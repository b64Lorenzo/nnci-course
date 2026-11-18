#include "nnci/assignments.hpp"
#include "nnci/logging.hpp"
#include "perceptron_backend.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace nnci {
namespace {

template <typename Consumer>
void run_trials(
    std::size_t dimensions, std::size_t patterns, std::size_t trials,
    std::size_t max_epochs, double threshold, std::mt19937_64& random,
    Logger* logger, bool cuda, Consumer consume) {
    if (!cuda) {
        for (std::size_t trial = 0; trial < trials; ++trial) {
            const auto examples = make_random_binary_data(patterns, dimensions, random());
            consume(trial, train_rosenblatt_perceptron(examples, max_epochs, threshold, logger));
        }
        return;
    }

    const auto batch_limit = detail::perceptron_cuda_batch_size(patterns, dimensions);
    for (std::size_t first = 0; first < trials;) {
        const auto count = std::min(batch_limit, trials - first);
        std::vector<detail::PerceptronDataset> batch;
        batch.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            batch.push_back(make_random_binary_data(patterns, dimensions, random()));
        }
        const auto results = detail::train_perceptron_cuda(batch, max_epochs, threshold, logger);
        for (std::size_t i = 0; i < count; ++i) {
            consume(first + i, results[i]);
        }
        first += count;
    }
}

constexpr std::size_t capacity_batch_memory_limit = 64 * 1024 * 1024;

class ThreadPool {
public:
    explicit ThreadPool(std::size_t count) {
        try {
            workers_.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                workers_.emplace_back([this] { run(); });
            }
        } catch (...) {
            {
                std::lock_guard lock(mutex_);
                stopping_ = true;
            }
            ready_.notify_all();
            for (auto& worker : workers_) {
                if (worker.joinable()) worker.join();
            }
            throw;
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    template <typename Function>
    auto submit(Function&& function) {
        using Result = std::invoke_result_t<Function>;
        auto task = std::make_shared<std::packaged_task<Result()>>(
            std::forward<Function>(function));
        auto future = task->get_future();
        {
            std::lock_guard lock(mutex_);
            if (stopping_) throw std::runtime_error("CPU study worker pool is stopping");
            tasks_.emplace([task] { (*task)(); });
        }
        ready_.notify_one();
        return future;
    }

private:
    void run() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable ready_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

struct CapacityTask {
    std::size_t study_index;
    std::size_t point_index;
    std::size_t trial_index;
    detail::PerceptronDataset examples;
    detail::CudaTrainingOptions options;
};

struct CapacityPointProgress {
    std::size_t completed = 0;
    std::size_t successes = 0;
    bool started = false;
};

std::size_t pattern_count(double alpha, std::size_t dimensions) {
    const auto rounded = std::llround(alpha * static_cast<double>(dimensions));
    return std::max<std::size_t>(1, static_cast<std::size_t>(rounded));
}

void validate_study(const CapacityStudySpec& study) {
    if (study.dimensions == 0 || study.alphas.empty() ||
        study.datasets_per_point == 0 || study.max_epochs == 0 ||
        !std::isfinite(study.threshold)) {
        throw std::invalid_argument(
            "Dimension, alpha values, trials, and epochs must be positive; threshold must be finite");
    }
    for (const double alpha : study.alphas) {
        const double patterns = alpha * static_cast<double>(study.dimensions);
        if (!std::isfinite(patterns) || alpha <= 0.0 ||
            patterns >= static_cast<double>(std::numeric_limits<long long>::max())) {
            throw std::invalid_argument("Alpha values must be positive and in range");
        }
    }
}

std::size_t host_task_bytes(std::size_t patterns, std::size_t dimensions) {
    const auto maximum = std::numeric_limits<std::size_t>::max();
    const auto multiply = [maximum](std::size_t left, std::size_t right) {
        return left != 0 && right > maximum / left ? maximum : left * right;
    };
    const auto add = [maximum](std::size_t left, std::size_t right) {
        return right > maximum - left ? maximum : left + right;
    };
    const auto input_bytes = multiply(multiply(patterns, dimensions), sizeof(double));
    const auto examples_bytes = multiply(patterns, sizeof(LabeledExample));
    const auto strength_bytes = multiply(patterns, sizeof(double));
    const auto weight_bytes = multiply(dimensions, sizeof(double));
    return add(add(add(input_bytes, examples_bytes), strength_bytes), weight_bytes);
}

std::size_t task_count(const std::vector<CapacityStudySpec>& studies) {
    const auto maximum = std::numeric_limits<std::size_t>::max();
    std::size_t count = 0;
    for (const auto& study : studies) {
        if (study.alphas.size() > (maximum - count) / study.datasets_per_point) {
            return maximum;
        }
        count += study.alphas.size() * study.datasets_per_point;
    }
    return count;
}

}  // namespace

std::vector<LabeledExample> make_random_binary_data(
    std::size_t count, std::size_t dimensions, std::uint64_t seed) {
    if (count == 0 || dimensions == 0) {
        throw std::invalid_argument("Data count and dimension must be positive");
    }

    std::mt19937_64 random(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::bernoulli_distribution positive(0.5);
    std::vector<LabeledExample> examples;
    examples.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        Vector input(dimensions);
        for (double& value : input) {
            value = normal(random);
        }
        examples.push_back({std::move(input), positive(random) ? 1 : -1});
    }
    return examples;
}

std::vector<std::vector<CapacityPoint>> estimate_perceptron_capacity_parallel(
    const std::vector<CapacityStudySpec>& studies,
    CapacityExecutionMode mode,
    Logger* logger) {
    if (studies.empty()) {
        throw std::invalid_argument("At least one capacity study is required");
    }
    if (mode != CapacityExecutionMode::CpuParallel && mode != CapacityExecutionMode::Cuda) {
        throw std::invalid_argument("Unknown capacity execution mode");
    }
    for (const auto& study : studies) validate_study(study);

    const auto total_tasks = task_count(studies);
    const auto hardware_threads = std::max<std::size_t>(
        1, std::thread::hardware_concurrency());
    const auto worker_count = mode == CapacityExecutionMode::CpuParallel
        ? std::min(total_tasks, hardware_threads)
        : std::size_t{0};
    const auto batch_limit = mode == CapacityExecutionMode::Cuda
        ? detail::cuda_batch_dataset_limit
        : worker_count;
    const auto study_group_size = batch_limit;
    std::unique_ptr<ThreadPool> workers;
    if (mode == CapacityExecutionMode::CpuParallel) {
        workers = std::make_unique<ThreadPool>(worker_count);
        if (logger != nullptr) {
            logger->write(LogLevel::Debug, "capacity",
                "scheduler=cpu-parallel; workers=" + std::to_string(worker_count) +
                "; batch_memory_limit=" + std::to_string(capacity_batch_memory_limit));
        }
    } else if (logger != nullptr) {
        logger->write(LogLevel::Debug, "capacity",
            "scheduler=cuda-batched; dataset_limit=" + std::to_string(batch_limit) +
            "; batch_memory_limit=" + std::to_string(capacity_batch_memory_limit));
    }

    std::vector<std::vector<CapacityPoint>> points(studies.size());
    std::vector<std::vector<CapacityPointProgress>> progress(studies.size());
    std::vector<bool> study_started(studies.size(), false);
    for (std::size_t index = 0; index < studies.size(); ++index) {
        points[index].resize(studies[index].alphas.size());
        progress[index].resize(studies[index].alphas.size());
    }

    std::vector<CapacityTask> batch;
    batch.reserve(batch_limit);
    std::size_t batch_bytes = 0;

    const auto process_batch = [&] {
        if (batch.empty()) return;
        std::vector<PerceptronResult> trained;
        if (mode == CapacityExecutionMode::Cuda) {
            std::vector<detail::PerceptronDataset> datasets;
            std::vector<detail::CudaTrainingOptions> options;
            datasets.reserve(batch.size());
            options.reserve(batch.size());
            for (auto& task : batch) {
                datasets.push_back(std::move(task.examples));
                options.push_back(task.options);
            }
            trained = detail::train_perceptron_cuda(datasets, options, logger);
        } else {
            std::vector<std::future<PerceptronResult>> futures;
            futures.reserve(batch.size());
            for (auto& task : batch) {
                auto examples = std::move(task.examples);
                const auto options = task.options;
                futures.push_back(workers->submit(
                    [examples = std::move(examples), options] {
                        return train_rosenblatt_perceptron(
                            examples, options.max_epochs, options.threshold);
                    }));
            }
            trained.reserve(futures.size());
            for (auto& future : futures) trained.push_back(future.get());
        }

        for (std::size_t index = 0; index < batch.size(); ++index) {
            const auto& task = batch[index];
            const auto& study = studies[task.study_index];
            const auto patterns = pattern_count(
                study.alphas[task.point_index], study.dimensions);
            auto& state = progress[task.study_index][task.point_index];
            if (trained[index].converged) ++state.successes;
            ++state.completed;

            if (logger != nullptr) {
                logger->write(LogLevel::Debug, "capacity",
                    "capacity study point trial completed; P=" +
                    std::to_string(patterns) + "; alpha=" +
                    std::to_string(study.alphas[task.point_index]) +
                    "; trial=" + std::to_string(task.trial_index + 1) +
                    "; successes=" + std::to_string(state.successes));
            }

            if (state.completed == study.datasets_per_point) {
                const double rate = static_cast<double>(state.successes) /
                                    static_cast<double>(study.datasets_per_point);
                const double effective_alpha = static_cast<double>(patterns) /
                                               static_cast<double>(study.dimensions);
                points[task.study_index][task.point_index] = {
                    effective_alpha, patterns, state.successes,
                    study.datasets_per_point, rate
                };
                if (logger != nullptr) {
                    logger->write(LogLevel::Info, "capacity",
                        "P=" + std::to_string(patterns) +
                        "; alpha=" + std::to_string(effective_alpha) +
                        "; successes=" + std::to_string(state.successes) + "/" +
                        std::to_string(study.datasets_per_point) +
                        "; Q_ls=" + std::to_string(rate));
                }
            }
        }
        batch.clear();
        batch_bytes = 0;
    };

    struct StudyCursor {
        std::size_t study_index;
        std::size_t point_index = 0;
        std::size_t trial_index = 0;
        std::mt19937_64 random;

        StudyCursor(std::size_t index, std::uint64_t seed)
            : study_index(index), random(seed) {}
    };

    for (std::size_t first_study = 0; first_study < studies.size();
         first_study += study_group_size) {
        const auto last_study = std::min(studies.size(), first_study + study_group_size);
        std::vector<StudyCursor> cursors;
        cursors.reserve(last_study - first_study);
        for (std::size_t index = first_study; index < last_study; ++index) {
            cursors.emplace_back(index, studies[index].seed);
        }

        bool remaining = true;
        while (remaining) {
            remaining = false;
            for (auto& cursor : cursors) {
                const auto& study = studies[cursor.study_index];
                while (cursor.point_index < study.alphas.size() &&
                       cursor.trial_index == study.datasets_per_point) {
                    ++cursor.point_index;
                    cursor.trial_index = 0;
                }
                if (cursor.point_index == study.alphas.size()) continue;
                remaining = true;

                const auto alpha = study.alphas[cursor.point_index];
                const auto patterns = pattern_count(alpha, study.dimensions);
                const auto bytes = mode == CapacityExecutionMode::Cuda
                    ? detail::perceptron_cuda_dataset_bytes(patterns, study.dimensions)
                    : host_task_bytes(patterns, study.dimensions);
                const bool exceeds_memory = !batch.empty() &&
                    (batch_bytes >= capacity_batch_memory_limit ||
                     bytes > capacity_batch_memory_limit - batch_bytes);
                if (batch.size() >= batch_limit || exceeds_memory) process_batch();

                if (!study_started[cursor.study_index]) {
                    study_started[cursor.study_index] = true;
                    if (logger != nullptr) {
                        logger->write(LogLevel::Debug, "capacity",
                            "capacity study started; N=" +
                            std::to_string(study.dimensions) +
                            "; max_epochs=" + std::to_string(study.max_epochs) +
                            "; datasets_per_point=" +
                            std::to_string(study.datasets_per_point));
                    }
                }
                auto& state = progress[cursor.study_index][cursor.point_index];
                if (!state.started) {
                    state.started = true;
                    if (logger != nullptr) {
                        logger->write(LogLevel::Debug, "capacity",
                            "capacity study point started; P=" +
                            std::to_string(patterns) + "; alpha=" +
                            std::to_string(alpha));
                    }
                }

                auto examples = make_random_binary_data(
                    patterns, study.dimensions, cursor.random());
                batch.push_back({
                    cursor.study_index, cursor.point_index, cursor.trial_index,
                    std::move(examples), {study.max_epochs, study.threshold}
                });
                ++cursor.trial_index;
                batch_bytes = bytes > capacity_batch_memory_limit
                    ? capacity_batch_memory_limit
                    : batch_bytes + bytes;
                if (batch.size() >= batch_limit ||
                    batch_bytes >= capacity_batch_memory_limit) {
                    process_batch();
                }
            }
        }
    }
    process_batch();

    for (std::size_t study_index = 0; study_index < studies.size(); ++study_index) {
        for (const auto& state : progress[study_index]) {
            if (state.completed != studies[study_index].datasets_per_point) {
                throw std::logic_error("Capacity study batch lost a dataset result");
            }
        }
    }
    return points;
}

std::vector<Vector> collect_converged_strengths(
    std::size_t dimensions,
    std::size_t patterns,
    std::size_t trials,
    std::size_t max_epochs,
    std::uint64_t seed,
    double threshold,
    Logger* logger,
    bool cuda) {
    if (dimensions == 0 || patterns == 0 || trials == 0 || max_epochs == 0) {
        throw std::invalid_argument(
            "Dimension, patterns, trials, and epochs must be positive");
    }

    std::mt19937_64 random(seed);
    std::vector<Vector> strengths;
    strengths.reserve(trials);

    run_trials(dimensions, patterns, trials, max_epochs, threshold, random, logger, cuda,
        [&](std::size_t, const PerceptronResult& result) {
            if (result.converged) {
                strengths.push_back(result.strengths);
            }
        });
    return strengths;
}

std::vector<CapacityPoint> estimate_perceptron_capacity(
    std::size_t dimensions,
    const std::vector<double>& alphas,
    std::size_t datasets_per_point,
    std::size_t max_epochs,
    std::uint64_t seed,
    double threshold,
    Logger* logger,
    bool cuda) {
    if (cuda) {
        return estimate_perceptron_capacity_parallel(
            {{dimensions, alphas, datasets_per_point, max_epochs, seed, threshold}},
            CapacityExecutionMode::Cuda, logger).front();
    }
    if (dimensions == 0 || alphas.empty() ||
        datasets_per_point == 0 || max_epochs == 0) {
        throw std::invalid_argument(
            "Dimension, alpha values, trials, and epochs must be positive");
    }

    for (const double alpha : alphas) {
        const double patterns = alpha * static_cast<double>(dimensions);
        if (!std::isfinite(patterns) || alpha <= 0.0 ||
            patterns >= static_cast<double>(std::numeric_limits<long long>::max())) {
            throw std::invalid_argument("Alpha values must be positive and in range");
        }
    }

    std::mt19937_64 random(seed);
    std::vector<CapacityPoint> result;
    result.reserve(alphas.size());

    if (logger != nullptr) {
        logger->write(LogLevel::Debug, "capacity",
            "capacity study started; N=" + std::to_string(dimensions) +
            "; max_epochs=" + std::to_string(max_epochs) +
            "; datasets_per_point=" + std::to_string(datasets_per_point));
    }

    for (const double alpha : alphas) {
        const auto patterns = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::llround(
                alpha * static_cast<double>(dimensions))));

        if (logger != nullptr) {
            logger->write(LogLevel::Debug, "capacity",
                "capacity study point started; P=" + std::to_string(patterns) +
                "; alpha=" + std::to_string(alpha));
        }

        std::size_t successes = 0;
        run_trials(dimensions, patterns, datasets_per_point, max_epochs, threshold,
            random, logger, cuda, [&](std::size_t trial, const PerceptronResult& trained) {
            if (trained.converged) {
                ++successes;
            }

            if (logger != nullptr) {
                logger->write(LogLevel::Debug, "capacity",
                    "capacity study point trial completed; P=" +
                    std::to_string(patterns) + "; alpha=" + std::to_string(alpha) +
                    "; trial=" + std::to_string(trial + 1) +
                    "; successes=" + std::to_string(successes));
            }
        });

        const double success_rate =
            static_cast<double>(successes) / static_cast<double>(datasets_per_point);
        const double effective_alpha =
            static_cast<double>(patterns) / static_cast<double>(dimensions);
        result.push_back({effective_alpha, patterns, successes,
                          datasets_per_point, success_rate});

        if (logger != nullptr) {
            logger->write(LogLevel::Info, "capacity",
                "P=" + std::to_string(patterns) +
                "; alpha=" + std::to_string(effective_alpha) +
                "; successes=" + std::to_string(successes) + "/" +
                std::to_string(datasets_per_point) +
                "; Q_ls=" + std::to_string(success_rate));
        }
    }
    return result;
}

}  // namespace nnci
