#include "nnci/assignments.hpp"
#include "../src/perceptron_backend.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_equal(const nnci::PerceptronResult& cpu, const nnci::PerceptronResult& gpu) {
    require(cpu.weights == gpu.weights, "CPU/GPU weights must match exactly");
    require(cpu.epochs == gpu.epochs, "CPU/GPU epochs must match");
    require(cpu.updates == gpu.updates, "CPU/GPU updates must match");
    require(cpu.strengths == gpu.strengths, "CPU/GPU pattern strengths must match exactly");
    require(cpu.converged == gpu.converged, "CPU/GPU convergence must match");
}

void test_last_epoch_separable_parity() {
    const std::vector<nnci::LabeledExample> examples{
        {{1.0, 0.0}, 1}, {{0.0, 1.0}, -1}
    };
    const auto cpu = nnci::train_rosenblatt_perceptron(examples, 1, 0.0);
    require(cpu.converged && cpu.epochs == 1 && cpu.updates == 2,
            "The last allowed epoch must count as a successful solution");
    require_equal(cpu, nnci::train_rosenblatt_perceptron(examples, 1, 0.0, nullptr, true));
}

void test_batch_budget_is_checked_before_allocation() {
    require(nnci::detail::perceptron_cuda_batch_size(2, 2) == 1024,
            "Small independent trials should fill a parallel batch");
    const auto count = nnci::detail::perceptron_cuda_batch_size(1000, 1000);
    require(count > 0 && count < 1024, "Large datasets must reduce the batch size");
    for (const auto shape : std::vector<std::pair<std::size_t, std::size_t>>{
             {0, 1}, {1, 0}, {1, std::numeric_limits<std::size_t>::max()},
             {std::numeric_limits<std::size_t>::max(), 2}, {10000, 10000}}) {
        bool rejected = false;
        try {
            (void)nnci::detail::perceptron_cuda_batch_size(shape.first, shape.second);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "Invalid, overflowed and over-budget shapes must be rejected");
    }
    const auto exact_patterns = (nnci::detail::cuda_batch_memory_limit - sizeof(double) - 128) /
                                (2 * sizeof(double) + sizeof(int));
    require(nnci::detail::perceptron_cuda_batch_size(exact_patterns, 1) == 1,
            "The exact resident-memory boundary must fit one dataset");
    bool rejected_boundary = false;
    try {
        (void)nnci::detail::perceptron_cuda_batch_size(exact_patterns + 1, 1);
    } catch (const std::runtime_error&) {
        rejected_boundary = true;
    }
    require(rejected_boundary, "One pattern beyond the memory boundary must be rejected");
    std::vector<nnci::detail::PerceptronDataset> excessive(1025, {{{1.0}, 1}});
    bool rejected = false;
    try {
        (void)nnci::detail::train_perceptron_cuda(excessive, 1, 0, nullptr);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("batch limit") != std::string::npos;
    }
    require(rejected, "The backend must enforce its budget even for direct batch callers");
}

void compare_twice(const std::vector<nnci::LabeledExample>& examples,
                   std::size_t epochs, double threshold) {
    const auto cpu = nnci::train_rosenblatt_perceptron(examples, epochs, threshold);
    const auto gpu = nnci::train_rosenblatt_perceptron(examples, epochs, threshold, nullptr, true);
    require_equal(cpu, gpu);
    require_equal(gpu, nnci::train_rosenblatt_perceptron(examples, epochs, threshold, nullptr, true));
}

void test_threshold_and_nonseparable_semantics() {
    const std::vector<nnci::LabeledExample> contradictory{{{1.0}, 1}, {{1.0}, -1}};
    compare_twice(contradictory, 19, 0.0);
    const auto negative = nnci::train_rosenblatt_perceptron(contradictory, 19, -0.5, nullptr, true);
    require(negative.converged && negative.epochs == 1 && negative.updates == 0 &&
            negative.weights == nnci::Vector{0.0} && negative.strengths == nnci::Vector{0.0, 0.0},
            "A no-update epoch must converge immediately, including at negative threshold");
    compare_twice(contradictory, 19, -0.5);
    const std::vector<nnci::LabeledExample> repeated{{{1.0}, 1}, {{1.0}, 1}};
    const auto positive = nnci::train_rosenblatt_perceptron(repeated, 1, 2.0, nullptr, true);
    require(positive.converged && positive.updates == 2 && positive.weights == nnci::Vector{2.0},
            "Update uses <= threshold; final convergence uses > 0, independently of threshold");
    compare_twice(repeated, 1, 2.0);
}

void test_seeded_parallel_batch_and_feature_order() {
    for (const auto dimensions : {3U, 7U, 17U}) {
        std::vector<nnci::detail::PerceptronDataset> datasets;
        for (std::uint64_t seed = 0; seed < 193; ++seed) {
            datasets.push_back(nnci::make_random_binary_data(dimensions * 2, dimensions, seed));
        }
        for (const double threshold : {0.0, 0.25, -0.25}) {
            const auto first = nnci::detail::train_perceptron_cuda(datasets, 23, threshold, nullptr);
            const auto second = nnci::detail::train_perceptron_cuda(datasets, 23, threshold, nullptr);
            require(first.size() == datasets.size(), "The partial GPU block must retain every result");
            for (std::size_t trial = 0; trial < datasets.size(); ++trial) {
                require_equal(nnci::train_rosenblatt_perceptron(datasets[trial], 23, threshold), first[trial]);
                require_equal(first[trial], second[trial]);
            }
        }
    }
    compare_twice({{{0.1, -0.2, 0.7}, 1}, {{-0.9, 0.3, 0.1}, -1}}, 5, 0);
    const double tiny = std::numeric_limits<double>::denorm_min() * 7;
    compare_twice({{{tiny, tiny, -tiny}, 1}}, 3, 0);
}

void test_work_chunks_resume_inside_patterns_and_epochs() {
    const auto long_features = nnci::make_random_binary_data(3, 8193, 42);
    compare_twice(long_features, 2, 0);
    const std::vector<nnci::LabeledExample> many_patterns(5003, {{0.0, 0.0, 0.0}, 1});
    compare_twice(many_patterns, 3, 0);
}

template <typename Function>
void require_code(Function function, nnci::PerceptronErrorCode expected) {
    try {
        function();
    } catch (const nnci::PerceptronError& error) {
        require(error.code() == expected, "The numeric/dataset/options error code must be preserved");
        return;
    }
    throw std::runtime_error("The invalid or non-finite training run was not rejected");
}

void test_error_codes_and_all_final_potentials() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (const bool cuda : {false, true}) {
        for (const auto& examples : std::vector<std::vector<nnci::LabeledExample>>{
                 {}, {{{}, 1}}, {{{1.0}, 0}}, {{{1.0}, 1}, {{1.0, 2.0}, -1}},
                 {{{nan}, 1}}, {{{inf}, -1}}}) {
            require_code([&] {
                (void)nnci::train_rosenblatt_perceptron(examples, 1, 0, nullptr, cuda);
            }, nnci::PerceptronErrorCode::InvalidDataset);
        }
        for (const double threshold : {nan, inf, -inf}) {
            require_code([&] {
                (void)nnci::train_rosenblatt_perceptron({{{1.0}, 1}}, 1, threshold, nullptr, cuda);
            }, nnci::PerceptronErrorCode::InvalidOptions);
        }
        require_code([&] {
            (void)nnci::train_rosenblatt_perceptron({{{1.0}, 1}}, 0, 0, nullptr, cuda);
        }, nnci::PerceptronErrorCode::InvalidOptions);
        require_code([&] {
            (void)nnci::train_rosenblatt_perceptron({{{0.0}, 1}, {{1e200}, 1}}, 1, 0, nullptr, cuda);
        }, nnci::PerceptronErrorCode::NumericalFailure);
    }
    require_code([&] {
        (void)nnci::detail::train_perceptron_cuda({{{{nan}, 1}}}, 1, 0, nullptr);
    }, nnci::PerceptronErrorCode::InvalidDataset);
    require_code([] {
        (void)nnci::detail::train_perceptron_cuda({{{{1.0}, 1}}}, 0, 0, nullptr);
    }, nnci::PerceptronErrorCode::InvalidOptions);
    // A later failing trial must not leak a partial vector of successful results.
    require_code([] {
        (void)nnci::detail::train_perceptron_cuda(
            {{{{1.0}, 1}, {{1.0}, 1}}, {{{0.0}, 1}, {{1e200}, 1}}}, 1, 0, nullptr);
    }, nnci::PerceptronErrorCode::NumericalFailure);
}
}

int main() {
    try {
        test_last_epoch_separable_parity();
        test_batch_budget_is_checked_before_allocation();
        test_threshold_and_nonseparable_semantics();
        test_seeded_parallel_batch_and_feature_order();
        test_work_chunks_resume_inside_patterns_and_epochs();
        test_error_codes_and_all_final_potentials();
        std::cout << "PASS native CUDA/CPU exact differential tests\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
