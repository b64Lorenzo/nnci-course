#include "perceptron_backend.hpp"

#include "assignment_utils.hpp"
#include "nnci/logging.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace nnci::detail {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + operation + ": " +
                                 cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : bytes_(checked_bytes(count)) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), bytes_), "allocation failed");
    }
    ~DeviceBuffer() { if (data_ != nullptr) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
    void zero() { check_cuda(cudaMemset(data_, 0, bytes_), "zero initialization failed"); }
    void upload(const T* data) {
        check_cuda(cudaMemcpy(data_, data, bytes_, cudaMemcpyHostToDevice), "upload failed");
    }
    void download(T* data) const {
        check_cuda(cudaMemcpy(data, data_, bytes_, cudaMemcpyDeviceToHost), "download failed");
    }
private:
    static std::size_t checked_bytes(std::size_t count) {
        if (count == 0 || count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::runtime_error("CUDA allocation size is out of range");
        }
        return count * sizeof(T);
    }
    T* data_ = nullptr;
    std::size_t bytes_;
};

enum Phase : int { Training, Updating, Checking, Finished, Failed };
struct TrainingState {
    std::size_t epochs = 0;
    std::size_t updates = 0;
    std::size_t pattern = 0;
    std::size_t feature = 0;
    double sum = 0.0;
    Phase phase = Training;
    bool updated = false;
    bool all_correct = true;
    bool converged = false;
};
static_assert(sizeof(TrainingState) <= 128);

__device__ void advance_training_pattern(TrainingState& state, std::size_t patterns) {
    if (++state.pattern == patterns) {
        ++state.epochs;
        state.pattern = 0;
        if (!state.updated) {
            state.converged = true;
            state.phase = Finished;
        } else {
            state.all_correct = true;
            state.phase = Checking;
        }
    } else {
        state.phase = Training;
    }
}

// Each launch has a fixed feature-operation budget, even for a huge single epoch.
// Resumable dot products retain the exact sequential CPU addition order.
__global__ void train_chunk(
    const double* inputs, const int* labels, double* weights, double* strengths,
    TrainingState* states, std::size_t datasets, std::size_t patterns,
    std::size_t dimensions, std::size_t max_epochs, double threshold) {
    const std::size_t trial = blockIdx.x * blockDim.x + threadIdx.x;
    if (trial >= datasets) return;
    auto state = states[trial];
    auto* w = weights + trial * dimensions;
    auto* x = strengths + trial * patterns;
    const auto* data = inputs + trial * patterns * dimensions;
    const auto* signs = labels + trial * patterns;

    for (unsigned int work = 0; work < 4096 &&
         state.phase != Finished && state.phase != Failed; ++work) {
        const auto offset = state.pattern * dimensions + state.feature;
        if (state.phase == Updating) {
            w[state.feature] += signs[state.pattern] * data[offset] /
                                static_cast<double>(dimensions);
            if (!isfinite(w[state.feature])) {
                state.phase = Failed;
                break;
            }
            if (++state.feature == dimensions) {
                state.feature = 0;
                x[state.pattern] += 1.0;
                ++state.updates;
                state.updated = true;
                advance_training_pattern(state, patterns);
            }
            continue;
        }

        state.sum += w[state.feature] * data[offset];
        if (++state.feature != dimensions) continue;
        const double potential = signs[state.pattern] * state.sum;
        state.feature = 0;
        state.sum = 0.0;
        if (!isfinite(potential)) {
            state.phase = Failed;
            break;
        }
        if (state.phase == Training) {
            if (potential <= threshold) {
                state.phase = Updating;
            } else {
                advance_training_pattern(state, patterns);
            }
        } else {
            if (potential <= 0.0) state.all_correct = false;
            // Check every final potential: an earlier miss must not hide a NaN/overflow.
            if (++state.pattern == patterns) {
                state.pattern = 0;
                if (state.all_correct) {
                    state.converged = true;
                    state.phase = Finished;
                } else if (state.epochs == max_epochs) {
                    state.phase = Finished;
                } else {
                    state.updated = false;
                    state.phase = Training;
                }
            }
        }
    }
    states[trial] = state;
}

void initialize_device(Logger* logger) {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "device discovery failed");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "device properties failed");
    if (logger != nullptr) {
        static std::once_flag logged;
        std::call_once(logged, [&] {
            logger->write(LogLevel::Info, "perceptron",
                "backend=CUDA; device=" + std::string(properties.name) +
                "; compute_capability=" + std::to_string(properties.major) + "." +
                std::to_string(properties.minor));
        });
    }
}

}  // namespace

std::vector<PerceptronResult> train_perceptron_cuda(
    const std::vector<PerceptronDataset>& datasets,
    std::size_t max_epochs, double threshold, Logger* logger) {
    if (max_epochs == 0 || !std::isfinite(threshold)) {
        throw PerceptronError(PerceptronErrorCode::InvalidOptions,
                              "Maximum epochs must be positive and threshold finite");
    }
    if (datasets.empty()) {
        throw PerceptronError(PerceptronErrorCode::InvalidDataset, "Training batch must not be empty");
    }
    std::size_t dimensions = 0;
    const auto patterns = datasets.front().size();
    try {
        dimensions = validate_labeled_examples(datasets.front());
        for (const auto& dataset : datasets) {
            if (validate_labeled_examples(dataset) != dimensions || dataset.size() != patterns) {
                throw std::invalid_argument("CUDA batch datasets must have matching shapes");
            }
        }
    } catch (const std::invalid_argument& error) {
        throw PerceptronError(PerceptronErrorCode::InvalidDataset, error.what());
    }
    if (datasets.size() > perceptron_cuda_batch_size(patterns, dimensions)) {
        throw std::runtime_error("CUDA batch exceeds the 64 MiB / 1024 dataset batch limit");
    }
    initialize_device(logger);
    const auto count = datasets.size();
    std::vector<double> inputs;
    std::vector<int> labels;
    inputs.reserve(count * patterns * dimensions);
    labels.reserve(count * patterns);
    for (const auto& dataset : datasets) {
        for (const auto& example : dataset) {
            inputs.insert(inputs.end(), example.input.begin(), example.input.end());
            labels.push_back(example.label);
        }
    }
    std::vector<TrainingState> states(count);
    DeviceBuffer<double> device_inputs(inputs.size());
    DeviceBuffer<int> device_labels(labels.size());
    DeviceBuffer<double> device_weights(count * dimensions);
    DeviceBuffer<double> device_strengths(count * patterns);
    DeviceBuffer<TrainingState> device_states(count);
    device_inputs.upload(inputs.data());
    device_labels.upload(labels.data());
    device_weights.zero();
    device_strengths.zero();
    device_states.upload(states.data());

    bool pending = true;
    std::size_t launches = 0;
    while (pending) {
        train_chunk<<<static_cast<unsigned int>((count + 127) / 128), 128>>>(
            device_inputs.get(), device_labels.get(), device_weights.get(),
            device_strengths.get(), device_states.get(), count, patterns,
            dimensions, max_epochs, threshold);
        check_cuda(cudaGetLastError(), "kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "kernel execution failed");
        ++launches;
        device_states.download(states.data());
        pending = false;
        for (const auto& state : states) {
            if (state.phase == Failed) {
                throw PerceptronError(PerceptronErrorCode::NumericalFailure,
                                      "Potential or weight became non-finite");
            }
            pending = pending || state.phase != Finished;
        }
    }

    if (logger != nullptr) {
        logger->write(LogLevel::Debug, "perceptron",
            "cuda_batch datasets=" + std::to_string(count) +
            "; launches=" + std::to_string(launches));
    }
    std::vector<double> weights(count * dimensions);
    std::vector<double> strengths(count * patterns);
    device_weights.download(weights.data());
    device_strengths.download(strengths.data());
    std::vector<PerceptronResult> results;
    results.reserve(count);
    for (std::size_t trial = 0; trial < count; ++trial) {
        results.push_back({
            Vector(weights.begin() + trial * dimensions, weights.begin() + (trial + 1) * dimensions),
            states[trial].epochs, states[trial].updates, states[trial].converged,
            Vector(strengths.begin() + trial * patterns, strengths.begin() + (trial + 1) * patterns)
        });
    }
    return results;
}

}  // namespace nnci::detail
