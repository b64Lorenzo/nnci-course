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
    const CudaDatasetDescriptor* descriptors, TrainingState* states,
    std::size_t datasets) {
    const std::size_t trial = blockIdx.x * blockDim.x + threadIdx.x;
    if (trial >= datasets) return;
    const auto descriptor = descriptors[trial];
    const auto patterns = descriptor.patterns;
    const auto dimensions = descriptor.dimensions;
    auto state = states[trial];
    auto* w = weights + descriptor.weight_offset;
    auto* x = strengths + descriptor.strength_offset;
    const auto* data = inputs + descriptor.input_offset;
    const auto* signs = labels + descriptor.label_offset;

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
            if (potential <= descriptor.threshold) {
                state.phase = Updating;
            } else {
                advance_training_pattern(state, patterns);
            }
        } else {
            if (potential <= 0.0) state.all_correct = false;
            // Check every final potential: an early miss must not hide a later overflow.
            if (++state.pattern == patterns) {
                state.pattern = 0;
                if (state.all_correct) {
                    state.converged = true;
                    state.phase = Finished;
                } else if (state.epochs == descriptor.max_epochs) {
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
    const std::vector<CudaTrainingOptions>& options,
    Logger* logger) {
    if (datasets.empty()) {
        throw PerceptronError(PerceptronErrorCode::InvalidDataset,
                              "Training batch must not be empty");
    }
    if (datasets.size() > cuda_batch_dataset_limit) {
        throw std::runtime_error("CUDA batch exceeds the 64 MiB / 1024 dataset batch limit");
    }
    if (options.size() != datasets.size()) {
        throw PerceptronError(PerceptronErrorCode::InvalidOptions,
                              "CUDA batch options must match the dataset count");
    }

    std::vector<CudaDatasetDescriptor> descriptors;
    descriptors.reserve(datasets.size());
    std::size_t batch_bytes = 0;
    std::size_t total_inputs = 0;
    std::size_t total_labels = 0;
    std::size_t total_weights = 0;
    std::size_t total_strengths = 0;
    const auto add_count = [](std::size_t current, std::size_t addition) {
        if (addition > std::numeric_limits<std::size_t>::max() - current) {
            throw std::runtime_error("CUDA allocation size is out of range");
        }
        return current + addition;
    };

    for (std::size_t trial = 0; trial < datasets.size(); ++trial) {
        if (options[trial].max_epochs == 0 || !std::isfinite(options[trial].threshold)) {
            throw PerceptronError(PerceptronErrorCode::InvalidOptions,
                                  "Maximum epochs must be positive and threshold finite");
        }
        const auto& dataset = datasets[trial];
        std::size_t dimensions = 0;
        try {
            dimensions = validate_labeled_examples(dataset);
        } catch (const std::invalid_argument& error) {
            throw PerceptronError(PerceptronErrorCode::InvalidDataset, error.what());
        }
        const auto patterns = dataset.size();
        const auto bytes = perceptron_cuda_dataset_bytes(patterns, dimensions);
        if (bytes > cuda_batch_memory_limit - batch_bytes) {
            throw std::runtime_error("CUDA batch exceeds the 64 MiB / 1024 dataset batch limit");
        }
        batch_bytes += bytes;

        const auto input_count = patterns * dimensions;
        descriptors.push_back({
            total_inputs, total_labels, total_weights, total_strengths,
            patterns, dimensions, options[trial].max_epochs, options[trial].threshold
        });
        total_inputs = add_count(total_inputs, input_count);
        total_labels = add_count(total_labels, patterns);
        total_weights = add_count(total_weights, dimensions);
        total_strengths = add_count(total_strengths, patterns);
    }

    std::vector<double> inputs;
    std::vector<int> labels;
    inputs.reserve(total_inputs);
    labels.reserve(total_labels);
    for (const auto& dataset : datasets) {
        for (const auto& example : dataset) {
            inputs.insert(inputs.end(), example.input.begin(), example.input.end());
            labels.push_back(example.label);
        }
    }

    initialize_device(logger);
    const auto count = datasets.size();
    std::vector<TrainingState> states(count);
    DeviceBuffer<double> device_inputs(total_inputs);
    DeviceBuffer<int> device_labels(total_labels);
    DeviceBuffer<double> device_weights(total_weights);
    DeviceBuffer<double> device_strengths(total_strengths);
    DeviceBuffer<CudaDatasetDescriptor> device_descriptors(count);
    DeviceBuffer<TrainingState> device_states(count);
    device_inputs.upload(inputs.data());
    device_labels.upload(labels.data());
    device_weights.zero();
    device_strengths.zero();
    device_descriptors.upload(descriptors.data());
    device_states.upload(states.data());

    bool pending = true;
    std::size_t launches = 0;
    while (pending) {
        train_chunk<<<static_cast<unsigned int>((count + 127) / 128), 128>>>(
            device_inputs.get(), device_labels.get(), device_weights.get(),
            device_strengths.get(), device_descriptors.get(), device_states.get(), count);
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
    std::vector<double> weights(total_weights);
    std::vector<double> strengths(total_strengths);
    device_weights.download(weights.data());
    device_strengths.download(strengths.data());
    std::vector<PerceptronResult> results;
    results.reserve(count);
    for (std::size_t trial = 0; trial < count; ++trial) {
        const auto& descriptor = descriptors[trial];
        results.push_back({
            Vector(weights.begin() + descriptor.weight_offset,
                   weights.begin() + descriptor.weight_offset + descriptor.dimensions),
            states[trial].epochs, states[trial].updates, states[trial].converged,
            Vector(strengths.begin() + descriptor.strength_offset,
                   strengths.begin() + descriptor.strength_offset + descriptor.patterns)
        });
    }
    return results;
}

std::vector<PerceptronResult> train_perceptron_cuda(
    const std::vector<PerceptronDataset>& datasets,
    std::size_t max_epochs, double threshold, Logger* logger) {
    return train_perceptron_cuda(
        datasets, std::vector<CudaTrainingOptions>(datasets.size(), {max_epochs, threshold}), logger);
}

}  // namespace nnci::detail
