#include "perceptron_backend.hpp"

#include "assignment_utils.hpp"
#include "nnci/logging.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

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

constexpr unsigned int warp_width = 32;
constexpr unsigned int warps_per_dataset = 8;
constexpr unsigned int threads_per_dataset = warp_width * warps_per_dataset;
constexpr unsigned int epochs_per_launch = 8;

struct TrainingState {
    std::size_t epochs = 0;
    std::size_t updates = 0;
    bool converged = false;
    bool failed = false;
};
static_assert(sizeof(TrainingState) <= cuda_batch_state_bytes);

// One cooperative block per dataset: each epoch reads a frozen selection mask.
// Shared-memory barriers separate selection, Hebbian reduction and new potentials.
__global__ void train_batch_epochs(
    const double* inputs, const int* labels, double* weights, double* strengths,
    double* potentials, unsigned char* selected,
    const CudaDatasetDescriptor* descriptors, TrainingState* states) {
    const auto trial = blockIdx.x;
    const auto descriptor = descriptors[trial];
    const auto patterns = descriptor.patterns;
    const auto dimensions = descriptor.dimensions;
    const auto thread = threadIdx.x;
    const auto lane = thread % warp_width;
    const auto warp = thread / warp_width;
    auto& state = states[trial];
    auto* w = weights + descriptor.weight_offset;
    auto* x = strengths + descriptor.strength_offset;
    auto* e = potentials + descriptor.label_offset;
    auto* active = selected + descriptor.label_offset;
    const auto* data = inputs + descriptor.input_offset;
    const auto* signs = labels + descriptor.label_offset;
    __shared__ double sums[threads_per_dataset];
    __shared__ unsigned long long counts[warps_per_dataset];
    __shared__ int failed;

    for (unsigned int step = 0; step < epochs_per_launch; ++step) {
        if (state.converged || state.failed || state.epochs == descriptor.max_epochs) return;
        if (thread == 0) failed = 0;
        __syncthreads();

        unsigned long long count = 0;
        for (std::size_t pattern = thread; pattern < patterns; pattern += threads_per_dataset) {
            active[pattern] = e[pattern] <= descriptor.threshold;
            if (active[pattern]) {
                x[pattern] += 1.0;
                ++count;
            }
        }
        for (unsigned int offset = warp_width / 2; offset != 0; offset /= 2) {
            count += __shfl_down_sync(0xffffffff, count, offset);
        }
        if (lane == 0) counts[warp] = count;
        __syncthreads();
        if (thread == 0) {
            std::size_t total = 0;
            for (const auto value : counts) total += static_cast<std::size_t>(value);
            if (total > ~std::size_t{0} - state.updates) failed = 1;
            else state.updates += total;
        }
        __syncthreads();

        // Eight pattern lanes accumulate coalesced 32-feature tiles, then reduce.
        for (std::size_t first = 0; first < dimensions; first += warp_width) {
            const auto feature = first + lane;
            double sum = 0.0;
            if (feature < dimensions) {
                for (std::size_t pattern = warp; pattern < patterns; pattern += warps_per_dataset) {
                    if (active[pattern]) sum += signs[pattern] * data[pattern * dimensions + feature];
                }
            }
            sums[thread] = sum;
            __syncthreads();
            for (unsigned int stride = warps_per_dataset / 2; stride != 0; stride /= 2) {
                if (warp < stride) sums[thread] += sums[thread + stride * warp_width];
                __syncthreads();
            }
            if (warp == 0 && feature < dimensions) {
                w[feature] += sums[lane] / static_cast<double>(dimensions);
                if (!isfinite(w[feature])) atomicExch(&failed, 1);
            }
            __syncthreads();
        }

        double minimum = CUDART_INF;
        for (std::size_t pattern = warp; pattern < patterns; pattern += warps_per_dataset) {
            double sum = 0.0;
            for (std::size_t feature = lane; feature < dimensions; feature += warp_width) {
                sum += w[feature] * data[pattern * dimensions + feature];
            }
            for (unsigned int offset = warp_width / 2; offset != 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            if (lane == 0) {
                const double potential = signs[pattern] * sum;
                e[pattern] = potential;
                if (!isfinite(potential)) atomicExch(&failed, 1);
                minimum = fmin(minimum, potential);
            }
        }
        if (lane == 0) sums[warp] = minimum;
        __syncthreads();
        if (warp == 0) {
            minimum = lane < warps_per_dataset ? sums[lane] : CUDART_INF;
            for (unsigned int offset = warp_width / 2; offset != 0; offset /= 2) {
                minimum = fmin(minimum, __shfl_down_sync(0xffffffff, minimum, offset));
            }
            if (lane == 0) {
                ++state.epochs;
                state.failed = failed != 0;
                state.converged = !state.failed && minimum > 0.0;
            }
        }
        __syncthreads();
    }
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
                std::to_string(properties.minor) + "; algorithm=synchronous-batch");
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
    DeviceBuffer<double> device_potentials(total_labels);
    DeviceBuffer<unsigned char> device_selected(total_labels);
    DeviceBuffer<CudaDatasetDescriptor> device_descriptors(count);
    DeviceBuffer<TrainingState> device_states(count);
    device_inputs.upload(inputs.data());
    device_labels.upload(labels.data());
    device_weights.zero();
    device_strengths.zero();
    device_potentials.zero();
    device_descriptors.upload(descriptors.data());
    device_states.upload(states.data());

    bool pending = true;
    std::size_t launches = 0;
    while (pending) {
        train_batch_epochs<<<static_cast<unsigned int>(count), threads_per_dataset>>>(
            device_inputs.get(), device_labels.get(), device_weights.get(),
            device_strengths.get(), device_potentials.get(), device_selected.get(),
            device_descriptors.get(), device_states.get());
        check_cuda(cudaGetLastError(), "kernel launch failed");
        check_cuda(cudaDeviceSynchronize(), "kernel execution failed");
        ++launches;
        device_states.download(states.data());
        pending = false;
        for (std::size_t trial = 0; trial < count; ++trial) {
            const auto& state = states[trial];
            if (state.failed) {
                throw PerceptronError(PerceptronErrorCode::NumericalFailure,
                                      "Hebbian sum, potential, weight or update count became non-finite/out of range");
            }
            pending = pending || (!state.converged && state.epochs < options[trial].max_epochs);
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
