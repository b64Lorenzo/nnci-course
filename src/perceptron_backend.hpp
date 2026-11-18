#pragma once

#include "nnci/assignments.hpp"

#include <algorithm>
#include <limits>

namespace nnci::detail {

using PerceptronDataset = std::vector<LabeledExample>;

inline constexpr std::size_t cuda_batch_memory_limit = 64 * 1024 * 1024;
inline constexpr std::size_t cuda_batch_dataset_limit = 1024;
inline constexpr std::size_t cuda_batch_state_bytes = 128;

struct CudaTrainingOptions {
    std::size_t max_epochs;
    double threshold;
};

struct CudaDatasetDescriptor {
    std::size_t input_offset;
    std::size_t label_offset;
    std::size_t weight_offset;
    std::size_t strength_offset;
    std::size_t patterns;
    std::size_t dimensions;
    std::size_t max_epochs;
    double threshold;
};

[[nodiscard]] inline std::size_t perceptron_cuda_dataset_bytes(
    std::size_t patterns, std::size_t dimensions) {
    const auto overflow = [] {
        throw std::runtime_error("CUDA dataset exceeds the 64 MiB batch limit");
    };
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (patterns == 0 || dimensions == 0 || patterns > maximum / dimensions) {
        overflow();
    }
    const auto input_values = patterns * dimensions;
    if (input_values > maximum / sizeof(double) ||
        patterns > maximum / (sizeof(int) + sizeof(double)) ||
        dimensions > maximum / sizeof(double)) {
        overflow();
    }
    const auto inputs = input_values * sizeof(double);
    const auto labels_and_strengths = patterns * (sizeof(int) + sizeof(double));
    const auto weights = dimensions * sizeof(double);
    const auto overhead = sizeof(CudaDatasetDescriptor) + cuda_batch_state_bytes;
    if (inputs > maximum - labels_and_strengths ||
        inputs + labels_and_strengths > maximum - weights ||
        inputs + labels_and_strengths + weights > maximum - overhead) {
        overflow();
    }
    const auto bytes = inputs + labels_and_strengths + weights + overhead;
    if (bytes > cuda_batch_memory_limit) {
        overflow();
    }
    return bytes;
}

// Bound resident device buffers before allocating or generating a batch.
[[nodiscard]] inline std::size_t perceptron_cuda_batch_size(
    std::size_t patterns, std::size_t dimensions) {
    return std::min(cuda_batch_dataset_limit,
                    cuda_batch_memory_limit / perceptron_cuda_dataset_bytes(patterns, dimensions));
}

// Private batch interface: independent datasets in, results in the same order out.
[[nodiscard]] std::vector<PerceptronResult> train_perceptron_cuda(
    const std::vector<PerceptronDataset>& datasets,
    const std::vector<CudaTrainingOptions>& options,
    Logger* logger);
[[nodiscard]] std::vector<PerceptronResult> train_perceptron_cuda(
    const std::vector<PerceptronDataset>& datasets,
    std::size_t max_epochs, double threshold, Logger* logger);

}  // namespace nnci::detail
