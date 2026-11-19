#pragma once

#include "nnci/assignments.hpp"

#include <algorithm>
#include <limits>

namespace nnci::detail {

using PerceptronDataset = std::vector<LabeledExample>;

inline constexpr std::size_t cuda_batch_state_bytes = 128;
inline constexpr std::size_t cuda_pattern_storage_bytes =
    sizeof(int) + 2 * sizeof(double) + sizeof(unsigned char);

struct CudaBatchLimits {
    std::size_t bytes;
    std::size_t datasets;
};

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
        throw std::runtime_error("CUDA dataset size is invalid or out of range");
    };
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (patterns == 0 || dimensions == 0 || patterns > maximum / dimensions) {
        overflow();
    }
    const auto input_values = patterns * dimensions;
    if (input_values > maximum / sizeof(double) ||
        patterns > maximum / cuda_pattern_storage_bytes ||
        dimensions > maximum / sizeof(double)) {
        overflow();
    }
    const auto inputs = input_values * sizeof(double);
    const auto pattern_buffers = patterns * cuda_pattern_storage_bytes;
    const auto weights = dimensions * sizeof(double);
    const auto overhead = sizeof(CudaDatasetDescriptor) + cuda_batch_state_bytes;
    if (inputs > maximum - pattern_buffers ||
        inputs + pattern_buffers > maximum - weights ||
        inputs + pattern_buffers + weights > maximum - overhead) {
        overflow();
    }
    const auto bytes = inputs + pattern_buffers + weights + overhead;
    return bytes;
}

// Preserve headroom; eight resident-block waves amortize batch transfers.
[[nodiscard]] inline CudaBatchLimits make_cuda_batch_limits(
    std::size_t free_bytes, std::size_t resident_blocks, std::size_t grid_limit) {
    const auto reserve = std::max<std::size_t>(256 * 1024 * 1024, free_bytes / 4);
    const auto bytes = free_bytes > reserve ? free_bytes - reserve : 0;
    const auto waves = resident_blocks > grid_limit / 8 ? grid_limit : resident_blocks * 8;
    return {bytes, std::min(waves, bytes / perceptron_cuda_dataset_bytes(1, 1))};
}

[[nodiscard]] CudaBatchLimits perceptron_cuda_limits(Logger* logger = nullptr);

[[nodiscard]] inline unsigned int perceptron_cuda_epoch_chunk(std::size_t input_values) {
    const auto work = 1024 * 1024 / std::max<std::size_t>(1, input_values);
    return static_cast<unsigned int>(std::clamp<std::size_t>(work, 1, 8));
}

[[nodiscard]] inline std::size_t perceptron_cuda_batch_size(
    std::size_t patterns, std::size_t dimensions,
    CudaBatchLimits limits = perceptron_cuda_limits()) {
    const auto count = std::min(limits.datasets,
        limits.bytes / perceptron_cuda_dataset_bytes(patterns, dimensions));
    if (count == 0) throw std::runtime_error("CUDA dataset exceeds available VRAM budget");
    return count;
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
