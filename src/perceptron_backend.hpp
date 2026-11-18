#pragma once

#include "nnci/assignments.hpp"

#include <algorithm>

namespace nnci::detail {

using PerceptronDataset = std::vector<LabeledExample>;

inline constexpr std::size_t cuda_batch_memory_limit = 64 * 1024 * 1024;
inline constexpr std::size_t cuda_batch_dataset_limit = 1024;

// Bound all resident device buffers, including state, before generating a batch.
[[nodiscard]] inline std::size_t perceptron_cuda_batch_size(
    std::size_t patterns, std::size_t dimensions) {
    constexpr auto limit = cuda_batch_memory_limit;
    constexpr auto per_pattern = sizeof(int) + sizeof(double);
    if (patterns == 0 || dimensions == 0 || dimensions > limit / sizeof(double) ||
        patterns > limit / (dimensions * sizeof(double) + per_pattern)) {
        throw std::runtime_error("CUDA dataset exceeds the 64 MiB batch limit");
    }
    const auto bytes = patterns * (dimensions * sizeof(double) + per_pattern) +
                       dimensions * sizeof(double) + 128;
    if (bytes > limit) {
        throw std::runtime_error("CUDA dataset exceeds the 64 MiB batch limit");
    }
    return std::min(cuda_batch_dataset_limit, limit / bytes);
}

// Private batch interface: independent datasets in, results in the same order out.
[[nodiscard]] std::vector<PerceptronResult> train_perceptron_cuda(
    const std::vector<PerceptronDataset>& datasets,
    std::size_t max_epochs, double threshold, Logger* logger);

}  // namespace nnci::detail
