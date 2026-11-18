#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace nnci::app {

struct PerceptronOptions {
    std::size_t dimensions = 20;
    std::size_t datasets = 50;
    std::size_t max_epochs = dimensions * datasets;
    std::size_t p_min = 0;
    std::size_t p_max = 0;
    std::size_t p_step = 0;
    bool p_min_explicit = false;
    bool p_max_explicit = false;
    bool p_step_explicit = false;
    bool max_epochs_explicit = false;
    std::uint64_t seed = 2026;
    double threshold = 0.0;
    std::filesystem::path output;

    bool capacity = false;
    bool strength = false;
    bool validation = false;
    bool cuda = false;
    bool parallelize = false;
    std::size_t patterns = 0;

    // All zero means --NCapacity was not supplied.
    std::size_t n_start = 0;
    std::size_t n_end = 0;
    std::size_t n_step = 0;

    [[nodiscard]] std::size_t epochs_for(std::size_t dimension_count) const;
};

[[nodiscard]] PerceptronOptions parse_perceptron_options(
    const std::vector<std::string_view>& arguments);

}  // namespace nnci::app
