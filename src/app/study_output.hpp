#pragma once

#include "nnci/assignments.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace nnci::app {

void write_capacity_csv(
    const std::filesystem::path& output,
    std::size_t dimensions,
    std::size_t max_epochs,
    std::uint64_t seed,
    const std::vector<nnci::CapacityPoint>& points);

void combine_capacity_csvs(
    const std::vector<std::filesystem::path>& inputs,
    const std::filesystem::path& output);

void write_strength_csv(
    const std::filesystem::path& path,
    const std::vector<nnci::Vector>& converged_strengths);

}  // namespace nnci::app
