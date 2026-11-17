#pragma once

#include "nnci/logging.hpp"
#include "perceptron_options.hpp"

#include <filesystem>

namespace nnci::app {

void run_perceptron(
    const PerceptronOptions& options,
    nnci::Logger& logger,
    const std::filesystem::path& executable_directory);

}  // namespace nnci::app
