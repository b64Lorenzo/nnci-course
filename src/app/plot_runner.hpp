#pragma once

#include <filesystem>

namespace nnci::app {

[[nodiscard]] int run_python_script(
    const std::filesystem::path& executable_directory,
    const std::filesystem::path& script,
    const std::filesystem::path& input,
    const std::filesystem::path& output);

}  // namespace nnci::app
