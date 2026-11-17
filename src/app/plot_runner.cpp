#include "plot_runner.hpp"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace nnci::app {

int run_python_script(
    const std::filesystem::path& executable_directory,
    const std::filesystem::path& script,
    const std::filesystem::path& input,
    const std::filesystem::path& output) {
    const auto script_path = executable_directory / script;
    if (!std::filesystem::is_regular_file(script_path)) {
        throw std::runtime_error(
            "Plotting script is missing beside the executable: " + script_path.string());
    }

#if defined(_WIN32)
    const std::vector<std::string> interpreters{"py -3", "python"};
    const std::string quiet_redirect = ">NUL 2>&1";
#else
    const std::vector<std::string> interpreters{"python3", "python"};
    const std::string quiet_redirect = ">/dev/null 2>&1";
#endif

    for (const auto& interpreter : interpreters) {
        const auto probe = interpreter + " -c \"import matplotlib\" " + quiet_redirect;
        if (std::system(probe.c_str()) != 0) {
            continue;
        }

        // Quotes preserve spaces in ordinary paths for the command shell.
        const std::string command =
            interpreter + " \"" + script_path.string() + "\" \"" + input.string() +
            "\" --output \"" + output.string() + "\"";
        return std::system(command.c_str());
    }

#if defined(_WIN32)
    throw std::runtime_error(
        "Could not find Python with matplotlib. Install it with: "
        "py -3 -m pip install -r requirements-plot.txt");
#else
    throw std::runtime_error(
        "Could not find Python with matplotlib. Install it with: "
        "python3 -m pip install -r requirements-plot.txt");
#endif
}

}  // namespace nnci::app
