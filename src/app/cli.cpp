#include "cli.hpp"

#include "nnci/assignments.hpp"
#include "nnci/logging.hpp"
#include "nnci/version.hpp"
#include "perceptron_command.hpp"
#include "perceptron_options.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace nnci::app {
namespace {

void print_usage() {
    std::cout << "Usage:\n"
        << "  nnci --version\n"
        << "  nnci [--debug] perceptron [options]\n\n"
        << "Perceptron options:\n"
        << "  --dimensions N    feature count (default: 20)\n"
        << "  --datasets nD     datasets per P (default: 50)\n"
        << "  --max-epochs nmax (default: 100)\n"
        << "  --p-min P         first pattern count (default: max(1, N/10))\n"
        << "  --p-max P         last pattern count (default: 3N)\n"
        << "  --p-step P        pattern-count step (default: max(1, N/10))\n"
        << "  --seed SEED       random seed (default: 2026)\n"
        << "  --threshold c     potential threshold (default: 0.0)\n"
        << "  --output FILE     CSV output path (default: results/capacity_N<N>.csv)\n"
        << "  --capacity        run capacity study and plot it (default: false)\n"
        << "  --NCapacity START END STEP  sweep N and create one combined plot\n"
        << "  --strength        run strength convergence study and plot it (default: false)\n"
        << "  --patterns P      patterns for single-run or strength study (default: N)\n"
        << "  --help            show this help message\n";
}

}  // namespace

int run_cli(int argc, char* argv[]) {
    const auto executable_directory =
        std::filesystem::absolute(argv[0]).parent_path();
    bool debug = false;
    std::string_view command;
    std::vector<std::string_view> arguments;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);

        if (argument == "--help") {
            print_usage();
            return 0;
        }
        if (argument == "--version") {
            std::cout << "nnci " << NNCI_VERSION << '\n';
            return 0;
        }
        if (argument == "--debug") {
            debug = true;
        } else if (command.empty()) {
            command = argument;
        } else {
            arguments.push_back(argument);
        }
    }

    if (command.empty()) {
        print_usage();
        return 0;
    }

    try {
        nnci::Logger logger(debug ? nnci::LogLevel::Debug : nnci::LogLevel::Info);
        logger.write(nnci::LogLevel::Info, "app",
                     "command started: " + std::string(command));
        logger.write(nnci::LogLevel::Debug, "app",
                     "debug-level diagnostics enabled");

        if (command == "perceptron") {
            run_perceptron(
                parse_perceptron_options(arguments), logger, executable_directory);
        } else {
            std::cerr << "Unknown command: " << command << '\n';
            print_usage();
            return 2;
        }

        logger.write(nnci::LogLevel::Info, "app",
                     "command completed: " + std::string(command));
        return 0;
    } catch (const nnci::PerceptronError& error) {
        const int code = static_cast<int>(error.code());
        std::cerr << "Perceptron error " << code << ": " << error.what() << '\n';
        return code;
    } catch (const std::exception& error) {
        std::cerr << "Run failed: " << error.what() << '\n';
        return 1;
    }
}

}  // namespace nnci::app
