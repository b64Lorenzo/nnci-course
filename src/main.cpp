#include "nnci/assignments.hpp"
#include "nnci/logging.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

struct StudyOptions {
    std::size_t dimensions = 20;
    std::size_t datasets = 50;
    std::size_t max_epochs = 100;
    std::size_t p_min = 0;
    std::size_t p_max = 0;
    std::size_t p_step = 0;
    std::uint64_t seed = 2026;
    std::filesystem::path output;
};

template <typename Value>
Value parse_number(std::string_view text, std::string_view option) {
    Value value{};
    const auto* const end = text.data() + text.size();
    const auto [position, error] = std::from_chars(text.data(), end, value);
    if (error != std::errc{} || position != end) {
        throw std::invalid_argument("Invalid value for " + std::string(option));
    }
    return value;
}

void print_usage() {
    std::cout << "Usage:\n"
              << "  nnci [--debug] a1-study [options]\n\n"
              << "Study options:\n"
              << "  --dimensions N    feature count (default: 20)\n"
              << "  --datasets nD     datasets per P (default: 50)\n"
              << "  --max-epochs nmax (default: 100)\n"
              << "  --p-min P         first pattern count (default: N/10)\n"
              << "  --p-max P         last pattern count (default: 3N)\n"
              << "  --p-step P        pattern-count step (default: N/10)\n"
              << "  --seed SEED       random seed (default: 2026)\n"
              << "  --output FILE     CSV output path (default: results/capacity_N<N>.csv)\n";
}

StudyOptions parse_study_options(const std::vector<std::string_view>& arguments) {
    StudyOptions options;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const auto option = arguments[i];
        if (i + 1 >= arguments.size()) {
            throw std::invalid_argument("Missing value for " + std::string(option));
        }
        const auto value = arguments[++i];

        if (option == "--dimensions") {
            options.dimensions = parse_number<std::size_t>(value, option);
        } else if (option == "--datasets") {
            options.datasets = parse_number<std::size_t>(value, option);
        } else if (option == "--max-epochs") {
            options.max_epochs = parse_number<std::size_t>(value, option);
        } else if (option == "--p-min") {
            options.p_min = parse_number<std::size_t>(value, option);
        } else if (option == "--p-max") {
            options.p_max = parse_number<std::size_t>(value, option);
        } else if (option == "--p-step") {
            options.p_step = parse_number<std::size_t>(value, option);
        } else if (option == "--seed") {
            options.seed = parse_number<std::uint64_t>(value, option);
        } else if (option == "--output") {
            options.output = std::filesystem::path(value);
        } else {
            throw std::invalid_argument("Unknown study option: " + std::string(option));
        }


    }

    if (options.dimensions == 0 || options.datasets == 0 || options.max_epochs == 0) {
        throw std::invalid_argument("Dimensions, datasets, and maximum epochs must be positive");
    }

    const std::size_t default_step = std::max<std::size_t>(1, options.dimensions / 10);
    if (options.p_step == 0) {
        options.p_step = default_step;
    }
    if (options.p_min == 0) {
        options.p_min = options.p_step;
    }
    if (options.p_max == 0) {
        if (options.dimensions > std::numeric_limits<std::size_t>::max() / 3) {
            throw std::invalid_argument("Dimensions are too large");
        }
        options.p_max = options.dimensions * 3;
    }
    if (options.p_min == 0 || options.p_max < options.p_min || options.p_step == 0) {
        throw std::invalid_argument("Pattern range and step must be positive and ordered");
    }
    if ((options.p_max - options.p_min) / options.p_step >= 1000000) {
        throw std::invalid_argument("Study is limited to 1000000 P values per run");
    }

    if (options.output.empty()) {
        options.output = std::filesystem::path("results") /
                         ("capacity_N" + std::to_string(options.dimensions) + ".csv");
    }
    return options;
}

std::vector<double> make_alphas(const StudyOptions& options) {
    std::vector<double> alphas;
    for (std::size_t patterns = options.p_min;;) {
        alphas.push_back(static_cast<double>(patterns) /
                         static_cast<double>(options.dimensions));
        if (options.p_step > options.p_max - patterns) {
            break;
        }
        patterns += options.p_step;
    }
    return alphas;
}

void write_capacity_csv(
    const StudyOptions& options, const std::vector<nnci::CapacityPoint>& points) {
    if (!options.output.parent_path().empty()) {
        std::filesystem::create_directories(options.output.parent_path());
    }

    std::ofstream file(options.output, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open output file: " + options.output.string());
    }

    file << "N,P,alpha,nD,nmax,successes,Q_ls,seed\n" << std::setprecision(17);
    for (const auto& point : points) {
        file << options.dimensions << ',' << point.examples << ',' << point.alpha << ','
             << point.datasets << ',' << options.max_epochs << ',' << point.successes << ','
             << point.success_rate << ',' << options.seed << '\n';
    }
    if (!file) {
        throw std::runtime_error("Could not write output file: " + options.output.string());
    }
}

void run_study(const StudyOptions& options, nnci::Logger& logger) {

    logger.write(nnci::LogLevel::Info, "study",
                 "options used: dimensions=" + std::to_string(options.dimensions) +
                 ", datasets=" + std::to_string(options.datasets) +
                 ", max_epochs=" + std::to_string(options.max_epochs) +
                 ", p_min=" + std::to_string(options.p_min) +
                 ", p_max=" + std::to_string(options.p_max) +
                 ", p_step=" + std::to_string(options.p_step) +
                 ", seed=" + std::to_string(options.seed) +
                 ", output=" + options.output.string());


    const auto points = nnci::estimate_perceptron_capacity(

        options.dimensions, make_alphas(options), options.datasets, options.max_epochs, options.seed, &logger);

    write_capacity_csv(options, points);

    std::cout << "Wrote " << points.size() << " capacity points to "
              << options.output.string() << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
    bool debug = false;
    std::string_view command;
    std::vector<std::string_view> arguments;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help") {
            print_usage();
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
        logger.write(nnci::LogLevel::Info, "app", "command started: " + std::string(command));
        logger.write(nnci::LogLevel::Debug, "app", "debug-level diagnostics enabled");

        if (command == "a1-study") {
            run_study(parse_study_options(arguments), logger);
        } else {
            std::cerr << "Unknown command: " << command << '\n';
            print_usage();
            return 2;
        }

        logger.write(nnci::LogLevel::Info, "app", "command completed: " + std::string(command));
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
