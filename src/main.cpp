#include "nnci/assignments.hpp"
#include "nnci/logging.hpp"
#include "nnci/version.hpp"

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

struct PerceptronOptions {
    std::size_t dimensions = 20;
    std::size_t datasets = 50;
    std::size_t max_epochs = 100;
    std::size_t p_min = 0;
    std::size_t p_max = 0;
    std::size_t p_step = 0;
    std::uint64_t seed = 2026;
    double threshold = 0.0;
    std::filesystem::path output;

    bool capacity = false;
    bool strength = false;
    std::size_t patterns = 0;


    // All zero means --NCapacity was not supplied.
    std::size_t n_start = 0;
    std::size_t n_end = 0;
    std::size_t n_step = 0;
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
        << "  nnci --version\n"
        << "  nnci [--debug] perceptron [options]\n\n"
        << "Perceptron options:\n"
        << "  --dimensions N    feature count (default: 20)\n"
        << "  --datasets nD     datasets per P (default: 50)\n"
        << "  --max-epochs nmax (default: 100)\n"
        << "  --p-min P         first pattern count (default: N/10)\n"
        << "  --p-max P         last pattern count (default: 3N)\n"
        << "  --p-step P        pattern-count step (default: N/10)\n"
        << "  --seed SEED       random seed (default: 2026)\n"
        << "  --threshold c     potential threshold (default: 0.0)\n"
        << "  --output FILE     CSV output path (default: results/capacity_N<N>.csv)\n"
        << "  --capacity        run capacity study and plot it (default: false)\n"
        << "  --NCapacity       sweep N from A through B by S, plotting each result\n"
        << "  --strength        run strength convergence study and plot it (default: false)\n"
        << "  --patterns P      patterns for single-run or strength study (default: N)\n"
        << "  --help            show this help message\n";
}

void finalize_options(PerceptronOptions& options) {
    if (options.dimensions == 0 ||
        options.datasets == 0 ||
        options.max_epochs == 0) {
        throw std::invalid_argument(
            "Dimensions, datasets, and maximum epochs must be positive");
    }

    if (options.patterns == 0)
        options.patterns = options.dimensions;

    if (options.patterns == 0)
        throw std::invalid_argument("Patterns must be positive");

    const std::size_t default_step =
        std::max<std::size_t>(1, options.dimensions / 10);

    if (options.p_step == 0)
        options.p_step = default_step;

    if (options.p_min == 0)
        options.p_min = options.p_step;

    if (options.p_max == 0) {
        if (options.dimensions >
            std::numeric_limits<std::size_t>::max() / 3) {
            throw std::invalid_argument("Dimensions are too large");
        }
        options.p_max = options.dimensions * 3;
    }

    if (options.p_min == 0 ||
        options.p_max < options.p_min ||
        options.p_step == 0) {
        throw std::invalid_argument(
            "Pattern range and step must be positive and ordered");
    }

    if ((options.p_max - options.p_min) / options.p_step >= 1'000'000) {
        throw std::invalid_argument(
            "Study is limited to 1000000 P values per run");
    }

    if (options.output.empty()) {
        options.output = std::filesystem::path("results") /
            ("capacity_N" + std::to_string(options.dimensions) + ".csv");
    }
}

PerceptronOptions parse_perceptron_options(
    const std::vector<std::string_view>& arguments) {
    PerceptronOptions options;
    bool saw_n_capacity = false;

    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const auto option = arguments[i];

        if (option == "--capacity") {
            options.capacity = true;
        }
        else if (option == "--strength") {
            options.strength = true;
        }
        else if (option == "--patterns") {
            if (i + 1 >= arguments.size()) {
                throw std::invalid_argument("Missing value for --patterns");
            }
            options.patterns = parse_number<std::size_t>(arguments[++i], option);
        }
        else if (option == "--NCapacity") {
            if (saw_n_capacity || i + 3 >= arguments.size()) {
                throw std::invalid_argument(
                    "--NCapacity requires START END STEP");
            }

            saw_n_capacity = true;
            options.n_start =
                parse_number<std::size_t>(arguments[++i], option);
            options.n_end =
                parse_number<std::size_t>(arguments[++i], option);
            options.n_step =
                parse_number<std::size_t>(arguments[++i], option);
        }
        else {
            if (i + 1 >= arguments.size()) {
                throw std::invalid_argument(
                    "Missing value for " + std::string(option));
            }

            const auto value = arguments[++i];

            if (option == "--dimensions") {
                options.dimensions = parse_number<std::size_t>(value, option);
            }
            else if (option == "--datasets") {
                options.datasets = parse_number<std::size_t>(value, option);
            }
            else if (option == "--max-epochs") {
                options.max_epochs = parse_number<std::size_t>(value, option);
            }
            else if (option == "--p-min") {
                options.p_min = parse_number<std::size_t>(value, option);
            }
            else if (option == "--p-max") {
                options.p_max = parse_number<std::size_t>(value, option);
            }
            else if (option == "--p-step") {
                options.p_step = parse_number<std::size_t>(value, option);
            }
            else if (option == "--seed") {
                options.seed = parse_number<std::uint64_t>(value, option);
            }
            else if (option == "--threshold") {
                options.threshold = parse_number<double>(value, option);
            }
            else if (option == "--output") {
                options.output = std::filesystem::path(value);
            }
            else {
                throw std::invalid_argument(
                    "Unknown perceptron option: " + std::string(option));
            }
        }
    }

    if (saw_n_capacity &&
        (options.n_start == 0 ||
            options.n_end < options.n_start ||
            options.n_step == 0)) {
        throw std::invalid_argument(
            "--NCapacity requires positive START and STEP, with END >= START");
    }

    if (options.capacity && saw_n_capacity) {
        throw std::invalid_argument(
            "Use either --capacity or --NCapacity, not both");
    }

    if (options.strength && (options.capacity || saw_n_capacity)) {
        throw std::invalid_argument(
            "Use --strength separately from capacity options");
    }

    finalize_options(options);
    return options;
}


std::vector<double> make_alphas(const PerceptronOptions& options) {
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
    const PerceptronOptions& options, const std::vector<nnci::CapacityPoint>& points) {
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

    // Paths are quoted so spaces in filenames work.
    const std::string command =
        "python \"" + script_path.string() + "\" \"" + input.string() +
        "\" --output \"" + output.string() + "\"";

    return std::system(command.c_str());
}

void run_capacity_csv(
    const PerceptronOptions& options,
    nnci::Logger& logger) {
    logger.write(nnci::LogLevel::Info, "study",
        "options used: dimensions=" + std::to_string(options.dimensions) +
        ", datasets=" + std::to_string(options.datasets) +
        ", max_epochs=" + std::to_string(options.max_epochs) +
        ", p_min=" + std::to_string(options.p_min) +
        ", p_max=" + std::to_string(options.p_max) +
        ", p_step=" + std::to_string(options.p_step) +
        ", seed=" + std::to_string(options.seed) +
        ", threshold=" + std::to_string(options.threshold) +
        ", output=" + options.output.string());

    const auto points = nnci::estimate_perceptron_capacity(
        options.dimensions,
        make_alphas(options),
        options.datasets,
        options.max_epochs,
        options.seed,
        options.threshold,
        &logger);

    write_capacity_csv(options, points);
    std::cout << "Wrote " << points.size() << " capacity points to "
              << options.output.string() << '\n';
}

void combine_capacity_csvs(
    const std::vector<std::filesystem::path>& inputs,
    const std::filesystem::path& output) {
    if (!output.parent_path().empty())
        std::filesystem::create_directories(output.parent_path());

    std::ofstream combined(output, std::ios::binary);
    if (!combined)
        throw std::runtime_error("Could not create combined capacity CSV");

    bool wrote_header = false;
    for (const auto& input_path : inputs) {
        std::ifstream input(input_path, std::ios::binary);
        if (!input)
            throw std::runtime_error("Could not read capacity CSV: " + input_path.string());

        std::string line;
        if (!std::getline(input, line))
            throw std::runtime_error("Capacity CSV is empty: " + input_path.string());
        if (!wrote_header) {
            combined << line << '\n';
            wrote_header = true;
        }
        while (std::getline(input, line))
            combined << line << '\n';
    }

    if (!combined)
        throw std::runtime_error("Could not write combined capacity CSV");
}

void run_capacity_and_plot(
    const PerceptronOptions& options,
    nnci::Logger& logger,
    const std::filesystem::path& executable_directory) {
    run_capacity_csv(options, logger);

    auto plot_path = options.output;
    plot_path.replace_extension(".png");

    if (run_python_script(
            executable_directory,
            "scripts/plot_capacity.py", options.output, plot_path) != 0) {
        throw std::runtime_error("Capacity plotting script failed");
    }

    std::cout << "Saved capacity plot to " << plot_path << '\n';
}

void run_n_capacity(
    const PerceptronOptions& options,
    nnci::Logger& logger,
    const std::filesystem::path& executable_directory) {
    std::vector<std::filesystem::path> csv_paths;

    for (std::size_t dimensions = options.n_start;;) {
        PerceptronOptions current = options;
        current.dimensions = dimensions;

        // Use a fresh N-dependent P range for each dimension.
        current.p_step = std::max<std::size_t>(1, dimensions / 10);
        current.p_min = current.p_step;

        if (dimensions >
            std::numeric_limits<std::size_t>::max() / 3) {
            throw std::invalid_argument("Dimensions are too large");
        }
        current.p_max = dimensions * 3;
        current.output = std::filesystem::path("results") /
            ("capacity_N" + std::to_string(dimensions) + ".csv");

        run_capacity_csv(current, logger);
        csv_paths.push_back(current.output);

        if (options.n_step > options.n_end - dimensions)
            break;

        dimensions += options.n_step;
    }

    const std::filesystem::path combined_csv =
        "results/capacity_N_sweep.csv";
    const std::filesystem::path combined_plot =
        "results/capacity_N_sweep.png";

    combine_capacity_csvs(csv_paths, combined_csv);

    if (run_python_script(
            executable_directory,
            "scripts/plot_capacity.py", combined_csv, combined_plot) != 0) {
        throw std::runtime_error("Combined capacity plotting script failed");
    }

    std::cout << "Saved combined capacity plot to " << combined_plot << '\n';
}

void write_strength_csv(
    const std::filesystem::path& path,
    const std::vector<nnci::Vector>& converged_strengths) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());

    std::ofstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Could not open strength CSV: " + path.string());

    file << "successful_case,mu,x_mu\n" << std::setprecision(17);
    for (std::size_t trial = 0; trial < converged_strengths.size(); ++trial) {
        const auto& values = converged_strengths[trial];
        for (std::size_t mu = 0; mu < values.size(); ++mu) {
            file << trial + 1 << ',' << mu + 1 << ',' << values[mu] << '\n';
        }
    }

    if (!file)
        throw std::runtime_error("Could not write strength CSV: " + path.string());
}

void run_strength(
    const PerceptronOptions& options,
    nnci::Logger& logger,
    const std::filesystem::path& executable_directory) {
    const auto strengths = nnci::collect_converged_strengths(
        options.dimensions,
        options.patterns,
        options.datasets,
        options.max_epochs,
        options.seed,
        options.threshold,
        &logger);


    if (strengths.empty()) {
        throw std::runtime_error(
            "No datasets converged; there are no strength values to plot");
    }

    const std::filesystem::path csv = "results/strengths.csv";
    const std::filesystem::path png = "results/strength_histogram.png";

    write_strength_csv(csv, strengths);

    if (run_python_script(
            executable_directory, "scripts/plot_strength.py", csv, png) != 0)
        throw std::runtime_error("Strength plotting script failed");

    std::cout << "Converged datasets: " << strengths.size()
        << "\nSaved strength histogram to " << png << '\n';
}

void run_perceptron(
    const PerceptronOptions& options,
    nnci::Logger& logger,
    const std::filesystem::path& executable_directory) {
    if (options.n_start != 0) {
        run_n_capacity(options, logger, executable_directory);
        return;
    }

    if (options.capacity) {
        run_capacity_and_plot(options, logger, executable_directory);
        return;
    }

    if (options.strength) {
        run_strength(options, logger, executable_directory);
        return;
    }

    // No study flag: train once on a randomly generated dataset.
    const auto examples = nnci::make_random_binary_data(
        options.patterns,
        options.dimensions,
        options.seed);

    const auto result = nnci::train_rosenblatt_perceptron(
        examples,
        options.max_epochs,
        options.threshold,
        &logger);

    std::cout << (result.converged ? "Converged" : "Did not converge")
        << " after " << result.epochs << " epochs; updates="
        << result.updates << '\n';
}



}  // namespace




int main(int argc, char* argv[]) {
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
        logger.write(nnci::LogLevel::Info, "app", "command started: " + std::string(command));
        logger.write(nnci::LogLevel::Debug, "app", "debug-level diagnostics enabled");

        if (command == "perceptron") {

            run_perceptron(
                parse_perceptron_options(arguments), logger, executable_directory);
        }
        else {
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
