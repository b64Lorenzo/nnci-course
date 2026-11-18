#include "perceptron_command.hpp"

#include "nnci/assignments.hpp"
#include "plot_runner.hpp"
#include "study_output.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nnci::app {
namespace {

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

    write_capacity_csv(
        options.output, options.dimensions, options.max_epochs, options.seed, points);
    std::cout << "Wrote " << points.size() << " capacity points to "
              << options.output.string() << '\n';
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

        // Keep explicit P bounds across the N sweep; otherwise derive defaults from N.
        const auto default_step = std::max<std::size_t>(1, dimensions / 10);
        current.p_step = options.p_step_explicit ? options.p_step : default_step;
        current.p_min = options.p_min_explicit ? options.p_min : current.p_step;

        if (dimensions >
            std::numeric_limits<std::size_t>::max() / 3) {
            throw std::invalid_argument("Dimensions are too large");
        }
        current.p_max = options.p_max_explicit ? options.p_max : dimensions * 3;
        if (current.p_min == 0 || current.p_step == 0 ||
            current.p_max < current.p_min) {
            throw std::invalid_argument(
                "Pattern range must be positive and ordered for every swept dimension");
        }
        if ((current.p_max - current.p_min) / current.p_step >= 1'000'000) {
            throw std::invalid_argument(
                "Study is limited to 1000000 P values per run");
        }
        current.output = std::filesystem::path("results") /
            ("capacity_N" + std::to_string(dimensions) + ".csv");

        run_capacity_csv(current, logger);
        csv_paths.push_back(current.output);

        if (options.n_step > options.n_end - dimensions) {
            break;
        }

        dimensions += options.n_step;
    }

    const std::filesystem::path combined_csv = "results/capacity_N_sweep.csv";
    const std::filesystem::path combined_plot = "results/capacity_N_sweep.png";

    combine_capacity_csvs(csv_paths, combined_csv);

    if (run_python_script(
            executable_directory,
            "scripts/plot_capacity.py", combined_csv, combined_plot) != 0) {
        throw std::runtime_error("Combined capacity plotting script failed");
    }

    std::cout << "Saved combined capacity plot to " << combined_plot << '\n';
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
            executable_directory, "scripts/plot_strength.py", csv, png) != 0) {
        throw std::runtime_error("Strength plotting script failed");
    }

    std::cout << "Converged datasets: " << strengths.size()
              << "\nSaved strength histogram to " << png << '\n';
}

void run_validation(nnci::Logger& logger) {
    const std::vector<nnci::LabeledExample> separable{
        {{1.0, 0.0}, 1}, {{0.0, 1.0}, -1}
    };
    const auto separable_result = nnci::train_rosenblatt_perceptron(
        separable, 1, 0.0, &logger);
    if (!separable_result.converged ||
        separable_result.epochs != 1 ||
        separable_result.updates != 2 ||
        separable_result.weights != nnci::Vector{0.5, -0.5} ||
        separable_result.strengths != nnci::Vector{1.0, 1.0}) {
        throw std::runtime_error("Separable theoretical validation failed");
    }

    std::cout << "Separable case (N=2, P=2, c=0):\n"
              << "  w=(" << separable_result.weights[0] << ", "
              << separable_result.weights[1] << ")\n"
              << "  x=(" << separable_result.strengths[0] << ", "
              << separable_result.strengths[1] << ")\n"
              << "  U=" << separable_result.updates << '\n'
              << "  converged=true, epochs=" << separable_result.epochs << '\n'
              << "  PASS\n";

    const std::vector<nnci::LabeledExample> non_separable{
        {{1.0, 0.0}, 1}, {{1.0, 0.0}, -1}
    };
    const auto non_separable_result = nnci::train_rosenblatt_perceptron(
        non_separable, 2, 0.0, &logger);
    if (non_separable_result.converged ||
        non_separable_result.epochs != 2 ||
        non_separable_result.updates != 4) {
        throw std::runtime_error("Non-separable theoretical validation failed");
    }

    std::cout << "Non-separable case (opposite labels for input (1,0)):\n"
              << "  converged=false after " << non_separable_result.epochs
              << " epochs (expected)\n"
              << "  PASS\n"
              << "PASS theoretical validation\n";
}

}  // namespace

void run_perceptron(
    const PerceptronOptions& options,
    nnci::Logger& logger,
    const std::filesystem::path& executable_directory) {
    if (options.validation) {
        run_validation(logger);
        return;
    }

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

}  // namespace nnci::app
