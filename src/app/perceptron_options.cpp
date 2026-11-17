#include "perceptron_options.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

namespace nnci::app {
namespace {

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

void finalize_options(PerceptronOptions& options) {
    if (options.dimensions == 0 ||
        options.datasets == 0 ||
        options.max_epochs == 0) {
        throw std::invalid_argument(
            "Dimensions, datasets, and maximum epochs must be positive");
    }

    // N sweeps need per-dimension defaults, so defer pattern-range validation.
    if (options.n_start != 0) {
        if ((options.p_min_explicit && options.p_min == 0) ||
            (options.p_max_explicit && options.p_max == 0) ||
            (options.p_step_explicit && options.p_step == 0)) {
            throw std::invalid_argument("Explicit pattern values must be positive");
        }
        if (options.p_min_explicit && options.p_max_explicit &&
            options.p_max < options.p_min) {
            throw std::invalid_argument("Pattern range must be positive and ordered");
        }
        return;
    }

    if (options.patterns == 0) {
        options.patterns = options.dimensions;
    }
    if (options.patterns == 0) {
        throw std::invalid_argument("Patterns must be positive");
    }

    const std::size_t default_step =
        std::max<std::size_t>(1, options.dimensions / 10);
    if (options.p_step == 0) {
        options.p_step = default_step;
    }
    if (options.p_min == 0) {
        options.p_min = options.p_step;
    }
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

}  // namespace

PerceptronOptions parse_perceptron_options(
    const std::vector<std::string_view>& arguments) {
    PerceptronOptions options;
    bool saw_n_capacity = false;

    for (std::size_t i = 0; i < arguments.size(); ++i) {
        const auto option = arguments[i];

        if (option == "--capacity") {
            options.capacity = true;
        } else if (option == "--strength") {
            options.strength = true;
        } else if (option == "--patterns") {
            if (i + 1 >= arguments.size()) {
                throw std::invalid_argument("Missing value for --patterns");
            }
            options.patterns = parse_number<std::size_t>(arguments[++i], option);
        } else if (option == "--NCapacity") {
            if (saw_n_capacity || i + 3 >= arguments.size()) {
                throw std::invalid_argument(
                    "--NCapacity requires START END STEP");
            }

            saw_n_capacity = true;
            options.n_start = parse_number<std::size_t>(arguments[++i], option);
            options.n_end = parse_number<std::size_t>(arguments[++i], option);
            options.n_step = parse_number<std::size_t>(arguments[++i], option);
        } else {
            if (i + 1 >= arguments.size()) {
                throw std::invalid_argument(
                    "Missing value for " + std::string(option));
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
                options.p_min_explicit = true;
            } else if (option == "--p-max") {
                options.p_max = parse_number<std::size_t>(value, option);
                options.p_max_explicit = true;
            } else if (option == "--p-step") {
                options.p_step = parse_number<std::size_t>(value, option);
                options.p_step_explicit = true;
            } else if (option == "--seed") {
                options.seed = parse_number<std::uint64_t>(value, option);
            } else if (option == "--threshold") {
                options.threshold = parse_number<double>(value, option);
            } else if (option == "--output") {
                options.output = std::filesystem::path(value);
            } else {
                throw std::invalid_argument(
                    "Unknown perceptron option: " + std::string(option));
            }
        }
    }

    if (saw_n_capacity &&
        (options.n_start == 0 || options.n_end < options.n_start ||
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

}  // namespace nnci::app
