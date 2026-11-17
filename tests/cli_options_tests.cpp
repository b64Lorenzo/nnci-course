#include "perceptron_options.hpp"

#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_invalid_argument(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_defaults_are_finalized_once() {
    const auto options = nnci::app::parse_perceptron_options({});
    require(options.dimensions == 20, "Default dimensions should be 20");
    require(options.datasets == 50, "Default dataset count should be 50");
    require(options.max_epochs == 100, "Default epoch count should be 100");
    require(options.patterns == 20, "Default pattern count should equal N");
    require(options.p_step == 2, "Default P step should be max(1, N/10)");
    require(options.p_min == 2, "Default P minimum should equal the P step");
    require(options.p_max == 60, "Default P maximum should equal 3N");
    require(options.output == "results/capacity_N20.csv",
            "Default capacity output path should be retained");
}

void test_explicit_values_and_n_sweep_options() {
    const auto options = nnci::app::parse_perceptron_options({
        "--dimensions", "8", "--datasets", "12", "--max-epochs", "30",
        "--p-min", "2", "--p-max", "6", "--p-step", "2",
        "--seed", "42", "--threshold", "0.25", "--capacity"
    });
    require(options.dimensions == 8 && options.datasets == 12 &&
                options.max_epochs == 30,
            "Explicit run counts should be parsed");
    require(options.p_min == 2 && options.p_max == 6 && options.p_step == 2,
            "Explicit pattern range should be parsed");
    require(options.p_min_explicit && options.p_max_explicit &&
                options.p_step_explicit,
            "Explicit P values should be tracked for N sweeps");
    require(options.seed == 42 && options.threshold == 0.25,
            "Seed and threshold should be parsed");
    require(options.capacity, "Capacity flag should be parsed");

    const auto sweep = nnci::app::parse_perceptron_options({
        "--NCapacity", "4", "10", "2"
    });
    require(sweep.n_start == 4 && sweep.n_end == 10 && sweep.n_step == 2,
            "N sweep values should be parsed");
    require(sweep.p_min == 0 && sweep.p_max == 0 && sweep.p_step == 0,
            "N sweep defaults should remain deferred until each dimension");
}

void test_conflicting_and_invalid_options_are_rejected() {
    require_invalid_argument([] {
        (void)nnci::app::parse_perceptron_options({
            "--capacity", "--NCapacity", "2", "4", "2"
        });
    }, "Capacity and NCapacity should be mutually exclusive");

    require_invalid_argument([] {
        (void)nnci::app::parse_perceptron_options({
            "--strength", "--capacity"
        });
    }, "Strength should be separate from capacity options");

    require_invalid_argument([] {
        (void)nnci::app::parse_perceptron_options({"--dimensions", "0"});
    }, "Zero dimensions should be rejected");

    require_invalid_argument([] {
        (void)nnci::app::parse_perceptron_options({"--unknown", "1"});
    }, "Unknown options should be rejected");
}

}  // namespace

int main() {
    try {
        test_defaults_are_finalized_once();
        test_explicit_values_and_n_sweep_options();
        test_conflicting_and_invalid_options_are_rejected();
        std::cout << "PASS CLI option tests\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
