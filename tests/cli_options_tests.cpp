#include "perceptron_options.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

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
    require(options.max_epochs == options.dimensions * options.datasets,
            "Default epoch limit should be N times nD");
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

void test_epoch_default_tracks_final_options() {
    for (const auto& arguments : std::vector<std::vector<std::string_view>>{
             {"--dimensions", "7", "--datasets", "3"},
             {"--datasets", "3", "--dimensions", "7"}}) {
        const auto options = nnci::app::parse_perceptron_options(arguments);
        require(options.max_epochs == options.dimensions * options.datasets,
                "The derived epoch limit must use the final N and nD");
    }
    require_invalid_argument([] {
        (void)nnci::app::parse_perceptron_options({"--max-epochs", "0"});
    }, "Explicit zero epochs must not mean automatic epochs");

    const auto largest = std::to_string(std::numeric_limits<std::size_t>::max());
    require_invalid_argument([&] {
        (void)nnci::app::parse_perceptron_options({
            "--dimensions", "2", "--datasets", largest
        });
    }, "Automatic epoch multiplication must reject overflow");
    const auto explicit_limit = nnci::app::parse_perceptron_options({
        "--dimensions", "2", "--datasets", largest, "--max-epochs", "1"
    });
    require(explicit_limit.max_epochs == 1,
            "An explicit epoch limit must not compute the unused default");
}

void test_cuda_is_an_opt_in_modifier() {
    require(!nnci::app::parse_perceptron_options({}).cuda,
            "CUDA must default to false");
    require(nnci::app::parse_perceptron_options({"--cuda"}).cuda,
            "Bare --cuda must opt in without consuming a value");
    for (const auto& arguments : std::vector<std::vector<std::string_view>>{
             {"--validation", "--cuda"}, {"--cuda", "--validation"}}) {
        const auto options = nnci::app::parse_perceptron_options(arguments);
        require(options.validation && options.cuda,
                "Validation must allow exactly the CUDA modifier in either order");
    }
    require_invalid_argument([] {
        (void)nnci::app::parse_perceptron_options({"--cuda", "false"});
    }, "--cuda must not accept a value");
    require_invalid_argument([] {
        (void)nnci::app::parse_perceptron_options({
            "--validation", "--cuda", "--capacity"
        });
    }, "Validation must reject other run options");
    require_invalid_argument([] {
        (void)nnci::app::parse_perceptron_options({"--validation", "--validation"});
    }, "Duplicate validation must not masquerade as a CUDA modifier");
}

void test_parallelize_is_an_opt_in_capacity_strategy() {
    require(!nnci::app::parse_perceptron_options({}).parallelize,
            "CPU study parallelism must default to false");
    const auto capacity = nnci::app::parse_perceptron_options({
        "--parallelize", "--capacity"
    });
    require(capacity.parallelize && capacity.capacity,
            "--parallelize should opt in for capacity studies");
    const auto sweep = nnci::app::parse_perceptron_options({
        "--NCapacity", "2", "4", "2", "--parallelize"
    });
    require(sweep.parallelize && sweep.n_start == 2,
            "--parallelize should opt in for NCapacity sweeps");
    require_invalid_argument([] {
        (void)nnci::app::parse_perceptron_options({"--parallelize"});
    }, "CPU parallelism must require a capacity study");
    require_invalid_argument([] {
        (void)nnci::app::parse_perceptron_options({
            "--capacity", "--parallelize", "--cuda"
        });
    }, "CPU parallelism and CUDA must not be combined");
    require_invalid_argument([] {
        (void)nnci::app::parse_perceptron_options({
            "--strength", "--parallelize"
        });
    }, "CPU parallelism must not be accepted by strength studies");
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
        test_epoch_default_tracks_final_options();
        test_cuda_is_an_opt_in_modifier();
        test_parallelize_is_an_opt_in_capacity_strategy();
        test_conflicting_and_invalid_options_are_rejected();
        std::cout << "PASS CLI option tests\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
