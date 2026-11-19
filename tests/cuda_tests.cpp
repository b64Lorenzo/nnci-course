#include "nnci/assignments.hpp"
#include "../src/perceptron_backend.hpp"
#include "batch_reference.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_equal(const nnci::PerceptronResult& expected, const nnci::PerceptronResult& gpu) {
    require(expected.weights.size() == gpu.weights.size(), "Batch weight dimensions must match");
    for (std::size_t i = 0; i < expected.weights.size(); ++i) {
        const auto tolerance = 1e-12 * (1.0 + std::abs(expected.weights[i]));
        require(std::isfinite(gpu.weights[i]) &&
                std::abs(expected.weights[i] - gpu.weights[i]) <= tolerance,
                "CUDA weights must match the independent synchronous batch reference");
    }
    require(expected.epochs == gpu.epochs, "Batch epochs must match");
    require(expected.updates == gpu.updates, "Batch selected-pattern counts must match");
    require(expected.strengths == gpu.strengths, "Batch pattern strengths must match exactly");
    require(expected.converged == gpu.converged, "Batch convergence must match");
}

void test_epoch_freezes_potentials_before_hebbian_sum() {
    const std::vector<nnci::LabeledExample> examples{{{1.0, 1.0}, 1}, {{1.0, 2.0}, 1}};
    const auto expected = nnci::test::batch_reference(examples, 1, 0.0);
    require(expected.weights == nnci::Vector{1.0, 1.5} && expected.updates == 2 &&
            expected.strengths == nnci::Vector{1.0, 1.0} && expected.converged,
            "The scalar oracle must match the hand-calculated Hebbian sum divided by N");
    const auto gpu = nnci::train_rosenblatt_perceptron(examples, 1, 0.0, nullptr, true);
    require(gpu.weights == expected.weights && gpu.updates == 2 &&
            gpu.strengths == expected.strengths && gpu.epochs == 1 && gpu.converged,
            "CUDA must freeze all potentials, then apply the full Hebbian sum / N once per epoch");
    const auto cpu = nnci::train_rosenblatt_perceptron(examples, 1, 0.0);
    require(cpu.weights == nnci::Vector{0.5, 0.5} && cpu.updates == 1,
            "The requested CUDA algorithm change must not change online CPU training");
}

void test_last_epoch_separable_parity() {
    const std::vector<nnci::LabeledExample> examples{
        {{1.0, 0.0}, 1}, {{0.0, 1.0}, -1}
    };
    const auto cpu = nnci::train_rosenblatt_perceptron(examples, 1, 0.0);
    require(cpu.converged && cpu.epochs == 1 && cpu.updates == 2,
            "The last allowed epoch must count as a successful solution");
    require_equal(cpu, nnci::train_rosenblatt_perceptron(examples, 1, 0.0, nullptr, true));
}

void test_batch_budget_is_checked_before_allocation() {
    require(nnci::detail::perceptron_cuda_batch_size(2, 2) == 1024,
            "Small independent trials should fill a parallel batch");
    const auto count = nnci::detail::perceptron_cuda_batch_size(1000, 1000);
    require(count > 0 && count < 1024, "Large datasets must reduce the batch size");
    for (const auto shape : std::vector<std::pair<std::size_t, std::size_t>>{
             {0, 1}, {1, 0}, {1, std::numeric_limits<std::size_t>::max()},
             {std::numeric_limits<std::size_t>::max(), 2}, {10000, 10000}}) {
        bool rejected = false;
        try {
            (void)nnci::detail::perceptron_cuda_batch_size(shape.first, shape.second);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "Invalid, overflowed and over-budget shapes must be rejected");
    }
    const auto exact_patterns =
        (nnci::detail::cuda_batch_memory_limit - sizeof(double) -
         sizeof(nnci::detail::CudaDatasetDescriptor) -
         nnci::detail::cuda_batch_state_bytes) /
        (3 * sizeof(double) + sizeof(int) + sizeof(unsigned char));
    require(nnci::detail::perceptron_cuda_batch_size(exact_patterns, 1) == 1,
            "The exact resident-memory boundary must fit one dataset");
    bool rejected_boundary = false;
    try {
        (void)nnci::detail::perceptron_cuda_batch_size(exact_patterns + 1, 1);
    } catch (const std::runtime_error&) {
        rejected_boundary = true;
    }
    require(rejected_boundary, "One pattern beyond the memory boundary must be rejected");
    std::vector<nnci::detail::PerceptronDataset> excessive(1025, {{{1.0}, 1}});
    bool rejected = false;
    try {
        (void)nnci::detail::train_perceptron_cuda(excessive, 1, 0, nullptr);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("batch limit") != std::string::npos;
    }
    require(rejected, "The backend must enforce its budget even for direct batch callers");
}

void compare_twice(const std::vector<nnci::LabeledExample>& examples,
                   std::size_t epochs, double threshold) {
    const auto cpu = nnci::test::batch_reference(examples, epochs, threshold);
    const auto gpu = nnci::train_rosenblatt_perceptron(examples, epochs, threshold, nullptr, true);
    require_equal(cpu, gpu);
    const auto repeat = nnci::train_rosenblatt_perceptron(examples, epochs, threshold, nullptr, true);
    require_equal(gpu, repeat);
    require(gpu.weights == repeat.weights, "Fixed CUDA reductions must be bitwise repeatable");
}

void test_threshold_and_nonseparable_semantics() {
    const std::vector<nnci::LabeledExample> contradictory{{{1.0}, 1}, {{1.0}, -1}};
    compare_twice(contradictory, 19, 0.0);
    const auto negative = nnci::train_rosenblatt_perceptron(contradictory, 19, -0.5, nullptr, true);
    require(!negative.converged && negative.epochs == 19 && negative.updates == 0 &&
            negative.weights == nnci::Vector{0.0} && negative.strengths == nnci::Vector{0.0, 0.0},
            "No active patterns must not imply convergence while min(E) is zero");
    compare_twice(contradictory, 19, -0.5);
    const std::vector<nnci::LabeledExample> repeated{{{1.0}, 1}, {{1.0}, 1}};
    const auto positive = nnci::train_rosenblatt_perceptron(repeated, 1, 2.0, nullptr, true);
    require(positive.converged && positive.updates == 2 && positive.weights == nnci::Vector{2.0},
            "Update uses <= threshold; final convergence uses > 0, independently of threshold");
    compare_twice(repeated, 1, 2.0);
}

void test_seeded_parallel_batch_and_feature_order() {
    for (const auto dimensions : {3U, 7U, 17U}) {
        std::vector<nnci::detail::PerceptronDataset> datasets;
        for (std::uint64_t seed = 0; seed < 193; ++seed) {
            datasets.push_back(nnci::make_random_binary_data(dimensions * 2, dimensions, seed));
        }
        for (const double threshold : {0.0, 0.25, -0.25}) {
            const auto first = nnci::detail::train_perceptron_cuda(datasets, 23, threshold, nullptr);
            const auto second = nnci::detail::train_perceptron_cuda(datasets, 23, threshold, nullptr);
            require(first.size() == datasets.size(), "The partial GPU block must retain every result");
            for (std::size_t trial = 0; trial < datasets.size(); ++trial) {
                require_equal(nnci::test::batch_reference(datasets[trial], 23, threshold), first[trial]);
                require_equal(first[trial], second[trial]);
                require(first[trial].weights == second[trial].weights,
                        "Seeded batch reductions must be deterministic");
            }
        }
    }
    compare_twice({{{0.1, -0.2, 0.7}, 1}, {{-0.9, 0.3, 0.1}, -1}}, 5, 0);
    const double tiny = std::numeric_limits<double>::denorm_min() * 7;
    compare_twice({{{tiny, tiny, -tiny}, 1}}, 3, 0);
    const std::vector<nnci::LabeledExample> subnormal{{{tiny, tiny, -tiny}, 1}};
    require(nnci::test::batch_reference(subnormal, 3, 0).weights ==
            nnci::train_rosenblatt_perceptron(subnormal, 3, 0, nullptr, true).weights,
            "Subnormal updates must not be hidden by the normal-scale tolerance");
}

void test_work_chunks_resume_inside_patterns_and_epochs() {
    const auto long_features = nnci::make_random_binary_data(3, 8193, 42);
    compare_twice(long_features, 2, 0);
    const std::vector<nnci::LabeledExample> many_patterns(5003, {{0.0, 0.0, 0.0}, 1});
    compare_twice(many_patterns, 3, 0);
}

void test_mixed_shape_batch_matches_cpu_and_keeps_order() {
    const std::vector<nnci::detail::PerceptronDataset> datasets{
        nnci::make_random_binary_data(4, 2, 31),
        nnci::make_random_binary_data(7, 4, 37),
        nnci::make_random_binary_data(3, 1, 41),
    };
    const std::vector<nnci::detail::CudaTrainingOptions> options{
        {7, 0.0}, {13, 0.25}, {1, -0.5}
    };
    const auto first = nnci::detail::train_perceptron_cuda(datasets, options, nullptr);
    const auto second = nnci::detail::train_perceptron_cuda(datasets, options, nullptr);
    require(first.size() == datasets.size(), "A mixed batch must retain every task");
    for (std::size_t task = 0; task < datasets.size(); ++task) {
        const auto cpu = nnci::test::batch_reference(
            datasets[task], options[task].max_epochs, options[task].threshold);
        require_equal(cpu, first[task]);
        require_equal(first[task], second[task]);
    }
}

void test_zero_minimum_and_batch_cancellation_exhaust_epochs() {
    const std::vector<nnci::LabeledExample> zero{{{1.0, 0.0}, 1}, {{0.0, 0.0}, -1}};
    const auto result = nnci::train_rosenblatt_perceptron(zero, 4, 0, nullptr, true);
    require(!result.converged && result.epochs == 4 && result.updates == 5 &&
            result.weights == nnci::Vector{0.5, 0.0} && result.strengths == nnci::Vector{1.0, 4.0},
            "A zero final potential must not satisfy strict min(E) > 0");
    const std::vector<nnci::LabeledExample> cancel{{{1.0}, 1}, {{1.0}, -1}};
    const auto cancelled = nnci::train_rosenblatt_perceptron(cancel, 17, 0, nullptr, true);
    require(!cancelled.converged && cancelled.epochs == 17 && cancelled.updates == 34 &&
            cancelled.weights == nnci::Vector{0.0} && cancelled.strengths == nnci::Vector{17, 17},
            "Cancelling Hebbian terms must continue across launches until max_epochs");
}

void test_capacity_and_strengths_use_batch_reference() {
    const std::vector<nnci::CapacityStudySpec> studies{
        {4, {0.5, 1.0, 1.5}, 7, 9, 17, 0.0}, {2, {0.5, 2.0}, 3, 3, 23, -0.5}
    };
    const auto actual = nnci::estimate_perceptron_capacity_parallel(
        studies, nnci::CapacityExecutionMode::Cuda);
    require(actual.size() == studies.size(), "Every mixed N study must be returned");
    for (std::size_t index = 0; index < studies.size(); ++index) {
        const auto& study = studies[index];
        std::mt19937_64 random(study.seed);
        require(actual[index].size() == study.alphas.size(), "Every P case must be returned");
        for (std::size_t point = 0; point < study.alphas.size(); ++point) {
            const auto patterns = std::max<std::size_t>(1, static_cast<std::size_t>(
                std::llround(study.alphas[point] * study.dimensions)));
            std::size_t successes = 0;
            for (std::size_t trial = 0; trial < study.datasets_per_point; ++trial) {
                const auto examples = nnci::make_random_binary_data(patterns, study.dimensions, random());
                successes += nnci::test::batch_reference(examples, study.max_epochs, study.threshold).converged;
            }
            const auto& result = actual[index][point];
            require(result.examples == patterns && result.successes == successes &&
                    result.datasets == study.datasets_per_point &&
                    result.success_rate == static_cast<double>(successes) / study.datasets_per_point,
                    "Mixed N/P capacity results must match batch learning, not online CPU training");
        }
    }
    std::mt19937_64 random(17);
    std::vector<nnci::Vector> expected;
    for (std::size_t trial = 0; trial < 19; ++trial) {
        const auto result = nnci::test::batch_reference(nnci::make_random_binary_data(3, 4, random()), 9, 0);
        if (result.converged) expected.push_back(result.strengths);
    }
    require(nnci::collect_converged_strengths(4, 3, 19, 9, 17, 0, nullptr, true) == expected,
            "Strength exports must retain only converged batch trials, in seed order");
}

template <typename Function>
void require_code(Function function, nnci::PerceptronErrorCode expected) {
    try {
        function();
    } catch (const nnci::PerceptronError& error) {
        require(error.code() == expected, "The numeric/dataset/options error code must be preserved");
        return;
    }
    throw std::runtime_error("The invalid or non-finite training run was not rejected");
}

void test_error_codes_and_all_final_potentials() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (const bool cuda : {false, true}) {
        for (const auto& examples : std::vector<std::vector<nnci::LabeledExample>>{
                 {}, {{{}, 1}}, {{{1.0}, 0}}, {{{1.0}, 1}, {{1.0, 2.0}, -1}},
                 {{{nan}, 1}}, {{{inf}, -1}}}) {
            require_code([&] {
                (void)nnci::train_rosenblatt_perceptron(examples, 1, 0, nullptr, cuda);
            }, nnci::PerceptronErrorCode::InvalidDataset);
        }
        for (const double threshold : {nan, inf, -inf}) {
            require_code([&] {
                (void)nnci::train_rosenblatt_perceptron({{{1.0}, 1}}, 1, threshold, nullptr, cuda);
            }, nnci::PerceptronErrorCode::InvalidOptions);
        }
        require_code([&] {
            (void)nnci::train_rosenblatt_perceptron({{{1.0}, 1}}, 0, 0, nullptr, cuda);
        }, nnci::PerceptronErrorCode::InvalidOptions);
        require_code([&] {
            (void)nnci::train_rosenblatt_perceptron({{{0.0}, 1}, {{1e200}, 1}}, 1, 0, nullptr, cuda);
        }, nnci::PerceptronErrorCode::NumericalFailure);
    }
    require_code([&] {
        (void)nnci::detail::train_perceptron_cuda({{{{nan}, 1}}}, 1, 0, nullptr);
    }, nnci::PerceptronErrorCode::InvalidDataset);
    require_code([] {
        (void)nnci::detail::train_perceptron_cuda({{{{1.0}, 1}}}, 0, 0, nullptr);
    }, nnci::PerceptronErrorCode::InvalidOptions);
    // A later failing trial must not leak a partial vector of successful results.
    require_code([] {
        (void)nnci::detail::train_perceptron_cuda(
            {{{{1.0}, 1}, {{1.0}, 1}}, {{{0.0}, 1}, {{1e200}, 1}}}, 1, 0, nullptr);
    }, nnci::PerceptronErrorCode::NumericalFailure);
}
}

int main() {
    try {
        test_epoch_freezes_potentials_before_hebbian_sum();
        test_last_epoch_separable_parity();
        test_batch_budget_is_checked_before_allocation();
        test_threshold_and_nonseparable_semantics();
        test_seeded_parallel_batch_and_feature_order();
        test_work_chunks_resume_inside_patterns_and_epochs();
        test_mixed_shape_batch_matches_cpu_and_keeps_order();
        test_error_codes_and_all_final_potentials();
        test_zero_minimum_and_batch_cancellation_exhaust_epochs();
        test_capacity_and_strengths_use_batch_reference();
        std::cout << "PASS native CUDA synchronous-batch reference and CPU compatibility tests\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
