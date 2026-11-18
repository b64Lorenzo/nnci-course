#include "nnci/assignments.hpp"
#include "../src/assignment_utils.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_random_binary_data_is_seeded() {
    const auto first = nnci::make_random_binary_data(8, 3, 17);
    const auto second = nnci::make_random_binary_data(8, 3, 17);
    require(first.size() == 8, "Data generator returned the wrong row count");

    for (std::size_t i = 0; i < first.size(); ++i) {
        require(first[i].input.size() == 3, "Generated input has the wrong dimension");
        require(first[i].input == second[i].input && first[i].label == second[i].label,
                "Generation must repeat for a fixed seed");
        require(first[i].label == -1 || first[i].label == 1,
                "Labels must be binary {-1, +1}");
    }
}

void test_dot_product() {
    const nnci::Vector left{1.0, 2.0, 3.0};
    const nnci::Vector right{4.0, -5.0, 6.0};
    require(nnci::detail::dot_product(left, right) == 12.0,
            "Dot product must sum pairwise products");
    require(nnci::detail::dot_product({}, {}) == 0.0,
            "The dot product of two empty vectors must be zero");

    bool rejected_mismatched_sizes = false;
    try {
        (void)nnci::detail::dot_product(left, {1.0});
    } catch (const std::invalid_argument&) {
        rejected_mismatched_sizes = true;
    }
    require(rejected_mismatched_sizes,
            "Dot product must reject vectors with different dimensions");
}

void test_perceptron_updates_sequentially_and_stops_when_solved() {
    const std::vector<nnci::LabeledExample> examples{
        {{1.0, 0.0}, 1}, {{-1.0, 0.0}, -1}
    };
    const auto result = nnci::train_rosenblatt_perceptron(examples, 1, 0);

    require(result.converged, "A solution found in the last allowed sweep must count as success");
    require(result.epochs == 1, "Training should stop in the sweep that finds a solution");
    require(result.updates == 1, "Update at zero stability, but not on the second sample");
    require(result.weights == nnci::Vector{0.5, 0.0}, "The update size must be 1/N");
}

void test_embedding_strengths_are_pattern_indexed() {
    const std::vector<nnci::LabeledExample> examples{
        {{1.0}, 1}, {{-1.0}, -1}, {{1.0}, 1}
    };
    const auto result = nnci::train_rosenblatt_perceptron(examples, 5, 0);

    require(result.strengths.size() == examples.size(),
            "There must be one embedding strength per training pattern");
    require(result.converged,
            "The constructed one-dimensional dataset should converge");
    require(result.strengths[0] == 1.0 &&
            result.strengths[1] == 0.0 &&
            result.strengths[2] == 0.0,
            "Each update must increment the strength of the pattern that caused it");
}

void test_non_finite_threshold_is_an_invalid_option() {
    for (const double threshold : {std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::infinity()}) {
        bool rejected = false;
        try {
            (void)nnci::train_rosenblatt_perceptron({{{1.0}, 1}}, 1, threshold);
        } catch (const nnci::PerceptronError& error) {
            rejected = error.code() == nnci::PerceptronErrorCode::InvalidOptions;
        }
        require(rejected, "Non-finite thresholds must be InvalidOptions (11), not convergence");
    }
}

}  // namespace

int main() {
    try {
        test_random_binary_data_is_seeded();
        test_dot_product();
        test_perceptron_updates_sequentially_and_stops_when_solved();
        test_embedding_strengths_are_pattern_indexed();
        test_non_finite_threshold_is_an_invalid_option();
        std::cout << "PASS Rosenblatt perceptron tests\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
