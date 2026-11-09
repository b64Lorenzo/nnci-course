#include "nnci/assignments.hpp"
#include "nnci/logging.hpp"
#include "../src/assignment_utils.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
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

void test_vector_length() {
    require(nnci::detail::length({3.0, 4.0}) == 5.0,
            "Vector length must be the Euclidean norm");
    require(nnci::detail::length({}) == 0.0,
            "The empty vector must have zero length");
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
    const auto result = nnci::train_rosenblatt_perceptron(examples, 1);

    require(result.converged, "A solution found in the last allowed sweep must count as success");
    require(result.epochs == 1, "Training should stop in the sweep that finds a solution");
    require(result.updates == 1, "Update at zero stability, but not on the second sample");
    require(result.weights == nnci::Vector{0.5, 0.0}, "The update size must be 1/N");
}

void test_perceptron_stops_at_maximum_sweeps() {
    const std::vector<nnci::LabeledExample> examples{
        {{1.0}, 1}, {{1.0}, -1}
    };
    const auto result = nnci::train_rosenblatt_perceptron(examples, 3);

    require(!result.converged, "Contradictory labels cannot be separated");
    require(result.epochs == 3, "Training must stop at the maximum sweep count");
    require(result.updates == 6, "At most nmax * P updates may be performed");
}

void test_capacity_study_is_reproducible() {
    const std::vector<double> alphas{0.25, 0.5};
    const auto first = nnci::estimate_perceptron_capacity(4, alphas, 4, 10, 23);
    const auto second = nnci::estimate_perceptron_capacity(4, alphas, 4, 10, 23);
    require(first.size() == alphas.size(), "Study returned the wrong number of points");

    for (std::size_t i = 0; i < first.size(); ++i) {
        require(first[i].examples == second[i].examples &&
                first[i].successes == second[i].successes &&
                first[i].success_rate == second[i].success_rate,
                "Study results must repeat for a fixed seed");
        require(first[i].datasets == 4 && first[i].successes <= first[i].datasets,
                "Study must report valid dataset counts");
        require(first[i].success_rate ==
                    static_cast<double>(first[i].successes) / first[i].datasets,
                "Success rate must equal the fraction of converged datasets");
    }
}

void test_perceptron_logs_and_reports_error_codes() {
    namespace fs = std::filesystem;
    const auto token = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = fs::temp_directory_path() / ("nnci-perceptron-test-" + std::to_string(token));
    fs::path log_file;

    {
        nnci::Logger logger(nnci::LogLevel::Debug, root);
        const std::vector<nnci::LabeledExample> valid{{{1.0}, 1}};
        const auto result = nnci::train_rosenblatt_perceptron(valid, 3, &logger);
        require(result.converged, "Logger-enabled training should still converge");

        const std::vector<nnci::LabeledExample> invalid{{{1.0}, 0}};
        bool invalid_data_code = false;
        try {
            (void)nnci::train_rosenblatt_perceptron(invalid, 3, &logger);
        } catch (const nnci::PerceptronError& error) {
            invalid_data_code = error.code() == nnci::PerceptronErrorCode::InvalidDataset &&
                                static_cast<int>(error.code()) == 10;
        }
        require(invalid_data_code, "Invalid datasets need a stable error code");

        bool invalid_options_code = false;
        try {
            (void)nnci::train_rosenblatt_perceptron(valid, 0, &logger);
        } catch (const nnci::PerceptronError& error) {
            invalid_options_code = error.code() == nnci::PerceptronErrorCode::InvalidOptions &&
                                   static_cast<int>(error.code()) == 11;
        }
        require(invalid_options_code, "Invalid options need a stable error code");

        const std::vector<nnci::LabeledExample> extreme{
            {{std::numeric_limits<double>::max()}, 1}
        };
        bool numerical_code = false;
        try {
            (void)nnci::train_rosenblatt_perceptron(extreme, 3, &logger);
        } catch (const nnci::PerceptronError& error) {
            numerical_code = error.code() == nnci::PerceptronErrorCode::NumericalFailure &&
                             static_cast<int>(error.code()) == 12;
        }
        require(numerical_code, "Non-finite arithmetic needs a stable error code");
        log_file = logger.log_file();
    }

    const auto contents = read_file(log_file);
    require(contents.find("training started") != std::string::npos,
            "Training should log its start");
    require(contents.find("epoch=1") != std::string::npos,
            "Training should log epoch summaries at debug level");
    require(contents.find("error_code=InvalidDataset") != std::string::npos &&
            contents.find("error_code=InvalidOptions") != std::string::npos &&
            contents.find("error_code=NumericalFailure") != std::string::npos,
            "Training should log typed errors and their codes");
    fs::remove_all(root);
}

}  // namespace

int main() {
    try {
        test_random_binary_data_is_seeded();
        test_vector_length();
        test_dot_product();
        test_perceptron_updates_sequentially_and_stops_when_solved();
        test_perceptron_stops_at_maximum_sweeps();
        test_capacity_study_is_reproducible();
        test_perceptron_logs_and_reports_error_codes();
        std::cout << "PASS Rosenblatt perceptron tests\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
