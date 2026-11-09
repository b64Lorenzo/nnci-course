#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nnci {

class Logger;

using Vector = std::vector<double>;

enum class PerceptronErrorCode : int {
    InvalidDataset = 10,
    InvalidOptions = 11,
    NumericalFailure = 12,
};

class PerceptronError final : public std::runtime_error {
public:
    PerceptronError(PerceptronErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    [[nodiscard]] PerceptronErrorCode code() const noexcept { return code_; }

private:
    PerceptronErrorCode code_;
};

struct LabeledExample {
    Vector input;
    int label;
};

struct PerceptronResult {
    Vector weights;
    std::size_t epochs;
    std::size_t updates;
    bool converged;
};

struct CapacityPoint {
    double alpha;
    std::size_t examples;
    std::size_t successes;
    std::size_t datasets;
    double success_rate;
};

[[nodiscard]] std::vector<LabeledExample> make_random_binary_data(
    std::size_t count, std::size_t dimensions, std::uint64_t seed);
[[nodiscard]] PerceptronResult train_rosenblatt_perceptron(
    const std::vector<LabeledExample>& examples, std::size_t max_epochs, double threshold,
    Logger* logger = nullptr);
[[nodiscard]] std::vector<CapacityPoint> estimate_perceptron_capacity(
    std::size_t dimensions, const std::vector<double>& alphas,
    std::size_t datasets_per_point, std::size_t max_epochs, std::uint64_t seed, double threshold,
    Logger* logger = nullptr);

}  // namespace nnci
