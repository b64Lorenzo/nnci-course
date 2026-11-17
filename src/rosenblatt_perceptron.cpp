#include "nnci/assignments.hpp"

#include "assignment_utils.hpp"
#include "nnci/logging.hpp"

#include <cmath>
#include <numeric>
#include <string>
#include <string_view>

namespace nnci {
namespace {

std::string_view error_name(PerceptronErrorCode code) {
    switch (code) {
        case PerceptronErrorCode::InvalidDataset: return "InvalidDataset";
        case PerceptronErrorCode::InvalidOptions: return "InvalidOptions";
        case PerceptronErrorCode::NumericalFailure: return "NumericalFailure";
    }
    return "Unknown";
}

void log_error(Logger* logger, const PerceptronError& error) {
    if (logger == nullptr) {
        return;
    }

    const auto code = error.code();
    logger->write(LogLevel::Error, "rosenblatt",
        "error_code=" + std::string(error_name(code)) +
        "(" + std::to_string(static_cast<int>(code)) + ") message=" + error.what());
}

double signed_potential(const LabeledExample& example, const Vector& weights) {
    const double potential = example.label * detail::dot_product(weights, example.input);
    if (!std::isfinite(potential)) {
        throw PerceptronError(
            PerceptronErrorCode::NumericalFailure, "Potential became non-finite");
    }
    return potential;
}

}  // namespace

PerceptronResult train_rosenblatt_perceptron(
    const std::vector<LabeledExample>& examples,
    std::size_t max_epochs,
    double threshold,
    Logger* logger) {
    if (max_epochs == 0) {
        PerceptronError error(
            PerceptronErrorCode::InvalidOptions, "Maximum epochs must be positive");
        log_error(logger, error);
        throw error;
    }

    std::size_t dimensions;
    try {
        dimensions = detail::validate_labeled_examples(examples);
    } catch (const std::invalid_argument& cause) {
        PerceptronError error(PerceptronErrorCode::InvalidDataset, cause.what());
        log_error(logger, error);
        throw error;
    }

    if (logger != nullptr) {
        logger->write(LogLevel::Debug, "rosenblatt",
            "training started; P=" + std::to_string(examples.size()) +
            "; N=" + std::to_string(dimensions) +
            "; max_epochs=" + std::to_string(max_epochs));
    }

    PerceptronResult result{
        Vector(dimensions, 0.0), 0, 0, false, Vector(examples.size(), 0.0)
    };

    try {
        for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {
            bool updated = false;
            std::size_t epoch_updates = 0;

            for (std::size_t mu = 0; mu < examples.size(); ++mu) {
                const auto& example = examples[mu];
                if (signed_potential(example, result.weights) <= threshold) {
                    for (std::size_t i = 0; i < dimensions; ++i) {
                        result.weights[i] += example.label * example.input[i] /
                                             static_cast<double>(dimensions);
                        if (!std::isfinite(result.weights[i])) {
                            throw PerceptronError(
                                PerceptronErrorCode::NumericalFailure,
                                "Weight became non-finite");
                        }
                    }

                    result.strengths[mu] += 1.0;
                    ++result.updates;
                    ++epoch_updates;
                    updated = true;
                }
            }

            result.epochs = epoch + 1;
            if (logger != nullptr) {
                logger->write(LogLevel::Debug, "rosenblatt",
                    "epoch=" + std::to_string(result.epochs) +
                    "; updates=" + std::to_string(epoch_updates) +
                    "; total_updates=" + std::to_string(result.updates) +
                    "; total strength=" + std::to_string(std::accumulate(
                        result.strengths.begin(), result.strengths.end(), 0.0)));
            }

            bool all_examples_correct = !updated;
            if (updated) {
                all_examples_correct = true;
                for (const auto& example : examples) {
                    if (logger != nullptr) {
                        logger->write(
                            LogLevel::Debug, "rosenblatt", "checking potential for example");
                    }
                    if (signed_potential(example, result.weights) <= 0.0) {
                        all_examples_correct = false;
                    }
                }
            }

            if (logger != nullptr) {
                logger->write(LogLevel::Debug, "rosenblatt",
                    "epoch=" + std::to_string(result.epochs) +
                    "; all_examples_correct=" +
                    (all_examples_correct ? "true" : "false"));
            }

            if (all_examples_correct) {
                result.converged = true;
                if (logger != nullptr) {
                    logger->write(LogLevel::Debug, "rosenblatt",
                        "converged; epochs=" + std::to_string(result.epochs) +
                        "; updates=" + std::to_string(result.updates));
                }
                break;
            }
        }

        if (!result.converged && logger != nullptr) {
            logger->write(LogLevel::Debug, "rosenblatt",
                "maximum epochs reached; epochs=" + std::to_string(result.epochs) +
                "; updates=" + std::to_string(result.updates));
        }
    } catch (const PerceptronError& error) {
        log_error(logger, error);
        throw;
    }

    return result;
}

}  // namespace nnci
