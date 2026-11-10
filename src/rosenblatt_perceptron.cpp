#include "nnci/assignments.hpp"

#include "assignment_utils.hpp"
#include "nnci/logging.hpp"

#include <cmath>
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
    std::string message = "error_code=";
    message += error_name(code);
    message += "(" + std::to_string(static_cast<int>(code)) + ") message=";
    message += error.what();
    logger->write(LogLevel::Error, "rosenblatt", message);
}

}  // namespace

PerceptronResult train_rosenblatt_perceptron(

    // Train a Rosenblatt perceptron on a given dataset of labeled examples using the specified maximum number of epochs and potential threshold.
    // The function returns a PerceptronResult containing the final weights, number of epochs, number of updates, convergence status, and strengths.
    // examples: vector of labeled examples (input features and labels) Each input is a vector of N features, and each label is either +1 or -1.
    // P is the number of examples, and N is the number of features.
    // Inner workings: loop over epochs (each epoch is the total of all examples), for each example compute potential, update weights if potential <= threshold, track updates and convergence.
    //
    // Inputs:
    //   - examples: vector of labeled examples (input features and labels) Each input is a vector of N features, and each label is either +1 or -1.
    //   - max_epochs: maximum number of epochs to train the perceptron
    //   - threshold: potential threshold for weight updates
    //   - logger: optional logger for debug information
    // Output:
    //   - A PerceptronResult containing the final weights, number of epochs,
    //     number of updates, convergence status, and strengths
    //
    // Version 1.0.0

    const std::vector<LabeledExample>& examples, std::size_t max_epochs, double threshold,
    Logger* logger) {

    // Validate the maximum epochs parameter

    if (max_epochs == 0) {
        PerceptronError error(PerceptronErrorCode::InvalidOptions, "Maximum epochs must be positive");

        log_error(logger, error);

        throw error;
    }

    std::size_t dimensions = 0;
    try {
        // Validate the dataset and determine the input dimension

        dimensions = detail::validate_labeled_examples(examples);

    } catch (const std::invalid_argument& cause) {

        PerceptronError error(PerceptronErrorCode::InvalidDataset, cause.what());

        log_error(logger, error);

        throw error;
    }

    if (logger != nullptr) {

        logger->write(LogLevel::Debug, "rosenblatt", "training started; P=" + std::to_string(examples.size()) +
                      "; N=" + std::to_string(dimensions) +
                      "; max_epochs=" + std::to_string(max_epochs));
    }

    // Initialize the perceptron weights to zero: tabula rasa

    PerceptronResult result{
        Vector(dimensions, 0.0),
        0,
        0,
        false,
        Vector(examples.size(), 0.0)
    };



    // Train the perceptron using the Rosenblatt learning algorithm

    try {
        for (std::size_t epoch = 0; epoch < max_epochs; ++epoch) {

            // Track whether any updates were made during this epoch

            bool updated = false;
            std::size_t epoch_updates = 0;

            // Iterate over each labeled example in the dataset from 1 to P and compute the potential and strength for each example and update the weights if necessary

            for (std::size_t mu = 0; mu < examples.size(); ++mu) {
                const auto& example = examples[mu];


                // Compute the potential for the current example: E^mu = S^mu_T * W * input^mu
                const double potential = example.label * detail::dot_product(result.weights, example.input);


                if (!std::isfinite(potential)) {
                    throw PerceptronError(PerceptronErrorCode::NumericalFailure, "Potential became non-finite");
                }

                if (potential <= threshold) {

                    // Update the weights according to the Rosenblatt rule

                    for (std::size_t i = 0; i < dimensions; ++i) {

                        result.weights[i] += example.label * example.input[i] / static_cast<double>(dimensions);

                        if (!std::isfinite(result.weights[i])) {
                            throw PerceptronError(PerceptronErrorCode::NumericalFailure, "Weight became non-finite");
                        }
                    }

                    // Increment the embedding strength of the current pattern.
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
                              "; total strength=" + std::to_string(detail::dot_product(result.strengths, Vector(result.strengths.size(), 1.0))));
            }

            // Check if all examples are now correctly classified -> Theta(-E^mu) = 0 ==> Potential > 0

            bool all_examples_correct = !updated;

            if (logger != nullptr) {
                logger->write(LogLevel::Debug, "rosenblatt",
                              "epoch=" + std::to_string(result.epochs) +
                              "; all_examples_correct=" +
                              (all_examples_correct ? "true" : "false"));
            }

            if (updated) {

                all_examples_correct = true;

                for (const auto& example : examples) {

                    if (logger != nullptr) {
                        logger->write(
                            LogLevel::Debug,
                            "rosenblatt",
                            "checking potential for example");
                    }

                    const double potential = example.label * detail::dot_product(result.weights, example.input);

                    if (!std::isfinite(potential)) {
                        throw PerceptronError(PerceptronErrorCode::NumericalFailure, "Potential became non-finite");
                    }
                    if (potential <= 0.0) {
                        all_examples_correct = false;
                    }
                }
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
