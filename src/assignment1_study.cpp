#include "nnci/assignments.hpp"
#include "nnci/logging.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace nnci {

std::vector<LabeledExample> make_random_binary_data(

    // Generate a dataset of random binary-labeled examples with normally distributed inputs
    // Each example consists of a vector of 'dimensions' normally distributed values and a label of either +1 or -1
    // Inputs:
    //   - count: number of examples to generate (P in the book)
    //   - dimensions: number of input features for each example (N in the book)
    //   - seed: random seed for reproducibility
    // Output:
    //   - A vector of LabeledExample, each containing a randomly generated input vector and
    //     a randomly assigned label of +1 or -1
    //
    // The input is a vector of N normally distributed values, and the label is either +1 or -1, representing the binary classification target for that example.
    //
    // version 1.0.0

    std::size_t count, std::size_t dimensions, std::uint64_t seed) {

    if (count == 0 || dimensions == 0) {
        throw std::invalid_argument("Data count and dimension must be positive");
    }

    std::mt19937_64 random(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::bernoulli_distribution positive(0.5);
    std::vector<LabeledExample> examples;
    examples.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        Vector input(dimensions);
        // Fill the input vector with normally distributed random values
        for (double& value : input) {
            value = normal(random);
        }
        examples.push_back({ std::move(input), positive(random) ? 1 : -1 });  // Assign label +1 or -1 with equal probability
        // example: {input: [0.5, -1.2, 0.3], label: +1}
        // example: {input: [-0.7, 0.8, -0.1], label: -1}
        // Lebel is S^mu_T from the book

    }
    return examples;
}

std::vector<PerceptronResult> strength_convergence_study(

    // Estimate the convergence of the perceptron strengths over multiple datasets and epochs
    // Inputs:
    //   - dimensions: size of input features (N)
    //   - alphas: vector of alpha values (P/N) to test
    //   - datasets_per_point: number of random datasets to generate for each alpha
    //   - max_epochs: maximum number of epochs for training
    //   - seed: random seed for reproducibility
    //   - threshold: potential threshold weight update
    //   - logger: optional logger for debug information
    // Output:
    //   - A vector of PerceptronResult, each containing the final weights,
    //     number of epochs, number of updates, convergence status, and strengths
    //
    // Version 1.0.0

    std::size_t dimensions, const std::vector<double>& alphas,
    std::size_t datasets_per_point, std::size_t max_epochs, std::uint64_t seed, double threshold,
    Logger* logger) {

    if (dimensions == 0 || alphas.empty() || datasets_per_point == 0 || max_epochs == 0) {
        throw std::invalid_argument("Dimension, alpha values, trials, and epochs must be positive");
    }
    for (const double alpha : alphas) {
        const double patterns = alpha * static_cast<double>(dimensions);
        if (!std::isfinite(patterns) || alpha <= 0.0 ||
            patterns >= static_cast<double>(std::numeric_limits<long long>::max())) {
            throw std::invalid_argument("Alpha values must be positive and in range");
        }
    }
    std::mt19937_64 random(seed);
    std::vector<PerceptronResult> results;
    results.reserve(alphas.size());
    if (logger != nullptr) {
        logger->write(LogLevel::Debug, "strength_convergence", "strength convergence study started; N=" + std::to_string(dimensions) + "; max_epochs=" + std::to_string(max_epochs) + "; datasets_per_point=" + std::to_string(datasets_per_point));
    }

    for (const double alpha : alphas) {
        const auto count = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(alpha * static_cast<double>(dimensions))));
        if (logger != nullptr) {
            logger->write(LogLevel::Debug, "strength_convergence", "strength convergence study point started; P=" + std::to_string(count) + "; alpha=" + std::to_string(alpha));
        }
        for (std::size_t trial = 0; trial < datasets_per_point; ++trial) {
            const auto examples = make_random_binary_data(count, dimensions, random());
            const auto result = train_rosenblatt_perceptron(examples, max_epochs, threshold, logger);
            results.push_back(result);
            if (logger != nullptr) {
                logger->write(LogLevel::Debug, "strength_convergence", "strength convergence study point trial completed; P=" + std::to_string(count) +
                    "; alpha=" + std::to_string(alpha) +
                    "; trial=" + std::to_string(trial + 1) +
                    "; epochs=" + std::to_string(result.epochs) +
                    "; updates=" + std::to_string(result.updates));
            }
        }
    }

    return results;
}


std::vector<Vector> collect_converged_strengths(
    // Collect the strengths of converged perceptron trials over multiple datasets and epochs
    // Inputs:
    //   - dimensions: size of input features (N)
    //   - patterns: number of patterns (P) to generate for each trial
    //   - trials: number of random datasets to generate
    //   - max_epochs: maximum number of epochs for training
    //   - seed: random seed for reproducibility
    //   - threshold: potential threshold weight update
    //   - logger: optional logger for debug information
    // Output:
    //   - A vector of Vector, each containing the strengths of a converged trial
    //
    // Version 1.0.0

    std::size_t dimensions, std::size_t patterns, std::size_t trials, std::size_t max_epochs, std::uint64_t seed, double threshold, Logger* logger) {
    if (dimensions == 0 || patterns == 0 || trials == 0 || max_epochs == 0) {
        throw std::invalid_argument("Dimension, patterns, trials, and epochs must be positive");
    }

    std::mt19937_64 random(seed);
    std::vector<Vector> strengths;
    strengths.reserve(trials);

    for (std::size_t trial = 0; trial < trials; ++trial) {
        const auto examples = make_random_binary_data(patterns, dimensions, random());
        const auto result = train_rosenblatt_perceptron(examples, max_epochs, threshold, logger);
        if (result.converged) {
            strengths.push_back(result.strengths);
        }
    }
    return strengths;
}


std::vector<CapacityPoint> estimate_perceptron_capacity(

    // Estimate the capacity of a perceptron by generating random datasets and training on them
    // Inputs:
    //   - dimensions: size of input features (N)
    //   - alphas: vector of alpha values (P/N) to test
    //   - datasets_per_point: number of random datasets to generate for each alpha
    //   - max_epochs: maximum number of epochs for training
    //   - seed: random seed for reproducibility
    //   - threshold: potential threshold weight update
    //   - logger: optional logger for debug information
    // Output:
    //   - A vector of CapacityPoint, each containing the alpha value, number of examples, and success rate
    //
    // Version 1.0.0

    std::size_t dimensions, const std::vector<double>& alphas,
    std::size_t datasets_per_point, std::size_t max_epochs, std::uint64_t seed, double threshold,
    Logger* logger) {

    if (dimensions == 0 || alphas.empty() || datasets_per_point == 0 || max_epochs == 0) {
        throw std::invalid_argument("Dimension, alpha values, trials, and epochs must be positive");
    }

    for (const double alpha : alphas) {
        const double patterns = alpha * static_cast<double>(dimensions);
        if (!std::isfinite(patterns) || alpha <= 0.0 ||
            patterns >= static_cast<double>(std::numeric_limits<long long>::max())) {
            throw std::invalid_argument("Alpha values must be positive and in range");
        }
    }

    std::mt19937_64 random(seed);
    std::vector<CapacityPoint> result;
    result.reserve(alphas.size());

    if (logger != nullptr) {
        logger->write(LogLevel::Debug, "capacity", "capacity study started; N=" + std::to_string(dimensions) + "; max_epochs=" + std::to_string(max_epochs) + "; datasets_per_point=" + std::to_string(datasets_per_point));
    }

    for (const double alpha : alphas) {

        const auto count = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(alpha * static_cast<double>(dimensions))));

        if (logger != nullptr) {
            logger->write(LogLevel::Debug, "capacity", "capacity study point started; P=" + std::to_string(count) + "; alpha=" + std::to_string(alpha));
        }

        // Generate datasets and train the perceptron on each one, counting successes
        std::size_t successes = 0;
        for (std::size_t trial = 0; trial < datasets_per_point; ++trial) {

            const auto examples = make_random_binary_data(count, dimensions, random());

            if (train_rosenblatt_perceptron(examples, max_epochs, threshold, logger).converged) {
                ++successes;
            }

            if (logger != nullptr) {
                logger->write(LogLevel::Debug, "capacity", "capacity study point trial completed; P=" + std::to_string(count) +
                    "; alpha=" + std::to_string(alpha) +
                    "; trial=" + std::to_string(trial + 1) +
                    "; successes=" + std::to_string(successes));
            }
        }

        // Compute the success rate and store the results for this alpha value

        const double success_rate = static_cast<double>(successes) / static_cast<double>(datasets_per_point);
        const double effective_alpha = static_cast<double>(count) / static_cast<double>(dimensions);

        result.push_back({effective_alpha, count, successes, datasets_per_point, success_rate});

        if (logger != nullptr) {
            logger->write(LogLevel::Info, "capacity",
                          "P=" + std::to_string(count) +
                          "; alpha=" + std::to_string(effective_alpha) +
                          "; successes=" + std::to_string(successes) + "/" +
                          std::to_string(datasets_per_point) +
                          "; Q_ls=" + std::to_string(success_rate));
        }
    }
    return result;
}

}  // namespace nnci
