#pragma once

#include "nnci/assignments.hpp"

#include <algorithm>
#include <cmath>

namespace nnci::test {

// Scalar specification oracle: freeze every E_mu before accumulating an epoch.
inline PerceptronResult batch_reference(
    const std::vector<LabeledExample>& examples, std::size_t max_epochs, double threshold) {
    const auto dimensions = examples.front().input.size();
    PerceptronResult result{Vector(dimensions, 0.0), 0, 0, false, Vector(examples.size(), 0.0)};
    Vector potentials(examples.size(), 0.0);
    while (result.epochs < max_epochs) {
        Vector hebbian(dimensions, 0.0);
        for (std::size_t pattern = 0; pattern < examples.size(); ++pattern) {
            if (potentials[pattern] <= threshold) {
                ++result.updates;
                ++result.strengths[pattern];
                for (std::size_t feature = 0; feature < dimensions; ++feature) {
                    hebbian[feature] += examples[pattern].label * examples[pattern].input[feature];
                }
            }
        }
        for (std::size_t feature = 0; feature < dimensions; ++feature) {
            result.weights[feature] += hebbian[feature] / static_cast<double>(dimensions);
        }
        ++result.epochs;
        for (std::size_t pattern = 0; pattern < examples.size(); ++pattern) {
            double sum = 0.0;
            for (std::size_t feature = 0; feature < dimensions; ++feature) {
                sum += result.weights[feature] * examples[pattern].input[feature];
            }
            potentials[pattern] = examples[pattern].label * sum;
            if (!std::isfinite(potentials[pattern])) {
                throw std::runtime_error("Batch reference potential is non-finite");
            }
        }
        result.converged = *std::min_element(potentials.begin(), potentials.end()) > 0.0;
        if (result.converged) break;
    }
    return result;
}

}  // namespace nnci::test
