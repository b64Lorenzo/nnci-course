#pragma once

#include "nnci/assignments.hpp"

#include <cmath>
#include <stdexcept>

namespace nnci::detail {

inline double dot_product(const Vector& left, const Vector& right) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("Vector dimensions do not match");
    }

    double result = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        result += left[i] * right[i];
    }
    return result;
}

inline double length(const Vector& values) {
    return std::sqrt(dot_product(values, values));
}

inline std::size_t validate_labeled_examples(const std::vector<LabeledExample>& examples) {
    if (examples.empty() || examples.front().input.empty()) {
        throw std::invalid_argument("Training data must not be empty");
    }

    const auto dimensions = examples.front().input.size();
    for (const auto& example : examples) {
        if (example.input.size() != dimensions) {
            throw std::invalid_argument("Training inputs must have equal dimensions");
        }
        if (example.label != -1 && example.label != 1) {
            throw std::invalid_argument("Labels must be -1 or +1");
        }
        for (const double value : example.input) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Training inputs must be finite");
            }
        }
    }
    return dimensions;
}

}  // namespace nnci::detail
