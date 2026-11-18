#include "study_output.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>

namespace nnci::app {
namespace {

void ensure_parent_directory(const std::filesystem::path& output) {
    const auto parent = output.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

long double binomial_lower_tail(std::size_t trials, std::size_t cutoff) {
    if (cutoff >= trials) {
        return 1.0L;
    }

    // Reflect the upper tail so the recurrence sums the smaller side.
    const bool complement = cutoff > trials / 2;
    const std::size_t lower_cutoff = complement ? trials - cutoff - 1 : cutoff;
    const long double n = static_cast<long double>(trials);
    const long double k = static_cast<long double>(lower_cutoff);
    long double term = std::exp(
        std::lgamma(n + 1.0L) - std::lgamma(k + 1.0L)
        - std::lgamma(n - k + 1.0L) - n * std::log(2.0L));
    long double sum = term;
    for (std::size_t i = lower_cutoff; i > 0; --i) {
        term *= static_cast<long double>(i)
            / static_cast<long double>(trials - i + 1);
        sum += term;
    }

    const long double probability = complement ? 1.0L - sum : sum;
    return std::clamp(probability, 0.0L, 1.0L);
}

double cover_probability(std::size_t dimensions, std::size_t patterns) {
    if (patterns <= dimensions) {
        return 1.0;
    }
    return static_cast<double>(binomial_lower_tail(patterns - 1, dimensions - 1));
}

double normal_approximation(std::size_t dimensions, std::size_t patterns) {
    if (patterns == 1) {
        return 1.0;
    }
    const double z = (2.0 * static_cast<double>(dimensions)
                      - static_cast<double>(patterns))
        / std::sqrt(static_cast<double>(patterns - 1));
    return 0.5 * std::erfc(-z / std::sqrt(2.0));
}

std::pair<double, double> wilson_interval(
    std::size_t successes, std::size_t datasets) {
    if (datasets == 0 || successes > datasets) {
        throw std::invalid_argument("invalid capacity success count");
    }
    constexpr double z = 1.96;
    const double n = static_cast<double>(datasets);
    const double rate = static_cast<double>(successes) / n;
    const double denominator = 1.0 + z * z / n;
    const double center = (rate + z * z / (2.0 * n)) / denominator;
    const double margin = z * std::sqrt(
        rate * (1.0 - rate) / n + z * z / (4.0 * n * n)) / denominator;
    return {std::clamp(center - margin, 0.0, 1.0),
            std::clamp(center + margin, 0.0, 1.0)};
}

}  // namespace

void write_capacity_csv(
    const std::filesystem::path& output,
    std::size_t dimensions,
    std::size_t max_epochs,
    std::uint64_t seed,
    const std::vector<nnci::CapacityPoint>& points) {
    ensure_parent_directory(output);

    std::ofstream file(output, std::ios::binary);
    if (!file) {
        throw std::runtime_error(
            "Could not open output file: " + output.string());
    }

    file << "N,P,alpha,nD,nmax,successes,Q_ls,Q_ls_variance_est,P_ls_exact,"
            "P_ls_normal,Q_ls_wilson95_low,Q_ls_wilson95_high,seed\n"
         << std::setprecision(17);
    for (const auto& point : points) {
        const auto interval = wilson_interval(point.successes, point.datasets);
        const double n = static_cast<double>(point.datasets);
        const double rate = static_cast<double>(point.successes) / n;
        const double variance = rate * (1.0 - rate) / n;
        file << dimensions << ',' << point.examples << ',' << point.alpha << ','
             << point.datasets << ',' << max_epochs << ',' << point.successes << ','
             << rate << ',' << variance << ','
             << cover_probability(dimensions, point.examples) << ','
             << normal_approximation(dimensions, point.examples) << ','
             << interval.first << ',' << interval.second << ',' << seed << '\n';
    }
    if (!file) {
        throw std::runtime_error(
            "Could not write output file: " + output.string());
    }
}

void combine_capacity_csvs(
    const std::vector<std::filesystem::path>& inputs,
    const std::filesystem::path& output) {
    ensure_parent_directory(output);

    std::ofstream combined(output, std::ios::binary);
    if (!combined) {
        throw std::runtime_error("Could not create combined capacity CSV");
    }

    bool wrote_header = false;
    for (const auto& input_path : inputs) {
        std::ifstream input(input_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error(
                "Could not read capacity CSV: " + input_path.string());
        }

        std::string line;
        if (!std::getline(input, line)) {
            throw std::runtime_error(
                "Capacity CSV is empty: " + input_path.string());
        }
        if (!wrote_header) {
            combined << line << '\n';
            wrote_header = true;
        }
        while (std::getline(input, line)) {
            combined << line << '\n';
        }
    }

    if (!combined) {
        throw std::runtime_error("Could not write combined capacity CSV");
    }
}

void write_strength_csv(
    const std::filesystem::path& path,
    const std::vector<nnci::Vector>& converged_strengths) {
    ensure_parent_directory(path);

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open strength CSV: " + path.string());
    }

    file << "successful_case,mu,x_mu\n" << std::setprecision(17);
    for (std::size_t trial = 0; trial < converged_strengths.size(); ++trial) {
        const auto& values = converged_strengths[trial];
        for (std::size_t mu = 0; mu < values.size(); ++mu) {
            file << trial + 1 << ',' << mu + 1 << ',' << values[mu] << '\n';
        }
    }

    if (!file) {
        throw std::runtime_error("Could not write strength CSV: " + path.string());
    }
}

}  // namespace nnci::app
