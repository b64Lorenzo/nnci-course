#include "study_output.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace nnci::app {
namespace {

void ensure_parent_directory(const std::filesystem::path& output) {
    const auto parent = output.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
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

    file << "N,P,alpha,nD,nmax,successes,Q_ls,seed\n" << std::setprecision(17);
    for (const auto& point : points) {
        file << dimensions << ',' << point.examples << ',' << point.alpha << ','
             << point.datasets << ',' << max_epochs << ',' << point.successes << ','
             << point.success_rate << ',' << seed << '\n';
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
