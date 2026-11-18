#include "nnci/assignments.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_unsupported(Function function) {
    try {
        function();
    } catch (const nnci::PerceptronError&) {
        throw std::runtime_error("Unavailable CUDA must be an infrastructure error, not a data error");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()).find("CUDA support is not enabled") != std::string::npos,
                "CPU-only builds must explain how to enable CUDA");
        return;
    }
    throw std::runtime_error("An explicit CUDA request must not silently fall back to CPU");
}
}

int main() {
    try {
        const std::vector<nnci::LabeledExample> examples{{{1.0}, 1}};
        const auto legacy = nnci::train_rosenblatt_perceptron(examples, 1, 0);
        const auto explicit_cpu = nnci::train_rosenblatt_perceptron(examples, 1, 0, nullptr, false);
        require(legacy.weights == explicit_cpu.weights, "Default API must stay CPU");
        require_unsupported([&] {
            (void)nnci::train_rosenblatt_perceptron(examples, 1, 0, nullptr, true);
        });
        require_unsupported([] {
            (void)nnci::estimate_perceptron_capacity(2, {1.0}, 2, 2, 17, 0, nullptr, true);
        });
        require_unsupported([] {
            (void)nnci::collect_converged_strengths(2, 2, 2, 2, 17, 0, nullptr, true);
        });
        std::cout << "PASS explicit CPU-only CUDA rejection and default API compatibility\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
