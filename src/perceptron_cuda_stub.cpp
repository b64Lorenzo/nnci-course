#include "perceptron_backend.hpp"

#include <stdexcept>

namespace nnci::detail {
namespace {
[[noreturn]] void unsupported_cuda() {
    throw std::runtime_error(
        "CUDA support is not enabled; rebuild with -DNNCI_ENABLE_CUDA=ON and a CUDA Toolkit");
}
}

CudaBatchLimits perceptron_cuda_limits(Logger*) { unsupported_cuda(); }

std::vector<PerceptronResult> train_perceptron_cuda(
    const std::vector<PerceptronDataset>&,
    const std::vector<CudaTrainingOptions>&,
    Logger*) {
    unsupported_cuda();
}

std::vector<PerceptronResult> train_perceptron_cuda(
    const std::vector<PerceptronDataset>&, std::size_t, double, Logger*) {
    unsupported_cuda();
}

}  // namespace nnci::detail
