#include "perceptron_backend.hpp"

#include <stdexcept>

namespace nnci::detail {

std::vector<PerceptronResult> train_perceptron_cuda(
    const std::vector<PerceptronDataset>&, std::size_t, double, Logger*) {
    throw std::runtime_error(
        "CUDA support is not enabled; rebuild with -DNNCI_ENABLE_CUDA=ON and a CUDA Toolkit");
}

}  // namespace nnci::detail
