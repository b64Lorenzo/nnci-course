#include "perceptron_backend.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void test_live_memory_and_occupancy_limits() {
    constexpr std::size_t mib = 1024 * 1024;
    constexpr std::size_t gib = 1024 * mib;
    const auto normal = nnci::detail::make_cuda_batch_limits(8 * gib, 64, 100000);
    require(normal.bytes == 6 * gib && normal.datasets == 512,
            "Budget must preserve 25% free VRAM and schedule eight resident-block waves");
    const auto larger = nnci::detail::make_cuda_batch_limits(16 * gib, 256, 100000);
    require(larger.bytes == 12 * gib && larger.datasets == 2048,
            "Both byte and dataset caps must grow beyond the old fixed limits");
    const auto low = nnci::detail::make_cuda_batch_limits(300 * mib, 64, 100000);
    require(low.bytes == 44 * mib, "Low-memory budgets must preserve the 256 MiB reserve");
    const auto grid = nnci::detail::make_cuda_batch_limits(8 * gib, 64, 37);
    require(grid.datasets == 37, "The device grid limit must bound the dataset count");
    for (const auto free : {std::size_t{0}, std::size_t{1}, 256 * mib}) {
        const auto empty = nnci::detail::make_cuda_batch_limits(free, 64, 100000);
        require(empty.bytes == 0 && empty.datasets == 0,
                "Insufficient memory must yield no allocation budget, never wrap or force one dataset");
    }
    const auto tiny = nnci::detail::make_cuda_batch_limits(256 * mib + 1, 64, 100000);
    require(tiny.datasets == 0, "A byte budget below minimum dataset storage must not admit work");
    const auto maximum = std::numeric_limits<std::size_t>::max();
    const auto extreme = nnci::detail::make_cuda_batch_limits(maximum, maximum, maximum);
    require(extreme.bytes < maximum && extreme.datasets > 0 &&
            extreme.datasets <= extreme.bytes / nnci::detail::perceptron_cuda_dataset_bytes(1, 1),
            "Occupancy and budget arithmetic must not overflow");
}

void test_shape_budget_boundaries() {
    const auto bytes = nnci::detail::perceptron_cuda_dataset_bytes(3, 2);
    require(nnci::detail::perceptron_cuda_batch_size(3, 2, {bytes, 100}) == 1,
            "A dataset at the exact byte boundary must fit");
    bool rejected = false;
    try {
        (void)nnci::detail::perceptron_cuda_batch_size(3, 2, {bytes - 1, 100});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "A dataset one byte over budget must be rejected before generation");
    require(nnci::detail::perceptron_cuda_batch_size(3, 2, {bytes * 7, 4}) == 4,
            "Dataset occupancy must cap a memory-rich batch");
    require(nnci::detail::perceptron_cuda_batch_size(3, 2, {bytes * 3, 100}) == 3,
            "Actual shape bytes must cap a memory-limited batch");
    require(nnci::detail::perceptron_cuda_dataset_bytes(10000, 10000) > 64 * 1024 * 1024,
            "The shape estimator must not impose the removed fixed 64 MiB ceiling");
    for (const auto shape : {std::pair<std::size_t, std::size_t>{0, 1}, {1, 0},
                             {std::numeric_limits<std::size_t>::max(), 2}}) {
        bool overflow = false;
        try { (void)nnci::detail::perceptron_cuda_dataset_bytes(shape.first, shape.second); }
        catch (const std::runtime_error&) { overflow = true; }
        require(overflow, "Invalid and overflowing shapes must still be rejected");
    }
}
}

int main() {
    try {
        test_live_memory_and_occupancy_limits();
        test_shape_budget_boundaries();
        require(nnci::detail::perceptron_cuda_epoch_chunk(1) == 8 &&
                nnci::detail::perceptron_cuda_epoch_chunk(1024 * 1024) == 1 &&
                nnci::detail::perceptron_cuda_epoch_chunk(std::numeric_limits<std::size_t>::max()) == 1,
                "Larger dynamic buffers must not keep eight huge epochs in one launch");
        std::cout << "PASS adaptive CUDA memory, occupancy and shape limits\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
