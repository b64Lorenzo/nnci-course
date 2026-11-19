# Development

Requires CMake 3.20+ and a C++20 compiler. Python 3 with Matplotlib is needed for plotting and plot tests.

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release
py -3 -m pip install -r requirements-plot.txt
ctest --test-dir build -C Release --output-on-failure
```

For single-configuration generators, set `-DCMAKE_BUILD_TYPE=Release` and omit `--config Release`.

## Layout

- `include/nnci/` — public library headers.
- `src/` — perceptron and study implementation.
- `src/app/` — private CLI, option parsing, output, and plotting code.
- `scripts/` — Python plotting scripts.
- `tests/` — C++ and Python tests.

`VERSION` is the release version. GitHub Actions builds, tests, and packages the Windows release.

## CUDA build and verification

`NNCI_ENABLE_CUDA=OFF` needs no CUDA compiler, toolkit, or runtime. `ON` enables the CUDA language and requires the toolkit; configuration fails rather than silently dropping support. CUDA 13.4.1 supports Visual Studio 2026/MSVC 195x and C++20 ([NVIDIA host compiler table](https://docs.nvidia.com/cuda/archive/13.4.1/cuda-installation-guide-microsoft-windows/index.html#system-requirements)). Use a supported toolkit/compiler combination; no unsupported-compiler flags are needed.

From a Developer Command Prompt with Ninja, follow the optional build in the main README. Set `CMAKE_CUDA_COMPILER` and `CUDAToolkit_ROOT` explicitly if the toolkit is not on the search path. With no architecture override, the project defaults to `89` (Ada, including RTX 4070). `CMAKE_CUDA_ARCHITECTURES` and `CUDAARCHS` overrides are preserved. For a multi-generation package, an example override is `-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"`; each selected architecture must be supported by the chosen toolkit. Unsuffixed architecture values include real machine code and virtual PTX ([CMake architecture semantics](https://cmake.org/cmake/help/v4.3/prop_tgt/CUDA_ARCHITECTURES.html)). This does not guarantee every future GPU/driver combination.

The CUDA runtime is static ([CMake CUDA_RUNTIME_LIBRARY](https://cmake.org/cmake/help/v4.3/prop_tgt/CUDA_RUNTIME_LIBRARY.html)); users still need a supported NVIDIA driver. CUDA flags are language-scoped so C++ warning flags are not passed to NVCC.

Run the full suite in both CPU-only and CUDA-enabled builds. Native GPU tests are deliberately not auto-skipped when a CUDA-enabled build has no usable device. They carry the `gpu` label, so a machine without a GPU can run the CPU regression subset using:

```powershell
ctest --test-dir build-cuda -LE gpu --output-on-failure
# On a real compatible GPU, also run:
ctest --test-dir build-cuda -L gpu --output-on-failure
```

`nnci_cuda_differential` checks an independent scalar batch reference, hand-calculated frozen-epoch updates, strict positive-minimum convergence, no-active/cancelling sums, mixed N/P options, long feature/pattern loops, subnormals, memory limits, and numerical errors. Normal-scale weights allow `1e-12 * (1 + abs(reference))` reduction roundoff; counters, strengths and convergence are exact, and deterministic reruns and the subnormal fixture require exact weights. `perceptron_cuda_cli` checks repeatable CUDA stdout/CSV bytes in every mode, automatic/explicit per-N budgets, batched N/P tasks and tails, zero-success negative-threshold cases, CPU-parallel parity and CPU-default operation with devices hidden. Online CPU and batch CUDA outputs are intentionally not equated. Checks remain active in Release builds.

For CPU capacity studies, `--parallelize` opts into a bounded worker pool that runs independent P/trial jobs concurrently; `--NCapacity` also schedules its independent N studies through the pool. The worker count is limited by available hardware concurrency and total work, and queued work is batched under a 64 MiB host-memory estimate. CPU execution stays sequential by default. `--parallelize` requires `--capacity` or `--NCapacity` and cannot be combined with `--cuda`.

## CUDA backend invariants

The private CUDA backend contract is an ordered batch of datasets with per-dataset shape and training options in `src/perceptron_backend.hpp`, implemented by `src/perceptron_cuda.cu` or the CPU-only stub. The public `estimate_perceptron_capacity_parallel` API schedules capacity studies in CPU-parallel or CUDA mode; existing public training/study functions append `bool cuda = false` after the logger parameter, preserving existing source calls. The CLI retains one `PerceptronOptions` and parser.

The host retains `mt19937_64`, normal/Bernoulli distributions, seed consumption, and trial/result order. Independent CPU workers retain online training. Each CUDA dataset owns one 256-thread cooperative block: freeze a pattern-selection mask, reduce Hebbian terms in coalesced 32-feature/eight-pattern tiles, divide the sum by N once, then reduce the new signed potentials to their minimum. Shared-memory barriers separate every phase; warp shuffles reduce dot products and minima ([NVIDIA synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html), [warp intrinsics](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-shuffle-functions)). There are no floating-point atomics. FP64 contraction remains disabled with `--fmad=false`; no fast math is used. Fixed reduction trees are repeatable but can differ from scalar addition order ([NVIDIA floating-point discussion](https://docs.nvidia.com/cuda/archive/12.9.1/floating-point/index.html)). Reproducibility is for a fixed toolchain/device, not across standard-library RNG implementations.

A CUDA batch contains at most 1024 datasets and 64 MiB of resident buffers, including inputs, labels, weights, strengths, potentials, selection masks, descriptors, and training state. Mixed N/P shapes and per-dataset thresholds/epoch limits remain supported. Checked shape arithmetic runs before generation/allocation; the new potential/mask buffers are included in the bound. Host generation, flattened transfers and result vectors consume additional batch-sized memory; successful strength output still scales with requested trials. An over-budget dataset is rejected. A kernel runs at most eight epochs per dataset, preserving weights, potentials and counters between launches. The resident-memory bound also bounds work within an epoch; no watchdog/TDR settings are changed. This is a bounded launch policy, not a hard wall-clock guarantee on arbitrary hardware.

Device allocations use RAII; allocation, transfer, launch, and synchronization failures are checked and reported, never retried on CPU. Invalid datasets/options/numerical failures retain codes 10/11/12; CUDA infrastructure failures use the CLI's generic code 1. Non-finite thresholds are explicitly invalid options. Backend/device information is logged on the host once per process when a logger is supplied; debug mode reports batch sizes and launch counts, not per-epoch CUDA diagnostics.

Host data generation, transfers, synchronization, and FP64 throughput can dominate. There is no universal speedup claim; single-run/validation mode uses a singleton batch.
