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

`nnci_cuda_differential` checks exact weights, epochs, updates, strengths, convergence, seeded parallel batches, launch continuation inside long feature/pattern loops, limits, and error codes. `perceptron_cuda_cli` compares CPU/GPU output and CSV bytes for every mode, checks automatic/explicit per-N epoch budgets, tests a 1024-trial batch plus its tail, and hides devices to verify CPU-default operation and explicit CUDA failure. CPU-only builds test clear CUDA rejection instead. Test checks remain active in Release builds.

## CUDA backend invariants

The sole private backend contract is an ordered dataset batch in `src/perceptron_backend.hpp`, implemented by `src/perceptron_cuda.cu` or the CPU-only stub. Public training/study functions append `bool cuda = false` after the logger parameter, preserving existing source calls. The CLI retains one `PerceptronOptions` and parser.

The host retains `mt19937_64`, normal/Bernoulli distributions, seed consumption, and trial/result order. Only independent trials are parallel: one GPU thread owns one dataset and performs its pattern and feature loops sequentially. Double-precision multiply/add contraction is disabled with `--fmad=false`; no fast math or reciprocal update rewrite is used ([NVCC fmad option](https://docs.nvidia.com/cuda/archive/13.4.1/cuda-compiler-driver-nvcc/index.html#fmad), [NVIDIA floating-point ordering discussion](https://docs.nvidia.com/cuda/archive/12.9.1/floating-point/index.html)). Reproducibility is for a fixed host toolchain, not across different standard-library random distributions.

A batch contains at most 1024 datasets and 64 MiB of resident buffers, including inputs, labels, weights, strengths, and training state. Checked shape arithmetic runs before batch generation/allocation. Host generation, flattened transfers, and result vectors also consume bounded batch-sized memory, but are not included in the device limit; accumulated successful strength output still scales with the requested trials. A single over-budget dataset is rejected with a CUDA infrastructure error. Each launch performs at most 4096 feature operations per thread and preserves partial dot products, updates, and epoch phase between launches. This bounds even a very large single epoch; no watchdog/TDR configuration is changed.

Device allocations use RAII; allocation, transfer, launch, and synchronization failures are checked and reported, never retried on CPU. Invalid datasets/options/numerical failures retain codes 10/11/12; CUDA infrastructure failures use the CLI's generic code 1. Non-finite thresholds are explicitly invalid options. Backend/device information is logged on the host once per process when a logger is supplied; debug mode reports batch sizes and launch counts, not per-epoch CUDA diagnostics.

Host data generation, transfers, synchronization, and FP64 throughput can dominate. There is no universal speedup claim; single-run/validation mode uses a singleton batch.
