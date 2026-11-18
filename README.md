# NNCI Course Project

A C++20 Rosenblatt perceptron for Assignment I. It generates random binary-labeled data and supports single runs, capacity studies, N sweeps, and strength plots.

## Build and test

Requires CMake 3.20+, a C++20 compiler, and Python 3 with Matplotlib for plotting and plot tests.

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release
py -3 -m pip install -r requirements-plot.txt
ctest --test-dir build -C Release --output-on-failure
```

## Run

Use `nnci.exe --help` for all options. Run this from the build output directory to check the fixed separable and non-separable cases:

```powershell
nnci.exe perceptron --validation
```

```powershell
# One training run
.\build\Release\nnci.exe perceptron --dimensions 20 --patterns 15 --seed 2026

# Capacity versus P for N=20
.\build\Release\nnci.exe perceptron --capacity --dimensions 20 --p-min 15 --p-max 60 --p-step 5

# Capacity sweep over N
.\build\Release\nnci.exe perceptron --NCapacity 20 60 20

# Strength histogram
.\build\Release\nnci.exe perceptron --strength --dimensions 20 --patterns 15
```

The default epoch limit is `N * nD` (`--dimensions` times `--datasets`), recalculated for every N in a sweep. Use `--max-epochs` to supply a fixed positive override.

The plotting scripts must stay beside `nnci.exe`. CSVs, plots, and logs are written relative to the current working directory.

## Optional CUDA backend

CPU execution is the default, including in CUDA-enabled builds. Build the optional backend with a CUDA Toolkit and a supported C++20 host compiler. For example, from a Visual Studio Developer Command Prompt with Ninja available:

```powershell
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DNNCI_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda
ctest --test-dir build-cuda --output-on-failure
.\build-cuda\nnci.exe perceptron --validation --cuda
.\build-cuda\nnci.exe perceptron --capacity --dimensions 20 --datasets 2048 --cuda
```

`--cuda` is a bare opt-in flag for single runs, capacity/N sweeps, strength studies, and validation. The build option `NNCI_ENABLE_CUDA` defaults to `OFF`; explicit `ON` requires a working CUDA compiler. Missing CUDA build support, a device, or a compatible driver produces a clear error, never a CPU fallback. A supported NVIDIA driver is required at runtime; the CUDA runtime is linked statically.

The backend parallelizes independent datasets, not the sequential updates within one dataset. Host RNG and trial order remain unchanged. Batches are bounded to 1024 datasets and 64 MiB of resident device buffers; a dataset that cannot fit is rejected. Work-bounded kernels resume even within a feature loop to avoid one unbounded epoch launch. Small or single-dataset jobs can be slower than CPU execution. See [development](doc/development.md) for architecture overrides and GPU test selection.

## More

- [Algorithm and experiment definitions](doc/algorithm.md)
- [Build and source layout](doc/development.md)
- [Assignment I run notes](assignments/README.md)
- [License scope](THIRD_PARTY_NOTICES.md)
