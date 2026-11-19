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

# Opt in to bounded CPU parallelism across independent capacity/N trials
.\build\Release\nnci.exe perceptron --NCapacity 20 60 20 --parallelize

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

`--parallelize` opts capacity and N-capacity studies into a bounded CPU worker pool for independent P/trial and N tasks. Without it, CPU studies remain sequential. It cannot be combined with `--cuda` and is rejected without a capacity study.

`--cuda` is a bare opt-in flag for single runs, capacity/N sweeps, strength studies, and validation. The build option `NNCI_ENABLE_CUDA` defaults to `OFF`; explicit `ON` requires a working CUDA compiler. Missing CUDA build support, a device, or a compatible driver produces a clear error, never a CPU fallback. A supported NVIDIA driver is required at runtime; the CUDA runtime is linked statically.

CUDA capacity/NCapacity execution still batches independent P, N, and trial datasets together. Since v0.3.2, each dataset also uses synchronous batch learning: classify every pattern from the same weights, reduce all selected Hebbian terms in parallel, apply their sum divided by N once, and recompute all potentials in parallel. Convergence requires `min(E) > 0`; no selected patterns alone is not convergence. CPU training remains online, so CPU and CUDA results need not match. See [algorithm](doc/algorithm.md) for the exact recurrence.

Host RNG and result order remain unchanged. CUDA batches use a live free-VRAM budget, leaving at least 256 MiB or 25% of the currently free VRAM unused (whichever is larger). The dataset cap follows kernel occupancy and GPU size, then actual dataset bytes further limit each batch. Limits refresh between batches. Large datasets use shorter epoch chunks, up to eight epochs per launch. Host staging requires additional RAM; competing GPU users can still change availability after a query. Small jobs can be slower than CPU execution. See [development](doc/development.md) for details.

## More

- [Algorithm and experiment definitions](doc/algorithm.md)
- [Build and source layout](doc/development.md)
- [Assignment I run notes](assignments/README.md)
- [License scope](THIRD_PARTY_NOTICES.md)
