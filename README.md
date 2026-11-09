# NNCI Course Projects

A compact C++20 repository for the Neural Networks and Computational Intelligence course. The implementation here focuses on Assignment I: the Rosenblatt perceptron and its linear-separability study. The perceptron source file contains only the training algorithm; data generation and experiment code live separately.

## Build and test

From a Visual Studio Developer Command Prompt or a terminal with CMake and a C++20 compiler:

```text
cmake -S . -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

Run on Windows:

```text
.\build\Debug\nnci.exe --help
.\build\Debug\nnci.exe a1-study --dimensions 20 --datasets 50 --max-epochs 100
py -m pip install -r requirements-plot.txt
py scripts/plot_capacity.py results/capacity_N20.csv --output results/capacity_N20.png
```

CMake defaults to the available generator. To pick one explicitly, check `cmake --help` for installed generators.

## Assignment I study

`a1-study` sweeps P, generates independent Gaussian feature vectors and random ±1 labels, trains each dataset, and writes the empirical success fraction `Q_ls` to CSV. Defaults are `N=20`, 50 datasets per P, 100 sweeps maximum, and P/N from 0.1 to 3.0 in steps of 0.1. Set `--dimensions 40` to run the second handout size; use `--p-min`, `--p-max`, and `--p-step` to adjust the P range.

For example, collect both N settings and plot empirical results against the theoretical curve:

```text
.\build\Debug\nnci.exe a1-study --dimensions 40 --output results/capacity_N40.csv
py scripts/plot_capacity.py results/capacity_N20.csv results/capacity_N40.csv --output results/capacity.png
```

The study uses a fixed seed by default; change it with `--seed`. The dashed theory curve uses the homogeneous, general-position result $P_{\mathrm{ls}}(N,P)=2^{1-P}\sum_{k=0}^{N-1}\binom{P-1}{k}$. Since `Q_ls` counts convergence within the finite `nmax`, it can sit below theory; raise `--max-epochs` to check whether runs are merely unfinished. Install the optional plotting dependency once with `py -m pip install -r requirements-plot.txt`. Generated CSV and image files under `results/` are ignored by Git.

This repository currently implements Assignment I. Its handout and short run guide are in [`assignments/`](assignments/README.md).

## Debugging

Use a Debug build. In Visual Studio, open this folder as a CMake project, select `nnci` as the startup target, and press **F5**. Set breakpoints in `src/main.cpp` or `src/rosenblatt_perceptron.cpp`. Run `nnci --debug a1-demo` to log each epoch. The training function accepts an optional `Logger*`; invalid input and numerical failures throw `PerceptronError` with stable codes 10 (invalid dataset), 11 (invalid options), and 12 (numerical failure). The CLI returns that code for perceptron errors.

## Logging

Each run creates `logs/<UTC-run-id>/run.log`. Records include timestamp, level, component, and message; they are written to both the log file and stderr. The logger filters by minimum level and escapes line breaks and delimiters so each event stays on one line. Logs and build output are ignored by Git.

## Layout

- `include/nnci/` — public logger and perceptron/study interfaces
- `src/rosenblatt_perceptron.cpp` — the training algorithm only
- `src/assignment1_study.cpp` — random data generation and repeated capacity experiments
- `scripts/plot_capacity.py` — CSV validation and empirical/theoretical plot
- `src/main.cpp` — simple demo and study commands
- `tests/` — minimal focused A1 checks
- `assignments/` — Assignment I handout and run guide

Keep each experiment reproducible: record its parameters and random seed, separate training from test data, and save generated results under that run's output directory. Do not commit private course datasets or raw experiment outputs.

## Continuous integration

`.github/workflows/ci.yml` builds and runs CTest on Windows and Ubuntu for pushes and pull requests.

## License and materials

The MIT license applies only to original repository code and general documentation. The included course handout remains under its original terms and is not covered by MIT; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## AI assistance

AI tools were used to improve the text, assist with code autocompletion, help set up this directory's CMake configuration (adapted from one of my older repositories), and help write tests for functions.
