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

Use `nnci.exe --help` for all options.

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

The plotting scripts must stay beside `nnci.exe`. CSVs, plots, and logs are written relative to the current working directory.

## More

- [Algorithm and experiment definitions](doc/algorithm.md)
- [Build and source layout](doc/development.md)
- [Assignment I run notes](assignments/README.md)
- [License scope](THIRD_PARTY_NOTICES.md)
