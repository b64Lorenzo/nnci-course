# NNCI Course Projects

A C++20 course project implementing the Rosenblatt perceptron and Assignment I experiments for random binary-labeled patterns. The command-line program supports a single training run, a fixed-dimension capacity study, a feature-dimension sweep, and an embedding-strength histogram for converged runs.

**The complete documentation is in [`doc/`](doc/README.md).** Start with the [user guide](doc/user-guide.md); see [algorithm notes](doc/algorithm.md), [development and testing](doc/development.md), and [troubleshooting](doc/troubleshooting.md) for details.

## Quick start

### Windows release package

Download `nnci-windows-x64.zip` from the [GitHub Releases page](https://github.com/b64Lorenzo/nnci-course/releases), extract the entire archive, and keep `nnci.exe`, `scripts/`, `doc/`, and the README files together. Plotting requires Python 3 and Matplotlib. Install Matplotlib with the same Windows Python launcher the program checks:

```powershell
py -3 -m pip install -r requirements-plot.txt
```

Then run a study from PowerShell in the extracted folder:

```powershell
.\nnci.exe perceptron --capacity --dimensions 20 --datasets 50 --max-epochs 100 --p-min 15 --p-max 60 --p-step 5 --seed 2026
```

The package resolves its plotting scripts relative to `nnci.exe`, so the executable can be launched from another working directory. Result and log paths are relative to the **current working directory**.

### Build from source

Requirements: CMake 3.20 or newer, a C++20 compiler, and (only for plotting) Python 3 with Matplotlib.

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
py -3 -m pip install -r requirements-plot.txt
.\build\Release\nnci.exe --help
```

Use the appropriate `-A` option for your Visual Studio architecture. With a single-configuration CMake generator, select the build type at configure time and omit `--config` from the build/test commands as appropriate.

## Command examples

```powershell
# One randomly generated dataset; defaults to P=N=20.
.\nnci.exe perceptron --dimensions 20 --patterns 15 --max-epochs 100 --seed 2026

# Sweep P for one N; writes a CSV and a plot automatically.
.\nnci.exe perceptron --capacity --dimensions 20 --datasets 50 --max-epochs 100 --p-min 15 --p-max 60 --p-step 5

# Sweep N and create one combined plot. The supplied P interval is used at every N.
.\nnci.exe perceptron --NCapacity 20 60 20 --datasets 50 --max-epochs 100 --p-min 15 --p-max 60 --p-step 5

# Histogram of pattern-indexed strengths from converged trials only.
.\nnci.exe perceptron --strength --dimensions 20 --patterns 15 --datasets 50 --max-epochs 100
```

`N` is the number of features; `P` is the number of patterns/examples. They are separate quantities. `--capacity` and `--NCapacity` are alternatives; run `--strength` separately. This release is CPU-only; there is no `--cuda` option.

## Outputs

Capacity studies write CSV files under `results/` and create PNG plots. `--NCapacity` also writes `results/capacity_N_sweep.csv` and `results/capacity_N_sweep.png`. Strength studies write `results/strengths.csv` and `results/strength_histogram.png`. Each run writes a log under `logs/<UTC-run-id>/run.log`. Generated results, logs, and build output are not tracked by Git.

## Documentation

- [`doc/README.md`](doc/README.md) — documentation index and package contents
- [`doc/user-guide.md`](doc/user-guide.md) — installation, CLI options, study modes, and outputs
- [`doc/algorithm.md`](doc/algorithm.md) — model, learning rule, capacity estimate, and strength definition
- [`doc/development.md`](doc/development.md) — build, tests, project layout, and release packaging
- [`doc/troubleshooting.md`](doc/troubleshooting.md) — common setup and runtime issues
- [`assignments/README.md`](assignments/README.md) — Assignment I context and experiment examples

## License

The MIT license covers original project code and documentation. The course handout in `assignments/` retains its own terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) before redistributing course materials.
