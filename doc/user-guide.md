# User guide

## 1. Install and prepare

### Windows release

1. Download `nnci-windows-x64.zip` from the project's GitHub Releases page.
2. Extract the **entire** archive. Do not separate `nnci.exe` from `scripts/`, `doc/`, or the plotting requirements file.
3. Open PowerShell in the extracted folder and install the plotting dependency using the Python launcher checked by the executable:

   ```powershell
   py -3 -m pip install -r requirements-plot.txt
   ```

4. Confirm the application and options:

   ```powershell
   .\nnci.exe --version
   .\nnci.exe --help
   ```

The executable can be called from a different working directory because it locates the plotting scripts beside itself. CSV, image, and log paths are relative to the working directory from which the command is launched.

### Build from source

Install CMake 3.20+ and a C++20 compiler. From a Visual Studio Developer PowerShell or a terminal where CMake can find the Visual Studio generator:

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
py -3 -m pip install -r requirements-plot.txt
.\build\Release\nnci.exe --help
```

If you use a single-configuration generator such as Ninja, configure with `-DCMAKE_BUILD_TYPE=Release`; omit the `--config Release` selector when building and testing. Plotting is the only feature that needs Python/Matplotlib; the C++ executable and non-plot build do not depend on Python.

## 2. Command shape

```text
nnci [--debug] perceptron [options]
nnci --version
```

`--help` prints the supported syntax. `--debug` may be placed before the command to enable detailed logs. The `perceptron` command always trains a single randomly generated dataset unless one of the study flags is selected.

### Parameters

| Option | Meaning | Default |
|---|---|---:|
| `--dimensions N` | Number of features in each example | 20 |
| `--patterns P` | Number of examples in a single run or each strength trial | `N` |
| `--datasets nD` | Independent datasets per P in a capacity study; number of trials in a strength study | 50 |
| `--max-epochs nmax` | Maximum number of complete training sweeps per dataset | 100 |
| `--p-min P` | First P value in a capacity sweep | `max(1, N/10)` |
| `--p-max P` | Largest allowed P in a capacity sweep | `3N` |
| `--p-step P` | Increment between P values | `max(1, N/10)` |
| `--seed SEED` | Seed for deterministic pseudorandom generation | 2026 |
| `--threshold c` | Potential threshold that triggers a weight update | 0.0 |
| `--output FILE` | Capacity CSV output path | `results/capacity_N<N>.csv` |

`N` and `P` are independent: `N` counts input features and `P` counts labeled examples/patterns. They are not interchangeable. The ratio `alpha = P/N` is computed for capacity results. P options are integer pattern counts, not alpha values.

Capacity P points start at `p-min` and increase by `p-step` while they do not exceed `p-max`; the final point need not equal `p-max` if the step does not land on it. Positive values are required. `--datasets` and `--max-epochs` must also be positive.

## 3. Run modes

Use only one study mode at a time. `--capacity` and `--NCapacity` cannot be combined; `--strength` must be run separately.

### Single training run

```powershell
.\nnci.exe perceptron --dimensions 20 --patterns 15 --max-epochs 100 --seed 2026
```

The program prints whether training converged within the epoch limit, the number of epochs used, and the number of weight updates. If `--patterns` is omitted, `P=N`.

### Capacity versus P at one N

```powershell
.\nnci.exe perceptron --capacity --dimensions 20 --datasets 50 --max-epochs 100 --p-min 15 --p-max 60 --p-step 5 --seed 2026
```

For every requested P, the program generates `nD` independent datasets and reports the empirical fraction that converged within `nmax` epochs. It writes the CSV path selected by `--output` (or `results/capacity_N20.csv` by default), then automatically creates a PNG beside that CSV by replacing its extension.

### Capacity sweep over N

```powershell
.\nnci.exe perceptron --NCapacity 20 60 20 --datasets 50 --max-epochs 100 --p-min 15 --p-max 60 --p-step 5 --seed 2026
```

The three positional values after `--NCapacity` are `START END STEP`. The sweep includes `START`, then adds `STEP` while the next value is no greater than `END`; `END` is included only when the interval lands on it. Each N gets its own capacity CSV, and all requested dimensions are combined into one `results/capacity_N_sweep.csv` and one `results/capacity_N_sweep.png`.

If supplied, `--p-min`, `--p-max`, and `--p-step` are the same absolute P counts for every N in the sweep. If omitted, defaults are recomputed for each N: `p-step=max(1,N/10)`, `p-min=p-step`, and `p-max=3N`. Choose custom P values that form a valid positive interval for every swept dimension. `--output` does not replace the fixed per-N/sweep filenames in this mode.

### Embedding-strength histogram

```powershell
.\nnci.exe perceptron --strength --dimensions 20 --patterns 15 --datasets 50 --max-epochs 100 --seed 2026
```

Each trial uses one randomly generated dataset with the specified N and P. Only trials that converge within the epoch limit contribute values. The program writes `results/strengths.csv` and automatically plots `results/strength_histogram.png`. See [Algorithm and experiment notes](algorithm.md) for the definition of `x_mu` and what the histogram counts.

## 4. Outputs

Output paths are relative to the process's current working directory, not the executable's directory. The default study files are:

```text
results/
├── capacity_N<N>.csv             # --capacity
├── capacity_N<N>.png             # --capacity
├── capacity_N<N>.csv             # one per N in --NCapacity
├── capacity_N_sweep.csv          # combined --NCapacity data
├── capacity_N_sweep.png          # combined --NCapacity plot
├── strengths.csv                 # --strength
└── strength_histogram.png         # --strength

logs/
└── <UTC-run-id>/run.log
```

Capacity CSV columns are `N,P,alpha,nD,nmax,successes,Q_ls,seed`. Strength CSV columns are `successful_case,mu,x_mu`, where `mu` identifies a pattern within a successful trial. Logs are also written to stderr. Generated `results/`, `logs/`, and build files are ignored by Git.

## 5. Plot dependencies and manual plotting

The study flags automatically call the Python plotter included beside the executable. On Windows, the executable checks `py -3` first and uses it only if Matplotlib imports successfully; it then tries `python`. Install with `py -3 -m pip ...` to target the same interpreter:

```powershell
py -3 -m pip install -r requirements-plot.txt
```

If you are using the source tree, you can invoke plotters directly:

```powershell
py -3 scripts/plot_capacity.py results/capacity_N20.csv --output results/capacity_N20.png
py -3 scripts/plot_capacity.py results/capacity_N_sweep.csv --output results/capacity_N_sweep.png
py -3 scripts/plot_strength.py results/strengths.csv --output results/strength_histogram.png
```

For several dimensions in one capacity plot, pass all of their CSV files as positional arguments to `plot_capacity.py`.

## 6. Reproducibility and interpretation

The default seed is 2026. Reusing the same seed and options gives repeatable pseudorandom data with the same toolchain. Record N, P bounds, dataset count, epoch limit, threshold, and seed when reporting a result. `Q_ls` is empirical convergence within a finite epoch limit, not a guarantee that the data are linearly inseparable when training stops early. See [Algorithm and experiment notes](algorithm.md) for the theory curve and its assumptions.

The implementation is CPU-only. CUDA/GPU execution is not available in this release.
