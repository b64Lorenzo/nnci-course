# Assignment I — Rosenblatt perceptron

[Course handout: `a1_2025.pdf`](a1_2025.pdf)

## Task

Generate independent standard-normal feature components and independent labels `-1` or `+1` with equal probability. Start with zero weights and present examples cyclically. Use the `c = 0` rule from §3.3.3 (printed p. 39) of *The Shallow and the Deep*: update by `label * input / N` whenever stability `E <= 0`. After a sweep, stop once every example has `E > 0`; otherwise continue up to `nmax` sweeps. Repeat `nD` datasets per pattern count `P` and report `Q_ls = successes / nD` against `alpha = P/N`.

## Build and run

From the repository root:

```text
cmake -S . -B build
cmake --build build --config Release
```

The handout's baseline is `N = 20, 40`, `alpha = 0.75, 1.00, 1.25, ..., 3.00`, `nD = 50`, and `nmax = 100`. Use integer `P` ranges to match those alpha values:

```powershell
.\build\Release\nnci.exe a1-study --dimensions 20 --datasets 50 --max-epochs 100 --p-min 15 --p-max 60 --p-step 5 --seed 2025 --output results/capacity_N20.csv
.\build\Release\nnci.exe a1-study --dimensions 40 --datasets 50 --max-epochs 100 --p-min 30 --p-max 120 --p-step 10 --seed 2040 --output results/capacity_N40.csv
```

Install Matplotlib once, then plot both results:

```text
py -m pip install -r requirements-plot.txt
py scripts/plot_capacity.py results/capacity_N20.csv results/capacity_N40.csv --output results/capacity.png
```

## Inputs

`a1-study` accepts:

| Option | Meaning | Default |
|---|---|---:|
| `--dimensions N` | Number of features | 20 |
| `--datasets nD` | Independent datasets per `P` | 50 |
| `--max-epochs nmax` | Maximum full sweeps per dataset | 100 |
| `--p-min P` | First pattern count | `N/10` |
| `--p-max P` | Last pattern count | `3N` |
| `--p-step P` | Increment in pattern count | `N/10` |
| `--seed SEED` | Reproducible random seed | 2026 |
| `--output FILE` | CSV destination | `results/capacity_N<N>.csv` |

`N`, `nD`, `nmax`, and the pattern bounds/step must be positive, with `p-min <= p-max`. `P` options are counts, not alpha values. Run `.\build\Release\nnci.exe --help` for the full command list.

## Outputs and logs

The generated files are:

```text
results/
├── capacity_N20.csv
├── capacity_N40.csv
└── capacity.png

logs/
└── <UTC-run-id>/
    └── run.log
```

CSV columns are `N,P,alpha,nD,nmax,successes,Q_ls,seed`. Normal logs record command start/end and a summary for each `P`. Add `--debug` before `a1-study` to log each training run's epoch and update counts. Perceptron errors are logged with codes 10 (invalid dataset), 11 (invalid options), or 12 (numerical failure); invalid CLI arguments print to stderr. Logs are also written to stderr.

Raw feature vectors, labels, learned weight vectors, and per-example stabilities are **not** saved. The CSV stores parameters and aggregate success counts/rates only. `results/` and `logs/` are ignored by Git.

## Report

Include the empirical plot with labeled axes and a legend, compare it with the theoretical curve, and briefly discuss differences (including the finite `nmax`). Example hand-in:

```text
submission/
├── report.pdf
├── source/                 # CMake, include/, src/, tests/, scripts/
└── results/
    ├── capacity_N20.csv
    ├── capacity_N40.csv
    └── capacity.png
```
