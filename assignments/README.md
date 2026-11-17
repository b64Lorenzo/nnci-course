# Assignment I — Rosenblatt perceptron

See the [course handout](a1_2025.pdf) for the full assignment.

## Baseline experiment

The handout uses `N = 20, 40`, `alpha = 0.75, 1.00, …, 3.00`, 50 datasets per P, and 100 epochs. Since `P = alpha × N`, run:

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release
py -3 -m pip install -r requirements-plot.txt

.\build\Release\nnci.exe perceptron --capacity --dimensions 20 --datasets 50 --max-epochs 100 --p-min 15 --p-max 60 --p-step 5 --seed 2025 --output results/capacity_N20.csv
.\build\Release\nnci.exe perceptron --capacity --dimensions 40 --datasets 50 --max-epochs 100 --p-min 30 --p-max 120 --p-step 10 --seed 2040 --output results/capacity_N40.csv
py -3 scripts/plot_capacity.py results/capacity_N20.csv results/capacity_N40.csv --output results/capacity.png
```

## Report

Plot empirical success against `alpha`, compare it with theory, and note the finite epoch limit and random seed. CSVs contain aggregate results only; raw feature vectors and labels are not saved. See [license scope](../THIRD_PARTY_NOTICES.md) before redistributing the handout.
