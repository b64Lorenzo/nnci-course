# Troubleshooting

## Plot reports that Matplotlib is missing

The Windows executable checks `py -3` first, then `python`, and chooses an interpreter only when it can import Matplotlib. Install into the same interpreter it probes:

```powershell
py -3 -m pip install -r requirements-plot.txt
py -3 -c "import matplotlib; print(matplotlib.__version__)"
```

If you have multiple Python installations, compare `py -0p` with `py -3 -c "import sys; print(sys.executable)"`. Installing Matplotlib into another interpreter (for example, `python -m pip` when `python` resolves to a different environment) will not satisfy the executable's check. Re-run the command from the extracted package and keep its `requirements-plot.txt` nearby for installation.

## Plot script is missing

Use the full `nnci-windows-x64.zip` release package and extract all of it. The expected layout is:

```text
nnci-windows-x64/
├── nnci.exe
├── README.md
├── README.txt
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── requirements-plot.txt
├── scripts/
│   ├── plot_capacity.py
│   └── plot_strength.py
└── doc/
```

Do not download or copy only `nnci.exe`. The app resolves scripts relative to the executable, but it still needs those script files in `scripts/`.

## Results are not in the package directory

Result and log folders are relative to the current working directory, while plotting scripts are resolved relative to the executable. If you launched the executable from another folder, look for `results/` and `logs/` there. Change directory before running or provide an absolute `--output` path for `--capacity` CSV output.

## No strength histogram was created

The strength study plots values only from trials that converged within `--max-epochs`. If no trial converges, the command stops with a clear error rather than making an empty histogram. Try a larger epoch limit, fewer patterns, more dimensions, or inspect the run log; record any changed parameters in your analysis.

## Capacity result differs from the theory curve

`Q_ls` is measured using a finite number of generated datasets (`--datasets`) and a finite training limit (`--max-epochs`). Sampling variability and runs that simply need more epochs can both reduce the empirical fraction. Increase the dataset count for a more stable estimate and increase the epoch limit to check whether apparent failures are unfinished; use a fixed seed when comparing changes.

## CLI rejects an option or value

Run `nnci.exe --help`. The supported command is `perceptron`; all study flags go after it. For example, use `nnci.exe perceptron --capacity ...`, not an obsolete `a1-study` command. `--NCapacity` requires exactly three values (`START END STEP`). Use `--capacity` or `--NCapacity`, not both; run `--strength` separately. N, P, counts, and steps must be positive, and the pattern range must be ordered.

## CUDA/GPU question

The current implementation is CPU-only and has no `--cuda` option. Use the standard `perceptron` command; the program does not silently choose or emulate a CUDA backend.
