import argparse
import csv
import math
import os
from fractions import Fraction
from math import comb
from pathlib import Path



_REQUIRED_COLUMNS = {"N", "P", "nD", "successes", "Q_ls"}


def theoretical_probability(dimensions: int, patterns: int) -> float:
    """Return the linear-separability probability for random labels."""
    if dimensions <= 0 or patterns <= 0:
        raise ValueError("N and P must be positive")

    terms = sum(comb(patterns - 1, k) for k in range(min(dimensions, patterns)))
    return float(Fraction(terms, 1 << (patterns - 1)))


def normal_approximation(dimensions: int, patterns: int) -> float:
    if dimensions <= 0 or patterns <= 0:
        raise ValueError("N and P must be positive")
    if patterns == 1:
        return 1.0
    z = (2 * dimensions - patterns) / math.sqrt(patterns - 1)
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def wilson_interval(successes: int, datasets: int) -> tuple[float, float]:
    if datasets <= 0 or not 0 <= successes <= datasets:
        raise ValueError("successes must be between zero and a positive dataset count")
    z = 1.96
    rate = successes / datasets
    denominator = 1.0 + z * z / datasets
    center = (rate + z * z / (2.0 * datasets)) / denominator
    margin = z * math.sqrt(
        rate * (1.0 - rate) / datasets + z * z / (4.0 * datasets**2)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def capacity_statistics(
    dimensions: int, patterns: int, datasets: int, successes: int
) -> dict[str, float]:
    rate = successes / datasets
    low, high = wilson_interval(successes, datasets)
    return {
        "Q_ls_variance_est": rate * (1.0 - rate) / datasets,
        "P_ls_exact": theoretical_probability(dimensions, patterns),
        "P_ls_normal": normal_approximation(dimensions, patterns),
        "Q_ls_wilson95_low": low,
        "Q_ls_wilson95_high": high,
    }


def read_capacity_csv(path: str | Path) -> list[dict[str, int | float]]:
    rows = []

    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None or not _REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected columns {', '.join(sorted(_REQUIRED_COLUMNS))}")

        for line_number, row in enumerate(reader, start=2):
            if not row or all(value is None for value in row.values()):
                continue
            try:
                dimensions = int(row["N"])
                patterns = int(row["P"])
                datasets = int(row["nD"])
                successes = int(row["successes"])
                rate = float(row["Q_ls"])

            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: invalid numeric value") from error

            if dimensions <= 0 or patterns <= 0 or datasets <= 0:
                raise ValueError(f"{path}:{line_number}: N, P, and nD must be positive")
            if not 0 <= successes <= datasets or not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
                raise ValueError(f"{path}:{line_number}: invalid success count or rate")
            if not math.isclose(rate, successes / datasets, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"{path}:{line_number}: Q_ls does not match successes/nD")

            statistics = capacity_statistics(dimensions, patterns, datasets, successes)
            for name, expected in statistics.items():
                if name not in reader.fieldnames:
                    continue
                try:
                    observed = float(row[name])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: invalid {name}") from error
                if not math.isfinite(observed) or not math.isclose(
                    observed, expected, rel_tol=1e-8, abs_tol=1e-10
                ):
                    raise ValueError(f"{path}:{line_number}: {name} is inconsistent")

            rows.append({
                "N": dimensions,
                "P": patterns,
                "nD": datasets,
                "successes": successes,
                "Q_ls": rate,
                **statistics,
            })

    if not rows:
        raise ValueError(f"{path}: no capacity results found")
    return rows


def plot_capacity(csv_paths: list[str | Path], output_path: str | Path) -> None:

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        installer = "py -3 -m pip" if os.name == "nt" else "python3 -m pip"
        raise RuntimeError(
            f"matplotlib is missing; install requirements-plot.txt with {installer}"
        ) from error

    by_dimension: dict[int, list[dict[str, int | float]]] = {}
    alpha_max = 2.0
    for path in csv_paths:
        for row in read_capacity_csv(path):
            dimensions, patterns = int(row["N"]), int(row["P"])
            alpha_max = max(alpha_max, patterns / dimensions)
            by_dimension.setdefault(dimensions, []).append(row)

    figure, axis = plt.subplots(figsize=(8, 5))
    for dimensions, rows in sorted(by_dimension.items()):
        rows.sort(key=lambda row: int(row["P"]))
        patterns = [int(row["P"]) for row in rows]

        if len(patterns) != len(set(patterns)):
            raise ValueError(f"duplicate P values for N={dimensions}")

        alpha = [p / dimensions for p in patterns]
        empirical = [float(row["Q_ls"]) for row in rows]
        lower = [float(row["Q_ls_wilson95_low"]) for row in rows]
        upper = [float(row["Q_ls_wilson95_high"]) for row in rows]
        exact = [float(row["P_ls_exact"]) for row in rows]
        normal = [float(row["P_ls_normal"]) for row in rows]

        lower_error = [max(0.0, q - low) for q, low in zip(empirical, lower)]
        upper_error = [max(0.0, high - q) for q, high in zip(empirical, upper)]
        empirical_line = axis.errorbar(
            alpha,
            empirical,
            yerr=[lower_error, upper_error],
            fmt="o-",
            capsize=3,
            label=f"empirical (95% Wilson), N={dimensions}",
        )
        color = empirical_line.lines[0].get_color()
        axis.plot(
            alpha, exact, color=color, linestyle="--",
            label=f"Cover exact, N={dimensions}",
        )
        axis.plot(
            alpha, normal, color=color, linestyle=":",
            label=f"CLT approximation, N={dimensions}",
        )

    axis.plot(
        [0.0, 2.0, 2.0, alpha_max], [1.0, 1.0, 0.0, 0.0],
        color="black", linestyle="-.",
        label="thermodynamic limit ($\\alpha_c=2$)",
    )
    axis.scatter(
        [2.0], [0.5], color="black", marker="D", zorder=5,
        label=r"$P=2N$: $P_{\mathrm{ls}}=1/2$",
    )
    axis.set_xlabel(r"$\alpha=P/N$")
    axis.set_ylabel(r"Probability ($P_{\mathrm{ls}}$, $Q_{\mathrm{ls}}$)")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlim(left=0.0)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Saved plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot empirical perceptron separability results.")
    parser.add_argument("csv", nargs="+", help="one or more capacity-study CSV files")
    parser.add_argument("--output", default="results/a1_capacity.png", help="output plot path")
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")

    arguments = parser.parse_args()

    try:
        plot_capacity(arguments.csv, arguments.output)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
