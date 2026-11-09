import argparse
import csv
import math
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


def read_capacity_csv(path: str | Path) -> list[dict[str, int | float]]:

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install matplotlib: py -m pip install -r requirements-plot.txt") from error

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

            rows.append({
                "N": dimensions,
                "P": patterns,
                "nD": datasets,
                "successes": successes,
                "Q_ls": rate,
            })

    if not rows:
        raise ValueError(f"{path}: no capacity results found")
    return rows


def plot_capacity(csv_paths: list[str | Path], output_path: str | Path) -> None:


    by_dimension: dict[int, list[dict[str, int | float]]] = {}
    for path in csv_paths:
        for row in read_capacity_csv(path):
            by_dimension.setdefault(int(row["N"]), []).append(row)

    figure, axis = plt.subplots(figsize=(8, 5))
    for dimensions, rows in sorted(by_dimension.items()):
        rows.sort(key=lambda row: int(row["P"]))
        patterns = [int(row["P"]) for row in rows]

        if len(patterns) != len(set(patterns)):
            raise ValueError(f"duplicate P values for N={dimensions}")

        alpha = [p / dimensions for p in patterns]
        empirical = [float(row["Q_ls"]) for row in rows]
        theory = [theoretical_probability(dimensions, p) for p in patterns]

        axis.plot(alpha, empirical, marker="o", label=f"empirical, N={dimensions}")
        axis.plot(alpha, theory, linestyle="--", label=f"theory, N={dimensions}")

    axis.set_xlabel(r"$\alpha=P/N$")
    axis.set_ylabel(r"$Q_{\mathrm{l.s.}}$")
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
