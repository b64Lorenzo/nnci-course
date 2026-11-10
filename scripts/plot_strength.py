import argparse
import csv
import math
from pathlib import Path


def read_strengths(path: str | Path) -> list[int]:
    values: list[int] = []

    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "x_mu" not in reader.fieldnames:
            raise ValueError(f"{path}: expected an 'x_mu' column")

        for line_number, row in enumerate(reader, start=2):
            try:
                value = float(row["x_mu"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid x_mu value"
                ) from error

            if not math.isfinite(value) or value < 0 or not value.is_integer():
                raise ValueError(
                    f"{path}:{line_number}: x_mu must be a non-negative integer"
                )

            values.append(int(value))

    if not values:
        raise ValueError(f"{path}: no embedding strengths found")

    return values


def plot_strength(csv_path: str | Path, output_path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Install matplotlib: py -m pip install -r requirements-plot.txt"
        ) from error

    values = read_strengths(csv_path)

    largest = max(values)
    bins = [strength - 0.5 for strength in range(largest + 2)]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=bins, edgecolor="black", align="mid")
    axis.set_xticks(range(largest + 1))
    axis.set_xlabel("Embedding strength x^μ (updates for pattern μ)")
    axis.set_ylabel("Frequency among patterns in converged runs")
    axis.set_title("Embedding-strength distribution in successful runs")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Saved plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Rosenblatt embedding strengths x_mu from converged runs."
    )
    parser.add_argument("csv", help="CSV containing converged-run x_mu values")
    parser.add_argument(
        "--output",
        default="results/strength_histogram.png",
        help="output image path",
    )
    arguments = parser.parse_args()

    try:
        plot_strength(arguments.csv, arguments.output)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
