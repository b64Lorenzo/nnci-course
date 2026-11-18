import csv
from math import comb, erfc, sqrt
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class NCapacityPatternRangeTest(unittest.TestCase):
    def test_explicit_pattern_range_is_used_for_each_dimension(self):
        executable = Path(sys.argv[1]).resolve()
        with tempfile.TemporaryDirectory(prefix="nnci-ncapacity-") as temporary:
            working_directory = Path(temporary)
            completed = subprocess.run(
                [
                    str(executable),
                    "perceptron",
                    "--NCapacity",
                    "2",
                    "4",
                    "2",
                    "--p-min",
                    "3",
                    "--p-max",
                    "5",
                    "--p-step",
                    "2",
                    "--datasets",
                    "2",
                    "--max-epochs",
                    "5",
                    "--seed",
                    "13",
                ],
                cwd=working_directory,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

            with (working_directory / "results/capacity_N_sweep.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))

        patterns_by_dimension = {}
        for row in rows:
            patterns_by_dimension.setdefault(int(row["N"]), []).append(int(row["P"]))

        self.assertEqual(patterns_by_dimension, {2: [3, 5], 4: [3, 5]})
        required = {
            "Q_ls_variance_est", "P_ls_exact", "P_ls_normal",
            "Q_ls_wilson95_low", "Q_ls_wilson95_high",
        }
        self.assertTrue(required.issubset(rows[0]))
        for row in rows:
            self.assertEqual(int(row["nmax"]), 5)
            dimensions, patterns = int(row["N"]), int(row["P"])
            datasets = int(row["nD"])
            successes = int(row["successes"])
            rate = successes / datasets
            variance = rate * (1.0 - rate) / datasets
            self.assertAlmostEqual(float(row["Q_ls_variance_est"]), variance)

            z = 1.96
            denominator = 1.0 + z * z / datasets
            center = (rate + z * z / (2.0 * datasets)) / denominator
            margin = z * sqrt(
                rate * (1.0 - rate) / datasets + z * z / (4.0 * datasets**2)
            ) / denominator
            self.assertAlmostEqual(float(row["Q_ls_wilson95_low"]), center - margin)
            self.assertAlmostEqual(float(row["Q_ls_wilson95_high"]), center + margin)

            expected = sum(comb(patterns - 1, k) for k in range(min(dimensions, patterns)))
            self.assertAlmostEqual(float(row["P_ls_exact"]), expected / 2 ** (patterns - 1))
            approximate = 1.0 if patterns == 1 else 0.5 * erfc(
                -(2 * dimensions - patterns) / sqrt(patterns - 1) / sqrt(2.0)
            )
            self.assertAlmostEqual(float(row["P_ls_normal"]), approximate)

    def test_default_epochs_follow_each_swept_dimension(self):
        executable = Path(sys.argv[1]).resolve()
        with tempfile.TemporaryDirectory(prefix="nnci-epochs-") as temporary:
            completed = subprocess.run(
                [str(executable), "perceptron", "--NCapacity", "2", "4", "2",
                 "--datasets", "3", "--p-min", "1", "--p-max", "1", "--p-step", "1"],
                cwd=temporary, capture_output=True, text=True, check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            with (Path(temporary) / "results/capacity_N_sweep.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["N"]) for row in rows], [2, 4])
            for row in rows:
                self.assertEqual(int(row["nmax"]), int(row["N"]) * int(row["nD"]))

    def test_default_p_max_is_derived_from_each_swept_dimension(self):
        executable = Path(sys.argv[1]).resolve()
        with tempfile.TemporaryDirectory(prefix="nnci-ncapacity-") as temporary:
            working_directory = Path(temporary)
            completed = subprocess.run(
                [
                    str(executable),
                    "perceptron",
                    "--NCapacity",
                    "24",
                    "24",
                    "1",
                    "--p-min",
                    "61",
                    "--datasets",
                    "1",
                    "--max-epochs",
                    "1",
                    "--seed",
                    "13",
                ],
                cwd=working_directory,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

            with (working_directory / "results/capacity_N_sweep.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual([int(row["P"]) for row in rows], [61, 63, 65, 67, 69, 71])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
