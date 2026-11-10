import csv
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
