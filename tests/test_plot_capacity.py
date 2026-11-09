import tempfile
import unittest
from pathlib import Path

from scripts.plot_capacity import read_capacity_csv, theoretical_probability


class PlotCapacityTests(unittest.TestCase):
    def test_theoretical_probability_matches_simple_cases(self):
        self.assertAlmostEqual(theoretical_probability(1, 2), 0.5)
        self.assertAlmostEqual(theoretical_probability(2, 3), 0.75)
        self.assertAlmostEqual(theoretical_probability(2, 4), 0.5)

    def test_read_capacity_csv_preserves_empirical_points(self):
        content = (
            "N,P,nD,successes,Q_ls\n"
            "2,2,20,20,1.0\n"
            "2,3,20,15,0.75\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capacity.csv"
            path.write_text(content, encoding="utf-8")
            rows = read_capacity_csv(path)

        self.assertEqual([row["P"] for row in rows], [2, 3])
        self.assertEqual([row["Q_ls"] for row in rows], [1.0, 0.75])


if __name__ == "__main__":
    unittest.main()
