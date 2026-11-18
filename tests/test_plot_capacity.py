import tempfile
import unittest
from pathlib import Path

from scripts.plot_capacity import (
    normal_approximation,
    plot_capacity,
    read_capacity_csv,
    theoretical_probability,
    wilson_interval,
)


class PlotCapacityTests(unittest.TestCase):
    def test_theoretical_probability_matches_simple_cases(self):
        self.assertAlmostEqual(theoretical_probability(1, 2), 0.5)
        self.assertAlmostEqual(theoretical_probability(2, 3), 0.75)
        self.assertAlmostEqual(theoretical_probability(2, 4), 0.5)

    def test_normal_approximation_handles_critical_and_single_pattern_cases(self):
        self.assertEqual(normal_approximation(5, 1), 1.0)
        self.assertAlmostEqual(normal_approximation(2, 4), 0.5)

    def test_wilson_interval_does_not_collapse_at_extremes(self):
        low, high = wilson_interval(20, 20)
        self.assertGreater(low, 0.0)
        self.assertEqual(high, 1.0)

        low, high = wilson_interval(0, 20)
        self.assertEqual(low, 0.0)
        self.assertLess(high, 1.0)

        low, high = wilson_interval(10, 20)
        self.assertAlmostEqual(low + high, 1.0)

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
        self.assertEqual(rows[0]["P_ls_exact"], 1.0)
        self.assertAlmostEqual(rows[1]["P_ls_exact"], 0.75)
        self.assertAlmostEqual(rows[1]["Q_ls_variance_est"], 0.75 * 0.25 / 20)
        self.assertGreater(rows[0]["Q_ls_wilson95_low"], 0.0)
        self.assertEqual(rows[0]["Q_ls_wilson95_high"], 1.0)

    def test_plot_capacity_handles_extreme_rates(self):
        content = (
            "N,P,nD,successes,Q_ls\n"
            "2,2,513,513,1.0\n"
            "2,3,513,0,0.0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "capacity.csv"
            output = Path(directory) / "capacity.png"
            source.write_text(content, encoding="utf-8")
            plot_capacity([source], output)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
