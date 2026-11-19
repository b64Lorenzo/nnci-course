"""Exercise real CPU/CUDA CLIs in isolated output directories."""
import csv
import io
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

EXECUTABLE = Path(sys.argv.pop(1)).resolve()
CUDA_ENABLED = sys.argv.pop(1) == "on"
SCENARIOS = {
    "single": ["--dimensions", "4", "--patterns", "6", "--datasets", "3", "--seed", "17"],
    "validation": ["--validation"],
    "capacity": ["--capacity", "--dimensions", "4", "--datasets", "3", "--p-min", "2", "--p-max", "10", "--p-step", "2", "--seed", "17"],
    "sweep": ["--NCapacity", "2", "4", "2", "--datasets", "3", "--seed", "17"],
    "sweep_explicit": ["--NCapacity", "2", "4", "2", "--datasets", "3", "--max-epochs", "5", "--seed", "17"],
    "strength": ["--strength", "--dimensions", "4", "--patterns", "3", "--datasets", "9", "--seed", "17"],
}


def csv_snapshot(directory):
    return {p.name: p.read_bytes() for p in Path(directory).glob("results/*.csv")}


def csv_rows(payload):
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))


class CudaCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CUDA_ENABLED:
            return
        with tempfile.TemporaryDirectory() as directory:
            probe = subprocess.run([str(EXECUTABLE), "--debug", "perceptron", "--validation", "--cuda"],
                                   cwd=directory, text=True, capture_output=True, timeout=90)
        if probe.returncode != 0:
            raise RuntimeError(probe.stderr)
        match = re.search(r"cuda_limits bytes=(\d+); dataset_limit=(\d+); free_bytes=(\d+); reserve_bytes=(\d+)", probe.stderr)
        if match is None:
            raise AssertionError("CUDA must report live VRAM and occupancy limits")
        cls.byte_limit, cls.dataset_limit, cls.free_bytes, cls.reserve_bytes = map(int, match.groups())

    def run_cli(self, directory, arguments, environment=None):
        return subprocess.run([str(EXECUTABLE), "perceptron", *arguments], cwd=directory,
                              env=environment, text=True, capture_output=True, timeout=90)

    def cuda_repeat(self, arguments):
        if not CUDA_ENABLED:
            self.skipTest("Native GPU contract")
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = self.run_cli(first_dir, [*arguments, "--cuda"])
            second = self.run_cli(second_dir, [*arguments, "--cuda"])
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first.stdout, second.stdout)
            self.assertEqual(first.stderr.count("backend=CUDA; device="), 1)
            self.assertIn("algorithm=synchronous-batch", first.stderr)
            snapshot = csv_snapshot(first_dir)
            self.assertEqual(snapshot, csv_snapshot(second_dir))
            for count, used, budget, cap in re.findall(
                    r"cuda_batch datasets=(\d+); launches=\d+; bytes=(\d+); byte_limit=(\d+); dataset_limit=(\d+)", first.stderr):
                self.assertLessEqual(int(count), int(cap))
                self.assertLessEqual(int(used), int(budget))
            return first, snapshot

    def test_all_modes_reject_cuda_in_cpu_only_build(self):
        if CUDA_ENABLED:
            self.skipTest("CPU-only build contract")
        for name, arguments in SCENARIOS.items():
            with self.subTest(mode=name), tempfile.TemporaryDirectory() as directory:
                result = self.run_cli(directory, [*arguments, "--cuda"])
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("CUDA support is not enabled", result.stderr)
                self.assertEqual(list(Path(directory).glob("results/*")), [])

    def test_native_all_modes_repeat_batch_results_and_epoch_budgets(self):
        counts = {"single": 0, "validation": 0, "capacity": 1, "sweep": 3, "sweep_explicit": 3, "strength": 1}
        for name, arguments in SCENARIOS.items():
            with self.subTest(mode=name):
                _, snapshot = self.cuda_repeat(arguments)
                self.assertEqual(len(snapshot), counts[name])
                if name in ("sweep", "sweep_explicit"):
                    for dimension in (2, 4):
                        rows = csv_rows(snapshot[f"capacity_N{dimension}.csv"])
                        expected = dimension * 3 if name == "sweep" else 5
                        self.assertEqual({int(row["nmax"]) for row in rows}, {expected})

    def test_parallelize_capacity_and_n_sweep_match_sequential_cpu(self):
        scenarios = {
            "capacity": ["--capacity", "--dimensions", "3", "--datasets", "8",
                         "--max-epochs", "5", "--p-min", "1", "--p-max", "3",
                         "--p-step", "1", "--seed", "17", "--debug"],
            "sweep": ["--NCapacity", "2", "4", "2", "--datasets", "3",
                      "--max-epochs", "5", "--p-min", "1", "--p-max", "4",
                      "--p-step", "1", "--seed", "17", "--debug"],
        }
        for name, arguments in scenarios.items():
            with self.subTest(mode=name), tempfile.TemporaryDirectory() as sequential_dir, tempfile.TemporaryDirectory() as parallel_dir:
                sequential = self.run_cli(sequential_dir, arguments)
                parallel = self.run_cli(parallel_dir, [*arguments, "--parallelize"])
                self.assertEqual(sequential.returncode, 0, sequential.stderr)
                self.assertEqual(parallel.returncode, 0, parallel.stderr)
                self.assertEqual(sequential.stdout, parallel.stdout)
                self.assertIn("scheduler=cpu-parallel; workers=", parallel.stderr)
                self.assertTrue(csv_snapshot(sequential_dir))
                self.assertEqual(csv_snapshot(sequential_dir), csv_snapshot(parallel_dir))

    def test_cuda_capacity_batches_all_p_and_n_trials_together(self):
        scenarios = {
            "capacity": (9, ["--debug", "--capacity", "--dimensions", "4", "--datasets", "3",
                             "--p-min", "2", "--p-max", "6", "--p-step", "2", "--seed", "17"]),
            "sweep": (12, ["--debug", "--NCapacity", "2", "4", "2", "--datasets", "2",
                           "--p-min", "1", "--p-max", "3", "--p-step", "1", "--seed", "17"]),
        }
        for name, (tasks, arguments) in scenarios.items():
            with self.subTest(mode=name):
                result, snapshot = self.cuda_repeat(arguments)
                self.assertTrue(snapshot)
                self.assertEqual(result.stderr.count("cuda_batch datasets="), 1, result.stderr)
                self.assertIn(f"cuda_batch datasets={tasks};", result.stderr)

    def test_cuda_capacity_chunks_mixed_p_at_the_batch_limit(self):
        if not CUDA_ENABLED:
            self.skipTest("Native GPU contract")
        trials = self.dataset_limit // 2 + 1
        result, _ = self.cuda_repeat([
            "--debug", "--capacity", "--dimensions", "2", "--datasets", str(trials),
            "--max-epochs", "2", "--p-min", "1", "--p-max", "2", "--p-step", "1", "--seed", "23"])
        counts = list(map(int, re.findall(r"cuda_batch datasets=(\d+);", result.stderr)))
        self.assertGreaterEqual(len(counts), 2)
        self.assertEqual(sum(counts), trials * 2)

    def test_parallel_studies_keep_tail_and_trial_order(self):
        if not CUDA_ENABLED:
            self.skipTest("Native GPU contract")
        trials = self.dataset_limit + 1
        for mode in ("--capacity", "--strength"):
            with self.subTest(mode=mode):
                result, snapshot = self.cuda_repeat([
                    "--debug", mode, "--dimensions", "2", "--patterns", "2", "--datasets", str(trials),
                    "--max-epochs", "4", "--seed", "23", "--p-min", "2", "--p-max", "2"])
                self.assertTrue(snapshot)
                counts = list(map(int, re.findall(r"cuda_batch datasets=(\d+);", result.stderr)))
                self.assertGreaterEqual(len(counts), 2)
                self.assertEqual(sum(counts), trials)

    def test_live_policy_preserves_free_vram(self):
        if not CUDA_ENABLED:
            self.skipTest("Native GPU contract")
        self.assertGreater(self.byte_limit, 0)
        self.assertGreater(self.dataset_limit, 0)
        self.assertEqual(self.byte_limit + self.reserve_bytes, self.free_bytes)
        self.assertGreaterEqual(self.reserve_bytes, 256 * 1024 * 1024)
        self.assertGreaterEqual(self.reserve_bytes, self.free_bytes // 4)

    def test_negative_threshold_never_reports_zero_potentials_as_converged(self):
        result, _ = self.cuda_repeat([
            "--dimensions", "2", "--patterns", "2", "--max-epochs", "3", "--threshold", "-0.5"])
        self.assertEqual(result.stdout.strip(), "Did not converge after 3 epochs; updates=0")
        _, snapshot = self.cuda_repeat([
            "--NCapacity", "2", "4", "2", "--datasets", "3", "--max-epochs", "3",
            "--p-min", "1", "--p-max", "3", "--p-step", "1", "--threshold", "-0.5"])
        self.assertEqual(len(snapshot), 3)
        for payload in snapshot.values():
            rows = csv_rows(payload)
            self.assertTrue(rows)
            self.assertTrue(all(int(row["successes"]) == 0 and float(row["Q_ls"]) == 0.0 for row in rows))

    def test_help_documents_cuda_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(directory, ["--help"])
            self.assertEqual(result.returncode, 0)
            self.assertIn("--cuda            opt in to CUDA (default: false)", result.stdout)
            self.assertIn("--parallelize", result.stdout)

    def test_hidden_gpu_leaves_cpu_default_usable(self):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
        with tempfile.TemporaryDirectory() as directory:
            cpu = self.run_cli(directory, SCENARIOS["single"], environment)
            self.assertEqual(cpu.returncode, 0, cpu.stderr)
            self.assertNotIn("backend=CUDA", cpu.stderr)
            if CUDA_ENABLED:
                gpu = self.run_cli(directory, [*SCENARIOS["single"], "--cuda"], environment)
                self.assertEqual(gpu.returncode, 1)
                self.assertIn("CUDA", gpu.stderr)

    def test_non_finite_threshold_has_error_11(self):
        for threshold in ("nan", "inf", "-inf"):
            with self.subTest(threshold=threshold), tempfile.TemporaryDirectory() as directory:
                arguments = ["--threshold", threshold, *(["--cuda"] if CUDA_ENABLED else [])]
                result = self.run_cli(directory, arguments)
                self.assertEqual(result.returncode, 11, result.stdout + result.stderr)
                self.assertIn("Perceptron error 11", result.stderr)


if __name__ == "__main__":
    unittest.main()
