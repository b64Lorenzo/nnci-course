"""Exercise the real CLI without writing results or logs in the source tree."""
import csv
import os
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


class CudaCliTests(unittest.TestCase):
    def run_cli(self, directory, arguments, environment=None):
        return subprocess.run([str(EXECUTABLE), "perceptron", *arguments], cwd=directory,
                              env=environment, text=True, capture_output=True, timeout=90)

    def test_all_modes_reject_cuda_in_cpu_only_build(self):
        if CUDA_ENABLED:
            self.skipTest("CPU-only build contract")
        for name, arguments in SCENARIOS.items():
            with self.subTest(mode=name), tempfile.TemporaryDirectory() as directory:
                result = self.run_cli(directory, [*arguments, "--cuda"])
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("CUDA support is not enabled", result.stderr)
                self.assertEqual(list(Path(directory).glob("results/*")), [])

    def test_native_all_modes_match_cpu_stdout_and_csv_bytes(self):
        if not CUDA_ENABLED:
            self.skipTest("Native GPU contract")
        for name, arguments in SCENARIOS.items():
            with self.subTest(mode=name), tempfile.TemporaryDirectory() as cpu_dir, tempfile.TemporaryDirectory() as gpu_dir:
                cpu = self.run_cli(cpu_dir, arguments)
                gpu = self.run_cli(gpu_dir, [*arguments, "--cuda"])
                self.assertEqual(cpu.returncode, 0, cpu.stderr)
                self.assertEqual(gpu.returncode, 0, gpu.stderr)
                self.assertEqual(cpu.stdout, gpu.stdout)
                self.assertNotIn("backend=CUDA", cpu.stderr)
                self.assertEqual(gpu.stderr.count("backend=CUDA; device="), 1)
                cpu_csv = {p.name: p.read_bytes() for p in Path(cpu_dir).glob("results/*.csv")}
                gpu_csv = {p.name: p.read_bytes() for p in Path(gpu_dir).glob("results/*.csv")}
                self.assertEqual(cpu_csv, gpu_csv)
                if name in ("sweep", "sweep_explicit"):
                    for dimension in (2, 4):
                        with (Path(gpu_dir) / f"results/capacity_N{dimension}.csv").open(newline="") as stream:
                            rows = list(csv.DictReader(stream))
                        expected = dimension * 3 if name == "sweep" else 5
                        self.assertEqual({int(row["nmax"]) for row in rows}, {expected})

    def test_help_documents_cuda_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(directory, ["--help"])
            self.assertEqual(result.returncode, 0)
            self.assertIn("--cuda            opt in to CUDA (default: false)", result.stdout)

    def test_parallel_studies_keep_tail_and_trial_order(self):
        if not CUDA_ENABLED:
            self.skipTest("Native GPU contract")
        for mode in ("--capacity", "--strength"):
            arguments = ["--debug", mode, "--dimensions", "2", "--patterns", "2",
                         "--datasets", "1025", "--max-epochs", "4", "--seed", "23",
                         "--p-min", "2", "--p-max", "2"]
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as cpu_dir, tempfile.TemporaryDirectory() as gpu_dir:
                cpu = self.run_cli(cpu_dir, arguments)
                gpu = self.run_cli(gpu_dir, [*arguments, "--cuda"])
                self.assertEqual(cpu.returncode, 0, cpu.stderr)
                self.assertEqual(gpu.returncode, 0, gpu.stderr)
                self.assertTrue("cuda_batch datasets=1024;" in gpu.stderr,
                                "The study must dispatch independent trials together")
                self.assertTrue("cuda_batch datasets=1;" in gpu.stderr,
                                "The final partial batch must not be dropped")
                cpu_csv = {p.name: p.read_bytes() for p in Path(cpu_dir).glob("results/*.csv")}
                gpu_csv = {p.name: p.read_bytes() for p in Path(gpu_dir).glob("results/*.csv")}
                self.assertEqual(cpu_csv, gpu_csv)

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
                arguments = ["--threshold", threshold]
                if CUDA_ENABLED:
                    arguments.append("--cuda")
                result = self.run_cli(directory, arguments)
                self.assertEqual(result.returncode, 11, result.stdout + result.stderr)
                self.assertIn("Perceptron error 11", result.stderr)


if __name__ == "__main__":
    unittest.main()
