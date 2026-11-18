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
                sequential_csv = {p.name: p.read_bytes() for p in Path(sequential_dir).glob("results/*.csv")}
                parallel_csv = {p.name: p.read_bytes() for p in Path(parallel_dir).glob("results/*.csv")}
                self.assertEqual(sequential_csv, parallel_csv)

    def test_cuda_capacity_batches_all_p_and_n_trials_together(self):
        if not CUDA_ENABLED:
            self.skipTest("Native GPU contract")
        scenarios = {
            "capacity": (9, ["--debug", "--capacity", "--dimensions", "4",
                              "--datasets", "3", "--p-min", "2", "--p-max", "6",
                              "--p-step", "2", "--seed", "17"]),
            "sweep": (12, ["--debug", "--NCapacity", "2", "4", "2",
                           "--datasets", "2", "--p-min", "1", "--p-max", "3",
                           "--p-step", "1", "--seed", "17"]),
        }
        for name, (tasks, arguments) in scenarios.items():
            with self.subTest(mode=name), tempfile.TemporaryDirectory() as cpu_dir, tempfile.TemporaryDirectory() as gpu_dir:
                cpu = self.run_cli(cpu_dir, arguments)
                gpu = self.run_cli(gpu_dir, [*arguments, "--cuda"])
                self.assertEqual(cpu.returncode, 0, cpu.stderr)
                self.assertEqual(gpu.returncode, 0, gpu.stderr)
                self.assertEqual(cpu.stdout, gpu.stdout)
                self.assertEqual(gpu.stderr.count("cuda_batch datasets="), 1, gpu.stderr)
                self.assertIn(f"cuda_batch datasets={tasks};", gpu.stderr)
                cpu_csv = {p.name: p.read_bytes() for p in Path(cpu_dir).glob("results/*.csv")}
                gpu_csv = {p.name: p.read_bytes() for p in Path(gpu_dir).glob("results/*.csv")}
                self.assertEqual(cpu_csv, gpu_csv)

    def test_cuda_capacity_chunks_mixed_p_at_the_batch_limit(self):
        if not CUDA_ENABLED:
            self.skipTest("Native GPU contract")
        arguments = ["--capacity", "--dimensions", "2", "--datasets", "513",
                     "--max-epochs", "2", "--p-min", "1", "--p-max", "2",
                     "--p-step", "1", "--seed", "23"]
        with tempfile.TemporaryDirectory() as cpu_dir, tempfile.TemporaryDirectory() as gpu_dir:
            cpu = self.run_cli(cpu_dir, arguments)
            gpu = self.run_cli(gpu_dir, ["--debug", *arguments, "--cuda"])
            self.assertEqual(cpu.returncode, 0, cpu.stderr)
            self.assertEqual(gpu.returncode, 0, gpu.stderr)
            self.assertEqual(cpu.stdout, gpu.stdout)
            self.assertEqual(gpu.stderr.count("cuda_batch datasets="), 2, gpu.stderr)
            self.assertIn("cuda_batch datasets=1024;", gpu.stderr)
            self.assertIn("cuda_batch datasets=2;", gpu.stderr)
            cpu_csv = {p.name: p.read_bytes() for p in Path(cpu_dir).glob("results/*.csv")}
            gpu_csv = {p.name: p.read_bytes() for p in Path(gpu_dir).glob("results/*.csv")}
            self.assertEqual(cpu_csv, gpu_csv)

    def test_help_documents_cuda_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_cli(directory, ["--help"])
            self.assertEqual(result.returncode, 0)
            self.assertIn("--cuda            opt in to CUDA (default: false)", result.stdout)
            self.assertIn("--parallelize", result.stdout)

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
