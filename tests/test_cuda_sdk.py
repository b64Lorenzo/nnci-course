"""Tests for the isolated Windows CUDA SDK bootstrap used by releases."""
import hashlib
import io
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import install_cuda as sdk

ROOT = Path(__file__).resolve().parents[1]


class CudaSdkTests(unittest.TestCase):
    @staticmethod
    def archive(extra=None):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as bundle:
            bundle.writestr("fixture/bin/nvcc.exe", b"unit-test fixture, not an executable")
            bundle.writestr("fixture/LICENSE", b"unit-test license fixture")
            if extra:
                bundle.writestr(extra, b"must not be extracted")
        payload = stream.getvalue()
        return payload, {
            "relative_path": "cuda_nvcc/windows-x86_64/component.zip",
            "sha256": hashlib.sha256(payload).hexdigest(), "size": str(len(payload)),
        }

    def test_installer_help_requires_an_explicit_output_directory(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/install_cuda.py"), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--output", result.stdout)
        self.assertIn("13.4.1", result.stdout)

    def test_verified_component_is_flattened_licensed_and_cached(self):
        payload, record = self.archive()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "sdk"
            with patch.object(sdk.urllib.request, "urlopen", return_value=io.BytesIO(payload)) as request:
                sdk.install_component(output, "cuda_nvcc", record)
                sdk.install_component(output, "cuda_nvcc", record)
                request.assert_called_once()
            self.assertEqual((output / "bin/nvcc.exe").read_bytes(), b"unit-test fixture, not an executable")
            self.assertEqual((output / "licenses/cuda_nvcc/LICENSE").read_bytes(), b"unit-test license fixture")

    def test_bad_component_metadata_is_rejected_before_network_or_output(self):
        _, record = self.archive()
        for key, value in (("relative_path", "https://other.invalid/component.zip"), ("sha256", "bad"), ("size", "-1")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "not-created"
                with patch.object(sdk.urllib.request, "urlopen") as request:
                    with self.assertRaises(ValueError):
                        sdk.install_component(output, "cuda_nvcc", {**record, key: value})
                    request.assert_not_called()
                self.assertFalse(output.exists())

    def test_checksum_failure_never_extracts_a_component(self):
        payload, record = self.archive()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(sdk.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    sdk.install_component(output, "cuda_nvcc", {**record, "sha256": "0" * 64})
            self.assertFalse((output / "bin").exists())

    def test_archive_traversal_is_rejected_before_extracting_any_member(self):
        for member in ("fixture/../escaped", "fixture/..\\escaped", "/escaped", "fixture/file:stream"):
            payload, record = self.archive(member)
            with self.subTest(member=member), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "sdk"
                with patch.object(sdk.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
                    with self.assertRaisesRegex(ValueError, "Unsafe"):
                        sdk.install_component(output, "cuda_nvcc", record)
                self.assertFalse((output / "bin").exists())
                self.assertFalse((Path(temporary) / "escaped").exists())
