"""Install a pinned, checksum-verified Windows CUDA SDK without system changes.

Source: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#parsing-redistrib-json
"""
import argparse
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

BASE_URL = "https://developer.download.nvidia.com/compute/cuda/redist/"
VERSION = "13.4.1"
COMPONENTS = ("cuda_nvcc", "cuda_cudart", "cuda_crt", "cccl", "libnvvm")


def install_component(output, name, record):
    relative = record["relative_path"]
    if not re.fullmatch(rf"{re.escape(name)}/windows-x86_64/[A-Za-z0-9_.-]+\.zip", relative):
        raise ValueError("Invalid CUDA component URL")
    digest = record["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid CUDA component SHA-256")
    size = int(record["size"])
    if not 0 < size <= 512 * 1024 * 1024:
        raise ValueError("Invalid CUDA component size")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / (name + ".zip")
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        with urllib.request.urlopen(BASE_URL + relative, timeout=120) as source, archive.open("wb") as target:
            copied = 0
            while chunk := source.read(min(1024 * 1024, size - copied + 1)):
                copied += len(chunk)
                if copied > size:
                    raise ValueError(f"CUDA download exceeds declared size: {name}")
                target.write(chunk)
    if archive.stat().st_size != size or hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise ValueError(f"CUDA component checksum/size mismatch: {name}")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError(f"Corrupt CUDA component archive: {name}")
        for item in bundle.infolist():
            parts = PurePosixPath(item.filename).parts
            if item.filename.startswith("/") or "\\" in item.filename or ".." in parts or any(":" in part for part in parts):
                raise ValueError("Unsafe CUDA archive path")
        for item in bundle.infolist():
            parts = PurePosixPath(item.filename).parts
            if len(parts) < 2 or item.is_dir():
                continue
            relative_path = Path(*parts[1:])
            if relative_path.name.upper().startswith(("LICENSE", "EULA")):
                relative_path = Path("licenses") / name / relative_path
            target = output / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    print(f"{name}: SHA-256 verified and extracted")


def install_cuda(output):
    with urllib.request.urlopen(BASE_URL + f"redistrib_{VERSION}.json", timeout=60) as source:
        manifest = json.load(source)
    for name in COMPONENTS:
        install_component(output, name, manifest[name]["windows-x86_64"])
    output = Path(output)
    if not (output / "bin/nvcc.exe").is_file():
        raise ValueError("CUDA SDK is missing nvcc.exe")
    (output / "redistrib-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"CUDA {VERSION} SDK ready: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Assemble an isolated NVIDIA CUDA {VERSION} Windows SDK")
    parser.add_argument("--output", required=True, type=Path, help="SDK output directory (no global installation)")
    install_cuda(parser.parse_args().output)
