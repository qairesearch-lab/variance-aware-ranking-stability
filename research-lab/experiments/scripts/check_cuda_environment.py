#!/usr/bin/env python3
"""Diagnose whether the workstation Python environment can use CUDA."""

from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys


def module_file(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    return None if spec is None else spec.origin


def run_nvidia_smi() -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            check=False,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except FileNotFoundError:
        return 127, "nvidia-smi was not found on PATH."
    except subprocess.TimeoutExpired:
        return 124, "nvidia-smi timed out."
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def main() -> None:
    print(f"python: {sys.version}")
    print(f"platform: {platform.platform()}")
    print(f"torch module: {module_file('torch')}")
    print(f"torchvision module: {module_file('torchvision')}")

    try:
        import torch
    except Exception as exc:
        print(f"ERROR: failed to import torch: {exc}")
        raise SystemExit(1)

    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"cuda:{index}: {torch.cuda.get_device_name(index)}")
        test = torch.randn(256, 256, device="cuda")
        result = test @ test
        torch.cuda.synchronize()
        print(f"cuda tensor smoke: ok shape={tuple(result.shape)}")
    else:
        print("CUDA is not available to PyTorch.")
        if torch.version.cuda is None:
            print("Diagnosis: this is almost certainly a CPU-only PyTorch build.")
        else:
            print("Diagnosis: PyTorch has CUDA runtime support, but the driver/GPU is not accessible.")

    code, output = run_nvidia_smi()
    print(f"nvidia-smi exit_code: {code}")
    print(output)
    if not torch.cuda.is_available():
        raise SystemExit(1)
    print("CUDA_ENVIRONMENT_OK")


if __name__ == "__main__":
    main()
