"""Build-time CUDA ABI checks and the isolated Jetson CUDA 12.9 launch gate."""

import ctypes as ct
from importlib import import_module
from importlib.metadata import distributions
import os
from pathlib import Path
import re
import sys


def cuda_abi_packages():
    names = {"torch", "torchaudio", "torchvision", "torchcodec", "triton", "vllm", "xformers", "flash-attn", "flashinfer"}
    prefixes = ("cuda-", "flash-attn-", "flashinfer-", "nvidia-", "pytorch-triton", "torch-stable-abi", "triton-")
    packages = {}
    for package in distributions():
        name = re.sub(r"[-_.]+", "-", package.name).lower()
        if name in names or name.startswith(prefixes):
            packages[name] = package.version
    required = {"torch", "torchaudio", "torchvision", "vllm", "triton"}
    if missing := required - packages.keys():
        raise RuntimeError(f"CUDA 12.9 base is missing packages: {', '.join(sorted(missing))}")
    return packages


def verify_build(before_path):
    import torch

    before = dict(line.split("==", 1) for line in Path(before_path).read_text().splitlines())
    after = cuda_abi_packages()
    changes = [f"{name}: {version} -> {after.get(name)}" for name, version in before.items() if after.get(name) != version]
    if changes:
        raise RuntimeError(f"CUDA base packages changed: {'; '.join(changes)}")
    if torch.version.cuda != "12.9":
        raise RuntimeError(f"Expected PyTorch CUDA 12.9, got {torch.version.cuda!r}")


def verify_runtime():
    compat = Path("/opt/tuber-cuda-compat").resolve(strict=True)
    libcuda = (compat / "libcuda.so.1").resolve(strict=True)
    if not libcuda.is_relative_to(compat):
        raise RuntimeError("Tegra libcuda must resolve inside the read-only compatibility mount")
    driver = ct.CDLL("libcuda.so.1")
    maps = Path("/proc/self/maps").read_text()
    if str(libcuda) not in maps:
        raise RuntimeError("The process did not load libcuda from the Tegra compatibility mount")
    init = driver.cuInit(0)
    if init:
        raise RuntimeError(f"cuInit failed with {init}")
    version = ct.c_int()
    result = driver.cuDriverGetVersion(ct.byref(version))
    if result or version.value != 12090:
        raise RuntimeError(f"Expected Tegra CUDA 12.9 driver API, got {version.value} (status {result})")

    import torch

    import_module("vllm._C")
    import_module("vllm_omni")

    if torch.version.cuda != "12.9" or torch.cuda.get_device_capability() != (8, 7):
        raise RuntimeError("Expected PyTorch CUDA 12.9 and an SM 8.7 Jetson")
    tensor = torch.ones((4, 4), device="cuda", dtype=torch.bfloat16)
    if (tensor @ tensor)[0, 0].item() != 4:
        raise RuntimeError("Jetson CUDA 12.9 BF16 compute probe failed")
    del tensor
    print(f"Tegra CUDA API {version.value}, PyTorch CUDA {torch.version.cuda}, SM 8.7 BF16 preflight passed", flush=True)


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["pins"]:
            for name, version in sorted(cuda_abi_packages().items()):
                print(f"{name}=={version}")
        case ["verify", before_path]:
            verify_build(before_path)
        case ["probe"]:
            verify_runtime()
        case ["serve"]:
            verify_runtime()
            model = Path(__file__).with_name("model.py")
            os.execv(sys.executable, [sys.executable, str(model), "serve"])
        case _:
            raise SystemExit("usage: cu129_runtime.py pins|verify FILE|probe|serve")
