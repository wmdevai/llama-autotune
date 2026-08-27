"""Tests for hardware.py subprocess-based detectors.

The platform-specific detectors shell out to native tools (lscpu,
nvidia-smi, rocm-smi, system_profiler, ...). These tests mock the
subprocess layer so each detector can be exercised deterministically.
"""

import json
from types import SimpleNamespace

from llama_autotune import hardware as hwmod
from llama_autotune.hardware import (
    _detect_amd_linux,
    _detect_amd_vram_linux,
    _detect_gpu_linux,
    _detect_gpu_macos,
    _detect_gpu_windows,
    _determine_backend,
    _get_cpu_name,
    _repair_windows_vram_nvidia,
    _verify_gpu_backend,
    _detect_vulkan_linux,
)
from llama_autotune.models import Backend, GpuVendor, HardwareInfo


# ── _determine_backend (missing AMD / INTEL branches) ────────────────


def test_determine_backend_amd():
    hw = HardwareInfo(gpu_vendor=GpuVendor.AMD)
    _determine_backend(hw)
    assert hw.backend == Backend.ROCM


def test_determine_backend_intel():
    hw = HardwareInfo(gpu_vendor=GpuVendor.INTEL)
    _determine_backend(hw)
    assert hw.backend == Backend.VULKAN


# ── _get_cpu_name ────────────────────────────────────────────────────


def test_get_cpu_name_linux(monkeypatch):
    monkeypatch.setattr(hwmod.sys, "platform", "linux")

    def fake_check_output(cmd, text=True, timeout=None):
        assert cmd[0] == "lscpu"
        return "Model name:            AMD Ryzen 9 7900X 12-Core Processor\n"

    monkeypatch.setattr(hwmod.subprocess, "check_output", fake_check_output)
    assert _get_cpu_name() == "AMD Ryzen 9 7900X 12-Core Processor"


def test_get_cpu_name_darwin(monkeypatch):
    monkeypatch.setattr(hwmod.sys, "platform", "darwin")

    def fake_check_output(cmd, text=True, timeout=None):
        assert cmd[:2] == ["sysctl", "-n"]
        return "Apple M1 Pro"

    monkeypatch.setattr(hwmod.subprocess, "check_output", fake_check_output)
    assert _get_cpu_name() == "Apple M1 Pro"


def test_get_cpu_name_windows(monkeypatch):
    monkeypatch.setattr(hwmod.sys, "platform", "win32")

    def fake_check_output(cmd, text=True, timeout=None):
        return "Intel(R) Core(TM) i7-12700K"

    monkeypatch.setattr(hwmod.subprocess, "check_output", fake_check_output)
    assert _get_cpu_name() == "Intel(R) Core(TM) i7-12700K"


def test_get_cpu_name_linux_falls_back(monkeypatch):
    monkeypatch.setattr(hwmod.sys, "platform", "linux")
    monkeypatch.setattr(
        hwmod.platform,
        "processor",
        lambda: "fallback-cpu",
    )

    def fake_check_output(cmd, text=True, timeout=None):
        raise OSError("lscpu missing")

    monkeypatch.setattr(hwmod.subprocess, "check_output", fake_check_output)
    assert _get_cpu_name() == "fallback-cpu"


# ── _detect_gpu_windows ──────────────────────────────────────────────


def test_detect_gpu_windows_nvidia(monkeypatch):
    payload = json.dumps(
        [
            {
                "Name": "NVIDIA GeForce RTX 4090",
                "AdapterRAM": 8 * 1024**3,
            }
        ]
    )

    monkeypatch.setattr(
        hwmod.subprocess,
        "check_output",
        lambda cmd, text=True, timeout=None, creationflags=0: payload,
    )

    info = HardwareInfo()
    _detect_gpu_windows(info)

    assert info.gpu_count == 1
    assert info.gpu_models == ["NVIDIA GeForce RTX 4090"]
    assert info.vram_per_gpu == [8.0]
    assert info.gpu_vendor == GpuVendor.NVIDIA


def test_detect_gpu_windows_single_object(monkeypatch):
    payload = json.dumps(
        {"Name": "AMD Radeon RX 7900 XTX", "AdapterRAM": 24 * 1024**3}
    )

    monkeypatch.setattr(
        hwmod.subprocess,
        "check_output",
        lambda cmd, text=True, timeout=None, creationflags=0: payload,
    )

    info = HardwareInfo()
    _detect_gpu_windows(info)

    assert info.gpu_count == 1
    assert info.gpu_models == ["AMD Radeon RX 7900 XTX"]
    assert info.gpu_vendor == GpuVendor.AMD


# ── _repair_windows_vram_nvidia ──────────────────────────────────────


def test_repair_windows_vram_nvidia_repairs(monkeypatch):
    info = HardwareInfo(
        gpu_count=1,
        gpu_models=["NVIDIA GeForce RTX 4090"],
        vram_per_gpu=[0.0],  # overflowed WMI value
        gpu_vendor=GpuVendor.NVIDIA,
    )

    def fake_check_output(cmd, text=True, timeout=None, creationflags=0):
        assert cmd[0] == "nvidia-smi"
        return "NVIDIA GeForce RTX 4090, 24564 MiB\n"

    monkeypatch.setattr(hwmod.subprocess, "check_output", fake_check_output)

    _repair_windows_vram_nvidia(info)

    assert info.vram_per_gpu == [24.0]
    assert info.gpu_models == ["NVIDIA GeForce RTX 4090"]
    assert info.gpu_count == 1
    assert info.gpu_vendor == GpuVendor.NVIDIA


def test_repair_windows_vram_nvidia_skips_when_plausible(monkeypatch):
    info = HardwareInfo(
        gpu_count=1,
        gpu_models=["NVIDIA GeForce RTX 4090"],
        vram_per_gpu=[24.0],
        gpu_vendor=GpuVendor.NVIDIA,
    )

    def boom(*args, **kwargs):
        raise AssertionError("nvidia-smi should not be called")

    monkeypatch.setattr(hwmod.subprocess, "check_output", boom)

    _repair_windows_vram_nvidia(info)

    assert info.vram_per_gpu == [24.0]


# ── _detect_gpu_linux ────────────────────────────────────────────────


def test_detect_gpu_linux_nvidia(monkeypatch):
    def fake_check_output(cmd, text=True, timeout=None):
        if cmd[0] == "nvidia-smi":
            return "NVIDIA GeForce RTX 4090, 24564 MiB\n"
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hwmod.subprocess, "check_output", fake_check_output)

    info = HardwareInfo()
    _detect_gpu_linux(info)

    assert info.gpu_vendor == GpuVendor.NVIDIA
    assert info.gpu_count == 1
    assert info.vram_per_gpu == [24.0]


def test_detect_gpu_linux_falls_back_to_amd(monkeypatch):
    def fake_check_output(cmd, text=True, timeout=None):
        if cmd[0] == "nvidia-smi":
            raise OSError("no nvidia-smi")
        if cmd[:2] == ["rocm-smi", "--showproductinfo"]:
            return "Name: gfx1100\nName: gfx1030\n"
        if cmd[:2] == ["rocm-smi", "--showmeminfo"]:
            return (
                "GPU[0] : VRAM Total Memory (B): 17163091968\n"
                "GPU[1] : VRAM Total Memory (B): 34326183936\n"
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(hwmod.subprocess, "check_output", fake_check_output)

    info = HardwareInfo()
    _detect_gpu_linux(info)

    assert info.gpu_vendor == GpuVendor.AMD
    assert info.gpu_count == 2
    assert info.gpu_models == ["gfx1100", "gfx1030"]
    assert info.vram_per_gpu == [16.0, 32.0]


def test_detect_gpu_linux_falls_back_to_vulkan(monkeypatch):
    def fake_check_output(cmd, text=True, timeout=None):
        raise OSError(f"missing tool: {cmd[0]}")

    monkeypatch.setattr(hwmod.subprocess, "check_output", fake_check_output)

    info = HardwareInfo()
    _detect_gpu_linux(info)

    assert info.gpu_count == 0
    assert info.gpu_vendor == GpuVendor.UNKNOWN


# ── _detect_amd_linux / _detect_amd_vram_linux ───────────────────────


def test_detect_amd_vram_linux_pads_missing(monkeypatch):
    monkeypatch.setattr(
        hwmod.subprocess,
        "check_output",
        lambda cmd, text=True, timeout=None: (
            "GPU[0] : VRAM Total Memory (B): 17163091968\n"
        ),
    )

    assert _detect_amd_vram_linux(2) == [16.0, 0.0]


def test_detect_amd_vram_linux_error_returns_zeros(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no rocm-smi")

    monkeypatch.setattr(hwmod.subprocess, "check_output", boom)

    assert _detect_amd_vram_linux(2) == [0.0, 0.0]


# ── _detect_vulkan_linux ─────────────────────────────────────────────


def test_detect_vulkan_linux(monkeypatch):
    output = (
        "GPU0: AMD Radeon RX 7900 XTX\n"
        "GPU1: Intel Arc A770\n"
    )

    monkeypatch.setattr(
        hwmod.subprocess,
        "check_output",
        lambda cmd, text=True, timeout=None: output,
    )

    info = HardwareInfo()
    _detect_vulkan_linux(info)

    assert info.gpu_count == 2
    assert info.gpu_models == [
        "AMD Radeon RX 7900 XTX",
        "Intel Arc A770",
    ]
    assert info.gpu_vendor == GpuVendor.UNKNOWN


# ── _detect_gpu_macos ────────────────────────────────────────────────


def test_detect_gpu_macos(monkeypatch):
    output = "    Chipset Model: Apple M1 Pro\n"

    monkeypatch.setattr(
        hwmod.subprocess,
        "check_output",
        lambda cmd, text=True, timeout=None: output,
    )

    info = HardwareInfo(ram_gb=32.0)
    _detect_gpu_macos(info)

    assert info.gpu_count == 1
    assert info.gpu_models == ["Apple M1 Pro"]
    assert info.vram_per_gpu == [24.0]
    assert info.gpu_vendor == GpuVendor.APPLE


def test_detect_gpu_macos_error_sets_apple(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no system_profiler")

    monkeypatch.setattr(hwmod.subprocess, "check_output", boom)

    info = HardwareInfo()
    _detect_gpu_macos(info)

    assert info.gpu_vendor == GpuVendor.APPLE


# ── _verify_gpu_backend ──────────────────────────────────────────────


def test_verify_gpu_backend_forces_cpu_when_no_gpu(monkeypatch):
    info = HardwareInfo(
        backend=Backend.CUDA,
        gpu_count=1,
        gpu_models=["NVIDIA GeForce RTX 4090"],
        vram_per_gpu=[24.0],
        gpu_vendor=GpuVendor.NVIDIA,
    )

    monkeypatch.setattr(
        "llama_autotune.benchmark.find_llama_bench",
        lambda: "/usr/bin/llama-bench",
    )

    result = SimpleNamespace(stdout="Available devices:\n  CPU\n")

    monkeypatch.setattr(
        hwmod.subprocess,
        "run",
        lambda *args, **kwargs: result,
    )

    _verify_gpu_backend(info)

    assert info.backend == Backend.CPU
    assert info.gpu_count == 0
    assert info.gpu_models == []
    assert info.vram_per_gpu == []


def test_verify_gpu_backend_keeps_gpu(monkeypatch):
    info = HardwareInfo(backend=Backend.CUDA, gpu_count=1)

    monkeypatch.setattr(
        "llama_autotune.benchmark.find_llama_bench",
        lambda: "/usr/bin/llama-bench",
    )

    result = SimpleNamespace(
        stdout="Available devices:\n  CUDA0: NVIDIA GeForce RTX 4090\n"
    )

    monkeypatch.setattr(
        hwmod.subprocess,
        "run",
        lambda *args, **kwargs: result,
    )

    _verify_gpu_backend(info)

    assert info.backend == Backend.CUDA
    assert info.gpu_count == 1


def test_verify_gpu_backend_skips_on_error(monkeypatch):
    info = HardwareInfo(backend=Backend.CUDA, gpu_count=1)

    monkeypatch.setattr(
        "llama_autotune.benchmark.find_llama_bench",
        lambda: "/usr/bin/llama-bench",
    )

    def boom(*args, **kwargs):
        raise OSError("llama-bench crashed")

    monkeypatch.setattr(hwmod.subprocess, "run", boom)

    _verify_gpu_backend(info)

    assert info.backend == Backend.CUDA
    assert info.gpu_count == 1
