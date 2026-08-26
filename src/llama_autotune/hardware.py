"""Cross-platform hardware detection.

Detects CPU (cores, name), RAM, and GPU (vendor, model, VRAM) using
platform-native tools: WMI on Windows, nvidia-smi / rocm-smi on Linux,
system_profiler on macOS. Falls back to ``llama-bench --list-devices``
to verify that detected GPUs are actually usable by llama.cpp.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys

import psutil

from .models import Backend, GpuVendor, HardwareInfo


def detect_hardware() -> HardwareInfo:
    """Detect all hardware on the current machine.

    Gathers CPU, RAM, and GPU information, then verifies that any
    detected GPU backend is actually usable by llama.cpp.

    Returns:
        A populated :class:`HardwareInfo` instance.
    """
    info = HardwareInfo()

    _detect_cpu(info)
    _detect_ram(info)
    _detect_gpu(info)
    _determine_backend(info)
    _verify_gpu_backend(info)

    return info


def _detect_cpu(info: HardwareInfo) -> None:
    """Populate CPU name, physical core count, and logical core count."""
    info.physical_cores = psutil.cpu_count(logical=False) or 0
    info.logical_cores = psutil.cpu_count(logical=True) or 0
    info.cpu_name = _get_cpu_name()


def _get_cpu_name() -> str:
    """Retrieve the CPU model name via platform-specific commands.

    Uses PowerShell (Windows), lscpu (Linux), or sysctl (macOS).

    Returns:
        CPU model string, or fallback from ``platform.processor()``.
    """
    if sys.platform == "win32":
        try:
            output = subprocess.check_output(
                [
                    "powershell",
                    "-Command",
                    "(Get-CimInstance Win32_Processor).Name",
                ],
                text=True,
                timeout=10,
            )
            return output.strip()
        except Exception:
            return platform.processor()
    elif sys.platform == "linux":
        try:
            output = subprocess.check_output(
                ["lscpu"], text=True, timeout=10
            )
            for line in output.splitlines():
                if "Model name" in line:
                    return line.split(":")[-1].strip()
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            output = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                timeout=10,
            )
            return output.strip()
        except Exception:
            pass
    return platform.processor()


def _detect_ram(info: HardwareInfo) -> None:
    """Populate total system RAM in gigabytes."""
    mem = psutil.virtual_memory()
    info.ram_gb = round(mem.total / (1024**3), 1)


def _detect_gpu(info: HardwareInfo) -> None:
    """Detect GPU(s) by dispatching to the platform-specific detector."""
    if sys.platform == "win32":
        _detect_gpu_windows(info)
    elif sys.platform == "linux":
        _detect_gpu_linux(info)
    elif sys.platform == "darwin":
        _detect_gpu_macos(info)


def _parse_nvidia_smi_csv(output: str) -> list[tuple[str, float]]:
    """Parse ``nvidia-smi --query-gpu=name,memory.total --format=csv``.

    Returns:
        A list of ``(name, vram_gb)`` tuples.
    """
    entries: list[tuple[str, float]] = []
    for line in output.strip().splitlines():
        parts = line.split(",")
        if len(parts) < 2:
            continue
        name = parts[0].strip()
        raw = parts[-1].strip().lower()
        is_gib = "gib" in raw
        mem = raw.replace("mib", "").replace("gib", "").strip()
        try:
            value = float(mem)
        except ValueError:
            continue
        vram_gb = value if is_gib else value / 1024.0
        entries.append((name, round(vram_gb, 1)))
    return entries


def _parse_rocm_vram(output: str) -> list[float]:
    """Parse VRAM values (in GB) from ``rocm-smi --showmeminfo vram``."""
    values: list[float] = []
    for line in output.splitlines():
        m = re.search(r"VRAM Total Memory \(B\):\s*([\d.]+)", line)
        if m:
            values.append(round(float(m.group(1)) / (1024**3), 1))
    return values


def _detect_gpu_windows(info: HardwareInfo) -> None:
    """Enumerate GPUs via WMI (Win32_VideoController).

    WMI's ``AdapterRAM`` is a uint32 that overflows on GPUs with more than
    4 GiB of VRAM, so a nvidia-smi pass is used to repair the values when
    they look implausible.
    """
    try:
        output = subprocess.check_output(
            [
                "powershell",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name, AdapterRAM | ConvertTo-Json",
            ],
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        import json

        data = json.loads(output)
        if not isinstance(data, list):
            data = [data]

        for gpu in data:
            name = gpu.get("Name", "") or ""
            ram_bytes = gpu.get("AdapterRAM") or 0
            vram_gb = round(ram_bytes / (1024**3), 1) if ram_bytes else 0.0

            vendor = _classify_gpu(name)
            if vendor != GpuVendor.UNKNOWN:
                info.gpu_count += 1
                info.gpu_models.append(name)
                info.vram_per_gpu.append(vram_gb)
                info.gpu_vendor = vendor

        _repair_windows_vram_nvidia(info)
    except Exception:
        pass


def _repair_windows_vram_nvidia(info: HardwareInfo) -> None:
    """Replace implausible WMI VRAM values with nvidia-smi data.

    WMI ``AdapterRAM`` is unreliable above 4 GiB; when an NVIDIA GPU shows
    a suspiciously small VRAM value, nvidia-smi (available on Windows) is
    queried for an accurate figure.
    """
    if info.gpu_vendor != GpuVendor.NVIDIA:
        return
    if info.vram_per_gpu and all(vram > 1.0 for vram in info.vram_per_gpu):
        return

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        parsed = _parse_nvidia_smi_csv(output)
        if parsed:
            info.gpu_models = [name for name, _ in parsed]
            info.vram_per_gpu = [vram for _, vram in parsed]
            info.gpu_count = len(parsed)
            info.gpu_vendor = GpuVendor.NVIDIA
    except Exception:
        pass


def _detect_gpu_linux(info: HardwareInfo) -> None:
    """Enumerate GPUs via nvidia-smi, rocm-smi, and vulkaninfo on Linux."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        parsed = _parse_nvidia_smi_csv(output)
        if parsed:
            info.gpu_count = len(parsed)
            info.gpu_models = [name for name, _ in parsed]
            info.vram_per_gpu = [vram for _, vram in parsed]
            info.gpu_vendor = GpuVendor.NVIDIA
    except Exception:
        pass
    if info.gpu_count == 0:
        _detect_amd_linux(info)
    if info.gpu_count == 0:
        _detect_vulkan_linux(info)


def _detect_amd_linux(info: HardwareInfo) -> None:
    """Detect AMD GPUs and their VRAM via rocm-smi."""
    try:
        output = subprocess.check_output(
            ["rocm-smi", "--showproductinfo"], text=True, timeout=10
        )
        for line in output.splitlines():
            m = re.search(r"Name:\s+(.+)", line)
            if m:
                info.gpu_count += 1
                info.gpu_models.append(m.group(1).strip())
                info.gpu_vendor = GpuVendor.AMD
        if info.gpu_count > 0:
            info.vram_per_gpu = _detect_amd_vram_linux(info.gpu_count)
    except Exception:
        pass


def _detect_amd_vram_linux(expected: int) -> list[float]:
    """Return VRAM values for ``expected`` AMD GPUs via rocm-smi."""
    try:
        output = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram"], text=True, timeout=10
        )
        values = _parse_rocm_vram(output)
        while len(values) < expected:
            values.append(0.0)
        return values[:expected]
    except Exception:
        return [0.0] * expected


def _detect_vulkan_linux(info: HardwareInfo) -> None:
    """Fall back to vulkaninfo for GPU names when no vendor tool exists."""
    try:
        output = subprocess.check_output(
            ["vulkaninfo", "--summary"], text=True, timeout=15
        )
        for line in output.splitlines():
            if "GPU" in line and ":" in line:
                info.gpu_count += 1
                info.gpu_models.append(line.split(":")[-1].strip())
                info.gpu_vendor = GpuVendor.UNKNOWN
    except Exception:
        pass


def _parse_macos_displays(
    output: str,
    ram_gb: float,
) -> tuple[list[str], list[float], GpuVendor]:
    """Parse ``system_profiler SPDisplaysDataType`` output.

    Extracts GPU names and VRAM. Apple Silicon reports no per-GPU VRAM
    (unified memory), so a conservative fraction of system RAM is used.
    """
    names: list[str] = []
    vrams: list[float] = []
    vendor = GpuVendor.UNKNOWN

    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Chipset Model"):
            name = stripped.split(":", 1)[-1].strip()
            names.append(name)
            vrams.append(0.0)
            vendor = _classify_gpu(name)
            continue
        m = re.search(
            r"VRAM \(Total\):\s*([\d.]+)\s*(MB|GB)",
            stripped,
            re.IGNORECASE,
        )
        if m and names:
            value = float(m.group(1))
            if m.group(2).upper() == "MB":
                value /= 1024.0
            vrams[-1] = round(value, 1)

    if vendor == GpuVendor.APPLE and names:
        vrams = [round(ram_gb * 0.75, 1)] * len(names)

    return names, vrams, vendor


def _detect_gpu_macos(info: HardwareInfo) -> None:
    """Enumerate GPUs via system_profiler on macOS."""
    try:
        output = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True,
            timeout=15,
        )
        names, vrams, vendor = _parse_macos_displays(output, info.ram_gb)
        info.gpu_models = names
        info.vram_per_gpu = vrams
        info.gpu_count = len(names)
        info.gpu_vendor = vendor
    except Exception:
        info.gpu_vendor = GpuVendor.APPLE


def _classify_gpu(name: str) -> GpuVendor:
    """Classify a GPU vendor by keyword matching on its name string.

    Args:
        name: GPU model name from the OS.

    Returns:
        The matching :class:`GpuVendor`, or UNKNOWN.
    """
    name_lower = name.lower()
    if any(x in name_lower for x in ["nvidia", "geforce", "rtx", "gtx", "tesla", "quadro"]):
        return GpuVendor.NVIDIA
    if any(x in name_lower for x in ["amd", "radeon", "rx ", "w7600", "w7800", "instinct"]):
        return GpuVendor.AMD
    if any(x in name_lower for x in ["intel", "arc", "iris", "uhd", "xe"]):
        return GpuVendor.INTEL
    if any(x in name_lower for x in ["apple", "m1", "m2", "m3", "m4", "m5"]):
        return GpuVendor.APPLE
    return GpuVendor.UNKNOWN


def _determine_backend(info: HardwareInfo) -> None:
    """Map GPU vendor to a llama.cpp backend string."""
    if info.gpu_vendor == GpuVendor.NVIDIA:
        info.backend = Backend.CUDA
    elif info.gpu_vendor == GpuVendor.AMD:
        info.backend = Backend.ROCM
    elif info.gpu_vendor == GpuVendor.INTEL:
        info.backend = Backend.VULKAN
    elif info.gpu_vendor == GpuVendor.APPLE:
        info.backend = Backend.METAL
    else:
        info.backend = Backend.CPU


def _list_devices_has_gpu(output: str) -> bool | None:
    """Inspect ``llama-bench --list-devices`` output for GPU devices.

    Returns:
        ``True`` when a non-CPU device is listed, ``False`` when the
        device list contains only CPU entries, and ``None`` when the
        output format is not recognised (the caller should then leave the
        backend unchanged).
    """
    in_devices = False
    gpu_seen = False

    for raw in output.splitlines():
        line = raw.strip()
        lowered = line.lower()
        if lowered.startswith("available devices"):
            in_devices = True
            continue
        if in_devices and line:
            if lowered == "cpu" or lowered.startswith("cpu"):
                continue
            gpu_seen = True

    if in_devices:
        return gpu_seen
    return None


def _verify_gpu_backend(info: HardwareInfo) -> None:
    """Cross-check GPU backend against ``llama-bench --list-devices``.

    If llama-bench reports no usable GPU devices, forces the backend
    to CPU to avoid generating invalid GPU configurations.
    """
    if info.backend == Backend.CPU:
        return
    try:
        from .benchmark import find_llama_bench

        bench_path = find_llama_bench()
        result = subprocess.run(
            [bench_path, "--list-devices"],
            capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if _list_devices_has_gpu(result.stdout) is False:
            info.backend = Backend.CPU
            info.gpu_count = 0
            info.gpu_models = []
            info.vram_per_gpu = []
    except Exception:
        pass
