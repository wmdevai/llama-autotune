from llama_autotune.hardware import (
    _classify_gpu,
    _determine_backend,
    _list_devices_has_gpu,
    _parse_macos_displays,
    _parse_nvidia_smi_csv,
    _parse_rocm_vram,
    detect_hardware,
)
from llama_autotune.models import Backend, GpuVendor, HardwareInfo


def test_detect_hardware():
    hw = detect_hardware()
    assert hw.physical_cores > 0
    assert hw.logical_cores > 0
    assert hw.ram_gb > 0
    assert hw.cpu_name != ""


def test_classify_nvidia():
    assert _classify_gpu("NVIDIA GeForce RTX 4090") == GpuVendor.NVIDIA
    assert _classify_gpu("Tesla V100") == GpuVendor.NVIDIA


def test_classify_amd():
    assert _classify_gpu("AMD Radeon RX 7900 XTX") == GpuVendor.AMD


def test_classify_intel():
    assert _classify_gpu("Intel Arc A770") == GpuVendor.INTEL
    assert _classify_gpu("Intel UHD Graphics") == GpuVendor.INTEL


def test_classify_apple():
    assert _classify_gpu("Apple M1") == GpuVendor.APPLE
    assert _classify_gpu("Apple M2 Max") == GpuVendor.APPLE


def test_classify_unknown():
    assert _classify_gpu("Unknown GPU") == GpuVendor.UNKNOWN


def test_determine_backend_nvidia():
    hw = HardwareInfo(gpu_vendor=GpuVendor.NVIDIA)
    _determine_backend(hw)
    assert hw.backend == Backend.CUDA


def test_determine_backend_cpu():
    hw = HardwareInfo()
    _determine_backend(hw)
    assert hw.backend == Backend.CPU


def test_determine_backend_apple():
    hw = HardwareInfo(gpu_vendor=GpuVendor.APPLE)
    _determine_backend(hw)
    assert hw.backend == Backend.METAL


# ── nvidia-smi CSV parsing ───────────────────────────────────────────


def test_parse_nvidia_smi_csv_mib():
    out = "NVIDIA GeForce RTX 4090, 24564 MiB\n"
    assert _parse_nvidia_smi_csv(out) == [("NVIDIA GeForce RTX 4090", 24.0)]


def test_parse_nvidia_smi_csv_gib():
    out = "NVIDIA GeForce RTX 4090, 24 GiB\n"
    assert _parse_nvidia_smi_csv(out) == [("NVIDIA GeForce RTX 4090", 24.0)]


def test_parse_nvidia_smi_csv_multiple():
    out = (
        "NVIDIA A100, 40960 MiB\n"
        "NVIDIA A10, 24576 MiB\n"
    )
    parsed = _parse_nvidia_smi_csv(out)
    assert parsed == [
        ("NVIDIA A100", 40.0),
        ("NVIDIA A10", 24.0),
    ]


def test_parse_nvidia_smi_csv_empty():
    assert _parse_nvidia_smi_csv("") == []


# ── rocm-smi VRAM parsing ────────────────────────────────────────────


def test_parse_rocm_vram():
    out = "GPU[0] : VRAM Total Memory (B): 17163091968\n"
    assert _parse_rocm_vram(out) == [16.0]


def test_parse_rocm_vram_multiple():
    out = (
        "GPU[0] : VRAM Total Memory (B): 17163091968\n"
        "GPU[1] : VRAM Total Memory (B): 34326183936\n"
    )
    assert _parse_rocm_vram(out) == [16.0, 32.0]


def test_parse_rocm_vram_empty():
    assert _parse_rocm_vram("no vram info") == []


# ── macOS display parsing ────────────────────────────────────────────


def test_parse_macos_displays_apple_silicon():
    out = "    Chipset Model: Apple M1 Pro\n"
    names, vrams, vendor = _parse_macos_displays(out, 32.0)
    assert names == ["Apple M1 Pro"]
    assert vrams == [24.0]  # 32 GB RAM * 0.75
    assert vendor == GpuVendor.APPLE


def test_parse_macos_displays_discrete_gpu_gb():
    out = (
        "    Chipset Model: AMD Radeon Pro 5500M\n"
        "      Type: GPU\n"
        "      VRAM (Total): 4 GB\n"
    )
    names, vrams, vendor = _parse_macos_displays(out, 32.0)
    assert names == ["AMD Radeon Pro 5500M"]
    assert vrams == [4.0]
    assert vendor == GpuVendor.AMD


def test_parse_macos_displays_vram_mb():
    out = (
        "    Chipset Model: Intel UHD Graphics 630\n"
        "      VRAM (Total): 1536 MB\n"
    )
    _, vrams, vendor = _parse_macos_displays(out, 32.0)
    assert vrams == [1.5]
    assert vendor == GpuVendor.INTEL


# ── llama-bench --list-devices parsing ───────────────────────────────


def test_list_devices_has_gpu_true():
    out = (
        "ggml_cuda_init: found 1 CUDA devices\n"
        "Available devices:\n"
        "  CUDA0: NVIDIA GeForce RTX 5060 Ti (15880 MiB, 15371 MiB free)\n"
    )
    assert _list_devices_has_gpu(out) is True


def test_list_devices_has_gpu_with_cpu_and_gpu():
    out = (
        "Available devices:\n"
        "  CPU\n"
        "  CUDA0: NVIDIA GeForce RTX 5060 Ti\n"
    )
    assert _list_devices_has_gpu(out) is True


def test_list_devices_has_gpu_false():
    out = "Available devices:\n  CPU\n"
    assert _list_devices_has_gpu(out) is False


def test_list_devices_has_gpu_unknown():
    out = "some unrelated output without a device section\n"
    assert _list_devices_has_gpu(out) is None
