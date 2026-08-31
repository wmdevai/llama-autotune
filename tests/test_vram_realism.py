"""Real-hardware VRAM realism tests.

These run real ``llama-bench`` workloads and read real GPU VRAM through
``nvidia-smi`` / ``rocm-smi``. They skip automatically when the
prerequisites (a vendor VRAM tool and a real GGUF model) are absent, so
they stay green in CI and on machines without a GPU.
"""

import os
import shutil

import pytest

from llama_autotune.benchmark import run_benchmark
from llama_autotune.constraints import estimate_vram, is_plausible
from llama_autotune.hardware import detect_hardware
from llama_autotune.heuristics import generate_initial_config
from llama_autotune.model_inspector import inspect_model
from llama_autotune.models import SearchConfig

MODEL_PATH = os.environ.get(
    "LLAMA_AUTOTUNE_TEST_MODEL",
    os.path.expanduser(
        "/home/walter/Modelli/llama.cpp/Qwen3-14B-Q5_K_M.gguf"
    ),
)


def _require_real_bench() -> None:
    if not shutil.which("nvidia-smi") and not shutil.which("rocm-smi"):
        pytest.skip("nvidia-smi/rocm-smi not available")
    if not os.path.isfile(MODEL_PATH):
        pytest.skip("real GGUF model not present")


def test_real_benchmark_reports_gpu_vram():
    """A real benchmark must populate BenchmarkResult.vram_usage."""
    _require_real_bench()

    result = run_benchmark(
        MODEL_PATH,
        SearchConfig(
            threads=12,
            n_gpu_layers=999,
            ctx_size=4096,
            flash_attn=True,
        ),
        repetitions=1,
        timeout=300,
        n_prompt=16,
        n_gen=8,
    )

    assert result.success is True
    assert result.vram_usage > 0.0


def test_initial_config_is_plausible():
    """The heuristic baseline must pass the constraint engine.

    Regression guard: if ``is_plausible`` rejects the config the tool itself
    generates as a starting point, the search can never explore improvements
    around it (this happened for full offload + large context on a 16 GB
    GPU because the VRAM estimate was too conservative).
    """
    _require_real_bench()

    hw = detect_hardware()
    model = inspect_model(MODEL_PATH)
    cfg = generate_initial_config(hw, model)

    assert is_plausible(cfg, model, hw) is True


def test_estimate_never_below_measured_vram():
    """The (full-context) VRAM estimate must not under-report the measured
    benchmark VRAM."""
    _require_real_bench()

    hw = detect_hardware()
    model = inspect_model(MODEL_PATH)
    cfg = SearchConfig(
        threads=12,
        n_gpu_layers=999,
        ctx_size=4096,
        flash_attn=True,
    )

    result = run_benchmark(
        MODEL_PATH,
        cfg,
        repetitions=1,
        timeout=300,
        n_prompt=16,
        n_gen=8,
    )

    assert result.success is True

    measured_gb = result.vram_usage / 1024.0
    assert estimate_vram(cfg, model, hw) >= measured_gb
