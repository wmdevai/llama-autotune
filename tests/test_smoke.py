"""End-to-end smoke tests for the optimization pipeline.

These tests exercise real llama-bench runs and the full optimizer. They skip
gracefully when no small GGUF model is available on the machine.
"""

import os
from pathlib import Path

import pytest

from llama_autotune.benchmark import run_benchmark
from llama_autotune.models import SearchConfig
from llama_autotune.optimizer import Optimizer

# Vocab-only files are a few MB; real small models are usually > 50 MB.
_MIN_MODEL_BYTES = 50_000_000
_MAX_MODEL_BYTES = 3_000_000_000


def _find_small_model() -> str | None:
    """Return a path to a small GGUF model, or None if unavailable."""
    env = os.environ.get("LLAMA_AUTOTUNE_SMOKE_MODEL")
    if env and os.path.isfile(env):
        return env

    for base in (
        Path.home() / "Modelli" / "llama.cpp",
        Path.home() / ".local" / "src" / "llama.cpp" / "models",
    ):
        if not base.is_dir():
            continue
        candidates = [
            path
            for path in base.rglob("*.gguf")
            if path.is_file()
            and _MIN_MODEL_BYTES < path.stat().st_size < _MAX_MODEL_BYTES
        ]
        if candidates:
            return str(min(candidates, key=lambda p: p.stat().st_size))

    return None


def test_smoke_run_benchmark():
    """A single llama-bench run must return a structured result."""
    model = _find_small_model()
    if model is None:
        pytest.skip("no small GGUF model available for smoke test")

    result = run_benchmark(
        model,
        SearchConfig(threads=2, n_gpu_layers=0, flash_attn=False),
        repetitions=1,
        timeout=180,
    )

    assert hasattr(result, "success")
    assert isinstance(result.success, bool)


def test_smoke_optimizer_single_trial():
    """A minimal optimization run must complete and return a config."""
    model = _find_small_model()
    if model is None:
        pytest.skip("no small GGUF model available for smoke test")

    opt = Optimizer(
        model_path=model,
        n_trials_stage_b=1,
        n_trials_stage_c=1,
    )

    best = opt.run()

    assert best is not None
