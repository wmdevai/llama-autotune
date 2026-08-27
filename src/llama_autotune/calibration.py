"""Per-quantization VRAM calibration persistence.

Calibration measures how much GPU memory a model really uses relative to its
file size (full offload, minimal context, so the KV cache is negligible) and
stores the result per quantization. The VRAM estimator then prefers measured
values over the heuristic defaults.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def calibrations_path() -> str:
    """Return the absolute path to the calibrations JSON file."""
    data_dir = os.path.join(Path.home(), ".llama-autotune")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "calibrations.json")


_calibrations: dict[str, dict] | None = None


def load_calibrations() -> dict[str, dict]:
    """Load the calibrations store, caching it in memory.

    Returns:
        A mapping of quantization string -> calibration record. Empty when
        the store does not exist yet or is unreadable.
    """
    global _calibrations

    if _calibrations is None:
        try:
            with open(calibrations_path(), encoding="utf-8") as f:
                _calibrations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            _calibrations = {}

    return _calibrations


def get_overhead_factor(quantization: str) -> float | None:
    """Return the calibrated overhead factor for a quantization, if any.

    Args:
        quantization: The quantization string (e.g. ``"Q5_K_M"``).

    Returns:
        The stored overhead factor, or ``None`` when the quantization has
        not been calibrated.
    """
    quant = (quantization or "").strip()
    if not quant:
        return None

    record = load_calibrations().get(quant)
    if not record:
        return None

    try:
        return float(record["overhead"])
    except (KeyError, TypeError, ValueError):
        return None


def save_calibration(
    quantization: str,
    measured_vram_mb: float,
    file_size_mb: float,
    model_path: str,
) -> dict:
    """Store a calibration record for a quantization.

    The raw measured ratio can be slightly below 1.0 because a short
    benchmark does not touch every memory page, so the stored overhead is
    clamped to a minimum of 1.0.

    Args:
        quantization: The quantization string.
        measured_vram_mb: Peak GPU VRAM measured in megabytes.
        file_size_mb: Model file size in megabytes.
        model_path: Path of the model used for the measurement.

    Returns:
        The stored calibration record.
    """
    quant = (quantization or "").strip()

    measured_ratio = (
        measured_vram_mb / file_size_mb
        if file_size_mb
        else 0.0
    )

    record = {
        "overhead": round(max(measured_ratio, 1.0), 2),
        "measured_ratio": round(measured_ratio, 3),
        "measured_vram_mb": round(measured_vram_mb, 1),
        "file_size_mb": round(file_size_mb, 1),
        "model_path": model_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    calibrations = load_calibrations()
    calibrations[quant] = record

    with open(calibrations_path(), "w", encoding="utf-8") as f:
        json.dump(
            calibrations,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return record


def list_calibrations() -> dict[str, dict]:
    """Return a copy of the calibration store."""
    return dict(load_calibrations())
