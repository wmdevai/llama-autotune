"""Tests for calibration.py — per-quantization VRAM calibration persistence."""


from llama_autotune import calibration
from llama_autotune.constraints import _overhead_factor
from llama_autotune.models import ModelInfo


def test_save_and_get_overhead(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibration,
        "calibrations_path",
        lambda: str(tmp_path / "calibrations.json"),
    )
    monkeypatch.setattr(calibration, "_calibrations", None)

    calibration.save_calibration(
        "Q5_K_M",
        measured_vram_mb=9952.0,
        file_size_mb=10025.0,
        model_path="/models/qwen.gguf",
    )

    assert calibration.get_overhead_factor("Q5_K_M") == 1.0


def test_overhead_is_clamped_to_at_least_one(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibration,
        "calibrations_path",
        lambda: str(tmp_path / "calibrations.json"),
    )
    monkeypatch.setattr(calibration, "_calibrations", None)

    record = calibration.save_calibration(
        "Q4_K_S",
        measured_vram_mb=14192.0,
        file_size_mb=14647.0,  # ratio ~0.969
        model_path="/models/qwen.gguf",
    )

    assert record["overhead"] == 1.0
    assert record["measured_ratio"] < 1.0


def test_get_overhead_unknown_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibration,
        "calibrations_path",
        lambda: str(tmp_path / "calibrations.json"),
    )
    monkeypatch.setattr(calibration, "_calibrations", None)

    assert calibration.get_overhead_factor("F16") is None


def test_load_calibrations_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibration,
        "calibrations_path",
        lambda: str(tmp_path / "does-not-exist.json"),
    )
    monkeypatch.setattr(calibration, "_calibrations", None)

    assert calibration.load_calibrations() == {}


def test_overhead_factor_prefers_calibration(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibration,
        "calibrations_path",
        lambda: str(tmp_path / "calibrations.json"),
    )
    monkeypatch.setattr(calibration, "_calibrations", None)

    # Q5_K_M hardcoded fallback is 1.05, but a stored 1.03 must win.
    calibration.save_calibration(
        "Q5_K_M",
        measured_vram_mb=10325.0,
        file_size_mb=10025.0,
        model_path="/models/qwen.gguf",
    )

    model = ModelInfo(quantization="Q5_K_M")
    assert _overhead_factor(model) == 1.03


def test_overhead_factor_falls_back_when_uncalibrated(monkeypatch, tmp_path):
    monkeypatch.setattr(
        calibration,
        "calibrations_path",
        lambda: str(tmp_path / "calibrations.json"),
    )
    monkeypatch.setattr(calibration, "_calibrations", None)

    # No calibration store: K-quants keep the heuristic 1.05.
    model = ModelInfo(quantization="Q5_K_M")
    assert _overhead_factor(model) == 1.05
