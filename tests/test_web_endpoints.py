"""Tests for the Web UI dashboard, inspect, benchmark and index endpoints."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from llama_autotune import web
from llama_autotune.models import (
    Backend,
    BenchmarkResult,
    GpuVendor,
    HardwareInfo,
    ModelInfo,
)


def _client():
    return TestClient(web.app)


def _hardware():
    return HardwareInfo(
        cpu_name="Test CPU",
        physical_cores=8,
        logical_cores=16,
        ram_gb=32.0,
        gpu_count=1,
        gpu_vendor=GpuVendor.NVIDIA,
        gpu_models=["NVIDIA GeForce RTX 4090"],
        vram_per_gpu=[24.0],
        backend=Backend.CUDA,
    )


def _model_info():
    return ModelInfo(
        path="/models/test.gguf",
        architecture="llama",
        parameters=7_000_000_000,
        quantization="Q4_K_M",
        n_layers=32,
        n_heads=32,
        training_context=8192,
    )


def test_index_returns_html():
    response = _client().get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower() or response.text.strip()


def test_dashboard(monkeypatch):
    monkeypatch.setattr(web, "detect_hardware", _hardware)
    monkeypatch.setattr(web, "find_llama_bench", lambda: "/usr/bin/llama-bench")
    monkeypatch.setattr(
        web,
        "find_llama_binary",
        lambda base: f"/usr/bin/{base}",
    )
    monkeypatch.setattr(web, "get_llama_version", lambda p: "b1234")
    monkeypatch.setattr(web, "_gpu_vram_used_mb", lambda: 8192.0)
    monkeypatch.setattr(
        web,
        "gpu_sensors",
        lambda: {"temperature_c": 62, "utilization_pct": 7},
    )

    response = _client().get("/api/dashboard")

    assert response.status_code == 200
    data = response.json()
    assert data["hardware"]["cpu_name"] == "Test CPU"
    assert data["hardware"]["backend"] == "cuda"
    assert data["llama_cpp"]["llama_bench"] == "/usr/bin/llama-bench"
    assert data["llama_cpp"]["version"] == "b1234"
    # New live/usage fields are populated.
    assert data["hardware"]["vram_used_gb"] == 8.0
    assert data["hardware"]["gpu_temperature_c"] == 62
    assert "system" in data
    assert "models_count" in data["system"]
    assert "calibrations_count" in data["system"]


def test_list_models(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    sub = model_dir / "sub"
    sub.mkdir()
    (sub / "model.gguf").write_bytes(b"GGUF")
    (model_dir / "notes.txt").write_text("not a model")

    monkeypatch.setattr(web, "MODEL_DIR", model_dir)

    response = _client().get("/api/models")

    assert response.status_code == 200
    data = response.json()
    assert len(data["models"]) == 1
    assert data["models"][0]["name"] == "model.gguf"


def test_inspect_requires_model_path():
    response = _client().post("/api/inspect", json={"model_path": "  "})
    assert response.status_code == 400


def test_inspect_missing_file():
    response = _client().post(
        "/api/inspect",
        json={"model_path": "/nonexistent/model.gguf"},
    )
    assert response.status_code == 404


def test_inspect_returns_model(monkeypatch, tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    monkeypatch.setattr(web, "inspect_model", lambda path: _model_info())

    response = _client().post(
        "/api/inspect",
        json={"model_path": str(model_file)},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["architecture"] == "llama"
    assert data["quantization"] == "Q4_K_M"


def test_benchmark_requires_model_path():
    response = _client().post("/api/benchmark", json={"model_path": "  "})
    assert response.status_code == 400


def test_benchmark_rejects_zero_repetitions(tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    response = _client().post(
        "/api/benchmark",
        json={"model_path": str(model_file), "repetitions": 0},
    )
    assert response.status_code == 400


def test_benchmark_runs(monkeypatch, tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    captured = {}

    def fake_run_benchmark(model_path, config, repetitions=None):
        captured["model_path"] = model_path
        captured["config"] = config
        captured["repetitions"] = repetitions
        return BenchmarkResult(success=True, generation_tps=42.0)

    monkeypatch.setattr(web, "run_benchmark", fake_run_benchmark)

    response = _client().post(
        "/api/benchmark",
        json={
            "model_path": str(model_file),
            "threads": 8,
            "repetitions": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["result"]["success"] is True
    assert data["result"]["generation_tps"] == 42.0
    assert captured["repetitions"] == 3
    assert captured["config"].threads == 8


def test_get_llama_version_success(monkeypatch):
    result = SimpleNamespace(stdout="version: 1 (b1234)\n", stderr="")
    monkeypatch.setattr(
        web.subprocess,
        "run",
        lambda *args, **kwargs: result,
    )

    assert web.get_llama_version("/usr/bin/llama-server") == "version: 1 (b1234)"


def test_get_llama_version_falls_back_to_stderr(monkeypatch):
    result = SimpleNamespace(stdout="", stderr="version: 2 (b9999)")
    monkeypatch.setattr(
        web.subprocess,
        "run",
        lambda *args, **kwargs: result,
    )

    assert web.get_llama_version("/usr/bin/llama-server") == "version: 2 (b9999)"


def test_get_llama_version_error_returns_unknown(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("no such binary")

    monkeypatch.setattr(web.subprocess, "run", boom)

    assert web.get_llama_version("/usr/bin/llama-server") == "Unknown"


# ── calibration endpoints ────────────────────────────────────────────


def test_calibrate_requires_model_path():
    response = _client().post("/api/calibrate", json={"model_path": "  "})
    assert response.status_code == 400


def test_calibrate_missing_file():
    response = _client().post(
        "/api/calibrate",
        json={"model_path": "/nonexistent/model.gguf"},
    )
    assert response.status_code == 404


def test_calibrate_runs_and_stores(monkeypatch, tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    monkeypatch.setattr(
        web,
        "inspect_model",
        lambda path: ModelInfo(
            quantization="Q5_K_M",
            file_size_gb=9.79,
        ),
    )
    monkeypatch.setattr(
        web,
        "detect_hardware",
        lambda: HardwareInfo(physical_cores=12),
    )
    monkeypatch.setattr(
        web,
        "run_benchmark",
        lambda *args, **kwargs: BenchmarkResult(
            success=True,
            vram_usage=9952.0,
        ),
    )
    monkeypatch.setattr(
        web.calibration,
        "calibrations_path",
        lambda: str(tmp_path / "calibrations.json"),
    )
    monkeypatch.setattr(web.calibration, "_calibrations", None)

    response = _client().post(
        "/api/calibrate",
        json={"model_path": str(model_file)},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["quantization"] == "Q5_K_M"
    assert data["measured_vram_mb"] == 9952.0
    assert data["overhead"] == 1.0

    # The stored calibration must now be listed.
    listing = _client().get("/api/calibrations").json()
    assert "Q5_K_M" in listing["calibrations"]


def test_calibrate_failure_reports_error(monkeypatch, tmp_path):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    monkeypatch.setattr(
        web,
        "inspect_model",
        lambda path: ModelInfo(quantization="Q5_K_M"),
    )
    monkeypatch.setattr(
        web,
        "detect_hardware",
        lambda: HardwareInfo(physical_cores=12),
    )
    monkeypatch.setattr(
        web,
        "run_benchmark",
        lambda *args, **kwargs: BenchmarkResult(success=False),
    )

    response = _client().post(
        "/api/calibrate",
        json={"model_path": str(model_file)},
    )

    assert response.status_code == 500


# ── presets / storage endpoints ─────────────────────────────────────


def test_presets_endpoint(monkeypatch, tmp_path):
    current = tmp_path / "presets.ini"
    recommended = tmp_path / "recommended.ini"
    current.write_text("version = 1\n", encoding="utf-8")
    recommended.write_text("version = 1\n[model]\n", encoding="utf-8")

    monkeypatch.setattr(web.presets, "CURRENT_PRESETS_PATH", current)
    monkeypatch.setattr(web.presets, "RECOMMENDED_PRESETS_PATH", recommended)

    data = _client().get("/api/presets").json()

    assert data["current"]["content"] == "version = 1\n"
    assert data["recommended"]["content"] == "version = 1\n[model]\n"
    assert data["differ"] is True


def test_storage_endpoints(monkeypatch, tmp_path):
    autotune = tmp_path / "autotune"
    autotune.mkdir()
    (autotune / "benchmarks.db").write_bytes(b"x" * 1024)

    monkeypatch.setattr(web.storage, "AUTOTUNE_DIR", autotune)
    monkeypatch.setattr(web.storage, "SLOTS_DIR", tmp_path / "slots")

    listing = _client().get("/api/storage").json()
    assert len(listing["items"]) == 1

    deleted = _client().post(
        "/api/storage/delete",
        json={"key": "benchmarks.db"},
    ).json()
    assert deleted["freed_bytes"] == 1024


def test_storage_delete_unknown_key(monkeypatch, tmp_path):
    monkeypatch.setattr(web.storage, "AUTOTUNE_DIR", tmp_path / "autotune")
    monkeypatch.setattr(web.storage, "SLOTS_DIR", tmp_path / "slots")

    response = _client().post(
        "/api/storage/delete",
        json={"key": "bogus"},
    )
    assert response.status_code == 400


def test_prune_stale_jobs(monkeypatch):
    now = 1_000_000.0
    monkeypatch.setattr(web.time, "time", lambda: now)

    web._jobs.clear()
    try:
        web._jobs["fresh"] = {"queue": None, "created": now - 10}
        web._jobs["stale"] = {
            "queue": None,
            "created": now - web._JOB_TTL_SECONDS - 1,
        }

        web._prune_stale_jobs()

        assert "fresh" in web._jobs
        assert "stale" not in web._jobs
    finally:
        web._jobs.clear()
