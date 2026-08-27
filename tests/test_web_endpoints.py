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

    response = _client().get("/api/dashboard")

    assert response.status_code == 200
    data = response.json()
    assert data["hardware"]["cpu_name"] == "Test CPU"
    assert data["hardware"]["backend"] == "cuda"
    assert data["llama_cpp"]["llama_bench"] == "/usr/bin/llama-bench"
    assert data["llama_cpp"]["version"] == "b1234"


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
