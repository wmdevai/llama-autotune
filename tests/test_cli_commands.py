"""Tests for CLI commands with mocked hardware/model/benchmark dependencies."""

from types import SimpleNamespace

from typer.testing import CliRunner

from llama_autotune.cli import app
from llama_autotune.models import (
    Backend,
    BenchmarkResult,
    GpuVendor,
    HardwareInfo,
    ModelInfo,
    SearchConfig,
)

runner = CliRunner()


def _fake_hardware() -> HardwareInfo:
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


def _fake_model_info() -> ModelInfo:
    return ModelInfo(
        path="/models/test.gguf",
        architecture="llama",
        parameters=7_000_000_000,
        quantization="Q4_K_M",
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        training_context=8192,
        is_moe=False,
        active_parameters=7_000_000_000,
        file_size_gb=4.0,
    )


def test_inspect_command_shows_hardware_and_model(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    monkeypatch.setattr(
        "llama_autotune.cli.detect_hardware",
        _fake_hardware,
    )
    monkeypatch.setattr(
        "llama_autotune.cli.inspect_model",
        lambda path: _fake_model_info(),
    )

    result = runner.invoke(app, ["inspect", str(model_file)])

    assert result.exit_code == 0
    assert "Test CPU" in result.output
    assert "NVIDIA GeForce RTX 4090" in result.output
    assert "llama" in result.output
    assert "Q4_K_M" in result.output
    assert "7,000,000,000" in result.output


def test_benchmark_command_uses_heuristic_config(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    monkeypatch.setattr(
        "llama_autotune.cli.detect_hardware",
        _fake_hardware,
    )
    monkeypatch.setattr(
        "llama_autotune.cli.inspect_model",
        lambda path: _fake_model_info(),
    )

    heuristic_config = SearchConfig(threads=8, batch_size=2048)
    monkeypatch.setattr(
        "llama_autotune.cli.generate_initial_config",
        lambda hw, model: heuristic_config,
    )

    captured = {}

    def fake_run_benchmark(model, config, repetitions=None):
        captured["config"] = config
        captured["repetitions"] = repetitions
        return BenchmarkResult(
            success=True,
            prompt_tps=1000.0,
            generation_tps=50.0,
            startup_time=2.0,
            memory_usage=512.0,
        )

    monkeypatch.setattr(
        "llama_autotune.cli.run_benchmark",
        fake_run_benchmark,
    )

    result = runner.invoke(app, ["benchmark", str(model_file), "-r", "5"])

    assert result.exit_code == 0
    assert captured["config"] == heuristic_config
    assert captured["repetitions"] == 5
    assert "Benchmark Results" in result.output


def test_benchmark_command_builds_config_from_params(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    monkeypatch.setattr(
        "llama_autotune.cli.detect_hardware",
        _fake_hardware,
    )
    monkeypatch.setattr(
        "llama_autotune.cli.inspect_model",
        lambda path: _fake_model_info(),
    )

    captured = {}

    def fake_run_benchmark(model, config, repetitions=None):
        captured["config"] = config
        return BenchmarkResult(success=True)

    monkeypatch.setattr(
        "llama_autotune.cli.run_benchmark",
        fake_run_benchmark,
    )

    result = runner.invoke(
        app,
        [
            "benchmark",
            str(model_file),
            "-t",
            "8",
            "-b",
            "512",
            "-c",
            "4096",
        ],
    )

    assert result.exit_code == 0
    config = captured["config"]
    assert config.threads == 8
    assert config.batch_size == 512
    assert config.ctx_size == 4096


def test_export_command_writes_profile(tmp_path):
    profile_path = tmp_path / "my_profile.json"

    result = runner.invoke(
        app,
        [
            "export",
            str(profile_path),
            "--model",
            "/models/test.gguf",
            "--hardware",
            "RTX 4090",
            "--score",
            "95.5",
        ],
    )

    assert result.exit_code == 0
    assert profile_path.exists()
    assert "Profile exported" in result.output

    import json

    data = json.loads(profile_path.read_text())
    assert data["model_path"] == "/models/test.gguf"
    assert data["hardware"] == "RTX 4090"
    assert data["score"] == 95.5


def test_import_command_shows_profile(tmp_path):
    import json

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "name": "test_profile",
                "args": ["-t", "8"],
                "model_path": "/models/test.gguf",
                "hardware": "Test CPU",
                "created": "2026-08-27T00:00:00+00:00",
                "score": 42.0,
            }
        )
    )

    result = runner.invoke(app, ["import", str(profile_path)])

    assert result.exit_code == 0
    assert "test_profile" in result.output
    assert "/models/test.gguf" in result.output
    assert "Test CPU" in result.output


def test_inspect_model_error_exits_nonzero(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    monkeypatch.setattr(
        "llama_autotune.cli.detect_hardware",
        _fake_hardware,
    )

    def boom(path):
        raise FileNotFoundError("cannot read model")

    monkeypatch.setattr("llama_autotune.cli.inspect_model", boom)

    result = runner.invoke(app, ["inspect", str(model_file)])

    assert result.exit_code == 1
    assert "Error reading model" in result.output


def test_benchmark_failure_exits_nonzero(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    monkeypatch.setattr(
        "llama_autotune.cli.detect_hardware",
        _fake_hardware,
    )
    monkeypatch.setattr(
        "llama_autotune.cli.inspect_model",
        lambda path: _fake_model_info(),
    )
    monkeypatch.setattr(
        "llama_autotune.cli.generate_initial_config",
        lambda hw, model: SearchConfig(threads=8),
    )
    monkeypatch.setattr(
        "llama_autotune.cli.run_benchmark",
        lambda *args, **kwargs: BenchmarkResult(success=False),
    )

    result = runner.invoke(app, ["benchmark", str(model_file)])

    assert result.exit_code == 1


def test_launch_command_execs_llama_server(tmp_path, monkeypatch):
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"GGUF")

    monkeypatch.setattr(
        "llama_autotune.cli.detect_hardware",
        _fake_hardware,
    )
    monkeypatch.setattr(
        "llama_autotune.cli.inspect_model",
        lambda path: _fake_model_info(),
    )
    monkeypatch.setattr(
        "llama_autotune.cli.generate_initial_config",
        lambda hw, model: SearchConfig(threads=8, ctx_size=4096),
    )
    monkeypatch.setattr(
        "llama_autotune.cli.find_llama_binary",
        lambda base: f"/usr/bin/{base}",
    )

    execv_calls = []

    def fake_execvp(binary, cmd):
        execv_calls.append((binary, cmd))

    monkeypatch.setattr("llama_autotune.cli.os.execvp", fake_execvp)

    result = runner.invoke(
        app,
        ["launch", str(model_file), "--host", "0.0.0.0", "--port", "9999"],
    )

    assert result.exit_code == 0
    assert execv_calls
    binary, cmd = execv_calls[0]
    assert binary == "/usr/bin/llama-server"
    assert "--host" in cmd and "0.0.0.0" in cmd
    assert "--port" in cmd and "9999" in cmd
