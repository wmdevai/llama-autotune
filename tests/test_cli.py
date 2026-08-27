"""Tests for cli.py — command registration and error handling."""

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from llama_autotune.cli import app
from llama_autotune.models import BenchmarkResult, SearchConfig

runner = CliRunner()


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "llama-autotune" in result.output


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["inspect", "benchmark", "search", "launch", "export", "import"]:
        assert cmd in result.output


def test_import_command_is_named_import():
    """The command must be `import` (as documented), not `import-cmd`."""
    result = runner.invoke(app, ["--help"])
    assert "import-cmd" not in result.output
    result = runner.invoke(app, ["import", "--help"])
    assert result.exit_code == 0


def test_inspect_missing_model_exits_nonzero():
    result = runner.invoke(app, ["inspect", "no_such_model.gguf"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_import_missing_profile_exits_nonzero():
    result = runner.invoke(app, ["import", "no_such_profile.json"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_search_invalid_objective_exits_nonzero(tmp_path):
    fake_model = tmp_path / "model.gguf"
    fake_model.write_bytes(b"GGUF")
    result = runner.invoke(app, ["search", str(fake_model), "--objective", "bogus"])
    assert result.exit_code == 1
    assert "Invalid objective" in result.output

@pytest.mark.parametrize(
    ("extra_args", "expected_slow"),
    [
        ([], False),
        (["--slow"], True),
    ],
)
def test_search_forwards_slow_option(
    tmp_path,
    monkeypatch,
    extra_args,
    expected_slow,
):
    """The CLI must forward --slow to Optimizer."""

    fake_model = tmp_path / "model.gguf"
    fake_model.write_bytes(b"GGUF")

    captured = {}

    class FakeOptimizer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

            self.hw = SimpleNamespace(
                cpu_name="Test CPU",
                physical_cores=4,
                logical_cores=8,
                ram_gb=16,
                backend=SimpleNamespace(value="cpu"),
            )

            self.model = SimpleNamespace(
                architecture="test",
                parameters=1_000,
                quantization="Q4_K_M",
                is_moe=False,
            )

            self.objective = kwargs["objective"]
            self._bench_reps = 1
            self._n_prompt = 16
            self._n_gen = 8
            self.best_score = 1.0
            self.total_evals = 1
            self._cache = {}

        def run(self):
            return SearchConfig(
                ctx_size=4096,
            )

        def validate_full_context(self, config):
            return BenchmarkResult(
                success=True,
                prompt_tps=10.0,
                generation_tps=20.0,
                startup_time=1.0,
                memory_usage=1000.0,
            )

    monkeypatch.setattr(
        "llama_autotune.cli.Optimizer",
        FakeOptimizer,
    )

    monkeypatch.setattr(
        "llama_autotune.cli.run_benchmark",
        lambda *args, **kwargs: BenchmarkResult(
            success=True,
            prompt_tps=10.0,
            generation_tps=20.0,
            startup_time=1.0,
            memory_usage=1000.0,
        ),
    )

    result = runner.invoke(
        app,
        [
            "search",
            str(fake_model),
            *extra_args,
        ],
    )

    assert result.exit_code == 0
    assert captured["slow"] is expected_slow

