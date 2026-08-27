"""Tests for candidates.py — pure candidate-generation helpers."""

from types import SimpleNamespace

from llama_autotune import candidates
from llama_autotune.search_space import ParamDef


def _categorical(*values):
    return ParamDef(
        "batch_size",
        0,
        0,
        is_categorical=True,
        categories=list(values),
    )


# ── grid_values ───────────────────────────────────────────────────────


def test_grid_values_categorical_returns_first_n_categories():
    param = _categorical(128, 256, 512, 1024)
    assert candidates.grid_values(param, 2) == [128, 256]


def test_grid_values_categorical_caps_at_available_categories():
    param = _categorical(128, 256)
    assert candidates.grid_values(param, 5) == [128, 256]


def test_grid_values_with_step():
    param = ParamDef("threads", 2, 8, step=2)
    assert candidates.grid_values(param, 3) == [2, 4, 6]


def test_grid_values_with_step_caps_at_high():
    param = ParamDef("threads", 2, 8, step=2)
    assert candidates.grid_values(param, 100) == [2, 4, 6, 8]


def test_grid_values_without_step_computes_uniform_step():
    param = ParamDef("threads", 2, 8)
    assert candidates.grid_values(param, 4) == [2, 4, 6, 8]


# ── sample_param ──────────────────────────────────────────────────────


def test_sample_param_categorical():
    param = _categorical(128, 256, 512)

    trial = SimpleNamespace()

    def suggest_categorical(name, choices):
        assert name == "batch_size"
        assert choices == [128, 256, 512]
        return 256

    trial.suggest_categorical = suggest_categorical

    assert candidates.sample_param(trial, "batch_size", param) == 256


def test_sample_param_int_with_step():
    param = ParamDef("threads", 2, 8, step=2)

    trial = SimpleNamespace()

    def suggest_int(name, low, high, step=None):
        assert name == "threads"
        assert low == 2
        assert high == 8
        assert step == 2
        return 4

    trial.suggest_int = suggest_int

    assert candidates.sample_param(trial, "threads", param) == 4


def test_sample_param_int_without_step():
    param = ParamDef("threads", 2, 8)

    trial = SimpleNamespace()

    def suggest_int(name, low, high, step=None):
        assert step is None
        return 6

    trial.suggest_int = suggest_int

    assert candidates.sample_param(trial, "threads", param) == 6
