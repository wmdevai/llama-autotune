"""Tests for Optimizer._score — objective-specific scoring."""

from types import SimpleNamespace

import optuna

from llama_autotune.models import BenchmarkResult, OptimizeObjective, SearchConfig
from llama_autotune.optimizer import Optimizer
from llama_autotune.web import OptimizeRequest
from llama_autotune.search_space import ParamDef


def test_web_optimize_request_defaults_match_optimizer():
    """Web API defaults must match the optimizer and CLI defaults."""

    request = OptimizeRequest(model_path="model.gguf")

    assert request.trials_b == 12
    assert request.trials_c == 20



def test_stage_b_uses_latest_best_config_between_parameters(
    monkeypatch,
):
    """Stage B must carry improvements forward between parameters."""

    opt = Optimizer.__new__(Optimizer)

    opt.objective = OptimizeObjective.BALANCED
    opt.n_trials_stage_b = 4
    opt._best_config = SearchConfig(
        threads=4,
        ctx_size=4096,
    )
    opt._best_score = 0.0
    opt._cache = {}
    opt._total_evals = 0

    opt.hw = SimpleNamespace(
        physical_cores=8,
    )
    opt.model = SimpleNamespace()

    space = {
        "threads": ParamDef(
            "threads",
            2,
            8,
            step=2,
        ),
        "ctx_size": ParamDef(
            "ctx_size",
            4096,
            16384,
            step=4096,
        ),
    }

    monkeypatch.setattr(
        "llama_autotune.optimizer.get_search_space",
        lambda *args, **kwargs: space,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.is_plausible",
        lambda *args, **kwargs: True,
    )

    evaluated = []

    scores = {
        (2, 4096): 1.0,
        (6, 4096): 10.0,
        (6, 8192): 20.0,
        (6, 16384): 5.0,
    }

    def evaluate(cfg):
        evaluated.append(
            (
                cfg.threads,
                cfg.ctx_size,
            )
        )

        return _result()

    def score(result, cfg):
        return scores[
            (
                cfg.threads,
                cfg.ctx_size,
            )
        ]

    opt._evaluate = evaluate
    opt._score = score

    opt._stage_b_local_search()

    assert evaluated == [
        (2, 4096),
        (6, 4096),
        (6, 8192),
        (6, 16384),
    ]

    assert opt._best_config.threads == 6
    assert opt._best_config.ctx_size == 8192
    assert opt._best_score == 20.0


def _make_optimizer(objective: OptimizeObjective) -> Optimizer:
    """Build an Optimizer without hardware detection or model inspection."""
    opt = Optimizer.__new__(Optimizer)
    opt.objective = objective
    return opt


def _result(gen=10.0, prompt=50.0, startup=5.0, memory=2000.0) -> BenchmarkResult:
    return BenchmarkResult(
        generation_tps=gen,
        prompt_tps=prompt,
        startup_time=startup,
        memory_usage=memory,
        success=True,
    )


def test_stage_b_skips_cached_candidates_and_reaches_budget(
    monkeypatch,
):
    """Stage B should reach its budget with new valid evaluations."""

    opt = Optimizer.__new__(Optimizer)

    opt.objective = OptimizeObjective.BALANCED
    opt.n_trials_stage_b = 2
    opt._best_config = SearchConfig(
        threads=4,
    )
    opt._best_score = 0.0
    opt._cache = {}
    opt._total_evals = 0

    opt.hw = SimpleNamespace(
        physical_cores=8,
    )
    opt.model = SimpleNamespace()

    space = {
        "threads": ParamDef(
            "threads",
            2,
            8,
            step=2,
        ),
    }

    monkeypatch.setattr(
        "llama_autotune.optimizer.get_search_space",
        lambda *args, **kwargs: space,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.is_plausible",
        lambda *args, **kwargs: True,
    )

    opt._local_values = (
        lambda param, current_value, count: [2, 6, 8]
    )

    cached_cfg = SearchConfig(
        threads=2,
    )

    opt._cache[
        opt._config_key(cached_cfg)
    ] = _result()

    evaluated = []

    def evaluate(cfg):
        key = opt._config_key(cfg)

        if key in opt._cache:
            return opt._cache[key]

        evaluated.append(cfg.threads)

        result = _result()
        opt._cache[key] = result
        opt._total_evals += 1

        return result

    def score(result, cfg):
        return float(cfg.threads)

    opt._evaluate = evaluate
    opt._score = score

    opt._stage_b_local_search()

    assert evaluated == [6, 8]
    assert opt._best_score == 8.0



def test_stage_b_reaches_budget_after_cached_and_duplicate_candidates(
    monkeypatch,
):
    """Stage B must reach its budget after cache hits and duplicates."""

    opt = Optimizer.__new__(Optimizer)

    opt.objective = OptimizeObjective.BALANCED
    opt.n_trials_stage_b = 2
    opt._best_config = SearchConfig(
        threads=4,
    )
    opt._best_score = 0.0
    opt._cache = {}
    opt._total_evals = 0

    opt.hw = SimpleNamespace(
        physical_cores=8,
    )
    opt.model = SimpleNamespace()

    space = {
        "threads": ParamDef(
            "threads",
            2,
            8,
            step=2,
        ),
    }

    monkeypatch.setattr(
        "llama_autotune.optimizer.get_search_space",
        lambda *args, **kwargs: space,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.is_plausible",
        lambda *args, **kwargs: True,
    )

    opt._local_values = (
        lambda param, current_value, count: [2, 2, 6, 8]
    )

    cached_cfg = SearchConfig(
        threads=2,
    )

    opt._cache[
        opt._config_key(cached_cfg)
    ] = _result()

    evaluated = []

    def evaluate(cfg):
        key = opt._config_key(cfg)

        if key in opt._cache:
            return opt._cache[key]

        evaluated.append(cfg.threads)

        result = _result()

        opt._cache[key] = result
        opt._total_evals += 1

        return result

    def score(result, cfg):
        return float(cfg.threads)

    opt._evaluate = evaluate
    opt._score = score

    opt._stage_b_local_search()

    assert evaluated == [6, 8]
    assert opt._best_score == 8.0



def test_stage_b_skips_duplicate_candidates(
    monkeypatch,
):
    """Stage B must not benchmark the same candidate twice."""

    opt = Optimizer.__new__(Optimizer)

    opt.objective = OptimizeObjective.BALANCED
    opt.n_trials_stage_b = 2
    opt._best_config = SearchConfig(
        threads=4,
    )
    opt._best_score = 0.0
    opt._cache = {}
    opt._total_evals = 0

    opt.hw = SimpleNamespace(
        physical_cores=8,
    )
    opt.model = SimpleNamespace()

    space = {
        "threads": ParamDef(
            "threads",
            2,
            8,
            step=2,
        ),
    }

    monkeypatch.setattr(
        "llama_autotune.optimizer.get_search_space",
        lambda *args, **kwargs: space,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.is_plausible",
        lambda *args, **kwargs: True,
    )

    opt._local_values = (
        lambda param, current_value, count: [2, 2, 6]
    )

    evaluated = []

    def evaluate(cfg):
        key = opt._config_key(cfg)

        if key in opt._cache:
            raise AssertionError(
                "Stage B attempted a duplicate benchmark"
            )

        evaluated.append(cfg.threads)

        result = _result()
        opt._cache[key] = result
        opt._total_evals += 1

        return result

    def score(result, cfg):
        return float(cfg.threads)

    opt._evaluate = evaluate
    opt._score = score

    opt._stage_b_local_search()

    assert evaluated == [2, 6]
    assert opt._best_score == 6.0



def test_stage_b_stops_when_unique_candidates_are_exhausted(
    monkeypatch,
):
    """Stage B must stop without duplicate benchmarks when candidates end."""

    opt = Optimizer.__new__(Optimizer)

    opt.objective = OptimizeObjective.BALANCED
    opt.n_trials_stage_b = 3
    opt._best_config = SearchConfig(
        threads=4,
    )
    opt._best_score = 0.0
    opt._cache = {}
    opt._total_evals = 0

    opt.hw = SimpleNamespace(
        physical_cores=8,
    )
    opt.model = SimpleNamespace()

    space = {
        "threads": ParamDef(
            "threads",
            2,
            4,
            step=2,
        ),
    }

    monkeypatch.setattr(
        "llama_autotune.optimizer.get_search_space",
        lambda *args, **kwargs: space,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.is_plausible",
        lambda *args, **kwargs: True,
    )

    opt._local_values = (
        lambda param, current_value, count: [2, 4]
    )

    evaluated = []

    def evaluate(cfg):
        key = opt._config_key(cfg)

        if key in opt._cache:
            raise AssertionError(
                "Stage B attempted a duplicate benchmark"
            )

        evaluated.append(cfg.threads)

        result = _result()
        opt._cache[key] = result
        opt._total_evals += 1

        return result

    def score(result, cfg):
        return float(cfg.threads)

    opt._evaluate = evaluate
    opt._score = score

    opt._stage_b_local_search()

    assert evaluated == [2, 4]
    assert opt._best_score == 4.0



def test_stage_c_trials_keep_frozen_base_after_new_best(
    monkeypatch,
):
    """Stage C trials must not inherit earlier Stage C improvements."""

    opt = Optimizer.__new__(Optimizer)

    opt.objective = OptimizeObjective.BALANCED
    opt.n_trials_stage_c = 2
    opt._best_config = SearchConfig(
        threads=4,
        ctx_size=4096,
    )
    opt._best_score = 0.0
    opt._cache = {}
    opt._total_evals = 0

    opt.hw = SimpleNamespace()
    opt.model = SimpleNamespace()

    space = {
        "threads": ParamDef(
            "threads",
            2,
            8,
            step=2,
        ),
        "ctx_size": ParamDef(
            "ctx_size",
            4096,
            16384,
            step=4096,
        ),
    }

    monkeypatch.setattr(
        "llama_autotune.optimizer.get_search_space",
        lambda *args, **kwargs: space,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.is_plausible",
        lambda *args, **kwargs: True,
    )

    def local_values(param, current_value, count):
        if param.name == "threads":
            return [6]

        return [8192]

    def strategic_ctx_values(
        param,
        current_value,
        count,
        include_current=True,
    ):
        return [8192]

    opt._local_values = local_values
    opt._strategic_ctx_values = strategic_ctx_values

    evaluated = []

    def evaluate(cfg):
        evaluated.append(
            (
                cfg.threads,
                cfg.ctx_size,
            )
        )
        return _result()

    scores = {
        (6, 4096): 10.0,
        (4, 8192): 20.0,
    }

    def score(result, cfg):
        return scores[
            (
                cfg.threads,
                cfg.ctx_size,
            )
        ]

    opt._evaluate = evaluate
    opt._score = score

    class FakeTrial:
        def __init__(self, number, values):
            self.number = number
            self._values = values

        def suggest_categorical(self, name, choices):
            value = self._values[name]
            assert value in choices
            return value

    class FakeStudy:
        def __init__(self):
            self.trials = []
            self._next = 0
            self._choices = [
                {
                    "threads": 6,
                    "ctx_size": 4096,
                },
                {
                    "threads": 4,
                    "ctx_size": 8192,
                },
            ]

        def optimize(
            self,
            objective_fn,
            n_trials,
            show_progress_bar,
        ):
            for _ in range(n_trials):
                values = self._choices[self._next]
                trial = FakeTrial(
                    self._next,
                    values,
                )
                self._next += 1

                score = objective_fn(trial)

                self.trials.append(
                    SimpleNamespace(
                        state=optuna.trial.TrialState.COMPLETE,
                        value=score,
                    )
                )

    fake_study = FakeStudy()

    monkeypatch.setattr(
        "llama_autotune.optimizer.optuna.create_study",
        lambda **kwargs: fake_study,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.optuna.samplers.TPESampler",
        lambda **kwargs: object(),
    )

    opt._stage_c_bayesian()

    assert evaluated == [
        (6, 4096),
        (4, 8192),
    ]

    assert opt._best_config.threads == 4
    assert opt._best_config.ctx_size == 8192
    assert opt._best_score == 20.0


def test_score_max_generation_tps():
    opt = _make_optimizer(OptimizeObjective.MAX_GENERATION_TPS)
    assert opt._score(_result(gen=42.0), SearchConfig()) == 42.0


def test_score_max_prompt_tps():
    opt = _make_optimizer(OptimizeObjective.MAX_PROMPT_TPS)
    assert opt._score(_result(prompt=123.0), SearchConfig()) == 123.0


def test_score_min_latency_is_negative_startup():
    opt = _make_optimizer(OptimizeObjective.MIN_LATENCY)
    assert opt._score(_result(startup=5.0), SearchConfig()) == -5.0


def test_score_min_latency_faster_beats_slower():
    """A 2s startup must outscore a 5s startup (higher is better)."""
    opt = _make_optimizer(OptimizeObjective.MIN_LATENCY)
    fast = opt._score(_result(startup=2.0), SearchConfig())
    slow = opt._score(_result(startup=5.0), SearchConfig())
    assert fast > slow


def test_score_min_latency_real_config_beats_failure_sentinel():
    """Any working config must outrank the old -1.0 failure sentinel.

    Regression test: stage C previously returned -1.0 for failed configs,
    which outranked legitimate min_latency scores like -5.0.
    """
    opt = _make_optimizer(OptimizeObjective.MIN_LATENCY)
    score = opt._score(_result(startup=0.5), SearchConfig())
    assert score > -1.0


def test_score_max_context_scales_with_ctx():
    opt = _make_optimizer(OptimizeObjective.MAX_CONTEXT)
    small = opt._score(_result(gen=100.0), SearchConfig(ctx_size=4096))
    large = opt._score(_result(gen=1.0), SearchConfig(ctx_size=32768))
    assert large > small


def test_score_max_context_gen_tps_breaks_ties():
    opt = _make_optimizer(OptimizeObjective.MAX_CONTEXT)
    slower = opt._score(_result(gen=5.0), SearchConfig(ctx_size=8192))
    faster = opt._score(_result(gen=10.0), SearchConfig(ctx_size=8192))
    assert faster > slower


def test_score_max_context_none_ctx():
    opt = _make_optimizer(OptimizeObjective.MAX_CONTEXT)
    score = opt._score(_result(gen=10.0), SearchConfig())
    assert score == 10.0 / 1000.0


def test_score_max_efficiency():
    opt = _make_optimizer(OptimizeObjective.MAX_EFFICIENCY)
    assert opt._score(_result(gen=10.0, memory=2000.0), SearchConfig()) == 10.0 / 2000.0


def test_score_balanced_defaults_to_generation():
    opt = _make_optimizer(OptimizeObjective.BALANCED)
    assert opt._score(_result(gen=7.0), SearchConfig()) == 7.0


def test_benchmark_prompt_tokens_scales_with_context():
    opt = Optimizer.__new__(Optimizer)
    opt._n_prompt = 512

    assert opt._benchmark_prompt_tokens(SearchConfig(ctx_size=4096)) == 512
    assert opt._benchmark_prompt_tokens(SearchConfig(ctx_size=8192)) == 1024
    assert opt._benchmark_prompt_tokens(SearchConfig(ctx_size=16384)) == 2048
    assert opt._benchmark_prompt_tokens(SearchConfig(ctx_size=24576)) == 3072


def test_benchmark_prompt_tokens_uses_base_without_context():
    opt = Optimizer.__new__(Optimizer)
    opt._n_prompt = 512

    assert opt._benchmark_prompt_tokens(SearchConfig()) == 512



def test_stage_b_uses_local_candidates_for_batch_and_ubatch(
    monkeypatch,
):
    """Stage B must search locally around batch and ubatch values."""

    opt = Optimizer.__new__(Optimizer)

    opt.objective = OptimizeObjective.BALANCED
    opt.n_trials_stage_b = 2
    opt._best_config = SearchConfig(
        batch_size=2048,
        ubatch_size=512,
    )
    opt._best_score = 0.0
    opt._cache = {}
    opt._total_evals = 0

    opt.hw = SimpleNamespace(
        physical_cores=8,
    )
    opt.model = SimpleNamespace()

    space = {
        "batch_size": ParamDef(
            "batch_size",
            128,
            8192,
            step=128,
        ),
        "ubatch_size": ParamDef(
            "ubatch_size",
            64,
            1024,
            step=64,
        ),
    }

    monkeypatch.setattr(
        "llama_autotune.optimizer.get_search_space",
        lambda *args, **kwargs: space,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.is_plausible",
        lambda *args, **kwargs: True,
    )

    local_calls = []

    def local_values(param, current_value, count):
        local_calls.append(
            (
                param.name,
                current_value,
                count,
            )
        )

        if param.name == "batch_size":
            return [1920]

        if param.name == "ubatch_size":
            return [448]

        raise AssertionError(
            f"Unexpected parameter: {param.name}"
        )

    opt._local_values = local_values

    evaluated = []

    def evaluate(cfg):
        evaluated.append(
            (
                cfg.batch_size,
                cfg.ubatch_size,
            )
        )

        result = _result()
        opt._total_evals += 1
        return result

    opt._evaluate = evaluate

    def score(result, cfg):
        return float(cfg.batch_size + cfg.ubatch_size)

    opt._score = score

    opt._stage_b_local_search()

    assert local_calls == [
        ("batch_size", 2048, 1),
        ("ubatch_size", 512, 1),
    ]

    assert evaluated == [
        (1920, 512),
        (1920, 448),
    ]



def test_stage_b_spread_values_cover_batch_range():
    opt = Optimizer.__new__(Optimizer)
    param = ParamDef("batch_size", 128, 8192, step=128)

    assert opt._stage_b_spread_values(param, 2) == [128, 8192]
    assert opt._stage_b_spread_values(param, 3) == [128, 4224, 8192]


def test_stage_b_spread_values_cover_ubatch_range():
    opt = Optimizer.__new__(Optimizer)
    param = ParamDef("ubatch_size", 64, 1024, step=64)

    assert opt._stage_b_spread_values(param, 2) == [64, 1024]
    assert opt._stage_b_spread_values(param, 3) == [64, 576, 1024]


def test_stage_b_uses_alternative_ngl_values_when_full_offload_is_baseline(
    monkeypatch,
):
    """Stage B must not spend its NGL budget on an already-cached baseline."""

    opt = Optimizer.__new__(Optimizer)

    opt.objective = OptimizeObjective.BALANCED
    opt.n_trials_stage_b = 3
    opt._best_config = SearchConfig(
        n_gpu_layers=999,
    )
    opt._best_score = 0.0
    opt._cache = {}
    opt._total_evals = 0

    opt.hw = SimpleNamespace(
        physical_cores=8,
    )
    opt.model = SimpleNamespace()

    space = {
        "n_gpu_layers": ParamDef(
            "n_gpu_layers",
            1,
            40,
            step=10,
        ),
    }

    monkeypatch.setattr(
        "llama_autotune.optimizer.get_search_space",
        lambda *args, **kwargs: space,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.is_plausible",
        lambda *args, **kwargs: True,
    )

    evaluated = []

    def evaluate(cfg):
        evaluated.append(cfg.n_gpu_layers)
        opt._total_evals += 1
        return _result()

    opt._evaluate = evaluate
    opt._score = lambda result, cfg: float(cfg.n_gpu_layers)

    opt._stage_b_local_search()

    assert evaluated == [
        40,
        39,
        38,
    ]

    assert opt._total_evals == 3


def test_stage_b_ngl_preserves_full_offload_semantic_value():
    opt = Optimizer.__new__(Optimizer)
    param = ParamDef('n_gpu_layers', 1, 40, step=10)

    assert opt._stage_b_ngl_values(
        param,
        current_value=999,
        count=3,
    ) == [999, 40, 39]


def test_strategic_ctx_values_preserves_current_and_fills_budget():
    opt = Optimizer.__new__(Optimizer)
    param = ParamDef("ctx_size", 1024, 32768, step=1024)

    assert opt._strategic_ctx_values(
        param,
        current_value=24576,
        count=3,
    ) == [24576, 28672, 32768]


def test_strategic_ctx_values_can_exclude_current_value():
    """Stage B must spend its context budget on alternative candidates."""
    opt = Optimizer.__new__(Optimizer)
    param = ParamDef("ctx_size", 4096, 16384, step=4096)

    assert opt._strategic_ctx_values(
        param,
        current_value=4096,
        count=2,
        include_current=False,
    ) == [8192, 16384]


def test_local_values_ngl_semantic_full_offload():
    """A semantic full-offload value explores concrete layers below the maximum."""
    opt = Optimizer.__new__(Optimizer)
    param = ParamDef("n_gpu_layers", 1, 40, step=10)

    assert opt._local_values(
        param,
        current_value=999,
        count=4,
    ) == [40, 39, 38, 37]


def test_local_values_ngl_numeric_maximum():
    """The numeric full-offload boundary is refined one layer at a time."""
    opt = Optimizer.__new__(Optimizer)
    param = ParamDef("n_gpu_layers", 1, 40, step=10)

    assert opt._local_values(
        param,
        current_value=40,
        count=4,
    ) == [39, 38, 37, 36]


def test_local_values_batch_maximum_preserves_step():
    """Batch refinement must remain aligned to the configured step."""
    opt = Optimizer.__new__(Optimizer)
    param = ParamDef("batch_size", 128, 8192, step=128)

    assert opt._local_values(
        param,
        current_value=8192,
        count=4,
    ) == [8064, 7936, 7808, 7680]


def test_local_values_ubatch_maximum_preserves_step():
    """Micro-batch refinement must remain aligned to the configured step."""
    opt = Optimizer.__new__(Optimizer)
    param = ParamDef("ubatch_size", 64, 1024, step=64)

    assert opt._local_values(
        param,
        current_value=1024,
        count=4,
    ) == [960, 896, 832, 768]


def test_stage_c_tpe_startup_trials_scale_with_budget(
    monkeypatch,
):
    """Stage C must scale TPE startup trials to its evaluation budget."""

    captured = {}

    original = optuna.samplers.TPESampler

    def sampler_factory(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    opt = Optimizer.__new__(Optimizer)

    opt.objective = OptimizeObjective.BALANCED
    opt.n_trials_stage_c = 4
    opt._best_config = SearchConfig(
        threads=4,
    )
    opt._best_score = 0.0
    opt._cache = {}
    opt._total_evals = 0
    opt.hw = SimpleNamespace(
        physical_cores=8,
    )
    opt.model = SimpleNamespace()

    space = {
        "threads": ParamDef(
            "threads",
            2,
            8,
            step=2,
        ),
    }

    monkeypatch.setattr(
        "llama_autotune.optimizer.get_search_space",
        lambda *args, **kwargs: space,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.is_plausible",
        lambda *args, **kwargs: True,
    )

    opt._evaluate = lambda cfg: _result()
    opt._score = lambda result, cfg: float(cfg.threads)

    monkeypatch.setattr(
        "llama_autotune.optimizer.optuna.samplers.TPESampler",
        sampler_factory,
    )

    opt._stage_c_bayesian()

    assert captured["seed"] == 42
    assert captured["n_startup_trials"] == 2


def test_stage_c_reaches_requested_valid_evaluations(
    monkeypatch,
):
    """Stage C should perform the requested number of valid evaluations."""

    opt = Optimizer.__new__(Optimizer)

    opt.objective = OptimizeObjective.BALANCED
    opt.n_trials_stage_c = 3
    opt._best_config = SearchConfig(
        threads=4,
        ctx_size=4096,
    )
    opt._best_score = 0.0
    opt._cache = {}
    opt._total_evals = 0

    opt.hw = SimpleNamespace(
        physical_cores=8,
    )
    opt.model = SimpleNamespace()

    space = {
        "threads": ParamDef(
            "threads",
            2,
            8,
            step=2,
        ),
    }

    monkeypatch.setattr(
        "llama_autotune.optimizer.get_search_space",
        lambda *args, **kwargs: space,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.is_plausible",
        lambda *args, **kwargs: True,
    )

    evaluated = []

    def evaluate(cfg):
        key = opt._config_key(cfg)

        if key in opt._cache:
            return opt._cache[key]

        evaluated.append(cfg.threads)

        result = _result()
        opt._cache[key] = result
        opt._total_evals += 1

        return result

    def score(result, cfg):
        return float(cfg.threads)

    opt._evaluate = evaluate
    opt._score = score

    opt._stage_c_bayesian()

    assert len(evaluated) == 3
    assert len(set(evaluated)) == 3
    assert opt._best_score == max(evaluated)



def test_stage_c_skips_preloaded_cache_and_reaches_budget(
    monkeypatch,
):
    """Stage C must skip cached configs and still reach its valid budget."""

    opt = Optimizer.__new__(Optimizer)

    opt.objective = OptimizeObjective.BALANCED
    opt.n_trials_stage_c = 2
    opt._best_config = SearchConfig(
        threads=4,
    )
    opt._best_score = 0.0
    opt._cache = {}
    opt._total_evals = 0

    opt.hw = SimpleNamespace(
        physical_cores=8,
    )
    opt.model = SimpleNamespace()

    space = {
        "threads": ParamDef(
            "threads",
            2,
            8,
            step=2,
        ),
    }

    monkeypatch.setattr(
        "llama_autotune.optimizer.get_search_space",
        lambda *args, **kwargs: space,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.is_plausible",
        lambda *args, **kwargs: True,
    )

    cached_cfg = SearchConfig(
        threads=2,
    )
    opt._cache[
        opt._config_key(cached_cfg)
    ] = _result()

    evaluated = []

    def evaluate(cfg):
        key = opt._config_key(cfg)

        if key in opt._cache:
            return opt._cache[key]

        evaluated.append(cfg.threads)

        result = _result()
        opt._cache[key] = result
        opt._total_evals += 1

        return result

    def score(result, cfg):
        return float(cfg.threads)

    opt._evaluate = evaluate
    opt._score = score

    opt._stage_c_bayesian()

    assert len(evaluated) == 2
    assert 2 not in evaluated
    assert len(set(evaluated)) == 2
    assert opt._best_score > 0.0



def test_stage_c_stops_immediately_when_unique_space_is_exhausted(
    monkeypatch,
):
    """Stage C must stop immediately when unique candidates are exhausted."""

    opt = Optimizer.__new__(Optimizer)

    opt.objective = OptimizeObjective.BALANCED
    opt.n_trials_stage_c = 3
    opt._best_config = SearchConfig(
        threads=4,
    )
    opt._best_score = 0.0
    opt._cache = {}
    opt._total_evals = 0

    opt.hw = SimpleNamespace(
        physical_cores=8,
    )
    opt.model = SimpleNamespace()

    space = {
        "threads": ParamDef(
            "threads",
            2,
            4,
            step=2,
        ),
    }

    monkeypatch.setattr(
        "llama_autotune.optimizer.get_search_space",
        lambda *args, **kwargs: space,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.is_plausible",
        lambda *args, **kwargs: True,
    )

    evaluated = []

    def evaluate(cfg):
        key = opt._config_key(cfg)

        if key in opt._cache:
            raise AssertionError(
                "Stage C attempted a duplicate benchmark"
            )

        evaluated.append(cfg.threads)

        result = _result()
        opt._cache[key] = result
        opt._total_evals += 1

        return result

    def score(result, cfg):
        return float(cfg.threads)

    opt._evaluate = evaluate
    opt._score = score

    original_create_study = optuna.create_study
    captured = {}

    def create_study(*args, **kwargs):
        study = original_create_study(
            *args,
            **kwargs,
        )

        captured["study"] = study

        return study

    monkeypatch.setattr(
        "llama_autotune.optimizer.optuna.create_study",
        create_study,
    )

    opt._stage_c_bayesian()

    assert set(evaluated) == {2, 4}

    study = captured["study"]

    assert len(
        [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
    ) == 2



def test_stage_c_stops_when_remaining_space_is_exhausted_by_cache(
    monkeypatch,
):
    """Stage C must stop after all remaining uncached candidates are tried."""

    opt = Optimizer.__new__(Optimizer)

    opt.objective = OptimizeObjective.BALANCED
    opt.n_trials_stage_c = 2
    opt._best_config = SearchConfig(
        threads=4,
    )
    opt._best_score = 0.0
    opt._cache = {}
    opt._total_evals = 0

    opt.hw = SimpleNamespace(
        physical_cores=8,
    )
    opt.model = SimpleNamespace()

    space = {
        "threads": ParamDef(
            "threads",
            2,
            4,
            step=2,
        ),
    }

    monkeypatch.setattr(
        "llama_autotune.optimizer.get_search_space",
        lambda *args, **kwargs: space,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.is_plausible",
        lambda *args, **kwargs: True,
    )

    cached_cfg = SearchConfig(
        threads=2,
    )

    opt._cache[
        opt._config_key(cached_cfg)
    ] = _result()

    evaluated = []

    def evaluate(cfg):
        key = opt._config_key(cfg)

        if key in opt._cache:
            raise AssertionError(
                "Stage C attempted a cached benchmark"
            )

        evaluated.append(cfg.threads)

        result = _result()
        opt._cache[key] = result
        opt._total_evals += 1

        return result

    def score(result, cfg):
        return float(cfg.threads)

    opt._evaluate = evaluate
    opt._score = score

    original_create_study = optuna.create_study
    captured = {}

    def create_study(*args, **kwargs):
        study = original_create_study(
            *args,
            **kwargs,
        )

        captured["study"] = study

        return study

    monkeypatch.setattr(
        "llama_autotune.optimizer.optuna.create_study",
        create_study,
    )

    opt._stage_c_bayesian()

    assert evaluated == [4]

    study = captured["study"]

    assert len(
        [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
    ) == 2



def test_stage_c_stops_when_unique_space_is_exhausted(
    monkeypatch,
):
    """Stage C must stop without duplicate benchmarks when space is exhausted."""

    opt = Optimizer.__new__(Optimizer)

    opt.objective = OptimizeObjective.BALANCED
    opt.n_trials_stage_c = 3
    opt._best_config = SearchConfig(
        threads=4,
    )
    opt._best_score = 0.0
    opt._cache = {}
    opt._total_evals = 0

    opt.hw = SimpleNamespace(
        physical_cores=8,
    )
    opt.model = SimpleNamespace()

    space = {
        "threads": ParamDef(
            "threads",
            2,
            4,
            step=2,
        ),
    }

    monkeypatch.setattr(
        "llama_autotune.optimizer.get_search_space",
        lambda *args, **kwargs: space,
    )

    monkeypatch.setattr(
        "llama_autotune.optimizer.is_plausible",
        lambda *args, **kwargs: True,
    )

    evaluated = []

    def evaluate(cfg):
        key = opt._config_key(cfg)

        if key in opt._cache:
            raise AssertionError(
                "Stage C attempted a duplicate benchmark"
            )

        evaluated.append(cfg.threads)

        result = _result()
        opt._cache[key] = result
        opt._total_evals += 1

        return result

    def score(result, cfg):
        return float(cfg.threads)

    opt._evaluate = evaluate
    opt._score = score

    opt._stage_c_bayesian()

    assert len(evaluated) == 2
    assert set(evaluated) == {2, 4}
    assert opt._best_score == 4.0



def test_run_normal_speed_executes_all_stages_in_order():
    """Normal-speed optimisation runs Stage A, B, and C in order."""

    opt = Optimizer.__new__(Optimizer)

    initial = SearchConfig(ctx_size=4096)
    best = SearchConfig(ctx_size=8192)

    opt.model_path = "test.gguf"
    opt.objective = OptimizeObjective.BALANCED
    opt.hw = SimpleNamespace(cpu_name="Test CPU")
    opt.slow = False
    opt._initial_config = initial
    opt._best_config = None
    opt._best_score = 0.0
    opt._total_evals = 0
    opt._speed_tier = "unknown"
    opt._n_prompt = 512
    opt._n_gen = 128
    opt._bench_reps = 3

    calls = []

    def estimate_speed():
        calls.append("estimate")
        opt._speed_tier = "fast"

    def stage_a():
        calls.append("stage_a")
        opt._best_config = best

    def stage_b():
        calls.append("stage_b")

    def stage_c():
        calls.append("stage_c")

    opt._estimate_speed = estimate_speed
    opt._stage_a_baseline = stage_a
    opt._stage_b_local_search = stage_b
    opt._stage_c_bayesian = stage_c

    result = opt.run()

    assert result is best
    assert calls == [
        "estimate",
        "stage_a",
        "stage_b",
        "stage_c",
    ]


def test_run_skips_later_stages_when_stage_a_finds_no_best_config():
    """Stage B and C require a successful Stage A or fallback result."""

    opt = Optimizer.__new__(Optimizer)

    initial = SearchConfig(ctx_size=4096)

    opt.model_path = "test.gguf"
    opt.objective = OptimizeObjective.BALANCED
    opt.hw = SimpleNamespace(cpu_name="Test CPU")
    opt.slow = False
    opt._initial_config = initial
    opt._best_config = None
    opt._best_score = 0.0
    opt._total_evals = 0
    opt._speed_tier = "unknown"
    opt._n_prompt = 512
    opt._n_gen = 128
    opt._bench_reps = 3

    calls = []

    def estimate_speed():
        calls.append("estimate")
        opt._speed_tier = "fast"

    def stage_a():
        calls.append("stage_a")
        # Leave _best_config as None to simulate complete failure.

    def stage_b():
        calls.append("stage_b")

    def stage_c():
        calls.append("stage_c")

    opt._estimate_speed = estimate_speed
    opt._stage_a_baseline = stage_a
    opt._stage_b_local_search = stage_b
    opt._stage_c_bayesian = stage_c

    result = opt.run()

    assert result is initial
    assert calls == [
        "estimate",
        "stage_a",
    ]



def test_run_very_slow_without_slow_mode_returns_initial_config():
    """Very slow hardware skips optimisation unless slow mode is enabled."""
    opt = Optimizer.__new__(Optimizer)

    initial = SearchConfig(ctx_size=4096)

    opt.model_path = "test.gguf"
    opt.objective = OptimizeObjective.BALANCED
    opt.hw = SimpleNamespace(cpu_name="Test CPU")
    opt.slow = False
    opt._initial_config = initial
    opt._best_config = None
    opt._best_score = 0.0
    opt._total_evals = 0
    opt._speed_tier = "unknown"
    opt._n_prompt = 512
    opt._n_gen = 128
    opt._bench_reps = 3

    calls = []

    def estimate_speed():
        calls.append("estimate")
        opt._speed_tier = "very_slow"

    def stage_a():
        calls.append("stage_a")

    def stage_b():
        calls.append("stage_b")

    def stage_c():
        calls.append("stage_c")

    opt._estimate_speed = estimate_speed
    opt._stage_a_baseline = stage_a
    opt._stage_b_local_search = stage_b
    opt._stage_c_bayesian = stage_c

    result = opt.run()

    assert result is initial
    assert calls == ["estimate"]
    assert opt._n_prompt == 64
    assert opt._n_gen == 32
    assert opt._bench_reps == 1


def test_run_very_slow_with_slow_mode_continues_optimization():
    """Slow mode allows optimisation to continue with a reduced workload."""
    opt = Optimizer.__new__(Optimizer)

    initial = SearchConfig(ctx_size=4096)
    best = SearchConfig(ctx_size=8192)

    opt.model_path = "test.gguf"
    opt.objective = OptimizeObjective.BALANCED
    opt.hw = SimpleNamespace(cpu_name="Test CPU")
    opt.slow = True
    opt._initial_config = initial
    opt._best_config = None
    opt._best_score = 0.0
    opt._total_evals = 0
    opt._speed_tier = "unknown"
    opt._n_prompt = 512
    opt._n_gen = 128
    opt._bench_reps = 3

    calls = []

    def estimate_speed():
        calls.append("estimate")
        opt._speed_tier = "very_slow"

    def stage_a():
        calls.append("stage_a")
        opt._best_config = best

    def stage_b():
        calls.append("stage_b")

    def stage_c():
        calls.append("stage_c")

    opt._estimate_speed = estimate_speed
    opt._stage_a_baseline = stage_a
    opt._stage_b_local_search = stage_b
    opt._stage_c_bayesian = stage_c

    result = opt.run()

    assert result is best
    assert calls == [
        "estimate",
        "stage_a",
        "stage_b",
        "stage_c",
    ]
    assert opt._n_prompt == 64
    assert opt._n_gen == 32
    assert opt._bench_reps == 1
