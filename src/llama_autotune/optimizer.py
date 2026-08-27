"""Three-stage LLM inference parameter optimizer.

Stage A — Rule-based baseline: evaluate the heuristic initial config; if it
fails, try simple thread-count fallbacks.

Stage B — Local grid search: sweep over the primary parameters (threads,
batch_size, ubatch_size, n_gpu_layers) one at a time, keeping the best so far.

Stage C — Bayesian optimisation: use Optuna's TPE sampler to explore the full
search space around the best config found by the earlier stages.
"""

from __future__ import annotations

import itertools

import logging
import math
import time
from typing import Any

import optuna

from .benchmark import run_benchmark
from .constraints import detect_oom_in_output, is_plausible
from .hardware import detect_hardware
from .heuristics import generate_initial_config, to_cpu_config
from .model_inspector import inspect_model
from .models import (
    Backend,
    BenchmarkResult,
    HardwareInfo,
    ModelInfo,
    OptimizeObjective,
    SearchConfig,
)
from .search_space import ParamDef, config_from_params, get_search_space
from . import candidates

logger = logging.getLogger(__name__)

class Optimizer:
    """Three-stage parameter optimizer for llama.cpp.

    Sequences a heuristic baseline (stage A), a local grid search (stage B),
    and a Bayesian optimisation pass (stage C) to find the best inference
    parameters for a given model and hardware combination.
    """

    def __init__(
        self,
        model_path: str,
        objective: OptimizeObjective = OptimizeObjective.BALANCED,
        n_trials_stage_b: int = 12,
        n_trials_stage_c: int = 20,
        llama_dir: str | None = None,
        slow: bool = False,
        cache: dict[str, BenchmarkResult] | None = None,
    ):
        """Initialise the optimizer.

        Args:
            model_path: Path to the GGUF model file.
            objective: Optimisation objective (e.g. balanced, max throughput).
            n_trials_stage_b: Maximum evaluations for the grid-search stage.
            n_trials_stage_c: Maximum evaluations for the Bayesian stage.
            llama_dir: Optional path to a custom llama.cpp directory. When set,
                the ``LLAMA_CPP_DIR`` environment variable is updated.
            slow: If True, fallback configs are tried even when the baseline is
                too slow for the host machine.
            cache: Optional pre-populated benchmark cache (config JSON ->
                result). Used to resume a previous search without repeating
                benchmarks.
        """
        self.model_path = model_path
        self.objective = objective
        self.n_trials_stage_b = n_trials_stage_b
        self.n_trials_stage_c = n_trials_stage_c
        self.slow = slow

        if llama_dir:
            import os

            os.environ["LLAMA_CPP_DIR"] = llama_dir

        self.hw: HardwareInfo = detect_hardware()
        self.model: ModelInfo = inspect_model(model_path)
        self._initial_config: SearchConfig = generate_initial_config(self.hw, self.model)
        self._best_config: SearchConfig | None = None
        self._best_score: float = 0.0
        self._total_evals: int = 0
        self._cache: dict[str, BenchmarkResult] = {}
        if cache:
            self._cache.update(cache)
        self._baseline_result: BenchmarkResult | None = None

        self._speed_tier: str = "unknown"
        self._speed_estimate: float = 0.0
        self._n_prompt: int = 512
        self._n_gen: int = 128
        self._bench_reps: int = 3

    def _benchmark_prompt_tokens(self, cfg: SearchConfig) -> int:
        """Return a context-aware prompt workload for a configuration.

        The hardware speed tier defines the baseline workload.  For larger
        contexts, scale the prompt up to one eighth of ``ctx_size`` while
        keeping a practical upper bound.
        """
        ctx_size = cfg.ctx_size
        if ctx_size is None or ctx_size <= 0:
            return self._n_prompt

        return max(self._n_prompt, min(ctx_size // 8, 3072))

    def _estimate_speed(self) -> None:
        """Run a minimal benchmark to determine hardware speed tier.

        Uses a small prompt (16 tokens) and short generation (8 tokens)
        with a single repetition, no warmup, and a 30-second timeout.
        Based on the measured generation throughput the method sets
        ``_speed_tier``, ``_n_prompt``, ``_n_gen``, and ``_bench_reps``
        so that subsequent evaluations are scaled to the hardware.
        """
        cfg = self._initial_config
        result = run_benchmark(self.model_path, cfg,
                               repetitions=1, timeout=30,
                               n_prompt=16, n_gen=8, no_warmup=True)
        if not result.success:
            self._speed_tier = "very_slow"
            logger.info("Speed probe failed — tier: very_slow")
            return

        tps = result.generation_tps
        if tps < 1:
            self._speed_tier = "very_slow"
        elif tps < 4:
            self._speed_tier = "slow"
            self._n_prompt, self._n_gen, self._bench_reps = 64, 32, 1
        elif tps < 15:
            self._speed_tier = "medium"
            self._n_prompt, self._n_gen, self._bench_reps = 256, 64, 2
        else:
            self._speed_tier = "fast"
        self._speed_estimate = tps
        logger.info(f"Speed tier: {self._speed_tier} (gen_tps={round(tps, 2)})")

    def run(self) -> SearchConfig:
        """Run all three optimisation stages and return the best config.

        Stages are executed sequentially: baseline (A), then local search (B),
        then Bayesian (C). If stage A fails completely and no fallback works,
        the initial heuristic config is returned.

        Returns:
            The best SearchConfig found, or the initial heuristic config if
            no successful evaluation was produced.
        """
        logger.info(f"Starting optimization — model={self.model_path} objective={self.objective.value} hw={self.hw.cpu_name}")

        self._estimate_speed()
        if self._speed_tier == "very_slow":
            self._n_prompt, self._n_gen, self._bench_reps = 64, 32, 1

            if not self.slow:
                logger.warning(
                    "Hardware too slow for benchmarking — using heuristic"
                )
                return self._initial_config

            logger.warning(
                "Hardware too slow for normal benchmarking — "
                "continuing in slow mode"
            )

        self._stage_a_baseline()
        if self._best_config is not None:
            self._stage_b_local_search()

        if self._best_config is not None:
            self._stage_c_bayesian()

        logger.info(f"Optimization complete — best_score={self._best_score} total_evals={self._total_evals}")
        return self._best_config or self._initial_config

    def _stage_a_baseline(self) -> None:
        """Evaluate the rule-based initial config.

        If the config succeeds its score is recorded.  On failure the method
        falls through to the fallback configs.
        """
        logger.info("========== STAGE A: BASELINE ==========")
        logger.info(f"[A] config={self._initial_config}")
        result = self._evaluate(self._initial_config)
        logger.info(
        f"[A] result success={result.success} "
        f"gen_tps={result.generation_tps} "
        f"prompt_tps={result.prompt_tps} "
        f"total_evals={self._total_evals}"
        )			
        if result.success:
            self._baseline_result = result
            self._best_config = self._initial_config
            self._best_score = self._score(result, self._initial_config)
            logger.info(f"Baseline score={self._best_score} gen_tps={result.generation_tps} prompt_tps={result.prompt_tps}")
        else:
            logger.warning("Baseline config failed, trying fallbacks")
            self._try_fallback_configs()

    def _try_fallback_configs(self) -> None:
        """Iterate generated fallback configs and use the first one that works.

        Once a fallback succeeds it becomes the new best config and iteration
        stops.
        """
        for cfg in self._generate_fallbacks():
            result = self._evaluate(cfg)
            if result.success:
                self._baseline_result = result
                self._best_config = cfg
                self._best_score = self._score(result, cfg)
                logger.info(f"Fallback worked (score={self._best_score})")
                return

    def _generate_fallbacks(self) -> list[SearchConfig]:
        """Build a list of fallback configurations.

        Produces configs with varying thread counts (physical cores, logical
        cores, half physical cores), first for the initial heuristic config
        and then for a CPU-only variant.

        Returns:
            A list of fallback SearchConfig objects.
        """
        fallbacks = []
        threads_opts = sorted({
            self.hw.physical_cores,
            self.hw.logical_cores,
            max(1, self.hw.physical_cores // 2),
        }, reverse=True)
        for t in threads_opts:
            cfg = self._initial_config.model_copy()
            cfg.threads = t
            fallbacks.append(cfg)
        cpu = to_cpu_config(self._initial_config)
        for t in threads_opts:
            cfg = cpu.model_copy()
            cfg.threads = t
            fallbacks.append(cfg)
        return fallbacks

    def _stage_b_local_search(self) -> None:
        """Local search with the benchmark budget distributed across parameters.

        The available Stage B budget is shared as evenly as possible among the
        primary parameters that actually exist in the current search space.
        Each parameter is explored around the best configuration found so far.
        """
        logger.info("========== STAGE B: LOCAL SEARCH ==========")

        budget = self.n_trials_stage_b

        if budget <= 0:
            logger.info("[B] SKIP budget=0")
            return

        space = get_search_space(
            self.hw,
            self.model,
            self.objective,
        )

        primary_params = [
            name
            for name in (
                "threads",
                "batch_size",
                "ubatch_size",
                "n_gpu_layers",
                "ctx_size",
            )
            if name in space
        ]

        if not primary_params:
            logger.info("[B] SKIP no primary parameters in search space")
            return

        base, extra = divmod(
            budget,
            len(primary_params),
        )

        allocations = {
            name: base + (
                1 if index < extra else 0
            )
            for index, name in enumerate(primary_params)
        }

        logger.info(
            f"[B] budget={budget} "
            f"parameters={primary_params} "
            f"allocations={allocations}"
        )

        logical_evals = 0

        for param_name in primary_params:
            allocation = allocations[param_name]

            if allocation <= 0:
                logger.info(
                    f"[B] SKIP parameter={param_name} "
                    f"reason=no_budget"
                )
                continue

            param = space[param_name]

            current_config = (
                self._best_config
                if self._best_config
                else SearchConfig()
            )
            current_value = getattr(
                current_config,
                param_name,
                None,
            )

            if param_name == "ctx_size":
                values = self._strategic_ctx_values(
                    param,
                    current_value,
                    count=allocation,
                    include_current=False,
                )
            elif param_name in {
                "batch_size",
                "ubatch_size",
            }:
                values = self._local_values(
                    param,
                    current_value,
                    count=allocation,
                )
            elif param_name == "n_gpu_layers":
                values = self._stage_b_ngl_values(
                    param,
                    current_value,
                    count=allocation,
                    include_current=False,
                )
            else:
                values = self._local_values(
                    param,
                    current_value,
                    count=allocation,
                )

            if not values:
                logger.info(
                    f"[B] SKIP parameter={param_name} "
                    f"reason=no_candidates"
                )
                continue

            logger.info(
                f"[B] PARAM={param_name} "
                f"allocation={allocation} "
                f"values={values}"
            )

            for val in values:
                if logical_evals >= budget:
                    break

                cfg = (
                    self._best_config.model_copy()
                    if self._best_config
                    else SearchConfig()
                )

                setattr(
                    cfg,
                    param_name,
                    val,
                )

                if not is_plausible(
                    cfg,
                    self.model,
                    self.hw,
                ):
                    logger.info(
                        f"[B] SKIP parameter={param_name} "
                        f"value={val} "
                        f"reason=not_plausible"
                    )
                    continue

                key = self._config_key(cfg)

                if key in self._cache:
                    logger.info(
                        f"[B] SKIP parameter={param_name} "
                        f"value={val} "
                        f"reason=cache_hit "
                        f"logical_evals={logical_evals}/{budget}"
                    )
                    continue

                logger.info(
                    f"[B] EVALUATE parameter={param_name} "
                    f"value={val} "
                    f"logical_evals={logical_evals}/{budget}"
                )

                result = self._evaluate(cfg)

                if result.success:
                    score = self._score(
                        result,
                        cfg,
                    )

                    logger.info(
                        f"[B] RESULT parameter={param_name} "
                        f"value={val} "
                        f"score={score} "
                        f"best={self._best_score}"
                    )

                    if score > self._best_score:
                        self._best_config = cfg
                        self._best_score = score

                        logger.info(
                            f"[B] NEW BEST "
                            f"parameter={param_name} "
                            f"value={val} "
                            f"score={score}"
                        )
                else:
                    logger.info(
                        f"[B] FAILED parameter={param_name} "
                        f"value={val}"
                    )

                # Stage B budget counts new valid benchmark evaluations.
                logical_evals += 1

        logger.info(
            f"[B] COMPLETE "
            f"logical_evals={logical_evals} "
            f"requested={budget} "
            f"total_benchmarks={self._total_evals}"
        )

    def _stage_c_bayesian(self) -> None:
        """Bayesian refinement around the best configuration.

        Stage C builds a small categorical search space around the best
        configuration produced by Stages A and B. Most parameters are sampled
        from nearby values generated by ``_local_values``. ``ctx_size`` uses
        progressively larger candidates generated by
        ``_strategic_ctx_values``.
        """
        logger.info("========== STAGE C: BAYESIAN LOCAL REFINEMENT ==========")

        target_evals = self.n_trials_stage_c

        if target_evals <= 0:
            logger.info("[C] SKIP target_evals=0")
            return

        if self._best_config is None:
            logger.warning("[C] SKIP no best configuration available")
            return

        # Freeze the Stage C starting point. Trial configurations must
        # always be constructed from the same base configuration; otherwise
        # later trials would inherit unrelated changes from earlier bests.
        stage_c_base = self._best_config.model_copy()

        space = get_search_space(
            self.hw,
            self.model,
            self.objective,
        )

        local_space: dict[str, list[Any]] = {}

        for pname, pdef in space.items():
            current_value = getattr(
                stage_c_base,
                pname,
                None,
            )

            if pname == "ctx_size":
                values = self._strategic_ctx_values(
                    pdef,
                    current_value,
                    4,
                )
            else:
                values = self._local_values(
                    pdef,
                    current_value,
                    4,
                )

            if current_value is not None:
                values = [
                    current_value,
                    *values,
                ]

            # Keep order while removing duplicates.
            values = list(dict.fromkeys(values))

            if values:
                local_space[pname] = values

        logger.info(
            f"[C] target_valid_evals={target_evals} "
            f"local_space={local_space}"
        )

        if not local_space:
            logger.info("[C] SKIP no local candidates")
            return

        # Enumerate the real finite candidate space. The Cartesian-product
        # size alone is only a theoretical upper bound because different
        # parameter combinations can map to the same SearchConfig, some
        # configurations can be implausible, and some can already be cached.
        candidate_params: dict[str, dict[str, Any]] = {}
        cached_keys: set[str] = set()

        names = list(local_space)

        for values in itertools.product(
            *(local_space[name] for name in names)
        ):
            params = dict(zip(names, values))

            cfg = config_from_params(
                params,
                stage_c_base,
            )

            if not is_plausible(
                cfg,
                self.model,
                self.hw,
            ):
                continue

            key = self._config_key(cfg)

            if key in candidate_params:
                continue

            candidate_params[key] = params

            if key in self._cache:
                cached_keys.add(key)

        available_keys = (
            set(candidate_params)
            - cached_keys
        )

        logger.info(
            f"[C] candidate_space="
            f"{len(candidate_params)} "
            f"benchmarkable_candidates={len(available_keys)} "
            f"cached_candidates={len(cached_keys)}"
        )

        if not available_keys:
            logger.info(
                "[C] STOP no benchmarkable candidates available"
            )
            return

        seen_keys: set[str] = set()
        valid_evals = 0

        # TPE samples an index into the enumerated *plausible* candidates.
        # Sampling the raw Cartesian product wasted trial slots on
        # implausible combinations (pruned as "unavailable"), so the
        # surrogate only ever saw the handful of plausible completions.
        ordered_candidates = list(candidate_params.values())

        def objective_fn(trial: optuna.Trial) -> float:
            nonlocal valid_evals

            index = trial.suggest_int(
                "candidate_index",
                0,
                len(ordered_candidates) - 1,
            )

            params = ordered_candidates[index]

            cfg = config_from_params(
                params,
                stage_c_base,
            )

            logger.info(
                f"[C] TRIAL={trial.number} params={params}"
            )

            key = self._config_key(cfg)

            if key in seen_keys:
                logger.info(
                    f"[C] PRUNED trial={trial.number} "
                    f"reason=duplicate_candidate "
                    f"valid_evals={valid_evals}/{target_evals}"
                )
                raise optuna.TrialPruned()

            seen_keys.add(key)

            cache_hit = key in cached_keys

            if cache_hit:
                result = self._cache[key]
            else:
                logger.info(
                    f"[C] EVALUATE trial={trial.number} "
                    f"cache_hit=False "
                    f"valid_evals={valid_evals}/{target_evals}"
                )

                result = self._evaluate(cfg)

                if result.success:
                    valid_evals += 1

            logger.info(
                f"[C] RESULT trial={trial.number} "
                f"success={result.success} "
                f"cache_hit={cache_hit} "
                f"valid_evals={valid_evals}/{target_evals} "
                f"total_evals={self._total_evals}"
            )

            if not result.success:
                logger.info(
                    f"[C] PRUNED trial={trial.number} "
                    f"reason=benchmark_failed"
                )
                raise optuna.TrialPruned()

            score = self._score(result, cfg)

            logger.info(
                f"[C] SCORE trial={trial.number} "
                f"score={score} "
                f"current_best={self._best_score}"
            )

            if score > self._best_score:
                self._best_config = cfg
                self._best_score = score

                logger.info(
                    f"[C] NEW BEST trial={trial.number} "
                    f"score={score} "
                    f"params={params}"
                )

            return score

        n_startup_trials = min(
            10,
            max(1, target_evals // 2),
        )

        logger.info(
            f"[C] TPE n_startup_trials={n_startup_trials} "
            f"target_valid_evals={target_evals} "
            f"available_candidates={len(available_keys)}"
        )

        # Suppress Optuna's own per-trial logging (its "Best is trial N"
        # lines refer to the local study only, which is confusing next to
        # the global best). Restored once the search finishes.
        original_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        try:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=42,
                    n_startup_trials=n_startup_trials,
                ),
            )

            # TPE drives the search instead of a deterministic enqueue.
            # ``objective_fn`` prunes duplicates and failed benchmarks, so
            # the loop continues until the valid-evaluation budget is spent
            # or the finite candidate space is exhausted. This single path
            # also serves the FakeStudy used by the unit tests (it only
            # exposes ``optimize``).
            patience = max(3, target_evals // 4)
            no_improvement_streak = 0

            while (
                valid_evals < target_evals
                and len(seen_keys) < len(ordered_candidates)
            ):
                best_before = self._best_score
                trials_before = len(study.trials)

                study.optimize(
                    objective_fn,
                    n_trials=1,
                    show_progress_bar=False,
                )

                if len(study.trials) == trials_before:
                    # Safety net: a sampler that produces no trial would
                    # otherwise loop forever.
                    logger.warning(
                        "[C] STOP sampler produced no trial"
                    )
                    break

                last_trial = study.trials[-1]

                if (
                    last_trial.state
                    == optuna.trial.TrialState.COMPLETE
                ):
                    if self._best_score > best_before:
                        no_improvement_streak = 0
                    else:
                        no_improvement_streak += 1

                    if (
                        valid_evals >= n_startup_trials
                        and no_improvement_streak >= patience
                    ):
                        logger.info(
                            f"[C] EARLY STOP no new global best for "
                            f"{no_improvement_streak} completed trials "
                            f"(patience={patience})"
                        )
                        break
        finally:
            optuna.logging.set_verbosity(original_verbosity)

        completed = sum(
            1
            for trial in study.trials
            if trial.state
            == optuna.trial.TrialState.COMPLETE
        )

        pruned = sum(
            1
            for trial in study.trials
            if trial.state
            == optuna.trial.TrialState.PRUNED
        )

        logger.info(
            f"[C] COMPLETE "
            f"target_valid_evals={target_evals} "
            f"actual_valid_evals={valid_evals} "
            f"total_trials={len(study.trials)} "
            f"completed={completed} "
            f"pruned={pruned} "
            f"remaining_candidates="
            f"{len(available_keys - seen_keys)} "
            f"total_benchmarks={self._total_evals} "
            f"best_score={self._best_score}"
        )

        if valid_evals < target_evals:
            if len(seen_keys) >= len(ordered_candidates):
                reason = "after exhausting benchmarkable candidates"
            else:
                reason = "after early stop"

            logger.warning(
                f"[C] STOPPED before target: "
                f"{valid_evals}/{target_evals} valid evaluations "
                f"{reason}"
            )

    def _evaluate(self, config: SearchConfig) -> BenchmarkResult:
        """Run a benchmark for the given config, caching the result."""

        key = self._config_key(config)

        if key in self._cache:
            logger.info(
                f"[EVAL] CACHE HIT key={key} "
                f"total_evals={self._total_evals}"
            )
            return self._cache[key]

        logger.info(
            f"[EVAL] RUN key={key} "
            f"next_total_eval={self._total_evals + 1} "
            f"config={config}"
        )

        self._total_evals += 1

        n_prompt = self._benchmark_prompt_tokens(config)

        result = run_benchmark(
            self.model_path,
            config,
            timeout=900,
            n_prompt=n_prompt,
            n_gen=self._n_gen,
            repetitions=self._bench_reps,
        )

        self._cache[key] = result

        logger.info(
            f"[EVAL] DONE key={key} "
            f"success={result.success} "
            f"total_evals={self._total_evals}"
        )

        if detect_oom_in_output(result.raw_output):
            result.success = False
            logger.warning(f"OOM detected — {key}")

        return result

    def _score(self, result: BenchmarkResult, config: SearchConfig) -> float:
        """Compute a scalar score from a benchmark result based on the objective.

        For ``MAX_CONTEXT`` the score is dominated by the configured context
        size (feasibility is enforced by the constraint engine and benchmark
        success), with generation throughput as a tiebreaker.

        Args:
            result: The benchmark result to score.
            config: The configuration that produced the result.

        Returns:
            A numerical score (higher is better).
        """
        if self.objective == OptimizeObjective.MAX_GENERATION_TPS:
            return result.generation_tps

        elif self.objective == OptimizeObjective.MAX_PROMPT_TPS:
            return result.prompt_tps

        elif self.objective == OptimizeObjective.MIN_LATENCY:
            return -result.startup_time

        elif self.objective == OptimizeObjective.MAX_EFFICIENCY:
            return result.generation_tps / max(result.memory_usage, 1)

        elif self.objective == OptimizeObjective.MAX_CONTEXT:
            return float(config.ctx_size or 0) + result.generation_tps / 1000.0

        elif self.objective == OptimizeObjective.BALANCED_CONTEXT:
            baseline = getattr(self, "_baseline_result", None)

            if baseline is None:
                return result.generation_tps

            generation_ratio = (
                result.generation_tps
                / max(baseline.generation_tps, 1e-9)
            )

            baseline_ctx = self._initial_config.ctx_size or 1
            ctx_ratio = (
                (config.ctx_size or 0)
                / max(baseline_ctx, 1)
            )

            return math.pow(generation_ratio, 0.60) * math.pow(
                ctx_ratio,
                0.40,
            )

        elif self.objective == OptimizeObjective.BALANCED:
            baseline = getattr(self, "_baseline_result", None)

            if baseline is None:
                return result.generation_tps

            generation_ratio = (
                result.generation_tps
                / max(baseline.generation_tps, 1e-9)
            )

            prompt_ratio = (
                result.prompt_tps
                / max(baseline.prompt_tps, 1e-9)
            )

            return math.pow(generation_ratio, 0.60) * math.pow(
                prompt_ratio,
                0.40,
            )

        else:
            return result.generation_tps

    def _strategic_ctx_values(
        self,
        param: ParamDef,
        current_value: Any,
        count: int,
        include_current: bool = True,
    ) -> list[Any]:
        return candidates.strategic_ctx_values(
            param,
            current_value,
            count,
            include_current=include_current,
        )

    def _local_values(
        self,
        param: ParamDef,
        current_value: Any,
        count: int,
    ) -> list[Any]:
        return candidates.local_values(
            param,
            current_value,
            count,
        )

    def _stage_b_ngl_values(
        self,
        param: ParamDef,
        current_value: Any,
        count: int,
        include_current: bool = True,
    ) -> list[Any]:
        return candidates.stage_b_ngl_values(
            param,
            current_value,
            count,
            include_current=include_current,
        )

    def _stage_b_spread_values(
        self,
        param: ParamDef,
        count: int,
    ) -> list[Any]:
        return candidates.stage_b_spread_values(param, count)

    def _grid_values(self, param: ParamDef, count: int) -> list[Any]:
        return candidates.grid_values(param, count)

    def _sample_param(
        self, trial: optuna.Trial, name: str, pdef: ParamDef
    ) -> Any:
        return candidates.sample_param(trial, name, pdef)

    def _config_key(self, config: SearchConfig) -> str:
        """Return a deterministic cache key for a config.

        Args:
            config: The search configuration.

        Returns:
            A JSON string uniquely representing the config.
        """
        return config.model_dump_json()

    @property
    def best_config(self) -> SearchConfig | None:
        """Return the best configuration found so far.

        Returns:
            The best SearchConfig, or None if no successful evaluation has
            been performed.
        """
        return self._best_config

    @property
    def best_score(self) -> float:
        """Return the score of the best configuration.

        Returns:
            The score associated with ``best_config`` (0.0 if none).
        """
        return self._best_score

    @property
    def total_evals(self) -> int:
        """Return the total number of benchmark evaluations performed.

        Returns:
            The cumulative evaluation count across all stages.
        """
        return self._total_evals
