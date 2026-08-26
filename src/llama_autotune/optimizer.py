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
        cfg = SearchConfig(threads=self.hw.physical_cores)
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
            f"{len(candidate_params) + len(cached_keys)} "
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

        def objective_fn(trial: optuna.Trial) -> float:
            nonlocal valid_evals

            params: dict[str, Any] = {}

            for pname, values in local_space.items():
                params[pname] = trial.suggest_categorical(
                    pname,
                    values,
                )

            cfg = config_from_params(
                params,
                stage_c_base,
            )

            logger.info(
                f"[C] TRIAL={trial.number} params={params}"
            )

            key = self._config_key(cfg)

            if key not in candidate_params:
                logger.info(
                    f"[C] PRUNED trial={trial.number} "
                    f"reason=unavailable_candidate "
                    f"valid_evals={valid_evals}/{target_evals}"
                )
                raise optuna.TrialPruned()

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
            f"target_valid_evals={target_evals}"
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=n_startup_trials,
            ),
        )

        # The benchmarkable space is finite and already known. For a real
        # Optuna study, enqueue a deterministic set of unique candidates so
        # categorical sampling cannot spend the Stage C budget on duplicates.
        #
        # Keep the FakeStudy fallback used by the unit tests: it supplies its
        # own FakeTrial values and intentionally exposes only optimize().
        if hasattr(study, "enqueue_trial"):
            ordered_candidates = list(candidate_params.values())
            requested_candidates = min(
                target_evals,
                len(ordered_candidates),
            )

            if requested_candidates == len(ordered_candidates):
                selected_candidates = ordered_candidates
            elif requested_candidates == 1:
                selected_candidates = [ordered_candidates[0]]
            else:
                last_index = len(ordered_candidates) - 1
                selected_indices = [
                    round(
                        index * last_index
                        / (requested_candidates - 1)
                    )
                    for index in range(requested_candidates)
                ]
                selected_candidates = [
                    ordered_candidates[index]
                    for index in selected_indices
                ]

            logger.info(
                f"[C] selected_candidates="
                f"{len(selected_candidates)} "
                f"available_candidates={len(available_keys)}"
            )

            for params in selected_candidates:
                study.enqueue_trial(params)

            study.optimize(
                objective_fn,
                n_trials=len(selected_candidates),
                show_progress_bar=False,
            )
        else:
            while (
                valid_evals < target_evals
                and len(seen_keys) < len(available_keys)
            ):
                study.optimize(
                    objective_fn,
                    n_trials=1,
                    show_progress_bar=False,
                )

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
            f"total_benchmarks={self._total_evals}"
        )

        if valid_evals < target_evals:
            logger.warning(
                f"[C] STOPPED before target: "
                f"{valid_evals}/{target_evals} valid evaluations "
                f"after exhausting benchmarkable candidates"
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
        """Generate strategic context-size candidates around the current value.

        When ``include_current`` is true, the current context is preserved as
        the first candidate. Otherwise, the requested budget is spent only on
        alternative context values. Candidates above the current value are
        preferred and distributed progressively towards the maximum supported
        context. At the upper bound, refinement proceeds downward in aligned
        steps.
        """
        if count <= 0:
            return []

        if current_value is None:
            current_value = param.low

        current = max(
            param.low,
            min(param.high, current_value),
        )

        step = param.step or 1

        current_value_normalized = (
            int(current)
            if isinstance(param.low, int)
            else current
        )

        values: list[Any] = []

        if include_current:
            values.append(current_value_normalized)

        if current >= param.high:
            distance = 1

            while len(values) < count:
                candidate = current - distance * step

                if candidate < param.low:
                    break

                value = (
                    int(candidate)
                    if isinstance(param.low, int)
                    else candidate
                )

                if value not in values:
                    values.append(value)

                distance += 1

            return values[:count]

        remaining = count - len(values)

        if remaining <= 0:
            return values[:count]

        # When the current value is excluded, Stage B should spend the
        # budget on real alternatives starting from the nearest aligned
        # value. Preserve the maximum context as the final candidate when
        # more than one alternative is requested.
        if not include_current:
            candidate = current + step

            while (
                len(values) < count - 1
                and candidate < param.high
            ):
                value = (
                    int(candidate)
                    if isinstance(param.low, int)
                    else candidate
                )

                if value not in values:
                    values.append(value)

                candidate += step

            if (
                len(values) < count
                and param.high not in values
            ):
                values.append(
                    int(param.high)
                    if isinstance(param.low, int)
                    else param.high
                )

            return values[:count]

        distance = param.high - current

        for index in range(1, remaining + 1):
            target = current + (
                distance * index / remaining
            )

            offset = target - param.low

            aligned = (
                param.low
                + ((offset + step - 1) // step) * step
            )

            if aligned > param.high:
                aligned = param.high

            value = (
                int(aligned)
                if isinstance(param.low, int)
                else aligned
            )

            if (
                value != current_value_normalized
                and value not in values
            ):
                values.append(value)

        candidate = current + step

        while (
            len(values) < count
            and candidate <= param.high
        ):
            value = (
                int(candidate)
                if isinstance(param.low, int)
                else candidate
            )

            if (
                value != current_value_normalized
                and value not in values
            ):
                values.append(value)

            candidate += step

        return values[:count]


    def _local_values(
        self,
        param: ParamDef,
        current_value: Any,
        count: int,
    ) -> list[Any]:
        """Generate values locally around the current parameter value.

        Numeric values are explored in alternating order around the current
        value: one step below, one step above, then progressively farther
        away. Values are kept within the parameter bounds and duplicates are
        avoided. If the current value lies outside the search-space bounds,
        it is clamped only as the center for local candidate generation.
        """
        if count <= 0:
            return []

        if param.is_categorical and param.categories:
            if current_value in param.categories:
                start = param.categories.index(current_value)
                values = []
                distance = 1

                while (
                    len(values) < count
                    and (
                        start - distance >= 0
                        or start + distance < len(param.categories)
                    )
                ):
                    for index in (
                        start - distance,
                        start + distance,
                    ):
                        if (
                            0 <= index < len(param.categories)
                            and len(values) < count
                        ):
                            values.append(param.categories[index])
                    distance += 1

                return values

            return param.categories[:count]

        if current_value is None:
            current_value = param.low

        above_high = current_value > param.high

        current = max(
            param.low,
            min(param.high, current_value),
        )

        # A value above the search-space maximum can represent a semantic
        # "maximum" setting, such as n_gpu_layers=999 for full offload.
        # For n_gpu_layers, treat both that representation and the explicit
        # numeric maximum as refinement around the full-offload boundary.
        if (
            param.name == "n_gpu_layers"
            and current >= param.high
        ):
            step = 1
        else:
            step = param.step or max(
                1,
                (param.high - param.low) // max(count, 1),
            )

        values = []

        if above_high and count > 0:
            values.append(
                int(current)
                if isinstance(param.low, int)
                else current
            )

        distance = 1

        while len(values) < count:
            added = False

            for candidate in (
                current - distance * step,
                current + distance * step,
            ):
                if (
                    param.low <= candidate <= param.high
                    and candidate not in values
                    and candidate != current
                    and len(values) < count
                ):
                    values.append(
                        int(candidate)
                        if isinstance(param.low, int)
                        else candidate
                    )
                    added = True

            if (
                current - distance * step < param.low
                and current + distance * step > param.high
            ):
                break

            if not added and (
                current - distance * step < param.low
                and current + distance * step > param.high
            ):
                break

            distance += 1

        return values

    def _stage_b_ngl_values(
        self,
        param: ParamDef,
        current_value: Any,
        count: int,
        include_current: bool = True,
    ) -> list[Any]:
        """Generate Stage B candidates for GPU layer offload.

        Optionally preserves a semantic full-offload value from the current
        configuration when it exceeds the numeric model layer range, then
        fills the remaining allocation with the highest concrete layer counts.
        """
        if count <= 0:
            return []

        values: list[Any] = []

        if (
            include_current
            and current_value is not None
            and current_value > param.high
        ):
            values.append(current_value)

        candidate = int(param.high)

        while len(values) < count and candidate >= int(param.low):
            if candidate not in values:
                values.append(candidate)
            candidate -= 1

        return values

    def _stage_b_spread_values(
        self,
        param: ParamDef,
        count: int,
    ) -> list[Any]:
        """Generate values spread across the full parameter range.

        Unlike ``_grid_values``, this method uses the available Stage B
        allocation to cover the low and high bounds of the parameter range,
        with evenly distributed intermediate values when more candidates are
        requested.
        """
        if count <= 0:
            return []

        if param.is_categorical and param.categories:
            if count == 1:
                return [param.categories[0]]

            if count >= len(param.categories):
                return param.categories[:]

            indices = [
                round(
                    index * (len(param.categories) - 1)
                    / (count - 1)
                )
                for index in range(count)
            ]

            return [
                param.categories[index]
                for index in indices
            ]

        if count == 1:
            return [
                int(param.low)
                if isinstance(param.low, int)
                else param.low
            ]

        if param.step:
            total_steps = int(
                (param.high - param.low) // param.step
            )

            if count > total_steps + 1:
                count = total_steps + 1

            indices = [
                round(
                    index * total_steps / (count - 1)
                )
                for index in range(count)
            ]

            values = [
                param.low + index * param.step
                for index in indices
            ]

            return [
                int(value)
                if isinstance(param.low, int)
                else value
                for value in values
            ]

        values = []

        for index in range(count):
            value = (
                param.low
                + (param.high - param.low)
                * index
                / (count - 1)
            )

            if isinstance(param.low, int):
                value = int(value)

            if value not in values:
                values.append(value)

        return values

    def _grid_values(self, param: ParamDef, count: int) -> list[Any]:
        """Generate up to ``count`` evenly-spaced values for a parameter.

        For categorical parameters the first ``count`` categories are
        returned.  For numeric parameters with a step size the values are
        produced by repeated addition; otherwise a uniform step is computed.

        Args:
            param: The parameter definition.
            count: Maximum number of values to return.

        Returns:
            A list of parameter values to evaluate.
        """
        if param.is_categorical and param.categories:
            return param.categories[:count]
        if param.step:
            values = []
            v = param.low
            while v <= param.high and len(values) < count:
                values.append(int(v) if isinstance(param.low, int) else v)
                v += param.step
            return values
        step = max(1, (param.high - param.low) // (count - 1))
        return list(range(param.low, param.high + 1, step))

    def _sample_param(
        self, trial: optuna.Trial, name: str, pdef: ParamDef
    ) -> Any:
        """Sample a parameter value for an Optuna trial.

        Delegates to the appropriate ``suggest_*`` method based on whether the
        parameter is categorical or integer-valued.

        Args:
            trial: The Optuna trial object.
            name: The parameter name.
            pdef: The parameter definition from the search space.

        Returns:
            A sampled value for the parameter.
        """
        if pdef.is_categorical and pdef.categories:
            return trial.suggest_categorical(name, pdef.categories)
        if pdef.step:
            return trial.suggest_int(name, pdef.low, pdef.high, step=int(pdef.step))
        return trial.suggest_int(name, pdef.low, pdef.high)

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
