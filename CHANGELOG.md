# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-08-27

### Fixed

- Stage C early-stopping heuristic was wrong: it compared the improvement
  between two *consecutive* trials against a 2% threshold and would have
  stopped the Bayesian stage after roughly two trials. Replaced with a
  principled rule that stops only after a streak of completed trials without
  a new global best (after the TPE startup phase).
- Stage C was not actually Bayesian: `study.enqueue_trial` fed a
  deterministic, evenly-spaced candidate list to Optuna, bypassing the TPE
  surrogate entirely. Stage C now lets TPE sample the categorical local space
  directly, pruning duplicates, implausible candidates and failed benchmarks
  inside the objective.
- SQLite sessions created by the CLI were never closed (leaking the
  connection pool). Added a `session_scope` context manager and used it in the
  `benchmark` and `search` commands.
- The VRAM estimate was too conservative for K-quants: the GQA-aware KV-cache
  estimate made full offload + large context on a 16 GB GPU estimate over the
  plausibility threshold, so the search could never explore the known-good
  baseline neighbourhood. `Q5` overhead is now calibrated to 1.05 from a real
  `nvidia-smi` measurement instead of the 1.15 default.

### Added

- Real GPU VRAM measurement during benchmarks: `run_benchmark` samples
  `nvidia-smi` / `rocm-smi` alongside the existing CPU RSS tracking and
  exposes the peak in `BenchmarkResult.vram_usage`. Real-hardware VRAM
  realism tests guard that the benchmark reports VRAM and that the heuristic
  initial config is plausible.

### Changed

- Optuna's per-trial logging ("Best is trial N …") is suppressed during
  Stage C and the global best score is logged explicitly, avoiding confusion
  between the local Optuna study and the global search result.

### Added

- Unit tests for the byte-level GGUF readers and synthetic-GGUF
  `inspect_model` coverage (`model_inspector` 38% → 96%), plus tests for
  `candidates.grid_values` / `sample_param`.
- Tests for the subprocess-based hardware detectors (lscpu, nvidia-smi,
  rocm-smi, system_profiler, vulkaninfo, `llama-bench --list-devices`), the
  `inspect`/`benchmark`/`launch`/`export`/`import` CLI commands,
  `run_benchmark`/`_watch_process`, and the dashboard/inspect/benchmark web
  endpoints (overall coverage 61% → 82%).

## [0.4.0] - 2026-08-26

### Added

- GQA/MQA-aware VRAM estimation: the KV-cache estimate now reads
  `attention.head_count_kv` from the GGUF header, derives the per-head
  dimension from the embedding length, honours `cache_type_k` / `cache_type_v`,
  and scales the cache by the GPU offload fraction (respecting
  `no_kv_offload`).
- VRAM detection for AMD on Linux (`rocm-smi --showmeminfo vram`), macOS
  (`system_profiler` with an Apple Silicon unified-memory fallback), and
  Windows (nvidia-smi repair of the unreliable WMI `AdapterRAM` value).
- `--resume` option for `search`: previously benchmarked configurations are
  reloaded from a new SQLite `trial_cache` table so an interrupted search can
  continue without repeating benchmarks.
- Live optimization progress in the Web UI via Server-Sent Events; the
  `/api/optimize` endpoint now runs the search in a background job and streams
  log output to `/api/optimize/events/{job_id}`.
- `candidates.py` module with the pure candidate-generation helpers, extracted
  from `optimizer.py` for isolated unit testing.
- `pytest-cov` coverage configuration and end-to-end smoke tests that run real
  `llama-bench` workloads when a small model is available.
- A roadmap section in the README tracking the P0/P1/P2 work items.

### Changed

- `n_gpu_layers` search range is now bounded by the number of layers that
  plausibly fit in available VRAM instead of a fixed cap of 200 layers.
- `batch_size` and `ubatch_size` are now categorical (powers of two) instead of
  fixed-step numeric ranges.
- `config_from_params` builds the config with a single
  `model_copy(update=...)` validation instead of re-validating once per key.
- `_verify_gpu_backend` now parses the `Available devices:` section of
  `llama-bench --list-devices` instead of relying on a brittle substring
  heuristic.
- The `llama-bench` not-found error now suggests setting `LLAMA_CPP_DIR`.

### Fixed

- Stage B no longer skips `ctx_size` exploration and Stage C re-enqueues cached
  candidates so cache-only spaces are exhausted correctly.
- A duplicate `study.optimize` call in the Stage C fallback loop that ran two
  trials per iteration.

## [0.3.0] - 2026-07-03

### Fixed

- The profile import command is now exposed as `llama-autotune import` (it was
  accidentally registered as `import-cmd`, contradicting the documentation).
- Binary resolution is now truly cross-platform: `llama-bench` / `llama-server`
  are resolved without the `.exe` suffix on Linux and macOS, and the system
  `PATH` is searched as a final fallback.
- `min_latency` optimization no longer lets failed configurations outrank
  working ones. Failed and implausible trials are pruned from the Bayesian
  stage instead of receiving a `-1.0` sentinel score that beat legitimate
  negative latency scores.
- `launch` on Windows now runs `llama-server` as a child process instead of
  `os.execvp`, which does not quote arguments correctly on Windows (paths
  containing spaces would break).

### Added

- Real memory measurement: benchmark runs now record the peak RAM of the
  llama-bench process (via psutil), since llama-bench does not report memory
  in its JSON output. This makes the `max_efficiency` objective meaningful —
  previously memory was always 0 and the objective degraded to generation
  speed.
- The `max_context` objective is now implemented: the search maximizes context
  size (validated by the constraint engine and benchmark success), with
  generation throughput as a tiebreaker. Previously it was a placeholder that
  optimized generation speed.
- CLI test suite (`tests/test_cli.py`) and objective scoring test suite
  (`tests/test_optimizer_scoring.py`).
- Continuous integration on Linux, Windows, and macOS via GitHub Actions.
- This changelog.

### Changed

- `--help` output is cleaner: usage examples moved to a single epilog instead
  of being appended to every command description.
- Tests that inspect real GGUF models now skip gracefully when the model file
  is not present on the machine.

## [0.2.0] - 2026-06

### Added

- Auto-scaling speed probe: benchmark size and trial budget scale to the
  measured hardware speed tier (`very_slow` / `slow` / `medium` / `fast`).
- MoE detection from GGUF headers (`expert_count`, `expert_used_count`),
  reporting active vs total parameters. Verified with OLMoE-1B-7B.
- Multi-GPU tensor-split search.
- SQLite persistence for benchmarks and launch profiles.

## [0.1.0] - 2026-06

### Added

- Initial release: hardware detection, GGUF inspection, llama-bench
  integration, heuristic config generation, 3-stage optimizer
  (baseline → grid search → Optuna), constraint engine with VRAM
  estimation and OOM detection, launch profiles, Typer CLI.
