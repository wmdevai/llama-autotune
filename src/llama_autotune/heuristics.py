"""Heuristic-based configuration generation for llama-autotune.

Provides functions that produce sane default configurations based on
detected hardware and model properties, without running benchmarks.
"""

from __future__ import annotations

from .models import Backend, HardwareInfo, ModelInfo, SearchConfig, SplitMode


def generate_initial_config(hw: HardwareInfo, model: ModelInfo) -> SearchConfig:
    """Create a reasonable starting SearchConfig for the given hardware and model.

    Args:
        hw: Detected hardware information.
        model: Model metadata.

    Returns:
        A populated SearchConfig with conservative defaults.
    """
    config = SearchConfig()

    _set_basics(config, model)
    _set_ctx(config, model)

    if hw.backend == Backend.CPU:
        _configure_cpu(config, hw)
    else:
        _configure_gpu(config, hw, model)

    return config


def _set_basics(config: SearchConfig, model: ModelInfo) -> None:
    """Fill config with baseline parameter values.

    Args:
        config: The configuration object to mutate.
        model: Model metadata (currently unused but reserved).
    """
    config.flash_attn = True
    config.n_gpu_layers = 999
    config.batch_size = 2048
    config.ubatch_size = 512


def _set_ctx(config: SearchConfig, model: ModelInfo) -> None:
    """Set a sensible context size based on the model's training context.

    Args:
        config: The configuration object to mutate.
        model: Model metadata containing training_context.
    """
    ctx = model.training_context
    if ctx > 0:
        config.ctx_size = min(ctx, 24576)
    else:
        config.ctx_size = 24576


def _configure_cpu(config: SearchConfig, hw: HardwareInfo) -> None:
    """Tweak config values for CPU-only execution.

    Args:
        config: The configuration object to mutate.
        hw: Hardware information used to set thread count.
    """
    config.threads = hw.physical_cores
    config.batch_size = min(config.batch_size or 2048, 512)
    config.ubatch_size = min(config.ubatch_size or 512, 256)
    config.n_gpu_layers = 0
    config.flash_attn = False


def _configure_gpu(config: SearchConfig, hw: HardwareInfo, model: ModelInfo) -> None:
    """Tweak config values for GPU-accelerated execution.

    Args:
        config: The configuration object to mutate.
        hw: Hardware information used to set thread count and split mode.
        model: Model metadata used to fit the config to available VRAM.
    """
    config.threads = max(1, min(hw.logical_cores, hw.physical_cores))
    if hw.gpu_count > 1:
        config.split_mode = SplitMode.LAYER

    _fit_to_vram(config, hw, model)


def _fit_to_vram(config: SearchConfig, hw: HardwareInfo, model: ModelInfo) -> None:
    """Adjust a GPU config so it fits in available VRAM.

    Reduces the number of offloaded layers first, then the context size,
    and finally falls back to CPU-only when nothing fits. This prevents the
    optimizer from starting from a full-offload baseline that only works
    thanks to GPU virtual-memory paging.
    """
    from .constraints import (
        estimate_max_offloadable_layers,
        is_plausible,
    )

    if is_plausible(config, model, hw):
        return

    if config.n_gpu_layers and config.n_gpu_layers > 0:
        config.n_gpu_layers = estimate_max_offloadable_layers(
            model,
            hw,
            ctx_size=config.ctx_size,
        )

    ctx = config.ctx_size or 24576
    while not is_plausible(config, model, hw) and ctx > 1024:
        ctx = max(1024, ctx // 2)
        config.ctx_size = ctx
        if config.n_gpu_layers:
            config.n_gpu_layers = estimate_max_offloadable_layers(
                model,
                hw,
                ctx_size=ctx,
            )

    if not is_plausible(config, model, hw):
        config.n_gpu_layers = 0
        config.flash_attn = False


def to_cpu_config(config: SearchConfig) -> SearchConfig:
    """Return a copy of *config* with all GPU-related features disabled.

    Args:
        config: The source configuration to clone.

    Returns:
        A new SearchConfig with n_gpu_layers set to 0 and flash_attn disabled.
    """
    cfg = config.model_copy()
    cfg.n_gpu_layers = 0
    cfg.flash_attn = False
    return cfg


def apply_search_config(hw: HardwareInfo, model: ModelInfo, base: SearchConfig) -> SearchConfig:
    """Return a copy of *base* with hardware-adjusted thread count applied.

    Args:
        hw: Hardware information used to cap thread count.
        model: Model metadata (currently unused but reserved).
        base: The base configuration to clone and adjust.

    Returns:
        A new SearchConfig with threads clamped to available logical/physical cores.
    """
    cfg = base.model_copy()
    cfg.threads = max(1, min(hw.logical_cores, hw.physical_cores))
    return cfg
