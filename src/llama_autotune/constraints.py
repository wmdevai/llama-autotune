"""Hardware-constraint checks for configuration validation.

Provides functions that estimate VRAM usage, detect out-of-memory (OOM)
conditions, and validate whether a given ``SearchConfig`` is plausible for
the current hardware and model.
"""

from __future__ import annotations

from .models import HardwareInfo, ModelInfo, SearchConfig


def estimate_vram(config: SearchConfig, model: ModelInfo, hw: HardwareInfo) -> float:
    """Estimate the total VRAM (in GB) required for a given configuration.

    Accounts for the fraction of the model offloaded to the GPU and for the
    KV cache belonging to offloaded layers.

    Args:
        config: The configuration to evaluate.
        model: Model metadata (file size, quantization, heads, etc.).
        hw: Hardware information (currently unused but reserved).

    Returns:
        Estimated VRAM usage in gigabytes.
    """
    gpu_fraction = _gpu_fraction(config, model)

    model_vram_gb = _estimate_model_vram(config, model, gpu_fraction)
    kv_cache_gb = _estimate_kv_cache(config, model, gpu_fraction)

    return model_vram_gb + kv_cache_gb


def _gpu_fraction(config: SearchConfig, model: ModelInfo) -> float:
    """Return the fraction of the model that lives on the GPU (0.0 to 1.0).

    ``None`` or ``0`` mean no GPU offload; values at or above the model's
    layer count are treated as full offload.
    """
    n_layers = model.n_layers or 1
    n_gpu_layers = config.n_gpu_layers

    if n_gpu_layers is None or n_gpu_layers <= 0:
        return 0.0
    if n_gpu_layers >= n_layers:
        return 1.0
    return n_gpu_layers / n_layers


def _estimate_model_vram(
    config: SearchConfig,
    model: ModelInfo,
    gpu_fraction: float,
) -> float:
    """Estimate the VRAM occupied by model weights.

    For both dense and MoE models the weights are distributed per
    transformer block, so offloading ``k`` of ``N`` layers moves roughly
    ``k / N`` of the model (attention *and* experts) onto the GPU. The
    uniform fraction is therefore a sound estimate for MoE as well.
    """
    overhead_factor = _overhead_factor(model)
    return model.file_size_gb * overhead_factor * gpu_fraction


def _overhead_factor(model: ModelInfo) -> float:
    """Return the runtime overhead multiplier for a quantization.

    K-quants were calibrated against real ``nvidia-smi`` measurements on
    Qwen3-14B / Qwen3.8-27B models (Q3_K, Q4_K_S, Q5_K_M): peak VRAM was
    0.96-0.99× the file size, so 1.05 is a slightly conservative runtime
    factor. The legacy / unmeasured quantizations keep their heuristic
    factors and should be calibrated the same way.
    """
    quant = model.quantization or ""
    if "_K" in quant:
        return 1.05
    if "IQ" in quant:
        return 1.1
    if "Q4" in quant:
        return 1.15
    if "Q8" in quant:
        return 1.2
    if "F16" in quant:
        return 1.3
    return 1.15


def _estimate_kv_cache(
    config: SearchConfig,
    model: ModelInfo,
    gpu_fraction: float,
) -> float:
    """Estimate the GPU-resident KV-cache size in gigabytes.

    Uses the number of KV heads (GQA/MQA-aware), the per-head dimension
    derived from the embedding length, and the configured K/V cache types
    (``cache_type_k`` / ``cache_type_v``). Only the KV cache belonging to
    offloaded layers counts toward VRAM; ``no_kv_offload`` moves it entirely
    to system RAM.
    """
    if config.no_kv_offload or gpu_fraction <= 0.0:
        return 0.0

    ctx = config.ctx_size or 4096
    n_layers = model.n_layers or 80
    n_kv_heads = model.n_kv_heads or model.n_heads or 32
    head_dim = _head_dim(model)

    bytes_k = _kv_cache_bytes_per_element(config.cache_type_k)
    bytes_v = _kv_cache_bytes_per_element(config.cache_type_v)

    kv_bytes = n_layers * n_kv_heads * head_dim * ctx * (bytes_k + bytes_v)

    return (kv_bytes / (1024**3)) * gpu_fraction


def _head_dim(model: ModelInfo) -> int:
    """Derive the per-head dimension from the embedding length.

    Falls back to 128 (the most common head dimension) when the embedding
    length or head count is unavailable.
    """
    n_heads = model.n_heads or 0
    if n_heads > 0 and model.embedding_length > 0:
        dim = model.embedding_length // n_heads
        if dim > 0:
            return dim
    return 128


def _kv_cache_bytes_per_element(cache_type: str | None) -> float:
    """Return bytes per element for a llama.cpp KV-cache type.

    Unknown or missing types default to F16 (2 bytes per element).
    """
    if not cache_type:
        return 2.0
    t = cache_type.lower()
    if t == "f32":
        return 4.0
    if t in ("f16", "bf16"):
        return 2.0
    if t in ("q8_0", "q8_1"):
        return 1.0
    if t in ("q4_0", "q4_1", "q4_k"):
        return 0.5
    if t in ("q5_0", "q5_1", "q5_k"):
        return 0.625
    if t == "q6_k":
        return 0.75
    return 2.0


def is_oom(config: SearchConfig, model: ModelInfo, hw: HardwareInfo) -> bool:
    """Check whether a configuration would exceed available VRAM.

    Args:
        config: The configuration to evaluate.
        model: Model metadata.
        hw: Hardware information (VRAM per GPU, RAM).

    Returns:
        True if the estimated VRAM exceeds what is available.
    """
    estimated = estimate_vram(config, model, hw)
    available = _available_vram(hw)
    return estimated > available


def _available_vram(hw: HardwareInfo) -> float:
    """Return the total usable VRAM (or RAM) in gigabytes.

    For GPU backends this is 90 % of the sum of per-GPU VRAM; for CPU-only
    backends it is 80 % of system RAM.

    Args:
        hw: Hardware information containing VRAM/RAM details.

    Returns:
        Usable memory in GB.
    """
    if hw.gpu_count > 0 and hw.vram_per_gpu:
        return sum(hw.vram_per_gpu) * 0.9
    return hw.ram_gb * 0.8


def estimate_max_offloadable_layers(
    model: ModelInfo,
    hw: HardwareInfo,
) -> int:
    """Estimate the maximum number of layers that fit in available VRAM.

    Used to bound the ``n_gpu_layers`` search range so that the search does
    not waste trials on configurations that cannot fit on the GPU.

    Args:
        model: Model metadata (layer count, file size, quantization).
        hw: Hardware information (per-GPU VRAM).

    Returns:
        An upper bound on ``n_gpu_layers``: at least 1 and at most the
        model's layer count (0 when the layer count is unknown).
    """
    n_layers = model.n_layers or 0
    if n_layers <= 0 or model.file_size_gb <= 0 or hw.gpu_count <= 0:
        return n_layers

    available = _available_vram(hw)
    per_layer = model.file_size_gb * _overhead_factor(model) / n_layers
    if per_layer <= 0:
        return n_layers

    by_vram = int(available / per_layer)
    return max(1, min(n_layers, by_vram))


def is_plausible(config: SearchConfig, model: ModelInfo, hw: HardwareInfo) -> bool:
    """Determine whether a configuration is worth benchmarking.

    Rejects configurations that would OOM or that request more threads
    than logical cores.

    Args:
        config: The configuration to validate.
        model: Model metadata.
        hw: Hardware information.

    Returns:
        True if the configuration is plausible.
    """
    if is_oom(config, model, hw):
        return False
    if config.threads is not None and config.threads > hw.logical_cores:
        return False
    return True


def detect_oom_in_output(output: str) -> bool:
    """Check a process output string for common OOM / CUDA error patterns.

    Args:
        output: The captured stdout/stderr text.

    Returns:
        True if any known OOM-related pattern was found.
    """
    oom_patterns = [
        "out of memory",
        "oom",
        "cuda error",
        "cudamalloc",
        "cannot allocate",
        "failed to allocate",
    ]
    output_lower = output.lower()
    return any(p in output_lower for p in oom_patterns)
