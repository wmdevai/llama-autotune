from llama_autotune.constraints import (
    _estimate_kv_cache,
    _head_dim,
    _kv_cache_bytes_per_element,
    detect_oom_in_output,
    estimate_max_offloadable_layers,
    estimate_vram,
    is_oom,
    is_plausible,
)
from llama_autotune.models import HardwareInfo, ModelInfo, SearchConfig

import pytest


def _hw() -> HardwareInfo:
    return HardwareInfo(
        cpu_name="Test",
        physical_cores=16,
        logical_cores=32,
        ram_gb=64,
        gpu_count=1,
        vram_per_gpu=[24.0],
    )


def _model() -> ModelInfo:
    return ModelInfo(
        architecture="test",
        parameters=7_000_000_000,
        quantization="Q4_K_M",
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        embedding_length=4096,
        file_size_gb=4.5,
    )


def test_estimate_vram():
    config = SearchConfig(ctx_size=4096, n_gpu_layers=999)
    hw = _hw()
    model = _model()
    estimated = estimate_vram(config, model, hw)
    assert estimated > 0


def test_estimate_vram_cpu_only_is_zero():
    """With no GPU offload the VRAM estimate must be zero (weights+KV in RAM)."""
    config = SearchConfig(ctx_size=4096)
    assert estimate_vram(config, _model(), _hw()) == 0.0


def test_estimate_vram_no_kv_offload_is_model_only():
    config = SearchConfig(ctx_size=4096, n_gpu_layers=999, no_kv_offload=True)
    model = _model()
    # model VRAM only: 4.5 GB * Q4_K overhead (1.05) = 4.725 GB
    assert estimate_vram(config, model, _hw()) == pytest.approx(4.725)


def test_is_oom_false():
    config = SearchConfig(ctx_size=4096)
    hw = _hw()
    model = _model()
    assert not is_oom(config, model, hw)


def test_is_plausible_too_many_threads():
    hw = _hw()
    hw.logical_cores = 8
    config = SearchConfig(threads=16)
    assert not is_plausible(config, _model(), hw)


def test_is_plausible_valid():
    config = SearchConfig(threads=8, ctx_size=4096)
    assert is_plausible(config, _model(), _hw())


def test_detect_oom_in_output():
    assert detect_oom_in_output("CUDA error: out of memory")
    assert detect_oom_in_output("failed to allocate memory")
    assert detect_oom_in_output("OOM")
    assert not detect_oom_in_output("normal output")
    assert not detect_oom_in_output("")


# ── KV cache estimation (GQA-aware) ──────────────────────────────────


def _kv_model(n_kv_heads: int, embedding_length: int = 4096) -> ModelInfo:
    return ModelInfo(
        n_layers=32,
        n_heads=32,
        n_kv_heads=n_kv_heads,
        embedding_length=embedding_length,
        file_size_gb=0.0,
    )


def test_head_dim_from_embedding_length():
    assert _head_dim(_kv_model(8, 4096)) == 128
    assert _head_dim(_kv_model(8, 3584)) == 112  # 3584 / 32


def test_head_dim_fallback_when_unknown():
    model = ModelInfo(n_layers=32, n_heads=32, embedding_length=0)
    assert _head_dim(model) == 128


def test_kv_cache_gqa_reduces_estimate():
    """Fewer KV heads (GQA) must shrink the KV cache estimate."""
    config = SearchConfig(ctx_size=4096, n_gpu_layers=999)
    gqa = _estimate_kv_cache(config, _kv_model(8), 1.0)
    dense = _estimate_kv_cache(config, _kv_model(32), 1.0)
    assert gqa < dense
    assert gqa == dense / 4  # 8 vs 32 KV heads


def test_kv_cache_respects_cache_type():
    """q8_0 K/V caches must be smaller than f16."""
    config_f16 = SearchConfig(ctx_size=4096, n_gpu_layers=999)
    config_q8 = SearchConfig(
        ctx_size=4096,
        n_gpu_layers=999,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
    )
    model = _kv_model(8)
    assert _estimate_kv_cache(config_q8, model, 1.0) < _estimate_kv_cache(
        config_f16, model, 1.0
    )


def test_kv_cache_respects_no_kv_offload():
    config = SearchConfig(ctx_size=4096, n_gpu_layers=999, no_kv_offload=True)
    assert _estimate_kv_cache(config, _kv_model(8), 1.0) == 0.0


def test_kv_cache_scales_with_gpu_fraction():
    """Partial offload must scale the KV cache proportionally."""
    config = SearchConfig(ctx_size=4096, n_gpu_layers=999)
    model = _kv_model(8)
    full = _estimate_kv_cache(config, model, 1.0)
    half = _estimate_kv_cache(config, model, 0.5)
    assert half == full / 2


def test_kv_cache_bytes_per_element_defaults_to_f16():
    assert _kv_cache_bytes_per_element(None) == 2.0


def test_max_offloadable_layers_accounts_for_kv():
    """A model with a large KV cache must not report full offload as fitting."""
    model = ModelInfo(
        n_layers=65,
        n_heads=24,
        n_kv_heads=4,
        embedding_length=5120,
        file_size_gb=14.3,
        quantization="Q4_K_S",
    )
    hw = HardwareInfo(gpu_count=1, vram_per_gpu=[15.9])

    max_ngl = estimate_max_offloadable_layers(model, hw)

    assert 1 <= max_ngl < model.n_layers


def test_max_offloadable_layers_full_when_small_kv():
    model = ModelInfo(
        n_layers=40,
        n_heads=32,
        n_kv_heads=8,
        embedding_length=4096,
        file_size_gb=4.0,
        quantization="Q4_K_M",
    )
    hw = HardwareInfo(gpu_count=1, vram_per_gpu=[24.0])

    assert estimate_max_offloadable_layers(model, hw) == model.n_layers
    assert _kv_cache_bytes_per_element("f16") == 2.0
    assert _kv_cache_bytes_per_element("q8_0") == 1.0
    assert _kv_cache_bytes_per_element("q4_0") == 0.5
