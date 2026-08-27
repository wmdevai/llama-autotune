from llama_autotune.heuristics import generate_initial_config
from llama_autotune.models import Backend, HardwareInfo, ModelInfo
from llama_autotune.constraints import is_plausible


def _cpu_hw() -> HardwareInfo:
    return HardwareInfo(
        cpu_name="Test CPU",
        physical_cores=16,
        logical_cores=32,
        ram_gb=64,
        backend=Backend.CPU,
    )


def _gpu_hw() -> HardwareInfo:
    return HardwareInfo(
        cpu_name="Test GPU",
        physical_cores=16,
        logical_cores=32,
        ram_gb=64,
        gpu_count=1,
        gpu_vendor="nvidia",
        gpu_models=["RTX 4090"],
        vram_per_gpu=[24.0],
        backend=Backend.CUDA,
    )


def _model() -> ModelInfo:
    return ModelInfo(
        architecture="qwen3",
        parameters=30_000_000_000,
        quantization="Q4_K_M",
        n_layers=80,
        n_heads=32,
        training_context=131072,
        file_size_gb=3.2,
    )


def test_generate_cpu_config():
    cfg = generate_initial_config(_cpu_hw(), _model())
    assert cfg.threads == 16
    assert cfg.n_gpu_layers == 0
    assert cfg.flash_attn is False


def test_generate_gpu_config():
    cfg = generate_initial_config(_gpu_hw(), _model())
    assert cfg.flash_attn is True
    assert cfg.batch_size == 2048
    assert cfg.ubatch_size == 512
    assert cfg.ctx_size == 24576


def test_config_ctx_capped():
    model = _model()
    model.training_context = 1000000
    cfg = generate_initial_config(_gpu_hw(), model)
    assert cfg.ctx_size == 24576


def test_config_no_training_ctx():
    model = _model()
    model.training_context = 0
    cfg = generate_initial_config(_cpu_hw(), model)
    assert cfg.ctx_size == 24576


def test_generate_gpu_config_partial_offload_when_kv_does_not_fit():
    """A model whose weights + KV exceed VRAM must get a partial-offload
    baseline (the server pre-allocates the KV cache at load)."""
    model = ModelInfo(
        architecture="qwen3",
        n_layers=65,
        n_heads=24,
        n_kv_heads=4,
        embedding_length=5120,
        file_size_gb=14.3,
        quantization="Q4_K_S",
        training_context=40960,
    )
    hw = _gpu_hw()
    hw.vram_per_gpu = [15.9]

    cfg = generate_initial_config(hw, model)

    assert cfg.n_gpu_layers < model.n_layers or cfg.ctx_size < 24576
    assert is_plausible(cfg, model, hw)


def test_generate_gpu_config_keeps_full_offload_when_fits():
    model = _model()
    model.file_size_gb = 3.2
    hw = _gpu_hw()
    hw.vram_per_gpu = [80.0]

    cfg = generate_initial_config(hw, model)

    assert cfg.n_gpu_layers == 999


def test_generate_gpu_config_partial_offload_when_weights_exceed_vram():
    """A model whose weights exceed VRAM must get a plausible partial offload."""
    model = ModelInfo(
        architecture="qwen3moe",
        n_layers=41,
        n_heads=24,
        n_kv_heads=4,
        embedding_length=5120,
        file_size_gb=20.75,
        quantization="Q4_K_M",
        training_context=40960,
    )
    hw = _gpu_hw()
    hw.vram_per_gpu = [15.9]

    cfg = generate_initial_config(hw, model)

    assert cfg.n_gpu_layers < model.n_layers
    assert is_plausible(cfg, model, hw)
