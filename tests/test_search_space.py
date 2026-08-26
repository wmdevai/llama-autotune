from llama_autotune.models import Backend, HardwareInfo, ModelInfo, OptimizeObjective, SearchConfig
from llama_autotune.search_space import config_from_params, get_search_space


def _hw() -> HardwareInfo:
    return HardwareInfo(
        cpu_name="Test CPU",
        physical_cores=16,
        logical_cores=32,
        ram_gb=64,
        backend=Backend.CPU,
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


def test_get_cpu_space():
    space = get_search_space(_hw(), _model(), OptimizeObjective.BALANCED)
    assert "threads" in space
    assert "batch_size" in space
    assert "ctx_size" in space


def test_get_gpu_space():
    hw = _hw()
    hw.backend = Backend.CUDA
    hw.gpu_count = 1
    hw.gpu_vendor = "nvidia"
    space = get_search_space(hw, _model(), OptimizeObjective.BALANCED)
    assert "n_gpu_layers" in space
    assert "batch_size" in space


def test_config_from_params():
    params = {"threads": 8, "batch_size": 1024}
    cfg = config_from_params(params)
    assert cfg.threads == 8
    assert cfg.batch_size == 1024


def test_config_from_params_with_base():
    base = SearchConfig(threads=16, ctx_size=4096)
    params = {"batch_size": 2048}
    cfg = config_from_params(params, base)
    assert cfg.threads == 16
    assert cfg.batch_size == 2048
    assert cfg.ctx_size == 4096


def test_max_context_objective():
    hw = _hw()
    space = get_search_space(hw, _model(), OptimizeObjective.MAX_CONTEXT)
    ctx = space.get("ctx_size")
    assert ctx is not None
    assert ctx.high >= 4096


def test_batch_ubatch_are_categorical():
    space = get_search_space(_hw(), _model(), OptimizeObjective.BALANCED)
    batch = space["batch_size"]
    ubatch = space["ubatch_size"]
    assert batch.is_categorical
    assert ubatch.is_categorical
    assert 8192 in batch.categories
    assert 1024 in ubatch.categories


def test_gpu_layers_capped_by_vram():
    hw = _hw()
    hw.backend = Backend.CUDA
    hw.gpu_count = 1
    hw.vram_per_gpu = [8.0]
    model = _model()
    model.file_size_gb = 50.0  # large model, small VRAM
    space = get_search_space(hw, model, OptimizeObjective.BALANCED)
    ngl = space["n_gpu_layers"]
    assert ngl.high < model.n_layers
    assert ngl.high >= 1


def test_gpu_layers_full_range_when_vram_plenty():
    hw = _hw()
    hw.backend = Backend.CUDA
    hw.gpu_count = 1
    hw.vram_per_gpu = [80.0]
    model = _model()
    space = get_search_space(hw, model, OptimizeObjective.BALANCED)
    assert space["n_gpu_layers"].high == model.n_layers
