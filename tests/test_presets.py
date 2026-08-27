"""Tests for presets.py — presets.ini generation and apply-with-backup."""

from llama_autotune import presets
from llama_autotune.models import SearchConfig


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(
        presets,
        "CURRENT_PRESETS_PATH",
        tmp_path / "current-presets.ini",
    )
    monkeypatch.setattr(
        presets,
        "RECOMMENDED_PRESETS_PATH",
        tmp_path / "recommended-presets.ini",
    )


def test_config_to_ini_lines():
    cfg = SearchConfig(
        n_gpu_layers=999,
        ctx_size=16384,
        batch_size=8192,
        ubatch_size=512,
        flash_attn=True,
        cache_type_k="q8_0",
    )

    lines = presets.config_to_ini_lines(cfg)

    assert "n-gpu-layers = 999" in lines
    assert "ctx-size = 16384" in lines
    assert "batch-size = 8192" in lines
    assert "ubatch-size = 512" in lines
    assert "flash-attn = on" in lines
    assert "cache-type-k = q8_0" in lines
    assert "load-on-startup = false" in lines


def test_generate_recommended_from_empty(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)

    content = presets.generate_recommended(
        SearchConfig(ctx_size=16384, batch_size=2048),
        "/models/Qwen3-14B-Q5_K_M.gguf",
    )

    assert "[Qwen3-14B-Q5_K_M]" in content
    assert "ctx-size = 16384" in content
    assert "version = 1" in content

    assert presets.recommended_content() == content


def test_generate_recommended_updates_existing_section(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)

    current = (
        "version = 1\n"
        "\n"
        "[*]\n"
        "flash-attn = on\n"
        "\n"
        "[Qwen3-14B-Q5_K_M]\n"
        "ctx-size = 4096\n"
        "batch-size = 512\n"
        "load-on-startup = false\n"
    )
    presets.CURRENT_PRESETS_PATH.write_text(current, encoding="utf-8")

    content = presets.generate_recommended(
        SearchConfig(ctx_size=16384, flash_attn=True),
        "/models/Qwen3-14B-Q5_K_M.gguf",
    )

    assert "[Qwen3-14B-Q5_K_M]" in content
    assert "ctx-size = 16384" in content
    assert "ctx-size = 4096" not in content
    # global section preserved
    assert "[*]" in content
    assert "flash-attn = on" in content


def test_apply_recommended_creates_backup(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)

    presets.CURRENT_PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    presets.CURRENT_PRESETS_PATH.write_text("old content", encoding="utf-8")

    presets.generate_recommended(
        SearchConfig(ctx_size=16384),
        "/models/test.gguf",
    )

    result = presets.apply_recommended()

    assert result["backup"] is not None
    assert presets.current_content() == presets.recommended_content()

    backup = tmp_path / "current-presets.ini.bak"
    backups = list(tmp_path.glob("current-presets.ini.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "old content"


def test_apply_recommended_without_recommended_raises(monkeypatch, tmp_path):
    _patch_paths(monkeypatch, tmp_path)

    import pytest

    with pytest.raises(FileNotFoundError):
        presets.apply_recommended()
