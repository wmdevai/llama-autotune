"""Tests for storage.py — generated files and KV-cache slot management."""

import pytest

from llama_autotune import storage


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "AUTOTUNE_DIR", tmp_path / "autotune")
    monkeypatch.setattr(storage, "SLOTS_DIR", tmp_path / "slots")


def test_list_storage_empty(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)

    data = storage.list_storage()

    assert data["items"] == []
    assert data["slots"] == []
    assert data["slots_total_bytes"] == 0


def test_list_storage_reports_files_and_slots(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)

    autotune = storage.AUTOTUNE_DIR
    autotune.mkdir()
    (autotune / "benchmarks.db").write_bytes(b"x" * 2048)
    (autotune / "calibrations.json").write_text("{}", encoding="utf-8")

    slots = storage.SLOTS_DIR
    slots.mkdir()
    (slots / "model-a").write_bytes(b"y" * 1024)
    (slots / "model-b").write_bytes(b"z" * 4096)

    data = storage.list_storage()

    assert len(data["items"]) == 2
    assert len(data["slots"]) == 2
    assert data["slots_total_bytes"] == 5120


def test_delete_file(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)

    storage.AUTOTUNE_DIR.mkdir()
    (storage.AUTOTUNE_DIR / "benchmarks.db").write_bytes(b"x" * 1024)

    result = storage.delete_item("benchmarks.db")

    assert result["freed_bytes"] == 1024
    assert not (storage.AUTOTUNE_DIR / "benchmarks.db").exists()


def test_delete_slot(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)

    storage.SLOTS_DIR.mkdir()
    (storage.SLOTS_DIR / "model-a").write_bytes(b"y" * 1024)

    result = storage.delete_item("slot:model-a")

    assert result["freed_bytes"] == 1024
    assert not (storage.SLOTS_DIR / "model-a").exists()


def test_delete_all_slots(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)

    storage.SLOTS_DIR.mkdir()
    (storage.SLOTS_DIR / "a").write_bytes(b"1" * 100)
    (storage.SLOTS_DIR / "b").write_bytes(b"2" * 200)

    result = storage.delete_item("slots")

    assert result["deleted_files"] == 2
    assert result["freed_bytes"] == 300
    assert list(storage.SLOTS_DIR.iterdir()) == []


def test_delete_unknown_key_raises(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)

    with pytest.raises(ValueError):
        storage.delete_item("nope")


def test_delete_missing_file_raises(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)

    with pytest.raises(FileNotFoundError):
        storage.delete_item("benchmarks.db")


def test_delete_slot_path_traversal_rejected(monkeypatch, tmp_path):
    _patch_dirs(monkeypatch, tmp_path)

    for key in (
        "slot:",
        "slot:.",
        "slot:..",
        "slot:../outside",
        "slot:a/b",
        "slot:/etc/passwd",
        "slot:..\\windows",
    ):
        with pytest.raises(ValueError):
            storage.delete_item(key)
