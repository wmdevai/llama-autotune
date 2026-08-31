"""Tests for the byte-level GGUF readers in model_inspector.py.

The GGUF readers are exercised with synthetic byte streams so the header
parsing and value decoding can be tested without a real model file.
"""

import io
import struct

import pytest

from llama_autotune import model_inspector as mi

# ── synthetic GGUF writer ─────────────────────────────────────────────


def _pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_typed_value(value) -> bytes:
    """Encode a Python value with its GGUF type tag."""
    if isinstance(value, str):
        return struct.pack("<I", mi.GGUF_TYPE_STRING) + _pack_string(value)
    if isinstance(value, bool):
        return struct.pack("<I", mi.GGUF_TYPE_BOOL) + struct.pack(
            "<B", int(value)
        )
    if isinstance(value, int):
        if value > 0xFFFFFFFF:
            return struct.pack("<I", mi.GGUF_TYPE_UINT64) + struct.pack(
                "<Q", value
            )
        return struct.pack("<I", mi.GGUF_TYPE_UINT32) + struct.pack(
            "<I", value
        )
    raise TypeError(f"unsupported value type: {type(value)}")


def _write_gguf(path, kv: dict) -> None:
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", len(kv)))  # metadata KV count
        for key, value in kv.items():
            f.write(_pack_string(key))
            f.write(_pack_typed_value(value))


# ── _read_string ──────────────────────────────────────────────────────


def test_read_string():
    stream = io.BytesIO(_pack_string("hello"))
    assert mi._read_string(stream) == "hello"


# ── _read_value ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("type_code", "payload", "expected"),
    [
        (mi.GGUF_TYPE_UINT8, struct.pack("<B", 7), 7),
        (mi.GGUF_TYPE_INT8, struct.pack("<b", -3), -3),
        (mi.GGUF_TYPE_UINT16, struct.pack("<H", 65535), 65535),
        (mi.GGUF_TYPE_INT16, struct.pack("<h", -123), -123),
        (mi.GGUF_TYPE_UINT32, struct.pack("<I", 4_000_000_000), 4_000_000_000),
        (mi.GGUF_TYPE_INT32, struct.pack("<i", -42), -42),
        (mi.GGUF_TYPE_FLOAT32, struct.pack("<f", 1.5), 1.5),
        (mi.GGUF_TYPE_BOOL, struct.pack("<B", 1), True),
        (mi.GGUF_TYPE_BOOL, struct.pack("<B", 0), False),
        (mi.GGUF_TYPE_UINT64, struct.pack("<Q", 2**60), 2**60),
        (mi.GGUF_TYPE_INT64, struct.pack("<q", -2**40), -(2**40)),
        (mi.GGUF_TYPE_FLOAT64, struct.pack("<d", 2.25), 2.25),
    ],
)
def test_read_value_scalar_types(type_code, payload, expected):
    stream = io.BytesIO(struct.pack("<I", type_code) + payload)
    assert mi._read_value(stream) == expected


def test_read_value_string():
    stream = io.BytesIO(
        struct.pack("<I", mi.GGUF_TYPE_STRING) + _pack_string("abc")
    )
    assert mi._read_value(stream) == "abc"


def test_read_value_array_of_strings():
    elements = _pack_string("x") + _pack_string("yy")
    stream = io.BytesIO(
        struct.pack("<I", mi.GGUF_TYPE_ARRAY)
        + struct.pack("<I", mi.GGUF_TYPE_STRING)
        + struct.pack("<Q", 2)
        + elements
    )
    assert mi._read_value(stream) == ["x", "yy"]


def test_read_value_unknown_type_returns_none():
    stream = io.BytesIO(struct.pack("<I", 999))
    assert mi._read_value(stream) is None


# ── _read_value_with_type ─────────────────────────────────────────────


def test_read_value_with_type_string():
    stream = io.BytesIO(_pack_string("no-tag"))
    assert mi._read_value_with_type(stream, mi.GGUF_TYPE_STRING) == "no-tag"


def test_read_value_with_type_uint32():
    stream = io.BytesIO(struct.pack("<I", 123))
    assert mi._read_value_with_type(stream, mi.GGUF_TYPE_UINT32) == 123


@pytest.mark.parametrize(
    ("type_code", "payload", "expected"),
    [
        (mi.GGUF_TYPE_UINT8, struct.pack("<B", 7), 7),
        (mi.GGUF_TYPE_INT8, struct.pack("<b", -3), -3),
        (mi.GGUF_TYPE_UINT16, struct.pack("<H", 65535), 65535),
        (mi.GGUF_TYPE_INT16, struct.pack("<h", -123), -123),
        (mi.GGUF_TYPE_INT32, struct.pack("<i", -42), -42),
        (mi.GGUF_TYPE_FLOAT32, struct.pack("<f", 1.5), 1.5),
        (mi.GGUF_TYPE_BOOL, struct.pack("<B", 1), True),
        (mi.GGUF_TYPE_UINT64, struct.pack("<Q", 2**60), 2**60),
        (mi.GGUF_TYPE_INT64, struct.pack("<q", -2**40), -(2**40)),
        (mi.GGUF_TYPE_FLOAT64, struct.pack("<d", 2.25), 2.25),
    ],
)
def test_read_value_with_type_scalars(type_code, payload, expected):
    stream = io.BytesIO(payload)
    assert mi._read_value_with_type(stream, type_code) == expected


def test_read_value_with_type_unknown():
    assert mi._read_value_with_type(io.BytesIO(), 999) is None


# ── _read_gguf_header ─────────────────────────────────────────────────


def test_read_gguf_header_roundtrip(tmp_path):
    path = tmp_path / "model.gguf"
    _write_gguf(
        path,
        {
            "general.architecture": "llama",
            "llama.block_count": 10,
            "general.parameter_count": 7_000_000_000,
        },
    )

    kv = mi._read_gguf_header(str(path))

    assert kv["general.architecture"] == "llama"
    assert kv["llama.block_count"] == 10
    assert kv["general.parameter_count"] == 7_000_000_000


def test_read_gguf_header_invalid_magic(tmp_path):
    path = tmp_path / "not_gguf.bin"
    path.write_bytes(b"NOPE" + b"\x00" * 64)

    assert mi._read_gguf_header(str(path)) == {}


# ── inspect_model on a synthetic GGUF ─────────────────────────────────


def test_inspect_model_synthetic_gguf(tmp_path):
    path = tmp_path / "model.gguf"
    _write_gguf(
        path,
        {
            "general.architecture": "llama",
            "general.file_type": 2,  # Q4_0
            "general.parameter_count": 7_000_000_000,
            "llama.block_count": 10,
            "llama.attention.head_count": 8,
            "llama.attention.head_count_kv": 2,
            "llama.embedding_length": 512,
            "llama.context_length": 4096,
        },
    )

    info = mi.inspect_model(path)

    assert info.architecture == "llama"
    assert info.quantization == "Q4_0"
    assert info.parameters == 7_000_000_000
    assert info.n_layers == 10
    assert info.n_heads == 8
    assert info.n_kv_heads == 2
    assert info.embedding_length == 512
    assert info.training_context == 4096
    assert info.is_moe is False
    assert info.active_parameters == 7_000_000_000


def test_inspect_model_kv_heads_falls_back_to_heads(tmp_path):
    path = tmp_path / "model.gguf"
    _write_gguf(
        path,
        {
            "general.architecture": "llama",
            "llama.block_count": 4,
            "llama.attention.head_count": 8,
            "llama.context_length": 4096,
        },
    )

    info = mi.inspect_model(path)

    assert info.n_heads == 8
    assert info.n_kv_heads == 8


def test_inspect_model_moe_active_params(tmp_path):
    path = tmp_path / "moe.gguf"
    _write_gguf(
        path,
        {
            "general.architecture": "olmoe",
            "general.size_label": "1B-7B",
            "olmoe.block_count": 16,
            "olmoe.attention.head_count": 16,
            "olmoe.expert_count": 64,
            "olmoe.expert_used_count": 8,
        },
    )

    info = mi.inspect_model(path)

    assert info.architecture == "olmoe"
    assert info.is_moe is True
    assert info.parameters == 7_000_000_000
    assert info.active_parameters == 1_000_000_000


def test_inspect_model_moe_expert_count_substring_fallback(tmp_path):
    """MoE detection falls back to any key containing 'expert_count'."""
    path = tmp_path / "moe-fallback.gguf"
    _write_gguf(
        path,
        {
            "general.architecture": "llama",
            "llama.block_count": 4,
            "llama.attention.head_count": 8,
            "some.other.expert_count": 64,
        },
    )

    info = mi.inspect_model(path)

    assert info.is_moe is True
