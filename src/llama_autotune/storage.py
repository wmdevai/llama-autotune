"""Storage overview and cleanup for generated files and KV-cache slots.

Lists the files llama-autotune generates (benchmark database, calibration
store, exported profiles) plus llama.cpp's persisted KV-cache slots, with
their sizes, and deletes them on request.
"""

from __future__ import annotations

import shutil
from pathlib import Path

AUTOTUNE_DIR = Path.home() / ".llama-autotune"
SLOTS_DIR = Path.home() / ".local" / "share" / "llama.cpp" / "slots"


def _dir_size(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024

    return f"{value:.1f} TB"


def list_storage() -> dict:
    """Return an overview of generated files and KV-cache slots."""
    items: list[dict] = []

    for name in ("benchmarks.db", "calibrations.json"):
        path = AUTOTUNE_DIR / name
        if path.is_file():
            size = path.stat().st_size
            items.append({
                "key": name,
                "name": name,
                "path": str(path),
                "size_bytes": size,
                "size_human": _human_size(size),
                "kind": "file",
                "source": "llama-autotune",
            })

    profiles = AUTOTUNE_DIR / "profiles"
    if profiles.is_dir():
        size = _dir_size(profiles)
        items.append({
            "key": "profiles",
            "name": "profiles/",
            "path": str(profiles),
            "size_bytes": size,
            "size_human": _human_size(size),
            "kind": "dir",
            "source": "llama-autotune",
        })

    slots: list[dict] = []
    slots_total = 0

    if SLOTS_DIR.is_dir():
        for f in sorted(SLOTS_DIR.iterdir()):
            if f.is_file():
                size = f.stat().st_size
                slots_total += size
                slots.append({
                    "name": f.name,
                    "size_bytes": size,
                    "size_human": _human_size(size),
                })

    return {
        "autotune_dir": str(AUTOTUNE_DIR),
        "slots_dir": str(SLOTS_DIR),
        "items": items,
        "slots": slots,
        "slots_total_bytes": slots_total,
        "slots_total_human": _human_size(slots_total),
    }


def _resolve_slot(name: str) -> Path:
    """Resolve a KV-cache slot name to a path inside ``SLOTS_DIR``.

    Rejects empty names, dot segments, and any name containing path
    separators or absolute paths so a caller-supplied key cannot escape
    the slots directory (path traversal).

    Args:
        name: The slot name portion of a ``slot:<name>`` key.

    Returns:
        The resolved path of the slot file inside ``SLOTS_DIR``.

    Raises:
        ValueError: When the name is empty, a dot segment, or would
            resolve outside ``SLOTS_DIR``.
    """
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or "\x00" in name
    ):
        raise ValueError(f"Chiave di archiviazione sconosciuta: slot:{name}")

    target = (SLOTS_DIR / name).resolve()
    slots_root = SLOTS_DIR.resolve()

    if target.parent != slots_root:
        raise ValueError(f"Chiave di archiviazione sconosciuta: slot:{name}")

    return target


def delete_item(key: str) -> dict:
    """Delete a known generated file, directory, or KV-cache slot.

    Args:
        key: One of ``benchmarks.db``, ``calibrations.json``, ``profiles``,
            ``slots`` (all slots), or ``slot:<name>``.

    Returns:
        A dict with the deletion summary (files removed and bytes freed).

    Raises:
        ValueError: For an unknown key.
        FileNotFoundError: When nothing exists for the key.
    """
    if key == "slots":
        freed = 0
        deleted = 0
        if SLOTS_DIR.is_dir():
            for f in SLOTS_DIR.iterdir():
                if f.is_file():
                    freed += f.stat().st_size
                    f.unlink()
                    deleted += 1
        return {
            "deleted": key,
            "deleted_files": deleted,
            "freed_bytes": freed,
            "freed_human": _human_size(freed),
        }

    if key == "benchmarks.db":
        target = AUTOTUNE_DIR / "benchmarks.db"
    elif key == "calibrations.json":
        target = AUTOTUNE_DIR / "calibrations.json"
    elif key == "profiles":
        target = AUTOTUNE_DIR / "profiles"
    elif key.startswith("slot:"):
        target = _resolve_slot(key[len("slot:"):])
    else:
        raise ValueError(f"Chiave di archiviazione sconosciuta: {key}")

    if target.is_dir():
        freed = _dir_size(target)
        shutil.rmtree(target)
        deleted_files = 1
    elif target.is_file():
        freed = target.stat().st_size
        target.unlink()
        deleted_files = 1
    else:
        raise FileNotFoundError(f"Nessun elemento da eliminare per: {key}")

    return {
        "deleted": key,
        "deleted_files": deleted_files,
        "freed_bytes": freed,
        "freed_human": _human_size(freed),
    }
