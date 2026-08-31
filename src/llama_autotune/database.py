"""SQLAlchemy ORM models and persistence helpers for llama-autotune.

Defines the database schema (``HardwareProfileModel``, ``ModelProfileModel``,
``BenchmarkModel``, ``LaunchProfileModel``) and convenience functions for
storing and retrieving benchmark results and launch profiles.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Float, Integer, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from .models import BenchmarkEntry, BenchmarkResult, LaunchProfile


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""
    pass


class HardwareProfileModel(Base):
    """Persisted hardware profile entry.

    Stores CPU/GPU topology, memory sizes, and backend type for a given
    machine. Corresponds to the ``hardware_profiles`` table.
    """
    __tablename__ = "hardware_profiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cpu_name: Mapped[str] = mapped_column(String)
    physical_cores: Mapped[int] = mapped_column(Integer)
    logical_cores: Mapped[int] = mapped_column(Integer)
    ram_gb: Mapped[float] = mapped_column(Float)
    gpu_count: Mapped[int] = mapped_column(Integer)
    gpu_vendor: Mapped[str] = mapped_column(String)
    gpu_models: Mapped[str] = mapped_column(Text)
    vram_per_gpu: Mapped[str] = mapped_column(Text)
    backend: Mapped[str] = mapped_column(String)


class ModelProfileModel(Base):
    """Persisted model profile entry.

    Stores model architecture metadata such as parameter count,
    quantization, layer/head counts, and file size. Corresponds to the
    ``model_profiles`` table.
    """
    __tablename__ = "model_profiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String, unique=True)
    architecture: Mapped[str] = mapped_column(String)
    parameters: Mapped[int] = mapped_column(Integer)
    quantization: Mapped[str] = mapped_column(String)
    n_layers: Mapped[int] = mapped_column(Integer)
    n_heads: Mapped[int] = mapped_column(Integer)
    training_context: Mapped[int] = mapped_column(Integer)
    is_moe: Mapped[int] = mapped_column(Integer)
    active_parameters: Mapped[int] = mapped_column(Integer)
    file_size_gb: Mapped[float] = mapped_column(Float)


class BenchmarkModel(Base):
    """Persisted benchmark result entry.

    Records a single benchmark run including throughput metrics, memory
    usage, the configuration JSON, and the optimisation objective.
    Corresponds to the ``benchmarks`` table.
    """
    __tablename__ = "benchmarks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    hardware_id: Mapped[str] = mapped_column(String)
    model_id: Mapped[str] = mapped_column(String)
    config_json: Mapped[str] = mapped_column(Text)
    prompt_tps: Mapped[float] = mapped_column(Float)
    generation_tps: Mapped[float] = mapped_column(Float)
    startup_time: Mapped[float] = mapped_column(Float)
    memory_usage: Mapped[float] = mapped_column(Float)
    vram_usage: Mapped[float] = mapped_column(Float)
    success: Mapped[int] = mapped_column(Integer)
    objective: Mapped[str] = mapped_column(String)
    timestamp: Mapped[str] = mapped_column(String)


class TrialCacheModel(Base):
    """Persisted per-trial benchmark cache for ``--resume`` support.

    Stores one row per benchmarked configuration so a subsequent search on
    the same model can skip configurations it has already evaluated.
    """
    __tablename__ = "trial_cache"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(String)
    config_json: Mapped[str] = mapped_column(Text)
    prompt_tps: Mapped[float] = mapped_column(Float)
    generation_tps: Mapped[float] = mapped_column(Float)
    startup_time: Mapped[float] = mapped_column(Float)
    memory_usage: Mapped[float] = mapped_column(Float)
    vram_usage: Mapped[float] = mapped_column(Float)
    success: Mapped[int] = mapped_column(Integer)
    raw_output: Mapped[str] = mapped_column(Text)


class LaunchProfileModel(Base):
    """Persisted launch-profile entry.

    Stores a named profile with its CLI arguments, model path, hardware
    description, and score. Corresponds to the ``launch_profiles`` table.
    """
    __tablename__ = "launch_profiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    args_json: Mapped[str] = mapped_column(Text)
    model_path: Mapped[str] = mapped_column(String)
    hardware: Mapped[str] = mapped_column(String)
    created: Mapped[str] = mapped_column(String)
    score: Mapped[float] = mapped_column(Float)


def get_db_path() -> str:
    """Return the default path for the SQLite benchmark database.

    The database is stored at ``~/.llama-autotune/benchmarks.db``. The
    parent directory is created if it does not exist.

    Returns:
        Absolute path to the database file.
    """
    data_dir = os.path.join(Path.home(), ".llama-autotune")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "benchmarks.db")


def get_session(db_path: str | None = None) -> Session:
    """Create and return a new SQLAlchemy session.

    Ensures all tables exist before returning the session.

    Args:
        db_path: Path to the SQLite database. Defaults to ``get_db_path()``.

    Returns:
        An open SQLAlchemy ``Session``.
    """
    if db_path is None:
        db_path = get_db_path()
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    _ensure_columns(engine)
    return Session(engine)


def _ensure_columns(engine) -> None:
    """Add columns introduced after the initial schema to existing DBs.

    ``Base.metadata.create_all`` creates missing tables but does not add new
    columns to tables that already exist, so this best-effort helper adds
    ``vram_usage`` when it is missing.
    """
    try:
        with engine.begin() as conn:
            for table in ("benchmarks", "trial_cache"):
                rows = conn.exec_driver_sql(
                    f"PRAGMA table_info({table})"
                ).fetchall()
                existing = {row[1] for row in rows}
                if "vram_usage" not in existing:
                    conn.exec_driver_sql(
                        f"ALTER TABLE {table} ADD COLUMN vram_usage FLOAT"
                    )
    except Exception:
        pass


@contextmanager
def session_scope(db_path: str | None = None):
    """Yield a session and close it (and its engine) on exit.

    Wraps :func:`get_session` so callers do not leak the SQLite connection
    pool. The session is closed and the underlying engine disposed in a
    ``finally`` block regardless of how the body exits.

    Args:
        db_path: Path to the SQLite database. Defaults to ``get_db_path()``.

    Yields:
        An open SQLAlchemy ``Session``.
    """
    session = get_session(db_path)
    bind = session.get_bind()
    try:
        yield session
    finally:
        session.close()
        if isinstance(bind, Engine):
            bind.dispose()


def save_benchmark(session: Session, entry: BenchmarkEntry) -> None:
    """Persist a benchmark entry to the database.

    Args:
        session: An active SQLAlchemy session.
        entry: The benchmark entry to save.
    """
    bm = BenchmarkModel(
        hardware_id=entry.hardware_id,
        model_id=entry.model_id,
        config_json=entry.config.model_dump_json(),
        prompt_tps=entry.result.prompt_tps,
        generation_tps=entry.result.generation_tps,
        startup_time=entry.result.startup_time,
        memory_usage=entry.result.memory_usage,
        vram_usage=entry.result.vram_usage,
        success=1 if entry.result.success else 0,
        objective=entry.objective.value,
        timestamp=entry.timestamp or datetime.now(timezone.utc).isoformat(),
    )
    session.add(bm)
    session.commit()


def save_launch_profile(session: Session, profile: LaunchProfile) -> None:
    """Upsert a launch profile into the database.

    If a profile with the same name already exists its fields are updated;
    otherwise a new row is inserted.

    Args:
        session: An active SQLAlchemy session.
        profile: The launch profile to persist.
    """
    existing = (
        session.query(LaunchProfileModel)
        .filter(LaunchProfileModel.name == profile.name)
        .first()
    )
    if existing:
        existing.args_json = json.dumps(profile.args)
        existing.model_path = profile.model_path
        existing.hardware = profile.hardware
        existing.score = profile.score
    else:
        model = LaunchProfileModel(
            name=profile.name,
            args_json=json.dumps(profile.args),
            model_path=profile.model_path,
            hardware=profile.hardware,
            created=profile.created,
            score=profile.score,
        )
        session.add(model)
    session.commit()


def save_trial_cache(
    session: Session,
    model_id: str,
    cache: dict[str, BenchmarkResult],
) -> None:
    """Upsert the optimizer's in-memory benchmark cache into the database.

    Args:
        session: An active SQLAlchemy session.
        model_id: Stable identifier for the model (e.g. its file path).
        cache: Mapping of config JSON to benchmark result.
    """
    for config_json, result in cache.items():
        existing = (
            session.query(TrialCacheModel)
            .filter(
                TrialCacheModel.model_id == model_id,
                TrialCacheModel.config_json == config_json,
            )
            .first()
        )

        values = {
            "prompt_tps": result.prompt_tps,
            "generation_tps": result.generation_tps,
            "startup_time": result.startup_time,
            "memory_usage": result.memory_usage,
            "vram_usage": result.vram_usage,
            "success": 1 if result.success else 0,
            "raw_output": result.raw_output or "",
        }

        if existing:
            for key, value in values.items():
                setattr(existing, key, value)
        else:
            session.add(
                TrialCacheModel(
                    model_id=model_id,
                    config_json=config_json,
                    **values,
                )
            )

    session.commit()


def load_trial_cache(
    session: Session,
    model_id: str,
) -> dict[str, BenchmarkResult]:
    """Load cached benchmark results keyed by config JSON.

    Args:
        session: An active SQLAlchemy session.
        model_id: Stable identifier for the model (e.g. its file path).

    Returns:
        A mapping of config JSON to ``BenchmarkResult``.
    """
    rows = (
        session.query(TrialCacheModel)
        .filter(TrialCacheModel.model_id == model_id)
        .all()
    )

    cache: dict[str, BenchmarkResult] = {}
    for row in rows:
        cache[row.config_json] = BenchmarkResult(
            prompt_tps=row.prompt_tps,
            generation_tps=row.generation_tps,
            startup_time=row.startup_time,
            memory_usage=row.memory_usage,
            vram_usage=row.vram_usage or 0.0,
            success=bool(row.success),
            raw_output=row.raw_output or "",
        )
    return cache
