import os
import tempfile

from llama_autotune.database import (
    BenchmarkModel,
    LaunchProfileModel,
    get_best_benchmark,
    get_session,
    load_trial_cache,
    save_benchmark,
    save_launch_profile,
    save_trial_cache,
)
from llama_autotune.models import (
    BenchmarkEntry,
    BenchmarkResult,
    LaunchProfile,
    OptimizeObjective,
    SearchConfig,
)


def _db_session():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    session = get_session(tmp.name)
    return session, tmp.name, session.bind


def test_save_and_get_benchmark():
    session, db_path, engine = _db_session()
    try:
        entry = BenchmarkEntry(
            hardware_id="test_cpu",
            model_id="test_model",
            config=SearchConfig(threads=8, batch_size=2048),
            result=BenchmarkResult(prompt_tps=5000, generation_tps=100, success=True),
            objective=OptimizeObjective.BALANCED,
        )
        save_benchmark(session, entry)

        best = get_best_benchmark(session, "test_model", "balanced")
        assert best is not None
        assert best.generation_tps == 100
        assert best.prompt_tps == 5000
    finally:
        session.close()
        engine.dispose()
        os.unlink(db_path)


def test_save_and_load_trial_cache():
    session, db_path, engine = _db_session()
    try:
        cache = {
            SearchConfig(threads=4).model_dump_json(): BenchmarkResult(
                generation_tps=10.0,
                success=True,
                raw_output="ok",
            ),
            SearchConfig(threads=8).model_dump_json(): BenchmarkResult(
                generation_tps=20.0,
                success=False,
                raw_output="oom",
            ),
        }

        save_trial_cache(session, "model.gguf", cache)

        loaded = load_trial_cache(session, "model.gguf")

        assert len(loaded) == 2
        key = SearchConfig(threads=4).model_dump_json()
        assert loaded[key].generation_tps == 10.0
        assert loaded[key].success is True

        key_fail = SearchConfig(threads=8).model_dump_json()
        assert loaded[key_fail].success is False
        assert loaded[key_fail].raw_output == "oom"
    finally:
        session.close()
        engine.dispose()
        os.unlink(db_path)


def test_save_trial_cache_upserts():
    session, db_path, engine = _db_session()
    try:
        key = SearchConfig(threads=4).model_dump_json()

        save_trial_cache(
            session,
            "model.gguf",
            {key: BenchmarkResult(generation_tps=10.0, success=True)},
        )
        save_trial_cache(
            session,
            "model.gguf",
            {key: BenchmarkResult(generation_tps=99.0, success=True)},
        )

        loaded = load_trial_cache(session, "model.gguf")
        assert len(loaded) == 1
        assert loaded[key].generation_tps == 99.0
    finally:
        session.close()
        engine.dispose()
        os.unlink(db_path)


def test_save_launch_profile():
    session, db_path, engine = _db_session()
    try:
        profile = LaunchProfile(
            name="test_profile",
            args=["--flash-attn", "-t", "8"],
            model_path="/models/test.gguf",
            hardware="test_cpu",
            score=95.0,
        )
        save_launch_profile(session, profile)

        result = (
            session.query(LaunchProfileModel)
            .filter(LaunchProfileModel.name == "test_profile")
            .first()
        )
        assert result is not None
        assert result.score == 95.0
    finally:
        session.close()
        engine.dispose()
        os.unlink(db_path)
