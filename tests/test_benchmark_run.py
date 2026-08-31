"""Tests for run_benchmark and _watch_process with mocked subprocess/psutil."""

import subprocess
from types import SimpleNamespace

from llama_autotune import benchmark as bm
from llama_autotune.models import SearchConfig


class FakeProc:
    """Minimal subprocess.Popen stand-in."""

    def __init__(self, returncode=0, stdout="", stderr="", poll_sequence=None):
        self.returncode = returncode
        self.pid = 12345
        self._stdout = stdout
        self._stderr = stderr
        self.args = ["llama-bench"]
        self._killed = False
        self._poll_calls = 0
        self._poll_sequence = poll_sequence or [returncode]

    def poll(self):
        index = min(self._poll_calls, len(self._poll_sequence) - 1)
        self._poll_calls += 1
        return self._poll_sequence[index]

    def communicate(self):
        return self._stdout, self._stderr

    def kill(self):
        self._killed = True


def _patch_run_benchmark_deps(monkeypatch, proc, peak_rss=0.0, peak_vram=0.0):
    monkeypatch.setattr(
        bm,
        "find_llama_bench",
        lambda: "/usr/bin/llama-bench",
    )
    monkeypatch.setattr(
        bm.subprocess,
        "Popen",
        lambda *args, **kwargs: proc,
    )
    monkeypatch.setattr(
        bm,
        "_watch_process",
        lambda p, timeout: (peak_rss, peak_vram),
    )


# ── run_benchmark ────────────────────────────────────────────────────


def test_run_benchmark_success(monkeypatch):
    stdout = '[{"n_prompt": 64, "n_gen": 0, "avg_ts": 1000.0},'
    stdout += ' {"n_prompt": 0, "n_gen": 32, "avg_ts": 50.0}]'
    proc = FakeProc(returncode=0, stdout=stdout)
    _patch_run_benchmark_deps(monkeypatch, proc, peak_rss=512.0)

    result = bm.run_benchmark(
        "/models/test.gguf",
        SearchConfig(threads=8),
        repetitions=3,
        timeout=300,
    )

    assert result.success is True
    assert result.prompt_tps == 1000.0
    assert result.generation_tps == 50.0
    assert result.memory_usage == 512.0  # peak RSS fallback
    assert result.vram_usage == 0.0  # mocked watcher returned 0


def test_run_benchmark_records_vram_usage(monkeypatch):
    proc = FakeProc(returncode=0, stdout='{"avg_ts": 50.0, "n_gen": 8}')
    _patch_run_benchmark_deps(monkeypatch, proc, peak_rss=512.0, peak_vram=4096.0)

    result = bm.run_benchmark("/models/test.gguf")

    assert result.success is True
    assert result.vram_usage == 4096.0


def test_run_benchmark_nonzero_returncode(monkeypatch):
    proc = FakeProc(returncode=1, stdout="", stderr="model load failed")
    _patch_run_benchmark_deps(monkeypatch, proc)

    result = bm.run_benchmark("/models/test.gguf")

    assert result.success is False
    assert "STDERR: model load failed" in result.raw_output


def test_run_benchmark_invalid_json(monkeypatch):
    proc = FakeProc(returncode=0, stdout="not json at all")
    _patch_run_benchmark_deps(monkeypatch, proc)

    result = bm.run_benchmark("/models/test.gguf")

    assert result.success is False


def test_run_benchmark_timeout(monkeypatch):
    proc = FakeProc(returncode=0, stdout="")

    monkeypatch.setattr(
        bm,
        "find_llama_bench",
        lambda: "/usr/bin/llama-bench",
    )
    monkeypatch.setattr(
        bm.subprocess,
        "Popen",
        lambda *args, **kwargs: proc,
    )

    def raise_timeout(p, timeout):
        raise subprocess.TimeoutExpired(cmd=["llama-bench"], timeout=timeout)

    monkeypatch.setattr(bm, "_watch_process", raise_timeout)

    result = bm.run_benchmark("/models/test.gguf")

    assert result.success is False
    assert result.raw_output == "[TIMEOUT]"


def test_run_benchmark_file_not_found(monkeypatch):
    monkeypatch.setattr(
        bm,
        "find_llama_bench",
        lambda: "/usr/bin/llama-bench",
    )

    def raise_fnf(*args, **kwargs):
        raise FileNotFoundError("llama-bench")

    monkeypatch.setattr(bm.subprocess, "Popen", raise_fnf)

    result = bm.run_benchmark("/models/test.gguf")

    assert result.success is False
    assert "llama-bench not found" in result.raw_output


def test_run_benchmark_generic_error(monkeypatch):
    monkeypatch.setattr(
        bm,
        "find_llama_bench",
        lambda: "/usr/bin/llama-bench",
    )

    def raise_runtime(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(bm.subprocess, "Popen", raise_runtime)

    result = bm.run_benchmark("/models/test.gguf")

    assert result.success is False
    assert "[ERROR] boom" in result.raw_output


# ── _watch_process ───────────────────────────────────────────────────


def test_watch_process_exits_immediately(monkeypatch):
    proc = FakeProc(returncode=0)

    class FakePsutil:
        Error = Exception

        @staticmethod
        def Process(pid):
            raise AssertionError("should not be called when already exited")

    monkeypatch.setattr(bm, "psutil", FakePsutil)

    assert bm._watch_process(proc, timeout=10) == (0.0, 0.0)


def test_watch_process_tracks_peak_rss(monkeypatch):
    proc = FakeProc(poll_sequence=[None, 0])

    class FakeChild:
        @staticmethod
        def memory_info():
            return SimpleNamespace(rss=512 * 1024)  # 0.5 MiB child

    class FakeProcHandle:
        @staticmethod
        def memory_info():
            return SimpleNamespace(rss=1024 * 1024)  # 1 MiB parent

        @staticmethod
        def children(recursive=True):
            return [FakeChild()]

    class FakePsutil:
        Error = Exception

        @staticmethod
        def Process(pid):
            return FakeProcHandle()

    monkeypatch.setattr(bm, "psutil", FakePsutil)
    monkeypatch.setattr(bm.time, "sleep", lambda _: None)
    monkeypatch.setattr(bm, "_gpu_vram_used_mb", lambda: 0.0)

    # 1 MiB parent + 0.5 MiB child = 1.5 MiB -> 1.5 MB
    assert bm._watch_process(proc, timeout=10) == (1.5, 0.0)


def test_watch_process_kills_on_timeout(monkeypatch):
    proc = FakeProc(poll_sequence=[None])  # never exits

    class FakePsutil:
        Error = Exception

        @staticmethod
        def Process(pid):
            raise FakePsutil.Error()

    monkeypatch.setattr(bm, "psutil", FakePsutil)

    times = iter([0.0, 100.0])

    monkeypatch.setattr(bm.time, "time", lambda: next(times))
    monkeypatch.setattr(bm.time, "sleep", lambda _: None)

    try:
        bm._watch_process(proc, timeout=10)
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError("expected TimeoutExpired")

    assert proc._killed is True
