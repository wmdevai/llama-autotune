"""Integrated Web UI for llama-autotune."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import calibration, presets, storage
from .benchmark import (
    _gpu_vram_used_mb,
    find_llama_bench,
    find_llama_binary,
    gpu_sensors,
    run_benchmark,
)
from .hardware import detect_hardware
from .model_inspector import inspect_model
from .models import OptimizeObjective, SearchConfig
from .optimizer import Optimizer

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web_static"
TEMPLATE_DIR = BASE_DIR / "web_templates"
MODEL_DIR = Path.home() / "Modelli" / "llama.cpp"


app = FastAPI(
    title="llama-autotune",
)


class _QueueLogHandler(logging.Handler):
    """A logging handler that forwards records into a thread-safe queue."""

    def __init__(self, sink: queue.Queue):
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink.put(
                {
                    "type": "log",
                    "level": record.levelname.lower(),
                    "message": self.format(record),
                }
            )
        except Exception:
            pass


# In-memory registry of running/just-finished optimization jobs.
_jobs: dict[str, dict] = {}

# Drop jobs whose SSE stream was never consumed after this many seconds.
_JOB_TTL_SECONDS = 24 * 3600


def _prune_stale_jobs() -> None:
    """Remove optimization jobs nobody consumed within the TTL window."""
    now = time.time()
    for job_id in [
        jid
        for jid, job in _jobs.items()
        if now - job.get("created", now) > _JOB_TTL_SECONDS
    ]:
        _jobs.pop(job_id, None)


# Liveness heartbeat used by the launcher mode: the web page pings
# /api/heartbeat, and the launcher stops the server when the pings stop.
_heartbeat_lock = threading.Lock()
_last_heartbeat: float = 0.0
_HEARTBEAT_TIMEOUT_SECONDS = 30.0


def _touch_heartbeat() -> None:
    global _last_heartbeat
    with _heartbeat_lock:
        _last_heartbeat = time.time()


def _heartbeat_age() -> float:
    with _heartbeat_lock:
        if _last_heartbeat == 0.0:
            return 0.0
        return time.time() - _last_heartbeat


class InspectRequest(BaseModel):
    model_path: str


class BenchmarkRequest(BaseModel):
    model_path: str
    threads: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    batch_size: Optional[int] = None
    ubatch_size: Optional[int] = None
    flash_attn: Optional[bool] = None
    cache_type_k: Optional[str] = None
    cache_type_v: Optional[str] = None
    repetitions: int = 1


class OptimizeRequest(BaseModel):
    model_path: str
    objective: str = "balanced"
    trials_b: int = 12
    trials_c: int = 20


class CalibrateRequest(BaseModel):
    model_path: str


class DeleteRequest(BaseModel):
    key: str


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = TEMPLATE_DIR / "index.html"

    if not index_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Web UI template not found: {index_path}",
        )

    return HTMLResponse(
        index_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/heartbeat")
def heartbeat() -> dict:
    """Record a liveness ping from the web page (launcher mode)."""
    _touch_heartbeat()
    return {"ok": True}


@app.get("/api/dashboard")
def dashboard() -> dict:
    hw = detect_hardware()

    llama_bench = find_llama_bench()
    llama_server = find_llama_binary("llama-server")

    vram_used_mb = _gpu_vram_used_mb()
    vram_total_gb = sum(hw.vram_per_gpu)
    vram_used_gb = round(vram_used_mb / 1024.0, 1)

    ram = psutil.virtual_memory()
    ram_used_gb = round((ram.total - ram.available) / (1024**3), 1)
    ram_total_gb = round(ram.total / (1024**3), 1)

    models_count = 0
    models_size_gb = 0.0
    if MODEL_DIR.is_dir():
        for path in MODEL_DIR.rglob("*.gguf"):
            if path.is_file():
                models_count += 1
                models_size_gb += path.stat().st_size / (1024**3)

    disk = shutil.disk_usage(
        MODEL_DIR if MODEL_DIR.is_dir() else Path.home()
    )

    from . import __version__

    sensors = gpu_sensors()

    return {
        "hardware": {
            "cpu_name": hw.cpu_name,
            "physical_cores": hw.physical_cores,
            "logical_cores": hw.logical_cores,
            "ram_gb": hw.ram_gb,
            "ram_used_gb": ram_used_gb,
            "ram_total_gb": ram_total_gb,
            "gpu_count": hw.gpu_count,
            "gpu_models": hw.gpu_models,
            "vram_per_gpu": hw.vram_per_gpu,
            "vram_total_gb": round(vram_total_gb, 1),
            "vram_used_gb": vram_used_gb,
            "vram_free_gb": round(vram_total_gb - vram_used_gb, 1),
            "gpu_temperature_c": sensors.get("temperature_c"),
            "gpu_utilization_pct": sensors.get("utilization_pct"),
            "backend": hw.backend.value,
        },
        "llama_cpp": {
            "llama_server": str(llama_server),
            "llama_bench": str(llama_bench),
            "version": get_llama_version(llama_server),
        },
        "system": {
            "version": __version__,
            "models_count": models_count,
            "models_size_gb": round(models_size_gb, 1),
            "calibrations_count": len(calibration.list_calibrations()),
            "disk_free_gb": round(disk.free / (1024**3), 1),
            "models_dir": str(MODEL_DIR),
        },
    }


@app.get("/api/models")
def list_models() -> dict:
    if not MODEL_DIR.is_dir():
        return {
            "directory": str(MODEL_DIR),
            "models": [],
        }

    models = sorted(
        path.resolve()
        for path in MODEL_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".gguf"
    )

    return {
        "directory": str(MODEL_DIR),
        "models": [
            {
                "name": path.name,
                "path": str(path),
                "relative_path": str(
                    path.relative_to(MODEL_DIR)
                ),
            }
            for path in models
        ],
    }


@app.post("/api/inspect")
def inspect(request: InspectRequest) -> dict:
    model_path = request.model_path.strip()

    if not model_path:
        raise HTTPException(
            status_code=400,
            detail="Model path is required.",
        )

    if not os.path.isfile(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {model_path}",
        )

    try:
        model = inspect_model(model_path)
        return model.model_dump()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to inspect model: {exc}",
        ) from exc


@app.post("/api/benchmark")
def benchmark(request: BenchmarkRequest) -> dict:
    model_path = request.model_path.strip()

    if not model_path:
        raise HTTPException(
            status_code=400,
            detail="Model path is required.",
        )

    if not os.path.isfile(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {model_path}",
        )

    if request.repetitions < 1:
        raise HTTPException(
            status_code=400,
            detail="Repetitions must be at least 1.",
        )

    config = SearchConfig(
        threads=request.threads,
        n_gpu_layers=request.n_gpu_layers,
        batch_size=request.batch_size,
        ubatch_size=request.ubatch_size,
        flash_attn=request.flash_attn,
        cache_type_k=request.cache_type_k,
        cache_type_v=request.cache_type_v,
    )

    result = run_benchmark(
        model_path,
        config,
        repetitions=request.repetitions,
    )

    return {
        "result": result.model_dump(),
        "config": config.model_dump(),
    }


@app.get("/api/calibrations")
def calibrations() -> dict:
    """Return the stored per-quantization VRAM calibrations."""
    return {"calibrations": calibration.list_calibrations()}


@app.post("/api/calibrate")
def calibrate(request: CalibrateRequest) -> dict:
    """Measure a model's real VRAM footprint and store the overhead factor.

    Runs one controlled benchmark with full offload and a minimal context so
    the KV cache is negligible and the measured VRAM is essentially the
    model weights. The result is stored per quantization and is then used by
    the VRAM estimator during optimization.
    """
    model_path = request.model_path.strip()

    if not model_path:
        raise HTTPException(
            status_code=400,
            detail="Model path is required.",
        )

    if not os.path.isfile(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {model_path}",
        )

    model = inspect_model(model_path)
    hw = detect_hardware()

    config = SearchConfig(
        threads=hw.physical_cores or 1,
        n_gpu_layers=999,
        ctx_size=4096,
        flash_attn=True,
    )

    result = run_benchmark(
        model_path,
        config,
        repetitions=1,
        timeout=600,
        n_prompt=16,
        n_gen=8,
    )

    if not result.success:
        raise HTTPException(
            status_code=500,
            detail=(
                "Calibration benchmark failed. The model may be too large "
                "for full GPU offload. "
                f"llama-bench output: {result.raw_output[:500]}"
            ),
        )

    if result.vram_usage <= 0.0:
        raise HTTPException(
            status_code=500,
            detail=(
                "No GPU VRAM was measured. Ensure nvidia-smi or rocm-smi is "
                "available and a GPU is present."
            ),
        )

    file_size_mb = model.file_size_gb * 1024.0

    record = calibration.save_calibration(
        model.quantization,
        result.vram_usage,
        file_size_mb,
        model_path,
    )

    return {
        "quantization": model.quantization,
        "file_size_mb": record["file_size_mb"],
        "measured_vram_mb": record["measured_vram_mb"],
        "measured_ratio": record["measured_ratio"],
        "overhead": record["overhead"],
        "model_path": record["model_path"],
        "timestamp": record["timestamp"],
    }


@app.get("/api/presets")
def presets_view() -> dict:
    """Return the current and recommended presets.ini."""
    current = presets.current_content()
    recommended = presets.recommended_content()

    return {
        "current": {
            "path": str(presets.CURRENT_PRESETS_PATH),
            "content": current,
        },
        "recommended": {
            "path": str(presets.RECOMMENDED_PRESETS_PATH),
            "content": recommended,
        },
        "differ": bool(current) and current != recommended,
    }


@app.post("/api/presets/apply")
def presets_apply() -> dict:
    """Overwrite the current presets.ini with the recommended one."""
    try:
        return presets.apply_recommended()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/storage")
def storage_view() -> dict:
    """Return generated files and KV-cache slots with their sizes."""
    return storage.list_storage()


@app.post("/api/storage/delete")
def storage_delete(request: DeleteRequest) -> dict:
    """Delete a generated file, directory, or KV-cache slot."""
    try:
        return storage.delete_item(request.key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/optimize")
def optimize(request: OptimizeRequest) -> dict:
    model_path = request.model_path.strip()

    if not model_path:
        raise HTTPException(
            status_code=400,
            detail="Model path is required.",
        )

    if not os.path.isfile(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {model_path}",
        )

    if request.trials_b < 1:
        raise HTTPException(
            status_code=400,
            detail="Stage B trials must be at least 1.",
        )

    if request.trials_c < 1:
        raise HTTPException(
            status_code=400,
            detail="Stage C trials must be at least 1.",
        )

    try:
        OptimizeObjective(request.objective)
    except ValueError as exc:
        valid = [item.value for item in OptimizeObjective]

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid optimization objective. "
                f"Choose from: {', '.join(valid)}"
            ),
        ) from exc

    _prune_stale_jobs()

    job_id = uuid.uuid4().hex
    sink: queue.Queue = queue.Queue()
    _jobs[job_id] = {"queue": sink, "created": time.time()}

    def worker() -> None:
        handler = _QueueLogHandler(sink)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = logging.getLogger("llama_autotune.optimizer")
        logger.addHandler(handler)

        result = None
        error = None

        try:
            result = _run_optimization(request)
        except Exception as exc:
            error = str(exc)
        finally:
            logger.removeHandler(handler)
            sink.put(
                {
                    "type": "done",
                    "result": result,
                    "error": error,
                }
            )

    threading.Thread(target=worker, daemon=True).start()

    return {"job_id": job_id}


def _run_optimization(request: OptimizeRequest) -> dict:
    """Run the full optimization and build the result payload."""
    objective = OptimizeObjective(request.objective)

    optimizer = Optimizer(
        model_path=request.model_path,
        objective=objective,
        n_trials_stage_b=request.trials_b,
        n_trials_stage_c=request.trials_c,
    )

    best_config = optimizer.run()

    # Reuse the exact benchmark result already evaluated by the optimizer.
    # This avoids running the same configuration a second time and keeps
    # the Baseline and Best Result values consistent with the optimization.
    best_result = optimizer.validate_full_context(best_config)

    baseline_result = optimizer._baseline_result

    try:
        presets.generate_recommended(best_config, request.model_path)
        recommended_presets = True
    except Exception:
        recommended_presets = False

    return {
        "objective": objective.value,
        "best_config": best_config.model_dump(),
        "best_score": optimizer._best_score,
        "total_evaluations": optimizer._total_evals,
        "baseline_result": (
            baseline_result.model_dump()
            if baseline_result is not None
            else None
        ),
        "final_result": best_result.model_dump(),
        "recommended_presets": recommended_presets,
    }


@app.get("/api/optimize/events/{job_id}")
async def optimize_events(job_id: str) -> StreamingResponse:
    """Stream optimization progress as Server-Sent Events."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown optimization job.",
        )

    sink: queue.Queue = job["queue"]

    async def stream():
        try:
            while True:
                try:
                    event = sink.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.25)
                    continue

                yield f"data: {json.dumps(event)}\n\n"

                if event.get("type") == "done":
                    break
        finally:
            _jobs.pop(job_id, None)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
    )


def get_llama_version(
    llama_server: str | Path,
) -> str:
    """Return llama-server version if available."""

    try:
        result = subprocess.run(
            [str(llama_server), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        output = result.stdout.strip()

        if output:
            return output

        return result.stderr.strip() or "Unknown"

    except Exception:
        return "Unknown"


app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)


def _notify(message: str) -> None:
    """Show a desktop notification (best-effort, Linux ``notify-send``)."""
    exe = shutil.which("notify-send")
    if not exe:
        return
    try:
        subprocess.Popen([exe, "llama-autotune", message])
    except Exception:
        pass


def _port_is_free(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _find_free_port(host: str, preferred: int) -> int:
    """Return *preferred* if available, otherwise a free ephemeral port."""
    if _port_is_free(host, preferred):
        return preferred

    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


_CHROMIUM_BROWSERS = ("chrom", "brave", "edge", "vivaldi", "opera")


def _open_browser(url: str, browser: str | None) -> None:
    """Open *url* in the requested browser, falling back to the default.

    Chromium-based browsers open in an "app window" (no address bar or
    toolbars), so the Web UI looks like a native application.
    """
    if browser and browser not in ("", "default"):
        exe = shutil.which(browser) or shutil.which(f"{browser}-browser")
        if exe:
            lower = browser.lower()
            if any(name in lower for name in _CHROMIUM_BROWSERS):
                subprocess.Popen([exe, f"--app={url}"])
            else:
                subprocess.Popen([exe, "--new-window", url])
            return

    import webbrowser

    webbrowser.open(url)


def run_web(
    host: str = "127.0.0.1",
    port: int = 8766,
    browser: str | None = None,
) -> None:
    """Launch the integrated Web UI.

    When *browser* is provided the page is opened in that browser and the
    server shuts down automatically once the page is closed (heartbeat
    timeout). Otherwise the server runs in the foreground as before.
    """
    if browser is None:
        uvicorn.run(app, host=host, port=port)
        return

    logger = logging.getLogger(__name__)

    actual_port = _find_free_port(host, port)
    if actual_port != port:
        logger.info("Port %s in use — using %s.", port, actual_port)

    url = f"http://{host}:{actual_port}"

    config = uvicorn.Config(app, host=host, port=actual_port)
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for the server to bind before opening the browser.
    while not server.started and thread.is_alive():
        time.sleep(0.1)

    if not server.started:
        logger.error("Web server failed to start.")
        _notify("Impossibile avviare la Web UI (porta occupata o errore).")
        thread.join(timeout=5.0)
        return

    _open_browser(url, browser)
    _touch_heartbeat()  # start the liveness clock once the page is loading
    _notify(f"Web UI avviata su {url}")

    logger.info(
        "Launcher mode: close the browser page to stop the server "
        f"({_HEARTBEAT_TIMEOUT_SECONDS:.0f}s timeout)"
    )

    try:
        while thread.is_alive():
            if _heartbeat_age() > _HEARTBEAT_TIMEOUT_SECONDS:
                logger.info("Heartbeat lost — shutting down.")
                server.should_exit = True
                break
            time.sleep(1.0)
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)

    _notify("Web UI arrestata.")
