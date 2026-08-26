"""Integrated Web UI for llama-autotune."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .benchmark import find_llama_bench, find_llama_binary, run_benchmark
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


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = TEMPLATE_DIR / "index.html"

    if not index_path.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Web UI template not found: {index_path}",
        )

    return HTMLResponse(
        index_path.read_text(encoding="utf-8")
    )


@app.get("/api/dashboard")
def dashboard() -> dict:
    hw = detect_hardware()

    llama_bench = find_llama_bench()
    llama_server = find_llama_binary("llama-server")

    return {
        "hardware": {
            "cpu_name": hw.cpu_name,
            "physical_cores": hw.physical_cores,
            "logical_cores": hw.logical_cores,
            "ram_gb": hw.ram_gb,
            "gpu_count": hw.gpu_count,
            "gpu_models": hw.gpu_models,
            "vram_per_gpu": hw.vram_per_gpu,
            "backend": hw.backend.value,
        },
        "llama_cpp": {
            "llama_server": str(llama_server),
            "llama_bench": str(llama_bench),
            "version": get_llama_version(llama_server),
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
        objective = OptimizeObjective(request.objective)
    except ValueError as exc:
        valid = [item.value for item in OptimizeObjective]

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid optimization objective. "
                f"Choose from: {', '.join(valid)}"
            ),
        ) from exc

    job_id = uuid.uuid4().hex
    sink: queue.Queue = queue.Queue()
    _jobs[job_id] = {"queue": sink}

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
    best_result = optimizer._evaluate(best_config)

    baseline_result = optimizer._baseline_result

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


def run_web(
    host: str = "127.0.0.1",
    port: int = 8766,
) -> None:
    """Launch the integrated Web UI."""

    uvicorn.run(
        app,
        host=host,
        port=port,
    )
