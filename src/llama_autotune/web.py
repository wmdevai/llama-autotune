"""Integrated Web UI for llama-autotune."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .benchmark import find_llama_bench, find_llama_binary, run_benchmark
from .hardware import detect_hardware
from .model_inspector import inspect_model
from .models import SearchConfig


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web_static"
TEMPLATE_DIR = BASE_DIR / "web_templates"


app = FastAPI(
    title="llama-autotune",
)


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
