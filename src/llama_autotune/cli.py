"""CLI interface for llama-autotune.

Provides commands for inspecting hardware/model info, running benchmarks,
auto-optimizing configuration via search, launching llama-server with tuned
settings, exporting/importing launch profiles, and launching the integrated
Web UI.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from .benchmark import find_llama_binary, run_benchmark
from .database import (
    load_trial_cache,
    save_benchmark,
    save_trial_cache,
    session_scope,
)
from .hardware import detect_hardware
from .heuristics import generate_initial_config
from .model_inspector import inspect_model
from .models import (
    BenchmarkEntry,
    OptimizeObjective,
    SearchConfig,
)
from .optimizer import Optimizer
from .profiles import create_profile, export_profile, import_profile
from .web import run_web

EXAMPLES = (
    "[bold]Examples:[/bold]\n\n"
    "  llama-autotune inspect model.gguf\n\n"
    "  llama-autotune benchmark model.gguf -r 3\n\n"
    "  llama-autotune search model.gguf "
    "--objective max_generation_tps --profile best.json\n\n"
    "  llama-autotune launch model.gguf\n\n"
    "  llama-autotune web\n\n"
    '  llama-autotune export my_profile.json '
    '--model model.gguf --hardware "RTX 4090"\n\n'
    "  llama-autotune import my_profile.json"
)


console = Console()

app = typer.Typer(
    name="llama-autotune",
    help="Benchmarking and optimization tool for llama.cpp",
    epilog=EXAMPLES,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)


def _require_model_file(path: str) -> None:
    if not os.path.isfile(path):
        console.print(
            f"[red]Error: model file not found: {path}[/red]"
        )
        raise typer.Exit(code=1)


@app.command(help="Inspect hardware and model metadata.")
def inspect(
    model: str = typer.Argument(
        ...,
        help="Path to GGUF model file",
    ),
):
    """Inspect hardware capabilities and model metadata.

    Detects system hardware (CPU, GPU, RAM) and reads model properties
    (architecture, parameter count, quantization, etc.) from a GGUF file,
    displaying both in formatted tables.
    """

    _require_model_file(model)

    hw = detect_hardware()

    try:
        model_info = inspect_model(model)
    except (FileNotFoundError, PermissionError) as e:
        console.print(
            f"[red]Error reading model: {e}[/red]"
        )
        raise typer.Exit(code=1)

    table = Table(
        title="Hardware",
        box=box.ROUNDED,
    )

    table.add_column(
        "Property",
        style="cyan",
    )

    table.add_column(
        "Value",
        style="green",
    )

    table.add_row(
        "CPU",
        hw.cpu_name,
    )

    table.add_row(
        "Cores",
        f"{hw.physical_cores}P / {hw.logical_cores}L",
    )

    table.add_row(
        "RAM",
        f"{hw.ram_gb} GB",
    )

    table.add_row(
        "GPU",
        ", ".join(hw.gpu_models) or "None",
    )

    table.add_row(
        "Backend",
        hw.backend.value,
    )

    table.add_row(
        "VRAM",
        ", ".join(
            f"{v} GB"
            for v in hw.vram_per_gpu
        ) or "N/A",
    )

    console.print(table)

    table2 = Table(
        title="Model",
        box=box.ROUNDED,
    )

    table2.add_column(
        "Property",
        style="cyan",
    )

    table2.add_column(
        "Value",
        style="green",
    )

    table2.add_row(
        "Path",
        model_info.path,
    )

    table2.add_row(
        "Architecture",
        model_info.architecture,
    )

    table2.add_row(
        "Parameters",
        f"{model_info.parameters:,}",
    )

    table2.add_row(
        "Quantization",
        model_info.quantization,
    )

    table2.add_row(
        "Layers",
        str(model_info.n_layers),
    )

    table2.add_row(
        "Heads",
        str(model_info.n_heads),
    )

    if model_info.n_kv_heads and model_info.n_kv_heads != model_info.n_heads:
        table2.add_row(
            "KV Heads",
            str(model_info.n_kv_heads),
        )

    table2.add_row(
        "Context Length",
        str(model_info.training_context),
    )

    table2.add_row(
        "MoE",
        "Yes" if model_info.is_moe else "No",
    )

    if model_info.is_moe:
        table2.add_row(
            "Active Params",
            f"{model_info.active_parameters:,}",
        )

    table2.add_row(
        "File Size",
        f"{model_info.file_size_gb} GB",
    )

    console.print(table2)


@app.command(help="Run a benchmark with optional custom parameters.")
def benchmark(
    model: str = typer.Argument(
        ...,
        help="Path to GGUF model file",
    ),
    threads: Optional[int] = typer.Option(
        None,
        "-t",
        "--threads",
        help="CPU threads",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "-b",
        "--batch-size",
        help="Batch size",
    ),
    ubatch_size: Optional[int] = typer.Option(
        None,
        "-ub",
        "--ubatch-size",
        help="Ubatch size",
    ),
    n_gpu_layers: Optional[int] = typer.Option(
        None,
        "-ngl",
        "--n-gpu-layers",
        help="GPU layers",
    ),
    flash_attn: Optional[bool] = typer.Option(
        None,
        "--flash-attn/--no-flash-attn",
        help="Flash attention",
    ),
    ctx_size: Optional[int] = typer.Option(
        None,
        "-c",
        "--ctx-size",
        help="Context size",
    ),
    repetitions: int = typer.Option(
        3,
        "-r",
        "--repetitions",
        help="Benchmark repetitions",
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "-o",
        "--output",
        help="Save result to file",
    ),
):
    """Run a benchmark with optional custom parameters."""

    _require_model_file(model)

    hw = detect_hardware()
    model_info = inspect_model(model)

    if any(
        x is not None
        for x in [
            threads,
            batch_size,
            ubatch_size,
            n_gpu_layers,
            flash_attn,
            ctx_size,
        ]
    ):
        config = SearchConfig(
            threads=threads,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            n_gpu_layers=n_gpu_layers,
            flash_attn=flash_attn,
            ctx_size=ctx_size,
        )
    else:
        config = generate_initial_config(
            hw,
            model_info,
        )

    console.print(
        "[bold]Running benchmark...[/bold]"
    )

    result = run_benchmark(
        model,
        config,
        repetitions=repetitions,
    )

    table = Table(
        title="Benchmark Results",
        box=box.ROUNDED,
    )

    table.add_column(
        "Metric",
        style="cyan",
    )

    table.add_column(
        "Value",
        style="green",
    )

    table.add_row(
        "Prompt TPS",
        f"{result.prompt_tps:.2f}",
    )

    table.add_row(
        "Generation TPS",
        f"{result.generation_tps:.2f}",
    )

    table.add_row(
        "Startup Time",
        f"{result.startup_time:.2f}s",
    )

    table.add_row(
        "Memory",
        f"{result.memory_usage:.1f} MB",
    )

    table.add_row(
        "Success",
        "Yes" if result.success else "No",
    )

    console.print(table)

    entry = BenchmarkEntry(
        hardware_id=hw.cpu_name,
        model_id=model_info.architecture,
        config=config,
        result=result,
    )

    with session_scope() as session:
        save_benchmark(
            session,
            entry,
        )

    if output_file:
        with open(
            output_file,
            "w",
        ) as f:
            json.dump(
                result.model_dump(),
                f,
                indent=2,
            )

        console.print(
            f"[green]Saved to {output_file}[/green]"
        )

    if not result.success:
        raise typer.Exit(code=1)


@app.command(help="Auto-optimize config via 3-stage search.")
def search(
    model: str = typer.Argument(
        ...,
        help="Path to GGUF model file",
    ),
    objective: str = typer.Option(
        "balanced",
        "--objective",
        "-O",
        help="Optimization objective",
    ),
    trials_b: int = typer.Option(
        12,
        "--trials-b",
        help="Stage B trials",
    ),
    trials_c: int = typer.Option(
        20,
        "--trials-c",
        help="Stage C trials",
    ),
    slow: bool = typer.Option(
        False,
        "--slow",
        help="Continue optimization with a reduced workload on very slow hardware",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Reuse cached benchmarks from a previous run for the same model",
    ),
    output_profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Save best profile to file",
    ),
):
    """Auto-optimize configuration via 3-stage search."""

    _require_model_file(model)

    try:
        obj = OptimizeObjective(objective)
    except ValueError:
        valid = [
            o.value
            for o in OptimizeObjective
        ]

        console.print(
            f"[red]Invalid objective. "
            f"Choose from: {', '.join(valid)}[/red]"
        )

        raise typer.Exit(code=1)

    cache = None

    if resume:
        with session_scope() as session:
            cache = load_trial_cache(
                session,
                model,
            )

        console.print(
            f"[green]Loaded {len(cache)} cached benchmarks "
            f"from a previous run[/green]"
        )

    opt = Optimizer(
        model_path=model,
        objective=obj,
        n_trials_stage_b=trials_b,
        n_trials_stage_c=trials_c,
        slow=slow,
        cache=cache,
    )

    console.print(
        "[bold]Phase 1: Hardware Detection[/bold]"
    )

    hw_table = Table(
        box=box.ROUNDED,
    )

    hw_table.add_column(
        "Property",
        style="cyan",
    )

    hw_table.add_column(
        "Value",
        style="green",
    )

    hw_table.add_row(
        "CPU",
        opt.hw.cpu_name,
    )

    hw_table.add_row(
        "Cores",
        f"{opt.hw.physical_cores}P / "
        f"{opt.hw.logical_cores}L",
    )

    hw_table.add_row(
        "RAM",
        f"{opt.hw.ram_gb} GB",
    )

    hw_table.add_row(
        "Backend",
        opt.hw.backend.value,
    )

    console.print(hw_table)

    console.print(
        "[bold]Phase 2: Model Inspection[/bold]"
    )

    mi = opt.model

    model_table = Table(
        box=box.ROUNDED,
    )

    model_table.add_column(
        "Property",
        style="cyan",
    )

    model_table.add_column(
        "Value",
        style="green",
    )

    model_table.add_row(
        "Architecture",
        mi.architecture,
    )

    model_table.add_row(
        "Parameters",
        f"{mi.parameters:,}",
    )

    model_table.add_row(
        "Quantization",
        mi.quantization,
    )

    model_table.add_row(
        "MoE",
        "Yes" if mi.is_moe else "No",
    )

    console.print(model_table)

    console.print(
        "[bold]Running optimization...[/bold]"
    )

    best_config = opt.run()

    result = opt.validate_full_context(best_config)

    table = Table(
        title=f"Best Config ({opt.objective.value})",
        box=box.ROUNDED,
    )

    table.add_column(
        "Parameter",
        style="cyan",
    )

    table.add_column(
        "Value",
        style="green",
    )

    table.add_row(
        "Prompt TPS",
        f"{result.prompt_tps:.2f}"
        if result.success
        else "N/A",
    )

    table.add_row(
        "Generation TPS",
        f"{result.generation_tps:.2f}"
        if result.success
        else "N/A",
    )

    table.add_row(
        "Score",
        f"{opt.best_score:.2f}",
    )

    table.add_row(
        "Evaluations",
        str(opt.total_evals),
    )

    if best_config:
        for field, val in best_config.model_dump(
            exclude_none=True
        ).items():
            table.add_row(
                field.replace("_", "-"),
                str(val),
            )

    console.print(table)

    if output_profile:
        hw_name = opt.hw.cpu_name.replace(
            " ",
            "_",
        )

        model_name = mi.architecture.replace(
            "/",
            "_",
        )

        profile = create_profile(
            name=(
                f"{hw_name}_"
                f"{model_name}_"
                f"{opt.objective.value}"
            ),
            args=(
                best_config.to_llama_args()
                if best_config
                else []
            ),
            model_path=model,
            hardware=opt.hw.cpu_name,
            score=opt.best_score,
        )

        path = export_profile(
            profile,
            output_profile,
        )

        console.print(
            f"[green]Profile saved to {path}[/green]"
        )

    if opt.total_evals > 0:
        entry = BenchmarkEntry(
            hardware_id=opt.hw.cpu_name,
            model_id=mi.architecture,
            config=best_config or SearchConfig(),
            result=result,
            objective=obj,
        )

        with session_scope() as session:
            save_benchmark(
                session,
                entry,
            )

            save_trial_cache(
                session,
                model,
                opt._cache,
            )


@app.command(help="Start llama-server with optimal config.")
def launch(
    model: str = typer.Argument(
        ...,
        help="Path to GGUF model file",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Path to profile JSON",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Server host",
    ),
    port: int = typer.Option(
        8080,
        "--port",
        help="Server port",
    ),
):
    """Start llama-server with optimal configuration."""

    _require_model_file(model)

    if profile:
        if not os.path.isfile(profile):
            console.print(
                f"[red]Error: profile file not found: "
                f"{profile}[/red]"
            )
            raise typer.Exit(code=1)

        prof = import_profile(profile)
        args = prof.args

    else:
        hw = detect_hardware()
        model_info = inspect_model(model)

        config = generate_initial_config(
            hw,
            model_info,
        )

        args = config.to_llama_args()

    server_exe = find_llama_binary(
        "llama-server"
    )

    cmd = [
        server_exe,
        "-m",
        model,
        "--host",
        host,
        "--port",
        str(port),
        *args,
    ]

    cmd_str = " ".join(cmd)

    console.print(
        "[bold]Starting llama-server:[/bold]"
    )

    console.print(
        f"  {cmd_str}"
    )

    if os.name == "nt":
        import subprocess

        try:
            proc = subprocess.run(cmd)

        except FileNotFoundError:
            console.print(
                f"[red]Error: llama-server not found at: "
                f"{server_exe}[/red]"
            )

            raise typer.Exit(code=1)

        raise typer.Exit(
            code=proc.returncode
        )

    else:
        os.execvp(
            cmd[0],
            cmd,
        )


@app.command(help="Export a launch profile to JSON.")
def export(
    profile: str = typer.Argument(
        ...,
        help="Profile name or path to save",
    ),
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Model path",
    ),
    hardware: str = typer.Option(
        "",
        "--hardware",
        help="Hardware description",
    ),
    score: float = typer.Option(
        0.0,
        "--score",
        help="Profile score",
    ),
):
    """Export a launch profile to JSON."""

    prof = create_profile(
        name=(
            Path(profile).stem
            if Path(profile).suffix
            else profile
        ),
        args=[],
        model_path=model,
        hardware=hardware,
        score=score,
    )

    path = export_profile(
        prof,
        profile,
    )

    console.print(
        f"[green]Profile exported to {path}[/green]"
    )


@app.command(
    name="import",
    help="Import a launch profile from JSON.",
)
def import_cmd(
    profile: str = typer.Argument(
        ...,
        help="Path to profile JSON",
    ),
):
    """Import a launch profile from JSON."""

    if not os.path.isfile(profile):
        console.print(
            f"[red]Error: profile file not found: "
            f"{profile}[/red]"
        )

        raise typer.Exit(code=1)

    prof = import_profile(profile)

    table = Table(
        title=f"Profile: {prof.name}",
        box=box.ROUNDED,
    )

    table.add_column(
        "Field",
        style="cyan",
    )

    table.add_column(
        "Value",
        style="green",
    )

    table.add_row(
        "Model",
        prof.model_path,
    )

    table.add_row(
        "Hardware",
        prof.hardware,
    )

    table.add_row(
        "Args",
        " ".join(prof.args),
    )

    table.add_row(
        "Score",
        f"{prof.score:.2f}",
    )

    console.print(table)


@app.command(
    help="Launch the integrated llama-autotune Web UI."
)
def web(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host address for the Web UI",
    ),
    port: int = typer.Option(
        8766,
        "--port",
        help="Port for the Web UI",
    ),
):
    """Launch the integrated llama-autotune Web UI."""

    console.print(
        "[bold green]"
        "Starting llama-autotune Web UI"
        "[/bold green]"
    )

    console.print(
        f"http://{host}:{port}"
    )

    run_web(
        host=host,
        port=port,
    )


def _version_callback(
    value: bool,
) -> None:
    """Print the package version and exit when --version is passed."""

    if value:
        from . import __version__

        console.print(
            f"llama-autotune {__version__}"
        )

        raise typer.Exit()


@app.callback()
def main_callback(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """Configure global logging for the CLI."""

    level = (
        logging.DEBUG
        if verbose
        else logging.INFO
    )

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(name)s: "
            "%(message)s"
        ),
    )


main = app
