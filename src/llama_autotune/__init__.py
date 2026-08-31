"""llama-autotune: automatic configuration optimisation for llama.cpp.

Exposes high-level building blocks:

- :class:`HardwareInfo` / ``detect_hardware`` — system introspection
- :class:`ModelInfo` / ``inspect_model`` — model metadata extraction
- ``generate_initial_config`` — heuristic-based default config
- :class:`Optimizer` — search-based tuning loop
- ``create_profile`` / ``export_profile`` / ``import_profile`` — profile persistence
- ``save_benchmark`` / ``save_launch_profile`` — database helpers
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("llama-autotune")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from .benchmark import run_benchmark
from .database import (
    get_session,
    save_benchmark,
    save_launch_profile,
    session_scope,
)
from .hardware import detect_hardware
from .heuristics import generate_initial_config
from .model_inspector import inspect_model
from .models import (
    Backend,
    BenchmarkEntry,
    BenchmarkResult,
    GpuVendor,
    HardwareInfo,
    LaunchProfile,
    ModelInfo,
    OptimizeObjective,
    SearchConfig,
    SplitMode,
)
from .optimizer import Optimizer
from .profiles import create_profile, export_profile, import_profile

__all__ = [
    "Backend",
    "BenchmarkEntry",
    "BenchmarkResult",
    "GpuVendor",
    "HardwareInfo",
    "LaunchProfile",
    "ModelInfo",
    "OptimizeObjective",
    "SearchConfig",
    "SplitMode",
    "detect_hardware",
    "inspect_model",
    "generate_initial_config",
    "run_benchmark",
    "Optimizer",
    "create_profile",
    "export_profile",
    "import_profile",
    "get_session",
    "save_benchmark",
    "save_launch_profile",
    "session_scope",
]
