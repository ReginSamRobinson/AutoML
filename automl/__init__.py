"""Resource-aware, constraint-driven AutoML package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .config import ResourceConstraints
from .engine import AutoMLEngine, AutoMLResult
from .dataset_analysis import DatasetAnalyzer
from .model_registry import ModelRegistry
from .constraint import ConstraintValidator
from .evaluation import Evaluator
from .resource_monitor import ResourceMonitor
from .visualizer import ResultVisualizer
from .exporter import export_to_html

# UI entry points pull in Flask/Plotly/Qt, so they are imported on first access
# to keep the core engine usable without the dashboard dependencies.
_LAZY_ATTRIBUTES = {
    "create_app": ".dashboard",
    "run_dashboard": ".dashboard",
    "run_dashboard_app": ".dashboard",
    "run_dashboard_qt": ".qt_dashboard",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRIBUTES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = getattr(import_module(module_name, __name__), name)
    except ImportError:
        if name != "run_dashboard_qt":
            raise
        value = None  # PySide6 is an optional extra
    globals()[name] = value
    return value


__all__ = [
    "AutoMLEngine",
    "AutoMLResult",
    "ResourceConstraints",
    "DatasetAnalyzer",
    "ModelRegistry",
    "ConstraintValidator",
    "Evaluator",
    "ResourceMonitor",
    "ResultVisualizer",
    "create_app",
    "run_dashboard",
    "run_dashboard_app",
    "run_dashboard_qt",
    "export_to_html",
]
