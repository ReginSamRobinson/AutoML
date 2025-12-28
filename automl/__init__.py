"""Resource-aware, constraint-driven AutoML package."""

from .config import ResourceConstraints
from .engine import AutoMLEngine, AutoMLResult
from .dataset_analysis import DatasetAnalyzer
from .model_registry import ModelRegistry
from .constraint import ConstraintValidator
from .evaluation import Evaluator
from .resource_monitor import ResourceMonitor
from .visualizer import ResultVisualizer
from .dashboard import create_app, run_dashboard
from .exporter import export_to_html

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
	"export_to_html",
]
