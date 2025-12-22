"""Configuration objects for the AutoML framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class ResourceConstraints:
    """User-defined resource limits for AutoML runs."""

    max_training_time: Optional[float] = None
    max_ram_mb: Optional[float] = None
    min_performance: Optional[float] = None
    interpretability_required: bool = False
    preferred_metrics: Optional[List[str]] = None
    additional_params: Optional[Dict[str, Union[str, float, int]]] = None


@dataclass
class SearchSpaceConfig:
    """Defines hyperparameter search boundaries for a model family."""

    model_name: str
    params: Dict[str, Union[List[Union[str, float, int]], Dict[str, Union[float, int]]]]
