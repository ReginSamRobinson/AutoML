"""Configuration objects for the AutoML framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class ResourceConstraints:
    """User-defined resource limits for AutoML runs.

    Attributes:
        max_training_time: Seconds allowed for evaluating one candidate
            (all cross-validation folds together).
        max_ram_mb: Extra memory in MB a candidate may use on top of the
            process baseline measured just before it starts.
        min_performance: Threshold on the primary metric. A lower bound for
            higher-is-better metrics (accuracy) and an upper bound for
            error metrics (rmse).
        max_total_time: Seconds for the whole search; no new candidates are
            started once it is spent.
    """

    max_training_time: Optional[float] = None
    max_ram_mb: Optional[float] = None
    min_performance: Optional[float] = None
    interpretability_required: bool = False
    preferred_metrics: Optional[List[str]] = None
    additional_params: Optional[Dict[str, Union[str, float, int]]] = None
    max_total_time: Optional[float] = None
