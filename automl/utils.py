"""Utility helpers for the AutoML framework."""

from __future__ import annotations

from typing import Dict


_METRIC_ALIASES: Dict[str, str] = {
    "f1_score": "f1",
    "root_mean_squared_error": "rmse",
    "neg_root_mean_squared_error": "rmse",
    "mean_squared_error": "mse",
    "neg_mean_squared_error": "mse",
    "mean_absolute_error": "mae",
    "neg_mean_absolute_error": "mae",
    "mean_absolute_percentage_error": "mape",
    "neg_mean_absolute_percentage_error": "mape",
}


def normalize_metric_name(metric: str) -> str:
    """Normalize metric identifiers to a canonical snake_case form."""

    canonical = metric.strip().lower().replace("-", "_").replace(" ", "_")
    return _METRIC_ALIASES.get(canonical, canonical)
