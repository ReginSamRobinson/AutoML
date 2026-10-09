"""Tests for the result visualizer."""

from __future__ import annotations

import pandas as pd

from automl import AutoMLResult, ResultVisualizer


def test_leaderboard_dataframe(iris_result: AutoMLResult) -> None:
    df = ResultVisualizer.leaderboard_to_dataframe(iris_result)

    assert isinstance(df, pd.DataFrame)
    assert {"Rank", "Model", "Combined Score"} <= set(df.columns)
    assert len(df) == len(iris_result.leaderboard)


def test_metrics_summary_uses_best_config_per_model(iris_result: AutoMLResult) -> None:
    summary = ResultVisualizer.metrics_summary(iris_result)

    assert summary
    best_by_model: dict[str, dict] = {}
    for entry in iris_result.leaderboard:  # rank-ordered
        best_by_model.setdefault(entry.result.model_name, entry.result.metrics)
    for model_name, metrics in summary.items():
        assert metrics["accuracy"] == round(best_by_model[model_name]["accuracy"], 4)


def test_resource_usage_summary(iris_result: AutoMLResult) -> None:
    df = ResultVisualizer.resource_usage_summary(iris_result)

    assert {"Model", "Training Time (s)", "Peak RAM (MB)"} <= set(df.columns)
    assert (df["Peak RAM (MB)"] >= 0).all()


def test_best_model_summary(iris_result: AutoMLResult) -> None:
    best_model = ResultVisualizer.best_model_summary(iris_result)

    assert isinstance(best_model, dict)
    if "status" not in best_model:
        assert {"Model", "Primary Metric", "All Metrics"} <= set(best_model)
