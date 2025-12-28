"""Tests for visualization and dashboard modules."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import load_iris

from automl import (
    AutoMLEngine,
    ConstraintValidator,
    DatasetAnalyzer,
    Evaluator,
    ModelRegistry,
    ResourceConstraints,
    ResourceMonitor,
    ResultVisualizer,
)


def test_result_visualizer_leaderboard_dataframe():
    """Test conversion of leaderboard to DataFrame."""
    dataset = load_iris(as_frame=True)
    X, y = dataset.data, dataset.target

    engine = AutoMLEngine(
        analyzer=DatasetAnalyzer(),
        registry=ModelRegistry(),
        validator_factory=lambda c: ConstraintValidator(c),
        evaluator=Evaluator(cv_folds=3),
        monitor=ResourceMonitor(sample_interval=0.1),
    )

    constraints = ResourceConstraints(
        max_training_time=10.0,
        max_ram_mb=1024.0,
        min_performance=0.5,
        interpretability_required=False,
        preferred_metrics=["accuracy"],
    )

    result = engine.run(X, y, constraints=constraints, metrics=["accuracy"])

    visualizer = ResultVisualizer()
    df = visualizer.leaderboard_to_dataframe(result)

    assert isinstance(df, pd.DataFrame)
    assert "Rank" in df.columns
    assert "Model" in df.columns
    assert "Combined Score" in df.columns
    assert len(df) > 0


def test_result_visualizer_metrics_summary():
    """Test extraction of metrics summary."""
    dataset = load_iris(as_frame=True)
    X, y = dataset.data, dataset.target

    engine = AutoMLEngine(
        analyzer=DatasetAnalyzer(),
        registry=ModelRegistry(),
        validator_factory=lambda c: ConstraintValidator(c),
        evaluator=Evaluator(cv_folds=3),
        monitor=ResourceMonitor(sample_interval=0.1),
    )

    constraints = ResourceConstraints(
        max_training_time=10.0,
        max_ram_mb=1024.0,
        min_performance=0.5,
        interpretability_required=False,
        preferred_metrics=["accuracy"],
    )

    result = engine.run(X, y, constraints=constraints, metrics=["accuracy"])

    visualizer = ResultVisualizer()
    metrics_summary = visualizer.metrics_summary(result)

    assert isinstance(metrics_summary, dict)
    assert len(metrics_summary) > 0
    for model_name, metrics in metrics_summary.items():
        assert isinstance(model_name, str)
        assert isinstance(metrics, dict)


def test_result_visualizer_resource_usage():
    """Test extraction of resource usage data."""
    dataset = load_iris(as_frame=True)
    X, y = dataset.data, dataset.target

    engine = AutoMLEngine(
        analyzer=DatasetAnalyzer(),
        registry=ModelRegistry(),
        validator_factory=lambda c: ConstraintValidator(c),
        evaluator=Evaluator(cv_folds=3),
        monitor=ResourceMonitor(sample_interval=0.1),
    )

    constraints = ResourceConstraints(
        max_training_time=10.0,
        max_ram_mb=1024.0,
        min_performance=0.5,
        interpretability_required=False,
        preferred_metrics=["accuracy"],
    )

    result = engine.run(X, y, constraints=constraints, metrics=["accuracy"])

    visualizer = ResultVisualizer()
    resource_df = visualizer.resource_usage_summary(result)

    assert isinstance(resource_df, pd.DataFrame)
    assert "Model" in resource_df.columns
    assert "Training Time (s)" in resource_df.columns
    assert "Peak RAM (MB)" in resource_df.columns


def test_result_visualizer_best_model():
    """Test extraction of best model summary."""
    dataset = load_iris(as_frame=True)
    X, y = dataset.data, dataset.target

    engine = AutoMLEngine(
        analyzer=DatasetAnalyzer(),
        registry=ModelRegistry(),
        validator_factory=lambda c: ConstraintValidator(c),
        evaluator=Evaluator(cv_folds=3),
        monitor=ResourceMonitor(sample_interval=0.1),
    )

    constraints = ResourceConstraints(
        max_training_time=10.0,
        max_ram_mb=1024.0,
        min_performance=0.5,
        interpretability_required=False,
        preferred_metrics=["accuracy"],
    )

    result = engine.run(X, y, constraints=constraints, metrics=["accuracy"])

    visualizer = ResultVisualizer()
    best_model = visualizer.best_model_summary(result)

    assert isinstance(best_model, dict)
    if "status" not in best_model:
        assert "Model" in best_model
        assert "Primary Metric" in best_model
        assert "All Metrics" in best_model
