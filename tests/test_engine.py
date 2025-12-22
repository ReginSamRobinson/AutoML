"""Integration tests for the AutoML engine."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer

from automl import (
    AutoMLEngine,
    ConstraintValidator,
    DatasetAnalyzer,
    Evaluator,
    ModelRegistry,
    ResourceConstraints,
    ResourceMonitor,
)


def test_automl_engine_selects_model() -> None:
    """Ensure the engine returns a valid best model under constraints."""

    dataset = load_breast_cancer(as_frame=True)
    features: pd.DataFrame = dataset.data
    target: pd.Series = dataset.target

    engine = AutoMLEngine(
        analyzer=DatasetAnalyzer(),
        registry=ModelRegistry(),
        validator_factory=lambda c: ConstraintValidator(c),
        evaluator=Evaluator(cv_folds=3),
        monitor=ResourceMonitor(sample_interval=0.1),
    )

    constraints = ResourceConstraints(
        max_training_time=30.0,
        max_ram_mb=1024.0,
        min_performance=0.85,
        interpretability_required=True,
        preferred_metrics=["accuracy"],
    )

    result = engine.run(features, target, constraints=constraints, metrics=["accuracy", "f1"])

    assert result.best_result is not None
    assert result.best_estimator is not None
    assert result.best_result.metrics["accuracy"] >= 0.0
    assert result.leaderboard, "Expected at least one leaderboard entry"