"""Integration tests for the AutoML engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, make_regression

from automl import ResourceConstraints

from conftest import FAST_CONSTRAINTS, make_engine


def test_automl_engine_selects_model() -> None:
    """Ensure the engine returns a valid best model under constraints."""

    dataset = load_breast_cancer(as_frame=True)
    constraints = ResourceConstraints(
        max_training_time=30.0,
        max_ram_mb=1024.0,
        min_performance=0.85,
        interpretability_required=True,
        preferred_metrics=["accuracy"],
    )

    result = make_engine().run(dataset.data, dataset.target, constraints=constraints, metrics=["accuracy", "f1"])

    assert result.best_result is not None
    assert result.best_estimator is not None
    assert result.best_result.metrics["accuracy"] >= 0.85
    assert result.leaderboard, "Expected at least one leaderboard entry"


def test_regression_with_classification_metrics_falls_back() -> None:
    """Dashboard defaults (accuracy/f1) must not break a regression dataset."""

    X, y = make_regression(n_samples=120, n_features=5, noise=5.0, random_state=0)
    constraints = ResourceConstraints(**FAST_CONSTRAINTS, min_performance=0.85, preferred_metrics=["accuracy"])

    with pytest.warns(UserWarning):
        result = make_engine().run(
            pd.DataFrame(X), pd.Series(y), constraints=constraints, metrics=["accuracy", "f1", "roc_auc"]
        )

    assert result.best_result is not None
    assert result.best_result.primary_metric == "rmse"
    assert not any(v.name == "runtime_error" for e in result.leaderboard for v in e.result.violations)


def test_regression_ranks_lower_error_higher() -> None:
    X, y = make_regression(n_samples=150, n_features=5, noise=1.0, random_state=0)
    constraints = ResourceConstraints(**FAST_CONSTRAINTS)

    result = make_engine().run(pd.DataFrame(X), pd.Series(y), constraints=constraints, metrics=["rmse"])

    # The linear model fits this linear data with ~1% of the trees' error; it must win.
    assert result.leaderboard[0].result.model_name == "elastic_net"


def test_missing_target_rows_are_dropped() -> None:
    dataset = load_breast_cancer(as_frame=True)
    target = dataset.target.astype(float).copy()
    target.iloc[:10] = np.nan

    result = make_engine().run(
        dataset.data,
        target,
        constraints=ResourceConstraints(**FAST_CONSTRAINTS, interpretability_required=True),
        metrics=["accuracy"],
    )

    assert result.best_result is not None


def test_string_labels_work_with_all_metrics() -> None:
    dataset = load_breast_cancer(as_frame=True)
    labels = dataset.target.map({0: "malignant", 1: "benign"})

    result = make_engine().run(
        dataset.data,
        labels,
        constraints=ResourceConstraints(**FAST_CONSTRAINTS),
        metrics=["accuracy", "f1", "precision", "recall", "roc_auc"],
    )

    for entry in result.leaderboard:
        assert not entry.result.violations or all(
            v.name != "runtime_error" for v in entry.result.violations
        ), entry.result.violations
    assert set(result.best_estimator.predict(dataset.data.head())) <= {"malignant", "benign"}


def test_max_total_time_stops_search() -> None:
    dataset = load_breast_cancer(as_frame=True)
    constraints = ResourceConstraints(max_total_time=0.0)

    result = make_engine().run(dataset.data, dataset.target, constraints=constraints, metrics=["accuracy"])

    assert len(result.leaderboard) == 1
