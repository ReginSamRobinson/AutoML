"""Shared fixtures for the AutoML test suite."""

from __future__ import annotations

import pytest
from sklearn.datasets import load_iris

from automl import (
    AutoMLEngine,
    AutoMLResult,
    ConstraintValidator,
    DatasetAnalyzer,
    Evaluator,
    ModelRegistry,
    ResourceConstraints,
    ResourceMonitor,
)


def make_engine(cv_folds: int = 3) -> AutoMLEngine:
    return AutoMLEngine(
        analyzer=DatasetAnalyzer(),
        registry=ModelRegistry(),
        validator_factory=lambda c: ConstraintValidator(c),
        evaluator=Evaluator(cv_folds=cv_folds),
        monitor=ResourceMonitor(sample_interval=0.05),
    )


# Tight limits keep the search small so the suite stays fast.
FAST_CONSTRAINTS = dict(max_training_time=10.0, max_ram_mb=1024.0)


@pytest.fixture(scope="session")
def iris_result() -> AutoMLResult:
    """A completed multiclass AutoML run, shared across tests."""

    dataset = load_iris(as_frame=True)
    constraints = ResourceConstraints(
        **FAST_CONSTRAINTS,
        min_performance=0.5,
        preferred_metrics=["accuracy"],
    )
    return make_engine().run(dataset.data, dataset.target, constraints=constraints, metrics=["accuracy", "f1"])
