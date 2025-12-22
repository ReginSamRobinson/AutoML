"""Example script for running the AutoML engine."""

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


def main() -> None:
    """Run an example AutoML search on the breast cancer dataset."""

    dataset = load_breast_cancer(as_frame=True)
    features: pd.DataFrame = dataset.data
    target: pd.Series = dataset.target

    analyzer = DatasetAnalyzer()
    registry = ModelRegistry()
    evaluator = Evaluator(cv_folds=5)
    monitor = ResourceMonitor(sample_interval=0.2)

    engine = AutoMLEngine(
        analyzer=analyzer,
        registry=registry,
        validator_factory=lambda constraints: ConstraintValidator(constraints),
        evaluator=evaluator,
        monitor=monitor,
    )

    constraints = ResourceConstraints(
        max_training_time=60.0,
        max_ram_mb=2048.0,
        min_performance=0.90,
        interpretability_required=False,
        preferred_metrics=["accuracy"],
    )

    metrics = ["accuracy", "f1", "roc_auc"]

    result = engine.run(features, target, constraints=constraints, metrics=metrics)

    print("Best model:")
    if result.best_result is None:
        print("  No valid model satisfied the constraints.")
    else:
        best = result.best_result
        print(f"  Model: {best.model_name}")
        print(f"  Primary Metric ({best.primary_metric}): {best.metrics[best.primary_metric]:.4f}")
        print(f"  Training Time: {best.resource_usage.training_time:.2f}s")
        if best.violations:
            print("  Constraint Violations:")
            for violation in best.violations:
                print(f"    - {violation.details}")
        else:
            print("  Constraint Violations: None")

    print("\nLeaderboard:")
    if not result.leaderboard:
        print("  No evaluation results available.")
    else:
        for entry in result.leaderboard:
            violations_summary = ", ".join(v.details for v in entry.result.violations) or "None"
            metrics_display = ", ".join(
                f"{name}={value:.4f}"
                for name, value in entry.result.metrics.items()
                if not name.endswith("_std")
            )
            print(
                f"  #{entry.rank} {entry.result.model_name} | Score={entry.combined_score:.3f} | "
                f"{metrics_display} | Time={entry.result.resource_usage.training_time:.2f}s | "
                f"RAM={entry.result.resource_usage.peak_ram_mb:.1f}MB | Violations={violations_summary}"
            )


if __name__ == "__main__":
    main()
