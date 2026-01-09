"""Example script for preparing and exploring the AutoML dashboard."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer

from automl import (
    AutoMLEngine,
    AutoMLResult,
    ConstraintValidator,
    DatasetAnalyzer,
    Evaluator,
    ModelRegistry,
    ResourceConstraints,
    ResourceMonitor,
    export_to_html,
    run_dashboard_qt,
)
from automl.dashboard import SAMPLE_DATASET_FILENAME
from automl.data_upload import create_upload_folder

warnings.filterwarnings("ignore", category=FutureWarning)


def prepare_sample_dataset() -> Tuple[Path, pd.DataFrame, pd.Series]:
    """Persist the sample dataset so it appears on the launch page."""

    upload_dir = Path(create_upload_folder())
    sample_path = upload_dir / SAMPLE_DATASET_FILENAME

    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy() if getattr(dataset, "frame", None) is not None else None

    if frame is None:
        feature_names = getattr(dataset, "feature_names", None)
        features = pd.DataFrame(dataset.data, columns=feature_names)
        if features.empty:
            features = pd.DataFrame(dataset.data)
        frame = features.assign(target=dataset.target)

    if "target" not in frame.columns:
        frame = frame.assign(target=dataset.target)

    frame.to_csv(sample_path, index=False)

    features_df = frame.drop(columns=["target"])
    target_series = frame["target"]
    return sample_path, features_df, target_series


def run_automl(features: pd.DataFrame, target: pd.Series) -> AutoMLResult:
    """Execute the AutoML search using the sample dataset."""

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
        max_training_time=30.0,
        max_ram_mb=2048.0,
        min_performance=0.85,
        interpretability_required=False,
        preferred_metrics=["accuracy"],
    )

    metrics = ["accuracy", "f1", "roc_auc"]
    return engine.run(features, target, constraints=constraints, metrics=metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the AutoML dashboard with an optional quick run.")
    parser.add_argument(
        "--auto-run",
        action="store_true",
        help="Execute the AutoML search immediately using the sample dataset.",
    )
    args = parser.parse_args()

    sample_path, features_df, target_series = prepare_sample_dataset()
    print(f"Sample dataset ready at {sample_path}")

    result: Optional[AutoMLResult] = None

    if args.auto_run:
        result = run_automl(features_df, target_series)

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

        export_to_html(result, "automl_report.html")
        print("\nReport saved to automl_report.html")
    else:
        print("AutoML run skipped. Use the dashboard to select and launch the sample dataset when you're ready.")

    if run_dashboard_qt is None:
        raise RuntimeError("PySide6 is required for run_dashboard_qt. Install it with `pip install PySide6`.")

    run_dashboard_qt(result)


if __name__ == "__main__":
    main()
