"""Model evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from .constraint import ConstraintViolation
from .dataset_analysis import DatasetSummary
from .resource_monitor import ResourceUsage


CLASSIFICATION_METRICS = frozenset({"accuracy", "balanced_accuracy", "f1", "roc_auc", "precision", "recall"})
REGRESSION_METRICS = frozenset({"r2", "rmse", "mse", "mae", "mape"})


def supported_metrics(task_type: str) -> frozenset[str]:
    """Return the canonical metric names usable for the given task type."""

    return CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS


@dataclass
class EvaluationResult:
    """Captures cross-validation results for a model."""

    model_name: str
    estimator: BaseEstimator
    params: Dict[str, object]
    metrics: Dict[str, float]
    metric_preferences: Dict[str, bool]
    primary_metric: str
    resource_usage: ResourceUsage
    violations: List[ConstraintViolation]


@dataclass
class MetricSpec:
    """Metadata defining how to evaluate and aggregate a metric."""

    scoring: Union[str, Callable]
    greater_is_better: bool


class Evaluator:
    """Runs cross-validation and computes metrics."""

    def __init__(self, cv_folds: int = 5, random_state: int | None = 42) -> None:
        if cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        self.cv_folds = cv_folds
        self.random_state = random_state

    def evaluate(
        self,
        estimator: BaseEstimator,
        features: pd.DataFrame,
        target: pd.Series,
        metrics: List[str],
        summary: DatasetSummary,
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        """Return aggregated performance metrics and their optimization direction."""

        metric_specs = {metric: self._resolve_metric(metric, summary) for metric in metrics}
        scoring = {metric: spec.scoring for metric, spec in metric_specs.items()}

        cv_strategy = self._build_cv(summary, target)

        cv_results = cross_validate(
            estimator,
            features,
            target,
            scoring=scoring,
            cv=cv_strategy,
            n_jobs=None,
            return_train_score=False,
            error_score="raise",
        )

        aggregated: Dict[str, float] = {}
        preferences: Dict[str, bool] = {}
        for metric, spec in metric_specs.items():
            score_key = f"test_{metric}"
            if score_key not in cv_results:
                raise KeyError(f"Cross-validation results missing metric '{metric}'")
            values = np.asarray(cv_results[score_key], dtype=float)
            if spec.greater_is_better:
                transformed = values
            else:
                transformed = -values
            mean_score = float(np.mean(transformed))
            std_score = float(np.std(transformed))
            aggregated[metric] = mean_score
            aggregated[f"{metric}_std"] = std_score
            preferences[metric] = spec.greater_is_better

        return aggregated, preferences

    def _build_cv(self, summary: DatasetSummary, target: pd.Series):
        """Create the appropriate cross-validation splitter."""

        if summary.task_type == "classification":
            return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        return KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

    def _resolve_metric(self, metric: str, summary: DatasetSummary) -> MetricSpec:
        """Map metric identifiers to sklearn scoring strings."""

        metric = metric.lower()
        classification = summary.task_type == "classification"
        n_classes = len(summary.class_balance) if summary.class_balance else 0

        # sklearn's binary f1/precision/recall scorers assume pos_label=1, which
        # fails for labels like "yes"/"no"; use the last sorted label instead.
        binary_scorers = {"f1": f1_score, "precision": precision_score, "recall": recall_score}
        if classification and n_classes == 2 and metric in binary_scorers:
            labels = sorted(summary.class_balance, key=str)
            if 1 not in labels:
                return MetricSpec(
                    scoring=make_scorer(binary_scorers[metric], pos_label=labels[-1]),
                    greater_is_better=True,
                )

        if metric == "accuracy" and classification:
            return MetricSpec(scoring="accuracy", greater_is_better=True)
        if metric == "balanced_accuracy" and classification:
            return MetricSpec(scoring="balanced_accuracy", greater_is_better=True)
        if metric == "f1" and classification:
            scorer = "f1" if n_classes <= 2 else "f1_macro"
            return MetricSpec(scoring=scorer, greater_is_better=True)
        if metric == "roc_auc" and classification:
            scorer = "roc_auc" if n_classes <= 2 else "roc_auc_ovr_weighted"
            return MetricSpec(scoring=scorer, greater_is_better=True)
        if metric == "precision" and classification:
            scorer = "precision" if n_classes <= 2 else "precision_macro"
            return MetricSpec(scoring=scorer, greater_is_better=True)
        if metric == "recall" and classification:
            scorer = "recall" if n_classes <= 2 else "recall_macro"
            return MetricSpec(scoring=scorer, greater_is_better=True)

        if metric == "r2" and not classification:
            return MetricSpec(scoring="r2", greater_is_better=True)
        if metric in {"rmse", "root_mean_squared_error"} and not classification:
            return MetricSpec(scoring="neg_root_mean_squared_error", greater_is_better=False)
        if metric == "mse" and not classification:
            return MetricSpec(scoring="neg_mean_squared_error", greater_is_better=False)
        if metric in {"mae", "mean_absolute_error"} and not classification:
            return MetricSpec(scoring="neg_mean_absolute_error", greater_is_better=False)
        if metric in {"mape", "mean_absolute_percentage_error"} and not classification:
            return MetricSpec(scoring="neg_mean_absolute_percentage_error", greater_is_better=False)

        raise ValueError(f"Unsupported metric '{metric}' for task '{summary.task_type}'")
