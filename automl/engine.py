"""Core AutoML engine orchestrating the workflow."""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid, ParameterSampler

from .config import ResourceConstraints
from .constraint import ConstraintValidator, ConstraintViolation
from .dataset_analysis import DatasetAnalyzer, DatasetSummary
from .evaluation import Evaluator, EvaluationResult, supported_metrics
from .leaderboard import Leaderboard, LeaderboardEntry
from .model_registry import ModelRegistry
from .resource_monitor import ResourceMonitor, ResourceUsage
from .utils import normalize_metric_name


@dataclass
class AutoMLResult:
    """Summary of an AutoML run."""

    best_result: Optional[EvaluationResult]
    best_estimator: Optional[BaseEstimator]
    leaderboard: List[LeaderboardEntry]


class AutoMLEngine:
    """Coordinates dataset analysis, model selection, and evaluation."""

    def __init__(
        self,
        analyzer: DatasetAnalyzer,
        registry: ModelRegistry,
        validator_factory: Callable[[ResourceConstraints], ConstraintValidator],
        evaluator: Evaluator,
        monitor: ResourceMonitor,
        random_state: int | None = 42,
    ) -> None:
        self.analyzer = analyzer
        self.registry = registry
        self.validator_factory = validator_factory
        self.evaluator = evaluator
        self.monitor = monitor
        self.random_state = random_state

    def run(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        constraints: ResourceConstraints,
        metrics: List[str],
    ) -> AutoMLResult:
        """Execute an AutoML search with the provided constraints."""

        features_df = features if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
        target_series = target if isinstance(target, pd.Series) else pd.Series(target, index=features_df.index)

        # Rows without a label cannot be used for supervised training.
        labelled = target_series.notna().to_numpy()
        if not labelled.all():
            features_df = features_df.loc[labelled]
            target_series = target_series.loc[labelled]
        if target_series.empty:
            raise ValueError("Target column has no non-missing values")

        summary = self.analyzer.summarize(features_df, target_series)
        normalized_metrics = self._prepare_metrics(metrics, summary, constraints)
        primary_metric = normalized_metrics[0]
        constraints = self._align_performance_threshold(constraints, metrics, primary_metric)

        validator = self.validator_factory(constraints)
        leaderboard = Leaderboard()

        model_names = self.registry.list_models(summary.task_type, constraints.interpretability_required)
        if constraints.interpretability_required:
            interpretable = [
                name for name in model_names if self.registry.is_interpretable(name, summary.task_type)
            ]
            if interpretable:
                model_names = interpretable

        rng = np.random.default_rng(self.random_state)
        search_started = perf_counter()

        for model_name in model_names:
            search_space = self.registry.get_search_space(model_name, constraints, summary)
            candidate_params = self._sample_parameters(search_space, rng, constraints, model_name)

            for params in candidate_params:
                if self._budget_exhausted(constraints, search_started, leaderboard):
                    break
                estimator = self.registry.build_estimator(model_name, summary)
                if params:
                    estimator.set_params(**params)

                try:
                    usage, eval_payload = self.monitor.measure(
                        self.evaluator.evaluate,
                        estimator,
                        features_df,
                        target_series,
                        normalized_metrics,
                        summary,
                    )
                    performance, preferences = eval_payload
                except Exception as exc:  # pragma: no cover - defensive logging path
                    usage = ResourceUsage(training_time=0.0, peak_ram_mb=0.0, avg_cpu_percent=0.0)
                    performance = {metric: float("nan") for metric in normalized_metrics}
                    preferences = {metric: True for metric in normalized_metrics}
                    violations = [ConstraintViolation(name="runtime_error", details=str(exc))]
                else:
                    violations = validator.validate(performance, usage, primary_metric, preferences)

                result = EvaluationResult(
                    model_name=model_name,
                    estimator=estimator,
                    params=deepcopy(params),
                    metrics=performance,
                    metric_preferences=preferences,
                    primary_metric=primary_metric,
                    resource_usage=usage,
                    violations=violations,
                )
                leaderboard.add(result)

        leaderboard_entries = leaderboard.build()
        best_result = leaderboard.best()
        best_estimator: Optional[BaseEstimator] = None

        if best_result is not None:
            best_estimator = self.registry.build_estimator(best_result.model_name, summary)
            if best_result.params:
                best_estimator.set_params(**best_result.params)
            best_estimator.fit(features_df, target_series)

        return AutoMLResult(
            best_result=best_result,
            best_estimator=best_estimator,
            leaderboard=leaderboard_entries,
        )

    def _prepare_metrics(
        self,
        metrics: List[str],
        summary: DatasetSummary,
        constraints: ResourceConstraints,
    ) -> List[str]:
        """Normalize metric names and ensure suitability for the task."""

        candidate_metrics: List[str] = []
        if constraints.preferred_metrics:
            candidate_metrics.extend(constraints.preferred_metrics)
        candidate_metrics.extend(metrics)

        if not candidate_metrics:
            default = "accuracy" if summary.task_type == "classification" else "rmse"
            candidate_metrics = [default]

        allowed = supported_metrics(summary.task_type)
        normalized: List[str] = []
        dropped: List[str] = []
        for metric in candidate_metrics:
            canonical = normalize_metric_name(metric)
            if canonical not in allowed:
                if metric not in dropped:
                    dropped.append(metric)
            elif canonical not in normalized:
                normalized.append(canonical)

        if dropped:
            warnings.warn(
                f"Ignoring metrics not supported for {summary.task_type}: {', '.join(dropped)}",
                stacklevel=3,
            )
        if not normalized:
            normalized = ["accuracy" if summary.task_type == "classification" else "rmse"]
        return normalized

    @staticmethod
    def _align_performance_threshold(
        constraints: ResourceConstraints,
        requested_metrics: List[str],
        primary_metric: str,
    ) -> ResourceConstraints:
        """Drop min_performance when it was set for a metric that is no longer primary.

        A threshold like 0.85 written for accuracy is meaningless once the task
        turns out to be regression and RMSE becomes the primary metric.
        """

        if constraints.min_performance is None:
            return constraints
        requested = list(constraints.preferred_metrics or []) + list(requested_metrics)
        if requested and normalize_metric_name(requested[0]) != primary_metric:
            warnings.warn(
                f"min_performance={constraints.min_performance} was set for '{requested[0]}', "
                f"which is unavailable; ignoring it for primary metric '{primary_metric}'",
                stacklevel=3,
            )
            return replace(constraints, min_performance=None)
        return constraints

    @staticmethod
    def _budget_exhausted(
        constraints: ResourceConstraints,
        search_started: float,
        leaderboard: Leaderboard,
    ) -> bool:
        """Return True once the overall search time budget is spent.

        At least one candidate is always evaluated so a result can be produced.
        """

        if constraints.max_total_time is None or not leaderboard.has_results:
            return False
        return perf_counter() - search_started >= constraints.max_total_time

    def _sample_parameters(
        self,
        search_space: dict[str, List[object]],
        rng: np.random.Generator,
        constraints: ResourceConstraints,
        model_name: str,
    ) -> List[dict[str, object]]:
        """Sample hyperparameter configurations respecting constraints."""

        if not search_space:
            return [{}]

        filtered_space = {key: values for key, values in search_space.items() if values}
        if not filtered_space:
            return [{}]

        budget = min(self._determine_budget(constraints, model_name), len(ParameterGrid(filtered_space)))
        sampler = ParameterSampler(filtered_space, n_iter=budget, random_state=int(rng.integers(0, 1_000_000)))
        unique_configs: List[dict[str, object]] = []
        for config in sampler:
            if config not in unique_configs:
                unique_configs.append(config)
        return unique_configs or [{}]

    def _determine_budget(self, constraints: ResourceConstraints, model_name: str) -> int:
        """Determine hyperparameter budget per model based on constraints."""

        base_budget = 10
        if constraints.max_training_time is not None:
            if constraints.max_training_time <= 60:
                base_budget = 5
            if constraints.max_training_time <= 30:
                base_budget = 3

        if constraints.interpretability_required and model_name not in {"logistic_regression", "elastic_net"}:
            base_budget = min(base_budget, 4)

        if constraints.max_ram_mb is not None and constraints.max_ram_mb <= 1024:
            base_budget = min(base_budget, 5)

        return max(1, base_budget)
