"""Core AutoML engine orchestrating the workflow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterSampler

from .config import ResourceConstraints
from .constraint import ConstraintValidator, ConstraintViolation
from .dataset_analysis import DatasetAnalyzer, DatasetSummary
from .evaluation import Evaluator, EvaluationResult
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
        target_series = target if isinstance(target, pd.Series) else pd.Series(target)

        summary = self.analyzer.summarize(features_df, target_series)
        normalized_metrics = self._prepare_metrics(metrics, summary, constraints)
        primary_metric = normalized_metrics[0]

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

        for model_name in model_names:
            search_space = self.registry.get_search_space(model_name, constraints, summary)
            candidate_params = self._sample_parameters(search_space, rng, constraints, model_name)

            for params in candidate_params:
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

        normalized = []
        for metric in candidate_metrics:
            canonical = normalize_metric_name(metric)
            if canonical not in normalized:
                normalized.append(canonical)
        return normalized

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

        budget = self._determine_budget(constraints, model_name)
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
