"""Unit tests for leaderboard ranking, constraints, and metric helpers."""

from __future__ import annotations

import pytest

from automl import ConstraintValidator, ResourceConstraints
from automl.constraint import ConstraintViolation
from automl.dataset_analysis import DatasetSummary
from automl.evaluation import EvaluationResult, Evaluator
from automl.leaderboard import FAILED_RUN_SCORE, Leaderboard
from automl.resource_monitor import ResourceUsage
from automl.utils import normalize_metric_name


def _result(name, value, *, time=1.0, ram=100.0, metric="accuracy", higher=True, violations=()):
    return EvaluationResult(
        model_name=name,
        estimator=None,
        params={},
        metrics={metric: value},
        metric_preferences={metric: higher},
        primary_metric=metric,
        resource_usage=ResourceUsage(training_time=time, peak_ram_mb=ram, avg_cpu_percent=50.0),
        violations=list(violations),
    )


def _ranked_names(board: Leaderboard) -> list[str]:
    return [entry.result.model_name for entry in board.build()]


class TestLeaderboard:
    def test_higher_metric_ranks_first_when_resources_equal(self) -> None:
        board = Leaderboard()
        board.add(_result("worse", 0.80))
        board.add(_result("better", 0.95))
        assert _ranked_names(board) == ["better", "worse"]

    def test_lower_error_ranks_first(self) -> None:
        board = Leaderboard()
        board.add(_result("high_error", 10.0, metric="rmse", higher=False))
        board.add(_result("low_error", 2.0, metric="rmse", higher=False))
        assert _ranked_names(board) == ["low_error", "high_error"]

    def test_cheaper_model_wins_on_equal_performance(self) -> None:
        board = Leaderboard()
        board.add(_result("slow", 0.9, time=50.0, ram=900.0))
        board.add(_result("fast", 0.9, time=1.0, ram=50.0))
        assert _ranked_names(board) == ["fast", "slow"]

    def test_failed_runs_rank_last_and_do_not_skew_others(self) -> None:
        board = Leaderboard()
        board.add(_result("good", 0.9, time=40.0, ram=800.0))
        board.add(
            _result(
                "crashed",
                float("nan"),
                time=0.0,
                ram=0.0,
                violations=[ConstraintViolation(name="runtime_error", details="boom")],
            )
        )
        board.add(_result("ok", 0.85, time=60.0, ram=900.0))

        entries = board.build()
        assert entries[-1].result.model_name == "crashed"
        assert entries[-1].combined_score == FAILED_RUN_SCORE
        assert board.best().model_name in {"good", "ok"}

    def test_best_skips_violating_results(self) -> None:
        board = Leaderboard()
        board.add(_result("violating", 0.99, violations=[ConstraintViolation("memory", "too much")]))
        board.add(_result("valid", 0.70))
        assert board.best().model_name == "valid"


class TestConstraintValidator:
    def _usage(self, time=1.0, ram=10.0) -> ResourceUsage:
        return ResourceUsage(training_time=time, peak_ram_mb=ram, avg_cpu_percent=0.0)

    def test_reports_time_and_memory(self) -> None:
        validator = ConstraintValidator(ResourceConstraints(max_training_time=5.0, max_ram_mb=100.0))
        violations = validator.validate({"accuracy": 0.9}, self._usage(time=9.0, ram=200.0), "accuracy", {})
        assert {v.name for v in violations} == {"training_time", "memory"}

    def test_error_metric_threshold_is_an_upper_bound(self) -> None:
        validator = ConstraintValidator(ResourceConstraints(min_performance=5.0))
        prefs = {"rmse": False}
        assert validator.validate({"rmse": 3.0}, self._usage(), "rmse", prefs) == []
        (violation,) = validator.validate({"rmse": 8.0}, self._usage(), "rmse", prefs)
        assert "exceeds maximum" in violation.details


class TestMetrics:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("F1-Score", "f1"),
            ("mean_squared_error", "mse"),
            ("root_mean_squared_error", "rmse"),
            ("mean_absolute_error", "mae"),
            ("Accuracy", "accuracy"),
        ],
    )
    def test_normalize_metric_name(self, raw: str, expected: str) -> None:
        assert normalize_metric_name(raw) == expected

    def test_unsupported_metric_for_task_raises(self) -> None:
        summary = DatasetSummary(
            n_rows=10, n_columns=1, task_type="regression", target_name="y",
            numeric_columns=["x"], categorical_columns=[],
        )
        with pytest.raises(ValueError):
            Evaluator()._resolve_metric("accuracy", summary)
