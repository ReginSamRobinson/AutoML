"""Tests for messy real-world inputs: missing values, encodings, column types and odd targets."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from automl import DatasetAnalyzer, ResourceConstraints
from automl.data_upload import get_dataset_preview, load_dataset

from conftest import FAST_CONSTRAINTS, make_engine

# Logistic regression / elastic net only, so engine tests stay fast.
QUICK = ResourceConstraints(**FAST_CONSTRAINTS, interpretability_required=True)


def _frame(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})


class TestUploads:
    def test_preview_with_missing_values_is_strict_json(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.nan] * 10, "b": [np.inf] * 20, "when": [pd.NaT] * 20})
        # Browsers reject NaN/Infinity tokens, so the payload must not contain them.
        json.dumps(get_dataset_preview(df), allow_nan=False, default=str)

    def test_latin1_semicolon_csv_loads(self, tmp_path) -> None:
        path = tmp_path / "data.csv"
        rows = "".join(f"café {i};{i * 1.5};{i % 2}\n" for i in range(20))
        path.write_bytes(("name;value;target\n" + rows).encode("latin-1"))

        df, error = load_dataset(str(path))

        assert error is None
        assert list(df.columns) == ["name", "value", "target"]
        assert df["name"].iloc[0] == "café 0"


class TestColumnHandling:
    def test_identifier_constant_and_datetime_columns(self) -> None:
        n = 200
        features = _frame(n).assign(
            customer_id=np.arange(1000, 1000 + n),
            ticket=[f"T-{i:05d}" for i in range(n)],
            constant="same",
            empty=np.nan,
            signup=pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            plan=np.tile(["basic", "pro", "team"], n)[:n],
        )

        summary = DatasetAnalyzer().summarize(features, pd.Series(np.arange(n) % 2))

        assert set(summary.dropped_columns) == {"customer_id", "ticket", "constant", "empty"}
        assert summary.datetime_columns == ["signup"]
        assert summary.categorical_columns == ["plan"]
        assert summary.numeric_columns == ["x1", "x2"]

    def test_engine_trains_on_mixed_column_types(self) -> None:
        n = 150
        features = _frame(n).assign(
            row_id=np.arange(n),
            stamp=pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
            mixed=[1, "a", 2.5] * (n // 3),  # numbers and text in one column
        )
        target = pd.Series(np.where(features.x1 > 0, "yes", "no"))

        result = make_engine().run(features, target, constraints=QUICK, metrics=["accuracy"])

        assert not any(v.name == "runtime_error" for e in result.leaderboard for v in e.result.violations)
        assert any("row_id" in note for note in result.notes)
        assert result.best_estimator.predict(features.head()).shape == (5,)


class TestTargets:
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([1.5, 2.5, 3.5, 10.0], "regression"),  # few distinct, but fractional
            ([0.0, 1.0], "classification"),  # integer labels read as float
            (["no", "yes"], "classification"),
        ],
    )
    def test_task_type_detection(self, values, expected) -> None:
        target = pd.Series(np.resize(values, 120))
        assert DatasetAnalyzer().summarize(_frame(), target).task_type == expected

    def test_task_type_override(self) -> None:
        target = pd.Series(np.resize([1.5, 2.5], 120))
        summary = DatasetAnalyzer().summarize(_frame(), target, task_type="classification")
        assert summary.task_type == "classification"

    def test_single_class_target_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="only one distinct value"):
            make_engine().run(_frame(), pd.Series(["a"] * 120), constraints=QUICK, metrics=["accuracy"])

    def test_singleton_classes_are_dropped_and_folds_reduced(self) -> None:
        features = _frame(60)
        target = pd.Series(["a"] * 30 + ["b"] * 27 + ["c"] * 2 + ["d"])

        result = make_engine(cv_folds=5).run(features, target, constraints=QUICK, metrics=["accuracy"])

        assert result.best_result is not None
        assert any("'d'" in note for note in result.notes)
        assert any("2-fold" in note for note in result.notes)


def test_closest_model_is_kept_when_no_candidate_passes() -> None:
    features = _frame()
    target = pd.Series(np.random.default_rng(1).integers(0, 2, 120))  # pure noise
    constraints = ResourceConstraints(**FAST_CONSTRAINTS, interpretability_required=True, min_performance=0.99)

    with pytest.warns(UserWarning, match="closest"):
        result = make_engine().run(features, target, constraints=constraints, metrics=["accuracy"])

    assert result.constraints_met is False
    assert result.best_result is not None and result.best_result.violations
    assert result.best_estimator is not None
