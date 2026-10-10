"""Dataset analysis utilities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

TASK_TYPES = ("classification", "regression")

# A text column with more distinct values than this, making up more than
# HIGH_CARDINALITY_RATIO of its rows, is treated as an identifier or free text.
HIGH_CARDINALITY_MIN_UNIQUE = 50
HIGH_CARDINALITY_RATIO = 0.5
MAX_CLASSES_INT = 15


@dataclass
class DatasetSummary:
    """Stores derived metadata about the dataset."""

    n_rows: int
    n_columns: int
    task_type: str
    target_name: str
    numeric_columns: List[str]
    categorical_columns: List[str]
    class_balance: Dict[Any, float] | None = None
    datetime_columns: List[str] = field(default_factory=list)
    # Columns the models will not see, mapped to the reason they were left out.
    dropped_columns: Dict[str, str] = field(default_factory=dict)


def parse_datetimes(values: pd.Series) -> pd.Series:
    """Parse values to UTC timestamps, turning anything unparseable into NaT."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
        except (TypeError, ValueError):  # pandas < 2.0 has no format="mixed"
            return pd.to_datetime(values, errors="coerce", utc=True)


def _looks_like_datetime(values: pd.Series) -> bool:
    """Return True when nearly every text value parses as a date or time."""

    sample = values.astype(str).head(200)
    # Dates and times have at least two numbers ("2024-01-05", "Jan 5, 2024", "12:30");
    # codes like "T-00042" have one, and the parser would wrongly accept them.
    if (sample.str.count(r"\d+") < 2).any() or sample.str.contains(r"[-/:,\s]").mean() < 0.9:
        return False
    return parse_datetimes(sample).notna().mean() >= 0.9


def _looks_like_row_id(name: object, values: pd.Series) -> bool:
    """Return True for an all-unique integer column that is a row number or ID."""

    lowered = str(name).strip().lower()
    if lowered.startswith("unnamed:") or lowered in {"id", "index"} or lowered.endswith("id"):
        return True
    ordered = values.sort_values()
    return bool(ordered.iloc[-1] - ordered.iloc[0] + 1 == len(ordered))


class DatasetAnalyzer:
    """Analyzes datasets to infer metadata for AutoML."""

    def summarize(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        task_type: Optional[str] = None,
    ) -> DatasetSummary:
        """Generate a dataset summary from feature matrix and target series.

        task_type forces "classification" or "regression" instead of inferring it.
        """

        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if not isinstance(target, pd.Series):
            target = pd.Series(target)

        if task_type is not None and task_type not in TASK_TYPES:
            raise ValueError(f"task_type must be one of {TASK_TYPES}, got {task_type!r}")

        target_clean = target.dropna()
        if task_type is None:
            task_type = self._infer_task_type(target_clean)

        class_balance: Dict[Any, float] | None = None
        if task_type == "classification" and not target_clean.empty:
            class_balance = target_clean.value_counts(normalize=True).to_dict()

        numeric_columns: List[str] = []
        categorical_columns: List[str] = []
        datetime_columns: List[str] = []
        dropped_columns: Dict[str, str] = {}

        for column in features.columns:
            values = features[column].dropna()
            n_unique = values.nunique()
            if values.empty:
                dropped_columns[column] = "all values missing"
            elif n_unique == 1:
                dropped_columns[column] = "same value in every row"
            elif pd.api.types.is_datetime64_any_dtype(values):
                datetime_columns.append(column)
            elif pd.api.types.is_bool_dtype(values) or pd.api.types.is_numeric_dtype(values):
                if (
                    pd.api.types.is_integer_dtype(values)
                    and n_unique == len(values)
                    and _looks_like_row_id(column, values)
                ):
                    dropped_columns[column] = "row identifier"
                else:
                    numeric_columns.append(column)
            elif _looks_like_datetime(values):
                datetime_columns.append(column)
            elif n_unique == len(values) or (
                n_unique > HIGH_CARDINALITY_MIN_UNIQUE and n_unique / len(values) > HIGH_CARDINALITY_RATIO
            ):
                dropped_columns[column] = f"identifier or free text, {n_unique:,} distinct values"
            else:
                categorical_columns.append(column)

        return DatasetSummary(
            n_rows=int(features.shape[0]),
            n_columns=int(features.shape[1]),
            task_type=task_type,
            target_name=str(target.name) if target.name else "target",
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            class_balance=class_balance,
            datetime_columns=datetime_columns,
            dropped_columns=dropped_columns,
        )

    @staticmethod
    def _infer_task_type(target: pd.Series) -> str:
        """Pick classification for labels and small sets of whole numbers, else regression."""

        if pd.api.types.is_datetime64_any_dtype(target):
            raise ValueError("Date/time target columns are not supported; choose a label or numeric column")
        if not pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target):
            return "classification"
        if target.nunique() > MAX_CLASSES_INT:
            return "regression"
        # Floats such as 0.0/1.0 (ints read next to missing values) are labels;
        # fractional values like 2.5 are measurements.
        if pd.api.types.is_integer_dtype(target) or (target % 1 == 0).all():
            return "classification"
        return "regression"
