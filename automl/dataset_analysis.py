"""Dataset analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


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


class DatasetAnalyzer:
    """Analyzes datasets to infer metadata for AutoML."""

    def summarize(self, features: pd.DataFrame, target: pd.Series) -> DatasetSummary:
        """Generate a dataset summary from feature matrix and target series."""

        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if not isinstance(target, pd.Series):
            target = pd.Series(target)

        target_clean = target.dropna()
        unique_values = target_clean.unique()
        n_unique = unique_values.size
        target_kind = target_clean.dtype.kind if not target_clean.empty else target.dtype.kind

        is_categorical = False
        if target_kind in {"O", "b"}:
            is_categorical = True
        elif target_kind in {"i", "u"} and n_unique <= 15:
            is_categorical = True
        elif target_kind in {"f"} and n_unique <= 10:
            is_categorical = True
        elif target_kind == "M":
            is_categorical = True

        task_type = "classification" if is_categorical else "regression"

        class_balance: Dict[Any, float] | None = None
        if task_type == "classification" and n_unique > 0:
            counts = target_clean.value_counts(normalize=True)
            class_balance = counts.to_dict()

        numeric_columns = [col for col in features.columns if pd.api.types.is_numeric_dtype(features[col])]
        categorical_columns = [col for col in features.columns if col not in numeric_columns]

        return DatasetSummary(
            n_rows=int(features.shape[0]),
            n_columns=int(features.shape[1]),
            task_type=task_type,
            target_name=str(target.name) if target.name else "target",
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            class_balance=class_balance,
        )
