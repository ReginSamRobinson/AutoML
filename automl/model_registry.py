"""Model registry and generation logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore

    HAS_XGB = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_XGB = False


if HAS_XGB:

    class LabelEncodedXGBClassifier(XGBClassifier):
        """XGBClassifier that accepts arbitrary labels (strings, non-contiguous ints).

        XGBoost requires classes encoded as 0..n-1; this encodes on fit and
        decodes on predict so it behaves like any sklearn classifier.
        """

        def fit(self, X, y, **kwargs):
            self._label_encoder = LabelEncoder().fit(y)
            super().fit(X, self._label_encoder.transform(y), **kwargs)
            self.classes_ = self._label_encoder.classes_
            return self

        def predict(self, X, **kwargs):
            encoded = np.asarray(super().predict(X, **kwargs)).astype(int)
            return self._label_encoder.inverse_transform(encoded)

from .config import ResourceConstraints
from .dataset_analysis import DatasetSummary


def _create_one_hot_encoder() -> OneHotEncoder:
    """Instantiate an OneHotEncoder compatible with multiple sklearn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - fallback for older versions
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(summary: DatasetSummary, scale_numeric: bool) -> ColumnTransformer:
    """Create a preprocessing column transformer based on dataset metadata."""

    numeric_pipeline_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_pipeline_steps.append(("scale", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_pipeline_steps) if summary.numeric_columns else "drop"
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", _create_one_hot_encoder()),
        ]
    ) if summary.categorical_columns else "drop"

    transformers: List[tuple[str, object, Iterable[str]]] = []
    if summary.numeric_columns:
        transformers.append(("numeric", numeric_pipeline, summary.numeric_columns))
    if summary.categorical_columns:
        transformers.append(("categorical", categorical_pipeline, summary.categorical_columns))

    if not transformers:
        # If we have no metadata, fall back to passing through all columns.
        return ColumnTransformer(transformers=[], remainder="passthrough")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_xgb_classifier_pipeline(summary: DatasetSummary, random_state: int | None) -> Pipeline:
    """Construct an XGBoost classifier pipeline with dataset-aware configuration."""

    params = dict(
        booster="gbtree",
        tree_method="hist",
        n_estimators=400,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )

    n_classes = len(summary.class_balance) if summary.class_balance else 0
    if n_classes > 2:
        params.update(objective="multi:softprob", eval_metric="mlogloss", num_class=n_classes)
    else:
        params.update(objective="binary:logistic", eval_metric="logloss")

    return Pipeline(
        steps=[
            ("preprocess", _build_preprocessor(summary, scale_numeric=False)),
            ("model", LabelEncodedXGBClassifier(**params)),
        ]
    )


def _build_xgb_regressor_pipeline(summary: DatasetSummary, random_state: int | None) -> Pipeline:
    """Construct an XGBoost regressor pipeline."""

    model = XGBRegressor(
        objective="reg:squarederror",
        booster="gbtree",
        tree_method="hist",
        n_estimators=400,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocess", _build_preprocessor(summary, scale_numeric=False)),
            ("model", model),
        ]
    )


@dataclass
class ModelDescriptor:
    """Defines how to build a model pipeline and its search space."""

    name: str
    interpretable: bool
    builder: Callable[[DatasetSummary, Optional[int]], Pipeline]
    base_search_space: Dict[str, List[object]]


class ModelRegistry:
    """Keeps track of available model templates and search spaces."""

    def __init__(self, random_state: int | None = 42) -> None:
        self.random_state = random_state
        self._classification_models: Dict[str, ModelDescriptor] = {}
        self._regression_models: Dict[str, ModelDescriptor] = {}
        self._register_models()

    def _register_models(self) -> None:
        """Populate registry with supported models."""

        self._classification_models = {
            "logistic_regression": ModelDescriptor(
                name="logistic_regression",
                interpretable=True,
                builder=lambda summary, seed: Pipeline(
                    steps=[
                        ("preprocess", _build_preprocessor(summary, scale_numeric=True)),
                        (
                            "model",
                            LogisticRegression(max_iter=500, class_weight="balanced"),
                        ),
                    ]
                ),
                base_search_space={
                    "model__C": [0.01, 0.1, 1.0, 5.0],
                    "model__solver": ["lbfgs", "sag"],
                },
            ),
            "random_forest": ModelDescriptor(
                name="random_forest",
                interpretable=False,
                builder=lambda summary, seed: Pipeline(
                    steps=[
                        ("preprocess", _build_preprocessor(summary, scale_numeric=False)),
                        (
                            "model",
                            RandomForestClassifier(random_state=seed, n_jobs=-1, class_weight="balanced"),
                        ),
                    ]
                ),
                base_search_space={
                    "model__n_estimators": [150, 300, 500],
                    "model__max_depth": [None, 10, 20, 40],
                    "model__min_samples_split": [2, 5, 10],
                    "model__max_features": ["sqrt", "log2", 0.5],
                },
            ),
            "gradient_boosting": ModelDescriptor(
                name="gradient_boosting",
                interpretable=False,
                builder=lambda summary, seed: Pipeline(
                    steps=[
                        ("preprocess", _build_preprocessor(summary, scale_numeric=False)),
                        (
                            "model",
                            GradientBoostingClassifier(random_state=seed),
                        ),
                    ]
                ),
                base_search_space={
                    "model__n_estimators": [100, 200, 300],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__max_depth": [3, 4, 5],
                },
            ),
        }

        if HAS_XGB:
            self._classification_models["xgboost"] = ModelDescriptor(
                name="xgboost",
                interpretable=False,
                builder=_build_xgb_classifier_pipeline,
                base_search_space={
                    "model__n_estimators": [200, 400, 600],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__max_depth": [3, 5, 7],
                    "model__subsample": [0.6, 0.8, 1.0],
                    "model__colsample_bytree": [0.6, 0.8, 1.0],
                },
            )

        self._regression_models = {
            "elastic_net": ModelDescriptor(
                name="elastic_net",
                interpretable=True,
                builder=lambda summary, seed: Pipeline(
                    steps=[
                        ("preprocess", _build_preprocessor(summary, scale_numeric=True)),
                        (
                            "model",
                            ElasticNet(max_iter=2000, random_state=seed),
                        ),
                    ]
                ),
                base_search_space={
                    "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
                    "model__l1_ratio": [0.15, 0.3, 0.5, 0.8],
                },
            ),
            "random_forest": ModelDescriptor(
                name="random_forest",
                interpretable=False,
                builder=lambda summary, seed: Pipeline(
                    steps=[
                        ("preprocess", _build_preprocessor(summary, scale_numeric=False)),
                        (
                            "model",
                            RandomForestRegressor(random_state=seed, n_jobs=-1),
                        ),
                    ]
                ),
                base_search_space={
                    "model__n_estimators": [150, 300, 500],
                    "model__max_depth": [None, 10, 20, 40],
                    "model__min_samples_split": [2, 5, 10],
                    "model__max_features": ["sqrt", "log2", 0.5],
                },
            ),
            "gradient_boosting": ModelDescriptor(
                name="gradient_boosting",
                interpretable=False,
                builder=lambda summary, seed: Pipeline(
                    steps=[
                        ("preprocess", _build_preprocessor(summary, scale_numeric=False)),
                        (
                            "model",
                            GradientBoostingRegressor(random_state=seed),
                        ),
                    ]
                ),
                base_search_space={
                    "model__n_estimators": [100, 200, 300],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__max_depth": [3, 4, 5],
                },
            ),
        }

        if HAS_XGB:
            self._regression_models["xgboost"] = ModelDescriptor(
                name="xgboost",
                interpretable=False,
                builder=_build_xgb_regressor_pipeline,
                base_search_space={
                    "model__n_estimators": [200, 400, 600],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                    "model__max_depth": [3, 5, 7],
                    "model__subsample": [0.6, 0.8, 1.0],
                    "model__colsample_bytree": [0.6, 0.8, 1.0],
                },
            )

    def list_models(self, task_type: str, interpretability_first: bool) -> List[str]:
        """Return ordered model names for the given task."""

        registry = self._classification_models if task_type == "classification" else self._regression_models
        interpretable = [name for name, desc in registry.items() if desc.interpretable]
        complex_models = [name for name, desc in registry.items() if not desc.interpretable]

        if interpretability_first:
            ordered = interpretable + [name for name in complex_models if name not in interpretable]
        else:
            ordered = list(registry.keys())

        return [name for name in ordered if name in registry]

    def build_estimator(self, model_name: str, summary: DatasetSummary) -> Pipeline:
        """Instantiate a model pipeline by name."""

        descriptor = self._get_descriptor(model_name, summary.task_type)
        return descriptor.builder(summary, self.random_state)

    def is_interpretable(self, model_name: str, task_type: str) -> bool:
        """Return whether the model is considered interpretable for the task."""

        descriptor = self._get_descriptor(model_name, task_type)
        return descriptor.interpretable

    def get_search_space(
        self,
        model_name: str,
        constraints: ResourceConstraints,
        summary: DatasetSummary,
    ) -> Dict[str, List[object]]:
        """Return the hyperparameter search configuration for a model."""

        descriptor = self._get_descriptor(model_name, summary.task_type)
        space = {param: list(values) for param, values in descriptor.base_search_space.items()}
        return self._adapt_search_space(model_name, space, constraints, summary)

    def _adapt_search_space(
        self,
        model_name: str,
        space: Dict[str, List[object]],
        constraints: ResourceConstraints,
        summary: DatasetSummary,
    ) -> Dict[str, List[object]]:
        """Adjust search space based on constraints and dataset metadata."""

        max_time = constraints.max_training_time or np.inf
        max_ram = constraints.max_ram_mb or np.inf

        def limit(param: str, max_len: int) -> None:
            if param in space and len(space[param]) > max_len:
                space[param] = space[param][:max_len]

        if model_name in {"random_forest", "gradient_boosting", "xgboost"}:
            if max_time <= 120 or summary.n_rows > 100_000:
                limit("model__n_estimators", 2)
            if max_time <= 60 or max_ram <= 1024:
                limit("model__n_estimators", 1)
                if "model__max_depth" in space:
                    space["model__max_depth"] = [val for val in space["model__max_depth"] if val is None or (isinstance(val, int) and val <= 10)]
            if constraints.interpretability_required and "model__max_depth" in space:
                space["model__max_depth"] = [val for val in space["model__max_depth"] if val is None or (isinstance(val, int) and val <= 8)]

        if model_name == "xgboost" and constraints.interpretability_required:
            space["model__max_depth"] = [val for val in space.get("model__max_depth", []) if isinstance(val, int) and val <= 5]

        if model_name == "logistic_regression" and summary.n_rows > 50_000:
            if "model__solver" in space and "saga" not in space["model__solver"]:
                space["model__solver"].append("saga")

        if model_name == "elastic_net":
            if max_time <= 60:
                limit("model__alpha", 3)
            if constraints.interpretability_required:
                space["model__l1_ratio"] = [val for val in space["model__l1_ratio"] if val <= 0.5]

        return space

    def _get_descriptor(self, model_name: str, task_type: str) -> ModelDescriptor:
        """Fetch descriptor for model ensuring it exists."""

        registry = self._classification_models if task_type == "classification" else self._regression_models
        if model_name not in registry:
            raise ValueError(f"Model '{model_name}' is not registered for task '{task_type}'.")
        return registry[model_name]
