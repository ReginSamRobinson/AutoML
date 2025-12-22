# Resource-Aware AutoML

Resource-Aware AutoML is a production-ready Python framework that automates model selection and hyperparameter tuning for tabular datasets while respecting strict resource and performance constraints.

## Key Features
- Tracks training time, peak RAM usage, and CPU utilization for every candidate model.
- Enforces user-defined constraints for time, memory, accuracy, and interpretability.
- Supports classification and regression with Logistic Regression, Elastic Net, Random Forest, Gradient Boosting, and optional XGBoost.
- Uses cross-validation and multi-metric evaluation with configurable metric priorities.
- Produces a ranked leaderboard with a combined performance/resource score and detailed constraint violations.

## Installation
1. Ensure Python 3.9 or later is available.
2. Install project dependencies:

```bash
pip install -e .[dev]
```

3. (Optional) Install XGBoost for gradient-boosted tree models with CPU acceleration:

```bash
pip install xgboost
```

## Quick Start

```python
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

dataset = load_breast_cancer(as_frame=True)
X: pd.DataFrame = dataset.data
y: pd.Series = dataset.target

engine = AutoMLEngine(
	analyzer=DatasetAnalyzer(),
	registry=ModelRegistry(),
	validator_factory=lambda c: ConstraintValidator(c),
	evaluator=Evaluator(cv_folds=5),
	monitor=ResourceMonitor(),
)

constraints = ResourceConstraints(
	max_training_time=120.0,
	max_ram_mb=4096.0,
	min_performance=0.9,
	interpretability_required=False,
	preferred_metrics=["accuracy"],
)

result = engine.run(X, y, constraints=constraints, metrics=["accuracy", "f1", "roc_auc"])

print(f"Best model: {result.best_result.model_name}")
for entry in result.leaderboard:
	print(entry)
```

For a full runnable example, execute:

```bash
python examples/run_example.py
```

## Roadmap
- Expand constraint types (e.g., energy consumption, GPU usage).
- Add Bayesian optimization and early stopping strategies.
- Provide integration hooks for experiment tracking platforms.

## License

This project is released under the MIT License. See the LICENSE file for details.
