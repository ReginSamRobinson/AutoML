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
2. Install the core engine plus the extras you need:

```bash
pip install -e .                 # core engine only
pip install -e ".[dashboard]"    # + Flask/Plotly web dashboard, xlsx/parquet uploads
pip install -e ".[xgboost]"      # + XGBoost models
pip install -e ".[qt]"           # + native Qt window (PySide6)
pip install -e ".[desktop]"      # + pywebview window
pip install -e ".[dev]"          # everything needed to run the tests
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

### Constraint semantics

| Field | Meaning |
| --- | --- |
| `max_training_time` | Seconds for one candidate, all CV folds together |
| `max_ram_mb` | Extra memory a candidate uses above the process baseline |
| `min_performance` | Lower bound for score metrics (accuracy, f1, …); upper bound for error metrics (rmse, mae, …). Ignored with a warning if its metric does not fit the task |
| `max_total_time` | Seconds for the whole search; no new candidates start once spent |

Metrics that don't fit the detected task (e.g. `accuracy` on a regression target) are skipped with a warning, falling back to `accuracy` / `rmse`.

## Dashboard & Visualization

The AutoML framework includes an interactive web-based dashboard for visualizing results.

### Features
- **Setup page**: upload datasets (or drag and drop), pick the target column, set constraints, and start a run
- **Selected model**: the chosen model, its parameters and key figures, compared with the next best model type
- **Trade-off chart**: primary metric vs. training time for every candidate; hollow points broke a limit
- **Resource use**: training time, extra memory and CPU for the best configuration of each model type
- **Leaderboard**: sortable table of every candidate with metrics, resource use and which limits it broke
- Light and dark themes; works offline (no external fonts or CDNs)

The UI lives in `automl/templates/` (Jinja) and `automl/static/` (CSS/JS).

### Using the Dashboard

The example script automatically launches the dashboard after completing the AutoML run:

```bash
python examples/run_example.py
```

Then open your browser to **http://127.0.0.1:5000** to view the results.

### Programmatic Usage

```python
from automl import run_dashboard, AutoMLEngine, ...

# ... run AutoML engine ...
result = engine.run(X, y, constraints=constraints, metrics=metrics)

# Launch dashboard
run_dashboard(result, host="127.0.0.1", port=5000, debug=False)
```

### API Endpoints

- `GET /` - Main dashboard page with embedded charts
- `GET /api/leaderboard` - JSON array of leaderboard entries
- `GET /api/charts/leaderboard` - Plotly JSON for rankings chart
- `GET /api/charts/metrics` - Plotly JSON for metrics chart
- `GET /api/charts/resources` - Plotly JSON for resource usage chart

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
