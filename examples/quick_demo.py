"""Quick demo of the visualization UI with minimal computation."""

from automl import ResultVisualizer, export_to_html, run_dashboard
from automl.engine import AutoMLResult
from automl.leaderboard import LeaderboardEntry
from automl.evaluation import EvaluationResult
from automl.resource_monitor import ResourceUsage
import warnings

warnings.filterwarnings('ignore')

# Create mock results for visualization demo
mock_results = []

# Mock evaluation result 1
eval1 = EvaluationResult(
    model_name="Random Forest",
    estimator=None,
    params={"n_estimators": 100, "max_depth": 10},
    metrics={"accuracy": 0.945, "f1": 0.942, "roc_auc": 0.978},
    metric_preferences={"accuracy": True, "f1": True, "roc_auc": True},
    primary_metric="accuracy",
    resource_usage=ResourceUsage(
        training_time=2.34,
        peak_ram_mb=128.5,
        avg_cpu_percent=85.2
    ),
    violations=[]
)

# Mock evaluation result 2
eval2 = EvaluationResult(
    model_name="XGBoost",
    estimator=None,
    params={"n_estimators": 150, "max_depth": 7},
    metrics={"accuracy": 0.938, "f1": 0.935, "roc_auc": 0.971},
    metric_preferences={"accuracy": True, "f1": True, "roc_auc": True},
    primary_metric="accuracy",
    resource_usage=ResourceUsage(
        training_time=3.21,
        peak_ram_mb=256.3,
        avg_cpu_percent=92.1
    ),
    violations=[]
)

# Mock evaluation result 3
eval3 = EvaluationResult(
    model_name="Logistic Regression",
    estimator=None,
    params={"C": 1.0, "max_iter": 500},
    metrics={"accuracy": 0.912, "f1": 0.908, "roc_auc": 0.954},
    metric_preferences={"accuracy": True, "f1": True, "roc_auc": True},
    primary_metric="accuracy",
    resource_usage=ResourceUsage(
        training_time=0.45,
        peak_ram_mb=64.2,
        avg_cpu_percent=45.3
    ),
    violations=[]
)

# Create leaderboard entries
entries = [
    LeaderboardEntry(rank=1, result=eval1, combined_score=0.945),
    LeaderboardEntry(rank=2, result=eval2, combined_score=0.938),
    LeaderboardEntry(rank=3, result=eval3, combined_score=0.912),
]

# Create mock result
result = AutoMLResult(
    best_result=eval1,
    best_estimator=None,
    leaderboard=entries
)

print("="*60)
print("AutoML Visualization UI Demo")
print("="*60)

# Visualize with ResultVisualizer
print("\nGenerating visualizations...")
visualizer = ResultVisualizer()

print("\nLeaderboard DataFrame:")
print(visualizer.leaderboard_to_dataframe(result))

print("\nBest Model Info:")
best_info = visualizer.best_model_summary(result)
for key, value in best_info.items():
    if key != "All Metrics":
        print(f"  {key}: {value}")

# Export HTML
print("\nExporting HTML report...")
export_to_html(result, "automl_demo_report.html")

# Launch dashboard
print("\nLaunching interactive dashboard...")
print("="*60)
print("Opening http://127.0.0.1:5000 in your browser...")
print("Press Ctrl+C to stop the server")
print("="*60)
run_dashboard(result, host="127.0.0.1", port=5000, debug=False)
