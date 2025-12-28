"""Quick Start: AutoML Visualization UI

This script demonstrates how to use the visualization components of the AutoML framework.
"""

from automl import (
    AutoMLEngine,
    ConstraintValidator,
    DatasetAnalyzer,
    Evaluator,
    ModelRegistry,
    ResourceConstraints,
    ResourceMonitor,
    ResultVisualizer,
    run_dashboard,
    export_to_html,
)
import pandas as pd
from sklearn.datasets import load_iris


def example_1_basic_visualization():
    """Example 1: Basic visualization with ResultVisualizer"""
    print("\n" + "="*60)
    print("Example 1: Basic Visualization")
    print("="*60)
    
    # Load data
    dataset = load_iris(as_frame=True)
    X, y = dataset.data, dataset.target
    
    # Run AutoML
    engine = AutoMLEngine(
        analyzer=DatasetAnalyzer(),
        registry=ModelRegistry(),
        validator_factory=lambda c: ConstraintValidator(c),
        evaluator=Evaluator(cv_folds=3),
        monitor=ResourceMonitor(),
    )
    
    constraints = ResourceConstraints(
        max_training_time=30.0,
        max_ram_mb=1024.0,
        min_performance=0.5,
        interpretability_required=False,
    )
    
    result = engine.run(X, y, constraints=constraints, metrics=["accuracy"])
    
    # Visualize with ResultVisualizer
    visualizer = ResultVisualizer()
    
    print("\n📊 Leaderboard DataFrame:")
    leaderboard_df = visualizer.leaderboard_to_dataframe(result)
    print(leaderboard_df)
    
    print("\n⭐ Best Model:")
    best_model = visualizer.best_model_summary(result)
    for key, value in best_model.items():
        print(f"  {key}: {value}")


def example_2_html_export():
    """Example 2: Export results to HTML report"""
    print("\n" + "="*60)
    print("Example 2: HTML Export")
    print("="*60)
    
    # Load data
    dataset = load_iris(as_frame=True)
    X, y = dataset.data, dataset.target
    
    # Run AutoML
    engine = AutoMLEngine(
        analyzer=DatasetAnalyzer(),
        registry=ModelRegistry(),
        validator_factory=lambda c: ConstraintValidator(c),
        evaluator=Evaluator(cv_folds=3),
        monitor=ResourceMonitor(),
    )
    
    constraints = ResourceConstraints(
        max_training_time=30.0,
        max_ram_mb=1024.0,
    )
    
    result = engine.run(X, y, constraints=constraints, metrics=["accuracy"])
    
    # Export to HTML
    export_to_html(result, output_path="iris_results.html")
    print("\n✅ HTML report exported to: iris_results.html")
    print("   Open the file in your browser to view the results!")


def example_3_web_dashboard():
    """Example 3: Launch interactive web dashboard"""
    print("\n" + "="*60)
    print("Example 3: Interactive Web Dashboard")
    print("="*60)
    
    # Load data
    dataset = load_iris(as_frame=True)
    X, y = dataset.data, dataset.target
    
    # Run AutoML
    engine = AutoMLEngine(
        analyzer=DatasetAnalyzer(),
        registry=ModelRegistry(),
        validator_factory=lambda c: ConstraintValidator(c),
        evaluator=Evaluator(cv_folds=3),
        monitor=ResourceMonitor(),
    )
    
    constraints = ResourceConstraints(
        max_training_time=30.0,
        max_ram_mb=1024.0,
    )
    
    result = engine.run(X, y, constraints=constraints, metrics=["accuracy"])
    
    # Launch dashboard
    print("\n🚀 Starting web dashboard...")
    print("   Open browser to: http://127.0.0.1:5000")
    print("   Press Ctrl+C to stop")
    run_dashboard(result, host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    import sys
    
    print("""
    AutoML Visualization Examples
    =============================
    
    Choose an example to run:
    1. Basic Visualization (ResultVisualizer)
    2. HTML Export
    3. Web Dashboard
    """)
    
    choice = input("Enter choice (1-3) [default: 1]: ").strip() or "1"
    
    if choice == "1":
        example_1_basic_visualization()
    elif choice == "2":
        example_2_html_export()
    elif choice == "3":
        example_3_web_dashboard()
    else:
        print("Invalid choice. Running example 1...")
        example_1_basic_visualization()
