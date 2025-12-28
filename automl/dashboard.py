"""Flask web UI for visualizing AutoML results with advanced professional design."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, jsonify
from plotly.subplots import make_subplots

from .engine import AutoMLResult
from .visualizer import ResultVisualizer


def create_app(result: Optional[AutoMLResult] = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.route("/")
    def index():
        """Render the main dashboard."""
        if result is None:
            return "<h1>No AutoML results to display</h1>"

        visualizer = ResultVisualizer()
        leaderboard_df = visualizer.leaderboard_to_dataframe(result)
        resource_df = visualizer.resource_usage_summary(result)
        best_model = visualizer.best_model_summary(result)

        html_content = _build_html_dashboard(leaderboard_df, resource_df, best_model, result)
        return html_content

    @app.route("/api/leaderboard")
    def api_leaderboard():
        """API endpoint for leaderboard data."""
        if result is None:
            return jsonify([])

        visualizer = ResultVisualizer()
        leaderboard_df = visualizer.leaderboard_to_dataframe(result)
        return jsonify(leaderboard_df.to_dict("records"))

    @app.route("/api/charts/leaderboard")
    def chart_leaderboard():
        """Generate leaderboard chart."""
        if result is None or not result.leaderboard:
            return jsonify({"error": "No data"})

        visualizer = ResultVisualizer()
        leaderboard_df = visualizer.leaderboard_to_dataframe(result)

        fig = px.bar(
            leaderboard_df,
            x="Model",
            y="Combined Score",
            title="Model Rankings by Combined Score",
            labels={"Combined Score": "Score", "Model": "Model"},
            color="Combined Score",
            color_continuous_scale=["#ff6b6b", "#ffd93d", "#6bcf7f"],
        )
        fig.update_layout(
            showlegend=False,
            height=380,
            xaxis_title="",
            yaxis_title="Combined Score",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            hovermode="x unified",
        )
        fig.update_traces(hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>")
        return fig.to_json()

    @app.route("/api/charts/metrics")
    def chart_metrics():
        """Generate metrics comparison chart."""
        if result is None or not result.leaderboard:
            return jsonify({"error": "No data"})

        visualizer = ResultVisualizer()
        metrics_summary = visualizer.metrics_summary(result)

        if not metrics_summary:
            return jsonify({"error": "No metrics"})

        # Get primary metric for all models
        primary_metric = result.leaderboard[0].result.primary_metric.upper()
        models = list(metrics_summary.keys())
        primary_scores = [metrics_summary[m].get(primary_metric, 0) for m in models]

        fig = px.bar(
            x=models,
            y=primary_scores,
            title=f"Primary Metric ({primary_metric}) Performance",
            labels={"x": "Model", "y": primary_metric},
            color=primary_scores,
            color_continuous_scale="Viridis",
        )
        fig.update_traces(
            texttemplate="%{y:.4f}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>" + primary_metric + ": %{y:.4f}<extra></extra>",
        )
        fig.update_layout(
            showlegend=False,
            height=380,
            xaxis_title="",
            yaxis_title=primary_metric,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            hovermode="x unified",
        )
        return fig.to_json()

    @app.route("/api/charts/resources")
    def chart_resources():
        """Generate resource usage chart."""
        if result is None or not result.leaderboard:
            return jsonify({"error": "No data"})

        visualizer = ResultVisualizer()
        resource_df = visualizer.resource_usage_summary(result)

        if resource_df.empty:
            return jsonify({"error": "No data"})

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"secondary_y": False}, {"secondary_y": False}]],
            subplot_titles=("Training Time", "Peak RAM Usage"),
        )

        fig.add_trace(
            go.Bar(
                x=resource_df["Model"],
                y=resource_df["Training Time (s)"],
                name="Time (s)",
                marker_color="#667eea",
                hovertemplate="<b>%{x}</b><br>Time: %{y:.2f}s<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=resource_df["Model"],
                y=resource_df["Peak RAM (MB)"],
                name="RAM (MB)",
                marker_color="#764ba2",
                hovertemplate="<b>%{x}</b><br>RAM: %{y:.0f}MB<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_xaxes(title_text="", row=1, col=2)
        fig.update_yaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="RAM (MB)", row=1, col=2)
        fig.update_layout(
            height=380,
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            hovermode="x unified",
        )
        return fig.to_json()

    return app


def _build_html_dashboard(
    leaderboard_df: pd.DataFrame,
    resource_df: pd.DataFrame,
    best_model: dict[str, object],
    result: AutoMLResult,
) -> str:
    """Build advanced professional HTML dashboard."""

    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AutoML Advanced Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --primary: #667eea;
                --secondary: #764ba2;
                --accent: #f093fb;
                --success: #11d4aa;
                --warning: #ff9800;
                --danger: #f44336;
                --dark: #1a1a2e;
                --light: #f8f9fa;
                --text: #2c3e50;
                --text-light: #7f8c8d;
                --border: #ecf0f1;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 25px 15px;
                color: var(--text);
            }
            
            .container {
                max-width: 1800px;
                margin: 0 auto;
            }
            
            .navbar {
                background: white;
                padding: 20px 30px;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.9);
            }
            
            .navbar-brand {
                font-size: 1.8em;
                font-weight: 800;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .stat-card {
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                border-left: 5px solid var(--primary);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
            }
            
            .stat-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
                border-left-color: var(--secondary);
            }
            
            .stat-label {
                font-size: 0.85em;
                color: var(--text-light);
                margin-bottom: 10px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1.2px;
            }
            
            .stat-value {
                font-size: 2.2em;
                font-weight: 900;
                color: var(--primary);
                margin-bottom: 5px;
            }
            
            .stat-unit {
                font-size: 0.75em;
                color: var(--text-light);
                font-weight: 600;
            }
            
            .hero-section {
                background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                padding: 50px 40px;
                border-radius: 18px;
                color: white;
                margin-bottom: 30px;
                box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4);
                position: relative;
                overflow: hidden;
            }
            
            .hero-section::before {
                content: '';
                position: absolute;
                top: -50%;
                right: -10%;
                width: 500px;
                height: 500px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 50%;
            }
            
            .hero-content {
                position: relative;
                z-index: 1;
            }
            
            .hero-title {
                font-size: 2.8em;
                font-weight: 900;
                margin-bottom: 15px;
                letter-spacing: -1px;
            }
            
            .hero-subtitle {
                font-size: 1.15em;
                opacity: 0.95;
                margin-bottom: 30px;
                font-weight: 500;
            }
            
            .best-model-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
            }
            
            .model-metric {
                background: rgba(255, 255, 255, 0.12);
                padding: 22px;
                border-radius: 12px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: all 0.3s ease;
            }
            
            .model-metric:hover {
                background: rgba(255, 255, 255, 0.18);
                transform: translateY(-3px);
            }
            
            .model-metric-label {
                font-size: 0.9em;
                opacity: 0.9;
                margin-bottom: 10px;
                font-weight: 700;
            }
            
            .model-metric-value {
                font-size: 2em;
                font-weight: 900;
            }
            
            .dashboard-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(550px, 1fr));
                gap: 25px;
                margin-bottom: 30px;
            }
            
            .card {
                background: white;
                border-radius: 18px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                padding: 30px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                border: 1px solid var(--border);
            }
            
            .card:hover {
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.12);
                border-color: var(--primary);
                transform: translateY(-5px);
            }
            
            .card-header {
                display: flex;
                align-items: center;
                margin-bottom: 25px;
                padding-bottom: 20px;
                border-bottom: 2px solid var(--light);
            }
            
            .card-icon {
                font-size: 2em;
                margin-right: 15px;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .card-title {
                font-size: 1.4em;
                font-weight: 800;
                color: var(--text);
            }
            
            .chart-container {
                height: 400px;
                margin: -15px -30px 0 -30px;
            }
            
            .table-wrapper {
                overflow-x: auto;
                border-radius: 12px;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.95em;
            }
            
            table thead {
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
            }
            
            table th {
                padding: 18px 14px;
                text-align: left;
                font-weight: 700;
                text-transform: uppercase;
                font-size: 0.8em;
                letter-spacing: 1px;
            }
            
            table td {
                padding: 16px 14px;
                border-bottom: 1px solid var(--border);
                font-weight: 500;
            }
            
            table tbody tr {
                transition: all 0.2s ease;
            }
            
            table tbody tr:hover {
                background: linear-gradient(90deg, rgba(102, 126, 234, 0.08), transparent);
            }
            
            table tbody tr:nth-child(even) {
                background: var(--light);
            }
            
            .rank-badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
                font-weight: 900;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }
            
            .metric-pill {
                display: inline-block;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                color: var(--primary);
                padding: 8px 14px;
                border-radius: 20px;
                font-size: 0.85em;
                font-weight: 700;
                margin: 3px;
                border: 1.5px solid var(--primary);
            }
            
            .footer {
                text-align: center;
                color: rgba(255, 255, 255, 0.8);
                margin-top: 50px;
                padding: 30px 20px;
            }
            
            .footer strong {
                color: white;
                font-weight: 700;
            }
            
            @media (max-width: 1024px) {
                .dashboard-grid {
                    grid-template-columns: 1fr;
                }
                .hero-title {
                    font-size: 2em;
                }
            }
            
            @media (max-width: 768px) {
                .stats-grid {
                    grid-template-columns: repeat(2, 1fr);
                }
                .card {
                    padding: 20px;
                }
                .chart-container {
                    height: 300px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="navbar">
                <div class="navbar-brand">
                    <i class="fas fa-robot"></i> AutoML Pro
                </div>
                <div style="font-size: 0.9em; color: var(--text-light);">
                    Advanced Machine Learning Dashboard v2.0
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label"><i class="fas fa-chart-line"></i> Total Models</div>
                    <div class="stat-value">""" + str(len(result.leaderboard)) + """</div>
                </div>
                <div class="stat-card" style="border-left-color: var(--success);">
                    <div class="stat-label"><i class="fas fa-crown"></i> Best Score</div>
                    <div class="stat-value">""" + (f"{best_model.get('Primary Metric Score', 0):.3f}" if not best_model.get("status") else "N/A") + """</div>
                </div>
                <div class="stat-card" style="border-left-color: var(--warning);">
                    <div class="stat-label"><i class="fas fa-hourglass"></i> Avg Time</div>
                    <div class="stat-value">""" + f"{resource_df['Training Time (s)'].mean():.1f}" + """<span class="stat-unit">s</span></div>
                </div>
                <div class="stat-card" style="border-left-color: var(--accent);">
                    <div class="stat-label"><i class="fas fa-database"></i> Peak Memory</div>
                    <div class="stat-value">""" + f"{resource_df['Peak RAM (MB)'].max():.0f}" + """<span class="stat-unit">MB</span></div>
                </div>
            </div>
            
            <div class="hero-section">
                <div class="hero-content">
                    <div class="hero-title">🏆 Performance Summary</div>
                    <div class="hero-subtitle">Comprehensive AutoML Evaluation Results</div>
    """

    if best_model.get("status"):
        html += f"<p style='font-size: 1.1em; opacity: 0.95;'>{best_model['status']}</p>"
    else:
        html += f"""
                    <div class="best-model-grid">
                        <div class="model-metric">
                            <div class="model-metric-label">✨ Top Model</div>
                            <div class="model-metric-value">{best_model.get('Model', 'N/A')}</div>
                        </div>
                        <div class="model-metric">
                            <div class="model-metric-label">⭐ {best_model.get('Primary Metric', 'Accuracy').upper()}</div>
                            <div class="model-metric-value">{best_model.get('Primary Metric Score', 0):.4f}</div>
                        </div>
                        <div class="model-metric">
                            <div class="model-metric-label">⏱️ Training</div>
                            <div class="model-metric-value">{best_model.get('Training Time (s)', 0):.2f}s</div>
                        </div>
                        <div class="model-metric">
                            <div class="model-metric-label">💾 Memory</div>
                            <div class="model-metric-value">{best_model.get('Peak RAM (MB)', 0):.0f}MB</div>
                        </div>
                    </div>
        """

    html += """
                </div>
            </div>
            
            <div class="dashboard-grid">
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon"><i class="fas fa-chart-bar"></i></div>
                        <div class="card-title">Model Rankings</div>
                    </div>
                    <div class="chart-container" id="chart-leaderboard"></div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon"><i class="fas fa-star"></i></div>
                        <div class="card-title">Metric Performance</div>
                    </div>
                    <div class="chart-container" id="chart-metrics"></div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <div class="card-icon"><i class="fas fa-tachometer-alt"></i></div>
                    <div class="card-title">Resource Efficiency</div>
                </div>
                <div class="chart-container" id="chart-resources"></div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <div class="card-icon"><i class="fas fa-table"></i></div>
                    <div class="card-title">Complete Leaderboard</div>
                </div>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
    """

    for col in leaderboard_df.columns:
        icon = ""
        if col == "Rank":
            icon = "🏅"
        elif col == "Model":
            icon = "🤖"
        elif "Score" in col:
            icon = "⭐"
        elif "Time" in col:
            icon = "⏱️"
        elif "RAM" in col:
            icon = "💾"
        html += f"<th>{icon} {col}</th>"

    html += """
                            </tr>
                        </thead>
                        <tbody>
    """

    for idx, row in leaderboard_df.iterrows():
        html += "<tr>"
        for col_name, value in zip(leaderboard_df.columns, row):
            if col_name == "Rank":
                html += f'<td><span class="rank-badge">{value}</span></td>'
            elif col_name in ["ACCURACY", "F1", "ROC_AUC"] and isinstance(value, (int, float)):
                html += f'<td><span class="metric-pill">{col_name.replace("_", " ")}: {value:.4f}</span></td>'
            else:
                html += f"<td>{value}</td>"
        html += "</tr>"

    html += """
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="footer">
                <p><strong>AutoML Framework</strong> - Advanced Machine Learning Pipeline</p>
                <p style="margin-top: 8px; opacity: 0.8;">Resource-Aware • Constraint-Driven • Performance-Optimized</p>
                <p style="margin-top: 8px; font-size: 0.85em; opacity: 0.7;">""" + pd.Timestamp.now().strftime("%B %d, %Y at %H:%M:%S") + """</p>
            </div>
        </div>

        <script>
            async function loadCharts() {
                try {
                    const [leaderboard, metrics, resources] = await Promise.all([
                        fetch('/api/charts/leaderboard').then(r => r.json()),
                        fetch('/api/charts/metrics').then(r => r.json()),
                        fetch('/api/charts/resources').then(r => r.json())
                    ]);

                    const commonLayout = {
                        font: { family: 'Inter, sans-serif', color: '#2c3e50', size: 12 },
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        margin: { t: 20, r: 20, b: 30, l: 60 }
                    };

                    Plotly.newPlot('chart-leaderboard', leaderboard.data, {...leaderboard.layout, ...commonLayout}, {responsive: true, displayModeBar: false});
                    Plotly.newPlot('chart-metrics', metrics.data, {...metrics.layout, ...commonLayout}, {responsive: true, displayModeBar: false});
                    Plotly.newPlot('chart-resources', resources.data, {...resources.layout, ...commonLayout}, {responsive: true, displayModeBar: false});
                } catch (error) {
                    console.error('Error loading charts:', error);
                }
            }

            window.addEventListener('load', loadCharts);
        </script>
    </body>
    </html>
    """

    return html


def run_dashboard(result: AutoMLResult, host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    """Run the dashboard server."""
    app = create_app(result)
    print(f"\n{'='*60}")
    print(f"🚀 Starting Advanced AutoML Dashboard")
    print(f"📍 Access at: http://{host}:{port}")
    print(f"⏹️  Press Ctrl+C to stop the server")
    print(f"{'='*60}\n")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
