"""Flask web UI for visualizing AutoML results with an explainable dark-themed dashboard."""

from __future__ import annotations

import html
import math
import threading
from typing import Optional
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from flask import Flask, jsonify, request, render_template_string
from plotly.subplots import make_subplots
from werkzeug.serving import make_server
from werkzeug.utils import secure_filename

from .engine import AutoMLResult
from .visualizer import ResultVisualizer
from .data_upload import allowed_file, load_dataset, get_dataset_preview, validate_dataset, create_upload_folder, ALLOWED_EXTENSIONS


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

        return _build_html_dashboard(leaderboard_df, resource_df, best_model, result)

    @app.route("/api/leaderboard")
    def api_leaderboard():
        """JSON payload for leaderboard values."""
        if result is None:
            return jsonify([])

        visualizer = ResultVisualizer()
        leaderboard_df = visualizer.leaderboard_to_dataframe(result)
        return jsonify(leaderboard_df.to_dict("records"))

    @app.route("/api/charts/leaderboard")
    def chart_leaderboard():
        """Generate final decision chart."""
        if result is None or not result.leaderboard:
            return jsonify({"error": "No data"})

        visualizer = ResultVisualizer()
        leaderboard_df = visualizer.leaderboard_to_dataframe(result)
        if leaderboard_df.empty or "Combined Score" not in leaderboard_df.columns:
            return jsonify({"error": "No data"})

        resource_df = visualizer.resource_usage_summary(result)

        combined_scores = pd.to_numeric(leaderboard_df["Combined Score"], errors="coerce").fillna(0.0)
        score_range = combined_scores.max() - combined_scores.min()
        if score_range > 1e-9:
            normalized_series = (combined_scores - combined_scores.min()) / score_range
        else:
            normalized_series = pd.Series([1.0] * len(combined_scores), index=combined_scores.index)

        normalized_values = normalized_series.tolist()
        models = leaderboard_df["Model"].tolist()
        ranks = leaderboard_df["Rank"].tolist()
        winner_idx = combined_scores.idxmax()
        winner_model = leaderboard_df.loc[winner_idx, "Model"]

        colors = [
            "rgba(95, 227, 192, 1)" if model == winner_model else "rgba(148, 163, 184, 0.45)"
            for model in models
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=models,
                    y=normalized_values,
                    marker=dict(color=colors),
                    customdata=list(zip(models, combined_scores.tolist(), ranks)),
                    text=[f"{score:.3f}" for score in combined_scores],
                    textposition="outside",
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                    "Combined Score: %{customdata[1]:.3f}<br>"
                    "Normalized: %{y:.2f}<extra></extra>",
                )
            ]
        )

        annotations = []
        if not resource_df.empty:
            time_series = pd.to_numeric(resource_df["Training Time (s)"], errors="coerce")
            ram_series = pd.to_numeric(resource_df["Peak RAM (MB)"], errors="coerce")
            time_threshold = time_series.quantile(0.75) if time_series.notna().any() else None
            ram_threshold = ram_series.quantile(0.75) if ram_series.notna().any() else None

            for idx, model in enumerate(models):
                if model == winner_model:
                    continue
                resource_row = resource_df[resource_df["Model"] == model]
                if resource_row.empty:
                    continue
                time_value = pd.to_numeric(resource_row["Training Time (s)"], errors="coerce").iloc[0]
                ram_value = pd.to_numeric(resource_row["Peak RAM (MB)"], errors="coerce").iloc[0]
                high_cost = False
                if (
                    time_threshold is not None
                    and pd.notna(time_value)
                    and time_value > time_threshold
                ):
                    high_cost = True
                if (
                    ram_threshold is not None
                    and pd.notna(ram_value)
                    and ram_value > ram_threshold
                ):
                    high_cost = True
                if high_cost:
                    annotations.append(
                        dict(
                            x=model,
                            y=min(normalized_values[idx] + 0.08, 1.08),
                            text="High Cost",
                            showarrow=False,
                            font=dict(color="#f87171", size=11),
                        )
                    )

        fig.update_layout(
            title="Combined Score (Normalized)",
            height=380,
            xaxis=dict(title="", showgrid=False, tickangle=-20),
            yaxis=dict(
                title="Normalized Combined Score",
                range=[0, 1.15],
                showgrid=False,
                zeroline=False,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#e6edf7"),
            hovermode="closest",
        )
        if annotations:
            fig.update_layout(annotations=annotations)

        return fig.to_json()

    @app.route("/api/charts/metrics")
    def chart_metrics():
        """Generate model quality chart."""
        if result is None or not result.leaderboard:
            return jsonify({"error": "No data"})

        visualizer = ResultVisualizer()
        leaderboard_df = visualizer.leaderboard_to_dataframe(result)
        metrics_summary = visualizer.metrics_summary(result)

        if leaderboard_df.empty or not metrics_summary:
            return jsonify({"error": "No data"})

        def extract_model_family(name: str) -> str:
            candidate = name.split("(")[0].split("-")[0].strip()
            return candidate or name

        grouped_best: dict[str, pd.Series] = {}
        for _, row in leaderboard_df.sort_values(by="Rank").iterrows():
            model_name = str(row["Model"])
            family = extract_model_family(model_name)
            if family not in grouped_best:
                grouped_best[family] = row

        if not grouped_best:
            return jsonify({"error": "No data"})

        best_rows = list(grouped_best.values())
        models = [str(row["Model"]) for row in best_rows]

        winner_model = (
            max(best_rows, key=lambda row: row.get("Combined Score", 0)).get("Model")
            if best_rows
            else None
        )

        metric_specs = [
            ("accuracy", "Accuracy"),
            ("f1", "F1 Score"),
            ("roc_auc", "ROC-AUC"),
        ]

        x_labels = [label for _, label in metric_specs]
        palette = [
            "#38bdf8",
            "#a855f7",
            "#f9a30b",
            "#f87171",
            "#14b8a6",
            "#818cf8",
            "#fb7185",
            "#22d3ee",
        ]

        fig = go.Figure()
        has_trace = False

        for index, model in enumerate(models):
            metrics = metrics_summary.get(model, {})
            lower_map = {k.lower(): v for k, v in metrics.items()}
            raw_values: list[float | None] = []
            for metric_key, _ in metric_specs:
                raw_values.append(lower_map.get(metric_key))

            series = pd.to_numeric(pd.Series(raw_values, index=x_labels), errors="coerce")
            if not series.notna().any():
                continue

            has_trace = True
            color = "#5fe3c0" if model == winner_model else palette[index % len(palette)]
            opacity = 1.0 if model == winner_model else 0.78

            fig.add_trace(
                go.Bar(
                    name=model,
                    x=x_labels,
                    y=series.fillna(0).tolist(),
                    marker=dict(color=color, opacity=opacity),
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    "Metric: %{x}<br>Value: %{y:.3f}<extra></extra>",
                )
            )

        if not has_trace:
            return jsonify({"error": "No metrics"})

        fig.update_layout(
            title="Model Quality Metrics",
            barmode="group",
            height=380,
            xaxis=dict(title="", showgrid=False),
            yaxis=dict(
                title="Metric Value",
                range=[0, 1.05],
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.18)",
                zeroline=False,
            ),
            legend=dict(title="Model", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#e6edf7"),
        )
        return fig.to_json()

    @app.route("/api/charts/resources")
    def chart_resources():
        """Generate resource cost chart."""
        if result is None or not result.leaderboard:
            return jsonify({"error": "No data"})

        visualizer = ResultVisualizer()
        resource_df = visualizer.resource_usage_summary(result)
        leaderboard_df = visualizer.leaderboard_to_dataframe(result)

        if resource_df.empty:
            return jsonify({"error": "No data"})

        models = resource_df["Model"].tolist()
        if not models:
            return jsonify({"error": "No data"})

        winner_model = (
            leaderboard_df.loc[leaderboard_df["Combined Score"].idxmax(), "Model"]
            if not leaderboard_df.empty and "Combined Score" in leaderboard_df.columns
            else models[0]
        )

        metric_candidates = [
            (
                "Training Time (s)",
                "Training Time",
                "s",
                "rgba(249, 158, 11, 0.9)",
                "rgba(148, 163, 184, 0.35)",
            ),
            (
                "Peak RAM (MB)",
                "Peak RAM",
                "MB",
                "rgba(168, 85, 247, 0.9)",
                "rgba(148, 163, 184, 0.35)",
            ),
            (
                "Avg CPU (%)",
                "CPU Usage",
                "%",
                "rgba(56, 189, 248, 0.9)",
                "rgba(148, 163, 184, 0.35)",
            ),
        ]

        prepared_metrics: list[tuple[str, str, str, str, str, pd.Series]] = []
        for column, title, unit, accent_color, muted_color in metric_candidates:
            if column not in resource_df.columns:
                continue
            series = pd.to_numeric(resource_df[column], errors="coerce")
            if not series.notna().any():
                continue
            prepared_metrics.append((column, title, unit, accent_color, muted_color, series))

        if not prepared_metrics:
            return jsonify({"error": "No data"})

        fig = make_subplots(
            rows=1,
            cols=len(prepared_metrics),
            subplot_titles=[f"{title} (Normalized)" for _, title, _, _, _, _ in prepared_metrics],
        )

        annotations = []

        for idx, (column, title, unit, accent_color, muted_color, series) in enumerate(
            prepared_metrics, start=1
        ):
            max_value = series.max()
            normalized = series / max_value if max_value and max_value > 0 else series * 0
            normalized_values = normalized.tolist()
            actual_values = series.tolist()
            colors = [
                accent_color if model == winner_model else muted_color for model in models
            ]
            unit_suffix = "%" if unit == "%" else f" {unit}"
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=normalized_values,
                    marker=dict(color=colors),
                    customdata=list(zip(models, actual_values)),
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                    f"{title}: " + "%{customdata[1]:.2f}" + unit_suffix + "<br>"
                    "Normalized: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=idx,
            )

            threshold = series.quantile(0.85) if series.notna().any() else None
            if threshold is not None and math.isfinite(threshold):
                for model, norm_val, actual_val in zip(models, normalized_values, actual_values):
                    if model == winner_model:
                        continue
                    if pd.notna(actual_val) and actual_val > threshold:
                        annotations.append(
                            dict(
                                x=model,
                                y=min(norm_val + 0.08, 1.08),
                                xref=f"x{idx}",
                                yref=f"y{idx}",
                                text="High Cost",
                                showarrow=False,
                                font=dict(color="#f87171", size=10),
                            )
                        )

        fig.update_layout(
            height=380,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#e6edf7"),
            hovermode="closest",
        )

        for axis_idx in range(1, len(prepared_metrics) + 1):
            fig.update_xaxes(row=1, col=axis_idx, title="", showgrid=False, tickangle=-20)
            fig.update_yaxes(
                row=1,
                col=axis_idx,
                title="Normalized",
                range=[0, 1.15],
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.15)",
                zeroline=False,
            )

        if annotations:
            fig.update_layout(annotations=annotations)

        return fig.to_json()

    # Dataset upload endpoints
    upload_folder = create_upload_folder()
    app.config['UPLOAD_FOLDER'] = upload_folder
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

    @app.route("/api/upload", methods=['POST'])
    def upload_dataset():
        """Handle dataset file upload."""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
            
            # Save file
            filename = secure_filename(file.filename)
            filepath = Path(app.config['UPLOAD_FOLDER']) / filename
            file.save(str(filepath))
            
            # Load and validate dataset
            df, load_error = load_dataset(str(filepath))
            if load_error:
                filepath.unlink()
                return jsonify({'error': load_error}), 400
            
            is_valid, validation_error = validate_dataset(df)
            if not is_valid:
                filepath.unlink()
                return jsonify({'error': validation_error}), 400
            
            # Get preview
            preview = get_dataset_preview(df)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'filepath': str(filepath),
                'preview': preview,
                'message': f'Successfully uploaded dataset with shape {df.shape[0]} rows × {df.shape[1]} columns'
            }), 200
        
        except Exception as e:
            return jsonify({'error': f'Upload failed: {str(e)}'}), 500

    @app.route("/api/datasets", methods=['GET'])
    def list_datasets():
        """List available uploaded datasets."""
        try:
            upload_dir = Path(app.config['UPLOAD_FOLDER'])
            datasets = []
            
            if upload_dir.exists():
                for file_path in upload_dir.iterdir():
                    if file_path.is_file() and allowed_file(file_path.name):
                        try:
                            stat = file_path.stat()
                            datasets.append({
                                'name': file_path.name,
                                'size': stat.st_size,
                                'modified': stat.st_mtime,
                                'path': str(file_path)
                            })
                        except Exception:
                            continue
            
            return jsonify({'datasets': sorted(datasets, key=lambda x: x['modified'], reverse=True)}), 200
        
        except Exception as e:
            return jsonify({'error': f'Failed to list datasets: {str(e)}'}), 500

    @app.route("/api/datasets/<filename>", methods=['GET'])
    def get_dataset_info(filename: str):
        """Get detailed information about a specific dataset."""
        try:
            filename = secure_filename(filename)
            filepath = Path(app.config['UPLOAD_FOLDER']) / filename
            
            if not filepath.exists():
                return jsonify({'error': 'Dataset not found'}), 404
            
            df, load_error = load_dataset(str(filepath))
            if load_error:
                return jsonify({'error': load_error}), 400
            
            preview = get_dataset_preview(df, rows=10)
            
            return jsonify({
                'name': filename,
                'size': filepath.stat().st_size,
                'preview': preview
            }), 200
        
        except Exception as e:
            return jsonify({'error': f'Failed to get dataset info: {str(e)}'}), 500

    @app.route("/api/datasets/<filename>", methods=['DELETE'])
    def delete_dataset(filename: str):
        """Delete an uploaded dataset."""
        try:
            filename = secure_filename(filename)
            filepath = Path(app.config['UPLOAD_FOLDER']) / filename
            
            if not filepath.exists():
                return jsonify({'error': 'Dataset not found'}), 404
            
            filepath.unlink()
            return jsonify({'success': True, 'message': f'Dataset {filename} deleted'}), 200
        
        except Exception as e:
            return jsonify({'error': f'Failed to delete dataset: {str(e)}'}), 500

    return app


def _build_html_dashboard(
    leaderboard_df: pd.DataFrame,
    resource_df: pd.DataFrame,
    best_model: dict[str, object],
    result: AutoMLResult,
) -> str:
    """Build dark-themed explainable HTML dashboard."""

    def extract_model_family(name: str | None) -> str:
        candidate = str(name or "").split("(")[0].split("-")[0].strip()
        return candidate or str(name or "")

    total_models = len(result.leaderboard)
    avg_time = float(resource_df["Training Time (s)"].mean()) if not resource_df.empty else 0.0
    peak_memory = float(resource_df["Peak RAM (MB)"].max()) if not resource_df.empty else 0.0

    best_overall_model = (
        str(best_model.get("Model"))
        if isinstance(best_model, dict) and not best_model.get("status") and best_model.get("Model")
        else None
    )

    best_overall_family = extract_model_family(best_overall_model) if best_overall_model else None

    leaderboard_families = pd.Series(dtype=str)
    leaderboard_other_family_mask = pd.Series(dtype=bool)
    if not leaderboard_df.empty and "Model" in leaderboard_df.columns:
        leaderboard_families = leaderboard_df["Model"].apply(extract_model_family)
        leaderboard_other_family_mask = pd.Series(True, index=leaderboard_df.index)
        leaderboard_other_family_mask &= leaderboard_df["Model"] != best_overall_model
        if best_overall_family:
            leaderboard_other_family_mask &= leaderboard_families != best_overall_family

    combined_score_series = (
        pd.to_numeric(leaderboard_df["Combined Score"], errors="coerce")
        if not leaderboard_df.empty and "Combined Score" in leaderboard_df.columns
        else pd.Series(dtype=float)
    )
    best_combined_score = float(combined_score_series.max()) if not combined_score_series.empty else 0.0
    runner_up_combined_score = None
    runner_up_combined_model = None
    if not combined_score_series.empty:
        competitor_scores = combined_score_series
        if not leaderboard_other_family_mask.empty:
            competitor_scores = competitor_scores[leaderboard_other_family_mask]
        competitor_scores = competitor_scores.dropna().sort_values(ascending=False)
        if not competitor_scores.empty:
            runner_up_combined_score = float(competitor_scores.iloc[0])
            runner_up_idx = competitor_scores.index[0]
            runner_up_combined_model = str(leaderboard_df.loc[runner_up_idx, "Model"])

    if best_overall_model and not combined_score_series.empty:
        mask = leaderboard_df["Model"] == best_overall_model
        if mask.any():
            best_combined_score = float(combined_score_series.loc[mask].iloc[0])

    best_model_metrics = best_model.get("All Metrics") if isinstance(best_model, dict) else {}
    best_model_accuracy = None
    if isinstance(best_model_metrics, dict):
        for key in ("accuracy", "Accuracy", "ACCURACY"):
            if key in best_model_metrics:
                try:
                    best_model_accuracy = float(best_model_metrics[key])
                except (TypeError, ValueError):
                    best_model_accuracy = None
                break

    accuracy_series = pd.Series(dtype=float)
    if not leaderboard_df.empty and "ACCURACY" in leaderboard_df.columns:
        accuracy_series = pd.to_numeric(leaderboard_df["ACCURACY"], errors="coerce")
        if accuracy_series.notna().any():
            if best_model_accuracy is None and best_overall_model:
                mask = leaderboard_df["Model"] == best_overall_model
                if mask.any():
                    try:
                        best_model_accuracy = float(accuracy_series.loc[mask].iloc[0])
                    except (TypeError, ValueError):
                        best_model_accuracy = None
            accuracy_idx = accuracy_series.idxmax()
            best_accuracy_model = str(leaderboard_df.loc[accuracy_idx, "Model"])
            best_accuracy_value = float(accuracy_series.loc[accuracy_idx])
        else:
            best_accuracy_model = None
            best_accuracy_value = None
    else:
        best_accuracy_model = None
        best_accuracy_value = None

    accuracy_runner_up_value = None
    accuracy_runner_up_model = None
    if not accuracy_series.empty and best_overall_model:
        accuracy_comparison = accuracy_series
        if not leaderboard_other_family_mask.empty:
            accuracy_comparison = accuracy_comparison[leaderboard_other_family_mask]
        else:
            accuracy_comparison = accuracy_comparison[leaderboard_df["Model"] != best_overall_model]
        accuracy_comparison = accuracy_comparison.dropna()
        if not accuracy_comparison.empty:
            accuracy_runner_up_index = accuracy_comparison.idxmax()
            accuracy_runner_up_value = float(accuracy_comparison.loc[accuracy_runner_up_index])
            accuracy_runner_up_model = str(leaderboard_df.loc[accuracy_runner_up_index, "Model"])

    accuracy_peer_baseline = None
    if best_overall_model and not accuracy_series.empty:
        accuracy_peer_series = accuracy_series
        if not leaderboard_other_family_mask.empty:
            accuracy_peer_series = accuracy_peer_series[leaderboard_other_family_mask]
        else:
            accuracy_peer_series = accuracy_peer_series[leaderboard_df["Model"] != best_overall_model]
        accuracy_peer_series = accuracy_peer_series.dropna()
        if not accuracy_peer_series.empty:
            accuracy_peer_baseline = float(accuracy_peer_series.median())

    time_series = pd.Series(dtype=float)
    ram_series = pd.Series(dtype=float)
    cpu_series = pd.Series(dtype=float)
    resource_other_family_mask = pd.Series(dtype=bool)
    if not resource_df.empty:
        time_series = pd.to_numeric(resource_df["Training Time (s)"], errors="coerce")
        ram_series = pd.to_numeric(resource_df["Peak RAM (MB)"], errors="coerce")
        if "Avg CPU (%)" in resource_df.columns:
            cpu_series = pd.to_numeric(resource_df["Avg CPU (%)"], errors="coerce")
        if "Model" in resource_df.columns:
            resource_other_family_mask = pd.Series(True, index=resource_df.index)
            resource_other_family_mask &= resource_df["Model"] != best_overall_model
            if best_overall_family:
                resource_families = resource_df["Model"].apply(extract_model_family)
                resource_other_family_mask &= resource_families != best_overall_family

    fastest_model = None
    lowest_memory_model = None
    time_runner_up_value = None
    time_runner_up_model = None
    ram_runner_up_value = None
    ram_runner_up_model = None
    cpu_runner_up_value = None
    cpu_runner_up_model = None
    cpu_best_model = None
    cpu_best_value = None
    if not time_series.empty and time_series.notna().any():
        fastest_idx = time_series.idxmin()
        fastest_model = str(resource_df.loc[fastest_idx, "Model"])
        time_comparison = time_series
        if not resource_other_family_mask.empty:
            time_comparison = time_comparison[resource_other_family_mask]
        elif best_overall_model:
            time_comparison = time_comparison[resource_df["Model"] != best_overall_model]
        time_series_ranked = time_comparison.dropna().sort_values(ascending=True)
        if not time_series_ranked.empty:
            runner_idx = time_series_ranked.index[0]
            time_runner_up_value = float(time_series_ranked.iloc[0])
            time_runner_up_model = str(resource_df.loc[runner_idx, "Model"])
    time_peer_baseline = None
    if best_overall_model and not time_series.empty:
        time_peer_series = time_series
        if not resource_other_family_mask.empty:
            time_peer_series = time_peer_series[resource_other_family_mask]
        else:
            time_peer_series = time_peer_series[resource_df["Model"] != best_overall_model]
        time_peer_series = time_peer_series.dropna()
        if not time_peer_series.empty:
            time_peer_baseline = float(time_peer_series.median())

    if not ram_series.empty and ram_series.notna().any():
        lowest_memory_idx = ram_series.idxmin()
        lowest_memory_model = str(resource_df.loc[lowest_memory_idx, "Model"])
        ram_comparison = ram_series
        if not resource_other_family_mask.empty:
            ram_comparison = ram_comparison[resource_other_family_mask]
        elif best_overall_model:
            ram_comparison = ram_comparison[resource_df["Model"] != best_overall_model]
        ram_series_ranked = ram_comparison.dropna().sort_values(ascending=True)
        if not ram_series_ranked.empty:
            ram_runner_idx = ram_series_ranked.index[0]
            ram_runner_up_value = float(ram_series_ranked.iloc[0])
            ram_runner_up_model = str(resource_df.loc[ram_runner_idx, "Model"])
    ram_peer_baseline = None
    if best_overall_model and not ram_series.empty:
        ram_peer_series = ram_series
        if not resource_other_family_mask.empty:
            ram_peer_series = ram_peer_series[resource_other_family_mask]
        else:
            ram_peer_series = ram_peer_series[resource_df["Model"] != best_overall_model]
        ram_peer_series = ram_peer_series.dropna()
        if not ram_peer_series.empty:
            ram_peer_baseline = float(ram_peer_series.median())
    cpu_peer_baseline = None
    if best_overall_model and not cpu_series.empty:
        cpu_peer_series = cpu_series
        if not resource_other_family_mask.empty:
            cpu_peer_series = cpu_peer_series[resource_other_family_mask]
        else:
            cpu_peer_series = cpu_peer_series[resource_df["Model"] != best_overall_model]
        cpu_peer_series = cpu_peer_series.dropna()
        if not cpu_peer_series.empty:
            cpu_peer_baseline = float(cpu_peer_series.median())

    if not cpu_series.empty and cpu_series.notna().any():
        cpu_series_ranked = cpu_series.dropna().sort_values(ascending=True)
        if not cpu_series_ranked.empty:
            cpu_idx = cpu_series_ranked.index[0]
            cpu_best_model = str(resource_df.loc[cpu_idx, "Model"])
            cpu_best_value = float(cpu_series_ranked.iloc[0])

        cpu_comparison = cpu_series
        if not resource_other_family_mask.empty:
            cpu_comparison = cpu_comparison[resource_other_family_mask]
        elif best_overall_model:
            cpu_comparison = cpu_comparison[resource_df["Model"] != best_overall_model]
        cpu_series_ranked = cpu_comparison.dropna().sort_values(ascending=True)
        if not cpu_series_ranked.empty:
            cpu_runner_idx = cpu_series_ranked.index[0]
            cpu_runner_up_value = float(cpu_series_ranked.iloc[0])
            cpu_runner_up_model = str(resource_df.loc[cpu_runner_idx, "Model"])

    best_model_time = (
        float(best_model.get("Training Time (s)"))
        if best_overall_model and best_model.get("Training Time (s)") is not None
        else None
    )
    best_model_ram = (
        float(best_model.get("Peak RAM (MB)"))
        if best_overall_model and best_model.get("Peak RAM (MB)") is not None
        else None
    )
    best_model_cpu = None
    if best_overall_model:
        if best_model.get("Avg CPU (%)") is not None:
            best_model_cpu = float(best_model.get("Avg CPU (%)"))
        elif cpu_best_model == best_overall_model:
            best_model_cpu = cpu_best_value
    if best_model_cpu is None and best_overall_model and not resource_df.empty and "Avg CPU (%)" in resource_df.columns:
        best_row_matches = resource_df[resource_df["Model"] == best_overall_model]
        if not best_row_matches.empty:
            try:
                best_model_cpu = float(pd.to_numeric(best_row_matches["Avg CPU (%)"], errors="coerce").iloc[0])
            except (TypeError, ValueError):
                best_model_cpu = None

    accuracy_margin = None
    if best_model_accuracy is not None and best_accuracy_value is not None:
        accuracy_margin = abs(best_accuracy_value - best_model_accuracy)

    overall_median_time = float(time_series.median()) if not time_series.empty else None
    lowest_ram_value = float(ram_series.min()) if not ram_series.empty else None

    combined_score_display = f"{best_combined_score:.3f}" if best_combined_score else "0.000"
    accuracy_display = f"{best_model_accuracy:.3f}" if best_model_accuracy is not None else "—"
    avg_time_display = f"{avg_time:.1f}"
    peak_memory_display = f"{peak_memory:.0f}"
    total_models_display = str(total_models)

    combined_score_tooltip = (
        "Balances Accuracy, Training Time, Peak RAM, and CPU usage so production realism matters more than raw accuracy."
    )

    badges: list[tuple[str, str | None]] = [
        ("Best Accuracy", best_accuracy_model),
        ("Fastest Training", fastest_model),
        ("Lowest Memory", lowest_memory_model),
        ("Best Overall", best_overall_model),
    ]
    winner_badges_html_parts: list[str] = []
    category_models = {model for _, model in badges if model}
    for label, model_name in badges:
        badge_classes = ["winner-badge"]
        if label == "Best Overall":
            badge_classes.append("winner-badge--overall")
        display_value = "—" if not model_name else html.escape(str(model_name))
        winner_badges_html_parts.append(
            f"<div class='{' '.join(badge_classes)}'><span class='badge-label'>{html.escape(label)}</span><span class='badge-value'>{display_value}</span></div>"
        )
    winner_badges_html = "".join(winner_badges_html_parts)

    def metric_value(value: float | None, precision: int = 3, suffix: str = "") -> str:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "—"
        try:
            formatted = f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            return "—"
        return f"{formatted}{suffix}"

    def percent_gap(baseline: float | None, comparator: float | None, higher_is_better: bool = True) -> float | None:
        if baseline is None or comparator is None:
            return None
        try:
            if math.isclose(comparator, 0.0, abs_tol=1e-12):
                return None
            if higher_is_better:
                return ((baseline - comparator) / abs(comparator)) * 100.0
            return ((comparator - baseline) / abs(comparator)) * 100.0
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def format_percent(value: float | None) -> str:
        if value is None:
            return ""
        return f"{value:.1f}%"

    accuracy_line = "✅ Accuracy insight unavailable."
    if best_model_accuracy is not None:
        if accuracy_runner_up_value is not None:
            pct = percent_gap(best_model_accuracy, accuracy_runner_up_value, higher_is_better=True)
            if pct is not None and pct > 0:
                rival_label = html.escape(accuracy_runner_up_model or "runner-up")
                accuracy_line = (
                    f"✅ Accuracy beats {rival_label} by {format_percent(pct)} "
                    f"({metric_value(best_model_accuracy, precision=3)} vs {metric_value(accuracy_runner_up_value, precision=3)})."
                )
        elif accuracy_peer_baseline is not None:
            pct = percent_gap(best_model_accuracy, accuracy_peer_baseline, higher_is_better=True)
            if pct is not None and pct > 0:
                accuracy_line = (
                    f"✅ Accuracy beats peer median by {format_percent(pct)} "
                    f"({metric_value(best_model_accuracy, precision=3)} vs {metric_value(accuracy_peer_baseline, precision=3)})."
                )
        if accuracy_line == "✅ Accuracy insight unavailable.":
            accuracy_line = f"✅ Accuracy at {metric_value(best_model_accuracy, precision=3)}."

    training_line = "⚡ Training time insight unavailable."
    if best_model_time is not None:
        comparator_value = None
        comparator_label = None
        if time_runner_up_value is not None and best_model_time < time_runner_up_value:
            comparator_value = time_runner_up_value
            comparator_label = time_runner_up_model or "runner-up"
        elif time_peer_baseline is not None and best_model_time < time_peer_baseline:
            comparator_value = time_peer_baseline
            comparator_label = "peer median"
        if comparator_value is not None:
            pct_time = percent_gap(best_model_time, comparator_value, higher_is_better=False)
            if pct_time is not None and pct_time > 0:
                training_line = (
                    f"⚡ Training faster than {html.escape(comparator_label)} by {format_percent(pct_time)} "
                    f"({metric_value(best_model_time, precision=2, suffix='s')} vs {metric_value(comparator_value, precision=2, suffix='s')})."
                )
        if training_line == "⚡ Training time insight unavailable.":
            training_line = f"⚡ Training completes in {metric_value(best_model_time, precision=2, suffix='s')} for rapid redeploys."

    memory_line = "💾 Resource usage insight unavailable."
    memory_bits: list[str] = []
    if best_model_ram is not None:
        ram_candidates: list[tuple[str, float]] = []
        if ram_runner_up_value is not None and best_model_ram < ram_runner_up_value:
            ram_candidates.append((ram_runner_up_model or "runner-up", ram_runner_up_value))
        if ram_peer_baseline is not None and best_model_ram < ram_peer_baseline:
            ram_candidates.append(("peer median", ram_peer_baseline))
        for label, value in ram_candidates:
            ram_pct = percent_gap(best_model_ram, value, higher_is_better=False)
            if ram_pct is not None and ram_pct > 0:
                memory_bits.append(
                    f"RAM {metric_value(best_model_ram, precision=0, suffix=' MB')} ({format_percent(ram_pct)} lower than {html.escape(label)})"
                )
                break
    if best_model_cpu is not None:
        cpu_candidates: list[tuple[str, float]] = []
        if cpu_runner_up_value is not None and best_model_cpu < cpu_runner_up_value:
            cpu_candidates.append((cpu_runner_up_model or "runner-up", cpu_runner_up_value))
        if cpu_peer_baseline is not None and best_model_cpu < cpu_peer_baseline:
            cpu_candidates.append(("peer median", cpu_peer_baseline))
        for label, value in cpu_candidates:
            cpu_pct = percent_gap(best_model_cpu, value, higher_is_better=False)
            if cpu_pct is not None and cpu_pct > 0:
                memory_bits.append(
                    f"CPU {metric_value(best_model_cpu, precision=1, suffix='%')} ({format_percent(cpu_pct)} lower than {html.escape(label)})"
                )
                break
    if memory_bits:
        memory_line = "💾 " + " • ".join(memory_bits)
    else:
        memory_line = "💾 Resource usage remains deployment-friendly."

    realism_line = "🧠 Combined score favours production realism over raw accuracy."
    if runner_up_combined_score is not None and math.isfinite(best_combined_score):
        combined_pct = percent_gap(best_combined_score, runner_up_combined_score, higher_is_better=True)
        if combined_pct is not None and combined_pct > 0:
            rival = html.escape(runner_up_combined_model or "runner-up")
            realism_line = (
                f"🧠 Combined score is {format_percent(combined_pct)} higher than {rival} "
                f"({metric_value(best_combined_score, precision=3)} vs {metric_value(runner_up_combined_score, precision=3)})."
            )

    if best_overall_model:
        why_card_html = (
            "<div class='why-card'>"
            "<div class='why-card-title'>Why this model won</div>"
            "<ul class='why-list'>"
            f"<li>{accuracy_line}</li>"
            f"<li>{training_line}</li>"
            f"<li>{memory_line}</li>"
            f"<li>{realism_line}</li>"
            "</ul>"
            "</div>"
        )
    else:
        why_card_html = (
            "<div class='why-card is-empty'>"
            "<div class='why-card-title'>Why this model won</div>"
            "<p class='why-placeholder'>No eligible model satisfied the constraints yet.</p>"
            "</div>"
        )

    def slugify(value: str) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug.strip("-") or "group"

    column_icons = {
        "Rank": "🏅",
        "Model": "🤖",
        "Combined Score": "⚖️",
        "Training Time (s)": "⏱️",
        "Peak RAM (MB)": "💾",
        "Avg CPU (%)": "🧠",
        "Violations": "⚠️",
    }

    entry_by_rank = {entry.rank: entry for entry in (result.leaderboard or [])}

    def summarize_params(params: dict[str, object] | None) -> str:
        if not params:
            return "Default settings"
        items = [f"{key}={params[key]}" for key in sorted(params)]
        summary = ", ".join(items[:3])
        if len(items) > 3:
            summary += ", …"
        return summary

    def describe_param_diff(base: dict[str, object] | None, other: dict[str, object] | None) -> str:
        base = base or {}
        other = other or {}
        diffs = []
        for key in sorted(set(base) | set(other)):
            base_val = base.get(key)
            other_val = other.get(key)
            if other_val != base_val:
                diffs.append(f"{key}={other_val}")
        if not diffs:
            return "Matches winning config"
        summary = ", ".join(diffs[:3])
        if len(diffs) > 3:
            summary += ", …"
        return summary

    if leaderboard_df.empty:
        header_html = "<th>Model</th>"
        body_html = "<tr><td>No evaluation results available.</td></tr>"
    else:
        header_cells: list[str] = []
        for col in leaderboard_df.columns:
            icon = column_icons.get(col, "")
            label = html.escape(str(col))
            if col == "Combined Score":
                header_cells.append(
                    "<th class='has-tooltip'>"
                    f"{icon} {label} <span class='info-icon info-icon--inline' data-tooltip='{combined_score_tooltip}'>ℹ️</span>"
                    "</th>"
                )
            else:
                header_cells.append(f"<th>{icon} {label}</th>")
        header_html = "".join(header_cells)

        metric_columns = {
            col
            for col in leaderboard_df.columns
            if col
            not in {
                "Rank",
                "Model",
                "Combined Score",
                "Training Time (s)",
                "Peak RAM (MB)",
                "Avg CPU (%)",
                "Violations",
            }
        }

        column_count = len(leaderboard_df.columns)

        table_rows: list[str] = []
        grouped: dict[str, list[pd.Series]] = {}
        for _, row in leaderboard_df.sort_values(by="Rank").iterrows():
            model_name = str(row["Model"])
            family = extract_model_family(model_name)
            grouped.setdefault(family, []).append(row)

        def format_cell(col_name: str, value, model_name: str, subtext: str | None = None) -> str:
            if pd.isna(value):
                display = "—"
            else:
                display = value
            if col_name == "Rank":
                try:
                    rank_display = int(value)
                except (TypeError, ValueError):
                    rank_display = value
                return f"<td><span class='rank-badge'>{rank_display}</span></td>"
            if col_name == "Model":
                model_title = html.escape(str(display))
                if subtext:
                    safe_subtext = html.escape(subtext)
                    return (
                        "<td class='model-name-cell'>"
                        f"<div class='model-name-primary'>{model_title}</div>"
                        f"<div class='model-name-subtext'>{safe_subtext}</div>"
                        "</td>"
                    )
                return f"<td class='model-name-cell'><div class='model-name-primary'>{model_title}</div></td>"
            if col_name == "Combined Score":
                score_value = metric_value(value, precision=3)
                chip_classes = ["score-chip"]
                if model_name == best_overall_model:
                    chip_classes.append("score-chip--winner")
                elif model_name in category_models:
                    chip_classes.append("score-chip--highlight")
                else:
                    chip_classes.append("score-chip--muted")
                return (
                    "<td>"
                    f"<span class='{' '.join(chip_classes)}' data-score='{score_value}'>{score_value}</span>"
                    "</td>"
                )
            if col_name == "Training Time (s)":
                return f"<td>{metric_value(value, precision=2, suffix=' s')}</td>"
            if col_name == "Peak RAM (MB)":
                return f"<td>{metric_value(value, precision=0, suffix=' MB')}</td>"
            if col_name == "Avg CPU (%)":
                return f"<td>{metric_value(value, precision=1, suffix='%')}</td>"
            if col_name == "Violations":
                try:
                    violations_value = int(value)
                except (TypeError, ValueError):
                    violations_value = value
                badge = "None" if violations_value == 0 else str(violations_value)
                badge_class = "violation-chip" if violations_value else "violation-chip violation-chip--none"
                return f"<td><span class='{badge_class}'>{badge}</span></td>"
            if col_name in metric_columns:
                metric_label = html.escape(col_name.replace("_", " "))
                metric_val = metric_value(value, precision=3)
                return f"<td><span class='metric-pill'>{metric_label}: {metric_val}</span></td>"
            return f"<td>{html.escape(str(display))}</td>"

        def saferank(row_series: pd.Series) -> float:
            try:
                return float(row_series["Rank"])
            except (TypeError, ValueError):
                return float("inf")

        sorted_families = sorted(
            grouped.items(),
            key=lambda item: saferank(min(item[1], key=lambda r: saferank(r))),
        )

        for family, rows in sorted_families:
            group_id = slugify(family)
            toggle_button = (
                f"<button class='group-toggle' data-group='{group_id}' aria-expanded='false'>View variations</button>"
            )
            table_rows.append(
                "<tr class='group-heading' data-group='"
                + group_id
                + "'><td colspan='"
                + str(column_count)
                + "'>"
                + f"<div class='group-heading-content'><span class='group-title'>{html.escape(family)}</span>{toggle_button}</div>"
                + "</td></tr>"
            )

            best_row = min(rows, key=lambda r: saferank(r))
            best_rank = best_row.get("Rank")
            best_entry = entry_by_rank.get(int(best_rank)) if not pd.isna(best_rank) else None
            best_params = best_entry.result.params if best_entry else {}
            best_subtext = f"Key params: {summarize_params(best_params)}"

            def build_row(row_series: pd.Series, primary: bool) -> None:
                model_name = str(row_series["Model"])
                rank_value = row_series.get("Rank")
                entry = None
                if rank_value is not None and not pd.isna(rank_value):
                    try:
                        entry = entry_by_rank.get(int(rank_value))
                    except (TypeError, ValueError):
                        entry = None
                params = entry.result.params if entry else {}
                if primary:
                    subtext = best_subtext
                else:
                    subtext = f"Diff vs best: {describe_param_diff(best_params, params)}"

                row_classes = ["group-row", "group-row-primary" if primary else "group-row-variant"]
                if not primary:
                    row_classes.append("collapsed")
                if model_name == best_overall_model:
                    row_classes.append("best-overall")
                elif model_name in category_models:
                    row_classes.append("category-winner")
                else:
                    row_classes.append("deemphasized")

                cells = []
                for col in leaderboard_df.columns:
                    if col == "Model":
                        cells.append(format_cell(col, row_series[col], model_name, subtext=subtext))
                    else:
                        cells.append(format_cell(col, row_series[col], model_name))

                table_rows.append(
                    f"<tr class='{' '.join(row_classes)}' data-group='{group_id}'>"
                    + "".join(cells)
                    + "</tr>"
                )

            build_row(best_row, primary=True)
            for variant_row in rows:
                if variant_row is best_row:
                    continue
                build_row(variant_row, primary=False)

        body_html = "".join(table_rows)

    ranking_toggle_html = (
        "<div class='ranking-toggle' role='group' aria-label='Ranking mode'>"
        "<span class='toggle-label'>Ranking mode</span>"
        "<div class='toggle-buttons'>"
        "<button type='button' class='toggle-btn' data-mode='accuracy'>Accuracy-first</button>"
        "<button type='button' class='toggle-btn active' data-mode='balanced'>Balanced</button>"
        "<button type='button' class='toggle-btn' data-mode='cost'>Cost-first</button>"
        "</div>"
        "</div>"
    )

    mode_copy_html = (
        "<p class='mode-copy' data-mode-copy>Balanced view: visibility weighted by Combined Score.</p>"
    )

    philosophy_statement = "The best model is the simplest model that meets performance requirements."
    generated_at = pd.Timestamp.now().strftime("%B %d, %Y at %H:%M:%S")

    parts: list[str] = []
    parts.append(
        """
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <title>AutoML Decision Dashboard</title>
        <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
        <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css\">
        <style>
            :root {
                --bg: #050b18;
                --panel: #0f172a;
                --panel-alt: #0b1120;
                --panel-soft: #131c2e;
                --accent: #5fe3c0;
                --accent-strong: #38bdf8;
                --accent-muted: rgba(95, 227, 192, 0.25);
                --danger: #f87171;
                --warning: #fbbf24;
                --success: #34d399;
                --text: #e6edf7;
                --text-muted: #94a3b8;
                --border: rgba(148, 163, 184, 0.25);
                --shadow: rgba(7, 13, 28, 0.6);
                --deemphasized: rgba(148, 163, 184, 0.35);
            }
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: radial-gradient(circle at top, rgba(56, 189, 248, 0.18), transparent 45%), var(--bg);
                min-height: 100vh;
                padding: 32px 18px 48px;
                color: var(--text);
                transition: background 0.4s ease;
            }
            body[data-mode=\"accuracy\"] .section-quality .section-title {
                color: var(--accent-strong);
            }
            body[data-mode=\"cost\"] .section-resources .section-title {
                color: var(--accent);
            }
            .container {
                max-width: 1600px;
                margin: 0 auto;
            }
            .navbar {
                background: var(--panel);
                padding: 20px 28px;
                border-radius: 18px;
                margin-bottom: 28px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border: 1px solid var(--border);
                box-shadow: 0 18px 45px var(--shadow);
            }
            .navbar-brand {
                font-size: 1.6rem;
                font-weight: 700;
                display: flex;
                align-items: center;
                gap: 10px;
                color: var(--text);
            }
            .navbar-context {
                font-size: 0.95rem;
                color: var(--text-muted);
            }
            .hero-section {
                background: var(--panel);
                border-radius: 22px;
                padding: 40px;
                margin-bottom: 24px;
                border: 1px solid var(--border);
                box-shadow: 0 24px 60px rgba(7, 13, 28, 0.45);
                position: relative;
                overflow: hidden;
            }
            .hero-section::before {
                content: "";
                position: absolute;
                top: -40%;
                right: -20%;
                width: 420px;
                height: 420px;
                background: radial-gradient(circle, rgba(95, 227, 192, 0.25), transparent);
            }
            .hero-content {
                position: relative;
                z-index: 1;
            }
            .hero-title {
                font-size: 2.3rem;
                font-weight: 800;
                margin-bottom: 12px;
            }
            .hero-subtitle {
                font-size: 1rem;
                color: var(--text-muted);
                margin-bottom: 24px;
            }
            .best-model-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 18px;
                margin-bottom: 24px;
            }
            .model-metric {
                background: var(--panel-alt);
                border-radius: 16px;
                padding: 20px;
                border: 1px solid rgba(56, 189, 248, 0.15);
                display: flex;
                flex-direction: column;
                justify-content: center;
                gap: 8px;
                min-height: 120px;
            }
            .model-metric-label {
                font-size: 0.85rem;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                color: var(--text-muted);
                margin-bottom: 10px;
            }
            .model-metric-value {
                font-size: 1.65rem;
                font-weight: 700;
                color: var(--text);
                word-break: break-word;
                line-height: 1.25;
            }
            .model-metric-value.highlight {
                color: var(--accent);
                font-size: 1.9rem;
            }
            .why-card {
                background: var(--panel-alt);
                border-radius: 18px;
                padding: 22px 24px;
                border: 1px solid rgba(95, 227, 192, 0.25);
            }
            .why-card-title {
                font-size: 1.1rem;
                font-weight: 700;
                margin-bottom: 12px;
            }
            .why-list {
                list-style: none;
                display: grid;
                gap: 10px;
            }
            .why-card.is-empty {
                border-style: dashed;
                color: var(--text-muted);
            }
            .winner-badges {
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                margin: 18px 0 24px;
            }
            .winner-badge {
                background: var(--panel-alt);
                border-radius: 40px;
                padding: 10px 16px;
                border: 1px solid var(--border);
                display: flex;
                gap: 8px;
                align-items: baseline;
            }
            .winner-badge--overall {
                border-color: rgba(95, 227, 192, 0.6);
                background: rgba(95, 227, 192, 0.12);
            }
            .badge-label {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--text-muted);
            }
            .badge-value {
                font-size: 0.95rem;
                font-weight: 600;
            }
            .section {
                margin-top: 32px;
            }
            .section-title {
                font-size: 1.4rem;
                font-weight: 700;
                margin-bottom: 6px;
            }
            .section-subtitle {
                font-size: 0.95rem;
                color: var(--text-muted);
                margin-bottom: 16px;
            }
            .card {
                background: var(--panel);
                border-radius: 20px;
                border: 1px solid var(--border);
                box-shadow: 0 18px 40px rgba(7, 13, 28, 0.4);
                padding: 24px 26px 12px;
                margin-bottom: 24px;
            }
            .card-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 14px;
            }
            .card-title {
                font-size: 1.15rem;
                font-weight: 700;
            }
            .card-icon {
                font-size: 1.4rem;
                color: var(--accent);
            }
            .chart-container {
                height: 380px;
            }
            .chart-empty {
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: var(--text-muted);
                font-size: 0.95rem;
            }
            .ranking-toggle {
                display: flex;
                align-items: center;
                gap: 12px;
                margin-top: 12px;
                flex-wrap: wrap;
            }
            .toggle-label {
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: var(--text-muted);
            }
            .toggle-buttons {
                display: inline-flex;
                background: var(--panel-alt);
                border-radius: 999px;
                border: 1px solid var(--border);
                padding: 4px;
            }
            .toggle-btn {
                border: none;
                background: transparent;
                color: var(--text-muted);
                font-size: 0.85rem;
                font-weight: 600;
                padding: 6px 16px;
                border-radius: 999px;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            .toggle-btn.active {
                background: var(--accent-muted);
                color: var(--accent);
            }
            .mode-copy {
                margin-top: 10px;
                font-size: 0.9rem;
                color: var(--text-muted);
            }
            .table-wrapper {
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid var(--border);
            }
            table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.92rem;
            }
            .model-name-cell {
                min-width: 220px;
            }
            .model-name-primary {
                font-weight: 600;
            }
            .model-name-subtext {
                margin-top: 4px;
                font-size: 0.78rem;
                color: var(--text-muted);
                white-space: normal;
                line-height: 1.3;
            }
            thead {
                background: rgba(9, 15, 28, 0.92);
            }
            th {
                padding: 14px 16px;
                text-align: left;
                color: var(--text-muted);
                font-size: 0.75rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                border-bottom: 1px solid var(--border);
            }
            td {
                padding: 14px 16px;
                border-bottom: 1px solid var(--border);
            }
            tbody tr.deemphasized {
                opacity: 0.55;
            }
            tbody tr.best-overall {
                box-shadow: inset 0 0 0 1px rgba(95, 227, 192, 0.4);
            }
            tbody tr.category-winner:not(.best-overall) {
                box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.35);
            }
            .group-heading {
                background: rgba(11, 17, 32, 0.85);
            }
            .group-heading-content {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .group-title {
                font-weight: 600;
                letter-spacing: 0.04em;
            }
            .group-toggle {
                background: transparent;
                border: 1px solid var(--border);
                border-radius: 999px;
                color: var(--text-muted);
                padding: 4px 12px;
                font-size: 0.8rem;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            .group-toggle:hover {
                color: var(--accent);
                border-color: var(--accent);
            }
            .group-row-variant.collapsed {
                display: none;
            }
            .rank-badge {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background: rgba(95, 227, 192, 0.15);
                color: var(--accent);
                font-weight: 700;
            }
            .metric-pill {
                display: inline-block;
                padding: 6px 12px;
                border-radius: 999px;
                font-size: 0.82rem;
                background: rgba(56, 189, 248, 0.12);
                color: var(--accent-strong);
            }
            .score-chip {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 72px;
                padding: 6px 12px;
                border-radius: 10px;
                font-weight: 600;
                transition: all 0.2s ease;
            }
            .score-chip--winner {
                background: rgba(95, 227, 192, 0.22);
                color: var(--accent);
            }
            .score-chip--highlight {
                background: rgba(56, 189, 248, 0.18);
                color: var(--accent-strong);
            }
            .score-chip--muted {
                background: rgba(148, 163, 184, 0.12);
                color: var(--text-muted);
            }
            .violation-chip {
                display: inline-flex;
                min-width: 52px;
                justify-content: center;
                background: rgba(248, 113, 113, 0.18);
                color: var(--danger);
                padding: 4px 8px;
                border-radius: 999px;
                font-weight: 600;
            }
            .violation-chip--none {
                background: rgba(52, 211, 153, 0.15);
                color: var(--success);
            }
            .info-icon {
                font-size: 0.85rem;
                cursor: help;
                color: var(--accent-strong);
                position: relative;
            }
            .info-icon::after {
                content: attr(data-tooltip);
                position: absolute;
                left: 50%;
                bottom: 140%;
                transform: translateX(-50%);
                background: rgba(11, 17, 32, 0.95);
                color: var(--text);
                padding: 10px 14px;
                border-radius: 10px;
                font-size: 0.8rem;
                line-height: 1.3;
                white-space: pre-line;
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.2s ease;
                width: 260px;
                border: 1px solid var(--border);
            }
            .info-icon:hover::after {
                opacity: 1;
            }
            .info-icon--inline {
                font-size: 0.75rem;
            }
            .winner-callout {
                margin-top: 16px;
                padding: 16px 18px;
                border-radius: 12px;
                background: rgba(95, 227, 192, 0.18);
                color: var(--accent);
                font-weight: 600;
            }
            .philosophy {
                margin-top: 18px;
                padding: 16px 20px;
                border-left: 3px solid rgba(56, 189, 248, 0.45);
                color: var(--text-muted);
                font-style: italic;
            }
            .footer {
                margin-top: 40px;
                text-align: center;
                color: var(--text-muted);
                font-size: 0.85rem;
            }
            .navbar-upload-btn {
                background: rgba(95, 227, 192, 0.15);
                border: 1px solid rgba(95, 227, 192, 0.4);
                color: var(--accent);
                padding: 8px 16px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 0.9rem;
                transition: all 0.2s ease;
                font-weight: 600;
            }
            .navbar-upload-btn:hover {
                background: rgba(95, 227, 192, 0.25);
                border-color: var(--accent);
            }
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.8);
            }
            .modal-content {
                background-color: var(--panel);
                margin: 5% auto;
                padding: 30px;
                border-radius: 12px;
                border: 1px solid var(--border);
                width: 90%;
                max-width: 500px;
            }
            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            .modal-title {
                font-size: 1.3rem;
                font-weight: 700;
            }
            .close {
                color: var(--text-muted);
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
                transition: color 0.2s;
            }
            .close:hover {
                color: var(--text);
            }
            .upload-area {
                border: 2px dashed var(--border);
                border-radius: 8px;
                padding: 40px 20px;
                text-align: center;
                cursor: pointer;
                transition: all 0.2s;
            }
            .upload-area:hover {
                border-color: var(--accent);
                background: rgba(95, 227, 192, 0.05);
            }
            .upload-area.dragover {
                border-color: var(--accent);
                background: rgba(95, 227, 192, 0.1);
            }
            #fileInput {
                display: none;
            }
            .upload-message {
                margin-top: 15px;
                padding: 12px;
                border-radius: 6px;
                font-size: 0.9rem;
            }
            .upload-message.success {
                background: rgba(52, 211, 153, 0.15);
                color: #34d399;
                border: 1px solid rgba(52, 211, 153, 0.3);
            }
            .upload-message.error {
                background: rgba(248, 113, 113, 0.15);
                color: #f87171;
                border: 1px solid rgba(248, 113, 113, 0.3);
            }
            @media (max-width: 960px) {
                .hero-section {
                    padding: 28px;
                }
            }
        </style>
    </head>
    <body data-mode=\"balanced\">
        <div class=\"container\">
            <div class=\"navbar\">
                <div class=\"navbar-brand\"><i class=\"fas fa-robot\"></i> AutoML Decision Hub</div>
                <div style=\"display: flex; gap: 20px; align-items: center;\">
                    <div class=\"navbar-context\">Resource-aware AutoML · Production Readiness Lens</div>
                    <button class=\"navbar-upload-btn\" onclick=\"openUploadModal()\" style=\"cursor: pointer;\">Upload Dataset</button>
                </div>
            </div>
            
            <!-- Upload Modal -->
            <div id=\"uploadModal\" class=\"modal\">
                <div class=\"modal-content\">
                    <div class=\"modal-header\">
                        <div class=\"modal-title\">Upload Dataset</div>
                        <span class=\"close\" onclick=\"closeUploadModal()\" style=\"cursor: pointer;\">&times;</span>
                    </div>
                    <div class=\"upload-area\" id=\"uploadArea\" onclick=\"document.getElementById('fileInput').click()\" style=\"cursor: pointer;\">
                        <div style=\"font-size: 3rem; margin-bottom: 10px;\">📁</div>
                        <div style=\"font-size: 1rem; margin-bottom: 5px; font-weight: 600;\">Drag and drop your file here</div>
                        <div style=\"color: var(--text-muted); font-size: 0.85rem;\">or click to browse</div>
                        <div style=\"color: var(--text-muted); font-size: 0.8rem; margin-top: 10px;\">Supported: CSV, Excel, JSON, Parquet</div>
                    </div>
                    <input type=\"file\" id=\"fileInput\" onchange=\"handleFileSelect(event)\" accept=\".csv,.xlsx,.xls,.json,.parquet\">
                    <div id=\"uploadMessage\"></div>
                </div>
            </div>
            
            <script>
                // Ensure modal functions are in global scope
                window.openUploadModal = function() {
                    var modal = document.getElementById('uploadModal');
                    if (modal) {
                        modal.style.display = 'block';
                        console.log('Modal opened');
                    }
                }
                
                window.closeUploadModal = function() {
                    var modal = document.getElementById('uploadModal');
                    if (modal) {
                        modal.style.display = 'none';
                        document.getElementById('uploadMessage').innerHTML = '';
                        document.getElementById('fileInput').value = '';
                    }
                }
                
                window.onclick = function(event) {
                    var modal = document.getElementById('uploadModal');
                    if (event.target == modal) {
                        modal.style.display = 'none';
                    }
                }
                
                var uploadArea = document.getElementById('uploadArea');
                if (uploadArea) {
                    uploadArea.addEventListener('dragover', function(e) {
                        e.preventDefault();
                        uploadArea.classList.add('dragover');
                    });
                    
                    uploadArea.addEventListener('dragleave', function() {
                        uploadArea.classList.remove('dragover');
                    });
                    
                    uploadArea.addEventListener('drop', function(e) {
                        e.preventDefault();
                        uploadArea.classList.remove('dragover');
                        var files = e.dataTransfer.files;
                        if (files.length > 0) {
                            handleFileUpload(files[0]);
                        }
                    });
                }
                
                window.handleFileSelect = function(event) {
                    var files = event.target.files;
                    if (files.length > 0) {
                        handleFileUpload(files[0]);
                    }
                }
                
                window.handleFileUpload = function(file) {
                    var formData = new FormData();
                    formData.append('file', file);
                    
                    var messageDiv = document.getElementById('uploadMessage');
                    messageDiv.innerHTML = '<div class=\"upload-message\">⏳ Uploading and validating...</div>';
                    
                    fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    })
                    .then(function(response) { return response.json(); })
                    .then(function(data) {
                        if (data.success) {
                            messageDiv.innerHTML = '<div class=\"upload-message success\">✅ ' + data.message + '</div>';
                            setTimeout(closeUploadModal, 2500);
                        } else {
                            messageDiv.innerHTML = '<div class=\"upload-message error\">❌ ' + data.error + '</div>';
                        }
                    })
                    .catch(function(error) {
                        messageDiv.innerHTML = '<div class=\"upload-message error\">❌ Upload failed: ' + error + '</div>';
                    });
                }
            </script>
"""
    )

    parts.append("            <div class=\"hero-section\">\n                <div class=\"hero-content\">\n")
    parts.append("                    <div class=\"hero-title\">Best Overall Model (Efficiency-Weighted)</div>\n")
    parts.append(
        "                    <div class=\"hero-subtitle\">Decision logic favours deployable performance over leaderboard vanity metrics.</div>\n"
    )

    if best_overall_model:
        combined_class = "model-metric-value highlight"
        parts.append("                    <div class=\"best-model-grid\">\n")
        parts.append(
            f'                        <div class="model-metric"><div class="model-metric-label">Model</div><div class="model-metric-value">{html.escape(best_overall_model)}</div></div>\n'
        )
        parts.append(
            f'                        <div class="model-metric"><div class="model-metric-label">Combined Score</div><div class="{combined_class}">{combined_score_display}</div></div>\n'
        )
        parts.append(
            f'                        <div class="model-metric"><div class="model-metric-label">Accuracy Only (Not Final Ranking)</div><div class="model-metric-value">{accuracy_display}</div></div>\n'
        )
        parts.append(
            f'                        <div class="model-metric"><div class="model-metric-label">Training Time</div><div class="model-metric-value">{metric_value(best_model_time, precision=2, suffix=" s")}</div></div>\n'
        )
        parts.append(
            f'                        <div class="model-metric"><div class="model-metric-label">Resource Cost</div><div class="model-metric-value">{metric_value(best_model_ram, precision=0, suffix=" MB")} · {metric_value(best_model_cpu, precision=1, suffix="%") } CPU</div></div>\n'
        )
        parts.append("                    </div>\n")
    else:
        status_message = html.escape(str(best_model.get("status", "No AutoML results to display")))
        parts.append(f"                    <p>{status_message}</p>\n")

    parts.append(f"{why_card_html}\n")
    parts.append("                </div>\n            </div>\n")

    parts.append(f'            <div class="winner-badges">{winner_badges_html}</div>\n')

    parts.append("            <div class=\"section section-quality\">\n")
    parts.append("                <div class=\"section-title\">Section 1 · Model Quality</div>\n")
    parts.append("                <div class=\"section-subtitle\">Accuracy, F1, and ROC-AUC comparisons by model.</div>\n")
    parts.append("                <div class=\"card chart-card\">\n")
    parts.append("                    <div class=\"card-header\">\n")
    parts.append("                        <div class=\"card-icon\"><i class=\"fas fa-bullseye\"></i></div>\n")
    parts.append("                        <div class=\"card-title\">Accuracy Only (Not Final Ranking)</div>\n")
    parts.append("                    </div>\n")
    parts.append("                    <div class=\"chart-container\" id=\"chart-quality\"></div>\n")
    parts.append("                </div>\n            </div>\n")

    parts.append("            <div class=\"section section-resources\">\n")
    parts.append("                <div class=\"section-title\">Section 2 · Resource Cost</div>\n")
    parts.append("                <div class=\"section-subtitle\">Training time, peak RAM, and CPU% normalised for comparison.</div>\n")
    parts.append("                <div class=\"card chart-card\">\n")
    parts.append("                    <div class=\"card-header\">\n")
    parts.append("                        <div class=\"card-icon\"><i class=\"fas fa-microchip\"></i></div>\n")
    parts.append("                        <div class=\"card-title\">Resource Profile by Model</div>\n")
    parts.append("                    </div>\n")
    parts.append("                    <div class=\"chart-container\" id=\"chart-cost\"></div>\n")
    parts.append("                </div>\n            </div>\n")

    parts.append("            <div class=\"card\">\n")
    parts.append("                <div class=\"card-header\">\n")
    parts.append("                    <div class=\"card-icon\"><i class=\"fas fa-table\"></i></div>\n")
    parts.append("                    <div class=\"card-title\">Leaderboard Detail</div>\n")
    parts.append("                </div>\n")
    parts.append("                <div class=\"table-wrapper\">\n")
    parts.append("                    <table>\n")
    parts.append("                        <thead><tr>" + header_html + "</tr></thead>\n")
    parts.append("                        <tbody>" + body_html + "</tbody>\n")
    parts.append("                    </table>\n")
    parts.append("                </div>\n")
    parts.append("            </div>\n")

    parts.append(
        f'            <div class="footer">Generated {html.escape(generated_at)} · Resource-aware AutoML Dashboard</div>\n'
    )

    parts.append(
        """
        </div>

        <script>
            async function loadCharts() {
                const endpoints = [
                    { id: 'chart-quality', url: '/api/charts/metrics' },
                    { id: 'chart-cost', url: '/api/charts/resources' }
                ];
                const commonLayout = {
                    font: { family: 'Inter, sans-serif', color: '#e6edf7', size: 12 },
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    margin: { t: 30, r: 20, b: 40, l: 60 }
                };

                for (const endpoint of endpoints) {
                    try {
                        const response = await fetch(endpoint.url);
                        const payload = await response.json();
                        const container = document.getElementById(endpoint.id);
                        if (!container) continue;
                        if (payload.error) {
                            container.innerHTML = `<div class=\"chart-empty\">${payload.error}</div>`;
                            continue;
                        }
                        Plotly.newPlot(endpoint.id, payload.data, Object.assign({}, payload.layout, commonLayout), { responsive: true, displayModeBar: false });
                    } catch (error) {
                        console.error('Error loading chart', endpoint.id, error);
                    }
                }
            }

            function initRankingToggle() {
                const buttons = document.querySelectorAll('.toggle-btn');
                const modeCopy = document.querySelector('[data-mode-copy]');
                const copyByMode = {
                    accuracy: 'Accuracy-first view: emphasises pure predictive quality.',
                    balanced: 'Balanced view: visibility weighted by Combined Score.',
                    cost: 'Cost-first view: highlights the most efficient deployments.'
                };

                buttons.forEach((button) => {
                    button.addEventListener('click', () => {
                        buttons.forEach((btn) => btn.classList.remove('active'));
                        button.classList.add('active');
                        const mode = button.dataset.mode || 'balanced';
                        document.body.setAttribute('data-mode', mode);
                        if (modeCopy) {
                            modeCopy.textContent = copyByMode[mode];
                        }
                    });
                });
            }

            function initGroupToggles() {
                document.querySelectorAll('.group-toggle').forEach((toggle) => {
                    toggle.addEventListener('click', () => {
                        const groupId = toggle.dataset.group;
                        const expanded = toggle.getAttribute('aria-expanded') === 'true';
                        const rows = document.querySelectorAll(`tr[data-group='${groupId}'].group-row-variant`);
                        rows.forEach((row) => {
                            if (expanded) {
                                row.classList.add('collapsed');
                            } else {
                                row.classList.remove('collapsed');
                            }
                        });
                        toggle.setAttribute('aria-expanded', (!expanded).toString());
                        toggle.textContent = expanded ? 'View variations' : 'Hide variations';
                    });
                });
            }

            window.addEventListener('load', () => {
                loadCharts();
                initRankingToggle();
                initGroupToggles();
            });
        </script>
    </body>
    </html>
    """
    )

    html_output = "".join(parts)
    return html_output


def run_dashboard(result: AutoMLResult, host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    """Run the dashboard server."""
    app = create_app(result)
    print(f"\n{'=' * 60}")
    print("🚀 Starting AutoML Decision Dashboard")
    print(f"📍 Access at: http://{host}:{port}")
    print("⏹️  Press Ctrl+C to stop the server")
    print(f"{'=' * 60}\n")
    app.run(host=host, port=port, debug=debug, use_reloader=False)


def run_dashboard_app(
    result: AutoMLResult,
    host: str = "127.0.0.1",
    port: int = 5000,
    width: int = 1280,
    height: int = 800,
) -> None:
    """Launch the dashboard in a desktop window instead of the browser.

    Spins up the Flask app in-process and embeds it in a pywebview window. The
    server shuts down automatically when the window closes.
    """

    try:
        import webview
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "pywebview is required for run_dashboard_app. Install the optional dependency first."
        ) from exc

    app = create_app(result)
    server = make_server(host, port, app)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://{host}:{port}"
    print(f"\n{'=' * 60}")
    print("🪟 Launching AutoML Decision Dashboard as a desktop window")
    print(f"📍 Embedded at: {url}")
    print(f"{'=' * 60}\n")

    def _on_closed() -> None:
        server.shutdown()

    webview.create_window("AutoML Decision Hub", url, width=width, height=height)
    webview.start(_on_closed)

