"""Flask web UI for visualizing AutoML results with an explainable dark-themed dashboard."""

from __future__ import annotations

import copy
import json
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from flask import Flask, Response, jsonify, render_template, request
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from werkzeug.serving import make_server
from werkzeug.utils import secure_filename

from .config import ResourceConstraints
from .constraint import ConstraintValidator
from .dataset_analysis import DatasetAnalyzer
from .evaluation import Evaluator
from .exporter import export_to_html
from .engine import AutoMLEngine, AutoMLResult
from .model_registry import ModelRegistry
from .resource_monitor import ResourceMonitor
from .visualizer import ResultVisualizer
from .data_upload import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    SAMPLE_DATASET_FILENAME,
    allowed_file,
    create_upload_folder,
    ensure_sample_dataset,
    get_dataset_preview,
    load_dataset,
    unique_upload_path,
    validate_dataset,
)


DEFAULT_DASHBOARD_SETTINGS: dict[str, object] = {
    "cv_folds": 5,
    "metrics": ["accuracy", "f1", "roc_auc"],
    "constraints": {
        "max_training_time": 60.0,
        "max_ram_mb": 4096.0,
        "min_performance": 0.85,
        "interpretability_required": False,
        "preferred_metrics": ["accuracy"],
    },
}


SETTINGS_FILENAME = "settings.json"


def _best_per_model(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the first (best-ranked) row for each model name."""

    if frame.empty or "Model" not in frame.columns:
        return frame
    return frame.drop_duplicates(subset="Model", keep="first").reset_index(drop=True)


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_float(value: object) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _normalize_metric_list(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        return tokens
    if isinstance(raw, list):
        tokens = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                tokens.append(item.strip())
        return tokens
    return []


def _collect_datasets(upload_dir: Path) -> list[dict[str, object]]:
    datasets: list[dict[str, object]] = []
    if upload_dir.exists():
        for file_path in upload_dir.iterdir():
            if file_path.is_file() and allowed_file(file_path.name) and file_path.name != SETTINGS_FILENAME:
                try:
                    stat = file_path.stat()
                    datasets.append(
                        {
                            "name": file_path.name,
                            "size": stat.st_size,
                            "modified": stat.st_mtime,
                            "path": str(file_path),
                        }
                    )
                except OSError:
                    continue
    return sorted(datasets, key=lambda item: item["modified"], reverse=True)


def _unique_metrics(metrics: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for metric in metrics:
        lowered = metric.lower()
        if lowered not in seen:
            seen.add(lowered)
            unique.append(metric)
    return unique


def _sanitize_settings_payload(
    base_settings: dict[str, object], payload: dict[str, object]
) -> dict[str, object]:
    merged = _deep_merge(DEFAULT_DASHBOARD_SETTINGS, base_settings)
    merged = _deep_merge(merged, payload)

    constraints_in: dict[str, object] = {}
    merged_constraints = merged.get("constraints")
    if isinstance(merged_constraints, dict):
        constraints_in = dict(merged_constraints)

    preferred_metrics = _normalize_metric_list(constraints_in.get("preferred_metrics"))
    if not preferred_metrics:
        fallback = []
        base_constraints = base_settings.get("constraints")
        if isinstance(base_constraints, dict):
            fallback = _normalize_metric_list(base_constraints.get("preferred_metrics"))
        if not fallback:
            default_constraints = DEFAULT_DASHBOARD_SETTINGS.get("constraints")
            if isinstance(default_constraints, dict):
                fallback = _normalize_metric_list(default_constraints.get("preferred_metrics"))
        preferred_metrics = fallback

    sanitized_constraints = {
        "max_training_time": _coerce_float(constraints_in.get("max_training_time")),
        "max_ram_mb": _coerce_float(constraints_in.get("max_ram_mb")),
        "min_performance": _coerce_float(constraints_in.get("min_performance")),
        "interpretability_required": _coerce_bool(constraints_in.get("interpretability_required")),
        "preferred_metrics": _unique_metrics(preferred_metrics) or _unique_metrics(
            _normalize_metric_list(DEFAULT_DASHBOARD_SETTINGS["constraints"].get("preferred_metrics"))
        ),
    }

    metrics_clean = _unique_metrics(_normalize_metric_list(merged.get("metrics")))
    if not metrics_clean:
        metrics_clean = _unique_metrics(_normalize_metric_list(base_settings.get("metrics")))
    if not metrics_clean:
        metrics_clean = _unique_metrics(_normalize_metric_list(DEFAULT_DASHBOARD_SETTINGS.get("metrics")))

    cv_folds_value = merged.get("cv_folds")
    try:
        cv_folds_clean = max(2, int(cv_folds_value))
    except (TypeError, ValueError):
        try:
            cv_folds_clean = max(2, int(DEFAULT_DASHBOARD_SETTINGS.get("cv_folds", 5)))
        except (TypeError, ValueError):
            cv_folds_clean = 5

    sanitized = copy.deepcopy(DEFAULT_DASHBOARD_SETTINGS)
    sanitized["cv_folds"] = cv_folds_clean
    sanitized["metrics"] = metrics_clean
    sanitized["constraints"] = sanitized_constraints
    return sanitized


def create_app(result: Optional[AutoMLResult] = None, upload_dir: Optional[str | Path] = None) -> Flask:
    """Create and configure the Flask application.

    upload_dir overrides where datasets and settings.json are stored
    (defaults to the project's uploads/ folder).
    """
    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.template_filter("fmt")
    def format_number(value: object, spec: str = "{:.4f}") -> str:
        number = _finite(value)
        return "—" if number is None else spec.format(number)

    if upload_dir is None:
        upload_folder = create_upload_folder()
    else:
        Path(upload_dir).mkdir(parents=True, exist_ok=True)
        upload_folder = str(upload_dir)
    upload_path = Path(upload_folder)
    ensure_sample_dataset(upload_path)

    app.config["UPLOAD_FOLDER"] = upload_folder
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

    settings_path = upload_path / SETTINGS_FILENAME
    try:
        raw_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        raw_settings = {}
    settings_cache = _sanitize_settings_payload(DEFAULT_DASHBOARD_SETTINGS, raw_settings)
    settings_path.write_text(json.dumps(settings_cache, indent=2), encoding="utf-8")

    state: dict[str, Optional[AutoMLResult]] = {"result": result}

    def get_result() -> Optional[AutoMLResult]:
        return state.get("result")

    def set_result(new_result: Optional[AutoMLResult]) -> None:
        state["result"] = new_result

    def get_settings() -> dict[str, object]:
        return copy.deepcopy(settings_cache)

    def persist_settings(new_settings: dict[str, object]) -> None:
        nonlocal settings_cache
        settings_cache = copy.deepcopy(new_settings)
        settings_path.write_text(json.dumps(settings_cache, indent=2), encoding="utf-8")

    def sanitize_settings(payload: dict[str, object]) -> dict[str, object]:
        return _sanitize_settings_payload(settings_cache, payload)

    @app.route("/")
    def index():
        """Render the main dashboard or the onboarding start page."""
        current = get_result()
        view = (request.args.get("view") or "").strip().lower()
        has_result = current is not None
        if view == "dashboard" and current is not None:
            return render_template(
                "results.html", active="results", has_result=True, view=_results_view(current)
            )

        sample_dataset = SAMPLE_DATASET_FILENAME if ensure_sample_dataset(upload_path) else None
        return render_template(
            "setup.html",
            active="setup",
            has_result=has_result,
            settings=get_settings(),
            defaults=DEFAULT_DASHBOARD_SETTINGS,
            allowed_extensions=sorted(ALLOWED_EXTENSIONS),
            max_upload_mb=MAX_FILE_SIZE // (1024 * 1024),
            sample_dataset=sample_dataset,
        )

    @app.route("/vendor/plotly.min.js")
    def plotly_js():
        """Serve the plotly.js bundled with the installed plotly package.

        Keeps the browser library in sync with the figure JSON produced here
        and works without internet access.
        """
        return Response(get_plotlyjs(), mimetype="application/javascript")

    @app.route("/api/leaderboard")
    def api_leaderboard():
        """JSON payload for leaderboard values."""
        current = get_result()
        if current is None:
            return jsonify([])

        visualizer = ResultVisualizer()
        leaderboard_df = visualizer.leaderboard_to_dataframe(current)
        return jsonify(leaderboard_df.to_dict("records"))

    @app.route("/api/charts/leaderboard")
    def chart_leaderboard():
        """Generate final decision chart."""
        current = get_result()
        if current is None or not current.leaderboard:
            return jsonify({"error": "No data"})

        visualizer = ResultVisualizer()
        # One bar per model family: its best-ranked configuration.
        leaderboard_df = _best_per_model(visualizer.leaderboard_to_dataframe(current))
        if leaderboard_df.empty or "Combined Score" not in leaderboard_df.columns:
            return jsonify({"error": "No data"})

        resource_df = _best_per_model(visualizer.resource_usage_summary(current))

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

        return Response(fig.to_json(), mimetype="application/json")

    @app.route("/api/charts/metrics")
    def chart_metrics():
        """Generate model quality chart."""
        current = get_result()
        if current is None or not current.leaderboard:
            return jsonify({"error": "No data"})

        visualizer = ResultVisualizer()
        leaderboard_df = visualizer.leaderboard_to_dataframe(current)
        metrics_summary = visualizer.metrics_summary(current)

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
        return Response(fig.to_json(), mimetype="application/json")

    @app.route("/api/charts/resources")
    def chart_resources():
        """Generate resource cost chart."""
        current = get_result()
        if current is None or not current.leaderboard:
            return jsonify({"error": "No data"})

        visualizer = ResultVisualizer()
        resource_df = _best_per_model(visualizer.resource_usage_summary(current))
        leaderboard_df = _best_per_model(visualizer.leaderboard_to_dataframe(current))

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

        return Response(fig.to_json(), mimetype="application/json")

    @app.route("/api/settings", methods=["GET", "POST"])
    def api_settings():
        """Read or persist dashboard run settings."""
        if request.method == "GET":
            return jsonify({"settings": get_settings()}), 200

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid payload"}), 400

        sanitized = sanitize_settings(payload)
        persist_settings(sanitized)
        return jsonify({"settings": sanitized, "message": "Settings saved"}), 200

    @app.route("/api/run", methods=["POST"])
    def api_run():
        """Execute a fresh AutoML run using uploaded datasets and saved settings."""
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid payload"}), 400

        dataset_name = payload.get("dataset")
        target_column = payload.get("target")
        if not dataset_name or not target_column:
            return jsonify({"error": "Dataset name and target column are required"}), 400

        filename = secure_filename(str(dataset_name))
        dataset_path = Path(app.config["UPLOAD_FOLDER"]) / filename
        if not dataset_path.exists():
            return jsonify({"error": "Dataset not found"}), 404

        df, load_error = load_dataset(str(dataset_path))
        if load_error:
            return jsonify({"error": load_error}), 400

        if str(target_column) not in df.columns:
            return jsonify({"error": f"Target column '{target_column}' not found"}), 400

        if df.shape[1] <= 1:
            return jsonify({"error": "Dataset must contain at least one feature column besides the target"}), 400

        features_df = df.drop(columns=[target_column])
        if features_df.shape[1] == 0:
            return jsonify({"error": "Dataset must contain at least one feature column besides the target"}), 400
        target_series = df[target_column]

        overrides: dict[str, object] = {}
        if isinstance(payload.get("settings"), dict):
            overrides = copy.deepcopy(payload["settings"])

        if isinstance(payload.get("constraints"), dict):
            constraint_override = copy.deepcopy(payload["constraints"])
            existing_constraints = overrides.get("constraints")
            if isinstance(existing_constraints, dict):
                overrides["constraints"] = _deep_merge(existing_constraints, constraint_override)
            else:
                overrides["constraints"] = constraint_override

        if payload.get("metrics") is not None:
            overrides["metrics"] = payload["metrics"]

        if payload.get("cv_folds") is not None and "cv_folds" not in overrides:
            overrides["cv_folds"] = payload["cv_folds"]

        sanitized = sanitize_settings(overrides)
        persist_settings(sanitized)

        constraint_payload = sanitized.get("constraints", {})
        constraints = ResourceConstraints(
            max_training_time=constraint_payload.get("max_training_time"),
            max_ram_mb=constraint_payload.get("max_ram_mb"),
            min_performance=constraint_payload.get("min_performance"),
            interpretability_required=constraint_payload.get("interpretability_required", False),
            preferred_metrics=constraint_payload.get("preferred_metrics"),
        )

        metrics = sanitized.get("metrics", [])
        if not isinstance(metrics, list) or not metrics:
            metrics = DEFAULT_DASHBOARD_SETTINGS["metrics"]

        engine = AutoMLEngine(
            analyzer=DatasetAnalyzer(),
            registry=ModelRegistry(),
            validator_factory=lambda cfg: ConstraintValidator(cfg),
            evaluator=Evaluator(cv_folds=sanitized["cv_folds"]),
            monitor=ResourceMonitor(),
        )

        try:
            result_obj = engine.run(
                features_df,
                target_series,
                constraints=constraints,
                metrics=[metric for metric in metrics if isinstance(metric, str) and metric],
            )
        except Exception as exc:
            return jsonify({"error": f"AutoML run failed: {exc}"}), 500

        set_result(result_obj)

        report_path = Path(upload_folder).parent / "automl_report.html"
        try:
            export_to_html(result_obj, str(report_path))
        except Exception:
            pass

        best_summary: dict[str, object] = {}
        if result_obj.best_result is not None:
            best = result_obj.best_result
            best_summary = {
                "model": best.model_name,
                "primary_metric": best.primary_metric,
                "primary_metric_value": best.metrics.get(best.primary_metric),
                "training_time": best.resource_usage.training_time,
                "peak_ram_mb": best.resource_usage.peak_ram_mb,
                "violations": [violation.details for violation in best.violations],
            }

        response = {
            "success": True,
            "message": "AutoML run complete",
            "leaderboard_size": len(result_obj.leaderboard),
            "best_result": best_summary,
            "redirect": "/?view=dashboard",
        }
        return jsonify(response), 200

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
            filepath = unique_upload_path(
                Path(app.config['UPLOAD_FOLDER']), secure_filename(file.filename)
            )
            filename = filepath.name
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
            datasets = _collect_datasets(Path(app.config['UPLOAD_FOLDER']))
            return jsonify({'datasets': datasets}), 200

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

            is_dataset = allowed_file(filename) and filename != SETTINGS_FILENAME
            if not is_dataset or not filepath.exists():
                return jsonify({'error': 'Dataset not found'}), 404

            filepath.unlink()
            return jsonify({'success': True, 'message': f'Dataset {filename} deleted'}), 200
        
        except Exception as e:
            return jsonify({'error': f'Failed to delete dataset: {str(e)}'}), 500

    return app


_MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "elastic_net": "Elastic net",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "xgboost": "XGBoost",
}

_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "f1": "F1",
    "roc_auc": "ROC AUC",
    "precision": "Precision",
    "recall": "Recall",
    "r2": "R²",
    "rmse": "RMSE",
    "mse": "MSE",
    "mae": "MAE",
    "mape": "MAPE",
}

_VIOLATION_LABELS = {
    "training_time": "Time limit",
    "memory": "Memory limit",
    "performance": "Below threshold",
    "runtime_error": "Failed",
}


def _model_label(name: str) -> str:
    if name in _MODEL_LABELS:
        return _MODEL_LABELS[name]
    return name.replace("_", " ").capitalize() if name.islower() else name


def _metric_label(name: str) -> str:
    return _METRIC_LABELS.get(name, name.replace("_", " ").upper())


def _finite(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _format_params(params: dict[str, object] | None) -> str:
    if not params:
        return "defaults"
    parts = []
    for key in sorted(params):
        value = params[key]
        shown = f"{value:g}" if isinstance(value, float) else str(value)
        parts.append(f"{key.removeprefix('model__')}={shown}")
    return ", ".join(parts)


def _comparison_rows(selected: dict, other: dict, primary: str, higher_is_better: bool) -> list[dict]:
    """Side-by-side figures for the selected model and the runner-up."""

    rows = []

    def add(label: str, a: float | None, b: float | None, fmt: str, better_if_higher: bool, delta: str) -> None:
        better = None
        if a is not None and b is not None and not math.isclose(a, b):
            better = (a > b) == better_if_higher
        rows.append({"label": label, "selected": fmt.format(a) if a is not None else "—",
                     "other": fmt.format(b) if b is not None else "—", "delta": delta, "better": better})

    a, b = selected["metrics"].get(primary), other["metrics"].get(primary)
    add(_metric_label(primary), a, b, "{:.4f}", higher_is_better,
        f"{a - b:+.4f}" if a is not None and b is not None else "")

    a, b = selected["time"], other["time"]
    if a and b:
        ratio = b / a
        delta = f"{ratio:.1f}× faster" if ratio >= 1 else f"{1 / ratio:.1f}× slower"
    else:
        delta = ""
    add("Training time", a, b, "{:.2f} s", False, delta)

    a, b = selected["ram"], other["ram"]
    add("Extra memory", a, b, "{:.1f} MB", False, f"{a - b:+.1f} MB" if a is not None and b is not None else "")

    a, b = selected["score"], other["score"]
    add("Score", a, b, "{:.3f}", True, f"{a - b:+.3f}" if a is not None and b is not None else "")
    return rows


def _results_view(result: AutoMLResult) -> dict[str, object]:
    """Flatten an AutoMLResult into what the results template and charts need."""

    entries = result.leaderboard or []
    view: dict[str, object] = {
        "rows": [],
        "metrics": [],
        "selected": None,
        "runner_up": None,
        "comparison": [],
        "closest": None,
        "primary": {"key": "", "label": "metric", "higher_is_better": True},
        "n_candidates": 0,
        "n_models": 0,
        "n_passing": 0,
        "n_failed": 0,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    if not entries:
        return view

    first = entries[0].result
    primary = first.primary_metric
    metric_names = [primary] + [m for m in first.metrics if not m.endswith("_std") and m != primary]
    higher_is_better = first.metric_preferences.get(primary, True)

    rows: list[dict] = []
    seen_models: set[str] = set()
    for entry in entries:
        res = entry.result
        failed = any(v.name == "runtime_error" for v in res.violations)
        rows.append(
            {
                "rank": entry.rank,
                "model": res.model_name,
                "label": _model_label(res.model_name),
                "params": _format_params(res.params),
                "metrics": {m: _finite(res.metrics.get(m)) for m in metric_names},
                "stds": {m: _finite(res.metrics.get(f"{m}_std")) for m in metric_names},
                "score": None if failed else _finite(entry.combined_score),
                "time": _finite(res.resource_usage.training_time) if not failed else None,
                "ram": _finite(res.resource_usage.peak_ram_mb) if not failed else None,
                "cpu": _finite(res.resource_usage.avg_cpu_percent) if not failed else None,
                "violations": [
                    {"label": _VIOLATION_LABELS.get(v.name, v.name), "details": v.details} for v in res.violations
                ],
                "failed": failed,
                "selected": result.best_result is not None and res is result.best_result,
                "best_of_model": res.model_name not in seen_models,
            }
        )
        seen_models.add(res.model_name)

    selected = next((row for row in rows if row["selected"]), None)
    runner_up = None
    if selected is not None:
        runner_up = next(
            (r for r in rows if r["best_of_model"] and r["model"] != selected["model"] and not r["failed"]),
            None,
        )

    view.update(
        rows=rows,
        metrics=[{"key": m, "label": _metric_label(m)} for m in metric_names],
        primary={"key": primary, "label": _metric_label(primary), "higher_is_better": higher_is_better},
        selected=selected,
        closest=rows[0] if selected is None else None,
        runner_up=runner_up,
        comparison=_comparison_rows(selected, runner_up, primary, higher_is_better) if selected and runner_up else [],
        n_candidates=len(rows),
        n_models=len(seen_models),
        n_passing=sum(1 for r in rows if not r["violations"]),
        n_failed=sum(1 for r in rows if r["failed"]),
    )
    return view


def run_dashboard(result: Optional[AutoMLResult] = None, host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
    """Run the dashboard server."""
    app = create_app(result)
    print(f"\n{'=' * 60}")
    print("🚀 Starting AutoML Decision Dashboard")
    print(f"📋 Launchpad: http://{host}:{port}/")
    print(f"📊 Results:  http://{host}:{port}/?view=dashboard")
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

