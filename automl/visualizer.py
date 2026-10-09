"""Visualization module for AutoML results."""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import pandas as pd

from .engine import AutoMLResult
from .leaderboard import LeaderboardEntry


class ResultVisualizer:
    """Converts AutoML results to visualization-friendly formats."""

    @staticmethod
    def leaderboard_to_dataframe(result: AutoMLResult) -> pd.DataFrame:
        """Convert leaderboard entries to a pandas DataFrame."""
        if not result.leaderboard:
            return pd.DataFrame()

        data = []
        for entry in result.leaderboard:
            row = {
                "Rank": entry.rank,
                "Model": entry.result.model_name,
                "Combined Score": round(entry.combined_score, 4),
                "Training Time (s)": round(entry.result.resource_usage.training_time, 2),
                "Peak RAM (MB)": round(entry.result.resource_usage.peak_ram_mb, 1),
                "Avg CPU (%)": round(entry.result.resource_usage.avg_cpu_percent, 1),
                "Violations": len(entry.result.violations),
            }
            # Add metrics
            for metric_name, metric_value in entry.result.metrics.items():
                if not metric_name.endswith("_std"):
                    row[metric_name.upper()] = round(metric_value, 4)

            data.append(row)

        return pd.DataFrame(data)

    @staticmethod
    def metrics_summary(result: AutoMLResult) -> dict[str, dict]:
        """Extract metrics for the best-ranked configuration of each model."""
        if not result.leaderboard:
            return {}

        summary = {}
        for entry in result.leaderboard:
            model_name = entry.result.model_name
            if model_name in summary:
                continue  # leaderboard is rank-ordered; keep the best config
            summary[model_name] = {}
            for metric_name, metric_value in entry.result.metrics.items():
                if not metric_name.endswith("_std"):
                    summary[model_name][metric_name] = round(metric_value, 4)

        return summary

    @staticmethod
    def resource_usage_summary(result: AutoMLResult) -> pd.DataFrame:
        """Extract resource usage data for visualization."""
        if not result.leaderboard:
            return pd.DataFrame()

        data = []
        for entry in result.leaderboard:
            data.append({
                "Model": entry.result.model_name,
                "Training Time (s)": entry.result.resource_usage.training_time,
                "Peak RAM (MB)": entry.result.resource_usage.peak_ram_mb,
                "Avg CPU (%)": entry.result.resource_usage.avg_cpu_percent,
            })

        return pd.DataFrame(data)

    @staticmethod
    def best_model_summary(result: AutoMLResult) -> dict[str, object]:
        """Extract best model information for display."""
        if result.best_result is None:
            return {"status": "No valid model satisfied the constraints"}

        best = result.best_result
        return {
            "Model": best.model_name,
            "Primary Metric": best.primary_metric,
            "Primary Metric Score": round(best.metrics[best.primary_metric], 4),
            "Training Time (s)": round(best.resource_usage.training_time, 2),
            "Peak RAM (MB)": round(best.resource_usage.peak_ram_mb, 1),
            "Avg CPU (%)": round(best.resource_usage.avg_cpu_percent, 1),
            "Violations": [v.details for v in best.violations] if best.violations else [],
            "All Metrics": {k: round(v, 4) for k, v in best.metrics.items() if not k.endswith("_std")},
        }
