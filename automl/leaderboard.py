"""Leaderboard logic for ranking model runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .evaluation import EvaluationResult


@dataclass
class LeaderboardEntry:
    """Holds ranking information for a model run."""

    rank: int
    result: EvaluationResult
    combined_score: float


FAILED_RUN_SCORE = -1.0  # below any score a completed run can reach


def _is_failed(result: EvaluationResult) -> bool:
    """Return True when the run crashed or produced no primary metric."""

    value = result.metrics.get(result.primary_metric, float("nan"))
    return any(v.name == "runtime_error" for v in result.violations) or bool(np.isnan(value))


class Leaderboard:
    """Maintains and sorts evaluation results."""

    def __init__(self, performance_weight: float = 0.7) -> None:
        if not 0.0 < performance_weight <= 1.0:
            raise ValueError("performance_weight must be in (0, 1]")
        self.performance_weight = performance_weight
        self.resource_weight = 1.0 - performance_weight
        self._results: List[EvaluationResult] = []
        self._entries: List[LeaderboardEntry] = []

    def add(self, result: EvaluationResult) -> None:
        """Add a new result to the leaderboard."""

        self._results.append(result)

    @property
    def has_results(self) -> bool:
        """Whether any result has been added."""

        return bool(self._results)

    def build(self) -> List[LeaderboardEntry]:
        """Return ranked leaderboard entries."""

        if not self._results:
            self._entries = []
            return self._entries

        # Failed runs report zero time/RAM; keeping them out of normalisation stops
        # them from looking like the most efficient candidates.
        completed = [res for res in self._results if not _is_failed(res)]
        failed = [res for res in self._results if _is_failed(res)]

        entries: List[LeaderboardEntry] = []
        if completed:
            primary_metric = completed[0].primary_metric
            primary_pref = completed[0].metric_preferences.get(primary_metric, True)
            performances = np.array([res.metrics[primary_metric] for res in completed], dtype=float)
            times = np.array([res.resource_usage.training_time for res in completed], dtype=float)
            memories = np.array([res.resource_usage.peak_ram_mb for res in completed], dtype=float)
            cpus = np.array([res.resource_usage.avg_cpu_percent for res in completed], dtype=float)

            perf_norm = self._normalize(performances, higher_is_better=primary_pref)
            time_norm = self._normalize(times, higher_is_better=False)
            memory_norm = self._normalize(memories, higher_is_better=False)
            cpu_norm = self._normalize(cpus, higher_is_better=False)

            for idx, result in enumerate(completed):
                resource_efficiency = 0.5 * time_norm[idx] + 0.3 * memory_norm[idx] + 0.2 * cpu_norm[idx]
                resource_cost = 1.0 - resource_efficiency
                combined = self.performance_weight * perf_norm[idx] - self.resource_weight * resource_cost
                if result.violations:
                    combined -= 0.1 * len(result.violations)
                entries.append(LeaderboardEntry(rank=0, result=result, combined_score=float(combined)))

        entries.sort(key=lambda entry: entry.combined_score, reverse=True)
        entries.extend(LeaderboardEntry(rank=0, result=res, combined_score=FAILED_RUN_SCORE) for res in failed)
        for rank, entry in enumerate(entries, start=1):
            entry.rank = rank

        self._entries = entries
        return self._entries

    def as_records(self) -> List[dict[str, object]]:
        """Return leaderboard entries as serializable dictionaries."""

        if not self._entries:
            self.build()
        records: List[dict[str, object]] = []
        for entry in self._entries:
            result = entry.result
            records.append(
                {
                    "rank": entry.rank,
                    "model": result.model_name,
                    "primary_metric": result.primary_metric,
                    "metric_value": result.metrics.get(result.primary_metric),
                    "training_time": result.resource_usage.training_time,
                    "peak_ram_mb": result.resource_usage.peak_ram_mb,
                    "avg_cpu_percent": result.resource_usage.avg_cpu_percent,
                    "combined_score": entry.combined_score,
                    "violations": [violation.details for violation in result.violations],
                }
            )
        return records

    def best(self) -> Optional[EvaluationResult]:
        """Return the highest-ranked valid result."""

        if not self._entries:
            self.build()

        for entry in self._entries:
            if not entry.result.violations:
                return entry.result
        return None

    @staticmethod
    def _normalize(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
        """Map values to [0, 1] where 1 is the best value in the given direction.

        NaN values are treated as the worst observed value.
        """

        if values.size == 0:
            return values

        if np.all(np.isnan(values)):
            return np.zeros_like(values)

        min_val = float(np.nanmin(values))
        max_val = float(np.nanmax(values))
        if np.isclose(max_val, min_val):
            return np.ones_like(values)
        filled = np.nan_to_num(values, nan=min_val if higher_is_better else max_val)
        scaled = (filled - min_val) / (max_val - min_val)
        return scaled if higher_is_better else 1.0 - scaled
