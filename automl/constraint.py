"""Constraint handling for the AutoML framework."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List

from .config import ResourceConstraints
from .resource_monitor import ResourceUsage


@dataclass
class ConstraintViolation:
    """Represents a violated constraint with context."""

    name: str
    details: str


class ConstraintValidator:
    """Validates resource and performance constraints."""

    def __init__(self, constraints: ResourceConstraints) -> None:
        self.constraints = constraints

    def validate(
        self,
        performance: Dict[str, float],
        resource_usage: ResourceUsage,
        primary_metric: str,
        metric_preferences: Dict[str, bool],
    ) -> List[ConstraintViolation]:
        """Return constraint violations for provided performance and resource usage."""

        violations: List[ConstraintViolation] = []

        if self.constraints.max_training_time is not None and resource_usage.training_time > self.constraints.max_training_time:
            violations.append(
                ConstraintViolation(
                    name="training_time",
                    details=f"Training time {resource_usage.training_time:.2f}s exceeds limit {self.constraints.max_training_time:.2f}s",
                )
            )

        if self.constraints.max_ram_mb is not None and resource_usage.peak_ram_mb > self.constraints.max_ram_mb:
            violations.append(
                ConstraintViolation(
                    name="memory",
                    details=f"Peak RAM {resource_usage.peak_ram_mb:.1f}MB exceeds limit {self.constraints.max_ram_mb:.1f}MB",
                )
            )

        if self.constraints.min_performance is not None:
            primary_value = performance.get(primary_metric)
            higher_is_better = metric_preferences.get(primary_metric, True)
            if (
                primary_value is None
                or math.isnan(primary_value)
                or (
                    higher_is_better
                    and primary_value < self.constraints.min_performance
                )
                or (
                    not higher_is_better
                    and primary_value > self.constraints.min_performance
                )
            ):
                value = float("nan") if primary_value is None else primary_value
                comparison = "is below minimum" if higher_is_better else "exceeds maximum"
                violations.append(
                    ConstraintViolation(
                        name="performance",
                        details=(
                            f"Metric '{primary_metric}' {value:.4f} {comparison} "
                            f"{self.constraints.min_performance:.4f}"
                        ),
                    )
                )

        return violations
