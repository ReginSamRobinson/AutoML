"""Custom exceptions for the AutoML framework."""

from __future__ import annotations


class ConstraintViolationError(Exception):
    """Raised when constraints are violated during AutoML."""


class ResourceLimitExceededError(Exception):
    """Raised when resource usage exceeds configured thresholds."""
