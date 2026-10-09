"""Resource monitoring utilities for AutoML runs."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from time import perf_counter, sleep
from typing import Callable, Dict, Tuple, TypeVar

import psutil


T = TypeVar("T")


@dataclass
class ResourceUsage:
    """Metrics captured for a model training run."""

    training_time: float
    peak_ram_mb: float  # peak memory above the process baseline
    avg_cpu_percent: float


class ResourceMonitor:
    """Tracks resource usage for function execution using psutil sampling."""

    def __init__(self, sample_interval: float = 0.2) -> None:
        if sample_interval <= 0:
            raise ValueError("sample_interval must be positive")
        self.sample_interval = sample_interval

    def measure(self, func: Callable[..., T], *args, **kwargs) -> Tuple[ResourceUsage, T]:
        """Execute the function while tracking resources and return usage and result."""

        process = psutil.Process()
        stop_event = Event()
        baseline_ram_mb = process.memory_info().rss / (1024 ** 2)
        peak_ram_mb: float = baseline_ram_mb
        cpu_samples: list[float] = []

        def sampler() -> None:
            nonlocal peak_ram_mb
            while not stop_event.is_set():
                try:
                    mem = process.memory_info().rss / (1024 ** 2)
                    peak_ram_mb = max(peak_ram_mb, mem)
                    cpu = process.cpu_percent(interval=self.sample_interval)
                    cpu_samples.append(cpu)
                except psutil.Error:  # pragma: no cover - process terminated
                    break

        # Prime cpu_percent to establish baseline measurement
        process.cpu_percent(interval=None)
        sampler_thread = Thread(target=sampler, daemon=True)

        start_time = perf_counter()
        sampler_thread.start()
        try:
            result = func(*args, **kwargs)
        finally:
            stop_event.set()
            sampler_thread.join(timeout=self.sample_interval * 2)
        end_time = perf_counter()

        elapsed = end_time - start_time
        avg_cpu = float(sum(cpu_samples) / len(cpu_samples)) if cpu_samples else 0.0
        usage = ResourceUsage(
            training_time=elapsed,
            peak_ram_mb=max(0.0, peak_ram_mb - baseline_ram_mb),
            avg_cpu_percent=avg_cpu,
        )
        return usage, result
