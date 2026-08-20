# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metrics collection and aggregation for Victor observability.

This module provides a metrics collection system with support for:
- Counters: Monotonically increasing values
- Gauges: Point-in-time values
- Histograms: Distribution of values
- Timers: Duration measurements

Design Patterns:
- Registry Pattern: MetricsRegistry for metric management
- Observer Pattern: Metrics subscribe to EventBus
- Singleton Pattern: Default registry instance

Example:
    from victor.observability.metrics import (
        MetricsRegistry,
        Counter,
        Gauge,
        Histogram,
    )

    registry = MetricsRegistry.get_instance()

    # Create metrics
    requests = registry.counter("requests_total", "Total requests")
    active_sessions = registry.gauge("active_sessions", "Active sessions")
    latency = registry.histogram("request_latency_ms", "Request latency")

    # Use metrics
    requests.increment()
    active_sessions.set(10)
    latency.observe(45.5)

    # Get all metrics
    all_metrics = registry.collect()
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, median, quantiles
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Metric Types
# =============================================================================


@dataclass
class MetricLabels:
    """Labels for metric identification."""

    labels: Dict[str, str] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(frozenset(self.labels.items()))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, MetricLabels):
            return self.labels == other.labels
        return False

    def with_label(self, key: str, value: str) -> "MetricLabels":
        """Create new labels with additional label."""
        new_labels = dict(self.labels)
        new_labels[key] = value
        return MetricLabels(new_labels)


class Metric(ABC):
    """Abstract base for all metric types."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize metric.

        Args:
            name: Metric name.
            description: Metric description.
            labels: Optional labels.
        """
        self.name = name
        self.description = description
        self._labels = MetricLabels(labels or {})
        self._lock = threading.RLock()
        self._created_at = datetime.now(timezone.utc)

    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """Collect metric value(s)."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset metric to initial state."""
        pass


class Counter(Metric):
    """Monotonically increasing counter.

    Only supports increment operations (no decrement).

    Example:
        requests = Counter("requests_total", "Total requests")
        requests.increment()
        requests.increment(5)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(name, description, labels)
        self._value: float = 0.0

    def increment(self, amount: float = 1.0) -> None:
        """Increment counter.

        Args:
            amount: Amount to increment (must be positive).
        """
        if amount < 0:
            raise ValueError("Counter increment must be positive")
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        """Get current value."""
        with self._lock:
            return self._value

    def collect(self) -> Dict[str, Any]:
        """Collect counter value."""
        return {
            "type": "counter",
            "name": self.name,
            "description": self.description,
            "labels": self._labels.labels,
            "value": self.value,
        }

    def reset(self) -> None:
        """Reset counter to zero."""
        with self._lock:
            self._value = 0.0


class Gauge(Metric):
    """Point-in-time value that can increase or decrease.

    Example:
        sessions = Gauge("active_sessions", "Active session count")
        sessions.set(10)
        sessions.increment()
        sessions.decrement(3)
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(name, description, labels)
        self._value: float = 0.0

    def set(self, value: float) -> None:
        """Set gauge value."""
        with self._lock:
            self._value = value

    def increment(self, amount: float = 1.0) -> None:
        """Increment gauge."""
        with self._lock:
            self._value += amount

    def decrement(self, amount: float = 1.0) -> None:
        """Decrement gauge."""
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        """Get current value."""
        with self._lock:
            return self._value

    def collect(self) -> Dict[str, Any]:
        """Collect gauge value."""
        return {
            "type": "gauge",
            "name": self.name,
            "description": self.description,
            "labels": self._labels.labels,
            "value": self.value,
        }

    def reset(self) -> None:
        """Reset gauge to zero."""
        with self._lock:
            self._value = 0.0


class Histogram(Metric):
    """Distribution of values with configurable buckets.

    Tracks count, sum, and bucket distribution for latency/size metrics.

    Example:
        latency = Histogram(
            "request_latency_ms",
            "Request latency",
            buckets=[10, 50, 100, 500, 1000]
        )
        latency.observe(45)
        latency.observe(250)
    """

    DEFAULT_BUCKETS = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> None:
        super().__init__(name, description, labels)
        self._buckets = buckets or self.DEFAULT_BUCKETS
        self._bucket_counts: Dict[float, int] = dict.fromkeys(self._buckets, 0)
        self._bucket_counts[float("inf")] = 0  # +Inf bucket
        self._sum: float = 0.0
        self._count: int = 0
        self._values: List[float] = []  # For percentile calculation
        self._max_values = 10000  # Limit stored values

    def observe(self, value: float) -> None:
        """Record a value observation.

        Args:
            value: Value to record.
        """
        with self._lock:
            self._count += 1
            self._sum += value

            # Update buckets
            for bucket in sorted(self._bucket_counts.keys()):
                if value <= bucket:
                    self._bucket_counts[bucket] += 1

            # Store for percentiles (with limit)
            if len(self._values) < self._max_values:
                self._values.append(value)

    @property
    def count(self) -> int:
        """Get observation count."""
        with self._lock:
            return self._count

    @property
    def sum(self) -> float:
        """Get sum of observations."""
        with self._lock:
            return self._sum

    @property
    def mean(self) -> Optional[float]:
        """Get mean of observations."""
        with self._lock:
            if self._count == 0:
                return None
            return self._sum / self._count

    def percentile(self, p: float) -> Optional[float]:
        """Get percentile value.

        Args:
            p: Percentile (0-100).

        Returns:
            Percentile value or None if no data.
        """
        with self._lock:
            if not self._values:
                return None
            if p == 50:
                return median(self._values)
            q = quantiles(self._values, n=100)
            idx = max(0, min(int(p) - 1, len(q) - 1))
            return q[idx]

    def collect(self) -> Dict[str, Any]:
        """Collect histogram data."""
        with self._lock:
            return {
                "type": "histogram",
                "name": self.name,
                "description": self.description,
                "labels": self._labels.labels,
                "count": self._count,
                "sum": self._sum,
                "mean": self.mean,
                "buckets": dict(self._bucket_counts),
                "p50": self.percentile(50),
                "p95": self.percentile(95),
                "p99": self.percentile(99),
            }

    def reset(self) -> None:
        """Reset histogram."""
        with self._lock:
            self._bucket_counts = dict.fromkeys(self._buckets, 0)
            self._bucket_counts[float("inf")] = 0
            self._sum = 0.0
            self._count = 0
            self._values.clear()


class Timer(Histogram):
    """Specialized histogram for timing operations.

    Can be used as context manager for automatic timing.

    Example:
        timer = Timer("operation_duration_ms", "Operation duration")

        # Manual timing
        with timer.time():
            do_operation()

        # Or as decorator
        @timer.timed
        def operation():
            pass
    """

    def time(self) -> "TimerContext":
        """Start timing context.

        Returns:
            Timer context manager.
        """
        return TimerContext(self)

    def timed(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator for timing functions.

        Args:
            func: Function to time.

        Returns:
            Decorated function.
        """
        import functools

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.time():
                return func(*args, **kwargs)

        return wrapper


class TimerContext:
    """Context manager for Timer."""

    def __init__(self, timer: Timer) -> None:
        self._timer = timer
        self._start: Optional[float] = None

    def __enter__(self) -> "TimerContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._start is not None:
            duration_ms = (time.perf_counter() - self._start) * 1000
            self._timer.observe(duration_ms)


# =============================================================================
# Metrics Registry (Registry Pattern)
# =============================================================================


class MetricsRegistry:
    """Central registry for all metrics.

    Provides metric creation, lookup, and collection.
    Uses singleton pattern for default instance.

    Example:
        registry = MetricsRegistry.get_instance()
        counter = registry.counter("my_counter", "Description")
        all_metrics = registry.collect()
    """

    _instance: Optional["MetricsRegistry"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "MetricsRegistry":
        """Get singleton registry instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        """Initialize registry."""
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.RLock()

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Counter:
        """Get or create a counter.

        Args:
            name: Metric name.
            description: Metric description.
            labels: Optional labels.

        Returns:
            Counter metric.
        """
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Counter(name, description, labels)
            return self._metrics[key]  # type: ignore

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Gauge:
        """Get or create a gauge.

        Args:
            name: Metric name.
            description: Metric description.
            labels: Optional labels.

        Returns:
            Gauge metric.
        """
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Gauge(name, description, labels)
            return self._metrics[key]  # type: ignore

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> Histogram:
        """Get or create a histogram.

        Args:
            name: Metric name.
            description: Metric description.
            labels: Optional labels.
            buckets: Optional bucket boundaries.

        Returns:
            Histogram metric.
        """
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Histogram(name, description, labels, buckets)
            return self._metrics[key]  # type: ignore

    def timer(
        self,
        name: str,
        description: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Timer:
        """Get or create a timer.

        Args:
            name: Metric name.
            description: Metric description.
            labels: Optional labels.

        Returns:
            Timer metric.
        """
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Timer(name, description, labels)
            return self._metrics[key]  # type: ignore

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create unique key for metric lookup."""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name

    def get(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[Metric]:
        """Get metric by name.

        Args:
            name: Metric name.
            labels: Optional labels.

        Returns:
            Metric or None.
        """
        key = self._make_key(name, labels)
        with self._lock:
            return self._metrics.get(key)

    def unregister(self, name: str, labels: Optional[Dict[str, str]] = None) -> bool:
        """Unregister a metric.

        Args:
            name: Metric name.
            labels: Optional labels.

        Returns:
            True if metric was removed.
        """
        key = self._make_key(name, labels)
        with self._lock:
            if key in self._metrics:
                del self._metrics[key]
                return True
            return False

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all metrics.

        Returns:
            List of metric dictionaries.
        """
        with self._lock:
            return [metric.collect() for metric in self._metrics.values()]

    def reset_all(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for metric in self._metrics.values():
                metric.reset()

    def clear(self) -> None:
        """Remove all metrics."""
        with self._lock:
            self._metrics.clear()

    @property
    def metric_count(self) -> int:
        """Get number of registered metrics."""
        with self._lock:
            return len(self._metrics)


# =============================================================================
# Metrics Collector (Observer Pattern)
# =============================================================================


class MetricsCollector:
    """Automatically collects metrics from Victor events.

    Subscribes to EventBus and updates metrics based on events.

    Example:
        from victor.core.events import ObservabilityBus, get_observability_bus
        from victor.observability.metrics import MetricsCollector

        collector = MetricsCollector()
        bus = get_observability_bus()
        collector.wire_observability_bus(bus)

        # Metrics are now updated automatically from events
    """

    def __init__(
        self,
        registry: Optional[MetricsRegistry] = None,
        prefix: str = "victor",
    ) -> None:
        """Initialize metrics collector.

        Args:
            registry: Metrics registry to use.
            prefix: Prefix for metric names.
        """
        self._registry = registry or MetricsRegistry.get_instance()
        self._prefix = prefix
        # Ad-hoc metrics recorded via the MetricsCollectorProtocol adapter
        # (``record_metric``). Kept separate from the fixed, event-driven metrics
        # so the adapter can accept arbitrary metric names without polluting the
        # shared registry.
        self._adhoc_metrics: Dict[str, float] = {}
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Create standard metrics."""
        # Tool metrics
        self.tool_calls = self._registry.counter(
            f"{self._prefix}_tool_calls_total",
            "Total tool calls",
        )
        self.tool_errors = self._registry.counter(
            f"{self._prefix}_tool_errors_total",
            "Total tool errors",
        )
        self.tool_duration = self._registry.histogram(
            f"{self._prefix}_tool_duration_ms",
            "Tool execution duration",
        )

        # Model metrics
        self.model_requests = self._registry.counter(
            f"{self._prefix}_model_requests_total",
            "Total model requests",
        )
        self.model_tokens = self._registry.counter(
            f"{self._prefix}_model_tokens_total",
            "Total tokens used",
        )
        self.model_latency = self._registry.histogram(
            f"{self._prefix}_model_latency_ms",
            "Model response latency",
        )

        # Session metrics
        self.active_sessions = self._registry.gauge(
            f"{self._prefix}_active_sessions",
            "Active sessions",
        )
        self.session_duration = self._registry.histogram(
            f"{self._prefix}_session_duration_seconds",
            "Session duration",
        )

        # State metrics
        self.state_transitions = self._registry.counter(
            f"{self._prefix}_state_transitions_total",
            "Total state transitions",
        )

        # Error metrics
        self.errors = self._registry.counter(
            f"{self._prefix}_errors_total",
            "Total errors",
        )

    def wire_observability_bus(self, observability_bus: Any) -> None:
        """Wire collector to ObservabilityBus.

        Args:
            observability_bus: ObservabilityBus instance.
        """
        # Subscribe to topic patterns for each category
        observability_bus.subscribe("tool.*", self._on_tool_event)
        observability_bus.subscribe("model.*", self._on_model_event)
        observability_bus.subscribe("state.*", self._on_state_event)
        observability_bus.subscribe("lifecycle.*", self._on_lifecycle_event)
        observability_bus.subscribe("error.*", self._on_error_event)

    def _on_tool_event(self, event: Any) -> None:
        """Handle tool events."""
        # Event has topic and data attributes
        if event.topic.endswith(".start") or event.topic == "tool.start":
            pass  # Track start for duration
        elif event.topic.endswith(".result") or event.topic == "tool.result":
            self.tool_calls.increment()
            if not event.data.get("success", True):
                self.tool_errors.increment()
            if "duration_ms" in event.data:
                self.tool_duration.observe(event.data["duration_ms"])

    def _on_model_event(self, event: Any) -> None:
        """Handle model events."""
        if event.topic.endswith(".request") or event.topic == "model.request":
            self.model_requests.increment()
        elif event.topic.endswith(".response") or event.topic == "model.response":
            if "total_tokens" in event.data and event.data["total_tokens"]:
                self.model_tokens.increment(event.data["total_tokens"])
            if "latency_ms" in event.data and event.data["latency_ms"]:
                self.model_latency.observe(event.data["latency_ms"])

    def _on_state_event(self, event: Any) -> None:
        """Handle state events."""
        if event.topic.endswith(".transition") or event.topic == "state.transition":
            self.state_transitions.increment()

    def _on_lifecycle_event(self, event: Any) -> None:
        """Handle lifecycle events."""
        if event.topic.endswith("session.start") or event.topic == "lifecycle.session.start":
            self.active_sessions.increment()
        elif event.topic.endswith("session.end") or event.topic == "lifecycle.session.end":
            self.active_sessions.decrement()
            # Check for duration in milliseconds or seconds
            duration_ms = event.data.get("duration_ms", 0)
            duration_sec = event.data.get("duration_seconds", 0)
            if duration_ms:
                self.session_duration.observe(duration_ms / 1000)  # Convert to seconds
            elif duration_sec:
                self.session_duration.observe(duration_sec)

    def _on_error_event(self, event: Any) -> None:
        """Handle error events."""
        self.errors.increment()

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary.

        Returns:
            Summary dictionary.
        """
        return {
            "tool_calls": self.tool_calls.value,
            "tool_errors": self.tool_errors.value,
            "tool_error_rate": (
                self.tool_errors.value / self.tool_calls.value if self.tool_calls.value > 0 else 0
            ),
            "model_requests": self.model_requests.value,
            "model_tokens": self.model_tokens.value,
            "active_sessions": self.active_sessions.value,
            "state_transitions": self.state_transitions.value,
            "errors": self.errors.value,
            "tool_latency_p50": self.tool_duration.percentile(50),
            "tool_latency_p95": self.tool_duration.percentile(95),
            "model_latency_p50": self.model_latency.percentile(50),
            "model_latency_p95": self.model_latency.percentile(95),
        }

    # =========================================================================
    # MetricsCollectorProtocol adapter (FEP-0014)
    # =========================================================================

    def record_metric(self, name: str, value: float, **tags: str) -> None:
        """Record a single ad-hoc metric observation (protocol adapter).

        Stores the value under ``name`` (tag-qualified when tags are supplied)
        in the ad-hoc metrics store. This complements the event-driven metrics
        wired via :meth:`wire_observability_bus`.

        Args:
            name: Metric name.
            value: Numeric value to record.
            **tags: Optional string-valued dimensional tags, appended to the key.
        """
        key = name
        if tags:
            tag_suffix = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            key = f"{name}{{{tag_suffix}}}"
        self._adhoc_metrics[key] = float(value)

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a point-in-time snapshot of collected metrics (protocol adapter).

        Merges the event-driven summary from :meth:`get_summary` with any ad-hoc
        metrics recorded via :meth:`record_metric`.

        Returns:
            A mapping of the collector's current metric state.
        """
        snapshot: Dict[str, Any] = dict(self.get_summary())
        snapshot.update(self._adhoc_metrics)
        return snapshot
