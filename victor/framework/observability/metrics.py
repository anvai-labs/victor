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

"""Observability metrics data models.

This module provides data models for collecting and aggregating metrics
from various components in the Victor framework.

Design Pattern: Value Object Pattern + Protocol Pattern
- Immutable metric data models
- Protocol-based metric source interface
- Type-safe metric collection

Phase 4: Enhance Observability with Unified Dashboard
"""

from __future__ import annotations

import csv
from victor.core.json_utils import json_dumps
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)
from collections import defaultdict


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"  # Cumulative value that only increases
    GAUGE = "gauge"  # Point-in-time value that can go up or down
    HISTOGRAM = "histogram"  # Distribution of values
    SUMMARY = "summary"  # Aggregated statistics (count, sum, avg, min, max)


@dataclass(frozen=True)
class MetricLabel:
    """A key-value label for a metric.

    Labels allow filtering and grouping of metrics.
    """

    key: str
    value: str

    def __post_init__(self):
        """Validate label format."""
        if not self.key:
            raise ValueError("Label key cannot be empty")
        if ":" in self.key:
            raise ValueError("Label key cannot contain ':'")


@dataclass(frozen=True)
class Metric:
    """Base metric data model.

    Attributes:
        name: Metric name (e.g., "cache_hits", "tool_calls")
        description: Human-readable description
        metric_type: Type of metric (counter, gauge, histogram, summary)
        labels: Optional labels for filtering/grouping
        timestamp: When the metric was recorded
    """

    name: str
    description: str
    metric_type: MetricType
    labels: tuple[MetricLabel, ...] = field(default_factory=tuple)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        """Validate metric format."""
        if not self.name:
            raise ValueError("Metric name cannot be empty")

        # Ensure label keys are unique
        label_keys = [label.key for label in self.labels]
        if len(label_keys) != len(set(label_keys)):
            raise ValueError("Metric label keys must be unique")

        # Guard against clock skew in timestamps (allow 1 minute tolerance)
        if self.timestamp > time.time() + 60:
            raise ValueError(f"Metric timestamp is too far in the future: {self.timestamp}")

    def with_labels(self, **kwargs: str) -> "Metric":
        """Return a new metric with additional labels.

        Args:
            **kwargs: Label key-value pairs

        Returns:
            New metric with merged labels
        """
        existing_labels = {label.key: label.value for label in self.labels}
        existing_labels.update(kwargs)

        new_labels = tuple(MetricLabel(key=k, value=v) for k, v in sorted(existing_labels.items()))

        return type(self)(
            name=self.name,
            description=self.description,
            metric_type=self.metric_type,
            labels=new_labels,
        )

    def get_label_value(self, key: str) -> Optional[str]:
        """Get label value by key.

        Args:
            key: Label key

        Returns:
            Label value or None if not found
        """
        for label in self.labels:
            if label.key == key:
                return label.value
        return None


@dataclass(frozen=True)
class CounterMetric(Metric):
    """A counter metric that only increases.

    Counters are used for cumulative values like:
    - Total requests
    - Total errors
    - Total cache hits

    Attributes:
        value: Current counter value
    """

    value: int = 0

    def __post_init__(self):
        """Initialize as counter type."""
        object.__setattr__(self, "metric_type", MetricType.COUNTER)

    def increment(self, amount: int = 1) -> "CounterMetric":
        """Return a new counter with incremented value.

        Args:
            amount: Amount to increment (default: 1)

        Returns:
            New counter metric
        """
        return type(self)(
            name=self.name,
            description=self.description,
            metric_type=MetricType.COUNTER,
            labels=self.labels,
            value=self.value + amount,
        )


@dataclass(frozen=True)
class GaugeMetric(Metric):
    """A gauge metric that can go up or down.

    Gauges are used for point-in-time values like:
    - Current memory usage
    - Active connections
    - Queue size

    Attributes:
        value: Current gauge value
        min_value: Minimum observed value
        max_value: Maximum observed value
    """

    value: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def __post_init__(self):
        """Initialize as gauge type."""
        object.__setattr__(self, "metric_type", MetricType.GAUGE)

    def set(self, value: float) -> "GaugeMetric":
        """Return a new gauge with updated value.

        Args:
            value: New gauge value

        Returns:
            New gauge metric with updated min/max
        """
        new_min = self.min_value
        new_max = self.max_value

        if new_min is None or value < new_min:
            new_min = value
        if new_max is None or value > new_max:
            new_max = value

        return type(self)(
            name=self.name,
            description=self.description,
            metric_type=MetricType.GAUGE,
            labels=self.labels,
            value=value,
            min_value=new_min,
            max_value=new_max,
        )


@dataclass(frozen=True)
class HistogramBucket:
    """A histogram bucket.

    Attributes:
        upper_bound: Upper bound of the bucket (exclusive)
        count: Number of observations in this bucket
    """

    upper_bound: float
    count: int = 0


@dataclass(frozen=True)
class HistogramMetric(Metric):
    """A histogram metric that tracks distribution of values.

    Histograms are used for distributions like:
    - Request latency
    - Response sizes
    - Processing times

    Attributes:
        count: Total number of observations
        sum: Sum of all observations
        buckets: Histogram buckets
    """

    count: int = 0
    sum: float = 0.0
    buckets: tuple[HistogramBucket, ...] = field(default_factory=tuple)

    def __post_init__(self):
        """Initialize as histogram type."""
        object.__setattr__(self, "metric_type", MetricType.HISTOGRAM)

    def observe(self, value: float) -> "HistogramMetric":
        """Return a new histogram with the observation added.

        Args:
            value: Observed value

        Returns:
            New histogram metric
        """
        new_count = self.count + 1
        new_sum = self.sum + value

        # Update buckets
        new_buckets = []
        for bucket in self.buckets:
            if value < bucket.upper_bound:
                new_buckets.append(
                    HistogramBucket(upper_bound=bucket.upper_bound, count=bucket.count + 1)
                )
            else:
                new_buckets.append(bucket)

        return type(self)(
            name=self.name,
            description=self.description,
            metric_type=MetricType.HISTOGRAM,
            labels=self.labels,
            count=new_count,
            sum=new_sum,
            buckets=tuple(new_buckets),
        )

    @property
    def average(self) -> float:
        """Average value of observations."""
        return self.sum / self.count if self.count > 0 else 0.0


@dataclass(frozen=True)
class SummaryMetric(Metric):
    """A summary metric with aggregated statistics.

    Summaries provide pre-calculated statistics like:
    - Cache hit rates
    - Success rates
    - Average response times

    Attributes:
        count: Total count
        sum: Sum of values
        average: Average value
        min_value: Minimum value
        max_value: Maximum value
        p50: 50th percentile (median)
        p95: 95th percentile
        p99: 99th percentile
    """

    count: int = 0
    sum: float = 0.0
    average: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None

    def __post_init__(self):
        """Initialize as summary type."""
        object.__setattr__(self, "metric_type", MetricType.SUMMARY)


@dataclass(frozen=True)
class MetricsSnapshot:
    """A snapshot of metrics at a point in time.

    Attributes:
        source_id: Unique identifier for the metrics source
        source_type: Type of metrics source (capability, cache, coordinator, etc.)
        timestamp: When the snapshot was taken
        metrics: All metrics in this snapshot
    """

    source_id: str
    source_type: str
    timestamp: float = field(default_factory=time.time)
    metrics: tuple[Metric, ...] = field(default_factory=tuple)

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name.

        Args:
            name: Metric name

        Returns:
            Metric or None if not found
        """
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def filter_by_labels(self, **kwargs: str) -> "MetricsSnapshot":
        """Return a new snapshot with only matching metrics.

        Args:
            **kwargs: Label filters (key=value)

        Returns:
            New snapshot with filtered metrics
        """
        filtered_metrics = []
        for metric in self.metrics:
            matches_all = True
            for key, value in kwargs.items():
                if metric.get_label_value(key) != value:
                    matches_all = False
                    break
            if matches_all:
                filtered_metrics.append(metric)

        return type(self)(
            source_id=self.source_id,
            source_type=self.source_type,
            metrics=tuple(filtered_metrics),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary.

        Returns:
            Dictionary representation of snapshot
        """
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "metrics": [
                {
                    "name": m.name,
                    "description": m.description,
                    "type": m.metric_type.value,
                    "labels": {label.key: label.value for label in m.labels},
                    "value": _extract_metric_value(m),
                }
                for m in self.metrics
            ],
        }


def _extract_metric_value(metric: Metric) -> Dict[str, Any]:
    """Extract value from metric based on type.

    Args:
        metric: Metric to extract value from

    Returns:
        Dictionary with metric-specific value fields
    """
    if isinstance(metric, CounterMetric):
        return {"value": metric.value}
    elif isinstance(metric, GaugeMetric):
        return {
            "value": metric.value,
            "min": metric.min_value,
            "max": metric.max_value,
        }
    elif isinstance(metric, HistogramMetric):
        return {
            "count": metric.count,
            "sum": metric.sum,
            "average": metric.average,
            "buckets": [{"upper_bound": b.upper_bound, "count": b.count} for b in metric.buckets],
        }
    elif isinstance(metric, SummaryMetric):
        return {
            "count": metric.count,
            "sum": metric.sum,
            "average": metric.average,
            "min": metric.min_value,
            "max": metric.max_value,
            "p50": metric.p50,
            "p95": metric.p95,
            "p99": metric.p99,
        }
    else:
        return {}


@runtime_checkable
class MetricSource(Protocol):
    """Protocol for components that can provide metrics.

    Any component that implements this protocol can be registered
    with the ObservabilityManager for metrics collection.
    """

    source_id: str
    source_type: str

    def get_metrics(self) -> MetricsSnapshot:
        """Get current metrics snapshot.

        Returns:
            MetricsSnapshot with current metrics
        """
        ...

    def get_observability_data(self) -> Dict[str, Any]:
        """Get observability data for dashboard.

        Returns:
            Dictionary with observability data
        """
        ...


@dataclass
class MetricsCollection:
    """A collection of metrics from multiple sources.

    Attributes:
        snapshots: Collection of metrics snapshots
        timestamp: When the collection was created
    """

    snapshots: List[MetricsSnapshot] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def add_snapshot(self, snapshot: MetricsSnapshot) -> None:
        """Add a metrics snapshot to the collection.

        Args:
            snapshot: Snapshot to add
        """
        self.snapshots.append(snapshot)

    def get_by_source_type(self, source_type: str) -> List[MetricsSnapshot]:
        """Get all snapshots from a specific source type.

        Args:
            source_type: Type of source (e.g., "capability", "cache")

        Returns:
            List of snapshots from matching sources
        """
        return [s for s in self.snapshots if s.source_type == source_type]

    def get_by_source_id(self, source_id: str) -> Optional[MetricsSnapshot]:
        """Get snapshot by source ID.

        Args:
            source_id: Unique source identifier

        Returns:
            Snapshot or None if not found
        """
        for snapshot in self.snapshots:
            if snapshot.source_id == source_id:
                return snapshot
        return None

    def aggregate_metric(self, metric_name: str) -> List[Metric]:
        """Aggregate all metrics with the given name across sources.

        Args:
            metric_name: Name of metric to aggregate

        Returns:
            List of metrics with matching names
        """
        metrics = []
        for snapshot in self.snapshots:
            metric = snapshot.get_metric(metric_name)
            if metric:
                metrics.append(metric)
        return metrics

    def to_dict(self) -> Dict[str, Any]:
        """Convert collection to dictionary.

        Returns:
            Dictionary representation of collection
        """
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "sources_count": len(self.snapshots),
            "snapshots": [s.to_dict() for s in self.snapshots],
            "by_type": {
                source_type: len(self.get_by_source_type(source_type))
                for source_type in set(s.source_type for s in self.snapshots)
            },
        }


# Common metric name constants


class MetricNames:
    """Standard metric names used across the framework."""

    # Capability metrics
    CAPABILITY_ACCESS_COUNT = "capability_access_count"
    CAPABILITY_LAST_ACCESSED = "capability_last_accessed"
    CAPABILITY_ERROR_COUNT = "capability_error_count"

    # Cache metrics
    CACHE_HIT_COUNT = "cache_hit_count"
    CACHE_MISS_COUNT = "cache_miss_count"
    CACHE_HIT_RATE = "cache_hit_rate"
    CACHE_SIZE = "cache_size"
    CACHE_EVICTION_COUNT = "cache_eviction_count"

    # Tool metrics
    TOOL_CALL_COUNT = "tool_call_count"
    TOOL_ERROR_COUNT = "tool_error_count"
    TOOL_LATENCY = "tool_latency"
    TOOL_SUCCESS_RATE = "tool_success_rate"

    # Coordinator metrics
    COORDINATOR_OPERATION_COUNT = "coordinator_operation_count"
    COORDINATOR_ERROR_COUNT = "coordinator_error_count"
    COORDINATOR_LATENCY = "coordinator_latency"

    # Vertical metrics
    VERTICAL_REQUEST_COUNT = "vertical_request_count"
    VERTICAL_ERROR_COUNT = "vertical_error_count"
    VERTICAL_LATENCY = "vertical_latency"

    # System metrics
    SYSTEM_MEMORY_USAGE = "system_memory_usage"
    SYSTEM_CPU_USAGE = "system_cpu_usage"
    SYSTEM_DISK_USAGE = "system_disk_usage"


__all__ = [
    # Enums
    "MetricType",
    # Data models
    "MetricLabel",
    "Metric",
    "CounterMetric",
    "GaugeMetric",
    "HistogramBucket",
    "HistogramMetric",
    "SummaryMetric",
    "MetricsSnapshot",
    "MetricsCollection",
    # Agent metrics
    "AgentMetrics",
    "ToolCallMetrics",
    "LLMCallMetrics",
    # Collector
    "MetricsCollector",
    "MetricsExporter",
    # Protocols
    "MetricSource",
    # Constants
    "MetricNames",
]


# =====================================================================
# Agent-Specific Metrics
# =====================================================================


@dataclass
class ToolCallMetrics:
    """Metrics for a single tool call.

    Tracks execution details for tool usage analysis.
    """

    tool_name: str
    start_time: float
    end_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def duration(self) -> Optional[float]:
        """Get call duration in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tool_name": self.tool_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "success": self.success,
            "error_message": self.error_message,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM call.

    Tracks model invocation details for cost and performance analysis.
    """

    provider: str
    model: str
    start_time: float
    end_time: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    success: bool = False
    error_message: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        """Get call duration in seconds."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
            + self.reasoning_tokens
        )

    @property
    def cached_tokens(self) -> int:
        """Get total cached tokens."""
        return self.cache_read_tokens + self.cache_write_tokens

    def estimated_cost(self, input_price: float = 0.0, output_price: float = 0.0) -> float:
        """Estimate cost based on token pricing.

        Args:
            input_price: Price per million input tokens
            output_price: Price per million output tokens

        Returns:
            Estimated cost in dollars
        """
        input_cost = (self.input_tokens / 1_000_000) * input_price
        output_cost = (self.output_tokens / 1_000_000) * output_price
        return input_cost + output_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "provider": self.provider,
            "model": self.model,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class AgentMetrics:
    """Enhanced metrics dataclass for Agent execution tracking.

    Provides comprehensive metrics for agent operations including:
    - Token usage (with breakdown by type)
    - Tool call tracking (with latency and success rates)
    - LLM call tracking (with model-specific stats)
    - Performance metrics (latency, throughput)
    - Error tracking and counts

    Example:
        >>> metrics = AgentMetrics(agent_id="my-agent")
        >>> metrics.start_timer("total_duration")
        >>> metrics.record_llm_call(LLMCallMetrics(...))
        >>> metrics.stop_timer("total_duration")
        >>> print(metrics.summary())
    """

    # Agent identification
    agent_id: str
    session_id: Optional[str] = None

    # Timestamps
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Token usage (cumulative)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_reasoning_tokens: int = 0

    # Tool tracking
    tool_calls: List[ToolCallMetrics] = field(default_factory=list)

    # LLM call tracking
    llm_calls: List[LLMCallMetrics] = field(default_factory=list)

    # State tracking
    state_transitions: int = 0
    current_state: Optional[str] = None

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Custom timers for performance measurement
    _timers: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used across all calls."""
        return (
            self.total_input_tokens
            + self.total_output_tokens
            + self.total_cache_read_tokens
            + self.total_cache_write_tokens
            + self.total_reasoning_tokens
        )

    @property
    def total_cached_tokens(self) -> int:
        """Get total cached tokens."""
        return self.total_cache_read_tokens + self.total_cache_write_tokens

    @property
    def duration(self) -> Optional[float]:
        """Get total execution duration."""
        if self.started_at is None or self.completed_at is None:
            return None
        return self.completed_at - self.started_at

    @property
    def tool_call_count(self) -> int:
        """Get total number of tool calls."""
        return len(self.tool_calls)

    @property
    def successful_tool_calls(self) -> int:
        """Get number of successful tool calls."""
        return sum(1 for tc in self.tool_calls if tc.success)

    @property
    def failed_tool_calls(self) -> int:
        """Get number of failed tool calls."""
        return sum(1 for tc in self.tool_calls if not tc.success)

    @property
    def tool_success_rate(self) -> float:
        """Get tool call success rate."""
        total = self.tool_call_count
        if total == 0:
            return 0.0
        return self.successful_tool_calls / total

    @property
    def llm_call_count(self) -> int:
        """Get total number of LLM calls."""
        return len(self.llm_calls)

    @property
    def successful_llm_calls(self) -> int:
        """Get number of successful LLM calls."""
        return sum(1 for llm in self.llm_calls if llm.success)

    @property
    def failed_llm_calls(self) -> int:
        """Get number of failed LLM calls."""
        return sum(1 for llm in self.llm_calls if not llm.success)

    @property
    def llm_success_rate(self) -> float:
        """Get LLM call success rate."""
        total = self.llm_call_count
        if total == 0:
            return 0.0
        return self.successful_llm_calls / total

    @property
    def total_llm_duration(self) -> float:
        """Get total LLM call duration."""
        return sum(llm.duration or 0 for llm in self.llm_calls)

    @property
    def total_tool_duration(self) -> float:
        """Get total tool call duration."""
        return sum(tc.duration or 0 for tc in self.tool_calls)

    @property
    def average_llm_duration(self) -> float:
        """Get average LLM call duration."""
        count = self.llm_call_count
        if count == 0:
            return 0.0
        return self.total_llm_duration / count

    @property
    def average_tool_duration(self) -> float:
        """Get average tool call duration."""
        count = self.tool_call_count
        if count == 0:
            return 0.0
        return self.total_tool_duration / count

    def start(self) -> None:
        """Mark the agent execution as started."""
        self.started_at = time.time()

    def complete(self) -> None:
        """Mark the agent execution as completed."""
        if self.started_at is None:
            self.started_at = self.created_at
        self.completed_at = time.time()

    def record_tool_call(self, metrics: ToolCallMetrics) -> None:
        """Record a tool call.

        Args:
            metrics: Tool call metrics to record
        """
        self.tool_calls.append(metrics)
        if metrics.success and metrics.end_time:
            # Update token counts if available
            self.total_input_tokens += metrics.input_tokens
            self.total_output_tokens += metrics.output_tokens

    def record_llm_call(self, metrics: LLMCallMetrics) -> None:
        """Record an LLM call.

        Args:
            metrics: LLM call metrics to record
        """
        self.llm_calls.append(metrics)
        if metrics.success and metrics.end_time:
            # Update token counts
            self.total_input_tokens += metrics.input_tokens
            self.total_output_tokens += metrics.output_tokens
            self.total_cache_read_tokens += metrics.cache_read_tokens
            self.total_cache_write_tokens += metrics.cache_write_tokens
            self.total_reasoning_tokens += metrics.reasoning_tokens

    def record_state_transition(self, new_state: str) -> None:
        """Record a state transition.

        Args:
            new_state: New state name
        """
        self.state_transitions += 1
        self.current_state = new_state

    def record_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an error.

        Args:
            error_type: Type of error (e.g., "ToolError", "ProviderError")
            error_message: Error message
            context: Optional additional context
        """
        self.errors.append(
            {
                "type": error_type,
                "message": error_message,
                "timestamp": time.time(),
                "context": context or {},
            }
        )

    def start_timer(self, name: str) -> None:
        """Start a named timer.

        Args:
            name: Timer name
        """
        self._timers[name] = time.time()

    def stop_timer(self, name: str) -> Optional[float]:
        """Stop a named timer and return duration.

        Args:
            name: Timer name

        Returns:
            Duration in seconds, or None if timer wasn't started
        """
        if name not in self._timers:
            return None
        duration = time.time() - self._timers[name]
        self._timers[name] = duration
        return duration

    def get_timer(self, name: str) -> Optional[float]:
        """Get timer value.

        Args:
            name: Timer name

        Returns:
            Duration in seconds if timer was stopped, None otherwise
        """
        value = self._timers.get(name)
        if value is None:
            return None
        # If still running (value is start time), return current duration
        if value < self.created_at:  # Started before created_at means it's a timestamp
            return None
        return value

    def get_tools_by_name(self, tool_name: str) -> List[ToolCallMetrics]:
        """Get all calls for a specific tool.

        Args:
            tool_name: Tool name to filter by

        Returns:
            List of tool call metrics for the specified tool
        """
        return [tc for tc in self.tool_calls if tc.tool_name == tool_name]

    def get_llm_calls_by_model(self, model: str) -> List[LLMCallMetrics]:
        """Get all calls for a specific model.

        Args:
            model: Model name to filter by

        Returns:
            List of LLM call metrics for the specified model
        """
        return [llm for llm in self.llm_calls if llm.model == model]

    def summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics.

        Returns:
            Dictionary with key metrics
        """
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "duration": self.duration,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "tokens": {
                "total": self.total_tokens,
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "cached": self.total_cached_tokens,
                "cache_read": self.total_cache_read_tokens,
                "cache_write": self.total_cache_write_tokens,
                "reasoning": self.total_reasoning_tokens,
            },
            "tool_calls": {
                "total": self.tool_call_count,
                "successful": self.successful_tool_calls,
                "failed": self.failed_tool_calls,
                "success_rate": self.tool_success_rate,
                "total_duration": self.total_tool_duration,
                "average_duration": self.average_tool_duration,
            },
            "llm_calls": {
                "total": self.llm_call_count,
                "successful": self.successful_llm_calls,
                "failed": self.failed_llm_calls,
                "success_rate": self.llm_success_rate,
                "total_duration": self.total_llm_duration,
                "average_duration": self.average_llm_duration,
            },
            "state": {
                "transitions": self.state_transitions,
                "current": self.current_state,
            },
            "errors": {
                "count": len(self.errors),
                "latest": self.errors[-1] if self.errors else None,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to full dictionary representation.

        Returns:
            Complete dictionary with all metrics
        """
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "state_transitions": self.state_transitions,
            "current_state": self.current_state,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "llm_calls": [llm.to_dict() for llm in self.llm_calls],
            "errors": self.errors,
            "summary": self.summary(),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json_dumps(self.to_dict(), indent=indent, default=str)


# =====================================================================
# Metrics Collector
# =====================================================================


class MetricsCollector:
    """Collects and aggregates metrics with sampling support.

    Provides:
    - Metric aggregation across multiple sources
    - Configurable sampling rates for high-frequency metrics
    - Time-windowed aggregation
    - Memory-efficient storage with downsampling

    Example:
        >>> collector = MetricsCollector(sample_rate=0.1)  # 10% sampling
        >>> collector.record_metric(CounterMetric(...))
        >>> snapshot = collector.get_snapshot()
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        max_samples: int = 10000,
        aggregation_window: float = 60.0,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            sample_rate: Sampling rate (0.0 to 1.0). 1.0 = collect all, 0.1 = 10%
            max_samples: Maximum number of samples to keep in memory
            aggregation_window: Time window for aggregation (seconds)
        """
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")

        self._sample_rate = sample_rate
        self._max_samples = max_samples
        self._aggregation_window = aggregation_window

        # Metric storage
        self._metrics: List[Metric] = []
        self._aggregated: Dict[str, Metric] = {}

        # Sampling counters
        self._sampled_count = 0
        self._total_count = 0

    def record(self, metric: Metric) -> bool:
        """Record a metric (subject to sampling).

        Args:
            metric: Metric to record

        Returns:
            True if metric was recorded, False if sampled out
        """
        self._total_count += 1

        # Apply sampling
        import random

        if random.random() > self._sample_rate:
            return False

        self._sampled_count += 1

        # Store metric
        self._metrics.append(metric)

        # Enforce max samples (FIFO eviction)
        if len(self._metrics) > self._max_samples:
            self._metrics.pop(0)

        return True

    def get_metrics(
        self,
        name: Optional[str] = None,
        metric_type: Optional[MetricType] = None,
        since: Optional[float] = None,
    ) -> List[Metric]:
        """Get collected metrics with optional filtering.

        Args:
            name: Filter by metric name
            metric_type: Filter by metric type
            since: Only include metrics after this timestamp

        Returns:
            List of matching metrics
        """
        filtered = self._metrics

        if name is not None:
            filtered = [m for m in filtered if m.name == name]

        if metric_type is not None:
            filtered = [m for m in filtered if m.metric_type == metric_type]

        if since is not None:
            filtered = [m for m in filtered if m.timestamp >= since]

        return filtered

    def aggregate_by_name(self, name: str) -> Optional[Metric]:
        """Aggregate all metrics with the given name.

        Args:
            name: Metric name to aggregate

        Returns:
            Aggregated metric or None if not found
        """
        metrics = self.get_metrics(name=name)

        if not metrics:
            return None

        # Aggregate based on first metric's type
        first = metrics[0]

        if isinstance(first, CounterMetric):
            total = sum(m.value for m in metrics)  # type: ignore
            return first.with_value(total)  # type: ignore
        elif isinstance(first, GaugeMetric):
            # Average for gauges
            values = [m.value for m in metrics]  # type: ignore
            avg = sum(values) / len(values)
            return first.set(avg)  # type: ignore
        else:
            # For other types, return the most recent
            return metrics[-1]

    def get_sampling_stats(self) -> Dict[str, Any]:
        """Get sampling statistics.

        Returns:
            Dictionary with sampling stats
        """
        actual_rate = self._sampled_count / self._total_count if self._total_count > 0 else 0.0

        return {
            "configured_rate": self._sample_rate,
            "actual_rate": actual_rate,
            "total_count": self._total_count,
            "sampled_count": self._sampled_count,
            "stored_count": len(self._metrics),
            "max_samples": self._max_samples,
        }

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()
        self._aggregated.clear()
        self._sampled_count = 0
        self._total_count = 0

    def record_metric(self, name: str, value: float, **tags: str) -> None:
        """Record a single metric observation (MetricsCollectorProtocol adapter).

        Adapts the ``(name, value, **tags)`` protocol surface onto this
        collector's :meth:`record`, which expects a :class:`Metric`. The value is
        wrapped as a :class:`GaugeMetric` and tags become metric labels.

        Args:
            name: Metric name.
            value: Numeric value to record.
            **tags: Optional string-valued dimensional tags, recorded as labels.
        """
        labels = tuple(MetricLabel(key=k, value=v) for k, v in sorted(tags.items()))
        self.record(
            GaugeMetric(
                name=name,
                description="",
                metric_type=MetricType.GAUGE,
                labels=labels,
                value=float(value),
            )
        )

    def get_snapshot(self) -> MetricsSnapshot:
        """Get a snapshot of current metrics.

        Returns:
            MetricsSnapshot with all collected metrics
        """
        return MetricsSnapshot(
            source_id="metrics_collector",
            source_type="collector",
            metrics=tuple(self._metrics),
        )


class MetricsExporter:
    """Export metrics to various formats.

    Supports:
    - JSON export
    - CSV export
    - Prometheus text format
    """

    @staticmethod
    def to_json(metrics: List[Metric], indent: Optional[int] = None) -> str:
        """Export metrics to JSON string.

        Args:
            metrics: List of metrics to export
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json_dumps(
            [_metric_to_dict(m) for m in metrics],
            indent=indent,
            default=str,
        )

    @staticmethod
    def to_json_file(
        metrics: List[Metric],
        path: str | Path,
        indent: Optional[int] = 2,
    ) -> None:
        """Export metrics to JSON file.

        Args:
            metrics: List of metrics to export
            path: Output file path
            indent: JSON indentation level
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(MetricsExporter.to_json(metrics, indent=indent))

    @staticmethod
    def to_csv(metrics: List[Metric]) -> str:
        """Export metrics to CSV string.

        Args:
            metrics: List of metrics to export

        Returns:
            CSV string
        """
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "name",
                "type",
                "timestamp",
                "labels",
                "value",
            ]
        )

        # Rows
        for metric in metrics:
            labels_str = ",".join(f"{label.key}={label.value}" for label in metric.labels)
            value_str = _format_metric_value(metric)

            writer.writerow(
                [
                    metric.name,
                    metric.metric_type.value,
                    metric.timestamp,
                    labels_str,
                    value_str,
                ]
            )

        return output.getvalue()

    @staticmethod
    def to_csv_file(metrics: List[Metric], path: str | Path) -> None:
        """Export metrics to CSV file.

        Args:
            metrics: List of metrics to export
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(MetricsExporter.to_csv(metrics))

    @staticmethod
    def to_prometheus(metrics: List[Metric]) -> str:
        """Export metrics to Prometheus text format.

        Args:
            metrics: List of metrics to export

        Returns:
            Prometheus text format string
        """
        lines = []

        for metric in metrics:
            # Create metric name and labels
            labels_str = (
                "{" + ",".join(f'{label.key}="{label.value}"' for label in metric.labels) + "}"
            )
            if len(metric.labels) == 0:
                labels_str = ""

            metric_line = f"{metric.name}{labels_str}"

            # Add value based on type
            if isinstance(metric, CounterMetric):
                lines.append(f"# TYPE {metric.name} counter")
                lines.append(f"{metric_line} {metric.value}")
            elif isinstance(metric, GaugeMetric):
                lines.append(f"# TYPE {metric.name} gauge")
                lines.append(f"{metric_line} {metric.value}")
            elif isinstance(metric, HistogramMetric):
                lines.append(f"# TYPE {metric.name} histogram")
                lines.append(f"{metric_line}_count {metric.count}")
                lines.append(f"{metric_line}_sum {metric.sum}")
                for bucket in metric.buckets:
                    le_label = '{le="%.1f"}' % bucket.upper_bound
                    lines.append(f"{metric.name}_bucket{le_label} {bucket.count}")
            elif isinstance(metric, SummaryMetric):
                lines.append(f"# TYPE {metric.name} summary")
                lines.append(f"{metric_line}_count {metric.count}")
                lines.append(f"{metric_line}_sum {metric.sum}")

        return "\n".join(lines)


def _metric_to_dict(metric: Metric) -> Dict[str, Any]:
    """Convert metric to dictionary.

    Args:
        metric: Metric to convert

    Returns:
        Dictionary representation
    """
    return {
        "name": metric.name,
        "description": metric.description,
        "type": metric.metric_type.value,
        "labels": {label.key: label.value for label in metric.labels},
        "timestamp": metric.timestamp,
        "datetime": datetime.fromtimestamp(metric.timestamp).isoformat(),
        "value": _extract_metric_value(metric),
    }


def _format_metric_value(metric: Metric) -> str:
    """Format metric value for CSV.

    Args:
        metric: Metric to format

    Returns:
        Formatted value string
    """
    if isinstance(metric, CounterMetric):
        return str(metric.value)
    elif isinstance(metric, GaugeMetric):
        return str(metric.value)
    elif isinstance(metric, HistogramMetric):
        return f"count={metric.count},sum={metric.sum},avg={metric.average:.2f}"
    elif isinstance(metric, SummaryMetric):
        return f"count={metric.count},avg={metric.average:.2f}"
    else:
        return ""
