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

"""Observability metrics for team and workflow usage.

This module provides Prometheus-compatible metrics for tracking:
- Team spawning and execution
- Workflow execution and outcomes
- Coordination decisions and their results

Metrics follow Prometheus naming conventions:
- Counter: _total suffix
- Histogram: _seconds or _bytes suffix
- Gauge: no suffix

Usage:
    from victor.observability.team_metrics import (
        record_team_spawned,
        record_team_completed,
        record_workflow_executed,
    )

    # Record team spawning
    record_team_spawned(
        team_name="feature_team",
        vertical="coding",
        task_type="feature",
        complexity="high",
    )

    # Record team completion
    record_team_completed(
        team_name="feature_team",
        success=True,
        duration_seconds=45.2,
        tool_calls=23,
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from victor.core.constants import DEFAULT_VERTICAL

logger = logging.getLogger(__name__)


# =============================================================================
# Metric Storage (in-memory for non-Prometheus environments)
# =============================================================================


@dataclass
class MetricValue:
    """Single metric value with labels."""

    value: float
    labels: Dict[str, str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class MetricSeries:
    """Time series for a metric."""

    name: str
    description: str
    metric_type: str  # "counter", "gauge", "histogram"
    values: List[MetricValue] = field(default_factory=list)

    def add(self, value: float, labels: Dict[str, str]) -> None:
        """Add a value to the series."""
        self.values.append(MetricValue(value=value, labels=labels))

    def get_total(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get total value, optionally filtered by labels."""
        if labels is None:
            return sum(v.value for v in self.values)

        return sum(
            v.value for v in self.values if all(v.labels.get(k) == labels.get(k) for k in labels)
        )


class MetricsRegistry:
    """Registry for all metrics.

    Provides in-memory storage for metrics when Prometheus client
    is not available, and exports to Prometheus format.
    """

    def __init__(self):
        """Initialize registry."""
        self._metrics: Dict[str, MetricSeries] = {}
        self._prometheus_available = self._check_prometheus()

        # Initialize Prometheus metrics if available
        if self._prometheus_available:
            self._init_prometheus_metrics()

    def _check_prometheus(self) -> bool:
        """Check if prometheus_client is available."""
        try:
            import prometheus_client  # noqa

            return True
        except ImportError:
            return False

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metric objects."""
        from prometheus_client import Counter, Histogram, Gauge

        # Team metrics
        self._team_spawned_counter = Counter(
            "victor_team_spawned_total",
            "Total teams spawned",
            ["team_name", "vertical", "task_type", "complexity", "trigger"],
        )

        self._team_completed_counter = Counter(
            "victor_team_completed_total",
            "Total teams completed",
            ["team_name", "success", "formation"],
        )

        self._team_duration_histogram = Histogram(
            "victor_team_duration_seconds",
            "Team execution duration in seconds",
            ["team_name", "formation"],
            buckets=(1, 5, 10, 30, 60, 120, 300, 600),
        )

        self._team_tool_calls_histogram = Histogram(
            "victor_team_tool_calls",
            "Number of tool calls per team execution",
            ["team_name", "formation"],
            buckets=(5, 10, 20, 50, 100, 200),
        )

        self._team_members_gauge = Gauge(
            "victor_team_members",
            "Number of members in a team",
            ["team_name"],
        )

        # Workflow metrics
        self._workflow_executed_counter = Counter(
            "victor_workflow_executed_total",
            "Total workflows executed",
            ["workflow_name", "mode", "trigger", "vertical"],
        )

        self._workflow_completed_counter = Counter(
            "victor_workflow_completed_total",
            "Total workflows completed",
            ["workflow_name", "success"],
        )

        self._workflow_duration_histogram = Histogram(
            "victor_workflow_duration_seconds",
            "Workflow execution duration in seconds",
            ["workflow_name"],
            buckets=(1, 5, 10, 30, 60, 120, 300, 600),
        )

        # Coordination metrics
        self._coordination_suggestion_counter = Counter(
            "victor_coordination_suggestion_total",
            "Total coordination suggestions made",
            ["mode", "task_type", "complexity", "action"],
        )

        self._coordination_accepted_counter = Counter(
            "victor_coordination_accepted_total",
            "Total coordination suggestions accepted",
            ["mode", "task_type", "suggestion_type"],
        )

    def register(
        self,
        name: str,
        description: str,
        metric_type: str,
    ) -> MetricSeries:
        """Register a metric series.

        Args:
            name: Metric name
            description: Human-readable description
            metric_type: Type (counter, gauge, histogram)

        Returns:
            MetricSeries instance
        """
        if name not in self._metrics:
            self._metrics[name] = MetricSeries(
                name=name,
                description=description,
                metric_type=metric_type,
            )
        return self._metrics[name]

    def record(
        self,
        name: str,
        value: float,
        labels: Dict[str, str],
    ) -> None:
        """Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            labels: Metric labels
        """
        if name in self._metrics:
            self._metrics[name].add(value, labels)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus exposition format string
        """
        lines = []
        for name, series in self._metrics.items():
            lines.append(f"# HELP {name} {series.description}")
            lines.append(f"# TYPE {name} {series.metric_type}")

            for value in series.values:
                label_str = ",".join(f'{k}="{v}"' for k, v in value.labels.items())
                lines.append(f"{name}{{{label_str}}} {value.value}")

        return "\n".join(lines)

    def export_json(self) -> Dict[str, Any]:
        """Export metrics as JSON.

        Returns:
            Dict with all metrics
        """
        result = {}
        for name, series in self._metrics.items():
            result[name] = {
                "description": series.description,
                "type": series.metric_type,
                "values": [
                    {
                        "value": v.value,
                        "labels": v.labels,
                        "timestamp": v.timestamp,
                    }
                    for v in series.values
                ],
            }
        return result


# Global registry instance
_registry: Optional[MetricsRegistry] = None


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry.

    Returns:
        MetricsRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry


# =============================================================================
# Team Metrics
# =============================================================================


def record_team_spawned(
    team_name: str,
    vertical: str = DEFAULT_VERTICAL,
    task_type: str = "unknown",
    complexity: str = "medium",
    trigger: str = "auto",
) -> None:
    """Record a team being spawned.

    Args:
        team_name: Name of the spawned team
        vertical: Vertical that spawned the team
        task_type: Type of task the team is handling
        complexity: Task complexity level
        trigger: What triggered the spawn (auto, manual, suggestion)
    """
    registry = get_metrics_registry()
    labels = {
        "team_name": team_name,
        "vertical": vertical,
        "task_type": task_type,
        "complexity": complexity,
        "trigger": trigger,
    }

    # Record in local registry
    series = registry.register(
        "victor_team_spawned_total",
        "Total teams spawned",
        "counter",
    )
    series.add(1, labels)

    # Record in Prometheus if available
    if registry._prometheus_available:
        registry._team_spawned_counter.labels(**labels).inc()

    logger.debug(
        f"Team spawned: {team_name} (vertical={vertical}, "
        f"task={task_type}, complexity={complexity}, trigger={trigger})"
    )


def record_team_completed(
    team_name: str,
    success: bool,
    duration_seconds: float,
    tool_calls: int = 0,
    formation: str = "sequential",
    member_count: int = 0,
) -> None:
    """Record a team completing execution.

    Args:
        team_name: Name of the completed team
        success: Whether execution succeeded
        duration_seconds: Execution duration
        tool_calls: Total tool calls made
        formation: Team formation used
        member_count: Number of team members
    """
    registry = get_metrics_registry()

    # Completion counter
    completion_labels = {
        "team_name": team_name,
        "success": str(success).lower(),
        "formation": formation,
    }

    series = registry.register(
        "victor_team_completed_total",
        "Total teams completed",
        "counter",
    )
    series.add(1, completion_labels)

    # Duration histogram
    duration_labels = {
        "team_name": team_name,
        "formation": formation,
    }

    duration_series = registry.register(
        "victor_team_duration_seconds",
        "Team execution duration",
        "histogram",
    )
    duration_series.add(duration_seconds, duration_labels)

    # Tool calls histogram
    if tool_calls > 0:
        tool_series = registry.register(
            "victor_team_tool_calls",
            "Tool calls per team",
            "histogram",
        )
        tool_series.add(tool_calls, duration_labels)

    # Prometheus recording
    if registry._prometheus_available:
        registry._team_completed_counter.labels(**completion_labels).inc()
        registry._team_duration_histogram.labels(**duration_labels).observe(duration_seconds)
        if tool_calls > 0:
            registry._team_tool_calls_histogram.labels(**duration_labels).observe(tool_calls)
        if member_count > 0:
            registry._team_members_gauge.labels(team_name=team_name).set(member_count)

    logger.debug(
        f"Team completed: {team_name} (success={success}, "
        f"duration={duration_seconds:.2f}s, tools={tool_calls})"
    )


# =============================================================================
# Workflow Metrics
# =============================================================================


def record_workflow_executed(
    workflow_name: str,
    mode: str = "build",
    trigger: str = "manual",
    vertical: str = DEFAULT_VERTICAL,
) -> None:
    """Record a workflow being executed.

    Args:
        workflow_name: Name of the executed workflow
        mode: Agent mode during execution
        trigger: What triggered the workflow
        vertical: Vertical context
    """
    registry = get_metrics_registry()
    labels = {
        "workflow_name": workflow_name,
        "mode": mode,
        "trigger": trigger,
        "vertical": vertical,
    }

    series = registry.register(
        "victor_workflow_executed_total",
        "Total workflows executed",
        "counter",
    )
    series.add(1, labels)

    if registry._prometheus_available:
        registry._workflow_executed_counter.labels(**labels).inc()

    logger.debug(f"Workflow executed: {workflow_name} (mode={mode}, trigger={trigger})")


def record_workflow_completed(
    workflow_name: str,
    success: bool,
    duration_seconds: float,
) -> None:
    """Record a workflow completing.

    Args:
        workflow_name: Name of the completed workflow
        success: Whether execution succeeded
        duration_seconds: Execution duration
    """
    registry = get_metrics_registry()

    completion_labels = {
        "workflow_name": workflow_name,
        "success": str(success).lower(),
    }

    series = registry.register(
        "victor_workflow_completed_total",
        "Total workflows completed",
        "counter",
    )
    series.add(1, completion_labels)

    duration_labels = {"workflow_name": workflow_name}
    duration_series = registry.register(
        "victor_workflow_duration_seconds",
        "Workflow execution duration",
        "histogram",
    )
    duration_series.add(duration_seconds, duration_labels)

    if registry._prometheus_available:
        registry._workflow_completed_counter.labels(**completion_labels).inc()
        registry._workflow_duration_histogram.labels(**duration_labels).observe(duration_seconds)

    logger.debug(
        f"Workflow completed: {workflow_name} (success={success}, "
        f"duration={duration_seconds:.2f}s)"
    )


# =============================================================================
# Coordination Metrics
# =============================================================================


def record_coordination_suggestion(
    mode: str,
    task_type: str,
    complexity: str,
    action: str,
    team_suggested: Optional[str] = None,
    workflow_suggested: Optional[str] = None,
) -> None:
    """Record a coordination suggestion being made.

    Args:
        mode: Current agent mode
        task_type: Classified task type
        complexity: Task complexity level
        action: Suggested action (none, suggest, auto_spawn)
        team_suggested: Name of suggested team (if any)
        workflow_suggested: Name of suggested workflow (if any)
    """
    registry = get_metrics_registry()
    labels = {
        "mode": mode,
        "task_type": task_type,
        "complexity": complexity,
        "action": action,
    }

    series = registry.register(
        "victor_coordination_suggestion_total",
        "Total coordination suggestions",
        "counter",
    )
    series.add(1, labels)

    if registry._prometheus_available:
        registry._coordination_suggestion_counter.labels(**labels).inc()

    logger.debug(
        f"Coordination suggestion: mode={mode}, task={task_type}, "
        f"complexity={complexity}, action={action}, "
        f"team={team_suggested}, workflow={workflow_suggested}"
    )


def record_coordination_accepted(
    mode: str,
    task_type: str,
    suggestion_type: str,  # "team" or "workflow"
    accepted: bool = True,
) -> None:
    """Record whether a coordination suggestion was accepted.

    Args:
        mode: Agent mode
        task_type: Task type
        suggestion_type: Type of suggestion (team or workflow)
        accepted: Whether user accepted the suggestion
    """
    registry = get_metrics_registry()
    labels = {
        "mode": mode,
        "task_type": task_type,
        "suggestion_type": suggestion_type,
    }

    series = registry.register(
        "victor_coordination_accepted_total",
        "Total coordination suggestions accepted",
        "counter",
    )
    series.add(1 if accepted else 0, labels)

    if registry._prometheus_available:
        registry._coordination_accepted_counter.labels(**labels).inc()


__all__ = [
    # Registry
    "MetricsRegistry",
    "get_metrics_registry",
    # Team metrics
    "record_team_spawned",
    "record_team_completed",
    # Workflow metrics
    "record_workflow_executed",
    "record_workflow_completed",
    # Coordination metrics
    "record_coordination_suggestion",
    "record_coordination_accepted",
]
