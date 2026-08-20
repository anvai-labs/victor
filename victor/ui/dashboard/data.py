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

"""Data provider for the metrics dashboard.

Aggregates metrics from various sources to provide unified data
for dashboard visualization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from victor.framework.observability import (
    AgentMetrics,
    CounterMetric,
    GaugeMetric,
    Metric,
    MetricNames,
    MetricType,
)


@dataclass
class DashboardMetric:
    """A dashboard-ready metric.

    Attributes:
        name: Metric name
        value: Current value
        previous_value: Previous value for trend calculation
        change: Percent change from previous
        trend: Trend direction ("up", "down", "neutral")
        timestamp: When the metric was recorded
        history: Recent historical values
    """

    name: str
    value: float
    previous_value: Optional[float] = None
    change: Optional[float] = None
    trend: str = "neutral"
    timestamp: float = field(default_factory=time.time)
    history: List[float] = field(default_factory=list)

    def add_history(self, value: float, max_history: int = 100) -> None:
        """Add a value to history.

        Args:
            value: Value to add
            max_history: Maximum history length
        """
        self.history.append(value)
        if len(self.history) > max_history:
            self.history.pop(0)

    def calculate_trend(self) -> None:
        """Calculate trend based on history."""
        if len(self.history) < 2:
            self.trend = "neutral"
            return

        recent = self.history[-5:] if len(self.history) >= 5 else self.history
        if len(recent) < 2:
            self.trend = "neutral"
            return

        # Simple linear regression
        x = list(range(len(recent)))
        y = recent
        n = len(recent)

        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        if slope > 0.01:
            self.trend = "up"
        elif slope < -0.01:
            self.trend = "down"
        else:
            self.trend = "neutral"


@dataclass
class DashboardData:
    """Complete dashboard data snapshot.

    Attributes:
        timestamp: When the data was collected
        session_id: Optional session identifier
        agent_id: Optional agent identifier
        metrics: Dashboard metrics by category
        token_usage: Token usage statistics
        tool_usage: Tool usage statistics
        llm_usage: LLM call statistics
        performance: Performance metrics
        errors: Recent errors
    """

    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    agent_id: Optional[str] = None

    # Metrics by category
    token_metrics: Dict[str, DashboardMetric] = field(default_factory=dict)
    tool_metrics: Dict[str, DashboardMetric] = field(default_factory=dict)
    llm_metrics: Dict[str, DashboardMetric] = field(default_factory=dict)
    performance_metrics: Dict[str, DashboardMetric] = field(default_factory=dict)

    # Summary data
    token_usage: Dict[str, Any] = field(default_factory=dict)
    tool_usage: Dict[str, Any] = field(default_factory=dict)
    llm_usage: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)

    # Errors
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def datetime(self) -> datetime:
        """Get timestamp as datetime."""
        return datetime.fromtimestamp(self.timestamp)


class DashboardDataProvider:
    """Provides data for the metrics dashboard.

    Aggregates metrics from:
    - AgentMetrics (from active agent sessions)
    - ObservabilityManager (system-wide metrics)
    - SessionCostTracker (cost tracking)

    Example:
        provider = DashboardDataProvider()

        # Get current dashboard data
        data = provider.get_dashboard_data()

        # Get historical data
        history = provider.get_history(hours=1)
    """

    def __init__(self) -> None:
        """Initialize the dashboard data provider."""
        self._agent_metrics: Dict[str, AgentMetrics] = {}
        self._history: List[DashboardData] = []
        self._max_history_hours: float = 24.0

    def register_agent(self, agent_id: str, metrics: AgentMetrics) -> None:
        """Register an agent's metrics.

        Args:
            agent_id: Agent identifier
            metrics: Agent metrics instance
        """
        self._agent_metrics[agent_id] = metrics

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent's metrics.

        Args:
            agent_id: Agent identifier
        """
        self._agent_metrics.pop(agent_id, None)

    def get_dashboard_data(self) -> DashboardData:
        """Get current dashboard data.

        Returns:
            DashboardData with current metrics
        """
        data = DashboardData()

        # Aggregate from all registered agents
        for agent_id, metrics in self._agent_metrics.items():
            data.agent_id = agent_id
            data.session_id = metrics.session_id

            # Token metrics
            self._aggregate_token_metrics(metrics, data)

            # Tool metrics
            self._aggregate_tool_metrics(metrics, data)

            # LLM metrics
            self._aggregate_llm_metrics(metrics, data)

            # Performance metrics
            self._aggregate_performance_metrics(metrics, data)

            # Errors
            data.errors.extend(metrics.errors)

        # Add to history
        self._history.append(data)
        self._cleanup_history()

        return data

    def _aggregate_token_metrics(self, metrics: AgentMetrics, data: DashboardData) -> None:
        """Aggregate token metrics.

        Args:
            metrics: Source metrics
            data: Dashboard data to update
        """
        total = DashboardMetric(
            name="total_tokens",
            value=float(metrics.total_tokens),
        )
        total.add_history(float(metrics.total_tokens))
        total.calculate_trend()
        data.token_metrics["total"] = total

        input_tokens = DashboardMetric(
            name="input_tokens",
            value=float(metrics.total_input_tokens),
        )
        input_tokens.add_history(float(metrics.total_input_tokens))
        input_tokens.calculate_trend()
        data.token_metrics["input"] = input_tokens

        output_tokens = DashboardMetric(
            name="output_tokens",
            value=float(metrics.total_output_tokens),
        )
        output_tokens.add_history(float(metrics.total_output_tokens))
        output_tokens.calculate_trend()
        data.token_metrics["output"] = output_tokens

        # Summary
        data.token_usage = {
            "total": metrics.total_tokens,
            "input": metrics.total_input_tokens,
            "output": metrics.total_output_tokens,
            "cached": metrics.total_cached_tokens,
            "cache_read": metrics.total_cache_read_tokens,
            "cache_write": metrics.total_cache_write_tokens,
            "reasoning": metrics.total_reasoning_tokens,
        }

    def _aggregate_tool_metrics(self, metrics: AgentMetrics, data: DashboardData) -> None:
        """Aggregate tool metrics.

        Args:
            metrics: Source metrics
            data: Dashboard data to update
        """
        total_calls = DashboardMetric(
            name="total_calls",
            value=float(metrics.tool_call_count),
        )
        total_calls.add_history(float(metrics.tool_call_count))
        total_calls.calculate_trend()
        data.tool_metrics["total_calls"] = total_calls

        success_rate = DashboardMetric(
            name="success_rate",
            value=float(metrics.tool_success_rate),
        )
        success_rate.add_history(float(metrics.tool_success_rate))
        success_rate.calculate_trend()
        data.tool_metrics["success_rate"] = success_rate

        avg_duration = DashboardMetric(
            name="avg_duration",
            value=float(metrics.average_tool_duration),
        )
        avg_duration.add_history(float(metrics.average_tool_duration))
        avg_duration.calculate_trend()
        data.tool_metrics["avg_duration"] = avg_duration

        # Per-tool stats
        per_tool: Dict[str, Dict[str, Any]] = {}
        for tool_call in metrics.tool_calls:
            tool_name = tool_call.tool_name
            if tool_name not in per_tool:
                per_tool[tool_name] = {
                    "calls": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_duration": 0.0,
                }

            stats = per_tool[tool_name]
            stats["calls"] += 1
            if tool_call.success:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
            if tool_call.duration:
                stats["total_duration"] += tool_call.duration

        # Calculate averages
        for stats in per_tool.values():
            if stats["calls"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["calls"]
                stats["success_rate"] = stats["successful"] / stats["calls"]

        data.tool_usage = {
            "total": metrics.tool_call_count,
            "successful": metrics.successful_tool_calls,
            "failed": metrics.failed_tool_calls,
            "success_rate": metrics.tool_success_rate,
            "total_duration": metrics.total_tool_duration,
            "average_duration": metrics.average_tool_duration,
            "by_tool": per_tool,
        }

    def _aggregate_llm_metrics(self, metrics: AgentMetrics, data: DashboardData) -> None:
        """Aggregate LLM metrics.

        Args:
            metrics: Source metrics
            data: Dashboard data to update
        """
        total_calls = DashboardMetric(
            name="total_calls",
            value=float(metrics.llm_call_count),
        )
        total_calls.add_history(float(metrics.llm_call_count))
        total_calls.calculate_trend()
        data.llm_metrics["total_calls"] = total_calls

        success_rate = DashboardMetric(
            name="success_rate",
            value=float(metrics.llm_success_rate),
        )
        success_rate.add_history(float(metrics.llm_success_rate))
        success_rate.calculate_trend()
        data.llm_metrics["success_rate"] = success_rate

        avg_duration = DashboardMetric(
            name="avg_duration",
            value=float(metrics.average_llm_duration),
        )
        avg_duration.add_history(float(metrics.average_llm_duration))
        avg_duration.calculate_trend()
        data.llm_metrics["avg_duration"] = avg_duration

        # Per-model stats
        per_model: Dict[str, Dict[str, Any]] = {}
        for llm_call in metrics.llm_calls:
            model = llm_call.model
            if model not in per_model:
                per_model[model] = {
                    "calls": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_duration": 0.0,
                    "total_tokens": 0,
                }

            stats = per_model[model]
            stats["calls"] += 1
            if llm_call.success:
                stats["successful"] += 1
            else:
                stats["failed"] += 1
            if llm_call.duration:
                stats["total_duration"] += llm_call.duration
            stats["total_tokens"] += llm_call.total_tokens

        # Calculate averages
        for stats in per_model.values():
            if stats["calls"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["calls"]
                stats["avg_tokens"] = stats["total_tokens"] / stats["calls"]
                stats["success_rate"] = stats["successful"] / stats["calls"]

        data.llm_usage = {
            "total": metrics.llm_call_count,
            "successful": metrics.successful_llm_calls,
            "failed": metrics.failed_llm_calls,
            "success_rate": metrics.llm_success_rate,
            "total_duration": metrics.total_llm_duration,
            "average_duration": metrics.average_llm_duration,
            "by_model": per_model,
        }

    def _aggregate_performance_metrics(self, metrics: AgentMetrics, data: DashboardData) -> None:
        """Aggregate performance metrics.

        Args:
            metrics: Source metrics
            data: Dashboard data to update
        """
        duration = DashboardMetric(
            name="duration",
            value=float(metrics.duration or 0),
        )
        if metrics.duration:
            duration.add_history(float(metrics.duration))
        duration.calculate_trend()
        data.performance_metrics["duration"] = duration

        state_transitions = DashboardMetric(
            name="state_transitions",
            value=float(metrics.state_transitions),
        )
        state_transitions.add_history(float(metrics.state_transitions))
        state_transitions.calculate_trend()
        data.performance_metrics["state_transitions"] = state_transitions

        data.performance = {
            "duration": metrics.duration,
            "state_transitions": metrics.state_transitions,
            "current_state": metrics.current_state,
            "tool_call_count": metrics.tool_call_count,
            "llm_call_count": metrics.llm_call_count,
        }

    def get_history(self, hours: float = 1.0) -> List[DashboardData]:
        """Get historical dashboard data.

        Args:
            hours: Number of hours of history to return

        Returns:
            List of historical dashboard data
        """
        cutoff = time.time() - (hours * 3600)
        return [d for d in self._history if d.timestamp >= cutoff]

    def _cleanup_history(self) -> None:
        """Remove old history entries."""
        cutoff = time.time() - (self._max_history_hours * 3600)
        self._history = [d for d in self._history if d.timestamp >= cutoff]


# Global data provider instance
_default_provider: Optional[DashboardDataProvider] = None


def get_dashboard_provider() -> DashboardDataProvider:
    """Get the global dashboard data provider.

    Returns:
        DashboardDataProvider instance
    """
    global _default_provider
    if _default_provider is None:
        _default_provider = DashboardDataProvider()
    return _default_provider
