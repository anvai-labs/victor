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

"""Metrics collection for A/B testing.

This module provides metrics collection and aggregation for A/B experiments.
"""

import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from victor.core.events import MessagingEvent, get_observability_bus
from victor.experiments.ab_testing.paths import get_default_ab_test_db_path

# ``stats`` is a lazy proxy (and ``_require_scipy`` its guard) defined in
# statistics.py — reusing it keeps scipy out of the CLI cold-start path.
from victor.experiments.ab_testing.statistics import _require_scipy, stats
from victor.experiments.ab_testing.models import (
    AggregatedMetrics,
    ExecutionMetrics,
)


class MetricsCollector:
    """Collects and aggregates metrics for A/B experiments.

    This class subscribes to the ObservabilityBus and collects metrics
    for experiment executions, aggregating them per variant.

    Usage:
        collector = MetricsCollector(storage_path="/path/to/experiments.db")
        await collector.start()

        # Metrics are automatically collected from ObservabilityBus

        # Get aggregated metrics
        metrics = await collector.get_variant_metrics(experiment_id, variant_id)
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        buffer_size: int = 100,
    ):
        """Initialize metrics collector.

        Args:
            storage_path: Path to SQLite database
            buffer_size: Number of metrics to buffer before flushing
        """
        if storage_path is None:
            storage_path = str(get_default_ab_test_db_path())

        self.storage_path = Path(storage_path).expanduser()
        self.buffer_size = buffer_size

        # Metrics buffer
        self._metrics_buffer: List[ExecutionMetrics] = []

        # In-memory aggregation cache
        self._aggregation_cache: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Event subscription handles
        self._subscription_handles: List = []

        # Ad-hoc metrics recorded via the MetricsCollectorProtocol adapter
        # (``record_metric``). The native surface here is async/DB-backed
        # experiment aggregation, so generic name/value observations are backed
        # by a small in-memory dict.
        self._adhoc_metrics: Dict[str, float] = {}

    # =========================================================================
    # MetricsCollectorProtocol adapter (FEP-0014)
    # =========================================================================

    def record_metric(self, name: str, value: float, **tags: str) -> None:
        """Record a single ad-hoc metric observation (protocol adapter).

        The native surface here collects experiment execution metrics from the
        ObservabilityBus into SQLite; this stores generic name/value
        observations so the collector satisfies ``MetricsCollectorProtocol``.

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

        Reports the buffered-execution count and any ad-hoc metrics recorded via
        :meth:`record_metric`. (Aggregated per-variant metrics are retrieved
        asynchronously via :meth:`get_variant_metrics`.)

        Returns:
            A mapping of the collector's current metric state.
        """
        snapshot: Dict[str, Any] = {
            "buffered_executions": len(self._metrics_buffer),
        }
        snapshot.update(self._adhoc_metrics)
        return snapshot

    async def start(self) -> None:
        """Start collecting metrics from ObservabilityBus."""
        bus = get_observability_bus()
        await bus.connect()

        # Subscribe to workflow events
        handle = bus.subscribe("workflow.*", self._on_workflow_event)
        self._subscription_handles.append(handle)

    async def stop(self) -> None:
        """Stop collecting metrics."""
        # Flush buffer
        await self._flush_buffer()

        # Unsubscribe from events
        bus = get_observability_bus()
        for handle in self._subscription_handles:
            await bus.unsubscribe(handle)

        self._subscription_handles.clear()

    async def _on_workflow_event(self, event: MessagingEvent) -> None:
        """Handle workflow execution events.

        Args:
            event: Workflow event from ObservabilityBus
        """
        # Check if event has experiment context
        experiment_id = event.data.get("experiment_id")
        variant_id = event.data.get("variant_id")

        if not experiment_id or not variant_id:
            return  # Not part of an experiment

        # Extract metrics from completed workflow
        if event.topic.endswith("completed") or event.topic.endswith("failed"):
            await self._record_execution(event)

    async def _record_execution(self, event: MessagingEvent) -> None:
        """Record execution metrics from event.

        Args:
            event: Workflow event
        """
        import uuid

        metrics = ExecutionMetrics(
            execution_id=event.data.get("execution_id", uuid.uuid4().hex),
            experiment_id=event.data["experiment_id"],
            variant_id=event.data["variant_id"],
            user_id=event.data.get("user_id", "unknown"),
            execution_time=event.data.get("duration_seconds", 0.0),
            node_times=event.data.get("node_times", {}),
            prompt_tokens=event.data.get("prompt_tokens", 0),
            completion_tokens=event.data.get("completion_tokens", 0),
            total_tokens=event.data.get("total_tokens", 0),
            tool_calls_count=event.data.get("tool_calls_count", 0),
            tool_calls_by_name=event.data.get("tool_calls_by_name", {}),
            tool_errors=event.data.get("tool_errors", 0),
            success=event.data.get("success", True),
            error_message=event.data.get("error"),
            estimated_cost=event.data.get("estimated_cost", 0.0),
            custom_metrics=event.data.get("custom_metrics", {}),
            timestamp=time.time(),
            workflow_name=event.data.get("workflow_name", ""),
            workflow_type=event.data.get("workflow_type", ""),
        )

        # Add to buffer
        self._metrics_buffer.append(metrics)

        # Flush buffer if full
        if len(self._metrics_buffer) >= self.buffer_size:
            await self._flush_buffer()

        # Update aggregation cache
        await self._update_aggregation(metrics)

    async def _flush_buffer(self) -> None:
        """Flush metrics buffer to storage."""
        if not self._metrics_buffer:
            return

        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        import json

        for metrics in self._metrics_buffer:
            cursor.execute(
                """
                INSERT INTO executions (
                    execution_id, experiment_id, variant_id, user_id,
                    execution_time, node_times_json,
                    prompt_tokens, completion_tokens, total_tokens,
                    tool_calls_count, tool_calls_by_name_json, tool_errors,
                    success, error_message, estimated_cost, custom_metrics_json,
                    timestamp, workflow_name, workflow_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.execution_id,
                    metrics.experiment_id,
                    metrics.variant_id,
                    metrics.user_id,
                    metrics.execution_time,
                    json.dumps(metrics.node_times),
                    metrics.prompt_tokens,
                    metrics.completion_tokens,
                    metrics.total_tokens,
                    metrics.tool_calls_count,
                    json.dumps(metrics.tool_calls_by_name),
                    metrics.tool_errors,
                    1 if metrics.success else 0,
                    metrics.error_message,
                    metrics.estimated_cost,
                    json.dumps(metrics.custom_metrics),
                    metrics.timestamp,
                    metrics.workflow_name,
                    metrics.workflow_type,
                ),
            )

        conn.commit()
        conn.close()

        # Clear buffer
        self._metrics_buffer.clear()

    async def _update_aggregation(self, metrics: ExecutionMetrics) -> None:
        """Update in-memory aggregation cache.

        Args:
            metrics: Execution metrics
        """
        key = f"{metrics.experiment_id}:{metrics.variant_id}"

        # Append to aggregation arrays
        self._aggregation_cache[key]["execution_times"].append(metrics.execution_time)
        self._aggregation_cache[key]["token_counts"].append(metrics.total_tokens)
        self._aggregation_cache[key]["tool_call_counts"].append(metrics.tool_calls_count)
        self._aggregation_cache[key]["successes"].append(1 if metrics.success else 0)
        self._aggregation_cache[key]["costs"].append(metrics.estimated_cost)

    async def get_variant_metrics(
        self,
        experiment_id: str,
        variant_id: str,
    ) -> Optional[AggregatedMetrics]:
        """Get aggregated metrics for a variant.

        Args:
            experiment_id: Experiment identifier
            variant_id: Variant identifier

        Returns:
            Aggregated metrics or None if no data
        """
        # Check cache first
        key = f"{experiment_id}:{variant_id}"
        if key in self._aggregation_cache:
            return self._compute_aggregated_metrics(self._aggregation_cache[key], variant_id)

        # Load from database
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT execution_time, total_tokens, tool_calls_count,
                   success, estimated_cost
            FROM executions
            WHERE experiment_id = ? AND variant_id = ?
        """,
            (experiment_id, variant_id),
        )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        # Build cache from database
        cache = {
            "execution_times": [row[0] for row in rows],
            "token_counts": [row[1] for row in rows],
            "tool_call_counts": [row[2] for row in rows],
            "successes": [1 if row[3] else 0 for row in rows],
            "costs": [row[4] for row in rows],
        }

        # Cache it
        self._aggregation_cache[key] = cache

        return self._compute_aggregated_metrics(cache, variant_id)

    def _compute_aggregated_metrics(
        self,
        cache: Dict[str, List[float]],
        variant_id: str,
    ) -> AggregatedMetrics:
        """Compute aggregated metrics from cache.

        Args:
            cache: Aggregation cache with raw data
            variant_id: Variant identifier

        Returns:
            Aggregated metrics
        """
        _require_scipy()
        execution_times = np.array(cache["execution_times"])
        token_counts = np.array(cache["token_counts"])
        tool_call_counts = np.array(cache["tool_call_counts"])
        successes = np.array(cache["successes"])
        costs = np.array(cache["costs"])

        # Execution time
        exec_mean = float(np.mean(execution_times))
        exec_median = float(np.median(execution_times))
        exec_std = float(np.std(execution_times))
        exec_p95 = float(np.percentile(execution_times, 95))

        # Confidence interval for execution time
        if len(execution_times) > 1:
            exec_ci = stats.t.interval(
                0.95,
                len(execution_times) - 1,
                loc=exec_mean,
                scale=stats.sem(execution_times),
            )
            exec_ci = (float(exec_ci[0]), float(exec_ci[1]))
        else:
            exec_ci = (exec_mean, exec_mean)

        # Token usage
        tokens_mean = float(np.mean(token_counts))
        tokens_median = float(np.median(token_counts))
        tokens_sum = int(np.sum(token_counts))

        # Tool usage
        tools_mean = float(np.mean(tool_call_counts))

        # Success rate
        success_count = int(np.sum(successes))
        success_rate = float(success_count / len(successes))

        # Confidence interval for success rate (Wilson score)
        if len(successes) > 0:
            success_ci = self._calculate_proportion_ci(success_count, len(successes))
        else:
            success_ci = (0.0, 0.0)

        # Cost
        total_cost = float(np.sum(costs))
        cost_mean = float(np.mean(costs))

        return AggregatedMetrics(
            variant_id=variant_id,
            sample_count=len(execution_times),
            execution_time_mean=exec_mean,
            execution_time_median=exec_median,
            execution_time_std=exec_std,
            execution_time_p95=exec_p95,
            execution_time_ci=exec_ci,
            total_tokens_mean=tokens_mean,
            total_tokens_median=tokens_median,
            total_tokens_sum=tokens_sum,
            tool_calls_mean=tools_mean,
            tool_errors_total=0,  # Track separately
            tool_error_rate=0.0,
            success_count=success_count,
            success_rate=success_rate,
            success_rate_ci=success_ci,
            total_cost=total_cost,
            cost_per_execution_mean=cost_mean,
            custom_metrics_aggregated={},
        )

    def _calculate_proportion_ci(
        self,
        successes: int,
        total: int,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Calculate confidence interval for proportion (Wilson score).

        Args:
            successes: Number of successes
            total: Total sample size
            confidence: Confidence level

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        _require_scipy()
        if total == 0:
            return (0.0, 0.0)

        p = successes / total
        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator

        return (center - margin, center + margin)

    async def get_all_variant_metrics(
        self,
        experiment_id: str,
    ) -> Dict[str, AggregatedMetrics]:
        """Get aggregated metrics for all variants in an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dictionary mapping variant_id to aggregated metrics
        """
        # Get all variants for this experiment
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT DISTINCT variant_id FROM executions
            WHERE experiment_id = ?
        """,
            (experiment_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        # Get metrics for each variant
        metrics = {}
        for row in rows:
            variant_id = row[0]
            variant_metrics = await self.get_variant_metrics(experiment_id, variant_id)
            if variant_metrics:
                metrics[variant_id] = variant_metrics

        return metrics

    async def get_execution_metrics(
        self,
        experiment_id: str,
        variant_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ExecutionMetrics]:
        """Get raw execution metrics.

        Args:
            experiment_id: Experiment identifier
            variant_id: Optional variant identifier filter
            limit: Maximum number of records to return

        Returns:
            List of execution metrics
        """
        conn = sqlite3.connect(str(self.storage_path))
        cursor = conn.cursor()

        if variant_id:
            cursor.execute(
                """
                SELECT execution_id, experiment_id, variant_id, user_id,
                       execution_time, node_times_json,
                       prompt_tokens, completion_tokens, total_tokens,
                       tool_calls_count, tool_calls_by_name_json, tool_errors,
                       success, error_message, estimated_cost, custom_metrics_json,
                       timestamp, workflow_name, workflow_type
                FROM executions
                WHERE experiment_id = ? AND variant_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (experiment_id, variant_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT execution_id, experiment_id, variant_id, user_id,
                       execution_time, node_times_json,
                       prompt_tokens, completion_tokens, total_tokens,
                       tool_calls_count, tool_calls_by_name_json, tool_errors,
                       success, error_message, estimated_cost, custom_metrics_json,
                       timestamp, workflow_name, workflow_type
                FROM executions
                WHERE experiment_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (experiment_id, limit),
            )

        import json

        rows = cursor.fetchall()
        conn.close()

        metrics = []
        for row in rows:
            metrics.append(
                ExecutionMetrics(
                    execution_id=row[0],
                    experiment_id=row[1],
                    variant_id=row[2],
                    user_id=row[3],
                    execution_time=row[4],
                    node_times=json.loads(row[5]),
                    prompt_tokens=row[6],
                    completion_tokens=row[7],
                    total_tokens=row[8],
                    tool_calls_count=row[9],
                    tool_calls_by_name=json.loads(row[10]),
                    tool_errors=row[11],
                    success=bool(row[12]),
                    error_message=row[13],
                    estimated_cost=row[14],
                    custom_metrics=json.loads(row[15]),
                    timestamp=row[16],
                    workflow_name=row[17],
                    workflow_type=row[18],
                )
            )

        return metrics

    def clear_cache(self) -> None:
        """Clear in-memory aggregation cache."""
        self._aggregation_cache.clear()
