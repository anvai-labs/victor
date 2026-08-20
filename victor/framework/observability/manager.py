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

"""Unified observability manager for Victor framework.

This module provides a centralized observability manager that aggregates
metrics from all components in the framework.

Design Pattern: Registry Pattern + Observer Pattern
- Central registry for metrics sources
- Automatic metrics collection
- Historical metrics tracking
- Thread-safe operations

Phase 4: Enhance Observability with Unified Dashboard

Integration Point:
    Use in CLI commands and dashboard for unified metrics

Example:
    manager = ObservabilityManager.get_instance()

    # Register a metrics source
    manager.register_source(my_capability)

    # Collect metrics from all sources
    collection = manager.collect_metrics()

    # Get dashboard data
    dashboard_data = manager.get_dashboard_data()
    print(f"Sources: {dashboard_data['total_sources']}")
    print(f"Cache hit rate: {dashboard_data['cache_hit_rate']:.2%}")
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Deque, Dict, List, Optional, Set
from weakref import WeakSet

import psutil

from victor.framework.observability.metrics import (
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    Metric,
    MetricLabel,
    MetricNames,
    MetricSource,
    MetricsCollection,
    MetricsSnapshot,
    SummaryMetric,
)

logger = logging.getLogger(__name__)


# Default configuration


@dataclass
class ObservabilityConfig:
    """Configuration for the observability manager.

    Attributes:
        max_history_size: Maximum number of historical snapshots to keep
        collection_timeout_seconds: Timeout for collecting metrics from a source
        enable_system_metrics: Whether to collect system metrics
        system_metrics_interval_seconds: Interval between system metrics collection
        history_retention_hours: How long to keep historical data
    """

    max_history_size: int = 1000
    collection_timeout_seconds: float = 5.0
    enable_system_metrics: bool = True
    system_metrics_interval_seconds: float = 60.0
    history_retention_hours: float = 24.0


@dataclass
class DashboardData:
    """Aggregated data for dashboard display.

    Attributes:
        timestamp: When the dashboard data was generated
        total_sources: Total number of registered sources
        sources_by_type: Count of sources by type
        cache_metrics: Aggregated cache metrics
        tool_metrics: Aggregated tool metrics
        coordinator_metrics: Aggregated coordinator metrics
        capability_metrics: Aggregated capability metrics
        system_metrics: System resource metrics
        alerts: List of alerts/warnings
    """

    timestamp: float = field(default_factory=time.time)
    total_sources: int = 0
    sources_by_type: Dict[str, int] = field(default_factory=dict)
    cache_metrics: Dict[str, Any] = field(default_factory=dict)
    tool_metrics: Dict[str, Any] = field(default_factory=dict)
    coordinator_metrics: Dict[str, Any] = field(default_factory=dict)
    capability_metrics: Dict[str, Any] = field(default_factory=dict)
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    vertical_metrics: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of dashboard data
        """
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "total_sources": self.total_sources,
            "sources_by_type": self.sources_by_type,
            "cache_metrics": self.cache_metrics,
            "tool_metrics": self.tool_metrics,
            "coordinator_metrics": self.coordinator_metrics,
            "capability_metrics": self.capability_metrics,
            "system_metrics": self.system_metrics,
            "vertical_metrics": self.vertical_metrics,
            "alerts": self.alerts,
        }


class ObservabilityManager:
    """Unified observability manager for Victor framework.

    Manages metrics collection from all registered sources and provides
    unified dashboard data.

    Features:
    - Thread-safe source registration
    - Automatic metrics collection
    - Historical metrics tracking
    - System metrics collection
    - Dashboard data aggregation
    - Alert generation

    Thread Safety:
        This class uses locks for thread-safe operations.

    Lifecycle:
        The singleton instance is created on first access via get_instance().
        Use close() to release resources.

    Example:
        manager = ObservabilityManager.get_instance()

        # Register a metrics source
        manager.register_source(my_capability)

        # Collect metrics
        collection = manager.collect_metrics()

        # Get dashboard data
        dashboard_data = manager.get_dashboard_data()

        print(f"Cache hit rate: {dashboard_data['cache_metrics']['hit_rate']:.2%}")
    """

    _instance: Optional[ObservabilityManager] = None
    _lock = threading.Lock()

    def __init__(self, config: Optional[ObservabilityConfig] = None):
        """Initialize the observability manager.

        Note: This class should be instantiated via get_instance()
        to ensure singleton behavior.

        Args:
            config: Manager configuration (uses defaults if None)
        """
        self._config = config or ObservabilityConfig()
        self._sources: WeakSet[MetricSource] = WeakSet()
        self._sources_lock = threading.RLock()

        # Historical metrics storage
        self._history: Deque[MetricsCollection] = deque()
        self._history_lock = threading.RLock()

        # System metrics tracking
        self._last_system_collection: float = 0.0

        # Statistics
        self._stats_lock = threading.RLock()
        self._collection_count: int = 0
        self._collection_errors: int = 0
        self._last_collection_time: float = 0.0

        logger.info("ObservabilityManager initialized")

    @classmethod
    def get_instance(cls, config: Optional[ObservabilityConfig] = None) -> ObservabilityManager:
        """Get the singleton observability manager instance.

        Args:
            config: Manager configuration (only used on first call)

        Returns:
            ObservabilityManager singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance.

        This is a dangerous method that should only be used in testing.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance = None

    def register_source(self, source: MetricSource) -> None:
        """Register a metrics source.

        Args:
            source: Component that implements MetricSource protocol

        Example:
            manager = ObservabilityManager.get_instance()
            manager.register_source(my_capability)
        """
        if not isinstance(source, MetricSource):
            logger.warning(f"Source {source} does not implement MetricSource protocol")
            return

        with self._sources_lock:
            self._sources.add(source)

        logger.info(f"Registered metrics source: {source.source_id} ({source.source_type})")

    def unregister_source(self, source: MetricSource) -> None:
        """Unregister a metrics source.

        Args:
            source: Component to unregister
        """
        with self._sources_lock:
            self._sources.discard(source)

        logger.info(f"Unregistered metrics source: {source.source_id}")

    def list_sources(self) -> List[str]:
        """List all registered source IDs.

        Returns:
            List of source IDs
        """
        with self._sources_lock:
            return [s.source_id for s in self._sources]

    def collect_metrics(self, include_system: bool = True) -> MetricsCollection:
        """Collect metrics from all registered sources.

        Args:
            include_system: Whether to include system metrics

        Returns:
            MetricsCollection with snapshots from all sources

        Example:
            collection = manager.collect_metrics()
            for snapshot in collection.snapshots:
                print(f"{snapshot.source_id}: {len(snapshot.metrics)} metrics")
        """
        collection = MetricsCollection()
        start_time = time.time()

        with self._sources_lock:
            sources = list(self._sources)

        # Collect from each source
        for source in sources:
            try:
                snapshot = source.get_metrics()
                collection.add_snapshot(snapshot)
            except Exception as e:
                logger.warning(f"Failed to collect metrics from {source.source_id}: {e}")
                with self._stats_lock:
                    self._collection_errors += 1

        # Add system metrics if enabled
        if include_system and self._config.enable_system_metrics:
            current_time = time.time()
            if (
                current_time - self._last_system_collection
                >= self._config.system_metrics_interval_seconds
            ):
                system_snapshot = self._collect_system_metrics()
                collection.add_snapshot(system_snapshot)
                self._last_system_collection = current_time

        # Store in history
        with self._history_lock:
            self._history.append(collection)
            # Trim history if needed
            while len(self._history) > self._config.max_history_size:
                self._history.popleft()

        # Update statistics
        with self._stats_lock:
            self._collection_count += 1
            self._last_collection_time = time.time() - start_time

        logger.debug(
            f"Collected metrics from {len(collection.snapshots)} sources "
            f"in {self._last_collection_time:.3f}s"
        )

        return collection

    def _collect_system_metrics(self) -> MetricsSnapshot:
        """Collect system resource metrics.

        Returns:
            MetricsSnapshot with system metrics
        """
        metrics: List[Metric] = []
        process = psutil.Process()

        # Memory usage
        memory_info = process.memory_info()
        metrics.append(
            GaugeMetric(
                name=MetricNames.SYSTEM_MEMORY_USAGE,
                description="Process memory usage in bytes",
                value=memory_info.rss,
                labels=(MetricLabel(key="type", value="rss"),),
            )
        )

        # CPU usage
        cpu_percent = process.cpu_percent()
        metrics.append(
            GaugeMetric(
                name=MetricNames.SYSTEM_CPU_USAGE,
                description="Process CPU usage percentage",
                value=cpu_percent,
                labels=(MetricLabel(key="type", value="process"),),
            )
        )

        # System-wide CPU
        system_cpu = psutil.cpu_percent()
        metrics.append(
            GaugeMetric(
                name=MetricNames.SYSTEM_CPU_USAGE,
                description="System CPU usage percentage",
                value=system_cpu,
                labels=(MetricLabel(key="type", value="system"),),
            )
        )

        return MetricsSnapshot(
            source_id="system",
            source_type="system",
            metrics=tuple(metrics),
        )

    def get_dashboard_data(self) -> DashboardData:
        """Get aggregated dashboard data.

        Collects metrics from all sources and aggregates them for dashboard display.

        Returns:
            DashboardData with aggregated metrics

        Example:
            data = manager.get_dashboard_data()
            print(f"Total sources: {data.total_sources}")
            print(f"Cache hit rate: {data.cache_metrics['hit_rate']:.2%}")
        """
        collection = self.collect_metrics()
        dashboard_data = DashboardData(timestamp=collection.timestamp)

        # Count sources by type
        type_counts: Dict[str, int] = {}
        for snapshot in collection.snapshots:
            type_counts[snapshot.source_type] = type_counts.get(snapshot.source_type, 0) + 1
        dashboard_data.sources_by_type = type_counts
        dashboard_data.total_sources = len(collection.snapshots)

        # Aggregate cache metrics
        dashboard_data.cache_metrics = self._aggregate_cache_metrics(collection)

        # Aggregate tool metrics
        dashboard_data.tool_metrics = self._aggregate_tool_metrics(collection)

        # Aggregate coordinator metrics
        dashboard_data.coordinator_metrics = self._aggregate_coordinator_metrics(collection)

        # Aggregate capability metrics
        dashboard_data.capability_metrics = self._aggregate_capability_metrics(collection)

        # Aggregate vertical metrics
        dashboard_data.vertical_metrics = self._aggregate_vertical_metrics(collection)

        # System metrics
        dashboard_data.system_metrics = self._extract_system_metrics(collection)

        # Generate alerts
        dashboard_data.alerts = self._generate_alerts(dashboard_data)

        return dashboard_data

    def _aggregate_cache_metrics(self, collection: MetricsCollection) -> Dict[str, Any]:
        """Aggregate cache metrics.

        Args:
            collection: Metrics collection

        Returns:
            Dictionary with aggregated cache metrics
        """
        result = {
            "total_hits": 0,
            "total_misses": 0,
            "hit_rate": 0.0,
            "total_size": 0,
            "evictions": 0,
        }

        for snapshot in collection.snapshots:
            if snapshot.source_type != "cache":
                continue

            for metric in snapshot.metrics:
                if metric.name == MetricNames.CACHE_HIT_COUNT:
                    if isinstance(metric, CounterMetric):
                        result["total_hits"] += metric.value
                elif metric.name == MetricNames.CACHE_MISS_COUNT:
                    if isinstance(metric, CounterMetric):
                        result["total_misses"] += metric.value
                elif metric.name == MetricNames.CACHE_EVICTION_COUNT:
                    if isinstance(metric, CounterMetric):
                        result["evictions"] += metric.value

        # Calculate hit rate
        total_requests = result["total_hits"] + result["total_misses"]
        if total_requests > 0:
            result["hit_rate"] = result["total_hits"] / total_requests

        return result

    def _aggregate_tool_metrics(self, collection: MetricsCollection) -> Dict[str, Any]:
        """Aggregate tool metrics.

        Args:
            collection: Metrics collection

        Returns:
            Dictionary with aggregated tool metrics
        """
        result = {
            "total_calls": 0,
            "total_errors": 0,
            "success_rate": 0.0,
            "average_latency": 0.0,
        }

        total_latency = 0.0
        latency_count = 0

        for snapshot in collection.snapshots:
            if snapshot.source_type != "tool":
                continue

            for metric in snapshot.metrics:
                if metric.name == MetricNames.TOOL_CALL_COUNT:
                    if isinstance(metric, CounterMetric):
                        result["total_calls"] += metric.value
                elif metric.name == MetricNames.TOOL_ERROR_COUNT:
                    if isinstance(metric, CounterMetric):
                        result["total_errors"] += metric.value
                elif metric.name == MetricNames.TOOL_LATENCY:
                    if isinstance(metric, HistogramMetric):
                        total_latency += metric.sum
                        latency_count += metric.count

        # Calculate success rate
        if result["total_calls"] > 0:
            result["success_rate"] = (result["total_calls"] - result["total_errors"]) / result[
                "total_calls"
            ]

        # Calculate average latency
        if latency_count > 0:
            result["average_latency"] = total_latency / latency_count

        return result

    def _aggregate_coordinator_metrics(self, collection: MetricsCollection) -> Dict[str, Any]:
        """Aggregate coordinator metrics.

        Args:
            collection: Metrics collection

        Returns:
            Dictionary with aggregated coordinator metrics
        """
        result = {
            "total_operations": 0,
            "total_errors": 0,
            "average_latency": 0.0,
        }

        total_latency = 0.0
        latency_count = 0

        for snapshot in collection.snapshots:
            if snapshot.source_type != "coordinator":
                continue

            for metric in snapshot.metrics:
                if metric.name == MetricNames.COORDINATOR_OPERATION_COUNT:
                    if isinstance(metric, CounterMetric):
                        result["total_operations"] += metric.value
                elif metric.name == MetricNames.COORDINATOR_ERROR_COUNT:
                    if isinstance(metric, CounterMetric):
                        result["total_errors"] += metric.value
                elif metric.name == MetricNames.COORDINATOR_LATENCY:
                    if isinstance(metric, HistogramMetric):
                        total_latency += metric.sum
                        latency_count += metric.count

        if latency_count > 0:
            result["average_latency"] = total_latency / latency_count

        return result

    def _aggregate_capability_metrics(self, collection: MetricsCollection) -> Dict[str, Any]:
        """Aggregate capability metrics.

        Args:
            collection: Metrics collection

        Returns:
            Dictionary with aggregated capability metrics
        """
        result = {
            "total_accesses": 0,
            "total_errors": 0,
            "capability_count": 0,
        }

        for snapshot in collection.snapshots:
            if snapshot.source_type != "capability":
                continue

            result["capability_count"] += 1

            for metric in snapshot.metrics:
                if metric.name == MetricNames.CAPABILITY_ACCESS_COUNT:
                    if isinstance(metric, CounterMetric):
                        result["total_accesses"] += metric.value
                elif metric.name == MetricNames.CAPABILITY_ERROR_COUNT:
                    if isinstance(metric, CounterMetric):
                        result["total_errors"] += metric.value

        return result

    def _aggregate_vertical_metrics(self, collection: MetricsCollection) -> Dict[str, Any]:
        """Aggregate vertical metrics.

        Args:
            collection: Metrics collection

        Returns:
            Dictionary with aggregated vertical metrics
        """
        result = {
            "total_requests": 0,
            "total_errors": 0,
            "average_latency": 0.0,
            "vertical_count": 0,
        }

        total_latency = 0.0
        latency_count = 0

        for snapshot in collection.snapshots:
            if snapshot.source_type != "vertical":
                continue

            result["vertical_count"] += 1

            for metric in snapshot.metrics:
                if metric.name == MetricNames.VERTICAL_REQUEST_COUNT:
                    if isinstance(metric, CounterMetric):
                        result["total_requests"] += metric.value
                elif metric.name == MetricNames.VERTICAL_ERROR_COUNT:
                    if isinstance(metric, CounterMetric):
                        result["total_errors"] += metric.value
                elif metric.name == MetricNames.VERTICAL_LATENCY:
                    if isinstance(metric, HistogramMetric):
                        total_latency += metric.sum
                        latency_count += metric.count

        if latency_count > 0:
            result["average_latency"] = total_latency / latency_count

        return result

    def _extract_system_metrics(self, collection: MetricsCollection) -> Dict[str, Any]:
        """Extract system metrics from collection.

        Args:
            collection: Metrics collection

        Returns:
            Dictionary with system metrics
        """
        result = {
            "memory_usage_bytes": 0,
            "cpu_usage_percent": 0.0,
        }

        for snapshot in collection.snapshots:
            if snapshot.source_type != "system":
                continue

            for metric in snapshot.metrics:
                if metric.name == MetricNames.SYSTEM_MEMORY_USAGE:
                    if isinstance(metric, GaugeMetric):
                        result["memory_usage_bytes"] = metric.value
                elif metric.name == MetricNames.SYSTEM_CPU_USAGE:
                    if isinstance(metric, GaugeMetric):
                        if metric.get_label_value("type") == "process":
                            result["cpu_usage_percent"] = metric.value

        return result

    def _generate_alerts(self, dashboard_data: DashboardData) -> List[Dict[str, Any]]:
        """Generate alerts based on dashboard data.

        Args:
            dashboard_data: Dashboard data

        Returns:
            List of alert dictionaries
        """
        alerts = []

        # Low cache hit rate alert
        cache_hit_rate = dashboard_data.cache_metrics.get("hit_rate", 0.0)
        if cache_hit_rate < 0.5:
            alerts.append(
                {
                    "severity": "warning",
                    "type": "low_cache_hit_rate",
                    "message": f"Low cache hit rate: {cache_hit_rate:.1%}",
                    "value": cache_hit_rate,
                }
            )

        # High error rate alert
        tool_success_rate = dashboard_data.tool_metrics.get("success_rate", 1.0)
        if tool_success_rate < 0.9:
            alerts.append(
                {
                    "severity": "error",
                    "type": "high_tool_error_rate",
                    "message": f"High tool error rate: {(1 - tool_success_rate):.1%}",
                    "value": 1 - tool_success_rate,
                }
            )

        # High memory usage alert
        if psutil is not None:
            memory_percent = (
                dashboard_data.system_metrics.get("memory_usage_bytes", 0)
                / psutil.virtual_memory().total
                * 100
            )
            if memory_percent > 80:
                alerts.append(
                    {
                        "severity": "warning",
                        "type": "high_memory_usage",
                        "message": f"High memory usage: {memory_percent:.1f}%",
                        "value": memory_percent,
                    }
                )

        return alerts

    def get_historical_data(
        self,
        source_id: Optional[str] = None,
        source_type: Optional[str] = None,
        hours: float = 1.0,
    ) -> List[MetricsCollection]:
        """Get historical metrics data.

        Args:
            source_id: Filter by specific source ID
            source_type: Filter by source type
            hours: Hours of history to return (default: 1.0)

        Returns:
            List of historical metrics collections

        Example:
            # Get last hour of data
            history = manager.get_historical_data(hours=1.0)

            # Get cache metrics for last 6 hours
            history = manager.get_historical_data(source_type="cache", hours=6.0)
        """
        cutoff_time = time.time() - (hours * 3600)
        result = []

        with self._history_lock:
            for collection in self._history:
                if collection.timestamp < cutoff_time:
                    continue

                if source_id is not None:
                    snapshot = collection.get_by_source_id(source_id)
                    if snapshot:
                        filtered_collection = MetricsCollection(timestamp=collection.timestamp)
                        filtered_collection.add_snapshot(snapshot)
                        result.append(filtered_collection)
                elif source_type is not None:
                    snapshots = collection.get_by_source_type(source_type)
                    if snapshots:
                        filtered_collection = MetricsCollection(timestamp=collection.timestamp)
                        for snapshot in snapshots:
                            filtered_collection.add_snapshot(snapshot)
                        result.append(filtered_collection)
                else:
                    result.append(collection)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get observability manager statistics.

        Returns:
            Dictionary with manager statistics
        """
        with self._stats_lock, self._history_lock:
            return {
                "collection_count": self._collection_count,
                "collection_errors": self._collection_errors,
                "last_collection_duration": self._last_collection_time,
                "registered_sources": len(self._sources),
                "history_size": len(self._history),
                "history_retention_hours": self._config.history_retention_hours,
            }

    def clear_history(self) -> None:
        """Clear all historical metrics data."""
        with self._history_lock:
            self._history.clear()
        logger.info("Cleared metrics history")

    def close(self) -> None:
        """Close the observability manager and release resources.

        Called automatically on application shutdown.
        """
        logger.info("Closing ObservabilityManager...")
        self.clear_history()
        with self._sources_lock:
            self._sources.clear()
        logger.info("ObservabilityManager closed")


__all__ = [
    "ObservabilityManager",
    "ObservabilityConfig",
    "DashboardData",
]
