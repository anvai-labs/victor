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

"""Observability framework for Victor.

This package provides unified observability for all components in the
Victor framework, including metrics collection, aggregation, and
dashboard integration.

Phase 4: Enhance Observability with Unified Dashboard

Main Components:
- ObservabilityManager: Central manager for metrics collection
- Metric data models: CounterMetric, GaugeMetric, HistogramMetric, SummaryMetric
- MetricsSnapshot: Point-in-time metrics from a source
- DashboardData: Aggregated data for dashboard display

Example:
    from victor.framework.observability import (
        ObservabilityManager,
        CounterMetric,
        GaugeMetric,
        MetricNames,
    )

    # Get the manager instance
    manager = ObservabilityManager.get_instance()

    # Register your component as a metrics source
    manager.register_source(my_component)

    # Collect metrics
    collection = manager.collect_metrics()

    # Get dashboard data
    dashboard = manager.get_dashboard_data()
    print(f"Cache hit rate: {dashboard['cache_metrics']['hit_rate']:.2%}")
"""

from victor.framework.observability.metrics import (
    CounterMetric,
    GaugeMetric,
    HistogramBucket,
    HistogramMetric,
    LLMCallMetrics,
    Metric,
    MetricLabel,
    MetricNames,
    MetricSource,
    MetricType,
    MetricsCollector,
    MetricsCollection,
    MetricsExporter,
    MetricsSnapshot,
    SummaryMetric,
    ToolCallMetrics,
    # Agent metrics
    AgentMetrics,
)

from victor.framework.observability.manager import (
    DashboardData,
    ObservabilityConfig,
    ObservabilityManager,
)

from victor.framework.observability.persistence import (
    InMemoryMetricsStore,
    MetricsStore,
    SQLiteMetricsStore,
)

__all__ = [
    # Manager
    "ObservabilityManager",
    "ObservabilityConfig",
    "DashboardData",
    # Metrics
    "Metric",
    "MetricType",
    "MetricLabel",
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
    # Persistence
    "MetricsStore",
    "SQLiteMetricsStore",
    "InMemoryMetricsStore",
    # Protocols
    "MetricSource",
    # Constants
    "MetricNames",
]
