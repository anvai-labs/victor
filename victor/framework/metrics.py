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

"""Framework-layer facade for metrics and telemetry.

This module provides a unified access point to metrics and telemetry components
from the framework layer. It re-exports existing implementations from core modules
without duplicating code, following the Facade Pattern.

Delegated modules:
- victor.observability.metrics: Counter, Gauge, Histogram, Timer metrics
- victor.telemetry: OpenTelemetry integration for tracing and metrics

Design Pattern: Facade
- Single import point for framework users
- No code duplication - pure re-exports
- Maintains backward compatibility with original modules
- Enables discovery of metrics capabilities through framework namespace

Example:
    from victor.framework import (
        # Metrics
        MetricsRegistry,
        Counter,
        Gauge,
        Histogram,
        Timer,

        # Telemetry
        setup_opentelemetry,
        get_tracer,
        get_meter,
    )

    # Get singleton registry
    registry = MetricsRegistry.get_instance()

    # Create metrics
    requests = registry.counter("requests_total", "Total API requests")
    active_sessions = registry.gauge("active_sessions", "Current active sessions")
    latency = registry.histogram("request_latency_ms", "Request latency in ms")

    # Use metrics
    requests.increment()
    requests.increment(labels={"provider": "anthropic"})
    active_sessions.set(10)
    latency.observe(45.5)

    # Timer context manager
    with Timer("operation_duration_ms", "Duration of operation").time():
        await perform_operation()

    # OpenTelemetry setup
    setup_opentelemetry(
        service_name="victor",
        otlp_endpoint="http://localhost:4317",
    )

    # Get tracers/meters
    tracer = get_tracer("my_module")
    with tracer.start_as_current_span("operation"):
        ...
"""

from __future__ import annotations

# =============================================================================
# Metrics System
# From: victor/observability/metrics.py
# =============================================================================
from victor.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    Metric,
    MetricLabels,
    MetricsCollector,
    MetricsRegistry,
    Timer,
    TimerContext,
)

# =============================================================================
# OpenTelemetry Integration
# From: victor/observability/telemetry/__init__.py
# =============================================================================
from victor.observability.telemetry import (
    get_meter,
    get_tracer,
    is_telemetry_enabled,
    setup_opentelemetry,
)

__all__ = [
    # Metrics
    "Counter",
    "Gauge",
    "Histogram",
    "Metric",
    "MetricLabels",
    "MetricsCollector",
    "MetricsRegistry",
    "Timer",
    "TimerContext",
    # Telemetry
    "get_meter",
    "get_tracer",
    "is_telemetry_enabled",
    "setup_opentelemetry",
]
