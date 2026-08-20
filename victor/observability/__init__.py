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

"""Victor Observability - Unified event bus and telemetry.

This module provides a centralized event bus for all Victor observations,
implementing the Pub/Sub pattern for decoupled event handling.

Architecture:
    ObservabilityBus (Singleton) from victor.core.events
        │
        ├── Subscribers (Observer Pattern)
        │   ├── LoggingSubscriber
        │   ├── MetricsSubscriber
        │   └── Custom handlers
        │
        └── Exporters (Strategy Pattern)
            ├── JsonLineExporter
            ├── CallbackExporter
            └── OpenTelemetryExporter

Example:
    from victor.core.events import get_observability_bus, MessagingEvent
    from victor.observability import JsonLineExporter

    # Get singleton bus
    bus = get_observability_bus()
    await bus.connect()

    # Subscribe to events
    async def on_tool_event(event: MessagingEvent):
        print(f"Tool {event.topic}: {event.data}")

    bus.subscribe("tool.*", on_tool_event)

    # Emit events
    await bus.emit("tool.read", {"path": "/tmp/test.txt"})

    # Add exporters for persistence
    bus.add_exporter(JsonLineExporter("events.jsonl"))
"""

# Event system - using canonical core/events
from victor.core.events import (
    ObservabilityBus,
    MessagingEvent,
    get_observability_bus,
)
from victor.observability.exporters import (
    BaseExporter,
    BufferedExporter,
    CallbackExporter,
    CompositeExporter,
    FilteringExporter,
    JsonLineExporter,
)
from victor.observability.hooks import (
    LoggingHook,
    MetricsHook,
    StateHookManager,
    StateTransitionHook,
    TransitionHistory,
    TransitionRecord,
)
from victor.observability.integration import (
    ObservabilityIntegration,
    ToolEventMiddleware,
    setup_observability,
)
from victor.observability.metrics import (
    Counter,
    Gauge,
    Histogram,
    Metric,
    MetricsCollector,
    MetricsRegistry,
    Timer,
)
from victor.observability.resilience import (
    BackoffStrategy,
    Bulkhead,
    BulkheadFullError,
    CircuitBreaker,
    CircuitBreakerError,
    ConstantBackoff,
    DecorrelatedJitterBackoff,
    ExponentialBackoff,
    LinearBackoff,
    RateLimiter,
    ResiliencePolicy,
    ObservabilityRetryConfig,
    retry_with_backoff,
    with_timeout,
)
from victor.observability.otel_exporter import (
    AsyncBatchingExporter,
    create_otel_exporter,
)
from victor.observability.debugger import AgentDebugger
from victor.observability.integration_audit import IntegrationAuditService
from victor.observability.tracing import (
    ExecutionSpan,
    ExecutionTracer,
    ToolCallRecord,
    ToolCallTracer,
)

# Conditional OTEL exports
try:
    from victor.observability.otel_exporter import (
        OpenTelemetryExporter,
        OTELSpanManager,
    )

    _OTEL_EXPORTS = ["OpenTelemetryExporter", "OTELSpanManager"]
except ImportError:
    _OTEL_EXPORTS = []

__all__ = [
    # Core (canonical event system)
    "ObservabilityBus",
    "MessagingEvent",
    "get_observability_bus",
    # Exporters
    "BaseExporter",
    "JsonLineExporter",
    "CallbackExporter",
    "CompositeExporter",
    "FilteringExporter",
    "BufferedExporter",
    "AsyncBatchingExporter",
    "create_otel_exporter",
    # Hooks
    "StateHookManager",
    "StateTransitionHook",
    "TransitionRecord",
    "TransitionHistory",
    "LoggingHook",
    "MetricsHook",
    # Integration
    "ObservabilityIntegration",
    "ToolEventMiddleware",
    "setup_observability",
    # Tracing
    "ExecutionTracer",
    "ExecutionSpan",
    "ToolCallTracer",
    "ToolCallRecord",
    # Debugging
    "AgentDebugger",
    # Audit
    "IntegrationAuditService",
    # Metrics
    "Metric",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "MetricsRegistry",
    "MetricsCollector",
    # Resilience
    "BackoffStrategy",
    "ExponentialBackoff",
    "LinearBackoff",
    "ConstantBackoff",
    "DecorrelatedJitterBackoff",
    "ObservabilityRetryConfig",
    "retry_with_backoff",
    "CircuitBreaker",
    "CircuitBreakerError",
    "Bulkhead",
    "BulkheadFullError",
    "RateLimiter",
    "ResiliencePolicy",
    "with_timeout",
] + _OTEL_EXPORTS

# Submodules - access via victor.observability.analytics, etc.
from victor.observability import analytics
from victor.observability import debug
from victor.observability import profiler
from victor.observability import telemetry
from victor.observability import pipeline

__version__ = "0.5.1"
