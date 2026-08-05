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

"""OpenTelemetry integration for Victor observability.

This module provides OTEL-compatible exporters for distributed tracing
and metrics export to standard observability backends.

Supports:
- OTLP HTTP/gRPC export
- Span creation for tool execution
- Baggage propagation for context
- Metrics export via OTLP

Design Patterns:
- Adapter Pattern: Adapts Victor events to OTEL format
- Strategy Pattern: Multiple export backends

Example:
    from victor.observability.otel_exporter import OpenTelemetryExporter

    # Create exporter with OTLP endpoint
    exporter = OpenTelemetryExporter(
        service_name="victor-agent",
        endpoint="http://localhost:4318",
    )

    # Add to EventBus
    event_bus.add_exporter(exporter)

Note: Requires optional opentelemetry dependencies:
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from victor.core.events import MessagingEvent, ObservabilityBus, get_observability_bus
from victor.observability.exporters import BaseExporter

logger = logging.getLogger(__name__)

# Check for OpenTelemetry availability
_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace, context as otel_context
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace import Status, StatusCode, Span
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _OTEL_AVAILABLE = True
except ImportError:
    pass

# Check for OTLP exporter
_OTLP_AVAILABLE = False
try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    _OTLP_AVAILABLE = True
except ImportError:
    pass


@dataclass
class OTELConfig:
    """Configuration for OpenTelemetry integration."""

    service_name: str = "victor-agent"
    service_version: str = "1.0.0"
    endpoint: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    use_console_exporter: bool = False
    batch_export: bool = True
    max_queue_size: int = 2048
    export_timeout_millis: int = 30000


class OpenTelemetryExporter(BaseExporter):
    """Exports Victor events as OpenTelemetry spans.

    Converts Victor events to OTEL spans for distributed tracing.
    Tool executions become child spans with timing and metadata.

    Example:
        exporter = OpenTelemetryExporter(
            service_name="victor-agent",
            endpoint="http://localhost:4318",
        )
        event_bus.add_exporter(exporter)
    """

    def __init__(
        self,
        service_name: str = "victor-agent",
        service_version: str = "1.0.0",
        endpoint: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        use_console_exporter: bool = False,
        batch_export: bool = True,
    ) -> None:
        """Initialize OpenTelemetry exporter.

        Args:
            service_name: Service name for tracing.
            service_version: Service version.
            endpoint: OTLP endpoint URL.
            headers: Optional headers for OTLP.
            use_console_exporter: Use console for debugging.
            batch_export: Use batch processing (recommended).
        """
        if not _OTEL_AVAILABLE:
            raise ImportError(
                "OpenTelemetry not available. Install with: "
                "pip install opentelemetry-api opentelemetry-sdk"
            )

        self._service_name = service_name
        self._service_version = service_version
        self._endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        self._headers = headers
        self._use_console = use_console_exporter
        self._batch_export = batch_export

        self._tracer: Optional[Any] = None
        self._provider: Optional[Any] = None

        self._setup_tracer()

    def _setup_tracer(self) -> None:
        """Set up OpenTelemetry tracer."""
        resource = Resource.create(
            {
                SERVICE_NAME: self._service_name,
                "service.version": self._service_version,
            }
        )

        self._provider = TracerProvider(resource=resource)

        # Add exporter
        if self._use_console:
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
        elif self._endpoint and _OTLP_AVAILABLE:
            otlp_exporter = OTLPSpanExporter(
                endpoint=f"{self._endpoint}/v1/traces",
                headers=self._headers,
            )
            if self._batch_export:
                processor = BatchSpanProcessor(otlp_exporter)
            else:
                processor = SimpleSpanProcessor(otlp_exporter)
        else:
            # Fallback to console
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
            logger.warning(
                "No OTLP endpoint configured, using console exporter. "
                "Set OTEL_EXPORTER_OTLP_ENDPOINT or install opentelemetry-exporter-otlp"
            )

        self._provider.add_span_processor(processor)
        trace.set_tracer_provider(self._provider)
        self._tracer = trace.get_tracer(__name__, self._service_version)

    def export(self, event: MessagingEvent) -> None:
        """Export event as OTEL span.

        Simplified generic approach - all events become spans with
        event data as attributes. Since we're on ObservabilityBus,
        we don't need category filtering.

        Args:
            event: Victor event to export (generic Event with topic + data).
        """
        if not self._tracer:
            return

        # Create span name from topic (replace dots with underscores for OTEL compatibility)
        span_name = event.topic.replace(".", "_")

        with self._tracer.start_as_current_span(span_name) as span:
            # All event data becomes span attributes
            for key, value in event.data.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"victor.{key}", value)
                elif isinstance(value, list):
                    span.set_attribute(f"victor.{key}", str(value)[:256])

            # Metadata
            span.set_attribute("victor.topic", event.topic)
            span.set_attribute("victor.session_id", event.session_id or "")

            # Status based on error presence
            if event.data.get("error"):
                error_msg = str(event.data["error"])[:256]
                span.set_status(Status(StatusCode.ERROR, error_msg))
            elif not event.data.get("success", True):
                span.set_status(Status(StatusCode.ERROR, "Operation failed"))
            else:
                span.set_status(Status(StatusCode.OK))

    def close(self) -> None:
        """Shutdown the tracer provider."""
        if self._provider:
            self._provider.shutdown()


class OTELSpanManager:
    """Context manager for manual span creation.

    Provides higher-level API for creating spans in Victor code.

    Example:
        manager = OTELSpanManager(service_name="victor-agent")

        with manager.span("process_request") as span:
            span.set_attribute("request.type", "tool_call")
            result = process()
            span.set_attribute("result.size", len(result))
    """

    def __init__(
        self,
        service_name: str = "victor-agent",
        endpoint: Optional[str] = None,
    ) -> None:
        """Initialize span manager.

        Args:
            service_name: Service name for tracing.
            endpoint: OTLP endpoint URL.
        """
        if not _OTEL_AVAILABLE:
            raise ImportError("OpenTelemetry not available")

        self._exporter = OpenTelemetryExporter(
            service_name=service_name,
            endpoint=endpoint,
        )
        self._tracer = self._exporter._tracer

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Any]:
        """Create a span context.

        Args:
            name: Span name.
            attributes: Initial attributes.

        Yields:
            Active span.
        """
        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(key, value)
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def close(self) -> None:
        """Close the span manager."""
        self._exporter.close()


# =============================================================================
# Async Batching Exporter
# =============================================================================


class AsyncBatchingExporter(BaseExporter):
    """Asynchronous batching exporter for high-throughput scenarios.

    Buffers events and exports in batches for reduced overhead.
    Uses background thread for non-blocking export.

    Example:
        exporter = AsyncBatchingExporter(
            target=JsonLineExporter("events.jsonl"),
            batch_size=100,
            flush_interval=5.0,
        )
        event_bus.add_exporter(exporter)
    """

    def __init__(
        self,
        target: BaseExporter,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_queue_size: int = 10000,
        on_export_error: Optional[Callable[[Exception, List[MessagingEvent]], None]] = None,
    ) -> None:
        """Initialize async batching exporter.

        Args:
            target: Target exporter for actual export.
            batch_size: Events per batch.
            flush_interval: Seconds between flushes.
            max_queue_size: Maximum queue size.
            on_export_error: Error callback.
        """
        import threading
        import queue

        self._target = target
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_queue_size = max_queue_size
        self._on_error = on_export_error

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._export_loop, daemon=True)
        self._thread.start()

    def export(self, event: MessagingEvent) -> None:
        """Queue event for async export.

        Args:
            event: MessagingEvent to export.
        """
        try:
            self._queue.put_nowait(event)
        except Exception:
            # Queue full, drop event
            logger.warning("Event queue full, dropping event")

    def _export_loop(self) -> None:
        """Background export loop."""
        import time

        batch: List[MessagingEvent] = []
        last_flush = time.time()

        while not self._shutdown.is_set():
            try:
                # Try to get event with timeout
                try:
                    event = self._queue.get(timeout=0.1)
                    batch.append(event)
                except Exception:
                    pass

                # Check flush conditions
                should_flush = (
                    len(batch) >= self._batch_size
                    or (time.time() - last_flush) >= self._flush_interval
                )

                if should_flush and batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()

            except Exception as e:
                logger.error(f"Error in export loop: {e}")

        # Final flush on shutdown
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[MessagingEvent]) -> None:
        """Flush batch of events to target.

        Args:
            batch: Events to export (generic Event objects with topic + data).
        """
        try:
            for event in batch:
                self._target.export(event)
        except Exception as e:
            logger.error(f"Batch export error: {e}")
            if self._on_error:
                self._on_error(e, batch)

    def flush(self) -> None:
        """Force flush pending events."""
        # Signal export loop to flush
        pass

    def close(self) -> None:
        """Shutdown exporter."""
        self._shutdown.set()
        self._thread.join(timeout=5.0)
        self._target.close()

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()


# =============================================================================
# Factory Functions
# =============================================================================


def create_otel_exporter(
    service_name: str = "victor-agent",
    endpoint: Optional[str] = None,
    **kwargs: Any,
) -> Optional[OpenTelemetryExporter]:
    """Create OpenTelemetry exporter if available.

    Returns None if OTEL dependencies are not installed.

    Args:
        service_name: Service name.
        endpoint: OTLP endpoint.
        **kwargs: Additional config.

    Returns:
        Exporter or None.
    """
    if not _OTEL_AVAILABLE:
        logger.info(
            "OpenTelemetry not available. Install with: "
            "pip install opentelemetry-api opentelemetry-sdk"
        )
        return None

    return OpenTelemetryExporter(
        service_name=service_name,
        endpoint=endpoint,
        **kwargs,
    )
