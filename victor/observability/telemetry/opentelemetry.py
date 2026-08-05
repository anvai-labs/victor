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

"""OpenTelemetry setup and configuration.

This module provides a unified interface for setting up OpenTelemetry
tracing and metrics with OTLP exporters. The setup is optional and
gracefully handles missing dependencies.

Usage:
    from victor.observability.telemetry import setup_opentelemetry, get_tracer, get_meter

    # Initialize OpenTelemetry (call once at startup)
    setup_opentelemetry(
        service_name="victor",
        service_version="0.5.1",
        otlp_endpoint="http://localhost:4317",  # Optional, defaults to env vars
    )

    # Get tracer/meter for instrumentation
    tracer = get_tracer("victor.agent")
    meter = get_meter("victor.agent")

    # Use in code
    with tracer.start_as_current_span("process_request") as span:
        span.set_attribute("user.id", user_id)
        # ... do work ...
"""

import logging
import os
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Track whether OpenTelemetry is available and configured
_otel_available = False
_otel_configured = False
_tracer: Optional[Any] = None
_meter: Optional[Any] = None

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION

    _otel_available = True
except ImportError:
    logger.debug(
        "OpenTelemetry not available. Install with: "
        "pip install opentelemetry-sdk opentelemetry-exporter-otlp"
    )


def is_telemetry_enabled() -> bool:
    """Check if OpenTelemetry is available and configured.

    Returns:
        True if telemetry is enabled and configured
    """
    return _otel_available and _otel_configured


def setup_opentelemetry(
    service_name: str = "victor",
    service_version: str = "0.5.1",
    otlp_endpoint: Optional[str] = None,
    enable_tracing: bool = True,
    enable_metrics: bool = True,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Initialize OpenTelemetry tracing and metrics.

    This function sets up:
    - A TracerProvider with BatchSpanProcessor and OTLP exporter
    - A MeterProvider with periodic OTLP metric reader
    - Resource attributes for service identification

    Args:
        service_name: Name of the service (default: "victor")
        service_version: Version of the service (default: "0.5.1")
        otlp_endpoint: OTLP collector endpoint (default: from OTEL_EXPORTER_OTLP_ENDPOINT env var)
        enable_tracing: Enable tracing (default: True)
        enable_metrics: Enable metrics (default: True)

    Returns:
        Tuple of (tracer, meter) or (None, None) if OpenTelemetry is not available
    """
    global _otel_configured, _tracer, _meter

    if not _otel_available:
        logger.warning(
            "OpenTelemetry SDK not installed. Telemetry will be disabled. "
            "Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp"
        )
        return None, None

    if _otel_configured:
        logger.debug("OpenTelemetry already configured, returning existing tracer/meter")
        return _tracer, _meter

    # Get endpoint from args or environment
    endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.debug(
            "No OTLP endpoint specified. Set OTEL_EXPORTER_OTLP_ENDPOINT "
            "or pass otlp_endpoint parameter to enable telemetry export."
        )

    try:
        # Create resource with service information
        resource = Resource.create(
            {
                SERVICE_NAME: service_name,
                SERVICE_VERSION: service_version,
                "deployment.environment": os.environ.get("VICTOR_ENV", "development"),
            }
        )

        # Set up tracing
        if enable_tracing:
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)

            if endpoint:
                span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
                tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
                logger.info(f"OpenTelemetry tracing enabled, exporting to {endpoint}")
            else:
                logger.debug("OpenTelemetry tracing enabled without exporter (no endpoint)")

            _tracer = trace.get_tracer(service_name, service_version)

        # Set up metrics
        if enable_metrics:
            if endpoint:
                metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
                metric_reader = PeriodicExportingMetricReader(
                    metric_exporter,
                    export_interval_millis=60000,  # Export every 60 seconds
                )
                meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
                metrics.set_meter_provider(meter_provider)
                logger.info(f"OpenTelemetry metrics enabled, exporting to {endpoint}")
            else:
                meter_provider = MeterProvider(resource=resource)
                metrics.set_meter_provider(meter_provider)
                logger.debug("OpenTelemetry metrics enabled without exporter (no endpoint)")

            _meter = metrics.get_meter(service_name, service_version)

        _otel_configured = True
        return _tracer, _meter

    except Exception as e:
        logger.error(f"Failed to setup OpenTelemetry: {e}")
        return None, None


def get_tracer(name: Optional[str] = None) -> Optional[Any]:
    """Get a tracer instance for instrumentation.

    Args:
        name: Optional tracer name (uses service name if not provided)

    Returns:
        Tracer instance or None if not configured
    """
    if not _otel_available:
        return None

    if not _otel_configured:
        # Return a no-op tracer if not configured
        return trace.get_tracer(name or "victor")

    if name:
        return trace.get_tracer(name)
    return _tracer


def get_meter(name: Optional[str] = None) -> Optional[Any]:
    """Get a meter instance for metrics.

    Args:
        name: Optional meter name (uses service name if not provided)

    Returns:
        Meter instance or None if not configured
    """
    if not _otel_available:
        return None

    if not _otel_configured:
        # Return a no-op meter if not configured
        return metrics.get_meter(name or "victor")

    if name:
        return metrics.get_meter(name)
    return _meter


def trace_tool_execution(tool_name: str):
    """Decorator to trace tool execution.

    Usage:
        @trace_tool_execution("my_tool")
        async def my_tool(**kwargs):
            ...
    """

    def decorator(func):
        if not _otel_available:
            return func

        async def wrapper(*args, **kwargs):
            tracer = get_tracer("victor.tools")
            if tracer is None:
                return await func(*args, **kwargs)

            with tracer.start_as_current_span(f"tool.{tool_name}") as span:
                span.set_attribute("tool.name", tool_name)
                span.set_attribute("tool.args_count", len(kwargs))
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("tool.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("tool.success", False)
                    span.set_attribute("tool.error", str(e))
                    raise

        return wrapper

    return decorator
