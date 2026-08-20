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

"""OpenTelemetry helpers for vertical loading observability.

This module provides span helpers and telemetry utilities for instrumenting
vertical loading operations, enabling production monitoring and debugging.

Design Principles:
    - Non-blocking telemetry (errors don't break loading)
    - Structured event data for analysis
    - Integration with OpenTelemetry standards
    - Support for both sync and async contexts
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class TelemetryStatus(str, Enum):
    """Status of a telemetry span or operation."""

    PENDING = "pending"  # Operation is in progress
    SUCCESS = "success"  # Operation completed successfully
    ERROR = "error"  # Operation failed
    SKIPPED = "skipped"  # Operation was skipped


@dataclass
class VerticalLoadSpan:
    """Context for a vertical loading operation.

    Attributes:
        vertical_name: Name of the vertical being loaded
        operation: Operation type (load, activate, discover, etc.)
        start_time_ns: Start time in nanoseconds
        end_time_ns: End time in nanoseconds
        status: Status of the operation (success, error, skipped)
        error: Error message if status is error
        metadata: Additional structured metadata
        extensions_loaded: Count of extensions loaded
        dependencies_resolved: List of resolved dependencies
    """

    vertical_name: str
    operation: str
    start_time_ns: int
    end_time_ns: Optional[int] = None
    status: TelemetryStatus = TelemetryStatus.PENDING
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extensions_loaded: int = 0
    dependencies_resolved: List[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        if self.end_time_ns is None:
            return None
        return (self.end_time_ns - self.start_time_ns) / 1_000_000

    @property
    def is_success(self) -> bool:
        """Check if operation succeeded."""
        return self.status == TelemetryStatus.SUCCESS

    @property
    def is_error(self) -> bool:
        """Check if operation failed."""
        return self.status == TelemetryStatus.ERROR


@contextmanager
def vertical_load_span(
    vertical_name: str,
    operation: str,
    emit_callback: Optional[Callable[[VerticalLoadSpan], None]] = None,
):
    """Context manager for instrumenting vertical loading operations.

    Args:
        vertical_name: Name of the vertical being loaded
        operation: Operation type (e.g., "load", "activate", "discover")
        emit_callback: Optional callback to emit the span to telemetry backend

    Yields:
        VerticalLoadSpan object for recording operation details

    Example:
        with vertical_load_span("coding", "load") as span:
            try:
                vertical = load_vertical("coding")
                span.status = TelemetryStatus.SUCCESS
            except Exception as e:
                span.status = TelemetryStatus.ERROR
                span.error = str(e)
    """
    span = VerticalLoadSpan(
        vertical_name=vertical_name,
        operation=operation,
        start_time_ns=time.time_ns(),
    )

    try:
        yield span
    finally:
        span.end_time_ns = time.time_ns()

        # Emit to callback if provided
        if emit_callback:
            try:
                emit_callback(span)
            except Exception as e:
                logger.debug(f"Failed to emit span telemetry: {e}")

        # Log slow operations (>1 second)
        if span.duration_ms and span.duration_ms > 1000:
            logger.warning(
                "Slow vertical operation: '%s' for '%s' took %.2fms",
                operation,
                vertical_name,
                span.duration_ms,
            )


class VerticalLoadTelemetry:
    """Telemetry recorder for vertical loading operations.

    This class provides methods for recording and emitting telemetry data
    for vertical loading operations, integrating with OpenTelemetry if available.

    Example:
        telemetry = VerticalLoadTelemetry()

        with telemetry.track_load("coding"):
            vertical = load_vertical("coding")

        metrics = telemetry.get_metrics()
    """

    def __init__(self) -> None:
        """Initialize the telemetry recorder."""
        self._spans: List[VerticalLoadSpan] = []
        self._active_spans: Dict[str, VerticalLoadSpan] = {}

    @contextmanager
    def track_load(
        self,
        vertical_name: str,
        operation: str = "load",
    ):
        """Context manager for tracking a vertical load operation.

        Args:
            vertical_name: Name of the vertical
            operation: Operation type

        Yields:
            VerticalLoadSpan for recording operation details
        """
        span = VerticalLoadSpan(
            vertical_name=vertical_name,
            operation=operation,
            start_time_ns=time.time_ns(),
        )

        self._active_spans[f"{vertical_name}:{operation}"] = span

        try:
            yield span
        finally:
            span.end_time_ns = time.time_ns()
            self._spans.append(span)
            del self._active_spans[f"{vertical_name}:{operation}"]

            # Emit to OpenTelemetry if available
            self._emit_span(span)

    def track_extension_load(
        self,
        vertical_name: str,
        extension_type: str,
        success: bool,
        duration_ms: float,
    ) -> None:
        """Track an extension load event.

        Args:
            vertical_name: Name of the vertical
            extension_type: Type of extension (e.g., "middleware", "safety")
            success: Whether the load succeeded
            duration_ms: Time taken to load
        """
        span = VerticalLoadSpan(
            vertical_name=vertical_name,
            operation="load_extension",
            start_time_ns=time.time_ns() - int(duration_ms * 1_000_000),
            end_time_ns=time.time_ns(),
            status="success" if success else "error",
            metadata={
                "extension_type": extension_type,
                "span_id": f"{vertical_name}:load_extension:{extension_type}",
            },
        )

        self._spans.append(span)
        self._emit_span(span)

        # Track slow extensions
        if success and duration_ms > 500:  # > 500ms
            logger.warning(
                "Slow extension load: '%s' for '%s' took %.2fms",
                extension_type,
                vertical_name,
                duration_ms,
            )

    def _emit_span(self, span: VerticalLoadSpan) -> None:
        """Emit span to OpenTelemetry if available.

        Args:
            span: Span to emit
        """
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import SERVICE_NAME

            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(
                f"vertical.{span.operation}",
                attributes={
                    "vertical.name": span.vertical_name,
                    "vertical.operation": span.operation,
                    "vertical.status": span.status,
                    "vertical.duration_ms": span.duration_ms or 0,
                    **span.metadata,
                },
            ) as otel_span:
                if span.is_error:
                    otel_span.set_status(trace.Status(trace.StatusCode.ERROR, span.error))
                elif span.status == TelemetryStatus.SKIPPED:
                    otel_span.set_status(trace.Status(trace.StatusCode.UNSET))

        except ImportError:
            # OpenTelemetry not installed - skip
            pass
        except Exception as e:
            logger.debug(f"Failed to emit OpenTelemetry span: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all recorded spans.

        Returns:
            Dict with metrics for operations, errors, slow operations
        """
        if not self._spans:
            return {}

        # Aggregate metrics
        operations: Dict[str, List[VerticalLoadSpan]] = {}
        for span in self._spans:
            key = f"{span.vertical_name}:{span.operation}"
            if key not in operations:
                operations[key] = []
            operations[key].append(span)

        # Calculate metrics
        total_ops = len(self._spans)
        success_ops = sum(1 for s in self._spans if s.is_success)
        error_ops = sum(1 for s in self._spans if s.is_error)

        # Calculate total duration
        total_duration_ms = sum(s.duration_ms or 0 for s in self._spans)
        avg_duration_ms = total_duration_ms / total_ops if total_ops > 0 else 0

        # Find slow operations
        slow_ops = [s for s in self._spans if (s.duration_ms or 0) > 1000]

        # Count by status
        by_status: Dict[str, int] = {}
        for span in self._spans:
            by_status[span.status] = by_status.get(span.status, 0) + 1

        return {
            "total_operations": total_ops,
            "successful": success_ops,
            "errors": error_ops,
            "success_rate": round(success_ops / total_ops, 4) if total_ops > 0 else 0.0,
            "avg_duration_ms": round(avg_duration_ms, 2),
            "total_duration_ms": round(total_duration_ms, 2),
            "slow_operations": len(slow_ops),
            "by_status": by_status,
        }

    def get_slow_operations(self, threshold_ms: float = 500.0) -> List[Dict[str, Any]]:
        """Get list of slow operations.

        Args:
            threshold_ms: Duration threshold for "slow" (default: 500ms)

        Returns:
            List of slow operation details
        """
        slow_ops = []
        for span in self._spans:
            if span.duration_ms and span.duration_ms > threshold_ms:
                slow_ops.append(
                    {
                        "vertical": span.vertical_name,
                        "operation": span.operation,
                        "duration_ms": span.duration_ms,
                        "status": span.status,
                    }
                )
        return slow_ops

    def clear(self) -> None:
        """Clear all recorded spans."""
        self._spans.clear()
        self._active_spans.clear()


# Global telemetry instance
_global_telemetry: Optional[VerticalLoadTelemetry] = None
_telemetry_lock = threading.RLock()


def get_telemetry() -> VerticalLoadTelemetry:
    """Get the global telemetry instance.

    Returns:
        VerticalLoadTelemetry singleton instance
    """
    global _global_telemetry
    with _telemetry_lock:
        if _global_telemetry is None:
            _global_telemetry = VerticalLoadTelemetry()
    return _global_telemetry


__all__ = [
    "VerticalLoadSpan",
    "vertical_load_span",
    "VerticalLoadTelemetry",
    "get_telemetry",
]
