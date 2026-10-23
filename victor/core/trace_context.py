"""Trace context propagation across core↔vertical boundary (OBS-2).

Provides a lightweight trace context that flows through vertical loading,
extension hook calls, and cross-boundary operations. Enables debugging
of errors that originate in verticals but surface in core.

Usage:
    from victor.core.trace_context import TraceContext, current_trace

    # Start a trace for a vertical operation
    with TraceContext.start("vertical.load", vertical="coding") as trace:
        result = load_vertical("coding")
        trace.add_event("loaded", status="ok")

    # Access current trace
    trace = current_trace()
    if trace:
        trace.add_event("extension.hook", hook="get_tools")
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# Context variable for propagating trace across async boundaries
_current_trace: contextvars.ContextVar[Optional["TraceContext"]] = contextvars.ContextVar(
    "_current_trace", default=None
)


@dataclass
class TraceEvent:
    """A single event within a trace span."""

    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceSpan:
    """A span within a trace — represents a unit of work."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[TraceEvent] = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is not None:
            return (self.end_time - self.start_time) * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                for e in self.events
            ],
        }


class TraceContext:
    """Trace context for cross-boundary operations.

    Carries a correlation ID and collects spans across the
    core↔vertical boundary. Uses Python contextvars for automatic
    propagation through async calls.
    """

    def __init__(self, trace_id: Optional[str] = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self._spans: List[TraceSpan] = []
        self._active_span: Optional[TraceSpan] = None

    @classmethod
    @contextmanager
    def start(
        cls,
        operation: str,
        trace_id: Optional[str] = None,
        **attributes: Any,
    ) -> Generator[TraceContext, None, None]:
        """Start a new trace context as a context manager.

        Sets the trace as the current context variable so child
        operations can access it via current_trace().

        Args:
            operation: Name of the operation being traced
            trace_id: Optional correlation ID (generated if not provided)
            **attributes: Additional attributes for the root span
        """
        parent = _current_trace.get()
        if parent is not None:
            # Nested — reuse parent trace, create child span
            ctx = parent
            span = ctx._start_span(operation, **attributes)
        else:
            # Root trace
            ctx = cls(trace_id=trace_id)
            span = ctx._start_span(operation, **attributes)

        token = _current_trace.set(ctx)
        try:
            yield ctx
        except Exception as e:
            span.status = "error"
            span.error = str(e)
            raise
        finally:
            span.end_time = time.time()
            _current_trace.reset(token)

    def _start_span(self, name: str, **attributes: Any) -> TraceSpan:
        """Start a new span within this trace."""
        parent_id = self._active_span.span_id if self._active_span else None
        span = TraceSpan(
            name=name,
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:12],
            parent_span_id=parent_id,
            attributes=attributes,
        )
        self._spans.append(span)
        self._active_span = span
        return span

    def add_event(self, name: str, **attributes: Any) -> None:
        """Add an event to the current active span."""
        if self._active_span:
            self._active_span.events.append(TraceEvent(name=name, attributes=attributes))

    def get_spans(self) -> List[TraceSpan]:
        """Get all spans in this trace."""
        return list(self._spans)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the trace context."""
        return {
            "trace_id": self.trace_id,
            "span_count": len(self._spans),
            "spans": [s.to_dict() for s in self._spans],
        }


def current_trace() -> Optional[TraceContext]:
    """Get the current trace context, or None if not in a trace."""
    return _current_trace.get()


def get_correlation_id() -> Optional[str]:
    """Get the current trace/correlation ID, or None."""
    trace = _current_trace.get()
    return trace.trace_id if trace else None
