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

"""Execution tracing with span trees.

This module provides execution tracing for debugging and observability.
Tracks execution spans with parent-child relationships for visualization.

SOLID Principles:
- SRP: Execution tracing only
- OCP: Extensible via event emission
- DIP: Depends on EventBus abstraction

Usage:
    from victor.observability.tracing import ExecutionTracer, SpanStatus
    from victor.core.events import MessagingEvent, ObservabilityBus, get_observability_bus

    event_bus = EventBus()
    tracer = ExecutionTracer(event_bus)

    # Start spans
    root_span = tracer.start_span("agent-1", "agent")
    child_span = tracer.start_span("agent-1", "tool", parent_id=root_span)

    # End spans
    tracer.end_span(child_span, status=SpanStatus.SUCCESS)
    tracer.end_span(root_span, status=SpanStatus.SUCCESS)

    # Get span tree
    tree = tracer.get_span_tree()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from victor.core.events import MessagingEvent, ObservabilityBus, get_observability_bus

logger = logging.getLogger(__name__)


class SpanStatus(str, Enum):
    """Status of an execution span."""

    RUNNING = "running"  # Span is currently active
    SUCCESS = "success"  # Span completed successfully
    ERROR = "error"  # Span failed with error


@dataclass
class ExecutionSpan:
    """Record of an execution span.

    Represents a unit of work in the execution flow (agent, tool, LLM call).

    Attributes:
        id: Unique span identifier
        parent_id: Parent span ID (None for root spans)
        agent_id: ID of the agent executing this span
        span_type: Type of span ("agent", "tool", "llm_call")
        start_time: Unix timestamp when span started
        end_time: Unix timestamp when span ended (None if running)
        status: Current status (running, success, error)
        metadata: Optional metadata about the span
        children: List of child span IDs
    """

    id: str
    parent_id: Optional[str]
    agent_id: str
    span_type: str
    start_time: float
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.RUNNING
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds.

        Returns:
            Duration in milliseconds, or None if span is still running
        """
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of span
        """
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "agent_id": self.agent_id,
            "span_type": self.span_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "metadata": self.metadata,
            "children": self.children,
        }


class ExecutionTracer:
    """Traces execution flow with parent-child relationships.

    Tracks execution spans forming a tree structure. Integrates with
    EventBus for real-time monitoring.

    SOLID: SRP (execution tracing only), DIP (depends on EventBus)

    Attributes:
        _event_bus: EventBus instance for emitting events
        _spans: Dictionary mapping span_id to ExecutionSpan
        _root_spans: List of root span IDs (spans with no parent)

    Example:
        >>> event_bus = EventBus()
        >>> tracer = ExecutionTracer(event_bus)
        >>>
        >>> # Start root span
        >>> root_id = tracer.start_span("agent-1", "agent")
        >>>
        >>> # Start child span
        >>> child_id = tracer.start_span("agent-1", "tool", parent_id=root_id)
        >>>
        >>> # End child span
        >>> tracer.end_span(child_id, status="success", result="output")
        >>>
        >>> # End root span
        >>> tracer.end_span(root_id, status="success")
        >>>
        >>> # Get span tree
        >>> tree = tracer.get_span_tree()
    """

    def __init__(self, event_bus: Any):
        """Initialize the execution tracer.

        Args:
            event_bus: EventBus instance for emitting span events
        """
        self._event_bus = event_bus
        self._spans: Dict[str, ExecutionSpan] = {}
        self._root_spans: List[str] = []

        logger.info("ExecutionTracer initialized")

    def start_span(
        self,
        agent_id: str,
        span_type: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new execution span.

        Args:
            agent_id: ID of the agent executing this span
            span_type: Type of span ("agent", "tool", "llm_call")
            parent_id: Optional parent span ID
            metadata: Optional metadata about the span

        Returns:
            New span ID
        """
        span_id = f"span-{len(self._spans)}-{time.time()}"

        span = ExecutionSpan(
            id=span_id,
            parent_id=parent_id,
            agent_id=agent_id,
            span_type=span_type,
            start_time=time.time(),
            metadata=metadata or {},
        )

        self._spans[span_id] = span

        # Link to parent
        if parent_id and parent_id in self._spans:
            self._spans[parent_id].children.append(span_id)
        else:
            # No parent or parent not found - this is a root span
            self._root_spans.append(span_id)

        # Publish span started event
        try:
            self._event_bus.emit(
                topic="lifecycle.span.started",
                data={
                    "span_id": span_id,
                    "parent_id": parent_id,
                    "agent_id": agent_id,
                    "span_type": span_type,
                    "start_time": span.start_time,
                    "category": "lifecycle",
                    **span.metadata,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to publish span_started event: {e}")

        logger.debug(
            f"Span started: {span_id} (type={span_type}, agent={agent_id}, parent={parent_id})"
        )

        return span_id

    def end_span(
        self,
        span_id: str,
        status: SpanStatus = SpanStatus.SUCCESS,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        """End an execution span.

        Args:
            span_id: Span ID to end
            status: Final status (SUCCESS, ERROR)
            result: Optional result data
            error: Optional error message if status is ERROR
        """
        if span_id not in self._spans:
            logger.warning(f"Span not found: {span_id}")
            return

        span = self._spans[span_id]
        span.end_time = time.time()
        span.status = status

        if result is not None:
            # Truncate result to avoid huge payloads
            result_str = str(result)[:500]
            span.metadata["result"] = result_str

        if error:
            span.metadata["error"] = error

        # Publish span ended event
        try:
            self._event_bus.emit(
                topic="lifecycle.span.ended",
                data={
                    "span_id": span_id,
                    "parent_id": span.parent_id,
                    "agent_id": span.agent_id,
                    "span_type": span.span_type,
                    "status": status,
                    "duration_ms": span.duration_ms,
                    "result": span.metadata.get("result"),
                    "error": error,
                    "category": "lifecycle",
                },
            )
        except Exception as e:
            logger.warning(f"Failed to publish span_ended event: {e}")

        logger.debug(f"Span ended: {span_id} (status={status}, duration={span.duration_ms:.2f}ms)")

    def get_span(self, span_id: str) -> Optional[ExecutionSpan]:
        """Get a span by ID.

        Args:
            span_id: Span ID to retrieve

        Returns:
            ExecutionSpan if found, None otherwise
        """
        return self._spans.get(span_id)

    def get_span_tree(self, root_id: Optional[str] = None) -> Dict[str, Any]:
        """Get span tree as nested dictionary.

        Args:
            root_id: Root span ID. If None, uses first root span.

        Returns:
            Nested dictionary representing span tree
        """
        if not self._spans:
            return {}

        if not root_id:
            if not self._root_spans:
                return {}
            root_id = self._root_spans[0]

        return self._build_tree(root_id)

    def _build_tree(self, span_id: str) -> Dict[str, Any]:
        """Build tree recursively from a span.

        Args:
            span_id: Root span ID

        Returns:
            Nested dictionary representing span tree
        """
        span = self._spans.get(span_id)
        if not span:
            return {}

        tree = span.to_dict()
        tree["children"] = [self._build_tree(child_id) for child_id in span.children]

        return tree

    def get_all_spans(self) -> List[ExecutionSpan]:
        """Get all recorded spans.

        Returns:
            List of all ExecutionSpan objects
        """
        return list(self._spans.values())

    def get_spans_by_agent(self, agent_id: str) -> List[ExecutionSpan]:
        """Get all spans for a specific agent.

        Args:
            agent_id: Agent ID to filter by

        Returns:
            List of spans for the agent
        """
        return [span for span in self._spans.values() if span.agent_id == agent_id]

    def get_spans_by_type(self, span_type: str) -> List[ExecutionSpan]:
        """Get all spans of a specific type.

        Args:
            span_type: Span type to filter by

        Returns:
            List of spans of the specified type
        """
        return [span for span in self._spans.values() if span.span_type == span_type]

    def get_active_spans(self) -> List[ExecutionSpan]:
        """Get all currently running spans.

        Returns:
            List of spans with status "running"
        """
        return [span for span in self._spans.values() if span.status == SpanStatus.RUNNING]

    def get_span_count(self) -> int:
        """Get total number of spans.

        Returns:
            Number of spans
        """
        return len(self._spans)

    def clear_spans(self) -> None:
        """Clear all recorded spans.

        Useful for long-running applications to prevent memory bloat.
        """
        count = len(self._spans)
        self._spans.clear()
        self._root_spans.clear()
        logger.info(f"Cleared {count} spans from tracer")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about recorded spans.

        Returns:
            Dictionary with span statistics
        """
        if not self._spans:
            return {"total": 0}

        # Count by type
        by_type: Dict[str, int] = {}
        for span in self._spans.values():
            by_type[span.span_type] = by_type.get(span.span_type, 0) + 1

        # Count by status
        by_status: Dict[str, int] = {}
        for span in self._spans.values():
            by_status[span.status] = by_status.get(span.status, 0) + 1

        # Calculate durations for completed spans
        completed_spans = [s for s in self._spans.values() if s.end_time is not None]
        if completed_spans:
            durations = [s.duration_ms for s in completed_spans if s.duration_ms is not None]
            if durations:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                min_duration = min(durations)
            else:
                avg_duration = max_duration = min_duration = 0.0
        else:
            avg_duration = max_duration = min_duration = 0.0

        return {
            "total": len(self._spans),
            "active": len(self.get_active_spans()),
            "completed": len(completed_spans),
            "by_type": by_type,
            "by_status": by_status,
            "avg_duration_ms": avg_duration,
            "max_duration_ms": max_duration,
            "min_duration_ms": min_duration,
        }


__all__ = [
    "ExecutionSpan",
    "ExecutionTracer",
]
