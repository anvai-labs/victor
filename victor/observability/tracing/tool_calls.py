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

"""Tool call tracing for debugging and observability.

This module provides tool call tracing linked to execution spans.
Tracks tool invocations with performance metrics.

SOLID Principles:
- SRP: Tool call tracing only
- OCP: Extensible via event emission
- DIP: Depends on EventBus abstraction

Usage:
    from victor.observability.tracing import ToolCallTracer
    from victor.core.events import MessagingEvent, ObservabilityBus, get_observability_bus

    event_bus = EventBus()
    tracer = ToolCallTracer(event_bus)

    # Record tool call
    call_id = tracer.record_call("read_file", {"file": "test.py"}, span_id="span-123")

    # Complete tool call
    tracer.complete_call(call_id, result="file content")

    # Fail tool call
    tracer.fail_call(call_id, error="File not found")

    # Get call history
    calls = tracer.get_calls(tool_name="read_file")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.core.events import MessagingEvent, ObservabilityBus, get_observability_bus

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """Record of a tool call.

    Links to ExecutionSpan for full execution context.

    Attributes:
        call_id: Unique call identifier
        parent_span_id: Parent execution span ID
        tool_name: Name of the tool being called
        arguments: Tool arguments
        result: Optional result data
        error: Optional error message if call failed
        start_time: Unix timestamp when call started
        end_time: Unix timestamp when call ended (None if running)
        duration_ms: Call duration in milliseconds (None if running)
    """

    call_id: str
    parent_span_id: str
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None

    @property
    def is_complete(self) -> bool:
        """Check if call is complete.

        Returns:
            True if call has completed (success or failure)
        """
        return self.end_time is not None

    @property
    def is_successful(self) -> bool:
        """Check if call was successful.

        Returns:
            True if call completed without error
        """
        return self.is_complete and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of tool call
        """
        return {
            "call_id": self.call_id,
            "parent_span_id": self.parent_span_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "is_complete": self.is_complete,
            "is_successful": self.is_successful,
        }


class ToolCallTracer:
    """Traces tool calls with execution span linkage.

    Records tool invocations linked to execution spans for debugging.
    Integrates with EventBus for real-time monitoring.

    SOLID: SRP (tool call tracing only), DIP (depends on EventBus)

    Attributes:
        _event_bus: EventBus instance for emitting events
        _calls: Dictionary mapping call_id to ToolCallRecord

    Example:
        >>> event_bus = EventBus()
        >>> tracer = ToolCallTracer(event_bus)
        >>>
        >>> # Record tool call
        >>> call_id = tracer.record_call(
        ...     tool_name="read_file",
        ...     arguments={"file": "test.py"},
        ...     parent_span_id="span-123"
        ... )
        >>>
        >>> # Complete successfully
        >>> tracer.complete_call(call_id, result="file content")
        >>>
        >>> # Get call history
        >>> calls = tracer.get_calls(tool_name="read_file")
    """

    def __init__(self, event_bus: Any):
        """Initialize the tool call tracer.

        Args:
            event_bus: EventBus instance for emitting tool call events
        """
        self._event_bus = event_bus
        self._calls: Dict[str, ToolCallRecord] = {}

        logger.info("ToolCallTracer initialized")

    def record_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        parent_span_id: str,
    ) -> str:
        """Record a tool call.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments
            parent_span_id: Parent execution span ID

        Returns:
            New call ID
        """
        call_id = f"tc-{len(self._calls)}-{time.time()}"

        record = ToolCallRecord(
            call_id=call_id,
            parent_span_id=parent_span_id,
            tool_name=tool_name,
            arguments=arguments,
        )

        self._calls[call_id] = record

        # Publish tool call started event
        try:
            self._event_bus.emit(
                topic="tool.call.started",
                data={
                    "call_id": call_id,
                    "parent_span_id": parent_span_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "start_time": record.start_time,
                    "category": "tool",
                },
            )
        except Exception as e:
            logger.warning(f"Failed to publish tool_call_started event: {e}")

        logger.debug(f"Tool call started: {call_id} (tool={tool_name}, span={parent_span_id})")

        return call_id

    def complete_call(self, call_id: str, result: Any) -> None:
        """Complete a tool call successfully.

        Args:
            call_id: Call ID to complete
            result: Result data from tool call
        """
        if call_id not in self._calls:
            logger.warning(f"Tool call not found: {call_id}")
            return

        record = self._calls[call_id]
        record.end_time = time.time()
        record.duration_ms = (record.end_time - record.start_time) * 1000
        record.result = result

        # Publish tool call completed event
        try:
            self._event_bus.emit(
                topic="tool.call.completed",
                data={
                    "call_id": call_id,
                    "parent_span_id": record.parent_span_id,
                    "tool_name": record.tool_name,
                    "result": str(result)[:500],
                    "duration_ms": record.duration_ms,
                    "category": "tool",
                },
            )
        except Exception as e:
            logger.warning(f"Failed to publish tool_call_completed event: {e}")

        logger.debug(f"Tool call completed: {call_id} (duration={record.duration_ms:.2f}ms)")

    def fail_call(self, call_id: str, error: str) -> None:
        """Mark a tool call as failed.

        Args:
            call_id: Call ID to fail
            error: Error message
        """
        if call_id not in self._calls:
            logger.warning(f"Tool call not found: {call_id}")
            return

        record = self._calls[call_id]
        record.end_time = time.time()
        record.duration_ms = (record.end_time - record.start_time) * 1000
        record.error = error

        # Publish tool call failed event
        try:
            self._event_bus.emit(
                topic="tool.call.failed",
                data={
                    "call_id": call_id,
                    "parent_span_id": record.parent_span_id,
                    "tool_name": record.tool_name,
                    "error": error,
                    "duration_ms": record.duration_ms,
                    "category": "tool",
                },
            )
        except Exception as e:
            logger.warning(f"Failed to publish tool_call_failed event: {e}")

        logger.debug(f"Tool call failed: {call_id} (error={error})")

    def get_call(self, call_id: str) -> Optional[ToolCallRecord]:
        """Get a tool call by ID.

        Args:
            call_id: Call ID to retrieve

        Returns:
            ToolCallRecord if found, None otherwise
        """
        return self._calls.get(call_id)

    def get_calls(
        self,
        tool_name: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ToolCallRecord]:
        """Get tool calls with optional filters.

        Args:
            tool_name: Filter by tool name (optional)
            parent_span_id: Filter by parent span ID (optional)
            limit: Maximum number of calls to return

        Returns:
            List of ToolCallRecord objects matching filters
        """
        calls = list(self._calls.values())

        if tool_name:
            calls = [c for c in calls if c.tool_name == tool_name]

        if parent_span_id:
            calls = [c for c in calls if c.parent_span_id == parent_span_id]

        # Sort by start time (most recent first) and limit
        calls.sort(key=lambda c: c.start_time, reverse=True)

        return calls[:limit]

    def get_calls_by_span(self, parent_span_id: str) -> List[ToolCallRecord]:
        """Get all tool calls for a specific execution span.

        Args:
            parent_span_id: Parent span ID

        Returns:
            List of tool calls for the span
        """
        return [c for c in self._calls.values() if c.parent_span_id == parent_span_id]

    def get_failed_calls(self, limit: int = 100) -> List[ToolCallRecord]:
        """Get failed tool calls.

        Args:
            limit: Maximum number of calls to return

        Returns:
            List of failed ToolCallRecord objects
        """
        calls = [c for c in self._calls.values() if not c.is_successful]
        calls.sort(key=lambda c: c.start_time, reverse=True)
        return calls[:limit]

    def get_all_calls(self) -> List[ToolCallRecord]:
        """Get all tool calls.

        Returns:
            List of all ToolCallRecord objects
        """
        return list(self._calls.values())

    def get_call_count(self) -> int:
        """Get total number of tool calls.

        Returns:
            Number of tool calls
        """
        return len(self._calls)

    def clear_calls(self) -> None:
        """Clear all recorded tool calls.

        Useful for long-running applications to prevent memory bloat.
        """
        count = len(self._calls)
        self._calls.clear()
        logger.info(f"Cleared {count} tool calls from tracer")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about recorded tool calls.

        Returns:
            Dictionary with tool call statistics
        """
        if not self._calls:
            return {"total": 0}

        # Count by tool name
        by_tool: Dict[str, int] = {}
        for call in self._calls.values():
            by_tool[call.tool_name] = by_tool.get(call.tool_name, 0) + 1

        # Count by status
        successful = len([c for c in self._calls.values() if c.is_successful])
        failed = len([c for c in self._calls.values() if c.is_complete and not c.is_successful])
        running = len([c for c in self._calls.values() if not c.is_complete])

        # Calculate durations for completed calls
        completed_calls = [c for c in self._calls.values() if c.duration_ms is not None]
        if completed_calls:
            durations = [c.duration_ms for c in completed_calls if c.duration_ms is not None]
            if durations:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                min_duration = min(durations)
            else:
                avg_duration = max_duration = min_duration = 0.0
        else:
            avg_duration = max_duration = min_duration = 0.0

        return {
            "total": len(self._calls),
            "successful": successful,
            "failed": failed,
            "running": running,
            "by_tool": by_tool,
            "avg_duration_ms": avg_duration,
            "max_duration_ms": max_duration,
            "min_duration_ms": min_duration,
        }


__all__ = [
    "ToolCallRecord",
    "ToolCallTracer",
]
