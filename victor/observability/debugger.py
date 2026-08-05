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

"""Unified debugging interface for agents.

This module provides AgentDebugger, a unified facade that combines
multiple tracers for comprehensive debugging and observability.

SOLID Principles:
- SRP: Debugging orchestration only
- OCP: Extensible via new tracers
- LSP: Substitutable with other debuggers
- ISP: Focused debugging interface
- DIP: Depends on tracer abstractions

Usage:
    from victor.observability.debugger import AgentDebugger
    from victor.core.events import ObservabilityBus as EventBus

    event_bus = EventBus()
    debugger = AgentDebugger(event_bus)

    # Get execution trace
    trace = debugger.get_execution_trace(agent_id="agent-1")

    # Get tool calls
    calls = debugger.get_tool_calls(tool_name="read_file")

    # Get state transitions
    transitions = debugger.get_state_transitions(scope="workflow")

    # Get performance summary
    summary = debugger.get_performance_summary()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor.observability.tracing.execution import ExecutionTracer
from victor.observability.tracing.tool_calls import ToolCallTracer
from victor.state.tracer import StateTracer

logger = logging.getLogger(__name__)


class AgentDebugger:
    """Unified debugging interface for agents.

    Facade pattern over specialized tracers. Provides simple API
    for debugging agent execution, tool calls, and state transitions.

    SOLID: SRP (orchestration only), DIP (depends on tracer abstractions)

    Attributes:
        _execution: ExecutionTracer for execution flow
        _tool_calls: ToolCallTracer for tool calls
        _state: StateTracer for state transitions

    Example:
        >>> from victor.core.events import ObservabilityBus as EventBus
        >>> from victor.observability.debugger import AgentDebugger
        >>>
        >>> event_bus = EventBus()
        >>> debugger = AgentDebugger(event_bus)
        >>>
        >>> # Get execution trace
        >>> trace = debugger.get_execution_trace()
        >>>
        >>> # Get tool calls
        >>> calls = debugger.get_tool_calls()
        >>>
        >>> # Get state transitions
        >>> transitions = debugger.get_state_transitions()
        >>>
        >>> # Get performance summary
        >>> summary = debugger.get_performance_summary()
    """

    def __init__(
        self,
        event_bus: Any,
        execution_tracer: Optional[ExecutionTracer] = None,
        tool_call_tracer: Optional[ToolCallTracer] = None,
        state_tracer: Optional[StateTracer] = None,
    ):
        """Initialize the agent debugger.

        Args:
            event_bus: EventBus instance for creating tracers if not provided
            execution_tracer: Optional ExecutionTracer instance
            tool_call_tracer: Optional ToolCallTracer instance
            state_tracer: Optional StateTracer instance
        """
        # Create tracers if not provided
        self._execution = execution_tracer or ExecutionTracer(event_bus)
        self._tool_calls = tool_call_tracer or ToolCallTracer(event_bus)
        self._state = state_tracer or StateTracer(event_bus)

        logger.info("AgentDebugger initialized")

    # ========================================================================
    # Execution Trace Methods
    # ========================================================================

    def get_execution_trace(
        self, agent_id: Optional[str] = None, span_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get execution trace for an agent.

        Args:
            agent_id: Filter by agent ID (optional)
            span_id: Root span ID (optional, uses first root if None)

        Returns:
            Execution span tree as nested dictionary
        """
        if span_id:
            return self._execution.get_span_tree(span_id)

        if agent_id:
            # Get spans for agent and build tree from first span
            agent_spans = self._execution.get_spans_by_agent(agent_id)
            if agent_spans:
                return self._execution.get_span_tree(agent_spans[0].id)

        # Get default tree
        return self._execution.get_span_tree()

    def get_execution_spans(
        self, agent_id: Optional[str] = None, span_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get execution spans as list.

        Args:
            agent_id: Filter by agent ID (optional)
            span_type: Filter by span type (optional)

        Returns:
            List of span dictionaries
        """
        spans = self._execution.get_all_spans()

        if agent_id:
            spans = [s for s in spans if s.agent_id == agent_id]

        if span_type:
            spans = [s for s in spans if s.span_type == span_type]

        return [span.to_dict() for span in spans]

    def get_active_spans(self) -> List[Dict[str, Any]]:
        """Get all currently running spans.

        Returns:
            List of active span dictionaries
        """
        spans = self._execution.get_active_spans()
        return [span.to_dict() for span in spans]

    # ========================================================================
    # Tool Call Methods
    # ========================================================================

    def get_tool_calls(
        self,
        tool_name: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get tool call history.

        Args:
            tool_name: Filter by tool name (optional)
            parent_span_id: Filter by parent span ID (optional)
            limit: Maximum number of calls to return

        Returns:
            List of tool call dictionaries
        """
        calls = self._tool_calls.get_calls(
            tool_name=tool_name, parent_span_id=parent_span_id, limit=limit
        )

        return [
            {
                "call_id": c.call_id,
                "parent_span_id": c.parent_span_id,
                "tool_name": c.tool_name,
                "arguments": c.arguments,
                "result": str(c.result)[:200] if c.result else None,
                "error": c.error,
                "duration_ms": c.duration_ms,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "is_complete": c.is_complete,
                "is_successful": c.is_successful,
            }
            for c in calls
        ]

    def get_failed_tool_calls(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get failed tool calls.

        Args:
            limit: Maximum number of calls to return

        Returns:
            List of failed tool call dictionaries
        """
        calls = self._tool_calls.get_failed_calls(limit)

        return [
            {
                "call_id": c.call_id,
                "parent_span_id": c.parent_span_id,
                "tool_name": c.tool_name,
                "arguments": c.arguments,
                "error": c.error,
                "duration_ms": c.duration_ms,
            }
            for c in calls
        ]

    def get_tool_calls_by_span(self, parent_span_id: str) -> List[Dict[str, Any]]:
        """Get all tool calls for a specific execution span.

        Args:
            parent_span_id: Parent span ID

        Returns:
            List of tool call dictionaries for the span
        """
        calls = self._tool_calls.get_calls_by_span(parent_span_id)

        return [call.to_dict() for call in calls]

    # ========================================================================
    # State Transition Methods
    # ========================================================================

    def get_state_transitions(
        self,
        scope: Optional[str] = None,
        key: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get state transition history.

        Args:
            scope: Filter by scope (optional)
            key: Filter by key (optional)
            limit: Maximum number of transitions to return

        Returns:
            List of state transition dictionaries
        """
        transitions = self._state.get_history(scope=scope, key=key, limit=limit)

        return [
            {
                "scope": t.scope,
                "key": t.key,
                "old_value": str(t.old_value)[:100] if t.old_value else None,
                "new_value": str(t.new_value)[:100] if t.new_value else None,
                "timestamp": t.timestamp,
                "metadata": t.metadata,
            }
            for t in transitions
        ]

    # ========================================================================
    # Performance Summary Methods
    # ========================================================================

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all tracers.

        Aggregates metrics from execution, tool calls, and state.

        Returns:
            Dictionary with performance metrics
        """
        # Get execution stats
        exec_stats = self._execution.get_statistics()

        # Get tool call stats
        tool_stats = self._tool_calls.get_statistics()

        # Get state stats
        state_stats = self._state.get_statistics()

        return {
            "execution": exec_stats,
            "tool_calls": tool_stats,
            "state": state_stats,
            "summary": {
                "total_spans": exec_stats.get("total", 0),
                "active_spans": exec_stats.get("active", 0),
                "total_tool_calls": tool_stats.get("total", 0),
                "failed_tool_calls": tool_stats.get("failed", 0),
                "total_state_transitions": state_stats.get("total", 0),
            },
        }

    def get_slow_tool_calls(
        self, threshold_ms: float = 1000.0, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get slow tool calls above threshold.

        Args:
            threshold_ms: Duration threshold in milliseconds
            limit: Maximum number of calls to return

        Returns:
            List of slow tool call dictionaries
        """
        all_calls = self._tool_calls.get_all_calls()

        # Filter by duration
        slow_calls = [
            c for c in all_calls if c.duration_ms is not None and c.duration_ms > threshold_ms
        ]

        # Sort by duration (slowest first) and limit
        slow_calls.sort(key=lambda c: c.duration_ms or 0, reverse=True)

        return [
            {
                "call_id": c.call_id,
                "parent_span_id": c.parent_span_id,
                "tool_name": c.tool_name,
                "arguments": c.arguments,
                "duration_ms": c.duration_ms,
            }
            for c in slow_calls[:limit]
        ]

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def clear_all(self) -> None:
        """Clear all tracing data.

        Useful for resetting between test runs or debugging sessions.
        """
        self._execution.clear_spans()
        self._tool_calls.clear_calls()
        self._state.clear_history()

        logger.info("Cleared all debugging data")

    def get_summary(self) -> Dict[str, Any]:
        """Get a brief summary of all debugging data.

        Returns:
            Dictionary with summary counts
        """
        perf = self.get_performance_summary()

        return {
            "spans": perf["execution"]["total"],
            "active_spans": perf["execution"]["active"],
            "tool_calls": perf["tool_calls"]["total"],
            "failed_tool_calls": perf["tool_calls"]["failed"],
            "state_transitions": perf["state"]["total"],
        }


__all__ = ["AgentDebugger"]
