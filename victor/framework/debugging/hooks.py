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

"""Debug hooks for StateGraph integration.

This module provides the DebugHook class which is the integration point
between StateGraph execution and the debugging system.

Key Classes:
    DebugHook: Hook for debugging workflow execution (injected into CompiledGraph)

Example:
    from victor.framework.debugging.hooks import DebugHook

    hook = DebugHook(session_id="debug-123", ...)

    # In CompiledGraph.invoke()
    await hook.before_node(node_id, state)

    # Execute node
    result = await node.execute(state)

    # After node
    await hook.after_node(node_id, state, error=None)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from victor.framework.debugging.breakpoints import (
    BreakpointManager,
    BreakpointPosition,
    WorkflowBreakpoint,
)
from victor.framework.debugging.execution import (
    ExecutionController,
    PauseContext,
)
from victor.framework.debugging.inspector import StateInspector

logger = logging.getLogger(__name__)


class DebugHook:
    """Hook for debugging workflow execution (injected into CompiledGraph).

    This class is the integration point between StateGraph execution
    and the debugging system. It checks breakpoints and controls
    execution pause/resume.

    Design:
        - Non-invasive: Only active when debugging is enabled
        - Minimal overhead: Fast path when no breakpoints
        - Async-safe: Properly handles async pause/resume

    Attributes:
        session_id: Debug session ID
        _breakpoint_mgr: BreakpointManager instance
        _execution_ctrl: ExecutionController instance
        _inspector: StateInspector instance
        _event_bus: EventBus for events
        _enabled: Whether debugging is active

    Example:
        hook = DebugHook(session_id="debug-123", ...)

        # In CompiledGraph.invoke()
        await hook.before_node(node_id, state)

        # Execute node
        result = await node.execute(state)

        # After node
        await hook.after_node(node_id, state, error=None)
    """

    def __init__(
        self,
        session_id: str,
        breakpoint_mgr: BreakpointManager,
        execution_ctrl: ExecutionController,
        inspector: StateInspector,
        event_bus: Any,
    ) -> None:
        """Initialize debug hook.

        Args:
            session_id: Debug session ID
            breakpoint_mgr: BreakpointManager instance
            execution_ctrl: ExecutionController instance
            inspector: StateInspector instance
            event_bus: EventBus for events
        """
        self.session_id = session_id
        self._breakpoint_mgr = breakpoint_mgr
        self._execution_ctrl = execution_ctrl
        self._inspector = inspector
        self._event_bus = event_bus
        self._enabled = True

    def disable(self) -> None:
        """Disable debug hook (no breakpoint checks)."""
        self._enabled = False

    def enable(self) -> None:
        """Enable debug hook."""
        self._enabled = True

    async def before_node(self, node_id: str, state: Dict[str, Any]) -> None:
        """Called before node execution.

        Checks for BEFORE position breakpoints and pauses if hit.

        Args:
            node_id: Node about to execute
            state: Current workflow state

        Raises:
            asyncio.CancelledError: If execution is cancelled during pause
        """
        if not self._enabled:
            return

        # Evaluate breakpoints
        breakpoints = self._breakpoint_mgr.evaluate_breakpoints(
            state=state,
            node_id=node_id,
            position=BreakpointPosition.BEFORE,
        )

        # Check if should pause
        if self._execution_ctrl.should_pause(node_id, state, breakpoints):
            # Capture snapshot before pause
            self._inspector.capture_snapshot(state, node_id)

            # Pause execution
            context = await self._execution_ctrl.pause(
                node_id=node_id,
                state=state,
                position=BreakpointPosition.BEFORE,
                breakpoints=breakpoints,
            )

            # Emit paused event
            self._emit_paused(context)

    async def after_node(
        self, node_id: str, state: Dict[str, Any], error: Optional[Exception] = None
    ) -> None:
        """Called after node execution.

        Checks for AFTER/ON_ERROR position breakpoints and pauses if hit.

        Args:
            node_id: Node that executed
            state: Current workflow state
            error: Exception if one occurred
        """
        if not self._enabled:
            return

        # Determine position
        position = BreakpointPosition.ON_ERROR if error else BreakpointPosition.AFTER

        # Evaluate breakpoints
        breakpoints = self._breakpoint_mgr.evaluate_breakpoints(
            state=state,
            node_id=node_id,
            position=position,
            error=error,
        )

        # Check if should pause
        if self._execution_ctrl.should_pause(node_id, state, breakpoints):
            # Capture snapshot after pause
            self._inspector.capture_snapshot(state, node_id)

            # Pause execution
            context = await self._execution_ctrl.pause(
                node_id=node_id,
                state=state,
                position=position,
                breakpoints=breakpoints,
                error=error,
            )

            # Emit paused event
            self._emit_paused(context)

    def _emit_paused(self, context: PauseContext) -> None:
        """Emit paused event.

        Args:
            context: Pause context
        """
        try:
            import asyncio

            if asyncio.iscoroutinefunction(self._event_bus.emit):
                asyncio.create_task(
                    self._event_bus.emit(
                        topic="debug.paused",
                        data={
                            "session_id": self.session_id,
                            "node_id": context.node_id,
                            "position": context.position.value,
                            "breakpoint_ids": context.breakpoint_ids,
                        },
                    )
                )
            else:
                self._event_bus.emit(
                    topic="debug.paused",
                    data={
                        "session_id": self.session_id,
                        "node_id": context.node_id,
                        "position": context.position.value,
                        "breakpoint_ids": context.breakpoint_ids,
                    },
                )
        except Exception as e:
            # Event emission failures shouldn't break debugging
            logger.debug(f"Failed to emit paused event: {e}")
