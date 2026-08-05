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

"""Execution control for workflow debugging.

This module provides pause/resume/step execution control for debugging
StateGraph workflows. Uses asyncio primitives for coordination.

Key Classes:
    ExecutionState: State of debug execution (RUNNING, PAUSED, STEPPING, TERMINATED)
    StepMode: Stepping modes (STEP_OVER, STEP_INTO, STEP_OUT)
    PauseContext: Context when execution is paused
    ExecutionController: Controls execution pause/resume (SRP)

Example:
    from victor.framework.debugging.execution import (
        ExecutionController,
        ExecutionState,
    )

    controller = ExecutionController(session_id="debug-123")

    # In workflow execution (DebugHook)
    if controller.should_pause(node_id, state):
        context = await controller.pause(node_id, state)
        # Execution waits here until resume

    # From debug client
    await controller.continue_execution()
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from victor.framework.debugging.breakpoints import (
    BreakpointPosition,
    WorkflowBreakpoint,
)


class ExecutionState(Enum):
    """State of debug execution.

    Attributes:
        RUNNING: Normal execution
        PAUSED: Paused at breakpoint
        STEPPING: Stepping through execution
        TERMINATED: Execution finished
    """

    RUNNING = "running"
    PAUSED = "paused"
    STEPPING = "stepping"
    TERMINATED = "terminated"


class StepMode(Enum):
    """Stepping modes.

    Attributes:
        STEP_OVER: Execute next node, don't enter sub-workflows
        STEP_INTO: Next node, enter sub-workflows
        STEP_OUT: Complete current sub-workflow
    """

    STEP_OVER = "step_over"
    STEP_INTO = "step_into"
    STEP_OUT = "step_out"


@dataclass
class PauseContext:
    """Context when execution is paused.

    Captures the state at breakpoint for inspection and resumption.

    Attributes:
        session_id: Debug session ID
        node_id: Current node ID
        position: Position relative to node execution
        state: Current workflow state
        breakpoint_ids: Breakpoints that were hit
        timestamp: Pause timestamp
        error: Exception if one occurred
        metadata: Additional metadata

    Example:
        context = PauseContext(
            session_id="debug-123",
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
            state={"task": "test", "errors": 0},
            breakpoint_ids=["bp-1"],
            timestamp=time.time()
        )
    """

    session_id: str
    node_id: str
    position: BreakpointPosition
    state: Dict[str, Any]
    breakpoint_ids: List[str]
    timestamp: float
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of pause context
        """
        return {
            "session_id": self.session_id,
            "node_id": self.node_id,
            "position": self.position.value,
            "state_keys": (
                list(self.state.to_dict().keys())
                if hasattr(self.state, "to_dict")
                else list(self.state.keys())
            ),
            "breakpoint_ids": self.breakpoint_ids,
            "timestamp": self.timestamp,
            "error": str(self.error) if self.error else None,
            "metadata": self.metadata,
        }


class ExecutionController:
    """Controls execution pause/resume for debugging (SRP).

    Manages the async coordination between workflow execution
    and debug commands (continue, step, pause).

    Thread Safety:
        This class uses asyncio primitives and is NOT thread-safe.
        Each debug session must have its own ExecutionController.

    Attributes:
        session_id: Unique debug session identifier
        state: Current execution state
        pause_context: Context when paused
        step_mode: Current stepping mode
        resume_event: Event for signaling resume

    Example:
        controller = ExecutionController(session_id="debug-123")

        # In workflow execution (DebugHook)
        if controller.should_pause(node_id, state):
            context = await controller.pause(node_id, state)
            # Execution waits here until resume

        # From debug client
        await controller.continue_execution()
    """

    def __init__(self, session_id: str) -> None:
        """Initialize execution controller.

        Args:
            session_id: Unique debug session ID
        """
        self.session_id = session_id
        self._state = ExecutionState.RUNNING
        self._pause_context: Optional[PauseContext] = None
        self._step_mode: Optional[StepMode] = None
        self._resume_event = asyncio.Event()
        self._pause_lock = asyncio.Lock()

    @property
    def state(self) -> ExecutionState:
        """Get current execution state.

        Returns:
            Current ExecutionState
        """
        return self._state

    @property
    def pause_context(self) -> Optional[PauseContext]:
        """Get current pause context.

        Returns:
            PauseContext if paused, None otherwise
        """
        return self._pause_context

    def should_pause(
        self,
        node_id: str,
        state: Dict[str, Any],
        breakpoints: List[WorkflowBreakpoint],
    ) -> bool:
        """Check if execution should pause.

        Called by DebugHook before/after node execution.

        Args:
            node_id: Current node ID
            state: Current workflow state
            breakpoints: Breakpoints that were hit

        Returns:
            True if execution should pause
        """
        # Pause if breakpoints hit
        if breakpoints:
            return True

        # Pause if in stepping mode
        if self._step_mode:
            return True

        # Pause if manually requested
        if self._state == ExecutionState.PAUSED:
            return True

        return False

    async def pause(
        self,
        node_id: str,
        state: Dict[str, Any],
        position: BreakpointPosition,
        breakpoints: List[WorkflowBreakpoint],
        error: Optional[Exception] = None,
    ) -> PauseContext:
        """Pause execution at breakpoint.

        Blocks until resume is called.

        Args:
            node_id: Current node ID
            state: Current workflow state
            position: Position relative to node
            breakpoints: Breakpoints that were hit
            error: Exception if one occurred

        Returns:
            PauseContext with captured state

        Thread Safety:
            Acquires pause lock to ensure only one pause at a time.
        """
        async with self._pause_lock:
            # Create pause context
            context = PauseContext(
                session_id=self.session_id,
                node_id=node_id,
                position=position,
                state=state,
                breakpoint_ids=[bp.id for bp in breakpoints],
                timestamp=time.time(),
                error=error,
            )

            self._pause_context = context
            self._state = ExecutionState.PAUSED

            # Clear resume event (wait for resume signal)
            self._resume_event.clear()

            # Wait for resume (blocks execution)
            await self._resume_event.wait()

            # Clear pause context after resume
            self._pause_context = None

            return context

    async def continue_execution(self) -> None:
        """Continue execution after pause.

        Signals the paused workflow to resume.
        """
        if self._state != ExecutionState.PAUSED:
            return

        self._step_mode = None
        self._state = ExecutionState.RUNNING
        self._resume_event.set()

    async def step_over(self) -> None:
        """Step to next node (don't enter sub-workflows)."""
        if self._state != ExecutionState.PAUSED:
            return

        self._step_mode = StepMode.STEP_OVER
        self._state = ExecutionState.STEPPING
        self._resume_event.set()

    async def step_into(self) -> None:
        """Step into next node (enter sub-workflows)."""
        if self._state != ExecutionState.PAUSED:
            return

        self._step_mode = StepMode.STEP_INTO
        self._state = ExecutionState.STEPPING
        self._resume_event.set()

    async def step_out(self) -> None:
        """Step out of current sub-workflow."""
        if self._state != ExecutionState.PAUSED:
            return

        self._step_mode = StepMode.STEP_OUT
        self._state = ExecutionState.STEPPING
        self._resume_event.set()

    async def pause_immediately(self) -> None:
        """Request immediate pause (user-requested pause).

        Sets state to PAUSED, which will be checked on next node.
        """
        self._state = ExecutionState.PAUSED

    def terminate(self) -> None:
        """Terminate execution (called on workflow completion)."""
        self._state = ExecutionState.TERMINATED
        self._resume_event.set()  # Release any waiting pause
