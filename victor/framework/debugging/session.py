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

"""Debug session management for workflow debugging.

This module provides the DebugSession class which manages a debug
session lifecycle, coordinating breakpoint manager, execution controller,
and state inspector.

Key Classes:
    DebugSessionConfig: Configuration for debug session
    DebugSession: Manages a debug session lifecycle

Example:
    from victor.framework.debugging.session import (
        DebugSession,
        DebugSessionConfig,
    )

    session = DebugSession(
        config=DebugSessionConfig(
            session_id="debug-123",
            workflow_id="code_review"
        ),
        event_bus=event_bus
    )

    # Set breakpoints
    session.set_breakpoint(node_id="analyze", position=BreakpointPosition.BEFORE)

    # Attach to workflow execution
    result = await workflow.invoke(
        input_state,
        debug_hook=session.create_hook()
    )

    # Control execution
    await session.continue_execution()

    # Inspect state
    state = session.get_current_state()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from victor.framework.debugging.breakpoints import (
    BreakpointManager,
    BreakpointPosition,
    WorkflowBreakpoint,
)
from victor.framework.debugging.execution import (
    ExecutionController,
    ExecutionState,
    PauseContext,
)
from victor.framework.debugging.hooks import DebugHook
from victor.framework.debugging.inspector import StateInspector


@dataclass
class DebugSessionConfig:
    """Configuration for debug session.

    Attributes:
        session_id: Unique debug session identifier
        workflow_id: Workflow identifier being debugged
        enable_breakpoints: Whether to enable breakpoints
        enable_exceptions: Whether to enable exception breakpoints
        session_timeout: Session timeout in seconds
        auto_resume: Whether to auto-resume on timeout

    Example:
        config = DebugSessionConfig(
            session_id="debug-123",
            workflow_id="code_review",
            enable_breakpoints=True,
            enable_exceptions=True
        )
    """

    session_id: str
    workflow_id: str
    enable_breakpoints: bool = True
    enable_exceptions: bool = True
    session_timeout: float = 3600.0  # 1 hour default
    auto_resume: bool = False  # Auto-resume on timeout


class DebugSession:
    """Manages a debug session lifecycle.

    Coordinates breakpoint manager, execution controller, and
    state inspector for a single debugging session.

    Attributes:
        config: Session configuration
        breakpoint_mgr: BreakpointManager instance
        execution_ctrl: ExecutionController instance
        inspector: StateInspector instance
        created_at: Session creation timestamp
        _event_bus: EventBus for events
        _active: Whether session is active
        _hook: DebugHook instance

    Example:
        session = DebugSession(
            config=DebugSessionConfig(
                session_id="debug-123",
                workflow_id="code_review"
            ),
            event_bus=event_bus
        )

        # Set breakpoints
        session.set_breakpoint(node_id="analyze", position=BreakpointPosition.BEFORE)

        # Attach to workflow execution
        result = await workflow.invoke(
            input_state,
            debug_hook=session.create_hook()
        )

        # Control execution
        await session.continue_execution()

        # Inspect state
        state = session.get_current_state()
    """

    def __init__(self, config: DebugSessionConfig, event_bus: Any) -> None:
        """Initialize debug session.

        Args:
            config: Session configuration
            event_bus: EventBus for events
        """
        self.config = config
        self._event_bus = event_bus
        # Use time.time() instead of event loop time to avoid "no event loop" errors
        import time

        self.created_at = time.time()

        # Create components
        self.breakpoint_mgr = BreakpointManager(event_bus)
        self.execution_ctrl = ExecutionController(config.session_id)
        self.inspector = StateInspector()

        # Session state
        self._active = True
        self._hook: Optional[DebugHook] = None

    def create_hook(self) -> DebugHook:
        """Create DebugHook for workflow execution.

        Returns:
            DebugHook configured for this session
        """
        self._hook = DebugHook(
            session_id=self.config.session_id,
            breakpoint_mgr=self.breakpoint_mgr,
            execution_ctrl=self.execution_ctrl,
            inspector=self.inspector,
            event_bus=self._event_bus,
        )
        return self._hook

    async def continue_execution(self) -> None:
        """Continue execution."""
        await self.execution_ctrl.continue_execution()

    async def step_over(self) -> None:
        """Step over next node."""
        await self.execution_ctrl.step_over()

    async def step_into(self) -> None:
        """Step into next node."""
        await self.execution_ctrl.step_into()

    async def step_out(self) -> None:
        """Step out of current sub-workflow."""
        await self.execution_ctrl.step_out()

    async def pause(self) -> None:
        """Request immediate pause."""
        await self.execution_ctrl.pause_immediately()

    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """Get current workflow state at pause.

        Returns:
            Current workflow state or None if not paused
        """
        context = self.execution_ctrl.pause_context
        return context.state if context else None

    def get_pause_context(self) -> Optional[PauseContext]:
        """Get current pause context.

        Returns:
            PauseContext or None if not paused
        """
        return self.execution_ctrl.pause_context

    def set_breakpoint(self, **kwargs) -> WorkflowBreakpoint:
        """Set a breakpoint (delegates to manager).

        Returns:
            Created WorkflowBreakpoint
        """
        return self.breakpoint_mgr.set_breakpoint(**kwargs)

    def clear_breakpoint(self, breakpoint_id: str) -> bool:
        """Clear a breakpoint (delegates to manager).

        Args:
            breakpoint_id: Breakpoint ID to clear

        Returns:
            True if breakpoint was found and cleared
        """
        return self.breakpoint_mgr.clear_breakpoint(breakpoint_id)

    def list_breakpoints(self, **kwargs) -> List[WorkflowBreakpoint]:
        """List breakpoints (delegates to manager).

        Returns:
            List of breakpoints
        """
        return self.breakpoint_mgr.list_breakpoints(**kwargs)

    async def stop(self) -> None:
        """Stop debug session."""
        self._active = False
        self.execution_ctrl.terminate()

        # Persist breakpoints if enabled
        await self.breakpoint_mgr.storage.persist()

    @property
    def is_active(self) -> bool:
        """Check if session is active.

        Returns:
            True if session is active
        """
        return self._active

    @property
    def is_paused(self) -> bool:
        """Check if execution is paused.

        Returns:
            True if execution is paused
        """
        return self.execution_ctrl.state == ExecutionState.PAUSED
