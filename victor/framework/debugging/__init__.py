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

"""Breakpoint and debugging system for StateGraph workflows.

This package provides comprehensive debugging capabilities for Victor's
StateGraph workflows, including breakpoints, execution control, state
inspection, and debug session management.

Key Components:
    BreakpointType: Types of breakpoints (NODE, CONDITIONAL, EXCEPTION, STATE)
    BreakpointPosition: Position relative to node execution (BEFORE, AFTER, ON_ERROR)
    WorkflowBreakpoint: A breakpoint in workflow execution
    BreakpointManager: Manages workflow breakpoints
    ExecutionController: Controls execution pause/resume
    StateInspector: Inspects workflow state
    DebugHook: Integration point with StateGraph
    DebugSession: Debug session facade

Example:
    from victor.framework.debugging import (
        DebugSession,
        DebugSessionConfig,
        BreakpointPosition,
    )

    # Create debug session
    session = DebugSession(
        config=DebugSessionConfig(
            session_id="debug-123",
            workflow_id="code_review"
        ),
        event_bus=event_bus
    )

    # Set breakpoint
    session.set_breakpoint(
        node_id="analyze",
        position=BreakpointPosition.BEFORE
    )

    # Attach to workflow
    result = await workflow.invoke(
        input_state,
        debug_hook=session.create_hook()
    )

    # Control execution
    await session.continue_execution()
"""

# Breakpoint types and management
from victor.framework.debugging.breakpoints import (
    BreakpointManager,
    BreakpointPosition,
    BreakpointStorage,
    BreakpointType,
    WorkflowBreakpoint,
)

# Execution control
from victor.framework.debugging.execution import (
    ExecutionController,
    ExecutionState,
    PauseContext,
    StepMode,
)

# State inspection
from victor.framework.debugging.inspector import (
    StateDiff,
    StateInspector,
    StateSnapshot,
)

# Debug protocol
from victor.framework.debugging.protocol import (
    DebugMessage,
    DebugMessageType,
    DebugProtocolHandler,
)

# Debug hooks
from victor.framework.debugging.hooks import DebugHook

# Debug session
from victor.framework.debugging.session import (
    DebugSession,
    DebugSessionConfig,
)

__all__ = [
    # Breakpoint types and management
    "BreakpointManager",
    "BreakpointPosition",
    "BreakpointStorage",
    "BreakpointType",
    "WorkflowBreakpoint",
    # Execution control
    "ExecutionController",
    "ExecutionState",
    "PauseContext",
    "StepMode",
    # State inspection
    "StateDiff",
    "StateInspector",
    "StateSnapshot",
    # Debug protocol
    "DebugMessage",
    "DebugMessageType",
    "DebugProtocolHandler",
    # Debug hooks
    "DebugHook",
    # Debug session
    "DebugSession",
    "DebugSessionConfig",
]
