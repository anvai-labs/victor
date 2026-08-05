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

"""Debug Adapter interface and base implementation.

This module defines the abstract interface for debug adapters using the
Strategy Pattern, allowing different debuggers (debugpy, lldb, gdb, etc.)
to be used interchangeably through a unified API.

Design Patterns:
    - Strategy Pattern: Each adapter implements the same interface
    - Factory Pattern: Registry creates adapters based on language
    - Observer Pattern: Events are emitted for state changes
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
    runtime_checkable,
)

from victor.observability.debug.protocol import (
    AttachConfiguration,
    Breakpoint,
    DebugSession,
    DebugState,
    EvaluateResult,
    LaunchConfiguration,
    Scope,
    SourceLocation,
    StackFrame,
    Thread,
    Variable,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DebugEventType(Enum):
    """Types of debug events."""

    INITIALIZED = "initialized"
    STOPPED = "stopped"
    CONTINUED = "continued"
    EXITED = "exited"
    TERMINATED = "terminated"
    THREAD_STARTED = "thread_started"
    THREAD_EXITED = "thread_exited"
    OUTPUT = "output"
    BREAKPOINT_CHANGED = "breakpoint_changed"
    MODULE_LOADED = "module_loaded"
    PROCESS_STARTED = "process_started"


@dataclass
class DebugEvent:
    """An event from the debug adapter."""

    type: DebugEventType
    session_id: str
    data: Dict[str, Any] = field(default_factory=dict)


# Event handler type
EventHandler = Callable[[DebugEvent], None]


@dataclass
class DebugAdapterCapabilities:
    """Capabilities advertised by a debug adapter.

    Adapters report what features they support, allowing the UI
    to enable/disable features accordingly.
    """

    # Breakpoint capabilities
    supports_conditional_breakpoints: bool = False
    supports_hit_conditional_breakpoints: bool = False
    supports_log_points: bool = False
    supports_function_breakpoints: bool = False
    supports_data_breakpoints: bool = False
    supports_instruction_breakpoints: bool = False

    # Execution control
    supports_step_back: bool = False
    supports_restart_frame: bool = False
    supports_goto_targets: bool = False
    supports_stepping_granularity: bool = False
    supports_terminate_request: bool = True
    supports_suspend_debuggee: bool = True

    # Evaluation
    supports_evaluate_for_hovers: bool = True
    supports_set_variable: bool = False
    supports_set_expression: bool = False
    supports_completions_request: bool = False

    # Exception handling
    supports_exception_options: bool = False
    supports_exception_filter_options: bool = False
    exception_breakpoint_filters: List[Dict[str, Any]] = field(default_factory=list)

    # Other
    supports_modules_request: bool = False
    supports_loaded_sources_request: bool = False
    supports_read_memory_request: bool = False
    supports_write_memory_request: bool = False
    supports_disassemble_request: bool = False

    # Language-specific
    supported_languages: List[str] = field(default_factory=list)


@runtime_checkable
class DebugAdapter(Protocol):
    """Protocol for debug adapters.

    Each language debugger (debugpy, lldb, gdb, etc.) implements this
    protocol to provide a unified debugging interface.

    Lifecycle:
        1. initialize() - Start the adapter
        2. launch() or attach() - Start/connect to program
        3. Debug operations (step, continue, breakpoints, etc.)
        4. disconnect() - End the session
        5. shutdown() - Clean up adapter resources
    """

    @property
    def name(self) -> str:
        """Adapter name (e.g., 'debugpy', 'lldb')."""
        ...

    @property
    def languages(self) -> List[str]:
        """Languages supported by this adapter."""
        ...

    @property
    def capabilities(self) -> DebugAdapterCapabilities:
        """Capabilities of this adapter."""
        ...

    # Lifecycle methods

    async def initialize(self) -> DebugAdapterCapabilities:
        """Initialize the debug adapter.

        Returns:
            Adapter capabilities
        """
        ...

    async def launch(self, config: LaunchConfiguration) -> DebugSession:
        """Launch a program for debugging.

        Args:
            config: Launch configuration

        Returns:
            New debug session
        """
        ...

    async def attach(self, config: AttachConfiguration) -> DebugSession:
        """Attach to a running process.

        Args:
            config: Attach configuration

        Returns:
            New debug session
        """
        ...

    async def disconnect(self, session_id: str, terminate: bool = False) -> None:
        """Disconnect from debug session.

        Args:
            session_id: Session to disconnect
            terminate: Whether to terminate the debuggee
        """
        ...

    async def shutdown(self) -> None:
        """Shutdown the adapter and release resources."""
        ...

    # Breakpoint operations

    async def set_breakpoints(
        self,
        session_id: str,
        source: Path,
        breakpoints: List[SourceLocation],
        conditions: Optional[Dict[int, str]] = None,
    ) -> List[Breakpoint]:
        """Set breakpoints in a source file.

        Args:
            session_id: Debug session ID
            source: Source file path
            breakpoints: Locations for breakpoints
            conditions: Optional conditions for each breakpoint (line -> condition)

        Returns:
            List of created breakpoints with verification status
        """
        ...

    async def set_function_breakpoints(
        self,
        session_id: str,
        names: List[str],
    ) -> List[Breakpoint]:
        """Set breakpoints on function names.

        Args:
            session_id: Debug session ID
            names: Function names to break on

        Returns:
            List of created breakpoints
        """
        ...

    async def clear_breakpoints(self, session_id: str, source: Path) -> None:
        """Clear all breakpoints in a source file.

        Args:
            session_id: Debug session ID
            source: Source file path
        """
        ...

    # Execution control

    async def continue_execution(self, session_id: str, thread_id: Optional[int] = None) -> None:
        """Continue execution.

        Args:
            session_id: Debug session ID
            thread_id: Optional specific thread to continue
        """
        ...

    async def pause(self, session_id: str, thread_id: Optional[int] = None) -> None:
        """Pause execution.

        Args:
            session_id: Debug session ID
            thread_id: Optional specific thread to pause
        """
        ...

    async def step_over(self, session_id: str, thread_id: int) -> None:
        """Step over (next line, skip function calls).

        Args:
            session_id: Debug session ID
            thread_id: Thread to step
        """
        ...

    async def step_into(self, session_id: str, thread_id: int) -> None:
        """Step into (enter function calls).

        Args:
            session_id: Debug session ID
            thread_id: Thread to step
        """
        ...

    async def step_out(self, session_id: str, thread_id: int) -> None:
        """Step out (return from current function).

        Args:
            session_id: Debug session ID
            thread_id: Thread to step
        """
        ...

    # Inspection

    async def get_threads(self, session_id: str) -> List[Thread]:
        """Get all threads in the debugged program.

        Args:
            session_id: Debug session ID

        Returns:
            List of threads
        """
        ...

    async def get_stack_trace(
        self,
        session_id: str,
        thread_id: int,
        start_frame: int = 0,
        levels: int = 20,
    ) -> List[StackFrame]:
        """Get stack trace for a thread.

        Args:
            session_id: Debug session ID
            thread_id: Thread to get stack for
            start_frame: First frame to return
            levels: Maximum number of frames

        Returns:
            List of stack frames
        """
        ...

    async def get_scopes(self, session_id: str, frame_id: int) -> List[Scope]:
        """Get variable scopes for a stack frame.

        Args:
            session_id: Debug session ID
            frame_id: Stack frame ID

        Returns:
            List of scopes (Locals, Globals, etc.)
        """
        ...

    async def get_variables(
        self,
        session_id: str,
        variables_reference: int,
        filter_type: Optional[str] = None,
        start: int = 0,
        count: int = 100,
    ) -> List[Variable]:
        """Get variables for a scope or container.

        Args:
            session_id: Debug session ID
            variables_reference: Reference from scope or variable
            filter_type: "indexed" or "named" to filter
            start: Start index for paging
            count: Number to return

        Returns:
            List of variables
        """
        ...

    async def evaluate(
        self,
        session_id: str,
        expression: str,
        frame_id: Optional[int] = None,
        context: str = "repl",
    ) -> EvaluateResult:
        """Evaluate an expression in the debug context.

        Args:
            session_id: Debug session ID
            expression: Expression to evaluate
            frame_id: Stack frame for context
            context: "watch", "repl", "hover", or "clipboard"

        Returns:
            Evaluation result
        """
        ...

    # Events

    def add_event_handler(self, handler: EventHandler) -> None:
        """Add an event handler for debug events.

        Args:
            handler: Callback for events
        """
        ...

    def remove_event_handler(self, handler: EventHandler) -> None:
        """Remove an event handler.

        Args:
            handler: Handler to remove
        """
        ...


class BaseDebugAdapter(ABC):
    """Base class for debug adapters with common functionality.

    Provides:
    - Event handling infrastructure
    - Session management
    - Logging
    - Common validation

    Subclasses must implement language-specific debug operations.
    """

    def __init__(self):
        """Initialize base adapter."""
        self._event_handlers: Set[EventHandler] = set()
        self._sessions: Dict[str, DebugSession] = {}
        self._lock = asyncio.Lock()
        self._capabilities: Optional[DebugAdapterCapabilities] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name."""
        ...

    @property
    @abstractmethod
    def languages(self) -> List[str]:
        """Supported languages."""
        ...

    @property
    def capabilities(self) -> DebugAdapterCapabilities:
        """Get adapter capabilities."""
        if self._capabilities is None:
            self._capabilities = self._get_default_capabilities()
        return self._capabilities

    @abstractmethod
    def _get_default_capabilities(self) -> DebugAdapterCapabilities:
        """Get default capabilities for this adapter."""
        ...

    def add_event_handler(self, handler: EventHandler) -> None:
        """Add event handler."""
        self._event_handlers.add(handler)

    def remove_event_handler(self, handler: EventHandler) -> None:
        """Remove event handler."""
        self._event_handlers.discard(handler)

    def _emit_event(self, event: DebugEvent) -> None:
        """Emit an event to all handlers."""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def _get_session(self, session_id: str) -> DebugSession:
        """Get session by ID, raising if not found."""
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        return session

    def _update_session_state(self, session_id: str, state: DebugState) -> None:
        """Update session state and emit event."""
        session = self._get_session(session_id)
        old_state = session.state
        session.state = state

        event_type = {
            DebugState.RUNNING: DebugEventType.CONTINUED,
            DebugState.STOPPED: DebugEventType.STOPPED,
            DebugState.TERMINATED: DebugEventType.TERMINATED,
        }.get(state)

        if event_type:
            self._emit_event(
                DebugEvent(
                    type=event_type,
                    session_id=session_id,
                    data={"old_state": old_state.value, "new_state": state.value},
                )
            )
