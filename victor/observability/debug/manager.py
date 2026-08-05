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

"""Debug Manager for orchestrating debug sessions.

The Debug Manager provides a high-level interface for debugging operations,
abstracting away the complexity of adapter management and session lifecycle.

Design Patterns:
    - Facade Pattern: Simplifies debug adapter interaction
    - Observer Pattern: Events for state changes
    - Command Pattern: Debug operations as commands for undo/redo

Usage:
    manager = DebugManager()

    # Start debugging
    session = await manager.launch("my_script.py", language="python")

    # Set breakpoints
    await manager.set_breakpoint(session.id, "my_script.py", line=10)

    # Step through code
    await manager.step_over(session.id)

    # Inspect state
    variables = await manager.get_variables(session.id, frame_id=0)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from victor.observability.debug.adapter import (
    DebugAdapter,
    DebugAdapterCapabilities,
    DebugEvent,
    EventHandler,
)
from victor.observability.debug.protocol import (
    AttachConfiguration,
    Breakpoint,
    DebugSession,
    EvaluateResult,
    LaunchConfiguration,
    Scope,
    SourceLocation,
    StackFrame,
    Thread,
    Variable,
)
from victor.observability.debug.registry import DebugAdapterRegistry, get_debug_registry

logger = logging.getLogger(__name__)


class DebugManager:
    """High-level debug session manager.

    Provides a unified interface for all debugging operations,
    managing multiple concurrent debug sessions across different
    languages and adapters.

    Features:
    - Multi-session management
    - Automatic adapter selection
    - Event aggregation
    - Session state tracking
    - Breakpoint persistence
    """

    def __init__(
        self,
        registry: Optional[DebugAdapterRegistry] = None,
        auto_discover: bool = True,
    ):
        """Initialize debug manager.

        Args:
            registry: Custom adapter registry (uses global if None)
            auto_discover: Whether to auto-discover adapters
        """
        self._registry = registry or get_debug_registry()
        self._sessions: Dict[str, DebugSession] = {}
        self._session_adapters: Dict[str, DebugAdapter] = {}
        self._event_handlers: Set[EventHandler] = set()
        self._lock = asyncio.Lock()

        # Auto-discover adapters if requested
        if auto_discover:
            self._registry.discover_adapters()

    # Session Management

    async def launch(
        self,
        program: str,
        language: Optional[str] = None,
        arguments: Optional[List[str]] = None,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        stop_on_entry: bool = False,
        **extra_options: Any,
    ) -> DebugSession:
        """Launch a program for debugging.

        Args:
            program: Path to program to debug
            language: Language (auto-detected if not specified)
            arguments: Command-line arguments
            working_directory: Working directory for program
            environment: Environment variables
            stop_on_entry: Stop at program entry point
            **extra_options: Language-specific options

        Returns:
            New debug session

        Raises:
            ValueError: If language cannot be determined or no adapter
        """
        program_path = Path(program).resolve()

        # Auto-detect language from extension if not specified
        if language is None:
            language = self._detect_language(program_path)
            if language is None:
                raise ValueError(
                    f"Cannot detect language for {program}. " f"Please specify language explicitly."
                )

        # Get adapter for language
        adapter = self._registry.get_adapter(language)

        # Initialize adapter if needed
        await adapter.initialize()

        # Create launch configuration
        config = LaunchConfiguration(
            program=program_path,
            language=language,
            arguments=arguments or [],
            working_directory=Path(working_directory) if working_directory else None,
            environment=environment or {},
            stop_on_entry=stop_on_entry,
            extra_options=extra_options,
        )

        # Launch program
        session = await adapter.launch(config)

        # Store session and adapter mapping
        async with self._lock:
            self._sessions[session.id] = session
            self._session_adapters[session.id] = adapter

        # Forward adapter events
        adapter.add_event_handler(self._forward_event)

        logger.info(f"Started debug session {session.id} for {program}")
        return session

    async def attach(
        self,
        process_id: Optional[int] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        language: str = "python",
        **extra_options: Any,
    ) -> DebugSession:
        """Attach to a running process.

        Args:
            process_id: Process ID to attach to
            host: Host for remote debugging
            port: Port for remote debugging
            language: Language of the process
            **extra_options: Language-specific options

        Returns:
            New debug session
        """
        adapter = self._registry.get_adapter(language)
        await adapter.initialize()

        config = AttachConfiguration(
            process_id=process_id,
            host=host,
            port=port,
            language=language,
            extra_options=extra_options,
        )

        session = await adapter.attach(config)

        async with self._lock:
            self._sessions[session.id] = session
            self._session_adapters[session.id] = adapter

        adapter.add_event_handler(self._forward_event)

        logger.info(f"Attached to process, session {session.id}")
        return session

    async def disconnect(self, session_id: str, terminate: bool = False) -> None:
        """Disconnect from a debug session.

        Args:
            session_id: Session to disconnect
            terminate: Whether to terminate the debuggee
        """
        adapter = self._get_adapter(session_id)
        await adapter.disconnect(session_id, terminate)

        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
            if session_id in self._session_adapters:
                del self._session_adapters[session_id]

        logger.info(f"Disconnected from session {session_id}")

    def get_session(self, session_id: str) -> Optional[DebugSession]:
        """Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session or None if not found
        """
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[DebugSession]:
        """List all active sessions.

        Returns:
            List of active sessions
        """
        return list(self._sessions.values())

    # Breakpoint Operations

    async def set_breakpoint(
        self,
        session_id: str,
        file: str,
        line: int,
        condition: Optional[str] = None,
        hit_condition: Optional[str] = None,
        log_message: Optional[str] = None,
    ) -> Breakpoint:
        """Set a breakpoint.

        Args:
            session_id: Debug session ID
            file: Source file path
            line: Line number (1-indexed)
            condition: Optional condition expression
            hit_condition: Optional hit count condition
            log_message: Optional log message (logpoint)

        Returns:
            Created breakpoint
        """
        adapter = self._get_adapter(session_id)
        file_path = Path(file).resolve()

        location = SourceLocation(path=file_path, line=line)
        conditions = {line: condition} if condition else None

        breakpoints = await adapter.set_breakpoints(session_id, file_path, [location], conditions)

        if breakpoints:
            bp = breakpoints[0]
            # Update session breakpoints
            session = self._sessions[session_id]
            key = str(file_path)
            if key not in session.breakpoints:
                session.breakpoints[key] = []
            session.breakpoints[key].append(bp)
            return bp

        raise RuntimeError(f"Failed to set breakpoint at {file}:{line}")

    async def remove_breakpoint(self, session_id: str, file: str, line: int) -> None:
        """Remove a breakpoint.

        Args:
            session_id: Debug session ID
            file: Source file path
            line: Line number
        """
        session = self._sessions.get(session_id)
        if not session:
            return

        file_path = Path(file).resolve()
        key = str(file_path)

        if key in session.breakpoints:
            # Filter out the breakpoint at this line
            session.breakpoints[key] = [
                bp for bp in session.breakpoints[key] if bp.source.line != line
            ]

            # Update adapter with remaining breakpoints
            adapter = self._get_adapter(session_id)
            remaining = [
                SourceLocation(path=bp.source.path, line=bp.source.line)
                for bp in session.breakpoints.get(key, [])
            ]
            await adapter.set_breakpoints(session_id, file_path, remaining)

    async def clear_all_breakpoints(self, session_id: str, file: str) -> None:
        """Clear all breakpoints in a file.

        Args:
            session_id: Debug session ID
            file: Source file path
        """
        adapter = self._get_adapter(session_id)
        file_path = Path(file).resolve()
        await adapter.clear_breakpoints(session_id, file_path)

        # Update session
        session = self._sessions.get(session_id)
        if session:
            key = str(file_path)
            session.breakpoints.pop(key, None)

    # Execution Control

    async def continue_execution(self, session_id: str) -> None:
        """Continue execution until next breakpoint."""
        adapter = self._get_adapter(session_id)
        await adapter.continue_execution(session_id)

    async def pause(self, session_id: str) -> None:
        """Pause execution."""
        adapter = self._get_adapter(session_id)
        await adapter.pause(session_id)

    async def step_over(self, session_id: str, thread_id: Optional[int] = None) -> None:
        """Step over (next line)."""
        adapter = self._get_adapter(session_id)
        thread_id = thread_id or self._get_current_thread(session_id)
        await adapter.step_over(session_id, thread_id)

    async def step_into(self, session_id: str, thread_id: Optional[int] = None) -> None:
        """Step into function."""
        adapter = self._get_adapter(session_id)
        thread_id = thread_id or self._get_current_thread(session_id)
        await adapter.step_into(session_id, thread_id)

    async def step_out(self, session_id: str, thread_id: Optional[int] = None) -> None:
        """Step out of function."""
        adapter = self._get_adapter(session_id)
        thread_id = thread_id or self._get_current_thread(session_id)
        await adapter.step_out(session_id, thread_id)

    # Inspection

    async def get_threads(self, session_id: str) -> List[Thread]:
        """Get all threads."""
        adapter = self._get_adapter(session_id)
        threads = await adapter.get_threads(session_id)

        # Update session
        session = self._sessions.get(session_id)
        if session:
            session.threads = threads

        return threads

    async def get_stack_trace(
        self, session_id: str, thread_id: Optional[int] = None
    ) -> List[StackFrame]:
        """Get stack trace for a thread."""
        adapter = self._get_adapter(session_id)
        thread_id = thread_id or self._get_current_thread(session_id)
        return await adapter.get_stack_trace(session_id, thread_id)

    async def get_scopes(self, session_id: str, frame_id: int) -> List[Scope]:
        """Get variable scopes for a stack frame."""
        adapter = self._get_adapter(session_id)
        return await adapter.get_scopes(session_id, frame_id)

    async def get_variables(self, session_id: str, scope_or_ref: int) -> List[Variable]:
        """Get variables for a scope or reference."""
        adapter = self._get_adapter(session_id)
        return await adapter.get_variables(session_id, scope_or_ref)

    async def evaluate(
        self,
        session_id: str,
        expression: str,
        frame_id: Optional[int] = None,
    ) -> EvaluateResult:
        """Evaluate an expression."""
        adapter = self._get_adapter(session_id)
        return await adapter.evaluate(session_id, expression, frame_id)

    # Events

    def add_event_handler(self, handler: EventHandler) -> None:
        """Add event handler."""
        self._event_handlers.add(handler)

    def remove_event_handler(self, handler: EventHandler) -> None:
        """Remove event handler."""
        self._event_handlers.discard(handler)

    def _forward_event(self, event: DebugEvent) -> None:
        """Forward event from adapter to handlers."""
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    # Helper Methods

    def _get_adapter(self, session_id: str) -> DebugAdapter:
        """Get adapter for a session."""
        adapter = self._session_adapters.get(session_id)
        if adapter is None:
            raise ValueError(f"No adapter for session: {session_id}")
        return adapter

    def _get_current_thread(self, session_id: str) -> int:
        """Get current thread ID for session."""
        session = self._sessions.get(session_id)
        if session and session.current_thread_id is not None:
            return session.current_thread_id
        if session and session.threads:
            return session.threads[0].id
        raise ValueError(f"No threads available for session: {session_id}")

    def _detect_language(self, path: Path) -> Optional[str]:
        """Detect language from file extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".rs": "rust",
            ".go": "golang",
            ".rb": "ruby",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".java": "java",
            ".kt": "kotlin",
            ".swift": "swift",
        }
        return extension_map.get(path.suffix.lower())

    # Utilities

    def supported_languages(self) -> List[str]:
        """Get list of languages with debug support."""
        return self._registry.supported_languages()

    def get_capabilities(self, language: str) -> DebugAdapterCapabilities:
        """Get capabilities for a language's debugger."""
        adapter = self._registry.get_adapter(language)
        return adapter.capabilities
