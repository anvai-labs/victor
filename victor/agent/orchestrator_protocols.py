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

"""Protocol-based interfaces for AgentOrchestrator.

This module defines lightweight protocol interfaces for orchestrator operations,
enabling lazy loading and reducing import dependencies.

Design Patterns:
- Protocol-Oriented Programming: Define interfaces without implementation
- Dependency Inversion: Depend on abstractions, not concrete implementations
- Facade Pattern: Provide simplified interfaces to complex subsystems

Example:
    >>> from victor.agent.orchestrator_protocols import IAgentOrchestrator
    >>>
    >>> def process_agent(orchestrator: IAgentOrchestrator):
    ...     # Use orchestrator without importing AgentOrchestrator
    ...     result = await orchestrator.run("Do something")
    ...     return result
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable

__all__ = [
    "IAgentOrchestrator",
    "IToolExecutor",
    "IMessageStore",
    "IStateManager",
    "ISessionManager",
    "IConversationController",
]


@runtime_checkable
class IAgentOrchestrator(Protocol):
    """
    Protocol-based interface for orchestrator operations.

    This protocol defines the public contract for orchestrator operations
    without importing the actual AgentOrchestrator implementation.
    This enables zero-import access to orchestrator functionality.

    Core Methods:
        run: Execute a single-turn task
        stream: Stream task execution with real-time events
        chat: Create a multi-turn chat session
        execute_tool: Execute a single tool
        reset: Reset orchestrator state

    Properties:
        messages: Get conversation messages
        settings: Get orchestrator settings
        session_id: Get current session ID
    """

    # Core execution methods
    async def run(self, task: str, **kwargs: Any) -> Any:
        """Execute a single-turn task.

        Args:
            task: Task description
            **kwargs: Additional execution parameters

        Returns:
            Execution result
        """
        ...

    async def stream(self, task: str, **kwargs: Any) -> AsyncIterator[Any]:
        """Stream task execution with real-time events.

        Args:
            task: Task description
            **kwargs: Additional execution parameters

        Yields:
            Streaming events (dict or Event objects)
        """
        ...

    def chat(self, **kwargs: Any) -> Any:
        """Create a multi-turn chat session.

        Args:
            **kwargs: Chat configuration parameters

        Returns:
            Chat session object
        """
        ...

    # Tool execution
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a single tool.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            **kwargs: Additional execution parameters

        Returns:
            Tool execution result
        """
        ...

    # Lifecycle
    async def aclose(self) -> None:
        """Close orchestrator and release resources."""
        ...

    # Properties
    @property
    def messages(self) -> List[Any]:
        """Get conversation messages."""
        ...

    @property
    def settings(self) -> Any:
        """Get orchestrator settings."""
        ...

    @property
    def session_id(self) -> str:
        """Get current session ID."""
        ...


@runtime_checkable
class IToolExecutor(Protocol):
    """[LEGACY PROTOCOL] Interface for tool execution.

    Superseded by IToolCoordinator for high-level operations.
    Defines interface for executing tools with retry, caching,
    and error handling.
    """

    async def execute_tool_call(
        self,
        tool_call: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a tool call.

        Args:
            tool_call: Tool call object (ExtractedToolCall or similar)
            **kwargs: Additional execution parameters

        Returns:
            Tool execution result with status and output
        """
        ...

    async def execute_tool_calls(
        self,
        tool_calls: List[Any],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Execute multiple tool calls.

        Args:
            tool_calls: List of tool call objects
            **kwargs: Additional execution parameters

        Returns:
            List of tool execution results
        """
        ...


@runtime_checkable
class IMessageStore(Protocol):
    """Protocol for message storage operations.

    Defines interface for managing conversation messages.
    """

    @property
    def messages(self) -> List[Any]:
        """Get all messages."""
        ...

    def add_message(self, message: Any) -> None:
        """Add a message to storage.

        Args:
            message: Message object to add
        """
        ...

    def get_messages(self, **filters: Any) -> List[Any]:
        """Get messages with optional filters.

        Args:
            **filters: Filter criteria

        Returns:
            Filtered list of messages
        """
        ...


@runtime_checkable
class IStateManager(Protocol):
    """Protocol for state management operations.

    Defines interface for managing session and conversation state.
    """

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            State value
        """
        ...

    def set_state(self, key: str, value: Any) -> None:
        """Set state value.

        Args:
            key: State key
            value: State value
        """
        ...

    def delete_state(self, key: str) -> None:
        """Delete state value.

        Args:
            key: State key to delete
        """
        ...


@runtime_checkable
class ISessionManager(Protocol):
    """Protocol for session lifecycle management.

    Defines interface for creating, resetting, and closing sessions.
    """

    async def create_session(self, **kwargs: Any) -> str:
        """Create a new session.

        Args:
            **kwargs: Session configuration parameters

        Returns:
            Session ID
        """
        ...

    async def reset_session(self, session_id: Optional[str] = None) -> None:
        """Reset session state.

        Args:
            session_id: Optional session ID (defaults to current session)
        """
        ...

    async def close_session(self, session_id: Optional[str] = None) -> None:
        """Close session and release resources.

        Args:
            session_id: Optional session ID (defaults to current session)
        """
        ...


@runtime_checkable
class IConversationController(Protocol):
    """Protocol for conversation control operations.

    Defines interface for managing conversation flow, context,
    and message processing.
    """

    async def process_message(
        self,
        message: Any,
        **kwargs: Any,
    ) -> Any:
        """Process a user message.

        Args:
            message: User message
            **kwargs: Additional processing parameters

        Returns:
            Processing result
        """
        ...

    async def stream_message(
        self,
        message: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream message processing.

        Args:
            message: User message
            **kwargs: Additional processing parameters

        Yields:
            Streaming events
        """
        ...

    @property
    def conversation_stage(self) -> str:
        """Get current conversation stage."""
        ...

    @property
    def is_complete(self) -> bool:
        """Check if conversation is complete."""
        ...
