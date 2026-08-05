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

"""Protocol adapters for coordinator dependencies.

This module provides adapters that expose orchestrator functionality
through protocol interfaces, enabling coordinators to depend on
abstractions rather than concrete implementations.

Design Patterns:
- Adapter Pattern: Convert orchestrator interface to protocol interfaces
- Dependency Inversion: Depend on abstractions, not concrete implementations
- Interface Segregation: Focused protocols for specific needs

Example:
    >>> from victor.agent.coordinators.protocol_dependencies import OrchestratorProtocolAdapter
    >>>
    >>> adapter = OrchestratorProtocolAdapter(orchestrator)
    >>> result = await adapter.execute_tool_call(tool_call)
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from victor.agent.orchestrator_protocols import (
    IAgentOrchestrator,
    IToolExecutor,
    IMessageStore,
    IStateManager,
    ISessionManager,
    IConversationController,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OrchestratorProtocolAdapter",
]


class OrchestratorProtocolAdapter:
    """
    Adapter that exposes orchestrator through protocol interfaces.

    This adapter enables coordinators to use protocol-based dependency
    injection rather than depending on AgentOrchestrator directly.

    Supported Protocols:
        - IAgentOrchestrator: Core orchestrator operations
        - IToolExecutor: Tool execution
        - IMessageStore: Message storage
        - IStateManager: State management
        - ISessionManager: Session lifecycle
        - IConversationController: Conversation control

    Example:
        >>> from victor.agent.coordinators.protocol_dependencies import OrchestratorProtocolAdapter
        >>>
        >>> # Create adapter from orchestrator
        >>> adapter = OrchestratorProtocolAdapter(orchestrator)
        >>>
        >>> # Use as IToolExecutor
        >>> result = await adapter.execute_tool_call(tool_call)
        >>>
        >>> # Use as IMessageStore
        >>> messages = adapter.messages
        >>> adapter.add_message(message)
    """

    def __init__(self, orchestrator: Any):
        """
        Initialize adapter with orchestrator instance.

        Args:
            orchestrator: AgentOrchestrator or compatible instance
        """
        self._orchestrator = orchestrator

    # IAgentOrchestrator implementation
    async def run(self, task: str, **kwargs: Any) -> Any:
        """Execute a single-turn task."""
        return await self._orchestrator.run(task, **kwargs)

    async def stream(self, task: str, **kwargs: Any) -> AsyncIterator[Any]:
        """Stream task execution with real-time events."""
        async for event in self._orchestrator.stream(task, **kwargs):
            yield event

    def chat(self, **kwargs: Any) -> Any:
        """Create a multi-turn chat session."""
        return self._orchestrator.chat(**kwargs)

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a single tool."""
        return await self._orchestrator.execute_tool(tool_name, arguments, **kwargs)

    async def aclose(self) -> None:
        """Close orchestrator and release resources."""
        if hasattr(self._orchestrator, "aclose"):
            await self._orchestrator.aclose()

    @property
    def messages(self) -> List[Any]:
        """Get messages from orchestrator."""
        return self._orchestrator.messages

    @property
    def settings(self) -> Any:
        """Get orchestrator settings."""
        return self._orchestrator.settings

    @property
    def session_id(self) -> str:
        """Get current session ID."""
        return self._orchestrator.session_id

    # IToolExecutor implementation (LEGACY)
    async def execute_tool_call(
        self,
        tool_call: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """[LEGACY] Execute tool call via orchestrator."""
        return await self._orchestrator.execute_tool(
            tool_call.name,
            tool_call.arguments,
            **kwargs,
        )

    async def execute_tool_calls(
        self,
        tool_calls: List[Any],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """[LEGACY] Execute multiple tool calls."""
        results = []
        for tool_call in tool_calls:
            result = await self.execute_tool_call(tool_call, **kwargs)
            results.append(result)
        return results

    # IMessageStore implementation
    def add_message(self, message: Any) -> None:
        """Add message to orchestrator."""
        if hasattr(self._orchestrator, "add_message"):
            self._orchestrator.add_message(message)
        elif hasattr(self._orchestrator, "_add_message"):
            self._orchestrator._add_message(message)
        else:
            logger.warning("OrchestratorProtocolAdapter: No add_message method found")

    def get_messages(self, **filters: Any) -> List[Any]:
        """Get messages with optional filters."""
        return self.messages

    # IStateManager implementation
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state from orchestrator."""
        if hasattr(self._orchestrator, "get_state"):
            return self._orchestrator.get_state(key, default)
        elif hasattr(self._orchestrator, "_session_state_manager"):
            return self._orchestrator._session_state_manager.get_state(key, default)
        else:
            logger.warning(f"OrchestratorProtocolAdapter: No state manager found for {key}")
            return default

    def set_state(self, key: str, value: Any) -> None:
        """Set state in orchestrator."""
        if hasattr(self._orchestrator, "set_state"):
            self._orchestrator.set_state(key, value)
        elif hasattr(self._orchestrator, "_session_state_manager"):
            self._orchestrator._session_state_manager.set_state(key, value)
        else:
            logger.warning(f"OrchestratorProtocolAdapter: No state manager found for {key}")

    def delete_state(self, key: str) -> None:
        """Delete state from orchestrator."""
        if hasattr(self._orchestrator, "delete_state"):
            self._orchestrator.delete_state(key)
        elif hasattr(self._orchestrator, "_session_state_manager"):
            self._orchestrator._session_state_manager.delete_state(key)
        else:
            logger.warning(f"OrchestratorProtocolAdapter: No state manager found for {key}")

    # ISessionManager implementation
    async def create_session(self, **kwargs: Any) -> str:
        """Create a new session."""
        if hasattr(self._orchestrator, "create_session"):
            return await self._orchestrator.create_session(**kwargs)
        raise NotImplementedError("Session creation not supported")

    async def reset_session(self, session_id: Optional[str] = None) -> None:
        """Reset session state."""
        if hasattr(self._orchestrator, "reset_session"):
            await self._orchestrator.reset_session(session_id)
        elif hasattr(self._orchestrator, "_conversation_controller"):
            await self._orchestrator._conversation_controller.reset_conversation()

    async def close_session(self, session_id: Optional[str] = None) -> None:
        """Close session and release resources."""
        if hasattr(self._orchestrator, "close_session"):
            await self._orchestrator.close_session(session_id)
        await self.aclose()

    # IConversationController implementation
    async def process_message(self, message: Any, **kwargs: Any) -> Any:
        """Process a user message."""
        if hasattr(self._orchestrator, "process_message"):
            return await self._orchestrator.process_message(message, **kwargs)
        # Fallback to run() method
        return await self.run(message)

    async def stream_message(
        self,
        message: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream message processing."""
        if hasattr(self._orchestrator, "stream_message"):
            async for event in self._orchestrator.stream_message(message, **kwargs):
                yield event
        else:
            # Fallback to stream() method
            async for event in self.stream(message, **kwargs):
                yield event

    @property
    def conversation_stage(self) -> str:
        """Get current conversation stage."""
        if hasattr(self._orchestrator, "conversation_stage"):
            return self._orchestrator.conversation_stage
        elif hasattr(self._orchestrator, "_conversation_state"):
            return str(self._orchestrator._conversation_state.current_stage)
        return "unknown"

    @property
    def is_complete(self) -> bool:
        """Check if conversation is complete."""
        if hasattr(self._orchestrator, "is_complete"):
            return self._orchestrator.is_complete
        elif hasattr(self._orchestrator, "_conversation_state"):
            from victor.agent.conversation.state_machine import ConversationStage

            return (
                self._orchestrator._conversation_state.current_stage == ConversationStage.COMPLETE
            )
        return False
