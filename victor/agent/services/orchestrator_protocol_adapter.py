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

"""Service-owned protocol adapter for orchestration runtime state.

This module provides adapter classes that adapt AgentOrchestrator to various
protocol interfaces, enabling the Dependency Inversion Principle (DIP) across
the orchestration layer.

Design Pattern: Adapter Pattern
- OrchestratorProtocolAdapter: Adapts AgentOrchestrator to protocol interfaces
- Enables coordinators to depend on protocols rather than concrete classes
- Facilitates unit testing with lightweight protocol mocks

The legacy `victor.agent.coordinators.protocol_adapters` module now re-exports
this implementation for compatibility.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio

    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.conversation.controller import ConversationController
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.message_history import MessageHistory
    from victor.agent.session_state_manager import SessionStateManager
    from victor.agent.task_analyzer import TaskAnalyzer
    from victor.agent.services.tool_planning_runtime import ToolPlanner
    from victor.agent.tool_selection import ToolSelector
    from victor.config.settings import Settings
    from victor.providers.base import BaseProvider, CompletionResponse, Message

logger = logging.getLogger(__name__)

# Internal orchestration/topology hints that ride **kwargs but are not provider
# request params (W3d/G7 D4). Kept in sync with the sandhi codec's backstop
# strip (victor/providers/sandhi_transport.py::_INTERNAL_KWARG_KEYS).
_INTERNAL_KWARG_KEYS = frozenset(
    {
        "execution_mode",
        "provider_hint",
        "escalation_target",
        "topology_action",
        "topology_kind",
        "topology_metadata",
        "task_type",
        "temperature_override",
    }
)


class OrchestratorProtocolAdapter:
    """Adapts AgentOrchestrator to protocol interfaces.

    The orchestrator implements multiple protocols through this adapter,
    allowing coordinators to depend on protocols rather than the
    concrete orchestrator class.

    This adapter implements:
    - ExecutionProvider: For executing model turns
    - ToolExecutor: For tool execution
    - MessageStore: For message storage
    - StateManager: For state management

    The adapter uses the existing orchestrator methods and properties,
    providing a protocol-based facade to the orchestrator's capabilities.
    """

    def __init__(self, orchestrator: "AgentOrchestrator") -> None:
        """Initialize the adapter with an orchestrator.

        Args:
            orchestrator: The AgentOrchestrator to adapt
        """
        self._orchestrator = orchestrator

    def __getattr__(self, name: str) -> Any:
        """Proxy unresolved compatibility/runtime attributes to the host orchestrator."""
        return getattr(self._orchestrator, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Proxy unresolved compatibility/runtime state writes to the host orchestrator."""
        if name == "_orchestrator" or hasattr(type(self), name):
            object.__setattr__(self, name, value)
            return
        setattr(self._orchestrator, name, value)

    def __delattr__(self, name: str) -> None:
        """Proxy unresolved compatibility/runtime state deletes to the host orchestrator."""
        if name == "_orchestrator" or hasattr(type(self), name):
            object.__delattr__(self, name)
            return
        delattr(self._orchestrator, name)

    def _resolve_runtime_owner_attr(self, name: str) -> Any:
        """Read an orchestrator runtime attribute without triggering MagicMock auto-creation."""
        orch = self._orchestrator
        instance_dict = getattr(orch, "__dict__", {})
        if isinstance(instance_dict, dict) and name in instance_dict:
            return instance_dict[name]
        if hasattr(type(orch), name):
            return getattr(orch, name, None)
        return None

    def _get_required_runtime_helper(self, getter_name: str) -> Any:
        """Return a canonical runtime helper instance or raise a clear compatibility error."""
        runtime_getter = self._resolve_runtime_owner_attr(getter_name)
        if not callable(runtime_getter):
            raise AttributeError(
                f"Canonical runtime helper {getter_name} is required on the orchestrator host"
            )
        return runtime_getter()

    def _get_planning_chat_runtime(self) -> Any:
        """Return the canonical planning runtime helper from the orchestrator host."""
        return self._get_required_runtime_helper("_get_planning_chat_runtime")

    def _get_context_limit_runtime(self) -> Any:
        """Return the canonical context-limit runtime helper from the orchestrator host."""
        return self._get_required_runtime_helper("_get_context_limit_runtime")

    def _require_runtime_helper_handler(self, getter_name: str, handler_name: str) -> Any:
        """Return a canonical runtime helper method or raise a clear compatibility error."""
        runtime = self._get_required_runtime_helper(getter_name)
        runtime_handler = getattr(runtime, handler_name, None)
        if not callable(runtime_handler):
            raise AttributeError(
                f"Canonical runtime helper {getter_name} must expose callable {handler_name}()"
            )
        return runtime_handler

    # =====================================================================
    # ExecutionProvider Implementation
    # =====================================================================

    async def execute_turn(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Execute a single model turn.

        Delegates to the orchestrator's provider.chat() method.

        Args:
            messages: Conversation history
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional tool definitions
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse from the model
        """
        # W3d/G7 D4: internal orchestration/topology hints ride **kwargs from
        # the runtime but are not provider params — strip them here so they
        # never reach a request body (the sandhi codec has a backstop strip,
        # but this is the root-cause fix and covers every provider path).
        kwargs = {key: value for key, value in kwargs.items() if key not in _INTERNAL_KWARG_KEYS}
        orch = self._orchestrator
        return await orch.provider.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )

    # =====================================================================
    # ToolExecutor Implementation
    # =====================================================================

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single tool with arguments.

        Delegates to the orchestrator's tool_executor.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Dict with tool execution result
        """
        orch = self._orchestrator
        context = {
            "code_manager": orch.code_manager,
            "provider": orch.provider,
            "model": orch.model,
            "tool_registry": orch.tools,
            "workflow_registry": orch.workflow_registry,
            "settings": orch.settings,
        }

        exec_result = await orch.tool_executor.execute(
            tool_name=tool_name,
            arguments=arguments,
            context=context,
        )

        return {
            "success": exec_result.success,
            "result": exec_result.result if exec_result.success else None,
            "error": exec_result.error if not exec_result.success else None,
        }

    async def execute_tool_calls(
        self,
        tool_calls: List[Any],
    ) -> List[Dict[str, Any]]:
        """Execute multiple tool calls via the canonical orchestrator surface.

        Prefer IToolCoordinator for new implementations.
        """
        orch = self._orchestrator
        return await orch.execute_tool_calls(tool_calls)

    async def chat(self, user_message: str, use_planning: bool = False) -> Any:
        """Execute a non-streaming chat turn through the canonical surface."""
        return await self._orchestrator.chat(
            user_message,
            use_planning=use_planning,
        )

    async def stream_chat(self, user_message: str, **kwargs: Any) -> AsyncIterator[Any]:
        """Execute a streaming chat turn through the canonical surface."""
        async for chunk in self._orchestrator.stream_chat(user_message, **kwargs):
            yield chunk

    def get_last_skill_match_info(self) -> Optional[Dict[str, Any]]:
        """Expose the sync chat fallback metadata surface."""
        return self._orchestrator.get_last_skill_match_info()

    # =====================================================================
    # MessageStore Implementation
    # =====================================================================

    @property
    def messages(self) -> List[Message]:
        """Get current message history."""
        return self._orchestrator.messages

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Add message to history.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
        """
        self._orchestrator.add_message(role, content, **metadata)

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from history.

        Args:
            limit: Optional limit on number of messages

        Returns:
            List of messages
        """
        messages = self._orchestrator.messages
        if limit is not None:
            return messages[-limit:]
        return messages

    def clear_messages(self) -> None:
        """Clear message history."""
        self._orchestrator.messages.clear()

    @property
    def conversation(self) -> "MessageHistory":
        """Get conversation manager."""
        return self._orchestrator.conversation

    # =====================================================================
    # StateManager Implementation
    # =====================================================================

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            State value or default
        """
        return getattr(self._orchestrator, key, default)

    def set_state(self, key: str, value: Any) -> None:
        """Set state value.

        Args:
            key: State key
            value: State value
        """
        setattr(self._orchestrator, key, value)

    def delete_state(self, key: str) -> bool:
        """Delete state value.

        Args:
            key: State key

        Returns:
            True if key was found and deleted, False otherwise
        """
        if hasattr(self._orchestrator, key):
            delattr(self._orchestrator, key)
            return True
        return False

    def get_all_state(self) -> Dict[str, Any]:
        """Get all state values.

        Returns:
            Dictionary of all state key-value pairs
        """
        # Return commonly accessed state attributes
        return {
            "tool_calls_used": self._orchestrator.tool_calls_used,
            "tool_budget": self._orchestrator.tool_budget,
            "model": self._orchestrator.model,
            "temperature": self._orchestrator.temperature,
            "max_tokens": self._orchestrator.max_tokens,
            "_system_added": (
                self._orchestrator.get_capability_value("system_prompt_added")
                if self._orchestrator.has_capability("system_prompt_added")
                else False
            ),
        }

    def clear_state(self) -> None:
        """Clear all state values.

        Note: This only clears temporary state, not configuration.
        """
        self._orchestrator.tool_calls_used = 0

    # =====================================================================
    # ChatContextProtocol Helpers (from chat_protocols.py)
    # =====================================================================

    @property
    def settings(self) -> "Settings":
        """Get settings."""
        return self._orchestrator.settings

    @property
    def conversation_controller(self) -> "ConversationController":
        """Get conversation controller."""
        return self._orchestrator.conversation_controller

    @property
    def _context_compactor(self) -> "ContextCompactor":
        """Get context compactor."""
        return self._orchestrator._context_compactor

    @property
    def _session_state(self) -> "SessionStateManager":
        """Get session state manager."""
        return self._orchestrator._session_state

    @property
    def _cumulative_token_usage(self) -> Dict[str, int]:
        """Get cumulative token usage."""
        return self._orchestrator._cumulative_token_usage

    def _check_context_overflow(self, max_context: int) -> bool:
        """Check if conversation exceeds context window limit."""
        return self._orchestrator._check_context_overflow(max_context)

    def _get_max_context_chars(self) -> int:
        """Get maximum context length in characters."""
        return self._orchestrator._get_max_context_chars()

    # =====================================================================
    # ToolContextProtocol Helpers (from chat_protocols.py)
    # =====================================================================

    @property
    def tool_selector(self) -> "ToolSelector":
        """Get tool selector."""
        return self._orchestrator.tool_selector

    @property
    def tool_adapter(self) -> Any:
        """Get tool adapter."""
        return self._orchestrator.tool_adapter

    @property
    def _tool_planner(self) -> "ToolPlanner":
        """Get tool planner."""
        return self._orchestrator._tool_planner

    @property
    def tool_budget(self) -> int:
        """Get tool budget."""
        return self._orchestrator.tool_budget

    @tool_budget.setter
    def tool_budget(self, value: int) -> None:
        """Set tool budget."""
        self._orchestrator.tool_budget = value

    @property
    def tool_calls_used(self) -> int:
        """Get tool calls used."""
        return self._orchestrator.tool_calls_used

    @tool_calls_used.setter
    def tool_calls_used(self, value: int) -> None:
        """Set tool calls used."""
        self._orchestrator.tool_calls_used = value

    @property
    def use_semantic_selection(self) -> bool:
        """Get semantic selection flag."""
        return self._orchestrator.use_semantic_selection

    @property
    def observed_files(self) -> set:
        """Get observed files set."""
        return self._orchestrator.observed_files

    def _model_supports_tool_calls(self) -> bool:
        """Check if model supports tool calls."""
        return self._orchestrator._model_supports_tool_calls()

    # =====================================================================
    # ProviderContextProtocol Helpers (from chat_protocols.py)
    # =====================================================================

    @property
    def task_classifier(self) -> Any:
        """Get task classifier."""
        return self._orchestrator.task_classifier

    @property
    def task_analyzer(self) -> "TaskAnalyzer":
        """Get task analyzer."""
        return self._orchestrator.task_analyzer

    @property
    def response_completer(self) -> Any:
        """Get response completer."""
        return self._orchestrator.response_completer

    @property
    def provider(self) -> "BaseProvider":
        """Get LLM provider."""
        return self._orchestrator.provider

    @property
    def model(self) -> str:
        """Get current model."""
        return self._orchestrator.model

    @property
    def temperature(self) -> float:
        """Get temperature."""
        return self._orchestrator.temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set temperature."""
        self._orchestrator.temperature = value

    @property
    def reasoning_effort(self) -> Optional[str]:
        """Get the configured reasoning effort (None = inherit/omit)."""
        return getattr(self._orchestrator, "reasoning_effort", None)

    def supports_reasoning_effort(self, model: Optional[str] = None) -> bool:
        """Whether the underlying provider accepts reasoning_effort for ``model``."""
        provider = getattr(self._orchestrator, "provider", None)
        if provider is None:
            return False
        try:
            return bool(provider.supports_reasoning_effort(model))
        except Exception:  # pragma: no cover - capability probe must not break the turn
            return False

    @property
    def max_tokens(self) -> int:
        """Get max tokens."""
        return self._orchestrator.max_tokens

    @property
    def profile(self) -> Any:
        """Get active profile configuration if available."""
        return getattr(self._orchestrator, "profile", None)

    @property
    def planning_model_override(self) -> Any:
        """Get CLI planning model override if available."""
        return getattr(self._orchestrator, "_planning_model_override", None)

    @property
    def _system_prompt_override(self) -> Any:
        """Get the current temporary system prompt override, if any."""
        return getattr(self._orchestrator, "_system_prompt_override", None)

    def set_system_prompt(self, prompt: str) -> None:
        """Set a temporary system prompt override for planning helpers."""
        self._orchestrator.set_system_prompt(prompt)

    @property
    def thinking(self) -> Any:
        """Get thinking configuration."""
        return self._orchestrator.thinking

    @property
    def task_classifier(self) -> Any:
        """Get task classifier."""
        return self._orchestrator.task_classifier

    @property
    def task_analyzer(self) -> "TaskAnalyzer":
        """Get task analyzer."""
        return self._orchestrator.task_analyzer

    @property
    def response_completer(self) -> Any:
        """Get response completer."""
        return self._orchestrator.response_completer

    @property
    def _provider_service(self) -> Any:
        """Get canonical provider service for chat compatibility helpers."""
        return self._orchestrator._provider_service

    @property
    def _cancel_event(self) -> Optional["asyncio.Event"]:
        """Get cancel event."""
        return self._orchestrator._cancel_event

    @_cancel_event.setter
    def _cancel_event(self, value: Optional["asyncio.Event"]) -> None:
        """Set cancel event."""
        self._orchestrator._cancel_event = value

    @property
    def _is_streaming(self) -> bool:
        """Get streaming flag."""
        return self._orchestrator._is_streaming

    @_is_streaming.setter
    def _is_streaming(self, value: bool) -> None:
        """Set streaming flag."""
        self._orchestrator._is_streaming = value

    def _check_cancellation(self) -> bool:
        """Check if cancellation requested."""
        return self._orchestrator._check_cancellation()


__all__ = [
    "OrchestratorProtocolAdapter",
]
