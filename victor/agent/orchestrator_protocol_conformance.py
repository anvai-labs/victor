"""Protocol-conformance accessors for the orchestrator facade.

Moved verbatim from ``victor/agent/orchestrator.py`` (facade decomposition):
stage/budget/file/iteration counters, provider/model properties and
hot-swap methods, and the system-prompt/message accessors. The
ToolsProtocol block and reset/context-manager/from_settings stay in the
orchestrator class body (service-delegation floors and patch targets).
"""

from typing import Any, Dict, List, Optional, Set

from victor.agent.conversation.state_machine import ConversationStage


class OrchestratorProtocolConformanceMixin:
    """Moved verbatim from orchestrator.py; see module docstring."""

    def get_stage(self) -> "ConversationStage":
        """Get current conversation stage (protocol method).

        Returns:
            Current ConversationStage enum value

        Note:
            Framework layer converts this to framework.state.Stage
        """
        if self.conversation_state:
            return self.conversation_state.get_stage()
        return ConversationStage.INITIAL

    def get_tool_calls_count(self) -> int:
        """Get total tool calls made (protocol method), matching get_tool_budget().

        Returns:
            Non-negative count of tool calls in this session
        """
        return int(getattr(self, "tool_calls_used", 0) or 0)

    def get_tool_budget(self) -> int:
        """Get the *enforcing* tool budget (protocol method).

        ``self.tool_budget`` seeds ``StreamingChatContext`` and halts the loop;
        ``UnifiedTaskTracker`` is advisory by design and must not be quoted here.

        Returns:
            Maximum allowed tool calls
        """
        return int(getattr(self, "tool_budget", 50) or 50)

    def get_observed_files(self) -> Set[str]:
        """Get files observed/read during conversation (protocol method).

        Returns:
            Set of absolute file paths
        """
        return set(getattr(self, "observed_files", []))

    def get_modified_files(self) -> Set[str]:
        """Get files modified during conversation (protocol method).

        Returns:
            Set of absolute file paths
        """
        if self.conversation_state and hasattr(self.conversation_state, "state"):
            return set(getattr(self.conversation_state.state, "modified_files", []))
        return set()

    def get_iteration_count(self) -> int:
        """Get current agent loop iteration count (protocol method).

        Returns:
            Non-negative iteration count
        """
        if self.unified_tracker:
            return self.unified_tracker.iteration_count
        return 0

    def get_max_iterations(self) -> int:
        """Get maximum allowed iterations (protocol method).

        Returns:
            Max iteration limit
        """
        if self.unified_tracker:
            return self.unified_tracker.max_iterations
        return 25

    # --- ProviderProtocol ---

    @property
    def current_provider(self) -> str:
        """Get current provider name (protocol property).

        Returns:
            Provider identifier (e.g., "anthropic", "openai")
        """
        return self.provider_name

    @property
    def current_model(self) -> str:
        """Get current model name (protocol property).

        Returns:
            Model identifier
        """
        return self.model

    async def switch_provider(
        self,
        provider_name: str,
        model: Optional[str] = None,
        on_switch: Optional[Any] = None,
    ) -> bool:
        """[CANONICAL] Switch provider via ProviderService.

        Delegates to ProviderService for orchestration, maintaining the
        legacy coordinator in sync for backward compatibility.

        Args:
            provider_name: Target provider name
            model: Optional specific model
            on_switch: Optional callback after switch
        """
        return await self._get_provider_management_runtime().switch_provider(
            provider_name,
            model,
            on_switch,
        )

    async def switch_model(self, model: str) -> bool:
        """[CANONICAL] Switch model via ProviderService."""
        return await self._get_provider_management_runtime().switch_model(model)

    def get_system_prompt(self) -> str:
        """Get current system prompt (protocol method).

        Returns:
            Complete system prompt string
        """
        return self._get_prompt_builder_runtime().get_system_prompt()

    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt (protocol method).

        Args:
            prompt: New system prompt (replaces existing)
        """
        self._get_prompt_builder_runtime().set_system_prompt(prompt)

    def append_to_system_prompt(self, content: str) -> None:
        """Append content to system prompt (protocol method).

        Args:
            content: Content to append
        """
        self._get_prompt_builder_runtime().append_to_system_prompt(content)

    # --- MessagesProtocol ---

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get conversation messages (protocol method).

        Returns:
            List of message dictionaries
        """
        return [{"role": m.role, "content": m.content} for m in self.conversation.messages]

    def get_message_count(self) -> int:
        """Get message count (protocol method).

        Returns:
            Number of messages in conversation
        """
        return len(self.conversation.messages)
