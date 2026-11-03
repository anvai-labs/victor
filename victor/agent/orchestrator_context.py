"""Context/cache accessors for the orchestrator facade.

Moved verbatim from ``victor/agent/orchestrator.py`` (facade decomposition).
get_context_metrics stays in the class body (service-delegation floor).
"""

import logging
from typing import Any, List, Optional

from victor.providers.base import Message

logger = logging.getLogger(__name__)


class OrchestratorContextMixin:
    """Moved verbatim from orchestrator.py; see module docstring."""

    @property
    def messages(self) -> List[Message]:
        """Get conversation messages (backward compatibility property).

        Returns:
            List of messages in conversation history
        """
        return self.conversation.messages

    def _get_conversation_message_count(self) -> int:
        """Return the current conversation length across history implementations."""
        from victor.agent.conversation.context_assembly import conversation_message_count

        return conversation_message_count(self)

    def _get_model_context_window(self) -> int:
        """Get context window size for the current model.

        Delegates to ContextManager when available (TD-002 refactoring).
        Falls back to direct implementation during __init__ before
        ContextManager is created.

        Returns:
            Context window size in tokens
        """
        # Delegate to ContextManager if available
        if hasattr(self, "_context_manager") and self._context_manager is not None:
            return self._context_manager.get_model_context_window()

        # Fallback for calls during __init__ before ContextManager is created
        try:
            from victor.config.config_loaders import get_provider_limits

            limits = get_provider_limits(self.provider_name, self.model)
            return limits.context_window
        except Exception as e:
            logger.warning(f"Could not load provider limits from config: {e}")
            return 128000  # Default safe value

    def _get_max_context_chars(self) -> int:
        """Get maximum context size in characters.

        Delegates to ContextManager (TD-002 refactoring).

        Returns:
            Maximum context size in characters
        """
        return self._context_manager.get_max_context_chars()

    def _check_cache_setting_enabled(self) -> bool:
        """Check if cache optimization is enabled in settings."""
        return self._get_prompt_builder_runtime().check_cache_setting_enabled()

    @property
    def _cache_optimization_enabled(self) -> bool:
        """Full API-level cache optimization — session-lock all tools + freeze prompt.

        Uses cached flag from _compute_cache_flags() when available.
        """
        return self._get_prompt_builder_runtime().is_cache_optimization_enabled()

    def _compute_cache_flags(self) -> None:
        """Compute and cache optimization flags once (called after provider init).

        Caches _kv_opt_cached and _cache_opt_cached so the @property accessors
        don't repeat try/except/getattr/hasattr chains on every access.
        """
        self._get_prompt_builder_runtime().compute_cache_flags()

    @property
    def _kv_optimization_enabled(self) -> bool:
        """KV prefix cache optimization — freeze prompt for prefix reuse.

        Uses cached flag from _compute_cache_flags() when available.
        """
        return self._get_prompt_builder_runtime().is_kv_optimization_enabled()

    async def warm_up_kv_cache(self) -> None:
        """Prime the KV cache by sending a minimal request with the system prompt.

        For KV prefix caching providers (Ollama, LMStudio), the first API call
        is always cold — full prefill computation. This method sends a 1-token
        completion to prime the KV cache so subsequent calls reuse the prefix.

        No-op when KV optimization is disabled.
        """
        await self._get_prompt_builder_runtime().warm_up_kv_cache()

    def _kv_prefix_fingerprint(self) -> str:
        """Compute a short fingerprint of the system prompt for KV cache observability.

        Returns a hex hash of the first 500 chars of the system prompt. Identical
        hashes across turns indicate the KV prefix is stable (likely cache hit).
        """
        return self._get_prompt_builder_runtime().kv_prefix_fingerprint()

    def get_session_tools(self) -> Optional[list]:
        """Get session-locked tools for cache-friendly API calls.

        Returns the currently enabled tool set (frozen at session start) so the
        tool prefix remains byte-identical across API calls in the same session.
        When cache optimization is disabled, returns None (use per-turn).
        """
        from victor.agent.conversation.context_assembly import session_tools

        return session_tools(self)

    def get_assembled_messages(
        self,
        current_query: Optional[str] = None,
        *,
        selected_tools: Optional[list[Any]] = None,
        planned_tools: Optional[list[Any]] = None,
        goals: Optional[list[str]] = None,
    ) -> List["Message"]:
        """Get context-assembled messages for provider calls.

        When context assembler is available: keeps system prompt + ledger +
        last N full turns + score-selected older messages within budget.
        Falls back to raw self.messages when assembler unavailable.

        When the canonical prompt pipeline is available, dynamic prompt guidance
        (skills, reminders, GEPA/MiPRO/CoT optimizations, credit guidance) is
        prepended to the last user message instead of being injected as system
        messages. For KV-capable providers this also keeps the system prompt
        byte-identical across turns for provider prefix caching.

        Implementation lives in victor.agent.conversation.context_assembly
        (self-analysis round 2, W3) — this facade method keeps its name and
        mock.patch target.
        """
        from victor.agent.conversation.context_assembly import assembled_messages_for_call

        return assembled_messages_for_call(
            self,
            current_query,
            selected_tools=selected_tools,
            planned_tools=planned_tools,
            goals=goals,
        )

    def _check_context_overflow(self, max_context_chars: int = 200000) -> bool:
        """Check if context is at risk of overflow.

        Args:
            max_context_chars: Maximum allowed context size in chars

        Returns:
            True if context is dangerously large
        """
        # Use the sync context manager for context overflow check.
        # ContextService is async and may not accept max_context_chars,
        # so delegate to the conversation controller's sync check.
        return self._context_manager.check_context_overflow(max_context_chars)

    def _init_conversation_embedding_store(self) -> None:
        """Initialize embedding store via the memory runtime."""
        from victor.agent.runtime.memory_runtime import (
            initialize_conversation_embedding_store,
        )

        store, cache = initialize_conversation_embedding_store(
            memory_manager=self.memory_manager,
        )
        self._conversation_embedding_store = store
        if cache is not None:
            self._pending_semantic_cache = cache
