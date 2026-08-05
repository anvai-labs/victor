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

"""Context management for conversation size and overflow handling.

This module provides centralized context window management extracted from
AgentOrchestrator as part of the TD-002 God Class refactoring effort.

The ContextManager handles:
- Model context window queries
- Character-to-token conversion with safety margins
- Context overflow detection and logging
- Proactive compaction coordination

Design Pattern: Dependency Injection
====================================
ContextManager receives its dependencies through the constructor rather
than creating them internally. This enables testability and reduces
coupling to concrete implementations.

Usage:
    config = ContextManagerConfig(max_context_chars=200000)
    context_manager = ContextManager(
        config=config,
        provider_name="anthropic",
        model="claude-sonnet-4-20250514",
        conversation_controller=controller,
        context_compactor=compactor,
        debug_logger=debug_logger,
        settings=settings,
    )

    # Check for overflow
    if context_manager.check_context_overflow():
        # Handle overflow...

    # Get max context in chars
    max_chars = context_manager.get_max_context_chars()
"""

from __future__ import annotations

import logging
import inspect
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from victor.providers.base import StreamChunk

if TYPE_CHECKING:
    from victor.agent.conversation.controller import (
        ConversationController,
        ContextMetrics,
    )
    from victor.agent.context_compactor import ContextCompactor
    from victor.agent.debug_logger import DebugLogger
    from victor.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class ContextManagerConfig:
    """Configuration for ContextManager.

    Attributes:
        max_context_chars: Override for maximum context size in characters.
            If None, derived from model context window.
        chars_per_token: Approximate characters per token for estimation.
            Default is 3.5 which is conservative for most models.
        safety_margin: Fraction of context window to use (0.0-1.0).
            Default is 0.8 (use 80% of available context).
        default_context_window: Fallback context window size in tokens
            if provider limits cannot be loaded.
    """

    max_context_chars: Optional[int] = None
    chars_per_token: float = 3.5
    safety_margin: float = 0.8
    default_context_window: int = 128000


class ContextManager:
    """Manages context window limits and overflow handling.

    This class centralizes context-related operations that were previously
    scattered across AgentOrchestrator. It provides:

    1. Model context window queries via provider limits config
    2. Character-based context size limits with token conversion
    3. Overflow detection with debug logging
    4. Proactive compaction coordination

    The class uses dependency injection to receive its collaborators,
    making it testable and loosely coupled.
    """

    def __init__(
        self,
        config: ContextManagerConfig,
        provider_name: str,
        model: str,
        conversation_controller: "ConversationController",
        context_compactor: Optional["ContextCompactor"] = None,
        context_service: Optional[Any] = None,
        debug_logger: Optional["DebugLogger"] = None,
        settings: Optional["Settings"] = None,
    ):
        """Initialize ContextManager.

        Args:
            config: Configuration for context limits and margins.
            provider_name: Name of the LLM provider (e.g., "anthropic").
            model: Model identifier (e.g., "claude-sonnet-4-20250514").
            conversation_controller: Controller for accessing context metrics.
            context_compactor: Optional compactor for proactive compaction.
            context_service: Optional context service for service-owned compaction.
            debug_logger: Optional debug logger for context size logging.
            settings: Optional settings for override values.
        """
        self._config = config
        self._provider_name = provider_name
        self._model = model
        self._conversation_controller = conversation_controller
        self._context_compactor = context_compactor
        self._context_service = context_service
        self._debug_logger = debug_logger
        self._settings = settings

    def get_model_context_window(self) -> int:
        """Get context window size for the current model.

        Queries the provider limits config for model-specific context window.

        Returns:
            Context window size in tokens
        """
        try:
            from victor.config.config_loaders import get_provider_limits

            limits = get_provider_limits(self._provider_name, self._model)
            return limits.context_window
        except Exception as e:
            logger.warning(f"Could not load provider limits from config: {e}")
            return self._config.default_context_window

    def get_max_context_chars(self) -> int:
        """Get maximum context size in characters.

        Derives from model context window, converting tokens to chars.
        Uses configured chars_per_token ratio with safety margin.

        Returns:
            Maximum context size in characters
        """
        # Check settings override first
        if self._settings is not None:
            settings_max = getattr(self._settings, "max_context_chars", None)
            if settings_max and settings_max > 0:
                return settings_max

        # Check config override
        if self._config.max_context_chars is not None and self._config.max_context_chars > 0:
            return self._config.max_context_chars

        # Calculate from model context window
        context_tokens = self.get_model_context_window()
        return int(context_tokens * self._config.chars_per_token * self._config.safety_margin)

    def check_context_overflow(self, max_context_chars: Optional[int] = None) -> bool:
        """Check if context is at risk of overflow.

        Args:
            max_context_chars: Maximum allowed context size in chars.
                If None, uses default from config/model.

        Returns:
            True if context is dangerously large
        """
        if max_context_chars is None:
            max_context_chars = self.get_max_context_chars()

        # Delegate to ConversationController
        metrics = self._conversation_controller.get_context_metrics()

        # Update debug logger if available
        if self._debug_logger is not None:
            self._debug_logger.log_context_size(metrics.char_count, metrics.estimated_tokens)

        if metrics.is_overflow_risk:
            logger.warning(
                f"Context overflow risk: {metrics.char_count:,} chars "
                f"(~{metrics.estimated_tokens:,} tokens). "
                f"Max: {metrics.max_context_chars:,} chars"
            )
            return True

        return False

    def handle_compaction(self, user_message: str) -> Optional["StreamChunk"]:
        """Perform proactive compaction if enabled.

        Args:
            user_message: Current user message for semantic relevance.

        Returns:
            StreamChunk with compaction notification if compaction occurred,
            None otherwise.
        """
        if self._has_context_service_compaction():
            service_handled, service_chunk = self._handle_context_service_compaction()
            if service_handled:
                return service_chunk

        if self._context_compactor is None:
            return None

        compaction_action = self._context_compactor.check_and_compact(current_query=user_message)
        if not compaction_action.action_taken:
            return None

        logger.info(
            f"Proactive compaction: {compaction_action.trigger.value}, "
            f"removed {compaction_action.messages_removed} messages, "
            f"freed {compaction_action.chars_freed:,} chars"
        )

        chunk: Optional[StreamChunk] = None
        if compaction_action.messages_removed > 0:
            chunk = StreamChunk(
                content=(
                    f"\n[context] Proactively compacted history "
                    f"({compaction_action.messages_removed} messages, "
                    f"{compaction_action.chars_freed:,} chars freed).\n"
                )
            )
            # Inject context reminder about compacted content
            self._conversation_controller.inject_compaction_context()

        return chunk

    def _handle_context_service_compaction(
        self,
    ) -> tuple[bool, Optional["StreamChunk"]]:
        """Use ContextService for sync compaction when it exposes sync methods."""
        recommendation_getter = getattr(
            self._context_service,
            "get_compaction_recommendation",
            None,
        )
        compact_context = getattr(self._context_service, "compact_context", None)

        recommendation = recommendation_getter()
        if inspect.isawaitable(recommendation):
            return False, None
        if isinstance(recommendation, dict) and not recommendation.get("should_compact", False):
            return True, None

        removed = compact_context(strategy=self._context_compaction_strategy(), min_messages=6)
        if inspect.isawaitable(removed):
            return False, None
        removed_count = int(removed or 0)
        if removed_count <= 0:
            return True, None

        logger.info(
            "ContextService proactive compaction: removed %s messages",
            removed_count,
        )
        self._conversation_controller.inject_compaction_context()
        return True, StreamChunk(
            content=(
                "\n[context] ContextService compacted history "
                f"({removed_count} messages removed).\n"
            )
        )

    async def handle_compaction_async(self, user_message: str) -> Optional["StreamChunk"]:
        """Perform proactive compaction asynchronously if enabled.

        Non-blocking version of handle_compaction for use in async hot paths.

        Args:
            user_message: Current user message for semantic relevance.

        Returns:
            StreamChunk with compaction notification if compaction occurred,
            None otherwise.
        """
        if self._has_context_service_compaction():
            return await self._handle_context_service_compaction_async()

        if self._context_compactor is None:
            return None

        compaction_action = await self._context_compactor.check_and_compact_async(
            current_query=user_message
        )
        if not compaction_action.action_taken:
            return None

        logger.info(
            f"Proactive compaction (async): {compaction_action.trigger.value}, "
            f"removed {compaction_action.messages_removed} messages, "
            f"freed {compaction_action.chars_freed:,} chars"
        )

        chunk: Optional[StreamChunk] = None
        if compaction_action.messages_removed > 0:
            chunk = StreamChunk(
                content=(
                    f"\n[context] Proactively compacted history "
                    f"({compaction_action.messages_removed} messages, "
                    f"{compaction_action.chars_freed:,} chars freed).\n"
                )
            )
            # Inject context reminder about compacted content
            self._conversation_controller.inject_compaction_context()

        return chunk

    def _has_context_service_compaction(self) -> bool:
        """Return whether ContextService owns async compaction for this manager."""
        if self._context_service is None:
            return False
        recommendation_getter = getattr(
            self._context_service,
            "get_compaction_recommendation",
            None,
        )
        compact_context = getattr(self._context_service, "compact_context", None)
        return callable(recommendation_getter) and callable(compact_context)

    async def _handle_context_service_compaction_async(self) -> Optional["StreamChunk"]:
        """Use ContextService for async compaction when it is wired."""
        recommendation_getter = getattr(
            self._context_service,
            "get_compaction_recommendation",
            None,
        )
        compact_context = getattr(self._context_service, "compact_context", None)

        recommendation = recommendation_getter()
        if inspect.isawaitable(recommendation):
            recommendation = await recommendation
        if isinstance(recommendation, dict) and not recommendation.get("should_compact", False):
            return None

        removed = compact_context(strategy=self._context_compaction_strategy(), min_messages=6)
        if inspect.isawaitable(removed):
            removed = await removed
        removed_count = int(removed or 0)
        if removed_count <= 0:
            return None

        logger.info(
            "ContextService proactive compaction: removed %s messages",
            removed_count,
        )
        self._conversation_controller.inject_compaction_context()
        return StreamChunk(
            content=(
                "\n[context] ContextService compacted history "
                f"({removed_count} messages removed).\n"
            )
        )

    def _context_compaction_strategy(self) -> str:
        return str(getattr(self._settings, "context_compaction_strategy", "tiered") or "tiered")

    def get_context_metrics(self) -> "ContextMetrics":
        """Get detailed context metrics.

        Returns:
            ContextMetrics with size and overflow information
        """
        return self._conversation_controller.get_context_metrics()

    @property
    def config(self) -> ContextManagerConfig:
        """Get the configuration."""
        return self._config

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def model(self) -> str:
        """Get the model identifier."""
        return self._model

    @property
    def is_background_compaction_running(self) -> bool:
        """Check if background compaction is currently active.

        Returns:
            True if background compaction task is running.
        """
        if self._context_compactor is None:
            return False
        return getattr(self._context_compactor, "_async_running", False)

    async def start_background_compaction(self, interval_seconds: float = 30.0) -> None:
        """Start background compaction task.

        Args:
            interval_seconds: How often to check for compaction (default 30s).
        """
        if self._context_compactor is None:
            return
        await self._context_compactor.start_background_compaction(interval_seconds)

    async def stop_background_compaction(self) -> None:
        """Stop background compaction task."""
        if self._context_compactor is None:
            return
        await self._context_compactor.stop_background_compaction()


def create_context_manager(
    provider_name: str,
    model: str,
    conversation_controller: "ConversationController",
    context_compactor: Optional["ContextCompactor"] = None,
    context_service: Optional[Any] = None,
    debug_logger: Optional["DebugLogger"] = None,
    settings: Optional["Settings"] = None,
    config: Optional[ContextManagerConfig] = None,
) -> ContextManager:
    """Factory function to create a ContextManager with defaults.

    This provides a convenient way to create a ContextManager with sensible
    defaults while allowing customization through the config parameter.

    Args:
        provider_name: Name of the LLM provider.
        model: Model identifier.
        conversation_controller: Controller for accessing context metrics.
        context_compactor: Optional compactor for proactive compaction.
        context_service: Optional context service for service-owned compaction.
        debug_logger: Optional debug logger for context size logging.
        settings: Optional settings for override values.
        config: Optional custom configuration. If None, uses defaults.

    Returns:
        Configured ContextManager instance
    """
    if config is None:
        config = ContextManagerConfig()

    return ContextManager(
        config=config,
        provider_name=provider_name,
        model=model,
        conversation_controller=conversation_controller,
        context_compactor=context_compactor,
        context_service=context_service,
        debug_logger=debug_logger,
        settings=settings,
    )
