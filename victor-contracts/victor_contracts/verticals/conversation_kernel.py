"""Conversation-manager kernel shared by every vertical's ``conversation_enhanced``.

Each vertical's ``EnhancedXConversationManager`` is the same shell: build a
``ConversationCoordinator``, hold the vertical's context object, delegate
message/history/summarization/stats to the coordinator, and expose
observability data keyed by the vertical. The vertical subclass keeps its
context dataclass, ``track_*`` recorders, its markdown summary renderer, and
its vertical-named context accessor::

    from victor_contracts.verticals import VerticalConversationManager

    class EnhancedDevOpsConversationManager(VerticalConversationManager):
        def __init__(self, **kwargs):
            super().__init__(
                DevOpsContext, "devops_context", "devops", **kwargs
            )

        def get_devops_context(self) -> DevOpsContext:
            return self.get_context()
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from victor_contracts import ConversationCoordinator, ConversationStats, TurnType

logger = logging.getLogger(__name__)


class VerticalConversationManager:
    """Coordinator-backed conversation manager with a vertical context object.

    Args:
        context_type: Zero-arg factory (usually the context dataclass) for the
            vertical's context object.
        context_key: The observability-data key for the context dict (e.g.
            ``"devops_context"``).
        vertical_name: The ``vertical`` value in observability data.
        max_history_turns / summarization_threshold / enable_deduplication /
            enable_statistics: Passed through to the coordinator.
    """

    def __init__(
        self,
        context_type: Callable[[], Any],
        context_key: str,
        vertical_name: str,
        *,
        max_history_turns: int = 50,
        summarization_threshold: int = 40,
        enable_deduplication: bool = True,
        enable_statistics: bool = True,
    ) -> None:
        self._conversation_coordinator = ConversationCoordinator(
            max_history_turns=max_history_turns,
            summarization_threshold=summarization_threshold,
            enable_deduplication=enable_deduplication,
            enable_statistics=enable_statistics,
        )
        self._context = context_type()
        self._context_key = context_key
        self._vertical_name = vertical_name
        logger.info(
            "%s initialized with max_turns=%s",
            type(self).__name__,
            max_history_turns,
        )

    # -- Message management (delegates to ConversationCoordinator) -----------

    def add_message(
        self,
        role: str,
        content: str,
        turn_type: TurnType,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Add a message; returns the turn id."""
        return self._conversation_coordinator.add_message(
            role, content, turn_type, metadata, tool_calls
        )

    def get_history(
        self,
        max_turns: Optional[int] = None,
        include_system: bool = True,
        include_tool: bool = True,
    ) -> List[Dict[str, Any]]:
        """Recent messages as dicts."""
        return self._conversation_coordinator.get_history(max_turns, include_system, include_tool)

    def clear_history(self, keep_summaries: bool = True) -> None:
        """Clear history, optionally keeping summaries."""
        self._conversation_coordinator.clear_history(keep_summaries)

    def needs_summarization(self) -> bool:
        """True when summarization is recommended."""
        return self._conversation_coordinator.needs_summarization()

    def add_summary(self, summary: str) -> None:
        """Record a conversation summary."""
        self._conversation_coordinator.add_summary(summary)

    # -- Accessors ------------------------------------------------------------

    def get_stats(self) -> ConversationStats:
        """Coordinator statistics."""
        return self._conversation_coordinator.get_stats()

    def get_context(self) -> Any:
        """The vertical's context object."""
        return self._context

    def get_conversation_coordinator(self) -> ConversationCoordinator:
        """The underlying coordinator."""
        return self._conversation_coordinator

    def get_observability_data(self) -> Dict[str, Any]:
        """Coordinator observability data plus the vertical context dict."""
        conv_obs = self._conversation_coordinator.get_observability_data()
        return {
            **conv_obs,
            self._context_key: self._context.to_dict(),
            "vertical": self._vertical_name,
        }
