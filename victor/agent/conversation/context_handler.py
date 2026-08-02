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

"""Context overflow detection and handling.

This module provides ContextOverflowHandler, which handles context
overflow detection and compaction. Extracted from ConversationManager
to follow the Single Responsibility Principle (SRP).

Part of SOLID-based refactoring to eliminate god class anti-pattern.
"""

import logging
from typing import Any, Dict, List, Optional

from victor.agent.conversation.controller import ConversationController, ContextMetrics
from victor.agent.protocols import IContextOverflowHandler

logger = logging.getLogger(__name__)


class ContextOverflowHandler(IContextOverflowHandler):
    """Handles context overflow detection and compaction.

    This class is responsible for:
    - Checking if context has overflowed
    - Handling context compaction
    - Getting context metrics
    - Getting memory context with token limits

    SRP Compliance: Focuses only on context management, delegating
    message storage, session management, and embeddings to specialized components.

    Attributes:
        _controller: ConversationController for context operations
        _max_context_chars: Maximum context size in characters
        _chars_per_token: Estimate for token calculation
    """

    def __init__(
        self,
        controller: ConversationController,
        max_context_chars: int = 200000,
        chars_per_token: int = 3,
    ):
        """Initialize the context overflow handler.

        Args:
            controller: ConversationController for context operations
            max_context_chars: Maximum context size in characters
            chars_per_token: Estimate for token calculation
        """
        self._controller = controller
        self._max_context_chars = max_context_chars
        self._chars_per_token = chars_per_token

    def check_overflow(self) -> bool:
        """Check if context has overflowed.

        Returns:
            True if overflow detected, False otherwise
        """
        metrics = self._controller.get_context_metrics()
        return metrics.total_chars > self._max_context_chars

    def handle_compaction(
        self,
        strategy: str = "auto",
        target_ratio: float = 0.7,
        preserve_system_prompt: bool = True,
    ) -> Optional[ContextMetrics]:
        """Handle context compaction.

        Args:
            strategy: Compaction strategy (auto, recent, semantic)
            target_ratio: Target ratio to reduce to
            preserve_system_prompt: Whether to preserve system prompt

        Returns:
            New context metrics after compaction, or None if not compacted
        """
        if not self.check_overflow():
            return self._controller.get_context_metrics()

        try:
            logger.info(
                f"Context overflow detected ({self._controller.get_context_metrics().total_chars} chars), "
                f"compacting with strategy='{strategy}'"
            )

            # Delegate to controller for compaction
            metrics = self._controller.compact_context(
                strategy=strategy,
                target_ratio=target_ratio,
                preserve_system_prompt=preserve_system_prompt,
            )

            logger.info(
                f"Compaction complete: {metrics.total_chars} chars, "
                f"{metrics.message_count} messages"
            )

            return metrics

        except Exception as e:
            logger.error(f"Compaction failed: {e}")
            return None

    def get_context_metrics(self) -> ContextMetrics:
        """Get current context metrics.

        Returns:
            ContextMetrics with current context statistics
        """
        return self._controller.get_context_metrics()

    def get_memory_context(
        self,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get memory context with optional token limit.

        Args:
            max_tokens: Optional token limit

        Returns:
            List of message dictionaries for memory context
        """
        # Convert messages to memory context format
        messages = self._controller.messages

        if max_tokens is None:
            # No limit, return all messages
            return [msg.to_dict() for msg in messages]

        # Calculate character limit
        char_limit = max_tokens * self._chars_per_token

        # Build context within character limit
        # Prioritize recent messages by iterating in reverse
        context = []
        total_chars = 0

        for message in reversed(messages):
            msg_dict = message.to_dict()
            msg_chars = len(msg_dict.get("content", ""))

            if total_chars + msg_chars > char_limit and context:
                # Would exceed limit, stop here
                break

            context.append(msg_dict)  # Append (will reverse once at end)
            total_chars += msg_chars

        # Reverse once to get correct order (oldest -> newest)
        return list(reversed(context))

    def set_max_context_chars(self, max_chars: int) -> None:
        """Set maximum context size in characters.

        Args:
            max_chars: Maximum context size
        """
        self._max_context_chars = max_chars

    def get_max_context_chars(self) -> int:
        """Get maximum context size in characters.

        Returns:
            Maximum context size
        """
        return self._max_context_chars
