# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Emergency Discard-All compaction (P2-2).

When context is critically full (>95%) and standard compaction isn't enough,
emergency compaction aggressively discards content while preserving user intent.

Safety features:
- Always preserves system prompt
- Always preserves user messages (original intent)
- Keeps minimum message count
- Logs emergency event for observability
- Injects warning message about context reset
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from victor.core.provider_messages import Message

logger = logging.getLogger(__name__)


@dataclass
class EmergencyCompactionConfig:
    """Configuration for emergency compaction.

    Attributes:
        critical_threshold: Utilization threshold to trigger emergency (0.0-1.0)
        min_messages_after_compact: Minimum messages to keep after compaction
        preserve_user_messages: Whether to preserve all user messages
        preserve_system_prompt: Whether to preserve system prompt
        max_tool_result_age: Maximum age (in turns) for tool results to keep
        inject_warning: Whether to inject warning about emergency compaction
    """

    critical_threshold: float = 0.95
    min_messages_after_compact: int = 3
    preserve_user_messages: bool = True
    preserve_system_prompt: bool = True
    max_tool_result_age: int = 2  # Keep only very recent tool results
    inject_warning: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.critical_threshold <= 1.0:
            raise ValueError(
                f"critical_threshold must be between 0.0 and 1.0, got {self.critical_threshold}"
            )
        if self.min_messages_after_compact < 1:
            raise ValueError(
                f"min_messages_after_compact must be at least 1, got {self.min_messages_after_compact}"
            )


class EmergencyCompactor:
    """Emergency compaction for critically full context.

    When standard compaction fails and context is at 95%+ capacity,
    this compactor aggressively discards content while preserving:
    - System prompt (essential for model behavior)
    - User messages (essential for intent preservation)
    - Most recent messages (for continuity)

    Example:
        compactor = EmergencyCompactor()
        compacted = compactor.compact(messages, current_turn=50)
    """

    def __init__(self, config: Optional[EmergencyCompactionConfig] = None):
        """Initialize the emergency compactor.

        Args:
            config: Optional custom configuration
        """
        self.config = config or EmergencyCompactionConfig()
        self._last_compaction_summary = ""

    def compact(
        self,
        messages: List[Message],
        current_turn: int,
    ) -> List[Message]:
        """Perform emergency compaction on messages.

        Args:
            messages: Current message list
            current_turn: Current turn number

        Returns:
            Compacted message list
        """
        result, summary = self.compact_with_summary(messages, current_turn)
        self._last_compaction_summary = summary
        return result

    def compact_with_summary(
        self,
        messages: List[Message],
        current_turn: int,
    ) -> tuple[List[Message], str]:
        """Perform emergency compaction and return summary.

        Args:
            messages: Current message list
            current_turn: Current turn number

        Returns:
            Tuple of (compacted messages, summary string)
        """
        if not messages:
            return [], "No messages to compact"

        original_count = len(messages)
        kept_messages = []

        # Always preserve system prompt first
        if self.config.preserve_system_prompt:
            for msg in messages:
                if msg.role == "system":
                    kept_messages.append(msg)
                    break

        # Preserve user messages if configured
        if self.config.preserve_user_messages:
            for msg in messages:
                if msg.role == "user":
                    kept_messages.append(msg)

        # Preserve very recent tool results (limit by max_tool_result_age)
        # This is done BEFORE general recent messages to prioritize tool results
        if self.config.max_tool_result_age > 0:
            tool_result_count = 0
            for msg in reversed(messages):
                if msg.role == "tool":
                    if tool_result_count < self.config.max_tool_result_age:
                        if msg not in kept_messages:
                            kept_messages.append(msg)
                            tool_result_count += 1

        # Preserve most recent messages (last N) to meet minimum
        # We work backwards from the end, skipping already-kept messages
        recent_count = 0
        min_recent = self.config.min_messages_after_compact
        for msg in reversed(messages):
            if recent_count >= min_recent:
                break
            # Skip messages already kept (system, user, tool)
            if msg in kept_messages:
                continue
            kept_messages.append(msg)
            recent_count += 1

        # Inject warning message if configured
        if self.config.inject_warning:
            warning = Message(
                role="user",
                content=(
                    "[Emergency: Context was critically full and has been reset. "
                    "Earlier conversation details were discarded to continue. "
                    "Use tools to re-read files if you need specific content.]"
                ),
            )
            kept_messages.append(warning)

        # Sort by original order (approximately)
        # For simplicity, we'll keep the order: system -> user -> recent
        result = self._reorder_messages(kept_messages, messages)

        # The injected warning is a new message not present in the original list,
        # so exclude it from the "removed" count to avoid negative values.
        injected = 1 if self.config.inject_warning else 0
        messages_removed = original_count - (len(result) - injected)

        # Log the emergency compaction
        logger.warning(
            f"Emergency compaction: {messages_removed} messages discarded "
            f"({original_count} -> {len(result)} messages) at turn {current_turn}"
        )

        summary = (
            f"Emergency compaction: discarded {messages_removed} messages "
            f"({original_count} -> {len(result)} kept)"
        )

        return result, summary

    def _reorder_messages(
        self,
        kept: List[Message],
        original: List[Message],
    ) -> List[Message]:
        """Reorder kept messages to match original order.

        Args:
            kept: Messages that were kept (may include new messages like warning)
            original: Original message list for order reference

        Returns:
            Reordered message list
        """
        ordered = []

        # First, preserve original order for messages that were in original
        original_set = set(id(m) for m in original)
        for msg in original:
            if msg in kept:
                ordered.append(msg)

        # Then, append new messages that weren't in original (like warning)
        for msg in kept:
            if id(msg) not in original_set:
                ordered.append(msg)

        return ordered

    def get_last_summary(self) -> str:
        """Get summary of last compaction.

        Returns:
            Summary string from last compact operation
        """
        return self._last_compaction_summary


def should_trigger_emergency_compaction(
    utilization: float,
    standard_compaction_failed: bool,
    critical_threshold: float = 0.95,
) -> bool:
    """Check if emergency compaction should be triggered.

    Args:
        utilization: Current context utilization (0.0-1.0)
        standard_compaction_failed: Whether standard compaction failed
        critical_threshold: Utilization threshold for emergency trigger

    Returns:
        True if emergency compaction should be triggered
    """
    return utilization >= critical_threshold and standard_compaction_failed


def emergency_compact(
    messages: List[Message],
    current_turn: int,
    config: Optional[EmergencyCompactionConfig] = None,
) -> List[Message]:
    """Convenience function for emergency compaction.

    Args:
        messages: Current message list
        current_turn: Current turn number
        config: Optional custom configuration

    Returns:
        Compacted message list
    """
    compactor = EmergencyCompactor(config)
    return compactor.compact(messages, current_turn)


# Singleton instance for convenience
_default_compactor: Optional[EmergencyCompactor] = None


def get_emergency_compactor() -> EmergencyCompactor:
    """Get the singleton emergency compactor instance.

    Returns:
        The default EmergencyCompactor instance
    """
    global _default_compactor
    if _default_compactor is None:
        _default_compactor = EmergencyCompactor()
    return _default_compactor
