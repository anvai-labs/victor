# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fast pruning before LLM compaction (P1 feature).

Based on OpenDev research (arXiv:2603.05344):
- Walk backwards through tool results
- Replace old tool results with [pruned] markers
- Preserves recency while reducing LLM compaction cost
- 30-40% reduction in compaction cost

This module provides fast pruning functionality that should be called
before expensive LLM-based compaction to reduce the context size
and cost.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Set

from victor.core.provider_messages import Message

logger = logging.getLogger(__name__)

# Distinctive tokens used to detect whether a later assistant message *references* a tool
# result: file paths, dotted names, snake/kebab identifiers, and long words. Common short prose
# is intentionally excluded so matching errs toward "referenced" (keep) — never the reverse.
_ANCHOR_RE = re.compile(r"[A-Za-z0-9_./-]{4,}")
_MAX_ANCHORS_PER_MESSAGE = 200


@dataclass
class FastPruningConfig:
    """Configuration for fast pruning before LLM compaction.

    Attributes:
        max_tool_result_age: Maximum age (in turns) for tool results before pruning
        max_pruned_chars: Maximum characters to keep in pruned marker preview
        prune_system_messages: Whether to allow pruning system messages
        prune_user_messages: Whether to allow pruning user messages (P0: should be False)
        tool_result_size_threshold: Minimum size for tool result to be pruned
        assistant_message_threshold: Minimum size for assistant message pruning
    """

    max_tool_result_age: int = 5  # Prune tool results older than N turns
    max_pruned_chars: int = 200  # Max chars to keep in pruned marker
    prune_system_messages: bool = False  # Never prune system messages by default
    prune_user_messages: bool = False  # Never prune user messages (P0: preserve intent)
    tool_result_size_threshold: int = 1000  # Min size for tool result pruning
    assistant_message_threshold: int = 2000  # Min size for assistant pruning
    # L1 (reference-aware pruning): when enabled, prune a tool result only if it is beyond the
    # recency window AND no *later* assistant message cites a distinctive anchor from it. This
    # attacks the context leak (full tool history re-sent every turn) without dropping output
    # the model still uses. Default OFF — flip only on C0 trace evidence (lossless + cheaper).
    enable_reference_tracking: bool = False
    reference_window_turns: int = 3  # Most-recent N tool results are always preserved


class FastPruner:
    """Fast pruning of old tool results before LLM compaction.

    Based on OpenDev research (arXiv:2603.05344):
    - Walk backwards through tool results
    - Replace old tool results with [pruned] markers
    - Preserves recency while reducing LLM compaction cost
    - 30-40% reduction in compaction cost

    The pruner operates on the following principles:
    1. Never prune user messages (P0: preserve original intent)
    2. Never prune system messages by default (they contain instructions)
    3. Prune large tool results that are old
    4. Preserve tool_call_id for proper pairing
    """

    def __init__(self, config: FastPruningConfig = None):
        """Initialize the fast pruner.

        Args:
            config: Optional custom configuration. If None, uses defaults.
        """
        self.config = config or FastPruningConfig()
        self._pruned_count = 0

    def prune_old_tool_results(
        self,
        messages: List[Message],
        current_turn: int,
    ) -> List[Message]:
        """Prune old tool results before LLM compaction.

        This method walks through messages and replaces large, old tool
        results with lightweight [pruned] markers. This reduces the
        context size before expensive LLM-based compaction.

        Args:
            messages: Current message list
            current_turn: Current turn number (for age calculation)

        Returns:
            Pruned message list (modified copy, original is not mutated)
        """
        if not messages:
            return messages

        self._pruned_count = 0

        # L1: reference-aware pruning replaces the size-only rule with "prune unreferenced,
        # out-of-window tool results" — preserving anything a later assistant turn still cites.
        if self.config.enable_reference_tracking:
            to_prune = self._reference_prune_indices(messages)
            pruned_messages = []
            for i, msg in enumerate(messages):
                if i in to_prune:
                    pruned_messages.append(self._create_pruned_marker(msg))
                    self._pruned_count += 1
                    logger.debug(
                        f"Reference-pruned unreferenced tool result ({len(msg.content)} chars)"
                    )
                else:
                    pruned_messages.append(msg)
            if self._pruned_count > 0:
                logger.info(
                    f"Reference-aware pruning: {self._pruned_count} unreferenced tool "
                    f"results pruned (turn {current_turn})"
                )
            return pruned_messages

        pruned_messages = []

        for msg in messages:
            if self._should_prune(msg, current_turn):
                # Create pruned marker
                pruned_msg = self._create_pruned_marker(msg)
                pruned_messages.append(pruned_msg)
                self._pruned_count += 1
                logger.debug(f"Fast-pruned {msg.role} message ({len(msg.content)} chars)")
            else:
                pruned_messages.append(msg)

        if self._pruned_count > 0:
            logger.info(
                f"Fast pruning: {self._pruned_count} messages pruned " f"(turn {current_turn})"
            )

        return pruned_messages

    def _extract_anchors(self, content: str) -> Set[str]:
        """Extract distinctive tokens (paths, dotted names, identifiers, long words).

        These are the signals that a later assistant message *used* a tool result — it cites a
        symbol, file, or value the tool surfaced. Common short prose is excluded so a match
        means "very likely referenced", and a non-match is the only thing that permits pruning.

        Args:
            content: Message content to scan.

        Returns:
            Lowercased set of anchor tokens (capped for performance).
        """
        anchors: Set[str] = set()
        for token in _ANCHOR_RE.findall(content or ""):
            if any(sep in token for sep in "_./-") or len(token) >= 6:
                anchors.add(token.lower())
                if len(anchors) >= _MAX_ANCHORS_PER_MESSAGE:
                    break
        return anchors

    def _reference_prune_indices(self, messages: List[Message]) -> Set[int]:
        """Indices of tool results safe to prune: out-of-window AND unreferenced afterwards.

        A tool result is preserved if it is among the most recent
        ``reference_window_turns`` tool results, OR if any assistant message *after* it shares a
        distinctive anchor with it. Only the remainder — old, large, and never cited again — is
        pruned. This is the safe-by-construction core of L1.

        Args:
            messages: The full message list (not mutated).

        Returns:
            Set of message indices to replace with pruned markers.
        """
        assistant_anchors = {
            i: self._extract_anchors(m.content)
            for i, m in enumerate(messages)
            if m.role == "assistant"
        }
        tool_indices = [i for i, m in enumerate(messages) if m.role == "tool"]
        window = max(0, self.config.reference_window_turns)
        protected = set(tool_indices[-window:]) if window else set()

        to_prune: Set[int] = set()
        for i in tool_indices:
            if i in protected:
                continue
            msg = messages[i]
            if len(msg.content) <= self.config.max_pruned_chars:
                continue  # too small to be worth pruning
            tool_anchors = self._extract_anchors(msg.content)
            referenced = bool(tool_anchors) and any(
                j > i and (tool_anchors & later_anchors)
                for j, later_anchors in assistant_anchors.items()
            )
            if not referenced:
                to_prune.add(i)
        return to_prune

    def _should_prune(self, msg: Message, current_turn: int) -> bool:
        """Determine if a message should be pruned.

        Args:
            msg: Message to evaluate
            current_turn: Current turn number

        Returns:
            True if message should be pruned
        """
        # Never prune system messages (unless configured and large)
        if msg.role == "system":
            if not self.config.prune_system_messages:
                return False
            # If pruning system messages, only prune large ones
            return len(msg.content) > self.config.tool_result_size_threshold

        # Never prune user messages (P0: preserve original intent)
        if msg.role == "user":
            if not self.config.prune_user_messages:
                return False
            return len(msg.content) > self.config.tool_result_size_threshold

        # Prune large tool results
        if msg.role == "tool":
            return len(msg.content) > self.config.tool_result_size_threshold

        # Prune large assistant messages (rare but possible)
        if msg.role == "assistant" and len(msg.content) > self.config.assistant_message_threshold:
            return True

        return False

    def _create_pruned_marker(self, msg: Message) -> Message:
        """Create a pruned marker message.

        The marker includes:
        - Original content length
        - Preview of content (from end, more likely to contain meaningful data)
        - Preserved tool_call_id for proper pairing

        Args:
            msg: Original message to create marker for

        Returns:
            New Message with pruned marker content
        """
        original_length = len(msg.content)

        # Take preview from END of content (more likely to contain meaningful data)
        preview_chars = min(self.config.max_pruned_chars, original_length)
        if original_length > preview_chars:
            truncated_preview = msg.content[-preview_chars:]
            preview = f"...{truncated_preview}"
        else:
            preview = msg.content

        return Message(
            role=msg.role,
            content=f"[pruned] Original was {original_length} chars: {preview}",
            tool_call_id=getattr(msg, "tool_call_id", None),
        )

    def get_pruned_count(self) -> int:
        """Get number of messages pruned in last operation.

        Returns:
            Number of messages pruned in the most recent prune_old_tool_results call
        """
        return self._pruned_count

    def estimate_size_reduction(
        self,
        messages: List[Message],
        current_turn: int,
    ) -> tuple[int, int]:
        """Estimate size reduction without actually pruning.

        Useful for deciding whether to perform fast pruning.

        Args:
            messages: Current message list
            current_turn: Current turn number

        Returns:
            Tuple of (original_size, estimated_pruned_size) in characters
        """
        original_size = sum(len(m.content) for m in messages)

        would_prune = 0
        pruned_size = 0

        # Mirror the active pruning rule so the caller's savings gate reflects reality.
        reference_prune = (
            self._reference_prune_indices(messages)
            if self.config.enable_reference_tracking
            else None
        )

        for i, msg in enumerate(messages):
            prune = (
                i in reference_prune
                if reference_prune is not None
                else self._should_prune(msg, current_turn)
            )
            if prune:
                # Estimate pruned marker size
                marker_size = len(f"[pruned] Original was {len(msg.content)} chars: ...")
                pruned_size += marker_size
                would_prune += 1
            else:
                pruned_size += len(msg.content)

        reduction = original_size - pruned_size
        reduction_pct = (reduction / original_size * 100) if original_size > 0 else 0

        logger.debug(
            f"Fast pruning estimate: {would_prune} messages, "
            f"{reduction} chars reduction ({reduction_pct:.1f}%)"
        )

        return original_size, pruned_size


# Singleton instance for convenience
_default_instance: Optional[FastPruner] = None


def get_fast_pruner() -> FastPruner:
    """Get the singleton fast pruner instance.

    Returns:
        The default FastPruner instance
    """
    global _default_instance
    if _default_instance is None:
        _default_instance = FastPruner()
    return _default_instance
