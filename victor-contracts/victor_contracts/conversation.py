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

"""SDK-owned conversation coordinator contracts and default implementation."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class TurnType(Enum):
    """Types of conversation turns."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    turn_id: str
    turn_type: TurnType
    content: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "turn_id": self.turn_id,
            "turn_type": self.turn_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "tool_calls": self.tool_calls,
        }


@dataclass
class ConversationStats:
    """Statistics about a conversation."""

    total_turns: int = 0
    user_turns: int = 0
    assistant_turns: int = 0
    tool_turns: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    start_time: float = 0.0
    last_activity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "total_turns": self.total_turns,
            "user_turns": self.user_turns,
            "assistant_turns": self.assistant_turns,
            "tool_turns": self.tool_turns,
            "tool_calls": self.tool_calls,
            "total_tokens": self.total_tokens,
            "start_time": self.start_time,
            "last_activity": self.last_activity,
            "duration_seconds": (self.last_activity - self.start_time if self.start_time else 0),
        }


@dataclass
class ConversationContext:
    """Context for the current conversation state."""

    turn_count: int = 0
    context_window_size: int = 128000
    max_history_turns: int = 50
    needs_summarization: bool = False
    summarization_threshold: int = 40

    def should_summarize(self) -> bool:
        """Return whether summarization should run."""
        return self.turn_count >= self.summarization_threshold


class ConversationCoordinator:
    """Coordinates conversation history, summarization, and telemetry."""

    def __init__(
        self,
        max_history_turns: int = 50,
        summarization_threshold: int = 40,
        context_window_size: int = 128000,
        enable_deduplication: bool = True,
        enable_statistics: bool = True,
    ):
        self._max_history_turns = max_history_turns
        self._summarization_threshold = summarization_threshold
        self._context_window_size = context_window_size
        self._enable_deduplication = enable_deduplication
        self._enable_statistics = enable_statistics

        self._turns: deque[ConversationTurn] = deque(maxlen=max_history_turns)
        self._turn_counter = 0
        self._message_hashes: Set[str] = set()
        self._context = ConversationContext(
            turn_count=0,
            context_window_size=context_window_size,
            max_history_turns=max_history_turns,
            summarization_threshold=summarization_threshold,
        )
        self._stats = ConversationStats()
        self._stats.start_time = time.time()
        self._stats.last_activity = self._stats.start_time
        self._summaries: List[str] = []
        self._last_summary_turn = 0

        logger.debug(
            "ConversationCoordinator initialized with max_turns=%s, summarization_threshold=%s",
            max_history_turns,
            summarization_threshold,
        )

    def add_message(
        self,
        role: str,
        content: str,
        turn_type: TurnType,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Add a message to the conversation."""
        del role
        self._turn_counter += 1
        turn_id = f"turn_{self._turn_counter}"

        turn = ConversationTurn(
            turn_id=turn_id,
            turn_type=turn_type,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
            tool_calls=tool_calls or [],
        )

        if self._enable_deduplication:
            content_hash = self._hash_content(content)
            if content_hash in self._message_hashes:
                logger.debug("Duplicate message detected: %s", turn_id)
                return turn_id
            self._message_hashes.add(content_hash)

        self._turns.append(turn)
        self._context.turn_count = self._turn_counter

        if self._enable_statistics:
            self._update_stats(turn)

        self._context.needs_summarization = self._context.should_summarize()
        logger.debug("Added turn %s, total turns: %s", turn_id, self._turn_counter)
        return turn_id

    def get_history(
        self,
        max_turns: Optional[int] = None,
        include_system: bool = True,
        include_tool: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get conversation history."""
        turns = list(self._turns)

        if not include_system:
            turns = [turn for turn in turns if turn.turn_type != TurnType.SYSTEM]
        if not include_tool:
            turns = [turn for turn in turns if turn.turn_type != TurnType.TOOL]
        if max_turns:
            turns = turns[-max_turns:]

        return [self._turn_to_message(turn) for turn in turns]

    def get_last_n_turns(self, n: int) -> List[ConversationTurn]:
        """Get the last n turns."""
        return list(self._turns)[-n:]

    def clear_history(self, keep_summaries: bool = True) -> None:
        """Clear conversation history."""
        self._turns.clear()
        self._message_hashes.clear()
        self._context.turn_count = 0
        self._context.needs_summarization = False

        if not keep_summaries:
            self._summaries.clear()
            self._last_summary_turn = 0

        logger.info("Conversation history cleared")

    def remove_turn(self, turn_id: str) -> bool:
        """Remove a specific turn."""
        for index, turn in enumerate(self._turns):
            if turn.turn_id == turn_id:
                removed = self._turns[index]
                del self._turns[index]

                if self._enable_statistics and removed.turn_type == TurnType.USER:
                    self._stats.user_turns = max(0, self._stats.user_turns - 1)
                elif self._enable_statistics and removed.turn_type == TurnType.ASSISTANT:
                    self._stats.assistant_turns = max(0, self._stats.assistant_turns - 1)

                logger.debug("Removed turn %s", turn_id)
                return True

        return False

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // 4

    def get_context_usage(self) -> Tuple[int, int]:
        """Get current context window usage."""
        used = sum(self.estimate_tokens(turn.content) for turn in self._turns)
        return used, self._context_window_size

    def get_context_utilization(self) -> float:
        """Get context window utilization."""
        used, total = self.get_context_usage()
        return used / total if total > 0 else 0.0

    def truncate_history_if_needed(self, max_tokens: Optional[int] = None) -> int:
        """Truncate history if context window is exceeded."""
        max_tok = max_tokens or self._context_window_size
        removed = 0

        while self.get_context_usage()[0] > max_tok and len(self._turns) > 2:
            for index, turn in enumerate(self._turns):
                if turn.turn_type != TurnType.SYSTEM:
                    del self._turns[index]
                    removed += 1
                    break

        if removed > 0:
            logger.info("Truncated %s turns to fit context window", removed)

        return removed

    def needs_summarization(self) -> bool:
        """Check whether summarization is recommended."""
        return self._context.needs_summarization

    def add_summary(self, summary: str) -> None:
        """Add a conversation summary."""
        self._summaries.append(summary)
        self._last_summary_turn = self._turn_counter

        turns_to_keep = self._max_history_turns // 2
        removed = 0
        if len(self._turns) > turns_to_keep:
            removed = len(self._turns) - turns_to_keep
            for _ in range(removed):
                self._turns.popleft()

        self._context.needs_summarization = False
        if removed > 0:
            logger.info("Added summary, cleared %s old turns", removed)

    def get_summaries(self) -> List[str]:
        """Get all conversation summaries."""
        return self._summaries.copy()

    def get_full_context(self) -> str:
        """Get full conversation context including summaries."""
        parts: List[str] = []

        if self._summaries:
            parts.append("# Previous Conversation Summaries")
            parts.extend(self._summaries)
            parts.append("")

        parts.append("# Recent Conversation")
        for turn in self._turns:
            parts.append(f"{turn.turn_type.value.upper()}: {turn.content}")

        return "\n".join(parts)

    def is_duplicate(self, content: str) -> bool:
        """Check if content is a duplicate."""
        return self._hash_content(content) in self._message_hashes

    def _hash_content(self, content: str) -> str:
        """Hash content for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()

    def get_stats(self) -> ConversationStats:
        """Get conversation statistics."""
        self._stats.total_turns = len(self._turns)
        self._stats.last_activity = time.time()
        return self._stats

    def get_stats_dict(self) -> Dict[str, Any]:
        """Get statistics as a dictionary."""
        return self.get_stats().to_dict()

    def _update_stats(self, turn: ConversationTurn) -> None:
        """Update statistics for a turn."""
        self._stats.total_turns = len(self._turns)
        self._stats.last_activity = turn.timestamp

        if turn.turn_type == TurnType.USER:
            self._stats.user_turns += 1
        elif turn.turn_type == TurnType.ASSISTANT:
            self._stats.assistant_turns += 1
        elif turn.turn_type == TurnType.TOOL:
            self._stats.tool_turns += 1
            self._stats.tool_calls += len(turn.tool_calls)

        self._stats.total_tokens += self.estimate_tokens(turn.content)

    def _turn_to_message(self, turn: ConversationTurn) -> Dict[str, Any]:
        """Convert a turn to a message dictionary."""
        return {
            "role": turn.turn_type.value,
            "content": turn.content,
            "timestamp": turn.timestamp,
            "turn_id": turn.turn_id,
            "metadata": turn.metadata,
        }

    def get_turn_count(self) -> int:
        """Get current turn count."""
        return self._turn_counter

    def is_empty(self) -> bool:
        """Check whether the conversation is empty."""
        return len(self._turns) == 0

    def reset(self) -> None:
        """Reset conversation state."""
        self._turns.clear()
        self._message_hashes.clear()
        self._turn_counter = 0
        self._context.turn_count = 0
        self._context.needs_summarization = False
        self._summaries.clear()
        self._last_summary_turn = 0

        self._stats = ConversationStats()
        self._stats.start_time = time.time()
        self._stats.last_activity = self._stats.start_time

        logger.info("ConversationCoordinator reset")

    def get_observability_data(self) -> Dict[str, Any]:
        """Get observability data for dashboard integration."""
        stats = self.get_stats()
        used_tokens, total_tokens = self.get_context_usage()

        return {
            "source_id": f"coordinator:conversation:{id(self)}",
            "source_type": "coordinator",
            "coordinator_type": "conversation",
            "stats": stats.to_dict(),
            "context": {
                "turn_count": self._turn_counter,
                "context_usage": {
                    "used_tokens": used_tokens,
                    "total_tokens": total_tokens,
                    "utilization": self.get_context_utilization(),
                },
                "needs_summarization": self._context.needs_summarization,
                "summary_count": len(self._summaries),
            },
        }


__all__ = [
    "TurnType",
    "ConversationTurn",
    "ConversationStats",
    "ConversationContext",
    "ConversationCoordinator",
]
