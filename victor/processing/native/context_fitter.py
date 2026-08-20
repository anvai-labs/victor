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

"""Context fitting functions with native acceleration.

Provides context window management for fitting messages into token budgets.
Uses Rust implementation when available for high-performance fitting,
falling back to pure Python implementation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from victor.processing.native._base import _NATIVE_AVAILABLE, _native


@dataclass
class FitResult:
    """Result of context fitting operation.

    Attributes:
        kept_indices: Indices of messages that fit within the budget
        total_tokens: Total token count of kept messages
        dropped_count: Number of messages dropped
        freed_tokens: Number of tokens freed by dropping messages
    """

    kept_indices: List[int]
    total_tokens: int
    dropped_count: int
    freed_tokens: int


def fit_context(
    messages: List[Dict[str, Any]],
    budget: int,
    strategy: str = "recency",
    preserve_system: bool = True,
) -> FitResult:
    """Fit messages into a token budget.

    Selects which messages to keep based on the given strategy,
    respecting the token budget. Uses Rust implementation when
    available for high-performance fitting.

    Args:
        messages: List of message dicts with 'role', 'content', and
                  optionally 'token_count' and 'priority' fields
        budget: Maximum token budget
        strategy: Fitting strategy - "recency" (keep newest),
                  "priority" (keep highest priority), or "balanced"
        preserve_system: Whether to always preserve system messages

    Returns:
        FitResult with indices of kept messages and statistics
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "fit_context"):
        try:
            # Build MessageSlot objects for Rust
            slots = []
            for i, msg in enumerate(messages):
                token_count = msg.get("token_count", len(msg.get("content", "").split()) * 13 // 10)
                priority = msg.get("priority", 1.0)
                role = msg.get("role", "user")
                recency = float(i) / max(len(messages), 1)
                slot = _native.MessageSlot(
                    index=i,
                    token_count=token_count,
                    priority=priority,
                    role=role,
                    recency=recency,
                )
                slots.append(slot)

            result = _native.fit_context(slots, budget, strategy, preserve_system)
            return FitResult(
                kept_indices=list(result.kept_indices),
                total_tokens=result.total_tokens,
                dropped_count=result.dropped_count,
                freed_tokens=result.freed_tokens,
            )
        except Exception:
            pass  # Fall through to Python implementation

    # Pure Python fallback
    return _fit_context_python(messages, budget, strategy, preserve_system)


def _fit_context_python(
    messages: List[Dict[str, Any]],
    budget: int,
    strategy: str,
    preserve_system: bool,
) -> FitResult:
    """Pure Python context fitting implementation.

    Args:
        messages: List of message dicts
        budget: Maximum token budget
        strategy: Fitting strategy
        preserve_system: Whether to preserve system messages

    Returns:
        FitResult with fitting results
    """
    if not messages:
        return FitResult(kept_indices=[], total_tokens=0, dropped_count=0, freed_tokens=0)

    # Calculate token counts for each message
    token_counts = []
    for msg in messages:
        count = msg.get("token_count", len(msg.get("content", "").split()) * 13 // 10)
        token_counts.append(count)

    total_all_tokens = sum(token_counts)

    # Identify system messages
    system_indices = set()
    if preserve_system:
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_indices.add(i)

    # Build candidate list (non-system messages that can be dropped)
    candidates = []
    for i in range(len(messages)):
        if i not in system_indices:
            priority = messages[i].get("priority", 1.0)
            recency = float(i) / max(len(messages), 1)
            candidates.append((i, token_counts[i], priority, recency))

    # Sort candidates by drop priority (what to drop first)
    if strategy == "recency":
        # Drop oldest first (lowest index = oldest)
        candidates.sort(key=lambda x: x[3])  # ascending recency
    elif strategy == "priority":
        # Drop lowest priority first
        candidates.sort(key=lambda x: x[2])  # ascending priority
    else:
        # Balanced: combine priority and recency
        candidates.sort(key=lambda x: x[2] * 0.5 + x[3] * 0.5)

    # Start with all messages, drop from front of sorted candidates
    kept = set(range(len(messages)))
    current_tokens = total_all_tokens

    for idx, tc, _pri, _rec in candidates:
        if current_tokens <= budget:
            break
        kept.discard(idx)
        current_tokens -= tc

    kept_indices = sorted(kept)
    kept_tokens = sum(token_counts[i] for i in kept_indices)
    dropped = len(messages) - len(kept_indices)
    freed = total_all_tokens - kept_tokens

    return FitResult(
        kept_indices=kept_indices,
        total_tokens=kept_tokens,
        dropped_count=dropped,
        freed_tokens=freed,
    )


def truncate_message(
    content: str,
    max_tokens: int,
    preserve_lines: bool = True,
) -> str:
    """Truncate a message to fit within a token limit.

    Uses Rust implementation when available for accurate BPE-aware
    truncation. Falls back to line-based or word-based truncation.

    Args:
        content: Message content to truncate
        max_tokens: Maximum number of tokens allowed
        preserve_lines: Whether to truncate at line boundaries

    Returns:
        Truncated content string
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "truncate_message"):
        try:
            return _native.truncate_message(content, max_tokens, preserve_lines)
        except Exception:
            pass  # Fall through to Python implementation

    # Pure Python fallback
    return _truncate_message_python(content, max_tokens, preserve_lines)


def _truncate_message_python(
    content: str,
    max_tokens: int,
    preserve_lines: bool,
) -> str:
    """Pure Python message truncation implementation.

    Args:
        content: Message content to truncate
        max_tokens: Maximum number of tokens allowed
        preserve_lines: Whether to truncate at line boundaries

    Returns:
        Truncated content string
    """
    if not content:
        return content

    if preserve_lines:
        lines = content.split("\n")
        kept_lines = []
        current_tokens = 0

        for line in lines:
            line_tokens = len(line.split()) * 13 // 10
            if line_tokens == 0:
                line_tokens = 1  # Empty lines still cost a token
            if current_tokens + line_tokens > max_tokens:
                break
            kept_lines.append(line)
            current_tokens += line_tokens

        return "\n".join(kept_lines)
    else:
        words = content.split()
        # Approximate: ~1.3 tokens per word
        max_words = max(1, max_tokens * 10 // 13)
        return " ".join(words[:max_words])


def batch_score_messages(
    priorities: List[int],
    timestamps: List[float],
) -> List[tuple]:
    """Score messages by priority (40%) and recency (60%), return sorted indices.

    Formula: score = 0.4 * (priority / 100) + 0.6 * (1 - age / max_age)

    Uses Rust implementation when available (3-10x faster for large lists).

    Args:
        priorities: List of priority values (0-100).
        timestamps: List of timestamps as epoch seconds.

    Returns:
        List of (index, score) tuples sorted by score descending.
    """
    if _NATIVE_AVAILABLE and hasattr(_native, "batch_score_messages"):
        try:
            return _native.batch_score_messages(priorities, timestamps)
        except Exception:
            pass  # Fall through to Python implementation

    return _batch_score_messages_python(priorities, timestamps)


def _batch_score_messages_python(
    priorities: List[int],
    timestamps: List[float],
) -> List[tuple]:
    """Pure Python batch scoring — reference implementation."""
    n = min(len(priorities), len(timestamps))
    if n == 0:
        return []

    max_ts = max(timestamps[:n])
    max_age = max(max_ts - t for t in timestamps[:n]) or 1e-9

    scored = []
    for i in range(n):
        priority_score = priorities[i] / 100.0
        age = max_ts - timestamps[i]
        recency_score = 1.0 - (age / max_age)
        score = priority_score * 0.4 + recency_score * 0.6
        scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
