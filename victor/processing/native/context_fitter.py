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

# Canonical strategy vocabulary — must match the Rust implementation's
# accepted set (rust/crates/python-bindings/src/context_fitter.rs). The
# Python fallback previously accepted a DISJOINT vocabulary
# (recency/priority/balanced), so "smart" silently meant "balanced" and
# "recency" silently meant Rust's fit_smart (co-design review U8-F1).
CANONICAL_STRATEGIES = frozenset({"smart", "priority", "fifo"})
_LEGACY_STRATEGY_ALIASES = {"recency": "fifo", "balanced": "smart"}
_DEFAULT_PRIORITY = 50


def _normalize_strategy(strategy: str) -> str:
    """Map legacy strategy names to the canonical vocabulary; raise on unknown.

    Raises loudly BEFORE any native call so a typo is visible regardless of
    whether the Rust wheel is installed.
    """
    canonical = _LEGACY_STRATEGY_ALIASES.get(strategy, strategy)
    if canonical not in CANONICAL_STRATEGIES:
        raise ValueError(
            f"unknown fit strategy {strategy!r} "
            f"(expected one of {sorted(CANONICAL_STRATEGIES)})"
        )
    return canonical


def _coerce_priority(value: Any) -> int:
    """Coerce a message priority to the u8 scale Rust expects (0-255).

    Floats previously raised TypeError inside PyO3 for every message lacking
    an explicit int priority, silently disabling the native path (U8-F4).
    """
    try:
        priority = int(value)
    except (TypeError, ValueError):
        priority = _DEFAULT_PRIORITY
    return max(0, min(255, priority))


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
    strategy: str = "smart",
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
        strategy: One of "smart", "priority", or "fifo" (default "smart").
                  Legacy names "recency" (-> fifo) and "balanced" (-> smart)
                  are accepted for backwards compatibility.
        preserve_system: Whether to always preserve system messages

    Returns:
        FitResult with indices of kept messages and statistics

    Raises:
        ValueError: On an unknown strategy name — validated before the
            native call so the error does not depend on whether the Rust
            wheel is installed.
    """
    strategy = _normalize_strategy(strategy)

    if _NATIVE_AVAILABLE and hasattr(_native, "fit_context"):
        try:
            # Build MessageSlot objects for Rust
            slots = []
            for i, msg in enumerate(messages):
                token_count = msg.get("token_count", len(msg.get("content", "").split()) * 13 // 10)
                priority = _coerce_priority(msg.get("priority", _DEFAULT_PRIORITY))
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

    A faithful port of the Rust algorithms (context_fitter.rs: fit_fifo /
    fit_priority / fit_smart) so both backends select identical messages —
    previously the fallback used drop-lowest heuristics with no smart
    pinning, diverging from the native path (co-design review U8-F1).

    Args:
        messages: List of message dicts
        budget: Maximum token budget
        strategy: One of "smart", "priority", "fifo"
        preserve_system: Whether to preserve system messages

    Returns:
        FitResult with fitting results
    """
    if not messages:
        return FitResult(kept_indices=[], total_tokens=0, dropped_count=0, freed_tokens=0)

    token_counts = [
        msg.get("token_count", len(msg.get("content", "").split()) * 13 // 10) for msg in messages
    ]
    total_all_tokens = sum(token_counts)
    n = len(messages)

    def _recency(i: int) -> float:
        return float(i) / max(n, 1)

    def _score(i: int) -> float:
        priority = _coerce_priority(messages[i].get("priority", _DEFAULT_PRIORITY))
        return (priority / 100.0) * 0.4 + _recency(i) * 0.6

    def _finalize(kept: set) -> FitResult:
        kept_indices = sorted(kept)
        kept_tokens = sum(token_counts[i] for i in kept_indices)
        return FitResult(
            kept_indices=kept_indices,
            total_tokens=kept_tokens,
            dropped_count=n - len(kept_indices),
            freed_tokens=total_all_tokens - kept_tokens,
        )

    if total_all_tokens <= budget:
        return _finalize(set(range(n)))

    if strategy == "fifo":
        # Mirror fit_fifo: pin system when preserving; walk the rest newest
        # first and keep until the remaining budget is exhausted.
        pinned = {i for i in range(n) if preserve_system and messages[i].get("role") == "system"}
        remaining = budget - sum(token_counts[i] for i in pinned)
        kept = set(pinned)
        used = 0
        for i in reversed(range(n)):
            if i in pinned:
                continue
            if used + token_counts[i] <= remaining:
                kept.add(i)
                used += token_counts[i]
        return _finalize(kept)

    if strategy == "priority":
        # Mirror fit_priority: pin system when preserving; score the rest
        # (0.4 * priority/100 + 0.6 * recency) and greedily keep the
        # highest-scoring messages that fit.
        pinned = {i for i in range(n) if preserve_system and messages[i].get("role") == "system"}
        remaining = budget - sum(token_counts[i] for i in pinned)
        rest = sorted((i for i in range(n) if i not in pinned), key=_score, reverse=True)
        kept = set(pinned)
        used = 0
        for i in rest:
            if used + token_counts[i] <= remaining:
                kept.add(i)
                used += token_counts[i]
        return _finalize(kept)

    # smart: mirror fit_smart — pin system messages, the first user message,
    # and the last 2 messages; score the rest and greedily keep what fits.
    pinned = {i for i in range(n) if messages[i].get("role") == "system"}
    for i in range(n):
        if messages[i].get("role") == "user":
            pinned.add(i)
            break
    if n >= 1:
        pinned.add(n - 1)
    if n >= 2:
        pinned.add(n - 2)

    pinned_cost = sum(token_counts[i] for i in pinned)
    if pinned_cost >= budget:
        return _finalize(pinned)

    remaining = budget - pinned_cost
    rest = sorted((i for i in range(n) if i not in pinned), key=_score, reverse=True)
    kept = set(pinned)
    used = 0
    for i in rest:
        if used + token_counts[i] <= remaining:
            kept.add(i)
            used += token_counts[i]
    return _finalize(kept)


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
