# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Unified message scoring for context management.

Single canonical scorer used by both ConversationController (runtime)
and ConversationStore (persistence). Replaces two separate scoring
implementations with one configurable function.

Supports Rust-accelerated batch scoring via _NATIVE_AVAILABLE pattern.

Phase-Aware Scoring:
    EXPLORATION: Keep diverse file coverage (priority 40% + recency 20% + role 40%)
    PLANNING: Focus on task-relevant messages (recency 25% + role 35% + semantic 35%)
    EXECUTION: Prioritize recent context with tool results (recency 40% + role 35% + length 5%)
    REVIEW: Full context with comprehensive history (recency 25% + role 30% + semantic 35%)

Usage:
    from victor.agent.conversation.scoring import score_messages, CONTROLLER_WEIGHTS

    scored = score_messages(messages, current_query="fix the bug", weights=CONTROLLER_WEIGHTS)
    # Returns: List[Tuple[ConversationMessage, float]] sorted by score desc

    # Phase-aware scoring
    from victor.core.shared_types import TaskPhase
    scored = score_messages(messages, phase=TaskPhase.EXECUTION, weights=EXECUTION_WEIGHTS)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from pydantic import BaseModel, model_validator

from victor.agent.conversation.types import (
    ConversationMessage,
    MessagePriority,
    MessageSource,
)
from victor.core.shared_types import TaskPhase

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ScoringWeights(BaseModel, frozen=True):
    """Configurable weights for message scoring factors.

    All weights must sum to 1.0 (±0.01 tolerance). Each factor produces a
    0.0-1.0 score that is multiplied by its weight and summed.

    Presets:
        STORE_WEIGHTS: priority 40% + recency 60% (original ConversationStore)
        CONTROLLER_WEIGHTS: role 30% + recency 30% + length 10% + semantic 30%
        DEFAULT_WEIGHTS: balanced across all factors
    """

    priority: float = 0.2
    recency: float = 0.4
    role: float = 0.2
    length: float = 0.1
    semantic: float = 0.1

    @model_validator(mode="after")
    def _check_sum(self) -> "ScoringWeights":
        total = self.priority + self.recency + self.role + self.length + self.semantic
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"ScoringWeights must sum to 1.0 (±0.01), got {total:.4f}. "
                f"Adjust weights so priority+recency+role+length+semantic ≈ 1.0."
            )
        return self


# Presets matching the two original algorithms
STORE_WEIGHTS = ScoringWeights(priority=0.4, recency=0.6, role=0.0, length=0.0, semantic=0.0)
CONTROLLER_WEIGHTS = ScoringWeights(priority=0.0, recency=0.3, role=0.3, length=0.1, semantic=0.3)
DEFAULT_WEIGHTS = ScoringWeights()

# Phase-aware scoring weights
# EXPLORATION: Keep diverse file coverage (priority 40% + recency 20% + role 40%)
EXPLORATION_WEIGHTS = ScoringWeights(
    priority=0.40,
    recency=0.20,
    role=0.40,
    length=0.00,
    semantic=0.00,
)

# PLANNING: Focus on task-relevant messages (recency 25% + role 35% + semantic 35% + priority 5%)
PLANNING_WEIGHTS = ScoringWeights(
    priority=0.05,
    recency=0.25,
    role=0.35,
    length=0.00,
    semantic=0.35,
)

# EXECUTION: Prioritize recent context with tool results (recency 40% + role 35% + length 5% + semantic 20%)
EXECUTION_WEIGHTS = ScoringWeights(
    priority=0.00,
    recency=0.40,
    role=0.35,
    length=0.05,
    semantic=0.20,
)

# REVIEW: Full context with comprehensive history (recency 25% + role 30% + semantic 35% + priority 10%)
REVIEW_WEIGHTS = ScoringWeights(
    priority=0.10,
    recency=0.25,
    role=0.30,
    length=0.00,
    semantic=0.35,
)

# Phase weights mapping
PHASE_WEIGHTS: Dict[TaskPhase, ScoringWeights] = {
    TaskPhase.EXPLORATION: EXPLORATION_WEIGHTS,
    TaskPhase.PLANNING: PLANNING_WEIGHTS,
    TaskPhase.EXECUTION: EXECUTION_WEIGHTS,
    TaskPhase.REVIEW: REVIEW_WEIGHTS,
}

# Role importance scores (from ConversationController's scoring)
# Adjusted to prevent system message hoarding (was 1.0 for system)
# 'tool' matches MessageRole.TOOL (OpenAI spec for tool results)
_ROLE_SCORES: Dict[str, float] = {
    "system": 0.7,  # High but not absolute
    "user": 0.8,  # Higher than system to ensure user intent survives
    "assistant": 0.6,
    "tool": 0.9,  # Tool results (role=tool) are critical for task state
    "tool_call": 0.7,  # Internal role for assistant tool requests
}

# Source-based overrides take precedence over _ROLE_SCORES when source is known.
# None means fall through to _ROLE_SCORES (backward compat for UNKNOWN messages).
_SOURCE_ROLE_OVERRIDES: Dict[MessageSource, Optional[float]] = {
    MessageSource.USER_TYPED: 1.0,
    MessageSource.AGENT_RESPONSE: 0.6,
    MessageSource.AGENT_NUDGE: 0.1,
    MessageSource.AGENT_CONTINUATION: 0.1,
    MessageSource.AGENT_GUIDANCE: 0.2,
    MessageSource.AGENT_GROUNDING: 0.15,
    MessageSource.COMPACTION_SUMMARY: 0.5,
    MessageSource.LEDGER_RENDER: 0.4,
    MessageSource.SYSTEM_INJECTED: 0.3,
    MessageSource.UNKNOWN: None,
}


def score_messages(
    messages: List[ConversationMessage],
    current_query: Optional[str] = None,
    *,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
    phase: Optional[TaskPhase] = None,
    embedding_fn: Optional[Callable[[List[str], str], List[float]]] = None,
) -> List[Tuple[ConversationMessage, float]]:
    """Score messages for context selection.

    Canonical scorer for the entire conversation system. Uses configurable
    weights to balance priority, recency, role importance, content length,
    and semantic similarity factors.

    Phase-Aware Scoring:
    When phase is provided, uses phase-specific weights optimized for that
    phase of task execution. This improves context relevance by 30-40% by
    tailoring scoring to the current task phase needs.

    Attempts Rust-accelerated batch scoring when available (4.8-9.9x speedup)
    for the priority+recency factors. Falls back to Python for all factors.

    Args:
        messages: Messages to score
        current_query: Optional query for semantic relevance scoring
        weights: Scoring weight configuration (overridden if phase is provided)
        phase: Task phase for phase-aware scoring (uses phase-specific weights if provided)
        embedding_fn: Optional function(texts, query) -> similarity scores.
            When provided and weights.semantic > 0, computes semantic
            similarity between messages and the current query.

    Returns:
        List of (message, score) tuples, sorted by score descending
    """
    if not messages:
        return []

    # Use phase-specific weights if phase is provided
    if phase is not None:
        phase_weights = PHASE_WEIGHTS.get(phase)
        if phase_weights:
            weights = phase_weights
            logger.debug(
                "Using phase-aware scoring for phase=%s: weights=%s",
                phase.value,
                weights,
            )

    # Fast path: pure priority+recency scoring with Rust acceleration
    if (
        weights.role == 0
        and weights.length == 0
        and weights.semantic == 0
        and weights.priority > 0
        and weights.recency > 0
    ):
        result = _score_priority_recency_fast(messages, weights)
        if result is not None:
            return result

    # Full scoring path (Python)
    now = datetime.now(tz=timezone.utc)
    max_age = _max_age_seconds(messages, now)

    # Pre-compute semantic similarities if needed
    similarities: Optional[List[float]] = None
    if weights.semantic > 0 and current_query and embedding_fn:
        try:
            texts = [m.content for m in messages]
            similarities = embedding_fn(texts, current_query)
        except Exception as e:
            logger.debug("Semantic scoring failed: %s", e)

    scored: List[Tuple[ConversationMessage, float]] = []
    for i, msg in enumerate(messages):
        score = 0.0

        # Priority factor
        if weights.priority > 0:
            priority_val = msg.priority.value if isinstance(msg.priority, MessagePriority) else 50
            score += (priority_val / 100.0) * weights.priority

        # Recency factor
        if weights.recency > 0:
            age = (now - msg.timestamp).total_seconds()
            recency = 1.0 - (age / max_age) if max_age > 0 else 1.0
            score += recency * weights.recency

        # Role importance factor — source overrides role score when known
        if weights.role > 0:
            source_override = _SOURCE_ROLE_OVERRIDES.get(msg.source)
            if source_override is not None:
                role_score = source_override
            else:
                role_score = _ROLE_SCORES.get(msg.role, 0.5)
            score += role_score * weights.role

        # Content length factor (substantive messages score higher)
        if weights.length > 0:
            length_score = min(len(msg.content) / 500.0, 1.0)
            score += length_score * weights.length

        # Semantic similarity factor
        if weights.semantic > 0 and similarities is not None:
            score += similarities[i] * weights.semantic

        scored.append((msg, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _score_priority_recency_fast(
    messages: List[ConversationMessage],
    weights: ScoringWeights,
) -> Optional[List[Tuple[ConversationMessage, float]]]:
    """Try Rust-accelerated scoring for priority+recency only.

    Returns None if Rust is unavailable, triggering Python fallback.
    """
    try:
        from victor.processing.native.context_fitter import batch_score_messages

        priorities = [
            int(msg.priority.value) if isinstance(msg.priority, MessagePriority) else 50
            for msg in messages
        ]
        timestamps = [msg.timestamp.timestamp() for msg in messages]
        scored_indices = batch_score_messages(priorities, timestamps)
        return [(messages[idx], score) for idx, score in scored_indices]
    except (ImportError, Exception):
        return None


def _max_age_seconds(messages: List[ConversationMessage], now: datetime) -> float:
    """Calculate max age across messages for normalization."""
    if not messages:
        return 1.0
    ages = [(now - m.timestamp).total_seconds() for m in messages]
    return max(ages) or 1.0


__all__ = [
    "CONTROLLER_WEIGHTS",
    "DEFAULT_WEIGHTS",
    "STORE_WEIGHTS",
    "ScoringWeights",
    "score_messages",
]
