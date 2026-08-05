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

"""RL-based cache eviction policy.

This module provides an intelligent cache eviction policy that uses
reinforcement learning to decide which entries to evict based on
predicted future value rather than simple LRU or TTL rules.

The policy learns from:
- Entry hit/miss patterns
- Tool context relevance
- Entry age and size
- Current task context

Sprint 3: Cache & Grounding Learners
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BoundedQTable:
    """Q-table with LRU eviction to prevent unbounded growth."""

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._table: Dict[str, float] = {}
        self._access_order: List[str] = []

    def get(self, key: str, default: float = 0.0) -> float:
        if key in self._table:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._table[key]
        return default

    def set(self, key: str, value: float) -> None:
        if key in self._table:
            self._access_order.remove(key)
        elif len(self._table) >= self.max_size:
            lru_key = self._access_order.pop(0)
            del self._table[lru_key]
        self._table[key] = value
        self._access_order.append(key)

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, key: str) -> bool:
        return key in self._table

    def clear(self) -> None:
        self._table.clear()
        self._access_order.clear()

    def keys(self) -> List[str]:
        return list(self._table.keys())


class EvictionAction(str, Enum):
    """Cache eviction actions."""

    KEEP = "keep"  # Keep entry in cache
    EVICT = "evict"  # Remove entry from cache
    PROMOTE = "promote"  # Move to higher tier (L1)
    DEMOTE = "demote"  # Move to lower tier (L2)


@dataclass
class CacheEntryState:
    """State representation for a cache entry.

    Attributes:
        key: Cache key
        tool_type: Type of tool that created entry
        entry_age_seconds: Age of entry in seconds
        hit_count: Number of times entry was accessed
        last_access_seconds: Time since last access
        size_bytes: Entry size in bytes
        context_relevance: Relevance to current task (0-1)
    """

    key: str
    tool_type: str = "unknown"
    entry_age_seconds: float = 0.0
    hit_count: int = 0
    last_access_seconds: float = 0.0
    size_bytes: int = 0
    context_relevance: float = 0.5

    def to_feature_tuple(self) -> tuple:
        """Convert to feature tuple for Q-table lookup.

        Returns bucketed/discretized features for manageable state space.
        """
        # Bucket age into categories
        if self.entry_age_seconds < 60:
            age_bucket = "fresh"  # < 1 min
        elif self.entry_age_seconds < 300:
            age_bucket = "recent"  # < 5 min
        elif self.entry_age_seconds < 900:
            age_bucket = "moderate"  # < 15 min
        else:
            age_bucket = "old"  # >= 15 min

        # Bucket hit count
        if self.hit_count == 0:
            hit_bucket = "unused"
        elif self.hit_count < 3:
            hit_bucket = "low"
        elif self.hit_count < 10:
            hit_bucket = "medium"
        else:
            hit_bucket = "high"

        # Bucket size
        if self.size_bytes < 1024:
            size_bucket = "small"  # < 1KB
        elif self.size_bytes < 10240:
            size_bucket = "medium"  # < 10KB
        else:
            size_bucket = "large"  # >= 10KB

        # Bucket relevance
        if self.context_relevance < 0.3:
            relevance_bucket = "low"
        elif self.context_relevance < 0.7:
            relevance_bucket = "medium"
        else:
            relevance_bucket = "high"

        return (self.tool_type, age_bucket, hit_bucket, size_bucket, relevance_bucket)


@dataclass
class EvictionDecision:
    """Decision from eviction policy.

    Attributes:
        action: Recommended action
        confidence: Confidence in decision (0-1)
        reason: Explanation for decision
        q_value: Q-value for this decision
    """

    action: EvictionAction
    confidence: float = 0.5
    reason: str = ""
    q_value: float = 0.0


class RLEvictionPolicy:
    """RL-based cache eviction policy.

    Uses Q-learning to decide which cache entries to evict based on
    learned value predictions. The policy learns from cache hit/miss
    feedback to optimize for hit rate.

    Q-learning update:
        Q(s, a) += α * (r + γ * max_a' Q(s', a') - Q(s, a))

    Where:
        - s: Entry state (tool_type, age, hits, size, relevance)
        - a: Action (keep, evict, promote, demote)
        - r: Reward (+1 for hit after keep, -0.5 for miss after keep,
                     +0.2 for evict that freed space for hit)

    Usage:
        policy = RLEvictionPolicy()

        # Get eviction decision
        state = CacheEntryState(key="key1", tool_type="code_search", ...)
        decision = policy.get_decision(state)

        # Execute decision
        if decision.action == EvictionAction.EVICT:
            cache.remove(state.key)

        # Record outcome (called on cache hit/miss)
        policy.record_hit(state)  # or record_miss(state)
    """

    # Q-learning parameters
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_DISCOUNT_FACTOR = 0.95
    DEFAULT_EPSILON = 0.1

    # Reward values
    REWARD_HIT_AFTER_KEEP = 1.0
    REWARD_MISS_AFTER_KEEP = -0.5
    REWARD_EVICT_FREED_SPACE = 0.2
    REWARD_EVICT_NEEDED_LATER = -0.3

    def __init__(
        self,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
        epsilon: float = DEFAULT_EPSILON,
        cache_eviction_learner: Optional[Any] = None,
    ):
        """Initialize eviction policy.

        Args:
            learning_rate: Q-learning alpha
            discount_factor: Q-learning gamma
            epsilon: Exploration rate
            cache_eviction_learner: Optional CacheEvictionLearner from RL framework
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Use external learner if provided
        self._external_learner = cache_eviction_learner

        # Local Q-table if no external learner
        self._q_table: Dict[tuple, Dict[EvictionAction, float]] = {}

        # Track recent decisions for feedback
        self._recent_decisions: Dict[str, Tuple[CacheEntryState, EvictionAction]] = {}
        self._max_tracked_decisions = 1000

        # Statistics
        self._total_decisions = 0
        self._hits_after_keep = 0
        self._misses_after_keep = 0
        self._evictions = 0

    def get_decision(
        self,
        state: CacheEntryState,
        cache_utilization: float = 0.5,
    ) -> EvictionDecision:
        """Get eviction decision for a cache entry.

        Args:
            state: Cache entry state
            cache_utilization: Current cache utilization (0-1)

        Returns:
            EvictionDecision with recommended action
        """
        self._total_decisions += 1

        # Get Q-values for this state
        q_values = self._get_q_values(state)

        # Decide action using epsilon-greedy
        import random

        if random.random() < self.epsilon:
            # Explore: random action
            action = random.choice(list(EvictionAction))
            reason = "Exploration"
        else:
            # Exploit: best Q-value
            action = max(q_values, key=q_values.get)
            reason = f"Q-value: {q_values[action]:.2f}"

        # Apply cache utilization heuristic
        if cache_utilization > 0.9 and action == EvictionAction.KEEP:
            # High utilization - bias toward eviction
            if q_values[EvictionAction.EVICT] > q_values[EvictionAction.KEEP] * 0.8:
                action = EvictionAction.EVICT
                reason = f"High utilization ({cache_utilization:.0%})"

        # Track decision for feedback
        self._track_decision(state, action)

        # Calculate confidence
        q_range = max(q_values.values()) - min(q_values.values())
        confidence = min(0.95, 0.5 + q_range)

        return EvictionDecision(
            action=action,
            confidence=confidence,
            reason=reason,
            q_value=q_values[action],
        )

    def record_hit(self, key: str) -> None:
        """Record cache hit for feedback.

        Args:
            key: Cache key that was hit
        """
        if key in self._recent_decisions:
            state, action = self._recent_decisions[key]

            if action == EvictionAction.KEEP:
                # Good decision - kept entry that was needed
                self._update_q_value(state, action, self.REWARD_HIT_AFTER_KEEP)
                self._hits_after_keep += 1

    def record_miss(self, key: str, was_evicted: bool = False) -> None:
        """Record cache miss for feedback.

        Args:
            key: Cache key that was missed
            was_evicted: Whether entry was previously evicted by policy
        """
        if key in self._recent_decisions:
            state, action = self._recent_decisions[key]

            if action == EvictionAction.KEEP:
                # Entry was kept but still missed (shouldn't happen normally)
                pass
            elif action == EvictionAction.EVICT and was_evicted:
                # Bad decision - evicted entry that was needed
                self._update_q_value(state, action, self.REWARD_EVICT_NEEDED_LATER)

            self._misses_after_keep += 1

    def record_eviction_success(self, key: str) -> None:
        """Record successful eviction that freed space for new entry.

        Args:
            key: Cache key that was evicted
        """
        if key in self._recent_decisions:
            state, action = self._recent_decisions[key]

            if action == EvictionAction.EVICT:
                # Good decision - freed space
                self._update_q_value(state, action, self.REWARD_EVICT_FREED_SPACE)
                self._evictions += 1

    def _get_q_values(self, state: CacheEntryState) -> Dict[EvictionAction, float]:
        """Get Q-values for all actions in given state.

        Args:
            state: Cache entry state

        Returns:
            Dictionary mapping actions to Q-values
        """
        if self._external_learner:
            # Use external learner
            try:
                rec = self._external_learner.get_recommendation(
                    provider="cache",
                    model="eviction",
                    task_type=state.tool_type,
                )
                if rec:
                    # Convert recommendation to Q-values
                    base_q = rec.value if isinstance(rec.value, (int, float)) else 0.5
                    return {
                        EvictionAction.KEEP: base_q,
                        EvictionAction.EVICT: 1.0 - base_q,
                        EvictionAction.PROMOTE: base_q * 0.8,
                        EvictionAction.DEMOTE: (1.0 - base_q) * 0.8,
                    }
            except Exception as e:
                logger.debug(f"External learner unavailable: {e}")

        # Use local Q-table
        feature_key = state.to_feature_tuple()

        if feature_key not in self._q_table:
            # Initialize with heuristic defaults
            self._q_table[feature_key] = self._get_default_q_values(state)

        return self._q_table[feature_key]

    def _get_default_q_values(self, state: CacheEntryState) -> Dict[EvictionAction, float]:
        """Get default Q-values based on heuristics.

        Args:
            state: Cache entry state

        Returns:
            Default Q-values for initialization
        """
        # Start with neutral values
        q_values = {
            EvictionAction.KEEP: 0.5,
            EvictionAction.EVICT: 0.5,
            EvictionAction.PROMOTE: 0.4,
            EvictionAction.DEMOTE: 0.4,
        }

        # Heuristic: high hit count = keep
        if state.hit_count >= 3:
            q_values[EvictionAction.KEEP] += 0.2
            q_values[EvictionAction.EVICT] -= 0.2

        # Heuristic: old unused = evict
        if state.entry_age_seconds > 600 and state.hit_count == 0:
            q_values[EvictionAction.EVICT] += 0.3
            q_values[EvictionAction.KEEP] -= 0.2

        # Heuristic: high relevance = keep
        if state.context_relevance > 0.7:
            q_values[EvictionAction.KEEP] += 0.15
            q_values[EvictionAction.PROMOTE] += 0.1

        # Heuristic: large size = demote
        if state.size_bytes > 10000:
            q_values[EvictionAction.DEMOTE] += 0.1

        return q_values

    def _update_q_value(
        self,
        state: CacheEntryState,
        action: EvictionAction,
        reward: float,
    ) -> None:
        """Update Q-value using Q-learning update rule.

        Args:
            state: State when decision was made
            action: Action that was taken
            reward: Reward received
        """
        if self._external_learner:
            # Delegate to external learner
            try:
                from victor.framework.rl.base import RLOutcome

                outcome = RLOutcome(
                    provider="cache",
                    model="eviction",
                    task_type=state.tool_type,
                    success=reward > 0,
                    quality_score=max(0, min(1, (reward + 1) / 2)),
                    metadata={
                        "action": action.value,
                        "hit_count": state.hit_count,
                        "age_seconds": state.entry_age_seconds,
                    },
                )
                self._external_learner.record_outcome(outcome)
                return
            except Exception as e:
                logger.debug(f"Failed to record to external learner: {e}")

        # Update local Q-table
        feature_key = state.to_feature_tuple()

        if feature_key not in self._q_table:
            self._q_table[feature_key] = self._get_default_q_values(state)

        current_q = self._q_table[feature_key][action]

        # Simplified Q-learning (no next state)
        target = reward
        new_q = current_q + self.learning_rate * (target - current_q)

        self._q_table[feature_key][action] = new_q

        logger.debug(f"Q-update: {feature_key}:{action.value} {current_q:.3f} -> {new_q:.3f}")

    def _track_decision(self, state: CacheEntryState, action: EvictionAction) -> None:
        """Track decision for later feedback.

        Args:
            state: Entry state
            action: Decision made
        """
        self._recent_decisions[state.key] = (state, action)

        # Limit tracked decisions
        if len(self._recent_decisions) > self._max_tracked_decisions:
            # Remove oldest entries
            oldest_keys = list(self._recent_decisions.keys())[: self._max_tracked_decisions // 2]
            for key in oldest_keys:
                del self._recent_decisions[key]

    def get_entries_to_evict(
        self,
        entries: List[CacheEntryState],
        target_count: int,
        cache_utilization: float = 0.9,
    ) -> List[str]:
        """Get list of entry keys to evict.

        Evaluates all entries and returns those recommended for eviction.

        Args:
            entries: List of cache entry states
            target_count: Target number of entries to evict
            cache_utilization: Current cache utilization

        Returns:
            List of keys to evict
        """
        # Score all entries
        scored_entries = []
        for entry in entries:
            decision = self.get_decision(entry, cache_utilization)
            if decision.action == EvictionAction.EVICT:
                # Higher score = more likely to evict
                scored_entries.append((entry.key, decision.q_value, decision))

        # Sort by Q-value for eviction (highest first)
        scored_entries.sort(key=lambda x: x[1], reverse=True)

        # Return top entries to evict
        return [key for key, _, _ in scored_entries[:target_count]]

    def export_metrics(self) -> Dict[str, Any]:
        """Export policy metrics.

        Returns:
            Dictionary with metrics
        """
        hit_rate = 0.0
        total_feedback = self._hits_after_keep + self._misses_after_keep
        if total_feedback > 0:
            hit_rate = self._hits_after_keep / total_feedback

        return {
            "total_decisions": self._total_decisions,
            "hits_after_keep": self._hits_after_keep,
            "misses_after_keep": self._misses_after_keep,
            "evictions": self._evictions,
            "hit_rate_after_keep": hit_rate,
            "q_table_size": len(self._q_table),
            "tracked_decisions": len(self._recent_decisions),
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
        }


# Global singleton
_rl_eviction_policy: Optional[RLEvictionPolicy] = None


def get_rl_eviction_policy(
    cache_eviction_learner: Optional[Any] = None,
) -> RLEvictionPolicy:
    """Get global RL eviction policy (lazy init).

    Args:
        cache_eviction_learner: Optional external learner

    Returns:
        RLEvictionPolicy singleton
    """
    global _rl_eviction_policy
    if _rl_eviction_policy is None:
        _rl_eviction_policy = RLEvictionPolicy(cache_eviction_learner=cache_eviction_learner)
    return _rl_eviction_policy
