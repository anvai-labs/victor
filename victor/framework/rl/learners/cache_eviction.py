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

"""RL learner for intelligent cache eviction policy.

This learner uses Q-learning to learn optimal cache eviction decisions
beyond simple TTL-based expiration.

Strategy:
- State: (cache_utilization_bucket, entry_age_bucket, hit_count, tool_type)
- Action: (evict, keep, promote_to_l1, demote_to_l2)
- Reward: hit_rate_improvement, memory_efficiency, latency_improvement

Sprint 3: Cache & Grounding Learners
"""

import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables
from victor.framework.rl.migration import RLTableMigrator

logger = logging.getLogger(__name__)


class CacheEvictionAction:
    """Possible cache eviction actions."""

    EVICT = "evict"
    KEEP = "keep"
    PROMOTE_L1 = "promote_to_l1"
    DEMOTE_L2 = "demote_to_l2"


class UtilizationLevel(str, Enum):
    """Cache utilization level for eviction decisions."""

    LOW = "low"  # < 50%
    MEDIUM = "medium"  # 50-75%
    HIGH = "high"  # 75-90%
    CRITICAL = "critical"  # > 90%


class CacheAge(str, Enum):
    """Cache entry age bucket."""

    FRESH = "fresh"  # < 1 minute
    RECENT = "recent"  # 1-5 minutes
    WARM = "warm"  # 5-60 minutes
    STALE = "stale"  # > 1 hour


class HitFrequency(str, Enum):
    """Cache entry hit frequency."""

    ZERO = "zero"  # No hits
    LOW = "low"  # 1-2 hits
    MEDIUM = "medium"  # 3-9 hits
    HIGH = "high"  # 10+ hits


class ToolType(str, Enum):
    """Tool type category for cache state representation."""

    SEARCH = "search"
    READ = "read"
    COMPUTE = "compute"
    OTHER = "other"


class CacheEvictionLearner(BaseLearner):
    """Learn optimal cache eviction policy using Q-learning.

    Uses prioritized experience replay with Q-learning to learn
    cache entry value predictions.

    Attributes:
        name: Always "cache_eviction"
        db: SQLite database connection
        learning_rate: Q-value update rate (alpha), default 0.1
        discount_factor: Future reward discount (gamma), default 0.95
        epsilon: Exploration rate, default 0.1
    """

    # Default Q-value for unseen state-action pairs
    DEFAULT_Q_VALUE = 0.0

    # Q-learning parameters
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_DISCOUNT_FACTOR = 0.95
    DEFAULT_EPSILON = 0.1

    # Minimum samples before confident recommendation
    MIN_SAMPLES_FOR_CONFIDENCE = 15

    # Valid actions
    ACTIONS = [
        CacheEvictionAction.EVICT,
        CacheEvictionAction.KEEP,
        CacheEvictionAction.PROMOTE_L1,
        CacheEvictionAction.DEMOTE_L2,
    ]

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        provider_adapter: Optional[Any] = None,
        discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
        epsilon: float = DEFAULT_EPSILON,
    ):
        """Initialize cache eviction learner.

        Args:
            name: Learner name (should be "cache_eviction")
            db_connection: SQLite database connection
            learning_rate: Q-value update rate (default 0.1)
            provider_adapter: Optional provider adapter
            discount_factor: Future reward discount (default 0.95)
            epsilon: Exploration rate (default 0.1)
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)

        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # In-memory caches for fast access
        self._q_values: Dict[str, Dict[str, float]] = {}  # state_key -> {action -> Q-value}
        self._visit_counts: Dict[str, Dict[str, int]] = {}  # state_key -> {action -> count}
        self._tool_value_estimates: Dict[str, float] = {}  # tool_name -> estimated value
        self._total_decisions: int = 0

        # Hit rate tracking per tool type
        self._tool_hit_rates: Dict[str, Tuple[int, int]] = {}  # tool_name -> (hits, misses)

        # Load state from database
        self._load_state()

    def _ensure_tables(self) -> None:
        """Migrate legacy per-learner tables to unified RL tables."""
        RLTableMigrator(self.db).run_if_needed(self.name, RLTableMigrator.migrate_cache_eviction)

    def _load_state(self) -> None:
        """Load state from database."""
        cursor = self.db.cursor()

        try:
            cursor.execute(
                f"SELECT state_key, action_key, q_value, visit_count FROM {Tables.RL_Q_VALUE}"
                f" WHERE learner_id = ?",
                (self.name,),
            )
            for row in cursor.fetchall():
                row_dict = dict(row)
                state_key = row_dict["state_key"]
                action = row_dict["action_key"]

                if state_key not in self._q_values:
                    self._q_values[state_key] = {}
                    self._visit_counts[state_key] = {}

                self._q_values[state_key][action] = row_dict["q_value"]
                self._visit_counts[state_key][action] = row_dict["visit_count"]
                self._total_decisions += row_dict["visit_count"]

        except Exception as e:
            logger.debug(f"RL: Could not load Q-values: {e}")

        try:
            cursor.execute(
                f"SELECT task_type, stat_key, stat_value FROM {Tables.RL_TASK_STAT}"
                f" WHERE learner_id = ?",
                (self.name,),
            )
            tool_stats: Dict[str, Dict[str, float]] = {}
            for row in cursor.fetchall():
                row_dict = dict(row)
                tool_name = row_dict["task_type"]
                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {}
                tool_stats[tool_name][row_dict["stat_key"]] = row_dict["stat_value"]

            for tool_name, stats in tool_stats.items():
                self._tool_value_estimates[tool_name] = stats.get("estimated_value", 0.5)
                self._tool_hit_rates[tool_name] = (
                    int(stats.get("hit_count", 0)),
                    int(stats.get("miss_count", 0)),
                )

        except Exception as e:
            logger.debug(f"RL: Could not load tool values: {e}")

        if self._q_values:
            logger.info(f"RL: Loaded {len(self._q_values)} cache eviction states from database")

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record cache eviction outcome and update Q-values.

        Expected metadata:
        - state_key: Discretized state (utilization:age:hits:tool_type)
        - action: Action taken (evict, keep, promote_to_l1, demote_to_l2)
        - tool_name: Name of the cached tool result
        - hit_after: 1 if cache was hit after keeping, 0 otherwise
        - memory_saved: Memory saved (positive for eviction, negative for keep)
        - latency_delta: Latency improvement (positive = faster)

        Args:
            outcome: Outcome with cache eviction data
        """
        state_key = outcome.metadata.get("state_key")
        action = outcome.metadata.get("action")
        tool_name = outcome.metadata.get("tool_name", "unknown")

        if not state_key or not action:
            logger.debug("RL: cache_eviction outcome missing required fields, skipping")
            return

        # Compute reward
        reward = self._compute_reward(outcome)

        # Get current Q-value
        old_q = self._get_q_value(state_key, action)

        # Simple TD update (no next state for cache decisions)
        new_q = old_q + self.learning_rate * (reward - old_q)

        # Update caches
        if state_key not in self._q_values:
            self._q_values[state_key] = {}
            self._visit_counts[state_key] = {}

        self._q_values[state_key][action] = new_q
        self._visit_counts[state_key][action] = self._visit_counts[state_key].get(action, 0) + 1
        self._total_decisions += 1

        # Update tool value estimate
        self._update_tool_value(tool_name, outcome)

        # Persist to database
        self._save_to_db(state_key, action, tool_name, reward, outcome)

        logger.debug(
            f"RL: Cache eviction {action} for '{tool_name}' Q-value: {old_q:.3f} → {new_q:.3f} "
            f"(reward={reward:.3f})"
        )

    def _save_to_db(
        self,
        state_key: str,
        action: str,
        tool_name: str,
        reward: float,
        outcome: RLOutcome,
    ) -> None:
        """Save Q-values and outcome to database."""
        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()

        # Save Q-value
        cursor.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.RL_Q_VALUE}
            (learner_id, state_key, action_key, q_value, visit_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.name,
                state_key,
                action,
                self._q_values[state_key][action],
                self._visit_counts[state_key][action],
                timestamp,
            ),
        )

        # Save eviction history to rl_transition
        hit_after = outcome.metadata.get("hit_after", 0)
        cursor.execute(
            f"""
            INSERT INTO {Tables.RL_TRANSITION}
            (learner_id, from_state, to_state, action, reward, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.name,
                state_key,
                action,
                tool_name,
                reward,
                str(hit_after),
                timestamp,
            ),
        )

        self.db.commit()

    def _update_tool_value(self, tool_name: str, outcome: RLOutcome) -> None:
        """Update tool value estimate based on cache outcome."""
        hit_after = outcome.metadata.get("hit_after", 0)
        action = outcome.metadata.get("action", "keep")

        # Update hit rate tracking
        if tool_name not in self._tool_hit_rates:
            self._tool_hit_rates[tool_name] = (0, 0)

        hits, misses = self._tool_hit_rates[tool_name]
        if action == CacheEvictionAction.KEEP:
            if hit_after:
                hits += 1
            else:
                misses += 1

        self._tool_hit_rates[tool_name] = (hits, misses)

        # Update value estimate
        total = hits + misses
        if total > 0:
            value = hits / total
        else:
            value = 0.5  # Default

        self._tool_value_estimates[tool_name] = value

        # Save to database via rl_task_stat (3 rows per tool_name)
        cursor = self.db.cursor()
        ts = datetime.now().isoformat()
        for stat_key, stat_value in (
            ("estimated_value", value),
            ("hit_count", float(hits)),
            ("miss_count", float(misses)),
        ):
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {Tables.RL_TASK_STAT}
                (learner_id, task_type, stat_key, stat_value, sample_count, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (self.name, tool_name, stat_key, stat_value, hits + misses, ts),
            )
        self.db.commit()

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get cache eviction recommendation for given state.

        Note: The 'provider' parameter is overloaded to contain the state_key.

        Args:
            provider: State key (overloaded parameter)
            model: Not used
            task_type: Task type for context

        Returns:
            Recommendation with best action and Q-value
        """
        state_key = provider  # Overloaded

        # Get all actions for this state
        actions = self._get_all_actions(state_key)

        if not actions:
            # Default to keep if no data
            return RLRecommendation(
                value=CacheEvictionAction.KEEP,
                confidence=0.3,
                reason="No learned data, defaulting to keep",
                sample_size=0,
                is_baseline=True,
            )

        # Get total visits for this state
        total_visits = sum(self._visit_counts.get(state_key, {}).values())

        # Epsilon-greedy selection
        import random

        if self._rng.random() < self.epsilon:
            action = self._rng.choice(self.ACTIONS)
            return RLRecommendation(
                value=action,
                confidence=0.3,
                reason=f"Exploration (ε={self.epsilon})",
                sample_size=total_visits,
                is_baseline=True,
            )

        # Exploitation: best action
        best_action = max(actions.keys(), key=lambda a: actions[a])
        best_q = actions[best_action]

        # Compute confidence
        action_visits = self._visit_counts.get(state_key, {}).get(best_action, 0)
        if action_visits < self.MIN_SAMPLES_FOR_CONFIDENCE:
            confidence = 0.3 + 0.2 * (action_visits / self.MIN_SAMPLES_FOR_CONFIDENCE)
            is_baseline = True
        else:
            confidence = min(0.95, 0.5 + 0.45 * (1 - math.exp(-action_visits / 20)))
            is_baseline = False

        return RLRecommendation(
            value=best_action,
            confidence=confidence,
            reason=f"Q={best_q:.3f}, visits={action_visits}",
            sample_size=total_visits,
            is_baseline=is_baseline,
        )

    def get_eviction_decision(
        self,
        utilization: float,
        age_seconds: float,
        hit_count: int,
        tool_name: str,
    ) -> Tuple[str, float]:
        """Get eviction decision for a cache entry.

        Convenience method that builds state key and returns decision.

        Args:
            utilization: Cache utilization (0-1)
            age_seconds: Entry age in seconds
            hit_count: Number of hits for this entry
            tool_name: Tool that produced the cached result

        Returns:
            Tuple of (action, confidence)
        """
        # Discretize state
        util_bucket = self._discretize_utilization(utilization)
        age_bucket = self._discretize_age(age_seconds)
        hit_bucket = self._discretize_hits(hit_count)
        tool_type = self._get_tool_type(tool_name)

        state_key = f"{util_bucket}:{age_bucket}:{hit_bucket}:{tool_type}"

        rec = self.get_recommendation(state_key, "", "cache")
        return (
            rec.value if rec else CacheEvictionAction.KEEP,
            rec.confidence if rec else 0.3,
        )

    def get_tool_value(self, tool_name: str) -> float:
        """Get estimated value for a tool's cache entries.

        Args:
            tool_name: Tool name

        Returns:
            Value estimate (0-1, higher = more valuable to cache)
        """
        return self._tool_value_estimates.get(tool_name, 0.5)

    def _get_q_value(self, state_key: str, action: str) -> float:
        """Get Q-value for a state-action pair."""
        return self._q_values.get(state_key, {}).get(action, self.DEFAULT_Q_VALUE)

    def _get_all_actions(self, state_key: str) -> Dict[str, float]:
        """Get all Q-values for a state."""
        return self._q_values.get(state_key, {})

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward from eviction outcome.

        Reward based on:
        - Hit rate improvement (60%): Did keeping lead to hits?
        - Memory efficiency (30%): Was memory used well?
        - Latency improvement (10%): Did it improve latency?

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (-1.0 to 1.0)
        """
        reward = 0.0
        action = outcome.metadata.get("action", "keep")

        # Hit rate component (60%)
        hit_after = outcome.metadata.get("hit_after", 0)
        if action == CacheEvictionAction.EVICT:
            # Eviction is neutral for hits (entry not available)
            reward += 0.0
        elif action == CacheEvictionAction.KEEP:
            # Reward for keeping if it was hit, penalize if not
            reward += 0.6 * (1.0 if hit_after else -0.5)
        elif action == CacheEvictionAction.PROMOTE_L1:
            # Higher reward for promotion if hit
            reward += 0.6 * (1.5 if hit_after else -0.3)
        elif action == CacheEvictionAction.DEMOTE_L2:
            # Moderate reward for demotion if hit
            reward += 0.6 * (0.5 if hit_after else 0.2)

        # Memory efficiency (30%)
        memory_saved = outcome.metadata.get("memory_saved", 0)
        memory_reward = min(1.0, memory_saved / 1000000) if memory_saved > 0 else 0
        reward += 0.3 * memory_reward

        # Latency improvement (10%)
        latency_delta = outcome.metadata.get("latency_delta", 0)
        latency_reward = min(1.0, latency_delta / 100) if latency_delta > 0 else 0
        reward += 0.1 * latency_reward

        return max(-1.0, min(1.0, reward))

    def _discretize_utilization(self, utilization: float) -> UtilizationLevel:
        """Discretize cache utilization to bucket."""
        if utilization < 0.5:
            return UtilizationLevel.LOW
        elif utilization < 0.75:
            return UtilizationLevel.MEDIUM
        elif utilization < 0.9:
            return UtilizationLevel.HIGH
        else:
            return UtilizationLevel.CRITICAL

    def _discretize_age(self, age_seconds: float) -> CacheAge:
        """Discretize entry age to bucket."""
        if age_seconds < 60:
            return CacheAge.FRESH  # < 1 minute
        elif age_seconds < 300:
            return CacheAge.RECENT  # < 5 minutes
        elif age_seconds < 3600:
            return CacheAge.WARM  # < 1 hour
        else:
            return CacheAge.STALE  # >= 1 hour

    def _discretize_hits(self, hit_count: int) -> HitFrequency:
        """Discretize hit count to bucket."""
        if hit_count == 0:
            return HitFrequency.ZERO
        elif hit_count < 3:
            return HitFrequency.LOW
        elif hit_count < 10:
            return HitFrequency.MEDIUM
        else:
            return HitFrequency.HIGH

    def _get_tool_type(self, tool_name: str) -> ToolType:
        """Get tool type category for state representation."""
        # Map tools to categories for generalization
        search_tools = {"code_search", "semantic_code_search", "web_search", "grep"}
        read_tools = {"read", "list_directory", "file_viewer"}
        compute_tools = {"code_review", "analyze_code", "refactor"}

        if tool_name in search_tools:
            return ToolType.SEARCH
        elif tool_name in read_tools:
            return ToolType.READ
        elif tool_name in compute_tools:
            return ToolType.COMPUTE
        else:
            return ToolType.OTHER

    def export_metrics(self) -> Dict[str, Any]:
        """Export learner metrics for monitoring.

        Returns:
            Dictionary with learner stats
        """
        return {
            "learner": self.name,
            "total_states": len(self._q_values),
            "total_decisions": self._total_decisions,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "tools_tracked": len(self._tool_value_estimates),
            "tool_values": dict(list(self._tool_value_estimates.items())[:10]),
        }
