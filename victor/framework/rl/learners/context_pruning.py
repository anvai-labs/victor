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

"""RL learner for optimal context pruning to minimize token usage.

This learner uses Q-learning to learn when and how aggressively to prune
conversation context to reduce token consumption while preserving task success.

Strategy:
- State: (context_utilization_bucket, tool_call_count, task_complexity, provider_type)
- Action: (no_prune, light_prune, moderate_prune, aggressive_prune)
- Reward: token_efficiency_gain * task_success - context_loss_penalty

Design:
- Learns per-provider optimal pruning thresholds (cloud vs local models differ)
- Balances token savings against task success rate
- Uses Thompson Sampling for exploration to find optimal pruning levels
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables
from victor.framework.rl.migration import RLTableMigrator

logger = logging.getLogger(__name__)


class PruningAction(str, Enum):
    """Possible context pruning actions."""

    NO_PRUNE = "no_prune"  # Keep all context
    LIGHT = "light"  # Prune 20% oldest tool results
    MODERATE = "moderate"  # Prune 40% oldest messages
    AGGRESSIVE = "aggressive"  # Prune to 50% target utilization


@dataclass
class PruningConfig:
    """Configuration for a pruning action."""

    action: PruningAction
    compaction_threshold: float  # When to trigger (0.0-1.0)
    compaction_target: float  # Target utilization after pruning (0.0-1.0)
    tool_result_max_chars: int  # Max chars per tool result
    min_messages_keep: int  # Minimum messages to preserve


# Default configs for each action
PRUNING_CONFIGS = {
    PruningAction.NO_PRUNE: PruningConfig(
        action=PruningAction.NO_PRUNE,
        compaction_threshold=0.95,
        compaction_target=0.90,
        tool_result_max_chars=12000,
        min_messages_keep=15,
    ),
    PruningAction.LIGHT: PruningConfig(
        action=PruningAction.LIGHT,
        compaction_threshold=0.80,
        compaction_target=0.65,
        tool_result_max_chars=8000,
        min_messages_keep=10,
    ),
    PruningAction.MODERATE: PruningConfig(
        action=PruningAction.MODERATE,
        compaction_threshold=0.70,
        compaction_target=0.50,
        tool_result_max_chars=5000,
        min_messages_keep=6,
    ),
    PruningAction.AGGRESSIVE: PruningConfig(
        action=PruningAction.AGGRESSIVE,
        compaction_threshold=0.60,
        compaction_target=0.40,
        tool_result_max_chars=3000,
        min_messages_keep=4,
    ),
}


class ContextPruningLearner(BaseLearner):
    """Learn optimal context pruning policy using Q-learning with Thompson Sampling.

    Uses Thompson Sampling for exploration to find optimal pruning levels
    that maximize token efficiency while maintaining task success.

    Attributes:
        name: Always "context_pruning"
        db: SQLite database connection
        learning_rate: Q-value update rate (alpha), default 0.15
        discount_factor: Future reward discount (gamma), default 0.90
    """

    # Q-learning parameters
    DEFAULT_LEARNING_RATE = 0.15
    DEFAULT_DISCOUNT_FACTOR = 0.90

    # Thompson Sampling priors (Beta distribution)
    PRIOR_ALPHA = 1.0
    PRIOR_BETA = 1.0

    # Minimum samples before confident recommendation
    MIN_SAMPLES_FOR_CONFIDENCE = 10

    # Valid actions
    ACTIONS = list(PruningAction)

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        provider_adapter: Optional[Any] = None,
        discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
    ):
        """Initialize context pruning learner.

        Args:
            name: Learner name (should be "context_pruning")
            db_connection: SQLite database connection
            learning_rate: Q-value update rate (default 0.15)
            provider_adapter: Optional provider adapter
            discount_factor: Future reward discount (default 0.90)
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)
        self.discount_factor = discount_factor

    def _ensure_tables(self) -> None:
        """Migrate legacy per-learner tables to unified RL tables."""
        RLTableMigrator(self.db).run_if_needed(self.name, RLTableMigrator.migrate_context_pruning)

    def _compute_reward(self, outcome: Any) -> float:
        """Compute reward from a standard RLOutcome.

        ContextPruningLearner primarily uses its own ``record_outcome()``
        with domain-specific arguments.  This method provides the
        ``BaseLearner`` abstract contract for the generic path.
        """
        success = getattr(outcome, "success", False)
        quality = getattr(outcome, "quality_score", 0.5)
        return (0.6 * (1.0 if success else -0.5)) + (0.4 * quality)

    def _discretize_state(
        self,
        context_utilization: float,
        tool_call_count: int,
        task_complexity: str,
        provider_type: str,
    ) -> str:
        """Convert continuous state to discrete state key.

        Args:
            context_utilization: Current context usage (0.0-1.0)
            tool_call_count: Number of tool calls made
            task_complexity: Task complexity (simple, medium, complex)
            provider_type: Provider type (cloud, local)

        Returns:
            State key string for Q-table lookup
        """
        # Discretize context utilization into buckets
        if context_utilization < 0.3:
            util_bucket = "low"
        elif context_utilization < 0.6:
            util_bucket = "medium"
        elif context_utilization < 0.8:
            util_bucket = "high"
        else:
            util_bucket = "critical"

        # Discretize tool call count
        if tool_call_count < 3:
            tool_bucket = "few"
        elif tool_call_count < 7:
            tool_bucket = "moderate"
        else:
            tool_bucket = "many"

        return f"{provider_type}:{util_bucket}:{tool_bucket}:{task_complexity}"

    def get_recommendation(
        self,
        context_utilization: float,
        tool_call_count: int,
        task_complexity: str = "medium",
        provider_type: str = "cloud",
    ) -> RLRecommendation:
        """Get recommended pruning action for current state.

        Uses Thompson Sampling to balance exploration and exploitation.

        Args:
            context_utilization: Current context usage (0.0-1.0)
            tool_call_count: Number of tool calls made
            task_complexity: Task complexity level
            provider_type: Provider type (cloud, local)

        Returns:
            RLRecommendation with action and config
        """
        state_key = self._discretize_state(
            context_utilization, tool_call_count, task_complexity, provider_type
        )

        cursor = self.db.cursor()

        # Get Q-values and visit counts for all actions in this state from rl_q_value
        cursor.execute(
            f"""
            SELECT action_key, q_value, visit_count
            FROM {Tables.RL_Q_VALUE}
            WHERE learner_id = ? AND state_key = ?
            """,
            (self.name, state_key),
        )
        rows = cursor.fetchall()

        # Build action stats dict (success_count estimated from q_value proportionally)
        action_stats = {}
        for row in rows:
            row_dict = dict(row)
            visits = row_dict["visit_count"]
            q_val = row_dict["q_value"]
            # Estimate successes from q_value (scaled 0-100) and visit count
            successes = max(0, int((q_val / 100.0) * visits))
            action_stats[row_dict["action_key"]] = {
                "q_value": q_val,
                "visits": visits,
                "successes": successes,
            }

        # Thompson Sampling: sample from Beta distribution for each action
        import random

        best_action = None
        best_sample = -float("inf")

        for action in self.ACTIONS:
            stats = action_stats.get(
                action.value,
                {"q_value": 0.0, "visits": 0, "successes": 0},
            )

            # Beta distribution parameters
            alpha = self.PRIOR_ALPHA + stats["successes"]
            beta = self.PRIOR_BETA + stats["visits"] - stats["successes"]

            # Sample from Beta distribution
            sample = self._rng.betavariate(alpha, max(1, beta))

            # Weight by Q-value if we have enough samples
            if stats["visits"] >= self.MIN_SAMPLES_FOR_CONFIDENCE:
                sample = 0.7 * sample + 0.3 * (stats["q_value"] / 100.0)

            if sample > best_sample:
                best_sample = sample
                best_action = action

        # Default to MODERATE if no data
        if best_action is None:
            best_action = PruningAction.MODERATE

        config = PRUNING_CONFIGS[best_action]

        total_visits = sum(s.get("visits", 0) for s in action_stats.values())
        confidence = min(1.0, total_visits / (self.MIN_SAMPLES_FOR_CONFIDENCE * len(self.ACTIONS)))

        return RLRecommendation(
            value=best_action.value,
            confidence=confidence,
            reason=f"Thompson Sampling selected {best_action.value} (sample={best_sample:.3f})",
            sample_size=total_visits,
            metadata={
                "state_key": state_key,
                "config": {
                    "compaction_threshold": config.compaction_threshold,
                    "compaction_target": config.compaction_target,
                    "tool_result_max_chars": config.tool_result_max_chars,
                    "min_messages_keep": config.min_messages_keep,
                },
                "exploration_sample": best_sample,
            },
        )

    def record_outcome(
        self,
        context_utilization: float,
        tool_call_count: int,
        action: str,
        task_success: bool,
        tokens_saved: int,
        task_complexity: str = "medium",
        provider_type: str = "cloud",
    ) -> None:
        """Record outcome of a pruning decision.

        Args:
            context_utilization: Context usage when decision was made
            tool_call_count: Number of tool calls at decision time
            action: The pruning action taken
            task_success: Whether the task completed successfully
            tokens_saved: Estimated tokens saved by pruning
            task_complexity: Task complexity level
            provider_type: Provider type
        """
        state_key = self._discretize_state(
            context_utilization, tool_call_count, task_complexity, provider_type
        )

        # Calculate reward: balance token savings vs task success
        # Normalize tokens_saved to 0-1 range (assume max savings ~50K tokens)
        normalized_savings = min(1.0, tokens_saved / 50000.0)
        success_bonus = 1.0 if task_success else -0.5  # Penalty for failure
        reward = (0.4 * normalized_savings + 0.6 * success_bonus) * 100  # Scale to 0-100

        cursor = self.db.cursor()
        now = datetime.now(timezone.utc).isoformat()

        # Read existing Q-value for this state-action pair
        existing = cursor.execute(
            f"SELECT q_value, visit_count FROM {Tables.RL_Q_VALUE}"
            f" WHERE learner_id = ? AND state_key = ? AND action_key = ?",
            (self.name, state_key, action),
        ).fetchone()

        if existing:
            old_q = dict(existing)["q_value"]
            visit_count = dict(existing)["visit_count"] + 1
            new_q = old_q + self.learning_rate * (reward - old_q)
        else:
            new_q = reward
            visit_count = 1

        # Write updated Q-value to rl_q_value
        cursor.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.RL_Q_VALUE}
            (learner_id, state_key, action_key, q_value, visit_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (self.name, state_key, action, new_q, visit_count, now),
        )

        # Update provider-level stats in rl_task_stat
        for stat_key, delta in (
            ("total_decisions", 1.0),
            ("total_tokens_saved", float(tokens_saved)),
        ):
            cursor.execute(
                f"""
                INSERT INTO {Tables.RL_TASK_STAT}
                (learner_id, task_type, stat_key, stat_value, sample_count, updated_at)
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(learner_id, task_type, stat_key) DO UPDATE SET
                    stat_value = stat_value + ?,
                    sample_count = sample_count + 1,
                    updated_at = excluded.updated_at
                """,
                (self.name, provider_type, stat_key, delta, now, delta),
            )

        self.db.commit()
        logger.debug(
            f"Recorded pruning outcome: state={state_key}, action={action}, "
            f"success={task_success}, tokens_saved={tokens_saved}, reward={reward:.1f}"
        )

    def get_stats(self, provider_type: Optional[str] = None) -> Dict[str, Any]:
        """Get learner statistics.

        Args:
            provider_type: Optional filter by provider type

        Returns:
            Dictionary with learning statistics
        """
        cursor = self.db.cursor()

        # Get per-action stats from rl_q_value
        if provider_type:
            cursor.execute(
                f"""
                SELECT action_key, SUM(visit_count), AVG(q_value)
                FROM {Tables.RL_Q_VALUE}
                WHERE learner_id = ? AND state_key LIKE ?
                GROUP BY action_key
                """,
                (self.name, f"{provider_type}:%"),
            )
        else:
            cursor.execute(
                f"""
                SELECT action_key, SUM(visit_count), AVG(q_value)
                FROM {Tables.RL_Q_VALUE}
                WHERE learner_id = ?
                GROUP BY action_key
                """,
                (self.name,),
            )

        action_stats = {}
        for row in cursor.fetchall():
            row_dict = dict(row)
            visits = row_dict["SUM(visit_count)"] or 0
            avg_q = row_dict["AVG(q_value)"] or 0.0
            action_stats[row_dict["action_key"]] = {
                "visits": visits,
                "avg_q_value": avg_q,
                "success_rate": max(0.0, avg_q / 100.0),
            }

        # Get overall stats from rl_task_stat
        cursor.execute(
            f"""
            SELECT stat_key, SUM(stat_value)
            FROM {Tables.RL_TASK_STAT}
            WHERE learner_id = ? AND stat_key IN ('total_decisions', 'total_tokens_saved')
            GROUP BY stat_key
            """,
            (self.name,),
        )
        totals = {dict(r)["stat_key"]: dict(r)["SUM(stat_value)"] for r in cursor.fetchall()}

        return {
            "total_decisions": int(totals.get("total_decisions", 0) or 0),
            "total_tokens_saved": int(totals.get("total_tokens_saved", 0) or 0),
            "action_stats": action_stats,
            "best_action": max(
                action_stats.items(),
                key=lambda x: x[1].get("avg_q_value", 0),
                default=(None, {}),
            )[0],
        }

    def reset(self) -> None:
        """Reset learner state (clear Q-values)."""
        cursor = self.db.cursor()
        cursor.execute(f"DELETE FROM {Tables.RL_Q_VALUE} WHERE learner_id = ?", (self.name,))
        cursor.execute(f"DELETE FROM {Tables.RL_TASK_STAT} WHERE learner_id = ?", (self.name,))
        self.db.commit()
        logger.info("Context pruning learner reset")
