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

"""RL learner for adaptive mode transitions using Q-learning.

This learner unifies the Q-learning from AdaptiveModeController with the
RLCoordinator framework for centralized observability and cross-learner analysis.

Strategy:
- Full Q-learning with temporal difference: Q(s,a) = Q(s,a) + α(r + γ*max(Q(s',a')) - Q(s,a))
- State: (current_mode, task_type, resource_ratio, quality_bucket)
- Action: (target_mode, budget_adjustment)
- Reward: task_completion, quality_score, efficiency

Sprint 2: Mode Controller Unification
"""

import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.framework.rl.migration import RLTableMigrator
from victor.core.schema import Tables

logger = logging.getLogger(__name__)


class RLAgentMode(Enum):
    """Extended agent operation modes for RL-based mode transitions.

    This is an extension of the base AgentMode (victor.agent.mode_controller.AgentMode)
    with additional modes (REVIEW, COMPLETE) for reinforcement learning state tracking.

    Renamed from AgentMode to be semantically distinct:
    - AgentMode (victor.agent.mode_controller): Base 3 modes (BUILD, PLAN, EXPLORE)
    - RLAgentMode: Extended 5 modes for RL state machine
    """

    EXPLORE = "explore"
    PLAN = "plan"
    BUILD = "build"
    REVIEW = "review"
    COMPLETE = "complete"


# Backward compatibility alias
AgentMode = RLAgentMode


class ModeTransitionLearner(BaseLearner):
    """Learn optimal mode transitions using Q-learning.

    Unifies the Q-learning from AdaptiveModeController with RLCoordinator
    for centralized storage and observability.

    Attributes:
        name: Always "mode_transition"
        db: SQLite database connection (shared with other learners)
        learning_rate: Q-value update rate (alpha), default 0.1
        discount_factor: Future reward discount (gamma), default 0.9
        epsilon: Exploration rate, default 0.1
    """

    # Default Q-value for unseen state-action pairs
    DEFAULT_Q_VALUE = 0.0

    # Q-learning parameters
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_DISCOUNT_FACTOR = 0.9
    DEFAULT_EPSILON = 0.1

    # Minimum samples before confident recommendation
    MIN_SAMPLES_FOR_CONFIDENCE = 10

    # Valid mode transitions
    VALID_TRANSITIONS = {
        AgentMode.EXPLORE: [AgentMode.PLAN, AgentMode.BUILD, AgentMode.COMPLETE],
        AgentMode.PLAN: [AgentMode.BUILD, AgentMode.EXPLORE, AgentMode.COMPLETE],
        AgentMode.BUILD: [AgentMode.REVIEW, AgentMode.EXPLORE, AgentMode.COMPLETE],
        AgentMode.REVIEW: [AgentMode.BUILD, AgentMode.COMPLETE],
        AgentMode.COMPLETE: [],
    }

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        provider_adapter: Optional[Any] = None,
        discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
        epsilon: float = DEFAULT_EPSILON,
    ):
        """Initialize mode transition learner.

        Args:
            name: Learner name (should be "mode_transition")
            db_connection: SQLite database connection
            learning_rate: Q-value update rate (default 0.1)
            provider_adapter: Optional provider adapter
            discount_factor: Future reward discount (default 0.9)
            epsilon: Exploration rate (default 0.1)
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)

        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # In-memory caches for fast access
        self._q_values: Dict[str, Dict[str, float]] = {}  # state_key -> {action_key -> Q-value}
        self._visit_counts: Dict[str, Dict[str, int]] = {}  # state_key -> {action_key -> count}
        self._task_stats: Dict[str, Dict[str, float]] = {}  # task_type -> stats
        self._total_transitions: int = 0

        # Load state from database
        self._load_state()

    def _ensure_tables(self) -> None:
        """Migrate legacy private tables into unified RL schema (idempotent)."""
        RLTableMigrator(self.db).run_if_needed(self.name, RLTableMigrator.migrate_mode_transition)

    def _load_state(self) -> None:
        """Load state from unified rl_q_value and rl_task_stat tables."""
        cursor = self.db.cursor()

        try:
            cursor.execute(
                f"SELECT state_key, action_key, q_value, visit_count FROM {Tables.RL_Q_VALUE} "
                f"WHERE learner_id = ?",
                (self.name,),
            )
            for row in cursor.fetchall():
                state_key, action_key, q_value, visit_count = row
                if state_key not in self._q_values:
                    self._q_values[state_key] = {}
                    self._visit_counts[state_key] = {}
                self._q_values[state_key][action_key] = q_value
                self._visit_counts[state_key][action_key] = visit_count
                self._total_transitions += visit_count
        except Exception as e:
            logger.debug(f"RL: Could not load Q-values: {e}")

        try:
            cursor.execute(
                f"SELECT task_type, stat_key, stat_value, sample_count FROM {Tables.RL_TASK_STAT} "
                f"WHERE learner_id = ?",
                (self.name,),
            )
            for task_type, stat_key, stat_value, sample_count in cursor.fetchall():
                if task_type not in self._task_stats:
                    self._task_stats[task_type] = {
                        "optimal_tool_budget": 10,
                        "avg_quality_score": 0.5,
                        "avg_completion_rate": 0.5,
                        "sample_count": 0,
                    }
                self._task_stats[task_type][stat_key] = stat_value
                self._task_stats[task_type]["sample_count"] = sample_count
        except Exception as e:
            logger.debug(f"RL: Could not load task stats: {e}")

        if self._q_values:
            logger.info(f"RL: Loaded {len(self._q_values)} mode transition states from database")

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record mode transition outcome and update Q-values.

        Expected metadata:
        - from_mode: Mode before transition
        - to_mode: Mode after transition
        - state_key: Discretized state representation
        - action_key: Action taken (target_mode:budget_adj)
        - next_state_key: State after transition (for TD update)
        - budget_adjustment: Tool budget adjustment made
        - tool_budget_used: Actual tools used
        - tool_budget_total: Total budget available

        Args:
            outcome: Outcome with transition data
        """
        from_mode = outcome.metadata.get("from_mode")
        to_mode = outcome.metadata.get("to_mode")
        state_key = outcome.metadata.get("state_key")
        action_key = outcome.metadata.get("action_key")

        if not all([from_mode, to_mode, state_key, action_key]):
            logger.debug("RL: mode_transition outcome missing required fields, skipping")
            return

        task_type = outcome.task_type or "default"
        next_state_key = outcome.metadata.get("next_state_key")

        # Compute reward
        reward = self._compute_reward(outcome)

        # Get current Q-value
        old_q = self._get_q_value(state_key, action_key)

        # Get max Q-value for next state (TD target)
        max_next_q = 0.0
        if next_state_key:
            next_actions = self._get_all_actions(next_state_key)
            if next_actions:
                max_next_q = max(next_actions.values())

        # Q-learning update: Q(s,a) = Q(s,a) + α(r + γ*max(Q(s',a')) - Q(s,a))
        td_target = reward + self.discount_factor * max_next_q
        new_q = old_q + self.learning_rate * (td_target - old_q)

        # Update caches
        if state_key not in self._q_values:
            self._q_values[state_key] = {}
            self._visit_counts[state_key] = {}

        self._q_values[state_key][action_key] = new_q
        self._visit_counts[state_key][action_key] = (
            self._visit_counts[state_key].get(action_key, 0) + 1
        )
        self._total_transitions += 1

        # Update task stats
        self._update_task_stats(outcome)

        # Persist to database
        self._save_to_db(state_key, action_key, from_mode, to_mode, task_type, outcome, reward)

        logger.debug(
            f"RL: Mode transition {from_mode}→{to_mode} Q-value: {old_q:.3f} → {new_q:.3f} "
            f"(reward={reward:.3f}, task={task_type})"
        )

    def _save_to_db(
        self,
        state_key: str,
        action_key: str,
        from_mode: str,
        to_mode: str,
        task_type: str,
        outcome: RLOutcome,
        reward: float,
    ) -> None:
        """Save Q-values and outcome to database."""
        cursor = self.db.cursor()
        timestamp = datetime.now().isoformat()

        import json as _json

        # Save Q-value to unified table
        cursor.execute(
            f"INSERT OR REPLACE INTO {Tables.RL_Q_VALUE} "
            f"(learner_id, state_key, action_key, q_value, visit_count, last_updated) "
            f"VALUES (?, ?, ?, ?, ?, ?)",
            (
                self.name,
                state_key,
                action_key,
                self._q_values[state_key][action_key],
                self._visit_counts[state_key][action_key],
                timestamp,
            ),
        )

        # Save transition history to unified table
        meta = _json.dumps(
            {
                "success": 1 if outcome.success else 0,
                "quality_score": outcome.quality_score,
                "task_type": task_type,
            }
        )
        cursor.execute(
            f"INSERT INTO {Tables.RL_TRANSITION} "
            f"(learner_id, from_state, to_state, action, reward, metadata, created_at) "
            f"VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self.name, from_mode, to_mode, action_key, reward, meta, timestamp),
        )

        self.db.commit()

    def _update_task_stats(self, outcome: RLOutcome) -> None:
        """Update task-type statistics."""
        task_type = outcome.task_type or "default"
        tool_budget_used = outcome.metadata.get("tool_budget_used", 0)
        _tool_budget_total = outcome.metadata.get("tool_budget_total", 10)  # noqa: F841

        if task_type not in self._task_stats:
            self._task_stats[task_type] = {
                "optimal_tool_budget": 10,
                "avg_quality_score": 0.5,
                "avg_completion_rate": 0.5,
                "sample_count": 0,
            }

        stats = self._task_stats[task_type]
        count = stats["sample_count"] + 1
        alpha = min(0.1, 1.0 / count)  # Decaying learning rate

        # Update averages
        stats["avg_quality_score"] = (1 - alpha) * stats[
            "avg_quality_score"
        ] + alpha * outcome.quality_score
        stats["avg_completion_rate"] = (1 - alpha) * stats["avg_completion_rate"] + alpha * (
            1.0 if outcome.success else 0.0
        )
        stats["sample_count"] = count

        # Update optimal budget based on outcome
        if outcome.success and outcome.quality_score >= 0.7:
            # Success: move budget toward actual usage
            target = max(tool_budget_used + 2, stats["optimal_tool_budget"] - 1)
            stats["optimal_tool_budget"] = int(
                (1 - alpha) * stats["optimal_tool_budget"] + alpha * target
            )
        elif not outcome.success:
            # Failure: increase budget
            stats["optimal_tool_budget"] = min(stats["optimal_tool_budget"] + 2, 50)

        # Save task stats to unified rl_task_stat table
        cursor = self.db.cursor()
        now = datetime.now().isoformat()
        for stat_key in (
            "optimal_tool_budget",
            "avg_quality_score",
            "avg_completion_rate",
        ):
            cursor.execute(
                f"INSERT OR REPLACE INTO {Tables.RL_TASK_STAT} "
                f"(learner_id, task_type, stat_key, stat_value, sample_count, updated_at) "
                f"VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.name,
                    task_type,
                    stat_key,
                    stats[stat_key],
                    stats["sample_count"],
                    now,
                ),
            )
        self.db.commit()

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get mode transition recommendation for given context.

        Note: The 'provider' parameter is overloaded to contain the state_key
        for API compatibility with BaseLearner.

        Args:
            provider: State key (overloaded parameter)
            model: Not used
            task_type: Task type for context

        Returns:
            Recommendation with best action and Q-value, or None
        """
        state_key = provider  # Overloaded

        # Get all actions for this state
        actions = self._get_all_actions(state_key)

        if not actions:
            return RLRecommendation(
                value="explore",  # Default to explore mode
                confidence=0.3,
                reason="No learned data for this state",
                sample_size=0,
                is_baseline=True,
            )

        # Get total visits for this state
        total_visits = sum(self._visit_counts.get(state_key, {}).values())

        # Epsilon-greedy: explore or exploit
        import random

        if self._rng.random() < self.epsilon:
            # Exploration: random action
            action = self._rng.choice(list(actions.keys()))
            _q_value = actions[action]  # noqa: F841
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

        # Compute confidence based on visit count
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

    def get_best_action(
        self,
        current_mode: str,
        task_type: str,
        tool_ratio: str,
        quality_bucket: str,
    ) -> Tuple[str, float, float]:
        """Get best action for given state components.

        Convenience method that builds the state key and returns the best action.

        Args:
            current_mode: Current agent mode
            task_type: Task type
            tool_ratio: Tool usage ratio bucket (low, mid_low, mid_high, high)
            quality_bucket: Quality score bucket (poor, fair, good, excellent)

        Returns:
            Tuple of (best_action, q_value, confidence)
        """
        state_key = f"{current_mode}:{task_type}:{tool_ratio}:{quality_bucket}"

        rec = self.get_recommendation(state_key, "", task_type)
        if rec is None:
            return ("explore:0", 0.0, 0.3)

        return (
            rec.value,
            rec.value if isinstance(rec.value, float) else 0.0,
            rec.confidence,
        )

    def get_optimal_budget(self, task_type: str) -> int:
        """Get learned optimal tool budget for a task type.

        Args:
            task_type: Task type

        Returns:
            Optimal tool budget (learned or default)
        """
        if task_type in self._task_stats:
            return self._task_stats[task_type]["optimal_tool_budget"]

        # Default budgets by task type
        defaults = {
            "code_generation": 8,
            "create_simple": 5,
            "create": 10,
            "edit": 10,
            "search": 8,
            "action": 15,
            "analysis_deep": 20,
            "analyze": 15,
            "design": 25,
            "general": 15,
        }
        return defaults.get(task_type, 10)

    def _get_q_value(self, state_key: str, action_key: str) -> float:
        """Get Q-value for a state-action pair."""
        return self._q_values.get(state_key, {}).get(action_key, self.DEFAULT_Q_VALUE)

    def _get_all_actions(self, state_key: str) -> Dict[str, float]:
        """Get all Q-values for a state."""
        return self._q_values.get(state_key, {})

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward from transition outcome.

        Reward based on:
        - Task completion (40%): Did the task succeed?
        - Quality score (30%): Quality of the result
        - Efficiency (20%): Tool/budget efficiency
        - Transition validity (10%): Was transition valid?

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (-1.0 to 1.0)
        """
        reward = 0.0

        # Task completion (40% weight)
        task_completed = outcome.metadata.get("task_completed", outcome.success)
        reward += 0.4 * (1.0 if task_completed else -0.5)

        # Quality score (30% weight)
        quality = outcome.quality_score
        reward += 0.3 * (2 * quality - 1)  # Map [0,1] to [-1,1]

        # Efficiency (20% weight)
        tool_budget_used = outcome.metadata.get("tool_budget_used", 0)
        tool_budget_total = outcome.metadata.get("tool_budget_total", 10)
        if tool_budget_total > 0:
            efficiency = 1.0 - (tool_budget_used / tool_budget_total)
            if task_completed:
                reward += 0.2 * efficiency  # Reward efficiency on success
            else:
                reward -= 0.2 * 0.5  # Penalize failure regardless of efficiency

        # Transition validity (10% weight)
        from_mode = outcome.metadata.get("from_mode", "explore")
        to_mode = outcome.metadata.get("to_mode", "explore")
        try:
            from_mode_enum = AgentMode(from_mode)
            to_mode_enum = AgentMode(to_mode)
            valid = to_mode_enum in self.VALID_TRANSITIONS.get(from_mode_enum, [])
            reward += 0.1 * (1.0 if valid else -1.0)
        except (ValueError, KeyError):
            pass  # Invalid modes, no reward adjustment

        return max(-1.0, min(1.0, reward))

    def get_task_stats(self, task_type: str) -> Dict[str, Any]:
        """Get statistics for a task type.

        Args:
            task_type: Task type

        Returns:
            Dictionary with task statistics
        """
        if task_type in self._task_stats:
            stats = self._task_stats[task_type].copy()
            stats["task_type"] = task_type
            return stats

        return {
            "task_type": task_type,
            "optimal_tool_budget": 10,
            "avg_quality_score": 0.5,
            "avg_completion_rate": 0.5,
            "sample_count": 0,
        }

    def export_metrics(self) -> Dict[str, Any]:
        """Export learner metrics for monitoring.

        Returns:
            Dictionary with learner stats
        """
        # Count states and actions
        total_state_action_pairs = sum(len(actions) for actions in self._q_values.values())

        return {
            "learner": self.name,
            "total_states": len(self._q_values),
            "total_state_action_pairs": total_state_action_pairs,
            "total_transitions": self._total_transitions,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "task_stats": self._task_stats,
        }
