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

"""Hierarchical RL policy for multi-step tasks.

This module implements a two-level hierarchical policy:

High-Level Policy (Meta-Controller):
- Selects which option (skill) to execute
- Operates on abstract goals (explore → plan → build → review)
- Learns to choose options based on task context

Low-Level Policy (Controller):
- Executes primitive actions within an option
- Handled by the Option's internal policy
- Terminates when option completes

Architecture:
┌─────────────────────────────────────┐
│      HIGH-LEVEL POLICY              │
│  (Strategy: explore→plan→build)     │
│  Actions: Set subgoal, switch mode  │
│  Reward: Task completion            │
└─────────────────┬───────────────────┘
                  │ subgoal (option selection)
                  ▼
┌─────────────────────────────────────┐
│      LOW-LEVEL POLICY               │
│  (Tactics: tool selection)          │
│  Actions: Select tool, parameters   │
│  Reward: Subgoal progress           │
└─────────────────────────────────────┘

Benefits:
- Better planning for complex, multi-step tasks
- Reusable low-level skills across different strategies
- More interpretable decision making

Sprint 5-6: Advanced Patterns - Hierarchical RL
"""

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.framework.rl.option_framework import (
    OptionRegistry,
    OptionResult,
    OptionState,
)

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalState:
    """State for hierarchical policy decision making.

    Attributes:
        task_type: Type of task (analysis, action, create, etc.)
        task_complexity: Estimated complexity (0-1)
        current_stage: Current workflow stage
        context_gathered: Amount of context (0-1 normalized)
        tools_used_count: Number of tools used
        iterations: Total iterations
        last_option: Last executed option name
        success_rate: Recent success rate
    """

    task_type: str = "unknown"
    task_complexity: float = 0.5
    current_stage: str = "initial"
    context_gathered: float = 0.0
    tools_used_count: int = 0
    iterations: int = 0
    last_option: Optional[str] = None
    success_rate: float = 0.5

    def to_key(self) -> str:
        """Convert to hashable key for Q-table."""
        complexity_bucket = (
            "low"
            if self.task_complexity < 0.4
            else ("medium" if self.task_complexity < 0.7 else "high")
        )
        context_bucket = (
            "low"
            if self.context_gathered < 0.3
            else ("medium" if self.context_gathered < 0.7 else "high")
        )
        return f"{self.task_type}:{complexity_bucket}:{self.current_stage}:{context_bucket}"


@dataclass
class PolicyDecision:
    """Decision from hierarchical policy.

    Attributes:
        option_name: Name of option to execute (or None for primitive action)
        primitive_action: Direct action if no option selected
        confidence: Confidence in the decision (0-1)
        reason: Explanation for the decision
        is_option: Whether this is an option or primitive action
    """

    option_name: Optional[str] = None
    primitive_action: Optional[str] = None
    confidence: float = 0.5
    reason: str = ""
    is_option: bool = True


class HierarchicalPolicy(BaseLearner):
    """Two-level hierarchical policy using options framework.

    The high-level policy selects which option to execute based on
    the current task context. Options are temporal abstractions that
    encapsulate multi-step behaviors (skills).

    Q-learning is used to learn the value of selecting each option
    in different states.

    Usage:
        policy = HierarchicalPolicy()

        # Get decision
        state = HierarchicalState(task_type="action", ...)
        decision = policy.get_decision(state)

        # Execute option or primitive action
        if decision.is_option:
            policy.start_option(decision.option_name, option_state)
            while policy.has_active_option():
                action = policy.step_option(option_state, reward)
                # Execute action...
            result = policy.get_option_result()
        else:
            # Execute primitive action
            pass

        # Record outcome
        policy.record_option_outcome(option_name, success, reward)
    """

    # Q-learning parameters
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_DISCOUNT_FACTOR = 0.95
    DEFAULT_EPSILON = 0.15  # Exploration rate

    # Option names
    OPTION_EXPLORE = "explore_codebase"
    OPTION_IMPLEMENT = "implement_feature"
    OPTION_DEBUG = "debug_issue"
    OPTION_REVIEW = "review_work"

    def __init__(
        self,
        name: str = "hierarchical_policy",
        db_connection: Optional[Any] = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        discount_factor: float = DEFAULT_DISCOUNT_FACTOR,
        epsilon: float = DEFAULT_EPSILON,
    ):
        """Initialize hierarchical policy.

        Args:
            name: Learner name
            db_connection: Optional database connection
            learning_rate: Q-learning alpha
            discount_factor: Q-learning gamma
            epsilon: Exploration rate
        """
        super().__init__(name, db_connection)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Q-table: state_key -> option_name -> Q-value
        self._q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(lambda: 0.5))

        # Visit counts for UCB exploration
        self._visit_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Option registry
        self._option_registry = OptionRegistry()

        # Current option tracking
        self._current_option_name: Optional[str] = None
        self._current_option_start_state: Optional[HierarchicalState] = None
        self._option_cumulative_reward: float = 0.0

        # Statistics
        self._total_decisions = 0
        self._option_completions: Dict[str, int] = defaultdict(int)
        self._option_success_rate: Dict[str, List[bool]] = defaultdict(list)

        # Load from database if available
        if db_connection:
            self._load_from_db()

    @property
    def option_names(self) -> List[str]:
        """Get available option names."""
        return [
            self.OPTION_EXPLORE,
            self.OPTION_IMPLEMENT,
            self.OPTION_DEBUG,
            self.OPTION_REVIEW,
        ]

    def get_decision(self, state: HierarchicalState) -> PolicyDecision:
        """Get decision from high-level policy.

        Uses epsilon-greedy Q-learning to select an option.

        Args:
            state: Current hierarchical state

        Returns:
            PolicyDecision with selected option or action
        """
        self._total_decisions += 1
        state_key = state.to_key()

        # Get available options based on current state
        option_state = self._to_option_state(state)
        available_options = self._option_registry.get_available_options(option_state)
        available_names = [opt.name for opt in available_options]

        if not available_names:
            # No options available, return primitive action
            return PolicyDecision(
                primitive_action="continue",
                confidence=0.5,
                reason="No options available",
                is_option=False,
            )

        # Epsilon-greedy selection
        if self._rng.random() < self.epsilon:
            # Explore: random selection
            selected = self._rng.choice(available_names)
            reason = "Exploration (random)"
        else:
            # Exploit: select best Q-value
            q_values = {name: self._q_table[state_key][name] for name in available_names}
            selected = max(q_values, key=q_values.get)
            reason = f"Exploitation (Q={q_values[selected]:.2f})"

        # Update visit count
        self._visit_counts[state_key][selected] += 1

        # Calculate confidence based on visit count
        visits = self._visit_counts[state_key][selected]
        confidence = min(0.95, 0.5 + 0.05 * visits)

        return PolicyDecision(
            option_name=selected,
            confidence=confidence,
            reason=reason,
            is_option=True,
        )

    def start_option(self, option_name: str, state: HierarchicalState) -> bool:
        """Start executing an option.

        Args:
            option_name: Name of option to start
            state: Current state

        Returns:
            True if option started successfully
        """
        option_state = self._to_option_state(state)

        if self._option_registry.start_option(option_name, option_state):
            self._current_option_name = option_name
            self._current_option_start_state = state
            self._option_cumulative_reward = 0.0
            logger.debug(f"HierarchicalPolicy: Started option '{option_name}'")
            return True

        return False

    def step_option(self, state: HierarchicalState, reward: float = 0.0) -> Optional[str]:
        """Step the current option.

        Args:
            state: Current state
            reward: Reward from previous action

        Returns:
            Next action or None if option terminated
        """
        if not self._current_option_name:
            return None

        self._option_cumulative_reward += reward
        option_state = self._to_option_state(state)

        return self._option_registry.step_active_option(option_state, reward)

    def has_active_option(self) -> bool:
        """Check if there's an active option."""
        return self._option_registry.active_option is not None

    def get_option_result(self) -> Optional[OptionResult]:
        """Get result of completed option."""
        return self._option_registry.terminate_active_option(success=True)

    def terminate_current_option(self, success: bool = True) -> Optional[OptionResult]:
        """Terminate current option early.

        Args:
            success: Whether termination is due to success

        Returns:
            OptionResult or None
        """
        result = self._option_registry.terminate_active_option(success)

        if result and self._current_option_name:
            # Update Q-value
            self._update_q_value(
                self._current_option_start_state,
                self._current_option_name,
                self._option_cumulative_reward,
                success,
            )

            self._current_option_name = None
            self._current_option_start_state = None

        return result

    def record_option_outcome(
        self,
        option_name: str,
        success: bool,
        reward: float,
        start_state: Optional[HierarchicalState] = None,
    ) -> None:
        """Record outcome for option execution.

        Args:
            option_name: Name of completed option
            success: Whether option succeeded
            reward: Total reward from option
            start_state: State when option started
        """
        state = start_state or self._current_option_start_state

        if state:
            self._update_q_value(state, option_name, reward, success)

        # Update statistics
        self._option_completions[option_name] += 1
        self._option_success_rate[option_name].append(success)
        # Keep last 100 outcomes
        self._option_success_rate[option_name] = self._option_success_rate[option_name][-100:]

        logger.debug(
            f"HierarchicalPolicy: Recorded outcome for '{option_name}' "
            f"success={success} reward={reward:.2f}"
        )

    def _update_q_value(
        self,
        state: HierarchicalState,
        option_name: str,
        reward: float,
        success: bool,
    ) -> None:
        """Update Q-value using Q-learning update rule.

        Args:
            state: Starting state
            option_name: Option that was executed
            reward: Reward received
            success: Whether successful
        """
        state_key = state.to_key()

        # Current Q-value
        current_q = self._q_table[state_key][option_name]

        # Compute target (simplified - no next state max Q for terminal)
        # In full implementation, would estimate next state value
        if success:
            target = reward + self.discount_factor * 0.5  # Assume moderate future value
        else:
            target = reward * 0.5  # Penalize failure

        # Q-learning update
        new_q = current_q + self.learning_rate * (target - current_q)
        self._q_table[state_key][option_name] = new_q

        logger.debug(f"Q-update: {state_key}:{option_name} {current_q:.3f} -> {new_q:.3f}")

    def _to_option_state(self, state: HierarchicalState) -> OptionState:
        """Convert HierarchicalState to OptionState.

        Args:
            state: Hierarchical state

        Returns:
            OptionState for option framework
        """
        return OptionState(
            current_mode=state.current_stage,
            iterations=state.iterations,
            context_size=int(state.context_gathered * 50000),  # Denormalize
            task_progress=state.context_gathered,  # Approximation
            last_tool_success=state.success_rate > 0.5,
        )

    # =========================================================================
    # BaseLearner interface implementation
    # =========================================================================

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get recommendation from high-level policy.

        Args:
            provider: Provider name
            model: Model name
            task_type: Task type

        Returns:
            Recommendation with suggested option
        """
        # Create state from context
        state = HierarchicalState(
            task_type=task_type,
            task_complexity=0.5,  # Default
            current_stage="initial",
        )

        decision = self.get_decision(state)

        return RLRecommendation(
            value=decision.option_name or decision.primitive_action,
            confidence=decision.confidence,
            reason=decision.reason,
            sample_size=self._total_decisions,
        )

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record outcome for learning.

        Args:
            outcome: Outcome data
        """
        # Create state from outcome
        state = HierarchicalState(
            task_type=outcome.task_type,
            success_rate=1.0 if outcome.success else 0.0,
        )

        # Determine which option was likely used
        option_name = outcome.metadata.get("option_name", self.OPTION_IMPLEMENT)

        self.record_option_outcome(
            option_name=option_name,
            success=outcome.success,
            reward=outcome.quality_score,
            start_state=state,
        )

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward from outcome.

        Args:
            outcome: Outcome data

        Returns:
            Computed reward
        """
        base_reward = outcome.quality_score

        if outcome.success:
            base_reward += 0.2
        else:
            base_reward -= 0.1

        return max(0.0, min(1.0, base_reward))

    def export_metrics(self) -> Dict[str, Any]:
        """Export policy metrics.

        Returns:
            Dictionary with metrics
        """
        # Calculate success rates
        success_rates = {}
        for option_name, outcomes in self._option_success_rate.items():
            if outcomes:
                success_rates[option_name] = sum(outcomes) / len(outcomes)

        return {
            "total_decisions": self._total_decisions,
            "option_completions": dict(self._option_completions),
            "option_success_rates": success_rates,
            "q_table_size": sum(len(v) for v in self._q_table.values()),
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
        }

    def _ensure_tables(self) -> None:
        """Create database tables for persistence."""
        if not self.db:
            return

        cursor = self.db.cursor()

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.name}_q_table (
                state_key TEXT NOT NULL,
                option_name TEXT NOT NULL,
                q_value REAL NOT NULL,
                visit_count INTEGER DEFAULT 0,
                updated_at REAL DEFAULT (julianday('now')),
                PRIMARY KEY (state_key, option_name)
            )
            """)

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.name}_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                option_name TEXT NOT NULL,
                success INTEGER NOT NULL,
                reward REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """)

        self.db.commit()

    def _load_from_db(self) -> None:
        """Load Q-table from database."""
        if not self.db:
            return

        cursor = self.db.cursor()

        try:
            cursor.execute(
                f"SELECT state_key, option_name, q_value, visit_count FROM {self.name}_q_table"
            )
            for row in cursor.fetchall():
                self._q_table[row[0]][row[1]] = row[2]
                self._visit_counts[row[0]][row[1]] = row[3]

            logger.info(
                f"HierarchicalPolicy: Loaded {sum(len(v) for v in self._q_table.values())} Q-values"
            )
        except Exception as e:
            logger.debug(f"HierarchicalPolicy: Could not load from database: {e}")

    def _save_to_db(self) -> None:
        """Save Q-table to database."""
        if not self.db:
            return

        cursor = self.db.cursor()

        for state_key, options in self._q_table.items():
            for option_name, q_value in options.items():
                visit_count = self._visit_counts[state_key][option_name]
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.name}_q_table
                    (state_key, option_name, q_value, visit_count)
                    VALUES (?, ?, ?, ?)
                    """,
                    (state_key, option_name, q_value, visit_count),
                )

        self.db.commit()


# Global singleton
_hierarchical_policy: Optional[HierarchicalPolicy] = None


def get_hierarchical_policy(
    db_connection: Optional[Any] = None,
) -> HierarchicalPolicy:
    """Get global hierarchical policy (lazy init).

    Args:
        db_connection: Optional database connection

    Returns:
        HierarchicalPolicy singleton
    """
    global _hierarchical_policy
    if _hierarchical_policy is None:
        _hierarchical_policy = HierarchicalPolicy(db_connection=db_connection)
    return _hierarchical_policy
