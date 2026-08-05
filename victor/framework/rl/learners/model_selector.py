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

"""RL learner for adaptive model/provider selection using Q-learning.

This learner uses Q-learning with epsilon-greedy exploration to select optimal
LLM models/providers based on historical session performance.

Strategy:
- Q-learning with epsilon-greedy exploration
- Multiple selection strategies: EPSILON_GREEDY, UCB, EXPLOIT_ONLY
- Task-type aware Q-values (different optimal models per task type)
- Reward based on: success, latency, token throughput, tool usage
- Epsilon decay schedule for exploration → exploitation transition
- Warm-up phase with higher learning rate
- Range: [0.0, 1.0] for Q-values

Migrated from: victor/agent/rl_model_selector.py
"""

from victor.core.json_utils import json_dumps, json_loads
from json import JSONDecodeError
import logging
import math
import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.core.schema import Tables
from victor.framework.rl.migration import RLTableMigrator

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Model selection strategy."""

    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"
    THOMPSON_SAMPLING = "thompson"
    EXPLOIT_ONLY = "exploit"


class ModelSelectorLearner(BaseLearner):
    """Learn optimal provider/model selection using Q-learning.

    Uses Q-learning with epsilon-greedy exploration to find optimal
    providers/models for each task type based on historical performance.

    Attributes:
        name: Always "model_selector"
        db: SQLite database connection
        learning_rate: Q-value update rate (alpha)
        epsilon: Exploration rate
        epsilon_decay: Decay rate per selection
        epsilon_min: Minimum exploration rate
        discount_factor: Future reward discount (gamma)
        ucb_c: UCB exploration constant
        strategy: Selection strategy
    """

    # Mock/test providers to exclude from Q-table
    MOCK_PROVIDERS = frozenset({"mock", "mock_provider", "dummy", "dummy-stream", "test"})

    # Warm-up threshold: use higher learning rate until this many real selections
    WARMUP_THRESHOLD = 100

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.1,
        provider_adapter: Optional[Any] = None,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        discount_factor: float = 0.95,
        ucb_c: float = 2.0,
        strategy: SelectionStrategy = SelectionStrategy.EPSILON_GREEDY,
    ):
        """Initialize model selector learner.

        Args:
            name: Learner name
            db_connection: SQLite database connection
            learning_rate: Q-learning alpha parameter
            provider_adapter: Optional provider adapter
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per selection
            epsilon_min: Minimum epsilon value
            discount_factor: Q-learning gamma parameter
            ucb_c: UCB exploration constant
            strategy: Selection strategy to use
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor
        self.ucb_c = ucb_c
        self.strategy = strategy

        # In-memory caches
        self._q_table: Dict[str, float] = {}
        self._selection_counts: Dict[str, int] = {}
        self._total_selections: int = 0
        self._q_table_by_task: Dict[str, Dict[str, float]] = {}
        self._task_selection_counts: Dict[str, Dict[str, int]] = {}

        # Priority 4 Phase 2: per-decision-type confidence threshold tracking
        # Must be initialized before _load_state() so that _load_state can populate it
        self._threshold_observations: Dict[str, list] = {}

        # Load state from database
        self._load_state()
        self.load_threshold_observations()

    def _ensure_tables(self) -> None:
        """Migrate legacy per-learner tables to unified RL tables."""
        RLTableMigrator(self.db).run_if_needed(self.name, RLTableMigrator.migrate_model_selector)

    def _load_state(self) -> None:
        """Load state from database."""
        cursor = self.db.cursor()

        try:
            # Load global Q-values (action_key='select') and task-specific Q-values
            cursor.execute(
                f"SELECT state_key, action_key, q_value, visit_count FROM {Tables.RL_Q_VALUE}"
                f" WHERE learner_id = ?",
                (self.name,),
            )
            for row in cursor.fetchall():
                row_dict = dict(row)
                provider = row_dict["state_key"]
                action_key = row_dict["action_key"]
                if action_key == "select":
                    self._q_table[provider] = row_dict["q_value"]
                    self._selection_counts[provider] = row_dict["visit_count"]
                    self._total_selections += row_dict["visit_count"]
                else:
                    # task-specific: action_key is the task_type
                    task_type = action_key
                    if provider not in self._q_table_by_task:
                        self._q_table_by_task[provider] = {}
                        self._task_selection_counts[provider] = {}
                    self._q_table_by_task[provider][task_type] = row_dict["q_value"]
                    self._task_selection_counts[provider][task_type] = row_dict["visit_count"]

            # Load epsilon, total_selections, and threshold observations from rl_param
            cursor.execute(
                f"SELECT param_key, param_value, value_text FROM {Tables.RL_PARAM}"
                f" WHERE learner_id = ?",
                (self.name,),
            )
            for row in cursor.fetchall():
                row_dict = dict(row)
                key = row_dict["param_key"]
                if key == "epsilon" and row_dict["param_value"] is not None:
                    self.epsilon = row_dict["param_value"]
                elif key == "total_selections" and row_dict["param_value"] is not None:
                    self._total_selections = int(row_dict["param_value"])
                elif key.startswith("threshold:") and row_dict["value_text"]:
                    decision_type = key[len("threshold:") :]
                    try:
                        self._threshold_observations[decision_type] = json_loads(
                            row_dict["value_text"]
                        )
                    except (JSONDecodeError, TypeError):
                        pass

        except Exception as e:
            logger.debug(f"RL: Could not load model_selector state: {e}")

        if self._q_table:
            logger.info(f"RL: Loaded {len(self._q_table)} provider Q-values from database")

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record model selection outcome and update Q-values.

        Expected metadata:
        - latency_seconds: Session duration
        - token_count: Total tokens
        - tool_calls_made: Number of tool calls
        - user_satisfaction: Optional user feedback (0-1)

        Args:
            outcome: Outcome with model selection data
        """
        provider = outcome.provider

        # Skip mock providers
        if provider in self.MOCK_PROVIDERS:
            logger.debug(f"RL: Skipping mock provider '{provider}'")
            return

        # Compute reward
        reward = self._compute_reward(outcome)

        # Get old Q-value
        old_q = self._q_table.get(provider, 0.5)

        # Compute effective learning rate (higher during warm-up)
        effective_lr = self._get_effective_learning_rate(provider)

        # Update Q-value: Q(s,a) ← Q(s,a) + α[r - Q(s,a)]
        new_q = max(0.0, min(1.0, old_q + effective_lr * (reward - old_q)))

        # Update global Q-table
        self._q_table[provider] = new_q
        self._selection_counts[provider] = self._selection_counts.get(provider, 0) + 1
        self._total_selections += 1

        # Update task-specific Q-table
        task_type = outcome.task_type
        if task_type:
            if provider not in self._q_table_by_task:
                self._q_table_by_task[provider] = {}
                self._task_selection_counts[provider] = {}

            old_task_q = self._q_table_by_task[provider].get(task_type, 0.5)
            new_task_q = max(0.0, min(1.0, old_task_q + effective_lr * (reward - old_task_q)))

            self._q_table_by_task[provider][task_type] = new_task_q
            self._task_selection_counts[provider][task_type] = (
                self._task_selection_counts[provider].get(task_type, 0) + 1
            )

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Persist to database
        self._save_to_db(provider, task_type, outcome.timestamp)

        logger.debug(
            f"RL: Updated Q({provider}): {old_q:.3f} → {new_q:.3f} "
            f"(reward={reward:.3f}, count={self._selection_counts[provider]})"
        )

    def _save_to_db(self, provider: str, task_type: Optional[str], timestamp: str) -> None:
        """Save Q-values and state to database."""
        cursor = self.db.cursor()

        # Save global Q-value
        cursor.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.RL_Q_VALUE}
            (learner_id, state_key, action_key, q_value, visit_count, last_updated)
            VALUES (?, ?, 'select', ?, ?, ?)
            """,
            (
                self.name,
                provider,
                self._q_table[provider],
                self._selection_counts[provider],
                timestamp,
            ),
        )

        # Save task-specific Q-value
        if task_type and provider in self._q_table_by_task:
            if task_type in self._q_table_by_task[provider]:
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO {Tables.RL_Q_VALUE}
                    (learner_id, state_key, action_key, q_value, visit_count, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.name,
                        provider,
                        task_type,
                        self._q_table_by_task[provider][task_type],
                        self._task_selection_counts[provider][task_type],
                        timestamp,
                    ),
                )

        # Save epsilon and total_selections
        cursor.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.RL_PARAM}
            (learner_id, param_key, param_value, updated_at)
            VALUES (?, 'epsilon', ?, ?)
            """,
            (self.name, self.epsilon, timestamp),
        )
        cursor.execute(
            f"""
            INSERT OR REPLACE INTO {Tables.RL_PARAM}
            (learner_id, param_key, param_value, updated_at)
            VALUES (?, 'total_selections', ?, ?)
            """,
            (self.name, float(self._total_selections), timestamp),
        )

        self.db.commit()

    def _get_effective_learning_rate(self, provider: str) -> float:
        """Get effective learning rate based on warm-up phase."""
        real_selections = sum(
            v for k, v in self._selection_counts.items() if k not in self.MOCK_PROVIDERS
        )
        provider_count = self._selection_counts.get(provider, 0)

        # Higher learning rate during warm-up
        if real_selections < self.WARMUP_THRESHOLD or provider_count < 10:
            return min(0.5, self.learning_rate * 3)
        return self.learning_rate

    def _get_q_value(self, provider: str, task_type: Optional[str] = None) -> float:
        """Get Q-value for a provider, optionally task-specific."""
        global_q = self._q_table.get(provider, 0.5)

        if not task_type or provider not in self._q_table_by_task:
            return global_q

        task_q = self._q_table_by_task[provider].get(task_type)
        # Blend: 70% task-specific, 30% global
        return 0.7 * task_q + 0.3 * global_q if task_q is not None else global_q

    def get_recommendation(
        self, provider: str, model: str, task_type: str
    ) -> Optional[RLRecommendation]:
        """Get recommended provider for selection.

        Note: This method signature matches BaseLearner but for model selector,
        we use it differently - provider param is used to pass available providers list.

        Args:
            provider: JSON string of available providers list (e.g., '["anthropic", "openai"]')
            model: Not used (kept for signature compatibility)
            task_type: Task type for context-aware selection

        Returns:
            Recommendation with selected provider and confidence, or None
        """
        # Parse available providers from provider param
        try:
            available_providers = json_loads(provider) if isinstance(provider, str) else [provider]
        except (JSONDecodeError, TypeError):
            available_providers = [provider]

        if not available_providers:
            return None

        # Ensure all providers have Q-values
        for p in available_providers:
            if p not in self._q_table:
                self._q_table[p] = 0.5  # Optimistic initial value
                self._selection_counts[p] = 0

        # Select provider using configured strategy
        if self.strategy == SelectionStrategy.EPSILON_GREEDY:
            selected, reason = self._select_epsilon_greedy(available_providers, task_type)
        elif self.strategy == SelectionStrategy.UCB:
            selected, reason = self._select_ucb(available_providers, task_type)
        elif self.strategy == SelectionStrategy.EXPLOIT_ONLY:
            selected, reason = self._select_exploit(available_providers, task_type)
        else:
            selected, reason = self._select_epsilon_greedy(available_providers, task_type)

        q_value = self._get_q_value(selected, task_type)
        count = self._selection_counts.get(selected, 0)
        confidence = min(0.95, 0.5 + 0.45 * (1 - math.exp(-count / 20)))

        return RLRecommendation(
            value=selected,
            confidence=confidence,
            reason=f"{reason} (Q={q_value:.3f}, n={count})",
            sample_size=count,
            is_baseline=False,
        )

    def _select_epsilon_greedy(
        self, providers: List[str], task_type: Optional[str] = None
    ) -> Tuple[str, str]:
        """Select using epsilon-greedy strategy."""
        # Exploration
        if self._rng.random() < self.epsilon:
            selected = self._rng.choice(providers)
            reason = f"Exploration (ε={self.epsilon:.2f})"
            return selected, reason

        # Exploitation
        return self._select_exploit(providers, task_type)

    def _select_exploit(
        self, providers: List[str], task_type: Optional[str] = None
    ) -> Tuple[str, str]:
        """Select provider with highest Q-value."""
        best_provider = max(providers, key=lambda p: self._get_q_value(p, task_type))
        reason = "Best Q-value"
        if task_type:
            reason += f" [task={task_type}]"
        return best_provider, reason

    def _select_ucb(self, providers: List[str], task_type: Optional[str] = None) -> Tuple[str, str]:
        """Select using Upper Confidence Bound (UCB) strategy."""
        if self._total_selections == 0:
            return self._rng.choice(providers), "No history, random selection"

        log_total = math.log(self._total_selections + 1)
        ucb_scores = {}

        for p in providers:
            q_value = self._get_q_value(p, task_type)
            count = self._selection_counts.get(p, 0)
            ucb_scores[p] = (
                float("inf") if count == 0 else q_value + self.ucb_c * math.sqrt(log_total / count)
            )

        best_provider = max(providers, key=lambda p: ucb_scores[p])
        ucb_score = ucb_scores[best_provider]

        if ucb_score == float("inf"):
            reason = "Unexplored (UCB=∞)"
        else:
            reason = f"UCB={ucb_score:.3f}"

        return best_provider, reason

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward signal from outcome.

        Reward based on:
        - Success: +1.0 / -0.5
        - Latency penalty (baseline 30s)
        - Tool usage bonus
        - Token throughput bonus (baseline 5 tok/s)
        - User satisfaction override (if available)

        Args:
            outcome: Outcome to compute reward for

        Returns:
            Reward value (-1.0 to 1.0)
        """
        reward = 1.0 if outcome.success else -0.5

        # Latency penalty
        latency = outcome.metadata.get("latency_seconds", 0)
        if latency > 30:
            reward -= min(0.1 * (latency - 30) / 60, 0.5)

        # Tool usage bonus
        tool_calls = outcome.metadata.get("tool_calls_made", 0)
        if tool_calls > 0:
            reward += 0.1

        # Token throughput bonus
        token_count = outcome.metadata.get("token_count", 0)
        if latency > 0 and token_count > 0:
            tok_per_sec = token_count / latency
            if tok_per_sec > 5:
                reward += min(0.05 * (tok_per_sec - 5) / 10, 0.2)

        # User satisfaction override
        user_satisfaction = outcome.metadata.get("user_satisfaction")
        if user_satisfaction is not None:
            reward = 0.7 * reward + 0.3 * (user_satisfaction * 2 - 0.5)

        return max(-1.0, min(1.0, reward))

    def get_provider_rankings(self) -> List[Dict[str, Any]]:
        """Get all providers ranked by Q-value.

        Returns:
            List of provider stats sorted by Q-value descending
        """
        rankings = []
        log_total = math.log(self._total_selections + 1) if self._total_selections > 0 else 0

        for provider, q_value in self._q_table.items():
            count = self._selection_counts.get(provider, 0)

            # Compute UCB score
            if count == 0:
                ucb_score = float("inf")
            else:
                ucb_score = q_value + self.ucb_c * math.sqrt(log_total / count)

            # Confidence
            confidence = min(0.95, 0.5 + 0.45 * (1 - math.exp(-count / 20)))

            rankings.append(
                {
                    "provider": provider,
                    "q_value": q_value,
                    "session_count": count,
                    "ucb_score": ucb_score if ucb_score != float("inf") else None,
                    "confidence": confidence,
                }
            )

        return sorted(rankings, key=lambda x: x["q_value"], reverse=True)

    def get_exploration_rate(self) -> float:
        """Get the current exploration rate (epsilon).

        Returns:
            Current epsilon value for exploration
        """
        return self.epsilon

    def get_strategy(self) -> SelectionStrategy:
        """Get the current selection strategy.

        Returns:
            Current selection strategy enum
        """
        return self.strategy

    # ------------------------------------------------------------------
    # Priority 4 Phase 2: Learned confidence thresholds for decision routing
    # ------------------------------------------------------------------

    def learn_confidence_threshold(
        self,
        decision_type: str,
        heuristic_confidence: float,
        used_llm: bool,
        success: bool,
    ) -> None:
        """Record a single decision observation to refine per-type thresholds.

        Tracks which confidence levels led to correct routing (heuristic vs LLM)
        so get_optimal_threshold() can return data-driven thresholds instead of
        static ones.

        Args:
            decision_type: Decision category (e.g. "task_type", "tool_necessity")
            heuristic_confidence: Confidence score the heuristic produced (0–1)
            used_llm: Whether the LLM path was actually taken
            success: Whether the decision outcome was successful
        """
        if not hasattr(self, "_threshold_observations"):
            self._threshold_observations: Dict[str, list] = {}

        if decision_type not in self._threshold_observations:
            self._threshold_observations[decision_type] = []

        self._threshold_observations[decision_type].append(
            {
                "confidence": heuristic_confidence,
                "used_llm": used_llm,
                "success": success,
            }
        )

        # Keep last 200 observations per type to bound memory
        self._threshold_observations[decision_type] = self._threshold_observations[decision_type][
            -200:
        ]

        self._persist_threshold(decision_type)

    def get_optimal_threshold(self, decision_type: str) -> Optional[float]:
        """Return the learned optimal confidence threshold for a decision type.

        Finds the threshold that maximises accuracy: heuristic correct when
        confidence ≥ threshold, LLM correct when confidence < threshold.

        Args:
            decision_type: Decision category to query

        Returns:
            Optimal threshold float, or None if insufficient data (<10 observations)
        """
        if not hasattr(self, "_threshold_observations"):
            return None

        observations = self._threshold_observations.get(decision_type, [])
        if len(observations) < 10:
            return None

        # Grid search over candidate thresholds
        best_threshold = 0.5
        best_accuracy = 0.0

        for candidate in [i / 20 for i in range(1, 20)]:  # 0.05 to 0.95
            correct = 0
            for obs in observations:
                heuristic_correct = obs["confidence"] >= candidate and obs["success"]
                llm_correct = obs["confidence"] < candidate and obs["success"]
                if heuristic_correct or llm_correct:
                    correct += 1

            accuracy = correct / len(observations)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = candidate

        return best_threshold

    def _persist_threshold(self, decision_type: str) -> None:
        """Persist threshold observations to DB for cross-session continuity."""
        observations = getattr(self, "_threshold_observations", {}).get(decision_type, [])
        if not observations:
            return
        try:
            cursor = self.db.cursor()
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {Tables.RL_PARAM}
                (learner_id, param_key, value_text, updated_at)
                VALUES (?, ?, ?, datetime('now'))
                """,
                (
                    self.name,
                    f"threshold:{decision_type}",
                    json_dumps(observations[-50:]),
                ),
            )
            self.db.commit()
        except Exception as e:
            logger.debug("model_selector: threshold persist failed: %s", e)

    def load_threshold_observations(self) -> None:
        """Load persisted threshold observations from DB on startup."""
        if not hasattr(self, "_threshold_observations"):
            self._threshold_observations = {}
        # Loaded during _load_state(); this is a no-op after migration to unified tables.
        # Kept for API compatibility with callers that invoke it directly.
