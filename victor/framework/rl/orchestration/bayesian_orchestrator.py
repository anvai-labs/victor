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

"""Bayesian orchestration service for Bayes-consistent orchestration.

This service integrates observation models, reliability weights, and VoI
computation to provide Bayes-consistent orchestration decisions.

Based on: "Position: agentic AI orchestration should be Bayes-consistent"
(arXiv:2605.00742, ICML 2026)
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from victor.agent.bayesian_task_analysis import BayesianTaskAnalysis
from victor.agent.task_analyzer import TaskComplexity, UnifiedTaskType
from victor.framework.rl.learners.agent_reliability import AgentReliabilityLearner
from victor.framework.rl.learners.observation_model import ObservationModelLearner
from victor.framework.rl.learners.voi_controller import VoIController

logger = logging.getLogger(__name__)


class BayesianOrchestrationService:
    """Service for Bayes-consistent orchestration decisions.

    Integrates:
    - Observation models P(agent_message | task_outcome)
    - Reliability weights α_i for downweighting noisy agents
    - Value of Information for query decisions
    - Belief state tracking and Bayesian updates

    This provides the orchestration layer with principled decision-making
    under uncertainty.
    """

    def __init__(
        self,
        db_connection: Any,
        observation_learner: ObservationModelLearner,
        reliability_learner: AgentReliabilityLearner,
        voi_controller: VoIController,
    ):
        """Initialize Bayesian orchestration service.

        Args:
            db_connection: SQLite database connection
            observation_learner: Learner for P(agent_message | task_outcome)
            reliability_learner: Learner for reliability weights α_i
            voi_controller: Controller for VoI computation
        """
        self.db = db_connection
        self.observation_learner = observation_learner
        self.reliability_learner = reliability_learner
        self.voi_controller = voi_controller

        # In-memory belief state cache
        self._belief_cache: Dict[str, BayesianTaskAnalysis] = {}

        # Ensure tables exist
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        self.db.execute("""CREATE TABLE IF NOT EXISTS rl_belief_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            belief_id TEXT NOT NULL,
            success_prob REAL NOT NULL,
            failure_prob REAL NOT NULL,
            entropy REAL NOT NULL,
            agent_id TEXT,
            message TEXT,
            timestamp TEXT NOT NULL
        )""")

        # Create indexes
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_belief_history_id " "ON rl_belief_history(belief_id)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_belief_history_timestamp "
            "ON rl_belief_history(timestamp)"
        )

    def create_belief_state(
        self,
        task_type: str,
        complexity: TaskComplexity,
        tool_budget: int,
        initial_belief: Dict[str, float],
    ) -> BayesianTaskAnalysis:
        """Create a new belief state for a task.

        Args:
            task_type: Type of task (e.g., "code_edit", "code_generation")
            complexity: Task complexity level
            tool_budget: Budget for tool calls
            initial_belief: Initial prior over outcomes, e.g., {"success": 0.5, "failure": 0.5}

        Returns:
            BayesianTaskAnalysis with unique belief_id
        """
        # Generate unique belief ID
        belief_id = str(uuid.uuid4())

        # Map task_type to UnifiedTaskType
        task_type_mapping = {
            "code_edit": UnifiedTaskType.EDIT,
            "code_generation": UnifiedTaskType.GENERATION,
            "code_search": UnifiedTaskType.SEARCH,
            "debugging": UnifiedTaskType.EDIT,
            "refactoring": UnifiedTaskType.EDIT,
        }
        unified_type = task_type_mapping.get(task_type, UnifiedTaskType.EDIT)

        # Create belief state
        belief = BayesianTaskAnalysis(
            complexity=complexity,
            tool_budget=tool_budget,
            complexity_confidence=0.8,
            unified_task_type=unified_type,
            outcome_belief=initial_belief.copy(),
        )

        # Add belief_id
        belief.belief_id = belief_id

        # Cache belief state
        self._belief_cache[belief_id] = belief

        # Record initial belief in history
        self._record_belief_snapshot(
            belief_id=belief_id,
            belief=belief,
            agent_id=None,
            message=None,
        )

        logger.info(
            f"Created belief state {belief_id} for task {task_type}, "
            f"initial_belief={initial_belief}"
        )

        return belief

    def get_belief_state(self, belief_id: str) -> Optional[BayesianTaskAnalysis]:
        """Get a belief state by ID.

        Args:
            belief_id: Unique belief state identifier

        Returns:
            BayesianTaskAnalysis or None if not found
        """
        # Check cache first
        if belief_id in self._belief_cache:
            return self._belief_cache[belief_id]

        # TODO: Load from database if not in cache
        logger.warning(f"Belief state {belief_id} not found in cache")
        return None

    def update_belief_with_message(
        self,
        belief_id: str,
        agent_id: str,
        message: str,
        confidence: float,
    ) -> Optional[BayesianTaskAnalysis]:
        """Update belief state using Bayesian posterior update.

        Computes: P(Y|D,z) ∝ P(Y|D) * P(z|Y)^α_i

        Where:
            P(Y|D) is current belief (prior)
            P(z|Y) is observation model (likelihood)
            α_i is reliability weight for downweighting

        Args:
            belief_id: Belief state to update
            agent_id: Agent providing the message
            message: Agent's message
            confidence: Agent's confidence (0.0-1.0)

        Returns:
            Updated BayesianTaskAnalysis or None if not found
        """
        belief = self.get_belief_state(belief_id)
        if not belief:
            return None

        # Get reliability weight for agent
        reliability_weight = self.reliability_learner.get_reliability_weight(agent_id)

        # Store reliability in belief for weighting
        belief.agent_reliability[agent_id] = reliability_weight

        # For Bayesian update, we need likelihood for each outcome
        # P(z|Y) for each possible outcome Y
        # We approximate this using the observation model

        # Simplified: create likelihood based on message category
        # Affirming messages → higher likelihood for success
        # Denying messages → higher likelihood for failure
        message_lower = message.lower()

        # Determine if message is affirming or denying
        affirm_keywords = [
            "yes",
            "correct",
            "right",
            "good",
            "success",
            "works",
            "will",
        ]
        deny_keywords = [
            "no",
            "wrong",
            "incorrect",
            "bad",
            "fail",
            "doesn't",
            "won't",
            "can't",
        ]

        is_affirming = any(kw in message_lower for kw in affirm_keywords)
        is_denying = any(kw in message_lower for kw in deny_keywords)

        # Create likelihood distribution
        if is_affirming:
            likelihood = {"success": 0.8, "failure": 0.2}
        elif is_denying:
            likelihood = {"success": 0.2, "failure": 0.8}
        else:
            # Uncertain
            likelihood = {"success": 0.5, "failure": 0.5}

        # Apply reliability weighting to likelihood
        # P(z|Y)^α_i
        weighted_likelihood = {
            outcome: prob**reliability_weight for outcome, prob in likelihood.items()
        }

        # Compute posterior
        belief.compute_posterior(
            prior=belief.outcome_belief,
            likelihood=weighted_likelihood,
            agent_id=agent_id,
        )

        # Record belief snapshot
        self._record_belief_snapshot(
            belief_id=belief_id,
            belief=belief,
            agent_id=agent_id,
            message=message,
        )

        logger.debug(
            f"Updated belief {belief_id} with message from {agent_id}: "
            f"success={belief.outcome_belief['success']:.3f}, "
            f"reliability={reliability_weight:.3f}"
        )

        return belief

    def should_query_agent(
        self,
        belief_id: str,
        agent_id: str,
        query_cost: float,
    ) -> bool:
        """Decide whether to query an agent based on VoI.

        Args:
            belief_id: Current belief state
            agent_id: Agent being considered
            query_cost: Cost of querying the agent

        Returns:
            True if expected information gain exceeds cost
        """
        belief = self.get_belief_state(belief_id)
        if not belief:
            return False

        return self.voi_controller.should_query(
            task_analysis=belief,
            agent_id=agent_id,
            query_cost=query_cost,
        )

    def select_best_agent_to_query(
        self,
        belief_id: str,
        agent_ids: List[str],
        query_cost: float,
    ) -> Optional[str]:
        """Select the best agent to query based on VoI ranking.

        Args:
            belief_id: Current belief state
            agent_ids: List of candidate agents
            query_cost: Cost of querying

        Returns:
            Agent ID with highest VoI, or None if no good candidates
        """
        belief = self.get_belief_state(belief_id)
        if not belief:
            return None

        # Rank agents by VoI
        rankings = self.voi_controller.rank_agents_by_voi(
            task_analysis=belief,
            agent_ids=agent_ids,
            query_cost=query_cost,
        )

        if not rankings:
            return None

        # Return agent with highest VoI
        # Only if VoI is positive
        if rankings[0]["voi"] > 0:
            return rankings[0]["agent_id"]

        return None

    def record_task_outcome(
        self,
        belief_id: str,
        agent_id: str,
        actual_outcome: str,
        agent_message: str,
        agent_confidence: float,
    ) -> None:
        """Record task outcome for learning.

        Updates all learners:
        - ObservationModelLearner: records P(message | outcome)
        - AgentReliabilityLearner: records prediction correctness
        - VoIController: records query outcome if applicable

        Args:
            belief_id: Belief state for the task
            agent_id: Agent that was consulted
            actual_outcome: Actual task outcome ("success", "failure", "partial")
            agent_message: Agent's message
            agent_confidence: Agent's confidence in message
        """
        # Determine if agent's prediction was correct
        # For simplicity: affirming message + success = correct
        # denying message + failure = correct
        message_lower = agent_message.lower()
        affirm_keywords = ["yes", "correct", "right", "good", "success", "works"]
        deny_keywords = ["no", "wrong", "incorrect", "bad", "fail", "doesn't", "won't"]

        is_affirming = any(kw in message_lower for kw in affirm_keywords)
        is_denying = any(kw in message_lower for kw in deny_keywords)

        was_correct = (is_affirming and actual_outcome == "success") or (
            is_denying and actual_outcome == "failure"
        )

        # Record observation in observation model
        self.observation_learner.record_observation(
            agent_id=agent_id,
            message=agent_message,
            actual_outcome=actual_outcome,
            confidence=agent_confidence,
        )

        # Record prediction result in reliability learner
        self.reliability_learner.record_prediction_result(
            agent_id=agent_id,
            was_correct=was_correct,
            calibration_error=abs(agent_confidence - (1.0 if actual_outcome == "success" else 0.0)),
        )

        logger.info(
            f"Recorded task outcome: agent={agent_id}, outcome={actual_outcome}, "
            f"correct={was_correct}"
        )

    def get_belief_history(self, belief_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get history of belief state changes.

        Args:
            belief_id: Belief state ID
            limit: Maximum number of history entries to return

        Returns:
            List of belief snapshots with timestamps
        """
        cursor = self.db.execute(
            """SELECT success_prob, failure_prob, entropy, agent_id, message, timestamp
               FROM rl_belief_history
               WHERE belief_id=?
               ORDER BY timestamp ASC
               LIMIT ?
            """,
            (belief_id, limit),
        )

        history = []
        for row in cursor.fetchall():
            success_prob, failure_prob, entropy, agent_id, message, timestamp = row
            history.append(
                {
                    "success_prob": success_prob,
                    "failure_prob": failure_prob,
                    "entropy": entropy,
                    "agent_id": agent_id,
                    "message": message,
                    "timestamp": timestamp,
                }
            )

        return history

    def _record_belief_snapshot(
        self,
        belief_id: str,
        belief: BayesianTaskAnalysis,
        agent_id: Optional[str],
        message: Optional[str],
    ) -> None:
        """Record a snapshot of belief state to history.

        Args:
            belief_id: Belief state ID
            belief: Current belief state
            agent_id: Agent that caused the update (if any)
            message: Agent's message (if any)
        """
        self.db.execute(
            """INSERT INTO rl_belief_history
               (belief_id, success_prob, failure_prob, entropy, agent_id, message, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                belief_id,
                belief.outcome_belief.get("success", 0.0),
                belief.outcome_belief.get("failure", 0.0),
                belief.belief_entropy,
                agent_id,
                message,
                datetime.now().isoformat(),
            ),
        )

    def cleanup_belief_state(self, belief_id: str) -> None:
        """Clean up a belief state from cache.

        Args:
            belief_id: Belief state ID to clean up
        """
        if belief_id in self._belief_cache:
            del self._belief_cache[belief_id]
            logger.debug(f"Cleaned up belief state {belief_id}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics for Bayesian orchestration.

        Returns:
            Dict with aggregate statistics
        """
        # Get statistics from all learners
        observation_stats = self.observation_learner.get_calibration_stats()
        reliability_stats = self.reliability_learner.get_all_agent_stats()
        voi_stats = self.voi_controller.get_all_voi_stats()

        # Count active belief states
        active_beliefs = len(self._belief_cache)

        return {
            "active_belief_states": active_beliefs,
            "agents_with_observations": len(observation_stats),
            "agents_with_reliability": len(reliability_stats),
            "agents_with_voi_stats": len(voi_stats),
        }
