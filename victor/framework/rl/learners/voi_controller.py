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

"""VoI Controller for Expected Value of Information computation.

This controller computes the Expected Value of Information (VoI) for
querying agents, enabling intelligent decision-making about when to
seek additional information in Bayesian orchestration.

Based on: "Position: agentic AI orchestration should be Bayes-consistent"
(arXiv:2605.00742, ICML 2026)
"""

from victor.core.json_utils import json_dumps
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from victor.agent.bayesian_task_analysis import BayesianTaskAnalysis
from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation
from victor.framework.rl.learners.agent_reliability import AgentReliabilityLearner
from victor.framework.rl.learners.observation_model import ObservationModelLearner

logger = logging.getLogger(__name__)


class VoIController(BaseLearner):
    """Compute Expected Value of Information for agent queries.

    Uses observation models and reliability weights to compute VoI:
        VoI = E[H[Y|D] - H[Y|D,z]] - cost

    Where:
        H[Y|D] is current entropy
        E[H[Y|D,z]] is expected posterior entropy after observation
        cost is the query cost

    Attributes:
        observation_learner: Learner for P(agent_message | task_outcome)
        reliability_learner: Learner for reliability weights α_i
    """

    def __init__(
        self,
        name: str,
        db_connection: Any,
        observation_learner: ObservationModelLearner,
        reliability_learner: AgentReliabilityLearner,
        learning_rate: float = 0.1,
        provider_adapter: Optional[Any] = None,
    ):
        """Initialize VoI controller.

        Args:
            name: Controller name (should be "voi_controller")
            db_connection: SQLite database connection
            observation_learner: Learner for observation models
            reliability_learner: Learner for reliability weights
            learning_rate: Update rate for parameters (not currently used)
            provider_adapter: Optional provider adapter for baselines
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)

        self.observation_learner = observation_learner
        self.reliability_learner = reliability_learner

    def _ensure_tables(self) -> None:
        """Ensure rl_voi_history table exists."""
        self.db.execute("""CREATE TABLE IF NOT EXISTS rl_voi_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            predicted_voi REAL NOT NULL,
            actual_information_gain REAL NOT NULL,
            query_cost REAL NOT NULL,
            was_beneficial BOOLEAN NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT
        )""")

        # Create indexes for faster lookups
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_voi_history_agent " "ON rl_voi_history(agent_id)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_voi_history_timestamp " "ON rl_voi_history(timestamp)"
        )

    def compute_voi(
        self,
        task_analysis: BayesianTaskAnalysis,
        agent_id: str,
        query_cost: float,
        reliability_weight: Optional[float] = None,
    ) -> float:
        """Compute Expected Value of Information for querying an agent.

        Args:
            task_analysis: Current task analysis with belief state
            agent_id: Agent being considered for query
            query_cost: Cost of making the query (in entropy units)
            reliability_weight: Optional precomputed weight override

        Returns:
            Expected information gain minus query cost. Positive values
            indicate the query is worth making.
        """
        # Current entropy: H[Y|D]
        current_entropy = task_analysis.belief_entropy

        # Get agent reliability weight
        if reliability_weight is None:
            reliability_weight = self.reliability_learner.get_reliability_weight(agent_id)

        # Expected information gain depends on:
        # 1. Current uncertainty (entropy)
        # 2. Agent reliability
        # 3. Observation model (likelihood quality)

        # Approximate expected entropy reduction
        # More reliable agents → more entropy reduction
        # Higher current entropy → more potential reduction
        expected_entropy_reduction = current_entropy * reliability_weight * 0.5

        # Expected information gain
        expected_gain = expected_entropy_reduction

        # Subtract cost to get VoI
        voi = expected_gain - query_cost

        logger.debug(
            f"VoI for {agent_id}: E[IG]={expected_gain:.4f}, "
            f"cost={query_cost:.4f}, VoI={voi:.4f}, "
            f"reliability={reliability_weight:.3f}"
        )

        return voi

    def should_query(
        self,
        task_analysis: BayesianTaskAnalysis,
        agent_id: str,
        query_cost: float,
    ) -> bool:
        """Decision rule: query if EVOI > 0.

        Args:
            task_analysis: Current task analysis with belief state
            agent_id: Agent being considered
            query_cost: Cost of the query

        Returns:
            True if the expected information gain exceeds cost
        """
        voi = self.compute_voi(task_analysis, agent_id, query_cost)
        return voi > 0

    def rank_agents_by_voi(
        self,
        task_analysis: BayesianTaskAnalysis,
        agent_ids: List[str],
        query_cost: float,
    ) -> List[Dict[str, Any]]:
        """Rank multiple agents by their expected VoI.

        Args:
            task_analysis: Current task analysis with belief state
            agent_ids: List of agents to rank
            query_cost: Cost of querying any agent

        Returns:
            List of agents ranked by VoI (highest first)
        """
        agent_vois = []

        for agent_id in agent_ids:
            reliability = self.reliability_learner.get_expected_reliability_weight(agent_id)
            voi = self.compute_voi(
                task_analysis,
                agent_id,
                query_cost,
                reliability_weight=reliability,
            )

            agent_vois.append(
                {
                    "agent_id": agent_id,
                    "voi": voi,
                    "reliability": reliability,
                    "should_query": voi > 0,
                }
            )

        # Sort by VoI descending
        agent_vois.sort(key=lambda x: x["voi"], reverse=True)

        return agent_vois

    def record_query_outcome(
        self,
        agent_id: str,
        predicted_voi: float,
        actual_information_gain: float,
        query_cost: float,
        was_beneficial: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record query outcome for learning.

        Args:
            agent_id: Agent that was queried
            predicted_voi: Predicted VoI before query
            actual_information_gain: Actual information gain from query
            query_cost: Cost of the query
            was_beneficial: Whether the query was beneficial
            metadata: Optional additional metadata
        """
        import json

        metadata_json = json_dumps(metadata) if metadata else None

        self.db.execute(
            """INSERT INTO rl_voi_history
               (agent_id, predicted_voi, actual_information_gain, query_cost, was_beneficial, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                predicted_voi,
                actual_information_gain,
                query_cost,
                was_beneficial,
                datetime.now().isoformat(),
                metadata_json,
            ),
        )

        logger.debug(
            f"Recorded query outcome: agent={agent_id}, "
            f"predicted_voi={predicted_voi:.4f}, actual_gain={actual_information_gain:.4f}, "
            f"beneficial={was_beneficial}"
        )

    def get_agent_voi_stats(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Return VoI statistics for an agent.

        Args:
            agent_id: Agent to get statistics for

        Returns:
            Dict with VoI metrics or None if agent not found
        """
        cursor = self.db.execute(
            """SELECT
                  COUNT(*) as query_count,
                  SUM(CASE WHEN was_beneficial THEN 1 ELSE 0 END) as beneficial_count,
                  AVG(predicted_voi) as avg_predicted_voi,
                  AVG(actual_information_gain) as avg_actual_gain,
                  AVG(query_cost) as avg_cost
               FROM rl_voi_history
               WHERE agent_id=?
            """,
            (agent_id,),
        )

        result = cursor.fetchone()

        if not result or result[0] == 0:
            return None

        query_count, beneficial_count, avg_predicted, avg_actual, avg_cost = result

        return {
            "query_count": query_count,
            "beneficial_count": beneficial_count,
            "beneficial_rate": (beneficial_count / query_count if query_count > 0 else 0.0),
            "avg_predicted_voi": avg_predicted,
            "avg_actual_gain": avg_actual,
            "avg_cost": avg_cost,
            "voi_accuracy": (1.0 - abs(avg_predicted - avg_actual) if avg_actual else 0.0),
        }

    def get_all_voi_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return VoI statistics for all agents.

        Returns:
            Dict mapping agent_id to VoI stats
        """
        cursor = self.db.execute("""SELECT
                  agent_id,
                  COUNT(*) as query_count,
                  SUM(CASE WHEN was_beneficial THEN 1 ELSE 0 END) as beneficial_count,
                  AVG(predicted_voi) as avg_predicted_voi,
                  AVG(actual_information_gain) as avg_actual_gain
               FROM rl_voi_history
               GROUP BY agent_id
            """)

        stats = {}
        for row in cursor.fetchall():
            agent_id, query_count, beneficial_count, avg_predicted, avg_actual = row

            stats[agent_id] = {
                "query_count": query_count,
                "beneficial_count": beneficial_count,
                "beneficial_rate": (beneficial_count / query_count if query_count > 0 else 0.0),
                "avg_predicted_voi": avg_predicted,
                "avg_actual_gain": avg_actual,
            }

        return stats

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record an RL outcome (BaseLearner interface).

        Args:
            outcome: RL outcome with VoI metadata
        """
        # Extract VoI data from metadata
        agent_id = outcome.metadata.get("agent_id")
        predicted_voi = outcome.metadata.get("predicted_voi", 0.0)
        actual_gain = outcome.metadata.get("actual_information_gain", 0.0)
        query_cost = outcome.metadata.get("query_cost", 0.0)
        was_beneficial = outcome.metadata.get("was_beneficial", False)

        if not agent_id:
            logger.warning(f"Missing agent_id in outcome metadata: " f"{outcome.metadata.keys()}")
            return

        # Record the query outcome
        self.record_query_outcome(agent_id, predicted_voi, actual_gain, query_cost, was_beneficial)

    def get_recommendation(self, context: Dict[str, Any]) -> Optional[RLRecommendation]:
        """Get VoI recommendation for given context.

        Args:
            context: Context dict with task_analysis, agent_id, query_cost

        Returns:
            RLRecommendation with VoI info
        """
        task_analysis = context.get("task_analysis")
        agent_id = context.get("agent_id")
        query_cost = context.get("query_cost", 0.1)

        if not task_analysis or not agent_id:
            return None

        # Compute VoI
        voi = self.compute_voi(task_analysis, agent_id, query_cost)

        # Get agent stats for confidence
        stats = self.get_agent_voi_stats(agent_id)

        if stats:
            sample_count = stats["query_count"]
            voi_accuracy = stats.get("voi_accuracy", 0.5)
            confidence = min(1.0, voi_accuracy * sample_count / 10.0)
        else:
            sample_count = 0
            confidence = 0.5  # Default confidence for new agents

        return RLRecommendation(
            value="voi_controller",
            confidence=confidence,
            reason=f"VoI {voi:.3f} for agent {agent_id}",
            sample_size=sample_count,
            is_baseline=False,
            metadata={
                "learner_name": "voi_controller",
                "recommendation_type": "voi",
                "agent_id": agent_id,
                "voi": voi,
                "should_query": voi > 0,
            },
        )

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward from VoI outcome.

        Args:
            outcome: RL outcome with VoI metadata

        Returns:
            Reward value (positive for good VoI predictions, negative for poor)
        """
        # Extract VoI data
        predicted_voi = outcome.metadata.get("predicted_voi", 0.0)
        actual_gain = outcome.metadata.get("actual_information_gain", 0.0)

        # Reward based on prediction accuracy
        # Good predictions: predicted ≈ actual
        # Bad predictions: large difference
        error = abs(predicted_voi - actual_gain)

        # Scale to [-1.0, 1.0]
        # Lower error = higher reward
        reward = max(-1.0, 1.0 - error * 2.0)

        logger.debug(
            f"Computed reward for VoI prediction: {reward:.4f} "
            f"(predicted={predicted_voi:.4f}, actual={actual_gain:.4f}, error={error:.4f})"
        )

        return reward
