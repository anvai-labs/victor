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

"""Agent reliability learner for reliability weights α_i.

This learner tracks how reliable each agent is at making predictions,
enabling reliability-weighted Bayesian updates in orchestration.

Based on: "Position: agentic AI orchestration should be Bayes-consistent"
(arXiv:2605.00742, ICML 2026)
"""

import logging
import random
from datetime import datetime
from typing import Any, Dict, Optional

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation

logger = logging.getLogger(__name__)

# Default Beta prior parameters for reliability
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0


class AgentReliabilityLearner(BaseLearner):
    """Learn reliability weights α_i for Bayesian updates.

    Tracks Beta distribution parameters for each agent to model
    their reliability in making predictions. High reliability agents
    get α_i > 1.0 (upweighting), low reliability get α_i < 1.0 (downweighting).

    The reliability weight is used to scale likelihoods:
        P(Y|D,z) ∝ P(Y|D) * P(z|Y)^α_i

    Attributes:
        _reliability_cache: In-memory cache of reliability weights
    """

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.1,
        provider_adapter: Optional[Any] = None,
    ):
        """Initialize agent reliability learner.

        Args:
            name: Learner name (should be "agent_reliability")
            db_connection: SQLite database connection
            learning_rate: Update rate for parameters (not currently used)
            provider_adapter: Optional provider adapter for baselines
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)

    def _ensure_tables(self) -> None:
        """Ensure rl_agent_reliability table exists."""
        self.db.execute(f"""CREATE TABLE IF NOT EXISTS rl_agent_reliability (
            agent_id TEXT NOT NULL PRIMARY KEY,
            alpha_reliability REAL NOT NULL DEFAULT {PRIOR_ALPHA},
            beta_reliability REAL NOT NULL DEFAULT {PRIOR_BETA},
            sample_count INTEGER NOT NULL DEFAULT 0,
            last_updated TEXT NOT NULL
        )""")

        # Create index for faster lookups
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_reliability_agent "
            "ON rl_agent_reliability(agent_id)"
        )

    def record_prediction_result(
        self,
        agent_id: str,
        was_correct: bool,
        calibration_error: float,
    ) -> None:
        """Update reliability model based on prediction result.

        Args:
            agent_id: Agent that made the prediction
            was_correct: Whether the prediction was correct
            calibration_error: Calibration error (0.0-1.0, higher = worse)
        """
        # Compute weight for this observation
        # Lower calibration error → higher weight
        # Weight in [0.5, 1.5] based on calibration error
        weight = 1.0 + (0.5 - calibration_error)
        weight = max(0.5, min(1.5, weight))  # Clamp to [0.5, 1.5]

        # Update Beta parameters
        # For correct predictions: increment alpha (reliability evidence)
        # For incorrect predictions: increment beta (unreliability evidence)
        # Apply weight to scale the increment
        alpha_increment = weight if was_correct else 0.0
        beta_increment = weight if not was_correct else 0.0

        self.db.execute(
            """INSERT INTO rl_agent_reliability (agent_id, alpha_reliability, beta_reliability, sample_count, last_updated)
               VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(agent_id)
            DO UPDATE SET
                alpha_reliability = alpha_reliability + ?,
                beta_reliability = beta_reliability + ?,
                sample_count = sample_count + 1,
                last_updated = ?
            """,
            (
                agent_id,
                PRIOR_ALPHA + alpha_increment,  # Initial alpha with increment
                PRIOR_BETA + beta_increment,  # Initial beta with increment
                1,  # Initial sample count
                datetime.now().isoformat(),
                alpha_increment,  # Increment alpha
                beta_increment,  # Increment beta
                datetime.now().isoformat(),
            ),
        )

        logger.debug(
            f"Recorded prediction result: agent={agent_id}, "
            f"correct={was_correct}, error={calibration_error:.3f}, "
            f"weight={weight:.3f}"
        )

    def get_reliability_weight(self, agent_id: str) -> float:
        """Get reliability weight α_i for an agent.

        Uses Thompson Sampling to sample from the Beta posterior,
        providing uncertainty estimates and exploration.

        Args:
            agent_id: Agent to get reliability weight for

        Returns:
            Reliability weight α_i. Values:
            - > 1.0: Upweight (more reliable)
            - = 1.0: Neutral (default)
            - < 1.0: Downweight (less reliable)
        """
        # Get Beta parameters
        params = self._get_parameters(agent_id)
        alpha = params["alpha"]
        beta = params["beta"]

        # Check if this is a new agent (only prior, no samples)
        cursor = self.db.execute(
            "SELECT sample_count FROM rl_agent_reliability WHERE agent_id=?",
            (agent_id,),
        )
        result = cursor.fetchone()

        # If no data or only prior (sample_count = 0), return neutral weight
        if not result or result[0] == 0:
            return 1.0

        # Thompson Sampling: sample from Beta(alpha, beta).
        #
        # Keep exploration, but do not let the sampled reliability cross the
        # neutral boundary once the posterior mean is already on one side of it.
        # Otherwise an agent with clearly sub-50% reliability can be randomly
        # upweighted above 1.0, which is operationally the opposite of what the
        # accumulated evidence says.
        posterior_mean = alpha / (alpha + beta)
        sampled_reliability = self._rng.betavariate(alpha, beta)
        if posterior_mean < 0.5:
            expected_reliability = min(sampled_reliability, 0.5 - 1e-12)
        elif posterior_mean > 0.5:
            expected_reliability = max(sampled_reliability, 0.5 + 1e-12)
        else:
            expected_reliability = sampled_reliability

        # Convert to weight: α_i = reliability / (1 - reliability)
        # This maps [0, 1] to [0, ∞]
        # We use a softer mapping: α_i = 0.5 + reliability
        # This maps [0, 1] to [0.5, 1.5]
        weight = 0.5 + expected_reliability

        # Clamp to reasonable range [0.1, 5.0]
        weight = max(0.1, min(5.0, weight))

        logger.debug(
            f"Reliability weight for {agent_id}: {weight:.3f} "
            f"(alpha={alpha:.1f}, beta={beta:.1f})"
        )

        return weight

    def get_expected_reliability_weight(self, agent_id: str) -> float:
        """Return the deterministic posterior-mean reliability weight for an agent."""
        stats = self.get_agent_reliability_stats(agent_id)
        if not stats or stats["sample_count"] == 0:
            return 1.0

        expected_reliability = stats["expected_reliability"]
        weight = 0.5 + expected_reliability
        return max(0.1, min(5.0, weight))

    def _get_parameters(self, agent_id: str) -> Dict[str, float]:
        """Get Beta parameters for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with "alpha" and "beta" parameters
        """
        cursor = self.db.execute(
            """SELECT alpha_reliability, beta_reliability
               FROM rl_agent_reliability
               WHERE agent_id=?
            """,
            (agent_id,),
        )

        result = cursor.fetchone()

        if result:
            alpha, beta = result
            return {"alpha": alpha, "beta": beta}
        else:
            # Return prior if no data
            return {"alpha": PRIOR_ALPHA, "beta": PRIOR_BETA}

    def get_agent_reliability_stats(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Return reliability statistics for an agent.

        Args:
            agent_id: Agent to get statistics for

        Returns:
            Dict with reliability metrics or None if agent not found
        """
        cursor = self.db.execute(
            """SELECT alpha_reliability, beta_reliability, sample_count
               FROM rl_agent_reliability
               WHERE agent_id=?
            """,
            (agent_id,),
        )

        result = cursor.fetchone()

        if not result:
            return None

        alpha, beta, sample_count = result

        # Expected reliability from Beta distribution
        # E[r] = alpha / (alpha + beta)
        expected_reliability = alpha / (alpha + beta)

        return {
            "alpha": alpha,
            "beta": beta,
            "sample_count": sample_count,
            "expected_reliability": expected_reliability,
        }

    def get_all_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return reliability statistics for all agents.

        Returns:
            Dict mapping agent_id to reliability stats
        """
        cursor = self.db.execute(
            """SELECT agent_id, alpha_reliability, beta_reliability, sample_count
               FROM rl_agent_reliability
            """
        )

        stats = {}
        for row in cursor.fetchall():
            agent_id, alpha, beta, sample_count = row

            expected_reliability = alpha / (alpha + beta)

            stats[agent_id] = {
                "alpha": alpha,
                "beta": beta,
                "sample_count": sample_count,
                "expected_reliability": expected_reliability,
            }

        return stats

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record an RL outcome (BaseLearner interface).

        This is called by the RL framework after task completion.
        We extract reliability data from the outcome metadata.

        Args:
            outcome: RL outcome with metadata containing reliability info
        """
        # Extract reliability data from metadata
        agent_id = outcome.metadata.get("agent_id")
        was_correct = outcome.metadata.get("prediction_correct", False)
        calibration_error = outcome.metadata.get("calibration_error", 0.5)

        if not agent_id:
            logger.warning(f"Missing agent_id in outcome metadata: " f"{outcome.metadata.keys()}")
            return

        # Record the prediction result
        self.record_prediction_result(agent_id, was_correct, calibration_error)

    def get_recommendation(self, context: Dict[str, Any]) -> Optional[RLRecommendation]:
        """Get reliability recommendation for given context.

        Args:
            context: Context dict with agent_id

        Returns:
            RLRecommendation with reliability info
        """
        agent_id = context.get("agent_id")

        if not agent_id:
            return None

        # Get current reliability weight
        weight = self.get_reliability_weight(agent_id)

        # Get stats for context
        stats = self.get_agent_reliability_stats(agent_id)

        if stats:
            sample_count = stats["sample_count"]
            expected_reliability = stats["expected_reliability"]
        else:
            sample_count = 0
            expected_reliability = 0.5

        # Determine confidence based on sample count
        confidence = min(1.0, sample_count / 10.0)  # Full confidence after 10 samples

        return RLRecommendation(
            value="agent_reliability",
            confidence=confidence,
            reason=f"Reliability weight {weight:.3f} for agent {agent_id}",
            sample_size=sample_count,
            is_baseline=False,
            metadata={
                "learner_name": "agent_reliability",
                "recommendation_type": "reliability_weight",
                "agent_id": agent_id,
                "reliability_weight": weight,
                "expected_reliability": expected_reliability,
            },
        )

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward from reliability outcome.

        Args:
            outcome: RL outcome with reliability metadata

        Returns:
            Reward value (positive for good reliability, negative for poor)
        """
        # Extract reliability data
        agent_id = outcome.metadata.get("agent_id")
        if not agent_id:
            return 0.0

        # Get reliability for this agent
        stats = self.get_agent_reliability_stats(agent_id)
        if not stats:
            return 0.0

        # Reward based on expected reliability
        # Higher reliability = higher reward
        expected_reliability = stats["expected_reliability"]

        # Scale to [-1.0, 1.0]
        reward = (expected_reliability - 0.5) * 2.0
        reward = max(-1.0, min(1.0, reward))

        logger.debug(
            f"Computed reward for {agent_id}: {reward:.4f} "
            f"(reliability={expected_reliability:.4f})"
        )

        return reward

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get overall reliability statistics across all agents.

        Returns:
            Dict with aggregate reliability metrics
        """
        cursor = self.db.execute("""SELECT agent_id,
                      sample_count,
                      CAST(alpha_reliability AS REAL) / (alpha_reliability + beta_reliability) as expected_reliability
               FROM rl_agent_reliability
            """)

        stats = {}
        for row in cursor.fetchall():
            agent_id, sample_count, expected_reliability = row
            stats[agent_id] = {
                "sample_count": sample_count,
                "expected_reliability": expected_reliability,
            }

        return stats
