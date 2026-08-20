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

"""Observation model learner for P(agent_message | task_outcome).

This learner tracks how reliable agents are at predicting task outcomes,
enabling Bayesian belief updates in orchestration.

Based on: "Position: agentic AI orchestration should be Bayes-consistent"
(arXiv:2605.00742, ICML 2026)
"""

import logging
import random
from datetime import datetime
from typing import Any, Dict, Optional

from victor.framework.rl.base import BaseLearner, RLOutcome, RLRecommendation

logger = logging.getLogger(__name__)

# Default Beta prior parameters (weakly informative)
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0

# Message categories for observation modeling
MESSAGE_CATEGORIES = ["affirm", "deny", "uncertain", "error"]


def categorize_message(message: str, outcome: str, confidence: float) -> str:
    """Categorize agent message for likelihood modeling.

    Args:
        message: The agent's message text
        outcome: Actual task outcome ("success", "failure", "partial")
        confidence: Agent's confidence in their message

    Returns:
        Category: "affirm", "deny", "uncertain", or "error"
    """
    message_lower = message.lower()

    # Check for error indicators (highest priority)
    error_keywords = ["error", "failed", "exception", "crash", "broken"]
    if any(keyword in message_lower for keyword in error_keywords):
        return "error"

    # Check for uncertainty indicators
    uncertain_keywords = [
        "maybe",
        "possibly",
        "might",
        "uncertain",
        "not sure",
        "could be",
        "probably",
        "likely",
    ]
    if any(keyword in message_lower for keyword in uncertain_keywords):
        return "uncertain"

    # Check for negative/denying indicators (agent predicts failure)
    # Check deny BEFORE affirm to handle "won't work" correctly
    deny_keywords = [
        "no",
        "wrong",
        "incorrect",
        "bad",
        "fail",
        "doesn't",
        "won't",
        "can't",
        "not work",
    ]
    if any(keyword in message_lower for keyword in deny_keywords):
        return "deny"

    # Check for positive/affirming indicators (agent predicts success)
    affirm_keywords = ["yes", "correct", "right", "good", "success", "works"]
    if any(keyword in message_lower for keyword in affirm_keywords):
        return "affirm"

    # If no clear keywords, use confidence as fallback
    if confidence > 0.7:
        return "affirm"
    elif confidence < 0.3:
        return "deny"
    else:
        return "uncertain"


class ObservationModelLearner(BaseLearner):
    """Learn P(agent_message | task_outcome) for Bayesian updates.

    Tracks Beta distribution parameters for each (agent, outcome, message_category)
    triplet to model how agent messages correlate with task outcomes.

    Attributes:
        _calibration_params: Dict mapping (agent_id, outcome, category) -> (alpha, beta)
    """

    def __init__(
        self,
        name: str,
        db_connection: Any,
        learning_rate: float = 0.1,
        provider_adapter: Optional[Any] = None,
    ):
        """Initialize observation model learner.

        Args:
            name: Learner name (should be "observation_model")
            db_connection: SQLite database connection
            learning_rate: Update rate for parameters (not currently used)
            provider_adapter: Optional provider adapter for baselines
        """
        super().__init__(name, db_connection, learning_rate, provider_adapter)

        # In-memory cache of calibration parameters
        self._calibration_params: Dict[str, Dict[str, Dict[str, float]]] = {}

    def _ensure_tables(self) -> None:
        """Ensure rl_observation_model table exists."""
        self.db.execute(f"""CREATE TABLE IF NOT EXISTS rl_observation_model (
            agent_id TEXT NOT NULL,
            outcome_type TEXT NOT NULL,
            message_category TEXT NOT NULL,
            alpha REAL NOT NULL DEFAULT {PRIOR_ALPHA},
            beta REAL NOT NULL DEFAULT {PRIOR_BETA},
            sample_count INTEGER NOT NULL DEFAULT 0,
            last_updated TEXT NOT NULL,
            PRIMARY KEY (agent_id, outcome_type, message_category)
        )""")

        # Create index for faster lookups
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_observation_model_agent "
            "ON rl_observation_model(agent_id)"
        )

    def record_observation(
        self,
        agent_id: str,
        message: str,
        actual_outcome: str,
        confidence: float,
    ) -> None:
        """Update likelihood model: P(message | outcome).

        Args:
            agent_id: Agent that provided the message
            message: The agent's message
            actual_outcome: Actual task outcome ("success", "failure", "partial")
            confidence: Agent's confidence in their message (0.0-1.0)
        """
        # Categorize the message
        category = categorize_message(message, actual_outcome, confidence)

        # Update Beta parameters for this (agent, outcome, category) triplet
        # Beta(α, β) models the distribution of calibration

        # Check if this is a "correct" prediction
        # For affirm category: correct if message matches outcome
        # For deny category: correct if message doesn't match outcome
        is_correct = self._is_correct_prediction(category, actual_outcome)

        # Increment alpha for correct predictions, beta for incorrect
        # On first observation, include the increment in the INSERT
        # On subsequent observations, the UPDATE clause adds the increment
        alpha_increment = 1 if is_correct else 0
        beta_increment = 0 if is_correct else 1

        self.db.execute(
            """INSERT INTO rl_observation_model (agent_id, outcome_type, message_category, alpha, beta, sample_count, last_updated)
               VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(agent_id, outcome_type, message_category)
            DO UPDATE SET
                alpha = alpha + ?,
                beta = beta + ?,
                sample_count = sample_count + 1,
                last_updated = ?
            """,
            (
                agent_id,
                actual_outcome,
                category,
                PRIOR_ALPHA + alpha_increment,  # Initial alpha with increment
                PRIOR_BETA + beta_increment,  # Initial beta with increment
                datetime.now().isoformat(),
                alpha_increment,  # Increment alpha
                beta_increment,  # Increment beta
                datetime.now().isoformat(),
            ),
        )

        logger.debug(
            f"Recorded observation: agent={agent_id}, outcome={actual_outcome}, "
            f"category={category}, correct={is_correct}"
        )

    def _is_correct_prediction(self, category: str, actual_outcome: str) -> bool:
        """Determine if agent's prediction was correct.

        Args:
            category: Message category ("affirm", "deny", "uncertain", "error")
            actual_outcome: Actual task outcome

        Returns:
            True if the prediction was correct
        """
        if category == "affirm":
            # Agent affirmed the outcome would happen - correct if outcome is success
            return actual_outcome == "success"
        elif category == "deny":
            # Agent denied the outcome would happen - correct if outcome is failure
            return actual_outcome == "failure"
        elif category == "uncertain":
            # Agent was uncertain, we can't say it was correct or incorrect
            return False  # Treat as incorrect for Bayesian updating
        elif category == "error":
            # Agent reported an error - correct if outcome is failure
            return actual_outcome == "failure"
        else:
            return False

    def get_likelihood(self, agent_id: str, message: str, outcome: str) -> float:
        """Return P(message | outcome) for Bayes update.

        Uses Thompson Sampling to sample from Beta posterior, enabling
        exploration of uncertain observation models.

        Args:
            agent_id: Agent that provided the message
            message: The agent's message (will be categorized)
            outcome: Task outcome to get likelihood for

        Returns:
            Likelihood P(message | outcome) in [0, 1]
        """
        # Categorize the message
        # Note: We don't have confidence here, so we use a heuristic
        category = self._categorize_message_heuristic(message)

        # Get Beta parameters for this (agent, outcome, category)
        params = self._get_parameters(agent_id, outcome, category)

        alpha = params["alpha"]
        beta = params["beta"]

        # Thompson Sampling provides uncertainty estimates and exploration.
        # For small sample counts we temper the draw back toward the posterior
        # mean so one noisy sample does not overwhelm sparse but consistent
        # evidence during orchestration decisions.
        sampled_likelihood = self._rng.betavariate(alpha, beta)
        sample_count = int(params["sample_count"])
        if sample_count <= 0:
            likelihood = sampled_likelihood
        elif sample_count < 10:
            posterior_mean = alpha / (alpha + beta)
            exploration_scale = sample_count / (sample_count + 5.0)
            likelihood = posterior_mean + exploration_scale * (sampled_likelihood - posterior_mean)
        else:
            likelihood = sampled_likelihood

        logger.debug(
            f"Sampled likelihood for {agent_id}/{category}/{outcome}: "
            f"{likelihood:.4f} (alpha={alpha:.1f}, beta={beta:.1f})"
        )

        return likelihood

    def _categorize_message_heuristic(self, message: str) -> str:
        """Categorize message without confidence (heuristic).

        Args:
            message: The agent's message

        Returns:
            Category: "affirm", "deny", "uncertain", or "error"
        """
        message_lower = message.lower()

        # Check for error indicators (highest priority)
        error_keywords = ["error", "failed", "exception", "crash", "broken"]
        if any(keyword in message_lower for keyword in error_keywords):
            return "error"

        # Check for uncertainty indicators
        uncertain_keywords = [
            "maybe",
            "possibly",
            "might",
            "uncertain",
            "not sure",
            "could be",
            "probably",
            "likely",
        ]
        if any(keyword in message_lower for keyword in uncertain_keywords):
            return "uncertain"

        # Check for negative/denying indicators (agent predicts failure)
        # Check deny BEFORE affirm to handle "won't work" correctly
        deny_keywords = [
            "no",
            "wrong",
            "incorrect",
            "bad",
            "fail",
            "doesn't",
            "won't",
            "can't",
            "not work",
        ]
        if any(keyword in message_lower for keyword in deny_keywords):
            return "deny"

        # Check for positive/affirming indicators (agent predicts success)
        affirm_keywords = ["yes", "correct", "right", "good", "success", "works"]
        if any(keyword in message_lower for keyword in affirm_keywords):
            return "affirm"

        # Default to uncertain if no clear signal
        return "uncertain"

    def _get_parameters(self, agent_id: str, outcome: str, category: str) -> Dict[str, float]:
        """Get Beta parameters for (agent, outcome, category).

        Args:
            agent_id: Agent identifier
            outcome: Task outcome
            category: Message category

        Returns:
            Dict with "alpha", "beta", and "sample_count" parameters
        """
        # Try database first
        cursor = self.db.execute(
            """SELECT alpha, beta, sample_count FROM rl_observation_model
               WHERE agent_id=? AND outcome_type=? AND message_category=?
            """,
            (agent_id, outcome, category),
        )

        result = cursor.fetchone()

        if result:
            alpha, beta, sample_count = result
            return {"alpha": alpha, "beta": beta, "sample_count": sample_count}
        else:
            # Return prior if no data
            return {"alpha": PRIOR_ALPHA, "beta": PRIOR_BETA, "sample_count": 0}

    def get_agent_calibration(self, agent_id: str) -> Dict[str, Any]:
        """Return calibration metrics for reliability weighting.

        Args:
            agent_id: Agent to get calibration for

        Returns:
            Dict with calibration metrics by category
        """
        cursor = self.db.execute(
            """SELECT outcome_type, message_category,
                      SUM(alpha) as total_alpha,
                      SUM(beta) as total_beta,
                      SUM(sample_count) as total_samples
               FROM rl_observation_model
               WHERE agent_id=?
               GROUP BY outcome_type, message_category
            """,
            (agent_id,),
        )

        # Group by message_category, aggregating calibration errors
        calibration_by_category = {}

        for row in cursor.fetchall():
            outcome_type, message_category, total_alpha, total_beta, total_samples = row

            if total_samples > 0:
                # Expected probability from Beta distribution
                # E[p] = alpha / (alpha + beta)
                expected_prob = total_alpha / (total_alpha + total_beta)

                # Empirical correctness rate (excluding priors)
                correct_predictions = total_alpha - PRIOR_ALPHA
                incorrect_predictions = total_beta - PRIOR_BETA
                total_predictions = correct_predictions + incorrect_predictions

                if total_predictions > 0:
                    empirical_freq = correct_predictions / total_predictions
                    # Calibration error: difference between expected and empirical
                    outcome_calibration_error = abs(expected_prob - empirical_freq)
                else:
                    outcome_calibration_error = 0.0

                # Accumulate calibration errors by message_category
                if message_category not in calibration_by_category:
                    calibration_by_category[message_category] = {
                        "total_error": 0.0,
                        "total_samples": 0,
                        "outcome_count": 0,
                    }

                calibration_by_category[message_category][
                    "total_error"
                ] += outcome_calibration_error
                calibration_by_category[message_category]["total_samples"] += total_samples
                calibration_by_category[message_category]["outcome_count"] += 1

        # Compute final calibration metrics
        calibration = {}
        for message_category, data in calibration_by_category.items():
            # Average calibration error across outcomes
            avg_calibration_error = data["total_error"] / data["outcome_count"]

            # For expected_prob, we need to look at the underlying data
            # Use a weighted average based on sample counts
            cursor2 = self.db.execute(
                """SELECT outcome_type, message_category,
                          SUM(alpha) as total_alpha,
                          SUM(beta) as total_beta,
                          SUM(sample_count) as total_samples
                   FROM rl_observation_model
                   WHERE agent_id=? AND message_category=?
                   GROUP BY outcome_type, message_category
                """,
                (agent_id, message_category),
            )

            weighted_expected_prob = 0.0
            total_weight = 0

            for row in cursor2.fetchall():
                outcome_type, msg_category, total_alpha, total_beta, total_samples = row
                expected_prob = total_alpha / (total_alpha + total_beta)
                weighted_expected_prob += expected_prob * total_samples
                total_weight += total_samples

            if total_weight > 0:
                avg_expected_prob = weighted_expected_prob / total_weight
            else:
                avg_expected_prob = 0.5

            calibration[message_category] = {
                "expected_prob": avg_expected_prob,
                "calibration_error": avg_calibration_error,
                "sample_count": data["total_samples"],
            }

        # If no data, return default calibration
        if not calibration:
            calibration["affirm"] = {
                "expected_prob": 0.5,  # Uniform prior
                "calibration_error": 0.0,
                "sample_count": 0,
            }

        return calibration

    def record_outcome(self, outcome: RLOutcome) -> None:
        """Record an RL outcome (BaseLearner interface).

        This is called by the RL framework after task completion.
        We extract observation data from the outcome metadata.

        Args:
            outcome: RL outcome with metadata containing observation info
        """
        # Extract observation data from metadata
        agent_id = outcome.metadata.get("agent_id")
        message = outcome.metadata.get("agent_message")
        actual_outcome = "success" if outcome.success else "failure"
        confidence = outcome.metadata.get("agent_confidence", 0.5)

        if not agent_id or not message:
            logger.warning(
                f"Missing agent_id or agent_message in outcome metadata: "
                f"{outcome.metadata.keys()}"
            )
            return

        # Record the observation
        self.record_observation(agent_id, message, actual_outcome, confidence)

    def get_recommendation(self, context: Dict[str, Any]) -> Optional[RLRecommendation]:
        """Get likelihood recommendation for given context.

        Args:
            context: Context dict with agent_id, message, outcome_type

        Returns:
            RLRecommendation with likelihood info
        """
        # This is used for querying the model
        # For observation model, we return likelihood info
        return RLRecommendation(
            value="observation_model",
            confidence=0.8,
            reason="Observation model for Bayesian belief updates",
            sample_size=0,  # Will be populated if needed
            is_baseline=False,
            metadata={
                "learner_name": "observation_model",
                "recommendation_type": "likelihood",
            },
        )

    def _compute_reward(self, outcome: RLOutcome) -> float:
        """Compute reward from observation outcome.

        Args:
            outcome: RL outcome with observation metadata

        Returns:
            Reward value (positive for good calibration, negative for poor)
        """
        # Extract calibration data
        agent_id = outcome.metadata.get("agent_id")
        if not agent_id:
            return 0.0

        # Get calibration for this agent
        calibration = self.get_agent_calibration(agent_id)
        if not calibration:
            return 0.0

        # Reward based on average calibration error across all categories
        total_error = sum(cat.get("calibration_error", 0.5) for cat in calibration.values())
        avg_error = total_error / len(calibration) if calibration else 0.5

        # Reward is negative of error (lower error = higher reward)
        # Scale to [-1.0, 1.0]
        reward = max(-1.0, min(1.0, -avg_error))

        logger.debug(
            f"Computed reward for {agent_id}: {reward:.4f} " f"(calibration_error={avg_error:.4f})"
        )

        return reward

    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get overall calibration statistics across all agents.

        Returns:
            Dict with aggregate calibration metrics
        """
        cursor = self.db.execute("""SELECT agent_id,
                      SUM(sample_count) as total_samples,
                      AVG(CAST(alpha AS REAL) / CAST(alpha + beta AS REAL)) as avg_expected_prob
               FROM rl_observation_model
               GROUP BY agent_id
            """)

        stats = {}
        for row in cursor.fetchall():
            agent_id, total_samples, avg_expected_prob = row
            stats[agent_id] = {
                "total_samples": total_samples,
                "avg_expected_prob": avg_expected_prob,
            }

        return stats
