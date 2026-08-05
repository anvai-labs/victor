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

"""Bayesian consensus formation with reliability weighting.

This module provides Bayesian consensus formation that integrates
observation models and reliability weights for robust multi-agent
decision making under uncertainty.

Based on: "Position: agentic AI orchestration should be Bayes-consistent"
(arXiv:2605.00742, ICML 2026)
"""

from victor.core.json_utils import json_dumps
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from victor.framework.rl.learners.correlation_tracker import CorrelationTracker
from victor.framework.rl.orchestration.bayesian_orchestrator import (
    BayesianOrchestrationService,
)

logger = logging.getLogger(__name__)


class BayesianConsensusBuilder:
    """Build Bayesian consensus from multiple agent opinions.

    Integrates with BayesianOrchestrationService to provide:
    - Reliability-weighted evidence combination
    - Correlation-aware pooling (dependence-adjusted)
    - Multiple consensus strategies (majority vote, Bayesian, etc.)
    - Belief state updates from consensus
    - Outcome tracking for learning

    This enhances Victor's consensus mechanisms with principled
    Bayesian reasoning under uncertainty.
    """

    def __init__(
        self,
        orchestration_service: BayesianOrchestrationService,
        correlation_tracker: Optional["CorrelationTracker"] = None,
    ):
        """Initialize Bayesian consensus builder.

        Args:
            orchestration_service: Bayesian orchestration service
            correlation_tracker: Optional correlation tracker for dependence-aware pooling
        """
        self.orchestration_service = orchestration_service
        self.db = orchestration_service.db
        self.correlation_tracker = correlation_tracker

        # Ensure tables exist
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        self.db.execute("""CREATE TABLE IF NOT EXISTS rl_bayesian_consensus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            belief_id TEXT NOT NULL,
            recommended_outcome TEXT NOT NULL,
            confidence REAL NOT NULL,
            agreement_level TEXT NOT NULL,
            agent_contributions TEXT,
            timestamp TEXT NOT NULL,
            actual_outcome TEXT,
            was_correct BOOLEAN
        )""")

        # Create indexes
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_consensus_belief_id "
            "ON rl_bayesian_consensus(belief_id)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_consensus_timestamp "
            "ON rl_bayesian_consensus(timestamp)"
        )

    def compute_consensus(
        self,
        belief_id: str,
        agent_messages: Dict[str, str],
        strategy: str = "weighted_bayesian",
    ) -> Dict[str, Any]:
        """Compute consensus from multiple agent opinions.

        Args:
            belief_id: Belief state for the task
            agent_messages: Dict mapping agent_id to message
            strategy: Consensus strategy ("majority_vote", "weighted_bayesian")

        Returns:
            Dict with consensus results
        """
        belief = self.orchestration_service.get_belief_state(belief_id)
        if not belief:
            logger.warning(f"Belief state {belief_id} not found")
            return None

        # Analyze messages
        agent_votes = self._analyze_messages(agent_messages)

        # Compute consensus based on strategy
        if strategy == "majority_vote":
            consensus_result = self._majority_vote_consensus(agent_votes)
        elif strategy == "weighted_bayesian":
            consensus_result = self._weighted_bayesian_consensus(
                belief_id, agent_votes, agent_messages
            )
        else:
            logger.warning(f"Unknown consensus strategy: {strategy}")
            consensus_result = self._majority_vote_consensus(agent_votes)

        # Determine agreement level
        agreement_level = self._compute_agreement_level(agent_votes)

        return {
            "recommended_outcome": consensus_result["outcome"],
            "confidence": consensus_result["confidence"],
            "agreement_level": agreement_level,
            "agent_contributions": consensus_result.get("contributions", {}),
            "strategy": strategy,
        }

    def compute_consensus_and_update_belief(
        self,
        belief_id: str,
        agent_messages: Dict[str, str],
        strategy: str = "weighted_bayesian",
    ) -> Dict[str, Any]:
        """Compute consensus and update belief state.

        Args:
            belief_id: Belief state to update
            agent_messages: Dict mapping agent_id to message
            strategy: Consensus strategy

        Returns:
            Dict with consensus results
        """
        # Compute consensus
        consensus = self.compute_consensus(
            belief_id=belief_id,
            agent_messages=agent_messages,
            strategy=strategy,
        )

        if not consensus:
            return None

        # Update belief with all agent messages
        for agent_id, message in agent_messages.items():
            # Estimate confidence (could be passed in)
            confidence = 0.8  # Default

            self.orchestration_service.update_belief_with_message(
                belief_id=belief_id,
                agent_id=agent_id,
                message=message,
                confidence=confidence,
            )

        return consensus

    def record_prediction_pairs(
        self,
        agent_messages: Dict[str, str],
        agent_votes: Dict[str, str],
        actual_outcome: str,
    ) -> None:
        """Record prediction pairs for correlation tracking.

        Args:
            agent_messages: Dict mapping agent_id to message
            agent_votes: Dict mapping agent_id to vote
            actual_outcome: Actual task outcome
        """
        if not self.correlation_tracker:
            return

        # Record all pairs
        agent_ids = list(agent_messages.keys())

        for i, agent_id_1 in enumerate(agent_ids):
            for agent_id_2 in agent_ids[i + 1 :]:
                # Get predictions (votes)
                prediction_1 = agent_votes.get(agent_id_1, "success")
                prediction_2 = agent_votes.get(agent_id_2, "success")

                # Record pair
                self.correlation_tracker.record_prediction_pair(
                    agent_id_1=agent_id_1,
                    agent_id_2=agent_id_2,
                    prediction_1=prediction_1,
                    prediction_2=prediction_2,
                    actual_outcome=actual_outcome,
                )

    def _analyze_messages(self, agent_messages: Dict[str, str]) -> Dict[str, str]:
        """Analyze agent messages to extract votes.

        Args:
            agent_messages: Dict mapping agent_id to message

        Returns:
            Dict mapping agent_id to vote ("success" or "failure")
        """
        agent_votes = {}

        for agent_id, message in agent_messages.items():
            message_lower = message.lower()

            # Check for success indicators
            success_keywords = [
                "yes",
                "correct",
                "right",
                "good",
                "success",
                "works",
                "will",
                "should",
            ]

            # Check for failure indicators
            failure_keywords = [
                "no",
                "wrong",
                "incorrect",
                "bad",
                "fail",
                "doesn't",
                "won't",
                "can't",
            ]

            success_count = sum(1 for kw in success_keywords if kw in message_lower)
            failure_count = sum(1 for kw in failure_keywords if kw in message_lower)

            if success_count > failure_count:
                agent_votes[agent_id] = "success"
            elif failure_count > success_count:
                agent_votes[agent_id] = "failure"
            else:
                # Default to success if unclear
                agent_votes[agent_id] = "success"

        return agent_votes

    def _majority_vote_consensus(self, agent_votes: Dict[str, str]) -> Dict[str, Any]:
        """Compute simple majority vote consensus.

        Args:
            agent_votes: Dict mapping agent_id to vote

        Returns:
            Dict with outcome and confidence
        """
        if not agent_votes:
            return {"outcome": "success", "confidence": 0.5}

        # Count votes
        success_votes = sum(1 for vote in agent_votes.values() if vote == "success")
        failure_votes = sum(1 for vote in agent_votes.values() if vote == "failure")
        total_votes = len(agent_votes)

        if success_votes > failure_votes:
            outcome = "success"
            confidence = success_votes / total_votes
        elif failure_votes > success_votes:
            outcome = "failure"
            confidence = failure_votes / total_votes
        else:
            # Tie - default to success with lower confidence
            outcome = "success"
            confidence = 0.5

        return {
            "outcome": outcome,
            "confidence": confidence,
        }

    def _weighted_bayesian_consensus(
        self,
        belief_id: str,
        agent_votes: Dict[str, str],
        agent_messages: Dict[str, str],
    ) -> Dict[str, Any]:
        """Compute reliability-weighted Bayesian consensus.

        Args:
            belief_id: Belief state for the task
            agent_votes: Dict mapping agent_id to vote
            agent_messages: Original messages

        Returns:
            Dict with outcome, confidence, and contributions
        """
        belief = self.orchestration_service.get_belief_state(belief_id)
        if not belief:
            return {"outcome": "success", "confidence": 0.5}

        # Get base reliability weights for all agents
        base_weights = {}
        for agent_id in agent_votes.keys():
            base_weights[agent_id] = (
                self.orchestration_service.reliability_learner.get_reliability_weight(agent_id)
            )

        # Adjust weights for correlations if tracker available
        if self.correlation_tracker:
            agent_ids = list(agent_votes.keys())
            adjusted_weights = self.correlation_tracker.get_adjusted_reliability_weights(
                agent_ids, base_weights
            )
        else:
            adjusted_weights = base_weights

        # Compute weighted scores
        contributions = {}
        weighted_success_score = 0.0
        weighted_failure_score = 0.0
        total_weight = 0.0

        for agent_id, vote in agent_votes.items():
            # Use adjusted weight
            weight = adjusted_weights.get(agent_id, 1.0)

            if vote == "success":
                weighted_success_score += weight
            else:
                weighted_failure_score += weight

            total_weight += weight

            contributions[agent_id] = {
                "vote": vote,
                "reliability": base_weights.get(agent_id, 1.0),
                "adjusted_weight": weight,
                "correlation_adjusted": self.correlation_tracker is not None,
            }

        # Determine outcome
        if weighted_success_score > weighted_failure_score:
            outcome = "success"
            confidence = weighted_success_score / total_weight if total_weight > 0 else 0.5
        elif weighted_failure_score > weighted_success_score:
            outcome = "failure"
            confidence = weighted_failure_score / total_weight if total_weight > 0 else 0.5
        else:
            # Tie
            outcome = "success"
            confidence = 0.5

        return {
            "outcome": outcome,
            "confidence": confidence,
            "contributions": contributions,
        }

    def _compute_agreement_level(self, agent_votes: Dict[str, str]) -> str:
        """Compute agreement level among agents.

        Args:
            agent_votes: Dict mapping agent_id to vote

        Returns:
            Agreement level: "unanimous", "partial", or "divergent"
        """
        if not agent_votes:
            return "unanimous"

        success_votes = sum(1 for vote in agent_votes.values() if vote == "success")
        total_votes = len(agent_votes)

        if success_votes == total_votes or success_votes == 0:
            return "unanimous"
        elif success_votes >= total_votes * 0.7 or success_votes <= total_votes * 0.3:
            return "partial"
        else:
            return "divergent"

    def record_consensus_outcome(
        self,
        belief_id: str,
        consensus: Dict[str, Any],
        actual_outcome: str,
        agent_messages: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record consensus outcome for learning.

        Args:
            belief_id: Belief state for the task
            consensus: Consensus result from compute_consensus
            actual_outcome: Actual task outcome
            agent_messages: Optional agent messages for correlation tracking
        """
        was_correct = consensus["recommended_outcome"] == actual_outcome

        import json

        self.db.execute(
            """INSERT INTO rl_bayesian_consensus
               (belief_id, recommended_outcome, confidence, agreement_level, agent_contributions, timestamp, actual_outcome, was_correct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                belief_id,
                consensus["recommended_outcome"],
                consensus["confidence"],
                consensus["agreement_level"],
                json_dumps(consensus.get("agent_contributions", {})),
                datetime.now().isoformat(),
                actual_outcome,
                was_correct,
            ),
        )

        logger.info(
            f"Recorded consensus outcome: recommended={consensus['recommended_outcome']}, "
            f"actual={actual_outcome}, correct={was_correct}"
        )

        # Record prediction pairs for correlation tracking
        if agent_messages and self.correlation_tracker:
            # Extract votes from agent contributions
            agent_votes = {}
            for agent_id, contribution in consensus.get("agent_contributions", {}).items():
                agent_votes[agent_id] = contribution.get("vote", "success")

            # Record pairs
            self.record_prediction_pairs(
                agent_messages=agent_messages,
                agent_votes=agent_votes,
                actual_outcome=actual_outcome,
            )

    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get consensus statistics.

        Returns:
            Dict with aggregate consensus metrics
        """
        cursor = self.db.execute("""SELECT
                  COUNT(*) as total_consensus,
                  SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct_count,
                  AVG(confidence) as avg_confidence,
                  AVG(CASE WHEN was_correct THEN 1 ELSE 0 END) as accuracy
               FROM rl_bayesian_consensus
            """)

        result = cursor.fetchone()

        if not result or result[0] == 0:
            return {
                "total_consensus": 0,
                "correct_count": 0,
                "accuracy": 0.0,
                "avg_confidence": 0.0,
            }

        total, correct, avg_confidence, accuracy = result

        return {
            "total_consensus": total,
            "correct_count": correct,
            "accuracy": accuracy if accuracy else 0.0,
            "avg_confidence": avg_confidence if avg_confidence else 0.0,
        }

    def get_agent_consensus_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get consensus statistics for a specific agent.

        Args:
            agent_id: Agent to get statistics for

        Returns:
            Dict with agent's consensus metrics
        """
        # This would require parsing agent_contributions JSON
        # For now, return a simple implementation
        cursor = self.db.execute(
            """SELECT COUNT(*) as total
               FROM rl_bayesian_consensus
               WHERE agent_contributions LIKE ?
            """,
            (f'%"{agent_id}"%',),
        )

        result = cursor.fetchone()

        if not result:
            return {"participation_count": 0}

        return {
            "participation_count": result[0],
        }
