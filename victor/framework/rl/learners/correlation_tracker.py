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

"""Correlation tracking for dependence-aware pooling.

This module provides correlation tracking between agent predictions
to enable dependence-aware Bayesian consensus formation.

When agents' predictions are correlated, their messages don't provide
independent evidence. This tracker detects and adjusts for such
correlations to avoid overcounting evidence.

Based on: "Position: agentic AI orchestration should be Bayes-consistent"
(arXiv:2605.00742, ICML 2026)
"""

import logging
import sqlite3
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CorrelationTracker:
    """Track correlations between agent predictions for dependence-aware pooling.

    Maintains pairwise correlation statistics between agents and adjusts
    effective sample sizes in consensus based on detected correlations.

    Attributes:
        db: SQLite database connection
        name: Tracker name for identification
    """

    def __init__(self, name: str, db_connection: sqlite3.Connection):
        """Initialize correlation tracker.

        Args:
            name: Tracker name for identification
            db_connection: SQLite database connection
        """
        self.name = name
        self.db = db_connection

        # Ensure tables exist
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        self.db.execute("""CREATE TABLE IF NOT EXISTS rl_agent_correlations (
            agent_id_1 TEXT NOT NULL,
            agent_id_2 TEXT NOT NULL,
            agreement_count INTEGER NOT NULL,
            disagreement_count INTEGER NOT NULL,
            total_pairs INTEGER NOT NULL,
            correlation_coefficient REAL NOT NULL,
            last_updated TEXT NOT NULL,
            PRIMARY KEY (agent_id_1, agent_id_2)
        )""")

        # Create indexes
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_correlation_agent1 "
            "ON rl_agent_correlations(agent_id_1)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_correlation_agent2 "
            "ON rl_agent_correlations(agent_id_2)"
        )

    def record_prediction_pair(
        self,
        agent_id_1: str,
        agent_id_2: str,
        prediction_1: str,
        prediction_2: str,
        actual_outcome: str,
    ) -> None:
        """Record a pair of predictions from two agents.

        Args:
            agent_id_1: First agent ID
            agent_id_2: Second agent ID
            prediction_1: First agent's prediction
            prediction_2: Second agent's prediction
            actual_outcome: Actual task outcome
        """
        # Normalize order (lexicographic to avoid duplicates)
        if agent_id_1 > agent_id_2:
            agent_id_1, agent_id_2 = agent_id_2, agent_id_1
            prediction_1, prediction_2 = prediction_2, prediction_1

        # Check if predictions agree
        agreement = prediction_1 == prediction_2

        # Update database
        cursor = self.db.execute(
            """SELECT agreement_count, disagreement_count, total_pairs
               FROM rl_agent_correlations
               WHERE agent_id_1 = ? AND agent_id_2 = ?
            """,
            (agent_id_1, agent_id_2),
        )

        row = cursor.fetchone()

        if row:
            # Update existing record
            agreement_count, disagreement_count, total_pairs = row

            if agreement:
                agreement_count += 1
            else:
                disagreement_count += 1

            total_pairs += 1

            # Recalculate correlation coefficient
            correlation = self._compute_correlation_coefficient(
                agreement_count, disagreement_count, total_pairs
            )

            self.db.execute(
                """UPDATE rl_agent_correlations
                   SET agreement_count = ?,
                       disagreement_count = ?,
                       total_pairs = ?,
                       correlation_coefficient = ?,
                       last_updated = ?
                   WHERE agent_id_1 = ? AND agent_id_2 = ?
                """,
                (
                    agreement_count,
                    disagreement_count,
                    total_pairs,
                    correlation,
                    datetime.now().isoformat(),
                    agent_id_1,
                    agent_id_2,
                ),
            )
        else:
            # Insert new record
            agreement_count = 1 if agreement else 0
            disagreement_count = 0 if agreement else 1
            total_pairs = 1

            correlation = self._compute_correlation_coefficient(
                agreement_count, disagreement_count, total_pairs
            )

            self.db.execute(
                """INSERT INTO rl_agent_correlations
                   (agent_id_1, agent_id_2, agreement_count, disagreement_count,
                    total_pairs, correlation_coefficient, last_updated)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id_1,
                    agent_id_2,
                    agreement_count,
                    disagreement_count,
                    total_pairs,
                    correlation,
                    datetime.now().isoformat(),
                ),
            )

        logger.debug(
            f"Recorded prediction pair: {agent_id_1} vs {agent_id_2}, "
            f"agreement={agreement}, correlation={correlation:.3f}"
        )

    def _compute_correlation_coefficient(
        self, agreement_count: int, disagreement_count: int, total_pairs: int
    ) -> float:
        """Compute correlation coefficient from agreement statistics.

        Uses Pearson correlation for binary predictions.
        correlation = (agreements - disagreements) / total_pairs

        Args:
            agreement_count: Number of agreeing pairs
            disagreement_count: Number of disagreeing pairs
            total_pairs: Total number of pairs

        Returns:
            Correlation coefficient in [-1, 1]
        """
        if total_pairs == 0:
            return 0.0

        # Pearson correlation for binary variables
        correlation = (agreement_count - disagreement_count) / total_pairs

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, correlation))

    def get_correlation(self, agent_id_1: str, agent_id_2: str) -> float:
        """Get correlation coefficient between two agents.

        Args:
            agent_id_1: First agent ID
            agent_id_2: Second agent ID

        Returns:
            Correlation coefficient in [-1, 1], or 0.0 if not found
        """
        # Normalize order
        if agent_id_1 > agent_id_2:
            agent_id_1, agent_id_2 = agent_id_2, agent_id_1

        cursor = self.db.execute(
            "SELECT correlation_coefficient FROM rl_agent_correlations "
            "WHERE agent_id_1 = ? AND agent_id_2 = ?",
            (agent_id_1, agent_id_2),
        )

        row = cursor.fetchone()

        if row:
            return row[0]
        else:
            return 0.0  # No correlation data, assume independent

    def get_correlation_matrix(self, agent_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix for a set of agents.

        Args:
            agent_ids: List of agent IDs

        Returns:
            Dict mapping (agent_id_1, agent_id_2) to correlation coefficient
        """
        matrix = defaultdict(dict)

        for i, agent_id_1 in enumerate(agent_ids):
            for agent_id_2 in agent_ids[i + 1 :]:
                correlation = self.get_correlation(agent_id_1, agent_id_2)
                matrix[agent_id_1][agent_id_2] = correlation
                matrix[agent_id_2][agent_id_1] = correlation

            # Self-correlation is 1.0
            matrix[agent_id_1][agent_id_1] = 1.0

        return dict(matrix)

    def compute_effective_sample_size(
        self, agent_ids: List[str], reliability_weights: Dict[str, float]
    ) -> float:
        """Compute effective sample size accounting for correlations.

        When agents are correlated, the effective sample size is smaller
        than the nominal sample size.

        ESS = sum(w_i) / (1 + sum_{i<j} 2*w_i*w_j*correlation_{ij} / sum(w))

        Args:
            agent_ids: List of agent IDs
            reliability_weights: Dict mapping agent_id to reliability weight

        Returns:
            Effective sample size
        """
        if not agent_ids:
            return 0.0

        # Sum of weights
        weight_sum = sum(reliability_weights.get(agent_id, 1.0) for agent_id in agent_ids)

        if weight_sum == 0:
            return 0.0

        # Sum of correlated weights
        correlated_sum = 0.0

        for i, agent_id_1 in enumerate(agent_ids):
            for agent_id_2 in agent_ids[i + 1 :]:
                weight_1 = reliability_weights.get(agent_id_1, 1.0)
                weight_2 = reliability_weights.get(agent_id_2, 1.0)
                correlation = self.get_correlation(agent_id_1, agent_id_2)

                correlated_sum += 2 * weight_1 * weight_2 * correlation

        # Effective sample size
        ess = weight_sum / (1 + correlated_sum / weight_sum) if weight_sum > 0 else 0.0

        return max(0.0, ess)  # Ensure non-negative

    def get_adjusted_reliability_weights(
        self,
        agent_ids: List[str],
        base_reliability_weights: Dict[str, float],
        correlation_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Get reliability weights adjusted for correlations.

        Downweights agents that are highly correlated with others
        to avoid overcounting evidence.

        Args:
            agent_ids: List of agent IDs
            base_reliability_weights: Base reliability weights
            correlation_threshold: Threshold for considering agents correlated

        Returns:
            Adjusted reliability weights
        """
        if not agent_ids:
            return {}

        adjusted_weights = {}

        for agent_id in agent_ids:
            # Start with base weight
            base_weight = base_reliability_weights.get(agent_id, 1.0)
            adjusted_weight = base_weight

            # Check correlations with other agents
            for other_agent_id in agent_ids:
                if agent_id == other_agent_id:
                    continue

                correlation = self.get_correlation(agent_id, other_agent_id)

                # If highly correlated, downweight
                if abs(correlation) > correlation_threshold:
                    # Downweight by (1 - |correlation|)
                    adjusted_weight *= 1.0 - abs(correlation)

            adjusted_weights[agent_id] = max(0.1, adjusted_weight)  # Minimum weight

        return adjusted_weights

    def get_correlation_stats(self, agent_id: Optional[str] = None) -> Dict[str, Dict]:
        """Get correlation statistics for agents.

        Args:
            agent_id: Specific agent ID, or None for all agents

        Returns:
            Dict mapping agent_id to correlation statistics
        """
        if agent_id:
            cursor = self.db.execute(
                """SELECT agent_id_2, agreement_count, disagreement_count,
                          total_pairs, correlation_coefficient
                   FROM rl_agent_correlations
                   WHERE agent_id_1 = ?
                   UNION ALL
                   SELECT agent_id_1, agreement_count, disagreement_count,
                          total_pairs, correlation_coefficient
                   FROM rl_agent_correlations
                   WHERE agent_id_2 = ?
                """,
                (agent_id, agent_id),
            )
        else:
            cursor = self.db.execute("""SELECT agent_id_1, agent_id_2, agreement_count,
                          disagreement_count, total_pairs, correlation_coefficient
                   FROM rl_agent_correlations
                """)

        stats = defaultdict(lambda: {"correlations": {}})

        for row in cursor:
            if agent_id:
                # Single agent query
                other_agent, agr_count, disagr_count, total, corr = row
                stats[agent_id]["correlations"][other_agent] = {
                    "correlation": corr,
                    "agreement_rate": agr_count / total if total > 0 else 0.0,
                    "total_pairs": total,
                }
            else:
                # All agents query
                agent_1, agent_2, agr_count, disagr_count, total, corr = row

                stats[agent_1]["correlations"][agent_2] = {
                    "correlation": corr,
                    "agreement_rate": agr_count / total if total > 0 else 0.0,
                    "total_pairs": total,
                }

                stats[agent_2]["correlations"][agent_1] = {
                    "correlation": corr,
                    "agreement_rate": agr_count / total if total > 0 else 0.0,
                    "total_pairs": total,
                }

        return dict(stats)

    def get_highly_correlated_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Get pairs of agents with high correlation.

        Args:
            threshold: Correlation threshold (default: 0.7)

        Returns:
            List of (agent_id_1, agent_id_2, correlation) tuples
        """
        cursor = self.db.execute(
            """SELECT agent_id_1, agent_id_2, correlation_coefficient
                   FROM rl_agent_correlations
                   WHERE ABS(correlation_coefficient) >= ?
                   ORDER BY ABS(correlation_coefficient) DESC
                """,
            (threshold,),
        )

        return cursor.fetchall()

    def cleanup_old_correlations(self, days_old: int = 30) -> int:
        """Clean up correlation data older than specified days.

        Args:
            days_old: Age in days (default: 30)

        Returns:
            Number of records cleaned up
        """
        cursor = self.db.execute(
            """DELETE FROM rl_agent_correlations
                   WHERE date(last_updated) < date('now', '-' || ? || ' days')
                """,
            (str(days_old),),
        )

        deleted_count = cursor.rowcount

        logger.info(f"Cleaned up {deleted_count} old correlation records")

        return deleted_count
