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

"""Bayesian orchestration monitoring and metrics visualization.

This module provides monitoring capabilities for the Bayesian orchestration
system, including metrics collection, visualization, and alerting.

Based on: "Position: agentic AI orchestration should be Bayes-consistent"
(arXiv:2605.00742, ICML 2026)
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BayesianMetricsMonitor:
    """Monitor and visualize Bayesian orchestration metrics.

    Provides real-time monitoring of:
    - Belief state evolution
    - Agent reliability trends
    - Observation model calibration
    - Value of Information accuracy
    - Consensus performance
    - Correlation detection
    """

    def __init__(self, db_connection: sqlite3.Connection):
        """Initialize metrics monitor.

        Args:
            db_connection: SQLite database connection
        """
        self.db = db_connection

    def get_belief_evolution(self, belief_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get belief state evolution over time.

        Args:
            belief_id: Belief state identifier
            limit: Maximum number of records to retrieve

        Returns:
            List of belief state snapshots
        """
        cursor = self.db.execute(
            """SELECT success_prob, failure_prob, entropy, agent_id, message, timestamp
               FROM rl_belief_history
               WHERE belief_id = ?
               ORDER BY timestamp ASC
               LIMIT ?
            """,
            (belief_id, limit),
        )

        evolution = []
        for row in cursor:
            evolution.append(
                {
                    "success_prob": row[0],
                    "failure_prob": row[1],
                    "entropy": row[2],
                    "agent_id": row[3],
                    "message": row[4],
                    "timestamp": row[5],
                }
            )

        return evolution

    def get_reliability_trends(
        self, agent_ids: Optional[List[str]] = None, days: int = 7
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get reliability weight trends for agents.

        Args:
            agent_ids: List of agent IDs, or None for all agents
            days: Number of days to look back

        Returns:
            Dict mapping agent_id to list of reliability snapshots
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        if agent_ids:
            placeholders = ",".join("?" * len(agent_ids))
            cursor = self.db.execute(
                f"""SELECT agent_id, alpha_reliability, beta_reliability, sample_count, last_updated
                   FROM rl_agent_reliability
                   WHERE agent_id IN ({placeholders})
                   AND last_updated >= ?
                   ORDER BY last_updated ASC
                """,
                agent_ids + [cutoff_date],
            )
        else:
            cursor = self.db.execute(
                """SELECT agent_id, alpha_reliability, beta_reliability, sample_count, last_updated
                   FROM rl_agent_reliability
                   WHERE last_updated >= ?
                   ORDER BY last_updated ASC
                """,
                (cutoff_date,),
            )

        trends = {}
        for row in cursor:
            agent_id = row[0]
            if agent_id not in trends:
                trends[agent_id] = []

            alpha, beta = row[1], row[2]
            expected_reliability = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5

            trends[agent_id].append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "expected_reliability": expected_reliability,
                    "sample_count": row[3],
                    "timestamp": row[4],
                }
            )

        return trends

    def get_observation_model_calibration(
        self, agent_id: Optional[str] = None, days: int = 7
    ) -> Dict[str, Any]:
        """Get observation model calibration statistics.

        Args:
            agent_id: Specific agent ID, or None for all agents
            days: Number of days to look back

        Returns:
            Calibration statistics
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        if agent_id:
            cursor = self.db.execute(
                """SELECT message_category, outcome_type, alpha, beta, sample_count
                   FROM rl_observation_model
                   WHERE agent_id = ? AND last_updated >= ?
                   ORDER BY agent_id, message_category, outcome_type
                """,
                (agent_id, cutoff_date),
            )
        else:
            cursor = self.db.execute(
                """SELECT agent_id, message_category, outcome_type, alpha, beta, sample_count
                   FROM rl_observation_model
                   WHERE last_updated >= ?
                   ORDER BY agent_id, message_category, outcome_type
                """,
                (cutoff_date,),
            )

        calibration = {}
        for row in cursor:
            if agent_id:
                key = (agent_id, row[1], row[2])
            else:
                key = (row[0], row[1], row[2])

            alpha, beta = row[3], row[4]
            expected_value = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5

            calibration[key] = {
                "alpha": alpha,
                "beta": beta,
                "expected_value": expected_value,
                "sample_count": row[5],
            }

        return calibration

    def get_voi_statistics(self, agent_id: Optional[str] = None, days: int = 7) -> Dict[str, Any]:
        """Get Value of Information statistics.

        Args:
            agent_id: Specific agent ID, or None for all agents
            days: Number of days to look back

        Returns:
            VoI statistics
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            if agent_id:
                cursor = self.db.execute(
                    """SELECT agent_id, predicted_voi, actual_information_gain, query_cost, was_beneficial
                       FROM rl_voi_history
                       WHERE agent_id = ? AND timestamp >= ?
                       ORDER BY timestamp DESC
                    """,
                    (agent_id, cutoff_date),
                )
            else:
                cursor = self.db.execute(
                    """SELECT agent_id, predicted_voi, actual_information_gain, query_cost, was_beneficial
                       FROM rl_voi_history
                       WHERE timestamp >= ?
                       ORDER BY timestamp DESC
                    """,
                    (cutoff_date,),
                )

            stats = {
                "total_queries": 0,
                "beneficial_queries": 0,
                "mean_predicted_voi": 0.0,
                "mean_actual_gain": 0.0,
                "mean_query_cost": 0.0,
                "beneficial_rate": 0.0,
                "per_agent": {},
            }

            for row in cursor:
                agent_id = row[0]
                predicted_voi = row[1]
                actual_gain = row[2]
                query_cost = row[3]
                was_beneficial = row[4]

                stats["total_queries"] += 1
                if was_beneficial:
                    stats["beneficial_queries"] += 1

                stats["mean_predicted_voi"] += predicted_voi
                stats["mean_actual_gain"] += actual_gain
                stats["mean_query_cost"] += query_cost

                if agent_id not in stats["per_agent"]:
                    stats["per_agent"][agent_id] = {
                        "total_queries": 0,
                        "beneficial_queries": 0,
                        "mean_predicted_voi": 0.0,
                        "mean_actual_gain": 0.0,
                    }

                agent_stats = stats["per_agent"][agent_id]
                agent_stats["total_queries"] += 1
                if was_beneficial:
                    agent_stats["beneficial_queries"] += 1
                agent_stats["mean_predicted_voi"] += predicted_voi
                agent_stats["mean_actual_gain"] += actual_gain

            # Compute means
            if stats["total_queries"] > 0:
                stats["mean_predicted_voi"] /= stats["total_queries"]
                stats["mean_actual_gain"] /= stats["total_queries"]
                stats["mean_query_cost"] /= stats["total_queries"]
                stats["beneficial_rate"] = stats["beneficial_queries"] / stats["total_queries"]

            for agent_id, agent_stats in stats["per_agent"].items():
                if agent_stats["total_queries"] > 0:
                    agent_stats["mean_predicted_voi"] /= agent_stats["total_queries"]
                    agent_stats["mean_actual_gain"] /= agent_stats["total_queries"]
                    agent_stats["beneficial_rate"] = (
                        agent_stats["beneficial_queries"] / agent_stats["total_queries"]
                    )

            return stats
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return {
                "total_queries": 0,
                "beneficial_queries": 0,
                "mean_predicted_voi": 0.0,
                "mean_actual_gain": 0.0,
                "mean_query_cost": 0.0,
                "beneficial_rate": 0.0,
                "per_agent": {},
            }

    def get_consensus_performance(self, days: int = 7) -> Dict[str, Any]:
        """Get consensus formation performance.

        Args:
            days: Number of days to look back

        Returns:
            Consensus performance statistics
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            cursor = self.db.execute(
                """SELECT recommended_outcome, confidence, agreement_level, was_correct
                   FROM rl_bayesian_consensus
                   WHERE timestamp >= ?
                   ORDER BY timestamp DESC
                """,
                (cutoff_date,),
            )

            stats = {
                "total_consensus": 0,
                "correct_count": 0,
                "accuracy": 0.0,
                "mean_confidence": 0.0,
                "agreement_distribution": {
                    "unanimous": 0,
                    "partial": 0,
                    "divergent": 0,
                },
                "by_agreement": {
                    "unanimous": {"total": 0, "correct": 0},
                    "partial": {"total": 0, "correct": 0},
                    "divergent": {"total": 0, "correct": 0},
                },
            }

            for row in cursor:
                confidence = row[1]
                agreement_level = row[2]
                was_correct = row[3]

                stats["total_consensus"] += 1
                if was_correct:
                    stats["correct_count"] += 1

                stats["mean_confidence"] += confidence
                stats["agreement_distribution"][agreement_level] += 1

                stats["by_agreement"][agreement_level]["total"] += 1
                if was_correct:
                    stats["by_agreement"][agreement_level]["correct"] += 1

            # Compute means
            if stats["total_consensus"] > 0:
                stats["accuracy"] = stats["correct_count"] / stats["total_consensus"]
                stats["mean_confidence"] /= stats["total_consensus"]

            return stats
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return {
                "total_consensus": 0,
                "correct_count": 0,
                "accuracy": 0.0,
                "mean_confidence": 0.0,
                "agreement_distribution": {
                    "unanimous": 0,
                    "partial": 0,
                    "divergent": 0,
                },
                "by_agreement": {
                    "unanimous": {"total": 0, "correct": 0},
                    "partial": {"total": 0, "correct": 0},
                    "divergent": {"total": 0, "correct": 0},
                },
            }

    def get_correlation_matrix(
        self, agent_ids: List[str], days: int = 7
    ) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix for agents.

        Args:
            agent_ids: List of agent IDs
            days: Number of days to look back

        Returns:
            Correlation matrix
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            matrix = {}
            for i, agent_id_1 in enumerate(agent_ids):
                matrix[agent_id_1] = {}
                for agent_id_2 in agent_ids:
                    cursor = self.db.execute(
                        """SELECT correlation_coefficient
                           FROM rl_agent_correlations
                           WHERE ((agent_id_1 = ? AND agent_id_2 = ?) OR (agent_id_1 = ? AND agent_id_2 = ?))
                           AND last_updated >= ?
                        """,
                        (agent_id_1, agent_id_2, agent_id_2, agent_id_1, cutoff_date),
                    )

                    row = cursor.fetchone()
                    if row:
                        matrix[agent_id_1][agent_id_2] = row[0]
                    else:
                        matrix[agent_id_1][agent_id_2] = 0.0

            return matrix
        except sqlite3.OperationalError:
            # Table doesn't exist yet, return identity matrix (no correlation)
            matrix = {}
            for agent_id_1 in agent_ids:
                matrix[agent_id_1] = {}
                for agent_id_2 in agent_ids:
                    # Identity matrix: 1.0 on diagonal, 0.0 elsewhere
                    matrix[agent_id_1][agent_id_2] = 1.0 if agent_id_1 == agent_id_2 else 0.0
            return matrix

    def get_system_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive system summary.

        Args:
            days: Number of days to look back

        Returns:
            System summary statistics
        """
        summary = {
            "period_days": days,
            "belief_states": self._count_belief_states(days),
            "agent_reliability": self._count_agent_reliability(days),
            "observation_models": self._count_observation_models(days),
            "voi_queries": self._count_voi_queries(days),
            "consensus_decisions": self._count_consensus_decisions(days),
            "correlations": self._count_correlations(days),
        }

        return summary

    def _count_belief_states(self, days: int) -> Dict[str, int]:
        """Count belief state records."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            cursor = self.db.execute(
                """SELECT COUNT(DISTINCT belief_id) FROM rl_belief_history WHERE timestamp >= ?""",
                (cutoff_date,),
            )

            return {"unique_belief_states": cursor.fetchone()[0]}
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return {"unique_belief_states": 0}

    def _count_agent_reliability(self, days: int) -> Dict[str, int]:
        """Count agent reliability records."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            cursor = self.db.execute(
                """SELECT COUNT(DISTINCT agent_id) FROM rl_agent_reliability WHERE last_updated >= ?""",
                (cutoff_date,),
            )

            return {"tracked_agents": cursor.fetchone()[0]}
        except sqlite3.OperationalError:
            return {"tracked_agents": 0}

    def _count_observation_models(self, days: int) -> Dict[str, int]:
        """Count observation model records."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            cursor = self.db.execute(
                """SELECT COUNT(*) FROM rl_observation_model WHERE last_updated >= ?""",
                (cutoff_date,),
            )

            return {"observation_records": cursor.fetchone()[0]}
        except sqlite3.OperationalError:
            return {"observation_records": 0}

    def _count_voi_queries(self, days: int) -> Dict[str, int]:
        """Count VoI query records."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            cursor = self.db.execute(
                """SELECT COUNT(*) FROM rl_voi_history WHERE timestamp >= ?""",
                (cutoff_date,),
            )

            return {"voi_queries": cursor.fetchone()[0]}
        except sqlite3.OperationalError:
            return {"voi_queries": 0}

    def _count_consensus_decisions(self, days: int) -> Dict[str, int]:
        """Count consensus decision records."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            cursor = self.db.execute(
                """SELECT COUNT(*) FROM rl_bayesian_consensus WHERE timestamp >= ?""",
                (cutoff_date,),
            )

            return {"consensus_decisions": cursor.fetchone()[0]}
        except sqlite3.OperationalError:
            return {"consensus_decisions": 0}

    def _count_correlations(self, days: int) -> Dict[str, int]:
        """Count correlation records."""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            cursor = self.db.execute(
                """SELECT COUNT(*) FROM rl_agent_correlations WHERE last_updated >= ?""",
                (cutoff_date,),
            )

            return {"correlation_pairs": cursor.fetchone()[0]}
        except sqlite3.OperationalError:
            return {"correlation_pairs": 0}


class MetricsExporter:
    """Export metrics for external visualization tools."""

    def __init__(self, monitor: BayesianMetricsMonitor):
        """Initialize metrics exporter.

        Args:
            monitor: Metrics monitor instance
        """
        self.monitor = monitor

    def export_belief_evolution_csv(
        self, belief_id: str, output_path: str, limit: int = 100
    ) -> None:
        """Export belief evolution to CSV.

        Args:
            belief_id: Belief state identifier
            output_path: Output CSV file path
            limit: Maximum records to export
        """
        evolution = self.monitor.get_belief_evolution(belief_id, limit)

        with open(output_path, "w") as f:
            f.write("timestamp,success_prob,failure_prob,entropy,agent_id,message\n")

            for snapshot in evolution:
                f.write(
                    f"{snapshot['timestamp']},"
                    f"{snapshot['success_prob']:.4f},"
                    f"{snapshot['failure_prob']:.4f},"
                    f"{snapshot['entropy']:.4f},"
                    f"{snapshot['agent_id'] or ''},"
                    f'"{snapshot["message"] or ""}"\n'
                )

        logger.info(f"Exported belief evolution to {output_path}")

    def export_reliability_trends_csv(
        self, output_path: str, agent_ids: Optional[List[str]] = None, days: int = 7
    ) -> None:
        """Export reliability trends to CSV.

        Args:
            output_path: Output CSV file path
            agent_ids: List of agent IDs, or None for all agents
            days: Number of days to look back
        """
        trends = self.monitor.get_reliability_trends(agent_ids, days)

        with open(output_path, "w") as f:
            f.write("agent_id,timestamp,alpha,beta,expected_reliability,sample_count\n")

            for agent_id, snapshots in trends.items():
                for snapshot in snapshots:
                    f.write(
                        f"{agent_id},"
                        f"{snapshot['timestamp']},"
                        f"{snapshot['alpha']:.4f},"
                        f"{snapshot['beta']:.4f},"
                        f"{snapshot['expected_reliability']:.4f},"
                        f"{snapshot['sample_count']}\n"
                    )

        logger.info(f"Exported reliability trends to {output_path}")

    def export_summary_json(self, output_path: str, days: int = 7) -> None:
        """Export system summary to JSON.

        Args:
            output_path: Output JSON file path
            days: Number of days to look back
        """
        import json

        summary = {
            "summary": self.monitor.get_system_summary(days),
            "consensus_performance": self.monitor.get_consensus_performance(days),
            "voi_statistics": self.monitor.get_voi_statistics(None, days),
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Exported system summary to {output_path}")


class ASCIIChartRenderer:
    """Render ASCII charts for terminal visualization."""

    @staticmethod
    def render_bar_chart(
        data: Dict[str, float],
        title: str,
        width: int = 50,
        height: int = 10,
    ) -> str:
        """Render horizontal bar chart.

        Args:
            data: Dict mapping label to value
            title: Chart title
            width: Chart width
            height: Chart height

        Returns:
            ASCII chart string
        """
        if not data:
            return f"{title}\n(No data)\n"

        max_value = max(data.values()) if data.values() else 1.0
        lines = [f"{title}", "=" * width]

        for label, value in sorted(data.items(), key=lambda x: x[1], reverse=True):
            bar_length = int((value / max_value) * (width - 20))
            bar = "█" * bar_length
            lines.append(f"{label:15s} |{bar:30s}| {value:.4f}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def render_line_chart(
        data: List[Tuple[str, float]],
        title: str,
        width: int = 60,
        height: int = 10,
    ) -> str:
        """Render line chart.

        Args:
            data: List of (label, value) tuples
            title: Chart title
            width: Chart width
            height: Chart height

        Returns:
            ASCII chart string
        """
        if not data:
            return f"{title}\n(No data)\n"

        # Extract values
        labels = [d[0] for d in data]
        values = [d[1] for d in data]

        min_value = min(values)
        max_value = max(values)
        value_range = max_value - min_value if max_value != min_value else 1.0

        lines = [f"{title}"]
        lines.append("=" * width)

        # Create chart rows
        for row in range(height, 0, -1):
            y_value = min_value + (row / height) * value_range
            line = f"{y_value:8.3f} |"

            for label, value in data:
                # Check if value is at this row
                if abs(value - y_value) < value_range / height:
                    line += "█"
                else:
                    line += " "

            lines.append(line)

        # Add x-axis labels (simplified)
        if len(labels) <= 10:
            label_line = "         |" + " ".join(labels[:10])
            lines.append(label_line)

        return "\n".join(lines) + "\n"

    @staticmethod
    def render_heatmap(
        matrix: Dict[str, Dict[str, float]],
        title: str,
    ) -> str:
        """Render correlation matrix heatmap.

        Args:
            matrix: Correlation matrix
            title: Chart title

        Returns:
            ASCII heatmap string
        """
        if not matrix:
            return f"{title}\n(No data)\n"

        agents = list(matrix.keys())
        lines = [f"{title}"]
        lines.append("=" * 60)

        # Header row
        header = "         |"
        for agent in agents[:8]:  # Limit to 8 agents for readability
            header += f"{agent[0]:3}|"
        lines.append(header)

        # Data rows
        for agent_1 in agents[:8]:
            row = f"{agent_1:8} |"
            for agent_2 in agents[:8]:
                correlation = matrix[agent_1].get(agent_2, 0.0)

                # Convert correlation to character
                if correlation >= 0.8:
                    ch = "█"  # Strong positive
                elif correlation >= 0.5:
                    ch = "▓"  # Moderate positive
                elif correlation >= 0.0:
                    ch = "▒"  # Weak positive
                elif correlation >= -0.5:
                    ch = "░"  # Weak negative
                else:
                    ch = " "  # Strong negative

                row += f"{ch}|"

            lines.append(row)

        # Legend
        lines.append("\nLegend: █ Strong+  ▓ Mod+  ▒ Weak+  ░ Weak-    Strong-")

        return "\n".join(lines) + "\n"
