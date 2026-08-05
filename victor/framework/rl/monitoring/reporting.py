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

"""Shared reporting helpers for Bayesian orchestration monitoring surfaces."""

from __future__ import annotations

import sqlite3
from typing import Optional, Sequence

from victor.core.database import get_database
from victor.framework.rl.monitoring.bayesian_monitor import (
    ASCIIChartRenderer,
    BayesianMetricsMonitor,
    MetricsExporter,
)

DEFAULT_BAYESIAN_LOOKBACK_DAYS = 7
SECTION_WIDTH = 60


def parse_agent_ids(raw_agent_ids: Optional[str]) -> Optional[list[str]]:
    """Parse a comma-separated agent ID list."""
    if raw_agent_ids is None:
        return None
    agent_ids = [agent_id.strip() for agent_id in raw_agent_ids.split(",") if agent_id.strip()]
    return agent_ids or None


class BayesianMonitoringService:
    """Shared formatter and export surface for Bayesian monitoring."""

    def __init__(self, db_connection: sqlite3.Connection):
        self.monitor = BayesianMetricsMonitor(db_connection)
        self.exporter = MetricsExporter(self.monitor)
        self.renderer = ASCIIChartRenderer()

    def render_summary(self, days: int = DEFAULT_BAYESIAN_LOOKBACK_DAYS) -> str:
        """Render a system summary."""
        system_summary = self.monitor.get_system_summary(days)
        lines = self._header(f"Bayesian Orchestration System Summary (Last {days} days)")
        lines.extend(
            [
                "Belief States:",
                f"  Unique belief states: {system_summary['belief_states']['unique_belief_states']}",
                "",
                "Agent Reliability:",
                f"  Tracked agents: {system_summary['agent_reliability']['tracked_agents']}",
                "",
                "Observation Models:",
                f"  Observation records: {system_summary['observation_models']['observation_records']}",
                "",
                "Value of Information:",
                f"  VoI queries: {system_summary['voi_queries']['voi_queries']}",
                "",
                "Consensus Decisions:",
                f"  Consensus decisions: {system_summary['consensus_decisions']['consensus_decisions']}",
                "",
                "Correlations:",
                f"  Correlation pairs: {system_summary['correlations']['correlation_pairs']}",
            ]
        )
        return "\n".join(lines)

    def render_reliability(
        self,
        agent_ids: Optional[Sequence[str]] = None,
        days: int = DEFAULT_BAYESIAN_LOOKBACK_DAYS,
    ) -> str:
        """Render agent reliability trends."""
        trends = self.monitor.get_reliability_trends(
            list(agent_ids) if agent_ids is not None else None,
            days,
        )
        if not trends:
            return "No reliability data found."

        lines = self._header(f"Agent Reliability (Last {days} days)")
        chart_data: dict[str, float] = {}
        for agent_id, snapshots in sorted(trends.items()):
            if not snapshots:
                continue
            latest = snapshots[-1]
            chart_data[agent_id] = latest["expected_reliability"]
            lines.extend(
                [
                    f"{agent_id}:",
                    f"  Expected reliability: {latest['expected_reliability']:.2%}",
                    f"  Sample count: {latest['sample_count']}",
                    f"  Alpha: {latest['alpha']:.2f}, Beta: {latest['beta']:.2f}",
                ]
            )

        lines.append("")
        lines.append(self.renderer.render_bar_chart(chart_data, "Reliability Comparison").rstrip())
        return "\n".join(lines)

    def render_consensus(self, days: int = DEFAULT_BAYESIAN_LOOKBACK_DAYS) -> str:
        """Render consensus performance statistics."""
        stats = self.monitor.get_consensus_performance(days)
        lines = self._header(f"Consensus Performance (Last {days} days)")
        lines.extend(
            [
                f"Total consensus decisions: {stats['total_consensus']}",
                f"Correct decisions: {stats['correct_count']}",
                f"Accuracy: {stats['accuracy']:.2%}",
                f"Mean confidence: {stats['mean_confidence']:.2%}",
                "",
                "Agreement Distribution:",
            ]
        )

        for level, count in stats["agreement_distribution"].items():
            lines.append(f"  {level.capitalize()}: {count}")

        lines.append("")
        lines.append("Accuracy by Agreement Level:")
        for level, level_stats in stats["by_agreement"].items():
            if level_stats["total"] > 0:
                accuracy = level_stats["correct"] / level_stats["total"]
                lines.append(
                    f"  {level.capitalize()}: {level_stats['correct']}/{level_stats['total']} ({accuracy:.2%})"
                )

        return "\n".join(lines)

    def render_voi(
        self,
        agent_id: Optional[str] = None,
        days: int = DEFAULT_BAYESIAN_LOOKBACK_DAYS,
    ) -> str:
        """Render Value of Information statistics."""
        stats = self.monitor.get_voi_statistics(agent_id, days)
        lines = self._header(f"Value of Information Statistics (Last {days} days)")
        lines.extend(
            [
                f"Total queries: {stats['total_queries']}",
                f"Beneficial queries: {stats['beneficial_queries']}",
                f"Beneficial rate: {stats['beneficial_rate']:.2%}",
                f"Mean predicted VoI: {stats['mean_predicted_voi']:.4f}",
                f"Mean actual gain: {stats['mean_actual_gain']:.4f}",
                f"Mean query cost: {stats['mean_query_cost']:.4f}",
            ]
        )

        if stats["per_agent"]:
            lines.append("")
            lines.append("Per-Agent Statistics:")
            for agent_name, agent_stats in sorted(stats["per_agent"].items()):
                lines.extend(
                    [
                        "",
                        f"  {agent_name}:",
                        f"    Total queries: {agent_stats['total_queries']}",
                        f"    Beneficial rate: {agent_stats['beneficial_rate']:.2%}",
                        f"    Mean predicted VoI: {agent_stats['mean_predicted_voi']:.4f}",
                        f"    Mean actual gain: {agent_stats['mean_actual_gain']:.4f}",
                    ]
                )

        return "\n".join(lines)

    def render_correlations(
        self,
        agent_ids: Sequence[str],
        days: int = DEFAULT_BAYESIAN_LOOKBACK_DAYS,
    ) -> str:
        """Render an agent correlation matrix and notable pairs."""
        matrix = self.monitor.get_correlation_matrix(list(agent_ids), days)
        lines = [
            self.renderer.render_heatmap(matrix, f"Agent Correlations (Last {days} days)").rstrip()
        ]
        lines.append("")
        lines.append("Highly Correlated Pairs (|correlation| > 0.7):")

        found_pair = False
        for index, agent_1 in enumerate(agent_ids):
            for agent_2 in agent_ids[index + 1 :]:
                correlation = matrix.get(agent_1, {}).get(agent_2, 0.0)
                if abs(correlation) > 0.7:
                    found_pair = True
                    lines.append(f"  {agent_1} <-> {agent_2}: {correlation:.3f}")

        if not found_pair:
            lines.append("  None")

        return "\n".join(lines)

    def render_belief(self, belief_id: str) -> str:
        """Render belief evolution details."""
        evolution = self.monitor.get_belief_evolution(belief_id)
        if not evolution:
            return f"No evolution data found for belief_id: {belief_id}"

        latest = evolution[-1]
        lines = self._header(f"Belief State Evolution: {belief_id}")
        lines.extend(
            [
                "Current State:",
                f"  Success probability: {latest['success_prob']:.2%}",
                f"  Failure probability: {latest['failure_prob']:.2%}",
                f"  Entropy: {latest['entropy']:.4f} nats",
                f"  Updates: {len(evolution)}",
            ]
        )

        if len(evolution) > 1:
            lines.append("")
            lines.append("Evolution (last 10 updates):")
            for snapshot in evolution[-10:]:
                timestamp = snapshot["timestamp"][:19]
                agent_name = snapshot["agent_id"] or "System"
                lines.append(
                    f"  {timestamp} | {agent_name:10} | "
                    f"P(success)={snapshot['success_prob']:.2f}, H={snapshot['entropy']:.3f}"
                )

        return "\n".join(lines)

    def export_summary_json(
        self,
        output_path: str,
        days: int = DEFAULT_BAYESIAN_LOOKBACK_DAYS,
    ) -> None:
        """Export the system summary to JSON."""
        self.exporter.export_summary_json(output_path, days)

    def export_reliability_csv(
        self,
        output_path: str,
        agent_ids: Optional[Sequence[str]] = None,
        days: int = DEFAULT_BAYESIAN_LOOKBACK_DAYS,
    ) -> None:
        """Export reliability trends to CSV."""
        self.exporter.export_reliability_trends_csv(
            output_path,
            list(agent_ids) if agent_ids is not None else None,
            days,
        )

    def export_belief_csv(self, belief_id: str, output_path: str) -> None:
        """Export belief evolution to CSV."""
        self.exporter.export_belief_evolution_csv(belief_id, output_path)

    @staticmethod
    def _header(title: str) -> list[str]:
        return ["", "=" * SECTION_WIDTH, title, "=" * SECTION_WIDTH, ""]


def get_bayesian_monitoring_service(
    db_connection: Optional[sqlite3.Connection] = None,
) -> BayesianMonitoringService:
    """Create the shared monitoring service."""
    connection = db_connection if db_connection is not None else get_database()
    return BayesianMonitoringService(connection)
