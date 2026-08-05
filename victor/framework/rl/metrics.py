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

"""RL metrics exporter for observability.

Provides metrics export in multiple formats:
- JSON: For dashboards and custom integrations
- Prometheus: For monitoring infrastructure

Metrics Categories:
    - Learner Outcomes: Success rates, quality scores per learner
    - Q-Values: Q-value distributions per learner
    - Exploration: Epsilon tracking, exploration vs exploitation rates
    - Performance: Learner update frequencies, outcome recording latencies

Usage:
    from victor.framework.rl.metrics import RLMetricsExporter, get_rl_metrics

    # Get global exporter
    exporter = get_rl_metrics()

    # Export as JSON
    metrics_json = exporter.export_json()

    # Export as Prometheus text format
    prometheus_text = exporter.export_prometheus()

    # Get summary for a specific learner
    summary = exporter.get_learner_summary("tool_selector")
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.rl.coordinator import RLCoordinator
    from victor.framework.rl.hooks import RLHookRegistry

logger = logging.getLogger(__name__)


@dataclass
class LearnerMetrics:
    """Metrics for a single RL learner."""

    learner_name: str
    outcome_count: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    avg_quality_score: float = 0.5
    last_update: Optional[str] = None

    # Exploration metrics
    exploration_count: int = 0
    exploitation_count: int = 0
    exploration_rate: float = 0.0
    current_epsilon: Optional[float] = None

    # Q-value stats
    q_value_count: int = 0
    avg_q_value: float = 0.5
    min_q_value: float = 0.0
    max_q_value: float = 1.0


@dataclass
class RLSystemMetrics:
    """Aggregate metrics for the RL system."""

    total_outcomes: int = 0
    total_learners: int = 0
    active_learners: List[str] = field(default_factory=list)
    inactive_learners: List[str] = field(default_factory=list)
    event_counts: Dict[str, int] = field(default_factory=dict)
    overall_success_rate: float = 0.0
    overall_quality: float = 0.5
    export_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RLMetricsExporter:
    """Exports RL metrics in multiple formats.

    Collects metrics from:
    - RLCoordinator: Learner outcomes, Q-values
    - RLHookRegistry: Event counts, exploration tracking
    - Database: Historical data aggregations

    Attributes:
        coordinator: RL coordinator for database access
        hooks: RL hook registry for event metrics
    """

    # Expected learners in the system
    EXPECTED_LEARNERS = [
        "tool_selector",
        "mode_transition",
        "continuation_patience",
        "continuation_prompts",
        "grounding_threshold",
        "semantic_threshold",
        "cache_eviction",
        "quality_weights",
        "model_selector",
        "team_composition",
        "workflow_execution",
        "cross_vertical",
        "prompt_template",
    ]

    def __init__(
        self,
        coordinator: Optional["RLCoordinator"] = None,
        hooks: Optional["RLHookRegistry"] = None,
    ):
        """Initialize metrics exporter.

        Args:
            coordinator: RL coordinator for database access
            hooks: RL hook registry for event metrics
        """
        self._coordinator = coordinator
        self._hooks = hooks

    def set_coordinator(self, coordinator: "RLCoordinator") -> None:
        """Set the RL coordinator."""
        self._coordinator = coordinator

    def set_hooks(self, hooks: "RLHookRegistry") -> None:
        """Set the RL hook registry."""
        self._hooks = hooks

    def get_learner_summary(self, learner_name: str) -> LearnerMetrics:
        """Get metrics summary for a specific learner.

        Args:
            learner_name: Name of the learner

        Returns:
            LearnerMetrics with current stats
        """
        metrics = LearnerMetrics(learner_name=learner_name)

        # Get outcome stats from database
        if self._coordinator is not None:
            try:
                cursor = self._coordinator.db.cursor()

                # Get outcome counts
                cursor.execute(
                    """
                    SELECT COUNT(*), SUM(success), AVG(quality_score)
                    FROM rl_outcome
                    WHERE learner_name = ?
                    """,
                    (learner_name,),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    metrics.outcome_count = row[0]
                    metrics.success_count = row[1] or 0
                    metrics.success_rate = metrics.success_count / metrics.outcome_count
                    metrics.avg_quality_score = row[2] or 0.5

                # Get last update time
                cursor.execute(
                    """
                    SELECT MAX(timestamp)
                    FROM rl_outcome
                    WHERE learner_name = ?
                    """,
                    (learner_name,),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    metrics.last_update = row[0]

            except Exception as e:
                logger.debug("Failed to get learner outcome stats: %s", e)

        # Get exploration metrics from hooks
        if self._hooks is not None:
            try:
                metrics.exploration_rate = self._hooks.get_exploration_rate(learner_name)
                metrics.exploration_count = self._hooks._exploration_counts.get(learner_name, 0)
                metrics.exploitation_count = self._hooks._exploitation_counts.get(learner_name, 0)

                # Get current epsilon from history
                history = self._hooks.get_epsilon_trend(learner_name, limit=1)
                if history:
                    metrics.current_epsilon = history[-1][1]

            except Exception as e:
                logger.debug("Failed to get exploration metrics: %s", e)

        return metrics

    def get_system_metrics(self) -> RLSystemMetrics:
        """Get aggregate metrics for the RL system.

        Returns:
            RLSystemMetrics with system-wide stats
        """
        metrics = RLSystemMetrics()

        # Get total outcomes and success rate from database
        if self._coordinator is not None:
            try:
                cursor = self._coordinator.db.cursor()

                # Total outcomes and success rate
                cursor.execute("""
                    SELECT COUNT(*), SUM(success), AVG(quality_score)
                    FROM rl_outcome
                    """)
                row = cursor.fetchone()
                if row and row[0]:
                    metrics.total_outcomes = row[0]
                    success_count = row[1] or 0
                    metrics.overall_success_rate = success_count / metrics.total_outcomes
                    metrics.overall_quality = row[2] or 0.5

                # Get active learners (those with outcomes)
                cursor.execute("""
                    SELECT DISTINCT learner_name
                    FROM rl_outcome
                    """)
                active = {row[0] for row in cursor.fetchall()}
                metrics.active_learners = list(active)
                metrics.inactive_learners = [
                    learner for learner in self.EXPECTED_LEARNERS if learner not in active
                ]
                metrics.total_learners = len(active)

            except Exception as e:
                logger.debug("Failed to get system metrics: %s", e)

        # Get event counts from hooks
        if self._hooks is not None:
            try:
                hook_metrics = self._hooks.get_metrics()
                metrics.event_counts = hook_metrics.get("event_counts", {})
            except Exception as e:
                logger.debug("Failed to get event counts: %s", e)

        return metrics

    def export_json(self) -> str:
        """Export all metrics as JSON.

        Returns:
            JSON string with all metrics
        """
        system = self.get_system_metrics()

        learners = {}
        for learner_name in self.EXPECTED_LEARNERS:
            learners[learner_name] = asdict(self.get_learner_summary(learner_name))

        data = {
            "system": asdict(system),
            "learners": learners,
        }

        return json_dumps(data, indent=2, default=str)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text exposition format.

        Returns:
            Prometheus-formatted metrics text
        """
        lines = []

        # System metrics
        system = self.get_system_metrics()
        lines.append("# HELP victor_rl_outcomes_total Total RL outcomes recorded")
        lines.append("# TYPE victor_rl_outcomes_total counter")
        lines.append(f"victor_rl_outcomes_total {system.total_outcomes}")

        lines.append("")
        lines.append("# HELP victor_rl_success_rate Overall RL success rate")
        lines.append("# TYPE victor_rl_success_rate gauge")
        lines.append(f"victor_rl_success_rate {system.overall_success_rate:.4f}")

        lines.append("")
        lines.append("# HELP victor_rl_quality_score Average quality score")
        lines.append("# TYPE victor_rl_quality_score gauge")
        lines.append(f"victor_rl_quality_score {system.overall_quality:.4f}")

        lines.append("")
        lines.append("# HELP victor_rl_active_learners Number of active learners")
        lines.append("# TYPE victor_rl_active_learners gauge")
        lines.append(f"victor_rl_active_learners {system.total_learners}")

        # Per-learner metrics
        lines.append("")
        lines.append("# HELP victor_rl_learner_outcomes Outcomes per learner")
        lines.append("# TYPE victor_rl_learner_outcomes counter")

        for learner_name in self.EXPECTED_LEARNERS:
            learner = self.get_learner_summary(learner_name)
            lines.append(
                f'victor_rl_learner_outcomes{{learner="{learner_name}"}} {learner.outcome_count}'
            )

        lines.append("")
        lines.append("# HELP victor_rl_learner_success_rate Success rate per learner")
        lines.append("# TYPE victor_rl_learner_success_rate gauge")

        for learner_name in self.EXPECTED_LEARNERS:
            learner = self.get_learner_summary(learner_name)
            lines.append(
                f'victor_rl_learner_success_rate{{learner="{learner_name}"}} {learner.success_rate:.4f}'
            )

        lines.append("")
        lines.append("# HELP victor_rl_exploration_rate Exploration rate per learner")
        lines.append("# TYPE victor_rl_exploration_rate gauge")

        for learner_name in self.EXPECTED_LEARNERS:
            learner = self.get_learner_summary(learner_name)
            lines.append(
                f'victor_rl_exploration_rate{{learner="{learner_name}"}} {learner.exploration_rate:.4f}'
            )

        # Event counts
        lines.append("")
        lines.append("# HELP victor_rl_events_total Events per type")
        lines.append("# TYPE victor_rl_events_total counter")

        for event_type, count in system.event_counts.items():
            lines.append(f'victor_rl_events_total{{event="{event_type}"}} {count}')

        # Priority 4: Learning from Execution metrics (gated by feature flag)
        try:
            from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager

            if get_feature_flag_manager().is_enabled(FeatureFlag.USE_LEARNING_FROM_EXECUTION):
                lines.extend(self._export_priority4_metrics())
        except Exception:
            pass

        return "\n".join(lines)

    def _export_priority4_metrics(self) -> List[str]:
        """Export Priority 4 (Learning from Execution) Prometheus metrics.

        Reads from existing learners and UsageAnalytics — no new DB queries.
        """
        lines: List[str] = []

        if self._coordinator is None:
            return lines

        # User feedback learner metrics
        try:
            fb_learner = self._coordinator.get_learner("user_feedback")
            if fb_learner is not None:
                stats = fb_learner.get_feedback_stats()  # type: ignore[attr-defined]
                lines.append("")
                lines.append("# HELP victor_rl_user_feedback_total Total user feedback events")
                lines.append("# TYPE victor_rl_user_feedback_total counter")
                lines.append(f"victor_rl_user_feedback_total {stats.get('total_feedback', 0)}")

                avg_rating = stats.get("avg_rating")
                if avg_rating is not None:
                    lines.append("")
                    lines.append(
                        "# HELP victor_rl_user_feedback_avg_rating Average user rating (0-1)"
                    )
                    lines.append("# TYPE victor_rl_user_feedback_avg_rating gauge")
                    lines.append(f"victor_rl_user_feedback_avg_rating {avg_rating:.4f}")
        except Exception as e:
            logger.debug("Priority4 metrics: user_feedback skipped: %s", e)

        # Model selector confidence thresholds
        try:
            model_learner = self._coordinator.get_learner("model_selector")
            if model_learner is not None and hasattr(model_learner, "get_optimal_threshold"):
                for decision_type in ("tool_necessity", "complexity", "task_type"):
                    threshold = model_learner.get_optimal_threshold(decision_type)
                    if threshold is not None:
                        lines.append("")
                        lines.append(
                            f"# HELP victor_rl_confidence_threshold_{decision_type} "
                            f"Learned confidence threshold for {decision_type}"
                        )
                        lines.append(f"# TYPE victor_rl_confidence_threshold_{decision_type} gauge")
                        lines.append(
                            f"victor_rl_confidence_threshold_{decision_type} {threshold:.4f}"
                        )
        except Exception as e:
            logger.debug("Priority4 metrics: model_selector thresholds skipped: %s", e)

        # Quality weights per-task personalization count
        try:
            qw_learner = self._coordinator.get_learner("quality_weights")
            if qw_learner is not None and hasattr(qw_learner, "export_metrics"):
                metrics = qw_learner.export_metrics()
                pref_count = metrics.get("user_preference_count", 0)
                lines.append("")
                lines.append(
                    "# HELP victor_rl_user_preferences_total Personalized quality weight preferences stored"
                )
                lines.append("# TYPE victor_rl_user_preferences_total counter")
                lines.append(f"victor_rl_user_preferences_total {pref_count}")
        except Exception as e:
            logger.debug("Priority4 metrics: quality_weights skipped: %s", e)

        return lines


# Global singleton
_metrics_exporter: Optional[RLMetricsExporter] = None


def get_rl_metrics(
    coordinator: Optional["RLCoordinator"] = None,
    hooks: Optional["RLHookRegistry"] = None,
) -> RLMetricsExporter:
    """Get the global RL metrics exporter.

    Args:
        coordinator: Optional coordinator to set
        hooks: Optional hooks registry to set

    Returns:
        Global RLMetricsExporter instance
    """
    global _metrics_exporter

    if _metrics_exporter is None:
        _metrics_exporter = RLMetricsExporter(coordinator, hooks)
    else:
        if coordinator is not None:
            _metrics_exporter.set_coordinator(coordinator)
        if hooks is not None:
            _metrics_exporter.set_hooks(hooks)

    return _metrics_exporter


def reset_rl_metrics() -> None:
    """Reset the global metrics exporter (for testing)."""
    global _metrics_exporter
    _metrics_exporter = None
