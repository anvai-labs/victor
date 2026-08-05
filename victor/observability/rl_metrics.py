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

This module provides metrics collection and export for the RL framework,
enabling monitoring of learner performance, policy drift, and system health.

Metrics Categories:
1. Per-Learner Metrics:
   - Q-value distributions
   - Sample counts
   - Success rates
   - Confidence levels

2. System-Wide Metrics:
   - Total outcomes recorded
   - Reward trends
   - Policy convergence indicators
   - Active experiments

3. Alert Metrics:
   - Policy degradation detection
   - Anomaly scores
   - Staleness indicators

Export Formats:
- Prometheus (via OpenTelemetry)
- JSON (for dashboards)
- Dict (for programmatic access)

Sprint 6: Observability & Polish
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.rl.coordinator import RLCoordinator

logger = logging.getLogger(__name__)


@dataclass
class LearnerMetrics:
    """Metrics for a single learner.

    Attributes:
        name: Learner name
        total_samples: Total samples recorded
        success_rate: Overall success rate
        avg_confidence: Average recommendation confidence
        q_value_mean: Mean Q-value
        q_value_std: Q-value standard deviation
        last_update_age_seconds: Seconds since last update
        contexts_learned: Number of unique contexts
    """

    name: str
    total_samples: int = 0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    q_value_mean: float = 0.5
    q_value_std: float = 0.0
    last_update_age_seconds: float = 0.0
    contexts_learned: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-wide RL metrics.

    Attributes:
        total_outcomes: Total outcomes across all learners
        active_learners: Number of active learners
        active_experiments: Number of running experiments
        avg_reward: Average computed reward
        policy_drift_score: Measure of policy change rate
        staleness_score: Measure of data freshness
    """

    total_outcomes: int = 0
    active_learners: int = 0
    active_experiments: int = 0
    avg_reward: float = 0.0
    policy_drift_score: float = 0.0
    staleness_score: float = 0.0
    curriculum_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class AlertMetrics:
    """Metrics for alerting on RL health.

    Attributes:
        degradation_detected: Whether policy degradation is detected
        degradation_learners: Learners showing degradation
        anomaly_score: Overall anomaly score (0-1)
        stale_learners: Learners with stale data
    """

    degradation_detected: bool = False
    degradation_learners: List[str] = field(default_factory=list)
    anomaly_score: float = 0.0
    stale_learners: List[str] = field(default_factory=list)


class RLMetricsCollector:
    """Collector for RL framework metrics.

    Aggregates metrics from all RL components and provides
    export functionality for monitoring systems.

    Usage:
        collector = RLMetricsCollector()
        collector.set_coordinator(rl_coordinator)

        # Collect and export
        metrics = collector.collect_all()
        prometheus_metrics = collector.export_prometheus()
        json_metrics = collector.export_json()
    """

    # Staleness threshold (seconds)
    STALENESS_THRESHOLD = 3600  # 1 hour

    # Degradation detection parameters
    DEGRADATION_WINDOW = 100  # Recent samples to consider
    DEGRADATION_THRESHOLD = 0.15  # 15% drop triggers alert

    def __init__(self):
        """Initialize metrics collector."""
        self._coordinator: Optional["RLCoordinator"] = None
        self._experiment_coordinator: Optional[Any] = None
        self._curriculum_controller: Optional[Any] = None
        self._feedback_integration: Optional[Any] = None

        # Historical data for trend analysis
        self._reward_history: List[float] = []
        self._success_history: Dict[str, List[bool]] = {}

        # Last collection time
        self._last_collection: float = 0.0

    def set_coordinator(self, coordinator: "RLCoordinator") -> None:
        """Set the RL coordinator for metrics collection.

        Args:
            coordinator: RLCoordinator instance
        """
        self._coordinator = coordinator

    def set_experiment_coordinator(self, coordinator: Any) -> None:
        """Set the experiment coordinator.

        Args:
            coordinator: ExperimentCoordinator instance
        """
        self._experiment_coordinator = coordinator

    def set_curriculum_controller(self, controller: Any) -> None:
        """Set the curriculum controller.

        Args:
            controller: CurriculumController instance
        """
        self._curriculum_controller = controller

    def set_feedback_integration(self, integration: Any) -> None:
        """Set the feedback integration.

        Args:
            integration: FeedbackIntegration instance
        """
        self._feedback_integration = integration

    def record_reward(self, reward: float) -> None:
        """Record a reward for trend analysis.

        Args:
            reward: Computed reward value
        """
        self._reward_history.append(reward)
        # Keep last 1000 rewards
        self._reward_history = self._reward_history[-1000:]

    def record_outcome(self, learner_name: str, success: bool) -> None:
        """Record an outcome for degradation detection.

        Args:
            learner_name: Name of the learner
            success: Whether the outcome was successful
        """
        if learner_name not in self._success_history:
            self._success_history[learner_name] = []

        self._success_history[learner_name].append(success)
        # Keep last DEGRADATION_WINDOW outcomes
        self._success_history[learner_name] = self._success_history[learner_name][
            -self.DEGRADATION_WINDOW :
        ]

    def collect_learner_metrics(self, learner_name: str) -> Optional[LearnerMetrics]:
        """Collect metrics for a specific learner.

        Args:
            learner_name: Name of the learner

        Returns:
            LearnerMetrics or None if learner not found
        """
        if not self._coordinator:
            return None

        learner = self._coordinator.get_learner(learner_name)
        if not learner:
            return None

        try:
            # Get learner's exported metrics
            exported = learner.export_metrics()

            # Calculate success rate from history
            history = self._success_history.get(learner_name, [])
            success_rate = sum(history) / len(history) if history else 0.5

            # Extract Q-value statistics if available
            q_values = []
            if hasattr(learner, "_q_values"):
                q_values = list(learner._q_values.values())
            elif hasattr(learner, "_weights"):
                # For learners using weights instead of Q-values
                for weights in learner._weights.values():
                    if isinstance(weights, dict):
                        q_values.extend(weights.values())

            q_mean = sum(q_values) / len(q_values) if q_values else 0.5
            q_std = (
                (sum((q - q_mean) ** 2 for q in q_values) / len(q_values)) ** 0.5
                if len(q_values) > 1
                else 0.0
            )

            return LearnerMetrics(
                name=learner_name,
                total_samples=exported.get("total_samples", exported.get("total_sessions", 0)),
                success_rate=success_rate,
                avg_confidence=0.7,  # Could be computed from recommendations
                q_value_mean=q_mean,
                q_value_std=q_std,
                last_update_age_seconds=0.0,  # Would need timestamp tracking
                contexts_learned=exported.get(
                    "total_contexts", exported.get("task_types_learned", 0)
                ),
                custom_metrics={
                    k: v
                    for k, v in exported.items()
                    if isinstance(v, (int, float)) and k not in ["total_samples", "total_sessions"]
                },
            )

        except Exception as e:
            logger.warning(f"Failed to collect metrics for {learner_name}: {e}")
            return None

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect system-wide metrics.

        Returns:
            SystemMetrics aggregated from all components
        """
        metrics = SystemMetrics()

        if self._coordinator:
            try:
                exported = self._coordinator.export_metrics()
                metrics.total_outcomes = exported.get("coordinator", {}).get("total_outcomes", 0)
                metrics.active_learners = len(exported.get("coordinator", {}).get("learners", {}))
            except Exception as e:
                logger.warning(f"Failed to collect coordinator metrics: {e}")

        if self._experiment_coordinator:
            try:
                exp_metrics = self._experiment_coordinator.export_metrics()
                metrics.active_experiments = exp_metrics.get("by_status", {}).get("running", 0)
            except Exception as e:
                logger.warning(f"Failed to collect experiment metrics: {e}")

        if self._curriculum_controller:
            try:
                curr_metrics = self._curriculum_controller.export_metrics()
                metrics.curriculum_distribution = curr_metrics.get("stage_distribution", {})
            except Exception as e:
                logger.warning(f"Failed to collect curriculum metrics: {e}")

        # Compute average reward
        if self._reward_history:
            metrics.avg_reward = sum(self._reward_history) / len(self._reward_history)

        # Compute policy drift score
        metrics.policy_drift_score = self._compute_policy_drift()

        # Compute staleness score
        metrics.staleness_score = self._compute_staleness()

        return metrics

    def collect_alert_metrics(self) -> AlertMetrics:
        """Collect metrics for alerting.

        Returns:
            AlertMetrics with degradation and anomaly indicators
        """
        alerts = AlertMetrics()

        # Check for degradation in each learner
        for learner_name, history in self._success_history.items():
            if len(history) >= self.DEGRADATION_WINDOW:
                # Compare recent half to earlier half
                mid = len(history) // 2
                early_rate = sum(history[:mid]) / mid if mid > 0 else 0.5
                recent_rate = sum(history[mid:]) / (len(history) - mid)

                if early_rate - recent_rate > self.DEGRADATION_THRESHOLD:
                    alerts.degradation_detected = True
                    alerts.degradation_learners.append(learner_name)

        # Compute anomaly score (simplified)
        if self._reward_history and len(self._reward_history) > 10:
            recent = self._reward_history[-10:]
            overall_mean = sum(self._reward_history) / len(self._reward_history)
            recent_mean = sum(recent) / len(recent)

            # Anomaly if recent rewards deviate significantly
            deviation = abs(recent_mean - overall_mean)
            alerts.anomaly_score = min(1.0, deviation * 2)

        return alerts

    def collect_all(self) -> Dict[str, Any]:
        """Collect all metrics.

        Returns:
            Dictionary with all metrics categories
        """
        self._last_collection = time.time()

        learner_metrics = {}
        learner_names = [
            "continuation_patience",
            "continuation_prompts",
            "semantic_threshold",
            "model_selector",
            "tool_selector",
            "mode_transition",
            "cache_eviction",
            "grounding_threshold",
            "quality_weights",
        ]

        for name in learner_names:
            metrics = self.collect_learner_metrics(name)
            if metrics:
                learner_metrics[name] = metrics

        return {
            "timestamp": datetime.now().isoformat(),
            "learners": {
                name: {
                    "total_samples": m.total_samples,
                    "success_rate": m.success_rate,
                    "q_value_mean": m.q_value_mean,
                    "q_value_std": m.q_value_std,
                    "contexts_learned": m.contexts_learned,
                    "custom_metrics": m.custom_metrics,
                }
                for name, m in learner_metrics.items()
            },
            "system": {
                "total_outcomes": (sm := self.collect_system_metrics()).total_outcomes,
                "active_learners": sm.active_learners,
                "active_experiments": sm.active_experiments,
                "avg_reward": sm.avg_reward,
                "policy_drift_score": sm.policy_drift_score,
                "staleness_score": sm.staleness_score,
                "curriculum_distribution": sm.curriculum_distribution,
            },
            "alerts": {
                "degradation_detected": (am := self.collect_alert_metrics()).degradation_detected,
                "degradation_learners": am.degradation_learners,
                "anomaly_score": am.anomaly_score,
                "stale_learners": am.stale_learners,
            },
        }

    def _compute_policy_drift(self) -> float:
        """Compute policy drift score.

        Returns:
            Drift score [0, 1] where higher means more drift
        """
        if len(self._reward_history) < 100:
            return 0.0

        # Compare first and last quarters
        quarter = len(self._reward_history) // 4
        early = self._reward_history[:quarter]
        recent = self._reward_history[-quarter:]

        early_mean = sum(early) / len(early)
        recent_mean = sum(recent) / len(recent)

        # Normalize drift to [0, 1]
        drift = abs(recent_mean - early_mean)
        return min(1.0, drift)

    def _compute_staleness(self) -> float:
        """Compute overall staleness score.

        Returns:
            Staleness score [0, 1] where higher means more stale
        """
        if self._last_collection == 0:
            return 0.0

        age = time.time() - self._last_collection
        return min(1.0, age / self.STALENESS_THRESHOLD)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-compatible metrics string
        """
        lines = []
        metrics = self.collect_all()

        # Learner metrics
        for learner_name, learner_data in metrics["learners"].items():
            safe_name = learner_name.replace("-", "_")

            lines.append(f"# HELP victor_rl_{safe_name}_samples Total samples for {learner_name}")
            lines.append(f"# TYPE victor_rl_{safe_name}_samples counter")
            lines.append(f'victor_rl_{safe_name}_samples {learner_data["total_samples"]}')

            lines.append(
                f"# HELP victor_rl_{safe_name}_success_rate Success rate for {learner_name}"
            )
            lines.append(f"# TYPE victor_rl_{safe_name}_success_rate gauge")
            lines.append(f'victor_rl_{safe_name}_success_rate {learner_data["success_rate"]:.4f}')

            lines.append(
                f"# HELP victor_rl_{safe_name}_q_value_mean Mean Q-value for {learner_name}"
            )
            lines.append(f"# TYPE victor_rl_{safe_name}_q_value_mean gauge")
            lines.append(f'victor_rl_{safe_name}_q_value_mean {learner_data["q_value_mean"]:.4f}')

        # System metrics
        lines.append("# HELP victor_rl_total_outcomes Total RL outcomes recorded")
        lines.append("# TYPE victor_rl_total_outcomes counter")
        lines.append(f'victor_rl_total_outcomes {metrics["system"]["total_outcomes"]}')

        lines.append("# HELP victor_rl_active_learners Number of active learners")
        lines.append("# TYPE victor_rl_active_learners gauge")
        lines.append(f'victor_rl_active_learners {metrics["system"]["active_learners"]}')

        lines.append("# HELP victor_rl_avg_reward Average computed reward")
        lines.append("# TYPE victor_rl_avg_reward gauge")
        lines.append(f'victor_rl_avg_reward {metrics["system"]["avg_reward"]:.4f}')

        lines.append("# HELP victor_rl_policy_drift Policy drift score")
        lines.append("# TYPE victor_rl_policy_drift gauge")
        lines.append(f'victor_rl_policy_drift {metrics["system"]["policy_drift_score"]:.4f}')

        # Alert metrics
        lines.append("# HELP victor_rl_degradation_detected Policy degradation alert")
        lines.append("# TYPE victor_rl_degradation_detected gauge")
        lines.append(
            f'victor_rl_degradation_detected {1 if metrics["alerts"]["degradation_detected"] else 0}'
        )

        lines.append("# HELP victor_rl_anomaly_score Anomaly detection score")
        lines.append("# TYPE victor_rl_anomaly_score gauge")
        lines.append(f'victor_rl_anomaly_score {metrics["alerts"]["anomaly_score"]:.4f}')

        return "\n".join(lines)

    def export_json(self) -> str:
        """Export metrics as JSON.

        Returns:
            JSON string with all metrics
        """
        return json.dumps(self.collect_all(), indent=2)

    def export_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary.

        Returns:
            Dictionary with all metrics
        """
        return self.collect_all()


# Global singleton
_rl_metrics_collector: Optional[RLMetricsCollector] = None


def get_rl_metrics_collector() -> RLMetricsCollector:
    """Get global RL metrics collector (lazy init).

    Returns:
        RLMetricsCollector singleton
    """
    global _rl_metrics_collector
    if _rl_metrics_collector is None:
        _rl_metrics_collector = RLMetricsCollector()
    return _rl_metrics_collector
