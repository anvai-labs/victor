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

"""Unit tests for RLMetricsCollector.

Tests the RL metrics collection and export for observability.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from victor.observability.rl_metrics import (
    RLMetricsCollector,
    LearnerMetrics,
    SystemMetrics,
    AlertMetrics,
    get_rl_metrics_collector,
)


@pytest.fixture
def collector() -> RLMetricsCollector:
    """Fixture for RLMetricsCollector."""
    return RLMetricsCollector()


@pytest.fixture
def mock_coordinator() -> MagicMock:
    """Fixture for mock RLCoordinator."""
    coordinator = MagicMock()
    coordinator.export_metrics.return_value = {
        "coordinator": {
            "total_outcomes": 100,
            "learners": {"tool_selector": {}, "model_selector": {}},
        }
    }
    return coordinator


@pytest.fixture
def mock_learner() -> MagicMock:
    """Fixture for mock learner."""
    learner = MagicMock()
    learner.export_metrics.return_value = {
        "total_samples": 50,
        "total_contexts": 5,
    }
    learner._q_values = {"ctx1": 0.7, "ctx2": 0.8, "ctx3": 0.6}
    return learner


class TestLearnerMetrics:
    """Tests for LearnerMetrics dataclass."""

    def test_learner_metrics_creation(self) -> None:
        """Test creating LearnerMetrics."""
        metrics = LearnerMetrics(
            name="test_learner",
            total_samples=100,
            success_rate=0.85,
            avg_confidence=0.9,
            q_value_mean=0.75,
            q_value_std=0.1,
            contexts_learned=10,
        )

        assert metrics.name == "test_learner"
        assert metrics.total_samples == 100
        assert metrics.success_rate == 0.85
        assert metrics.contexts_learned == 10

    def test_learner_metrics_defaults(self) -> None:
        """Test default values for LearnerMetrics."""
        metrics = LearnerMetrics(name="test")

        assert metrics.total_samples == 0
        assert metrics.success_rate == 0.0
        assert metrics.avg_confidence == 0.0
        assert metrics.q_value_mean == 0.5
        assert metrics.q_value_std == 0.0
        assert metrics.custom_metrics == {}


class TestSystemMetrics:
    """Tests for SystemMetrics dataclass."""

    def test_system_metrics_creation(self) -> None:
        """Test creating SystemMetrics."""
        metrics = SystemMetrics(
            total_outcomes=500,
            active_learners=5,
            active_experiments=2,
            avg_reward=0.75,
            policy_drift_score=0.1,
        )

        assert metrics.total_outcomes == 500
        assert metrics.active_learners == 5
        assert metrics.avg_reward == 0.75

    def test_system_metrics_curriculum_distribution(self) -> None:
        """Test curriculum distribution in SystemMetrics."""
        metrics = SystemMetrics(curriculum_distribution={"WARM_UP": 5, "BASIC": 3, "EXPERT": 1})

        assert metrics.curriculum_distribution["WARM_UP"] == 5
        assert metrics.curriculum_distribution["BASIC"] == 3


class TestAlertMetrics:
    """Tests for AlertMetrics dataclass."""

    def test_alert_metrics_no_alerts(self) -> None:
        """Test AlertMetrics with no alerts."""
        alerts = AlertMetrics()

        assert alerts.degradation_detected is False
        assert alerts.degradation_learners == []
        assert alerts.anomaly_score == 0.0
        assert alerts.stale_learners == []

    def test_alert_metrics_with_degradation(self) -> None:
        """Test AlertMetrics with degradation detected."""
        alerts = AlertMetrics(
            degradation_detected=True,
            degradation_learners=["tool_selector", "model_selector"],
            anomaly_score=0.8,
        )

        assert alerts.degradation_detected is True
        assert len(alerts.degradation_learners) == 2


class TestRLMetricsCollector:
    """Tests for RLMetricsCollector."""

    def test_initialization(self, collector: RLMetricsCollector) -> None:
        """Test collector initialization."""
        assert collector._coordinator is None
        assert collector._reward_history == []
        assert collector._success_history == {}

    def test_set_coordinator(
        self, collector: RLMetricsCollector, mock_coordinator: MagicMock
    ) -> None:
        """Test setting coordinator."""
        collector.set_coordinator(mock_coordinator)
        assert collector._coordinator == mock_coordinator

    def test_set_experiment_coordinator(self, collector: RLMetricsCollector) -> None:
        """Test setting experiment coordinator."""
        exp_coordinator = MagicMock()
        collector.set_experiment_coordinator(exp_coordinator)
        assert collector._experiment_coordinator == exp_coordinator

    def test_set_curriculum_controller(self, collector: RLMetricsCollector) -> None:
        """Test setting curriculum controller."""
        curriculum = MagicMock()
        collector.set_curriculum_controller(curriculum)
        assert collector._curriculum_controller == curriculum

    def test_record_reward(self, collector: RLMetricsCollector) -> None:
        """Test recording rewards."""
        collector.record_reward(0.5)
        collector.record_reward(0.8)
        collector.record_reward(0.7)

        assert len(collector._reward_history) == 3
        assert collector._reward_history == [0.5, 0.8, 0.7]

    def test_record_reward_truncation(self, collector: RLMetricsCollector) -> None:
        """Test reward history truncation at 1000."""
        for i in range(1100):
            collector.record_reward(i * 0.001)

        assert len(collector._reward_history) == 1000
        # Should keep most recent
        assert collector._reward_history[-1] == pytest.approx(1.099, abs=0.001)

    def test_record_outcome(self, collector: RLMetricsCollector) -> None:
        """Test recording outcomes."""
        collector.record_outcome("tool_selector", True)
        collector.record_outcome("tool_selector", False)
        collector.record_outcome("model_selector", True)

        assert len(collector._success_history["tool_selector"]) == 2
        assert len(collector._success_history["model_selector"]) == 1

    def test_record_outcome_truncation(self, collector: RLMetricsCollector) -> None:
        """Test outcome history truncation."""
        for i in range(150):
            collector.record_outcome("learner", i % 2 == 0)

        # Should keep DEGRADATION_WINDOW (100) outcomes
        assert len(collector._success_history["learner"]) == collector.DEGRADATION_WINDOW

    def test_collect_learner_metrics_no_coordinator(self, collector: RLMetricsCollector) -> None:
        """Test collecting learner metrics without coordinator."""
        result = collector.collect_learner_metrics("tool_selector")
        assert result is None

    def test_collect_learner_metrics(
        self,
        collector: RLMetricsCollector,
        mock_coordinator: MagicMock,
        mock_learner: MagicMock,
    ) -> None:
        """Test collecting learner metrics."""
        mock_coordinator.get_learner.return_value = mock_learner
        collector.set_coordinator(mock_coordinator)

        # Record some outcomes
        for _ in range(10):
            collector.record_outcome("tool_selector", True)
        for _ in range(5):
            collector.record_outcome("tool_selector", False)

        metrics = collector.collect_learner_metrics("tool_selector")

        assert metrics is not None
        assert metrics.name == "tool_selector"
        assert metrics.total_samples == 50
        assert metrics.success_rate == pytest.approx(0.667, abs=0.01)

    def test_collect_learner_metrics_learner_not_found(
        self, collector: RLMetricsCollector, mock_coordinator: MagicMock
    ) -> None:
        """Test collecting metrics when learner not found."""
        mock_coordinator.get_learner.return_value = None
        collector.set_coordinator(mock_coordinator)

        result = collector.collect_learner_metrics("unknown_learner")
        assert result is None

    def test_collect_system_metrics(
        self, collector: RLMetricsCollector, mock_coordinator: MagicMock
    ) -> None:
        """Test collecting system metrics."""
        collector.set_coordinator(mock_coordinator)

        # Add reward history
        for i in range(50):
            collector.record_reward(0.5 + i * 0.01)

        metrics = collector.collect_system_metrics()

        assert metrics.total_outcomes == 100
        assert metrics.active_learners == 2
        assert metrics.avg_reward > 0

    def test_collect_system_metrics_with_experiment_coordinator(
        self, collector: RLMetricsCollector
    ) -> None:
        """Test system metrics with experiment coordinator."""
        exp_coordinator = MagicMock()
        exp_coordinator.export_metrics.return_value = {"by_status": {"running": 3, "completed": 5}}
        collector.set_experiment_coordinator(exp_coordinator)

        metrics = collector.collect_system_metrics()
        assert metrics.active_experiments == 3

    def test_collect_system_metrics_with_curriculum(self, collector: RLMetricsCollector) -> None:
        """Test system metrics with curriculum controller."""
        curriculum = MagicMock()
        curriculum.export_metrics.return_value = {"stage_distribution": {"WARM_UP": 3, "BASIC": 2}}
        collector.set_curriculum_controller(curriculum)

        metrics = collector.collect_system_metrics()
        assert metrics.curriculum_distribution == {"WARM_UP": 3, "BASIC": 2}

    def test_collect_alert_metrics_no_degradation(self, collector: RLMetricsCollector) -> None:
        """Test alert metrics with no degradation."""
        # Record consistent success
        for _ in range(100):
            collector.record_outcome("learner", True)

        alerts = collector.collect_alert_metrics()
        assert alerts.degradation_detected is False
        assert alerts.degradation_learners == []

    def test_collect_alert_metrics_with_degradation(self, collector: RLMetricsCollector) -> None:
        """Test alert metrics with degradation."""
        # Record early successes followed by failures
        for _ in range(50):
            collector.record_outcome("learner", True)
        for _ in range(50):
            collector.record_outcome("learner", False)

        alerts = collector.collect_alert_metrics()
        assert alerts.degradation_detected is True
        assert "learner" in alerts.degradation_learners

    def test_collect_alert_metrics_anomaly_score(self, collector: RLMetricsCollector) -> None:
        """Test anomaly score calculation."""
        # Record consistent rewards
        for _ in range(100):
            collector.record_reward(0.5)
        # Then deviant rewards
        for _ in range(10):
            collector.record_reward(0.1)

        alerts = collector.collect_alert_metrics()
        assert alerts.anomaly_score > 0

    def test_compute_policy_drift(self, collector: RLMetricsCollector) -> None:
        """Test policy drift computation."""
        # Not enough data
        for _ in range(50):
            collector.record_reward(0.5)
        assert collector._compute_policy_drift() == 0.0

        # Add more data with drift
        for _ in range(100):
            collector.record_reward(0.3)
        drift = collector._compute_policy_drift()
        assert drift > 0

    def test_compute_staleness(self, collector: RLMetricsCollector) -> None:
        """Test staleness computation."""
        # No collection yet
        assert collector._compute_staleness() == 0.0

        # After collection
        collector.collect_all()
        # Immediate staleness should be ~0
        assert collector._compute_staleness() < 0.1

    def test_collect_all(
        self,
        collector: RLMetricsCollector,
        mock_coordinator: MagicMock,
        mock_learner: MagicMock,
    ) -> None:
        """Test collecting all metrics."""
        mock_coordinator.get_learner.return_value = mock_learner
        collector.set_coordinator(mock_coordinator)

        all_metrics = collector.collect_all()

        assert "timestamp" in all_metrics
        assert "learners" in all_metrics
        assert "system" in all_metrics
        assert "alerts" in all_metrics

    def test_export_prometheus(
        self, collector: RLMetricsCollector, mock_coordinator: MagicMock
    ) -> None:
        """Test Prometheus export."""
        collector.set_coordinator(mock_coordinator)

        prometheus_output = collector.export_prometheus()

        assert "victor_rl_" in prometheus_output
        assert "# HELP" in prometheus_output
        assert "# TYPE" in prometheus_output
        assert "victor_rl_total_outcomes" in prometheus_output
        assert "victor_rl_active_learners" in prometheus_output

    def test_export_prometheus_learner_metrics(
        self,
        collector: RLMetricsCollector,
        mock_coordinator: MagicMock,
        mock_learner: MagicMock,
    ) -> None:
        """Test Prometheus export includes learner metrics."""
        mock_coordinator.get_learner.return_value = mock_learner
        collector.set_coordinator(mock_coordinator)
        collector.record_outcome("tool_selector", True)

        prometheus_output = collector.export_prometheus()

        # Should have metrics for tool_selector
        assert "victor_rl_tool_selector_samples" in prometheus_output
        assert "victor_rl_tool_selector_success_rate" in prometheus_output

    def test_export_json(self, collector: RLMetricsCollector) -> None:
        """Test JSON export."""
        json_output = collector.export_json()

        # Should be valid JSON
        parsed = json.loads(json_output)
        assert "timestamp" in parsed
        assert "system" in parsed

    def test_export_dict(self, collector: RLMetricsCollector) -> None:
        """Test dict export."""
        dict_output = collector.export_dict()

        assert isinstance(dict_output, dict)
        assert "timestamp" in dict_output


class TestGlobalSingleton:
    """Tests for global singleton."""

    def test_get_rl_metrics_collector(self) -> None:
        """Test getting global singleton."""
        # Reset global
        import victor.observability.rl_metrics as module

        module._rl_metrics_collector = None

        collector1 = get_rl_metrics_collector()
        collector2 = get_rl_metrics_collector()

        assert collector1 is collector2

    def test_singleton_preserves_state(self) -> None:
        """Test singleton preserves state."""
        import victor.observability.rl_metrics as module

        module._rl_metrics_collector = None

        collector = get_rl_metrics_collector()
        collector.record_reward(0.5)

        collector2 = get_rl_metrics_collector()
        assert len(collector2._reward_history) == 1
