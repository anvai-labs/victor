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

"""Unit tests for ExperimentCoordinator.

Tests the A/B testing framework for RL policies.
"""

import pytest

from victor.framework.rl.experiment_coordinator import (
    ExperimentConfig,
    ExperimentCoordinator,
    ExperimentResult,
    ExperimentStatus,
    Variant,
    VariantMetrics,
    VariantType,
)


@pytest.fixture
def coordinator() -> ExperimentCoordinator:
    """Fixture for ExperimentCoordinator without database."""
    return ExperimentCoordinator()


@pytest.fixture
def sample_config() -> ExperimentConfig:
    """Fixture for a sample experiment config."""
    return ExperimentConfig(
        experiment_id="test_exp_001",
        name="Test Experiment",
        description="A test experiment",
        control=Variant(
            name="baseline",
            type=VariantType.CONTROL,
            config={
                "section_name": "GROUNDING_RULES",
                "provider": "anthropic",
                "strategy_name": "gepa",
            },
            description="Baseline selector",
        ),
        treatment=Variant(
            name="rl_selector",
            type=VariantType.TREATMENT,
            config={
                "section_name": "GROUNDING_RULES",
                "provider": "anthropic",
                "strategy_name": "prefpo",
            },
            description="RL-based selector",
        ),
        traffic_split=0.5,  # 50% for easier testing
        min_samples_per_variant=10,  # Low for testing
        significance_level=0.05,
    )


class TestVariantMetrics:
    """Tests for VariantMetrics."""

    def test_success_rate_empty(self) -> None:
        """Test success rate with no samples."""
        metrics = VariantMetrics(variant_name="test")
        assert metrics.success_rate == 0.0

    def test_success_rate_calculation(self) -> None:
        """Test success rate calculation."""
        metrics = VariantMetrics(
            variant_name="test",
            sample_count=10,
            success_count=7,
        )
        assert metrics.success_rate == 0.7

    def test_avg_quality_calculation(self) -> None:
        """Test average quality calculation."""
        metrics = VariantMetrics(
            variant_name="test",
            sample_count=4,
            total_quality=3.2,
        )
        assert metrics.avg_quality == 0.8

    def test_get_metric_custom(self) -> None:
        """Test getting custom metrics."""
        metrics = VariantMetrics(
            variant_name="test",
            sample_count=5,
            metric_sums={"latency": 500.0},
        )
        assert metrics.get_metric("latency") == 100.0
        assert metrics.get_metric("unknown") == 0.0


class TestExperimentCoordinator:
    """Tests for ExperimentCoordinator."""

    def test_create_experiment(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test creating an experiment."""
        result = coordinator.create_experiment(sample_config)

        assert result is True
        assert sample_config.experiment_id in coordinator._experiments
        assert coordinator._status[sample_config.experiment_id] == ExperimentStatus.DRAFT

    def test_create_duplicate_experiment_fails(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test that creating duplicate experiment fails."""
        coordinator.create_experiment(sample_config)
        result = coordinator.create_experiment(sample_config)

        assert result is False

    def test_start_experiment(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test starting an experiment."""
        coordinator.create_experiment(sample_config)
        result = coordinator.start_experiment(sample_config.experiment_id)

        assert result is True
        assert coordinator._status[sample_config.experiment_id] == ExperimentStatus.RUNNING

    def test_start_nonexistent_experiment_fails(self, coordinator: ExperimentCoordinator) -> None:
        """Test starting nonexistent experiment fails."""
        result = coordinator.start_experiment("nonexistent")
        assert result is False

    def test_pause_experiment(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test pausing an experiment."""
        coordinator.create_experiment(sample_config)
        coordinator.start_experiment(sample_config.experiment_id)
        result = coordinator.pause_experiment(sample_config.experiment_id)

        assert result is True
        assert coordinator._status[sample_config.experiment_id] == ExperimentStatus.PAUSED

    def test_assign_variant_not_running_returns_none(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test that assigning variant when not running returns None."""
        coordinator.create_experiment(sample_config)

        variant = coordinator.assign_variant(sample_config.experiment_id, "session_123")
        assert variant is None

    def test_assign_variant_consistent(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test that variant assignment is consistent for same session."""
        coordinator.create_experiment(sample_config)
        coordinator.start_experiment(sample_config.experiment_id)

        v1 = coordinator.assign_variant(sample_config.experiment_id, "session_123")
        v2 = coordinator.assign_variant(sample_config.experiment_id, "session_123")

        assert v1 == v2

    def test_assign_variant_distributes_traffic(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test that variant assignment distributes traffic."""
        coordinator.create_experiment(sample_config)
        coordinator.start_experiment(sample_config.experiment_id)

        control_count = 0
        treatment_count = 0

        for i in range(100):
            variant = coordinator.assign_variant(sample_config.experiment_id, f"session_{i}")
            if variant == sample_config.control.name:
                control_count += 1
            else:
                treatment_count += 1

        # With 50/50 split, should be roughly even
        assert control_count > 20
        assert treatment_count > 20

    def test_record_outcome(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test recording outcomes."""
        coordinator.create_experiment(sample_config)
        coordinator.start_experiment(sample_config.experiment_id)

        session_id = "session_test"
        coordinator.assign_variant(sample_config.experiment_id, session_id)
        coordinator.record_outcome(
            sample_config.experiment_id,
            session_id,
            success=True,
            quality_score=0.85,
            latency_ms=150.0,
        )

        # Check metrics updated
        metrics = coordinator._metrics[sample_config.experiment_id]
        total_samples = sum(m.sample_count for m in metrics.values())
        assert total_samples == 1

    def test_analyze_insufficient_samples(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test analysis with insufficient samples."""
        coordinator.create_experiment(sample_config)
        coordinator.start_experiment(sample_config.experiment_id)

        # Record just a few outcomes
        for i in range(5):
            session_id = f"session_{i}"
            coordinator.assign_variant(sample_config.experiment_id, session_id)
            coordinator.record_outcome(sample_config.experiment_id, session_id, success=True)

        result = coordinator.analyze_experiment(sample_config.experiment_id)

        assert result is not None
        assert result.is_significant is False
        assert "Insufficient" in result.recommendation

    def test_analyze_with_sufficient_samples(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test analysis with sufficient samples."""
        coordinator.create_experiment(sample_config)
        coordinator.start_experiment(sample_config.experiment_id)

        # Record enough outcomes with different success rates
        # Force specific assignments for control
        for i in range(sample_config.min_samples_per_variant + 5):
            session_id = f"control_session_{i}"
            coordinator._assignments.setdefault(sample_config.experiment_id, {})[
                session_id
            ] = sample_config.control.name
            coordinator.record_outcome(
                sample_config.experiment_id,
                session_id,
                success=i % 2 == 0,  # 50% success
                quality_score=0.5,
            )

        # Force treatment assignments with higher success
        for i in range(sample_config.min_samples_per_variant + 5):
            session_id = f"treatment_session_{i}"
            coordinator._assignments.setdefault(sample_config.experiment_id, {})[
                session_id
            ] = sample_config.treatment.name
            coordinator.record_outcome(
                sample_config.experiment_id,
                session_id,
                success=i % 4 != 0,  # 75% success
                quality_score=0.75,
            )

        result = coordinator.analyze_experiment(sample_config.experiment_id)

        assert result is not None
        assert "control" in result.details
        assert "treatment" in result.details

    def test_rollout_treatment(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test rolling out treatment."""
        coordinator.create_experiment(sample_config)

        result = coordinator.rollout_treatment(sample_config.experiment_id)

        assert result is True
        assert coordinator._status[sample_config.experiment_id] == ExperimentStatus.ROLLED_OUT

    def test_rollback_experiment(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test rolling back experiment."""
        coordinator.create_experiment(sample_config)

        result = coordinator.rollback_experiment(sample_config.experiment_id)

        assert result is True
        assert coordinator._status[sample_config.experiment_id] == ExperimentStatus.ROLLED_BACK

    def test_get_experiment_status(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test getting experiment status."""
        coordinator.create_experiment(sample_config)

        status = coordinator.get_experiment_status(sample_config.experiment_id)

        assert status is not None
        assert status["experiment_id"] == sample_config.experiment_id
        assert status["status"] == ExperimentStatus.DRAFT.value
        assert status["section_name"] == "GROUNDING_RULES"
        assert status["provider"] == "anthropic"
        assert status["control"]["strategy_name"] == "gepa"
        assert status["treatment"]["strategy_name"] == "prefpo"
        assert "control" in status
        assert "treatment" in status

    def test_get_status_nonexistent(self, coordinator: ExperimentCoordinator) -> None:
        """Test getting status of nonexistent experiment."""
        status = coordinator.get_experiment_status("nonexistent")
        assert status is None

    def test_list_experiments(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test listing experiments."""
        coordinator.create_experiment(sample_config)

        # Create another experiment
        config2 = ExperimentConfig(
            experiment_id="test_exp_002",
            name="Second Experiment",
            description="Another test",
            control=Variant("ctrl", VariantType.CONTROL),
            treatment=Variant("treat", VariantType.TREATMENT),
        )
        coordinator.create_experiment(config2)

        experiments = coordinator.list_experiments()

        assert len(experiments) == 2
        assert experiments[0]["section_name"] == "GROUNDING_RULES"
        assert experiments[0]["provider"] == "anthropic"
        assert experiments[0]["treatment"]["strategy_name"] == "prefpo"

    def test_export_metrics(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test metrics export."""
        coordinator.create_experiment(sample_config)
        coordinator.start_experiment(sample_config.experiment_id)

        metrics = coordinator.export_metrics()

        assert metrics["total_experiments"] == 1
        assert "by_status" in metrics
        assert metrics["by_status"]["running"] == 1

    def test_custom_metrics_in_outcome(
        self, coordinator: ExperimentCoordinator, sample_config: ExperimentConfig
    ) -> None:
        """Test recording custom metrics."""
        coordinator.create_experiment(sample_config)
        coordinator.start_experiment(sample_config.experiment_id)

        session_id = "session_custom"
        coordinator._assignments.setdefault(sample_config.experiment_id, {})[
            session_id
        ] = sample_config.control.name

        coordinator.record_outcome(
            sample_config.experiment_id,
            session_id,
            success=True,
            custom_metrics={"tool_count": 5, "tokens_used": 1000},
        )

        metrics = coordinator._metrics[sample_config.experiment_id][sample_config.control.name]
        assert metrics.metric_sums.get("tool_count") == 5
        assert metrics.metric_sums.get("tokens_used") == 1000
