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

"""Unit tests for PolicyManager.

Tests the policy lifecycle management, auto-checkpointing, and rollback.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from victor.framework.rl.policy_manager import (
    PolicyManager,
    PolicyState,
    PolicyStage,
    RollbackEvent,
    get_policy_manager,
)
from victor.framework.rl.checkpoint_store import CheckpointStore


@pytest.fixture
def temp_storage_path() -> Path:
    """Fixture for temporary storage path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "checkpoints"


@pytest.fixture
def checkpoint_store(temp_storage_path: Path) -> CheckpointStore:
    """Fixture for checkpoint store."""
    return CheckpointStore(storage_path=temp_storage_path)


@pytest.fixture
def mock_coordinator() -> MagicMock:
    """Fixture for mock RLCoordinator."""
    coordinator = MagicMock()
    return coordinator


@pytest.fixture
def mock_learner() -> MagicMock:
    """Fixture for mock learner."""
    learner = MagicMock()
    learner.name = "tool_selector"
    learner._q_values = {"ctx1": 0.7, "ctx2": 0.8}
    learner._sample_counts = {"ctx1": 10, "ctx2": 5}
    learner.export_metrics.return_value = {"total_samples": 15}
    return learner


@pytest.fixture
def manager(mock_coordinator: MagicMock, checkpoint_store: CheckpointStore) -> PolicyManager:
    """Fixture for PolicyManager."""
    return PolicyManager(coordinator=mock_coordinator, checkpoint_store=checkpoint_store)


class TestPolicyState:
    """Tests for PolicyState dataclass."""

    def test_policy_state_creation(self) -> None:
        """Test creating policy state."""
        state = PolicyState(
            learner_name="tool_selector",
            current_version="v1.0.0",
            stage=PolicyStage.PRODUCTION,
        )

        assert state.learner_name == "tool_selector"
        assert state.current_version == "v1.0.0"
        assert state.stage == PolicyStage.PRODUCTION

    def test_policy_state_defaults(self) -> None:
        """Test default values for policy state."""
        state = PolicyState(learner_name="test")

        assert state.current_version == "v0.0.0"
        assert state.stage == PolicyStage.DEVELOPMENT
        assert state.shadow_version is None
        assert state.canary_traffic == 0
        assert state.performance_baseline == {}
        assert state.auto_checkpoint_threshold == 100


class TestPolicyStage:
    """Tests for PolicyStage enum."""

    def test_policy_stages(self) -> None:
        """Test policy stage values."""
        assert PolicyStage.DEVELOPMENT.value == "development"
        assert PolicyStage.STAGING.value == "staging"
        assert PolicyStage.CANARY.value == "canary"
        assert PolicyStage.PRODUCTION.value == "production"
        assert PolicyStage.DEPRECATED.value == "deprecated"


class TestRollbackEvent:
    """Tests for RollbackEvent dataclass."""

    def test_rollback_event_creation(self) -> None:
        """Test creating rollback event."""
        event = RollbackEvent(
            learner_name="tool_selector",
            from_version="v2.0.0",
            to_version="v1.0.0",
            reason="Performance degradation",
            metrics_before={"success_rate": 0.5},
        )

        assert event.learner_name == "tool_selector"
        assert event.from_version == "v2.0.0"
        assert event.to_version == "v1.0.0"
        assert event.timestamp is not None


class TestPolicyManager:
    """Tests for PolicyManager."""

    def test_initialization(self, manager: PolicyManager) -> None:
        """Test manager initialization."""
        assert manager._policy_states == {}
        assert manager._recent_outcomes == {}
        assert manager._rollback_history == []

    def test_set_coordinator(
        self, checkpoint_store: CheckpointStore, mock_coordinator: MagicMock
    ) -> None:
        """Test setting coordinator."""
        manager = PolicyManager(checkpoint_store=checkpoint_store)
        manager.set_coordinator(mock_coordinator)

        assert manager._coordinator == mock_coordinator

    def test_get_policy_state_creates_new(self, manager: PolicyManager) -> None:
        """Test getting policy state creates new if not exists."""
        state = manager.get_policy_state("new_learner")

        assert state.learner_name == "new_learner"
        assert state.stage == PolicyStage.DEVELOPMENT

    def test_get_policy_state_returns_existing(self, manager: PolicyManager) -> None:
        """Test getting existing policy state."""
        state1 = manager.get_policy_state("learner")
        state1.stage = PolicyStage.PRODUCTION

        state2 = manager.get_policy_state("learner")

        assert state2.stage == PolicyStage.PRODUCTION

    def test_enable_auto_checkpoint(self, manager: PolicyManager) -> None:
        """Test enabling auto-checkpoint."""
        manager.enable_auto_checkpoint("learner", threshold=50)

        state = manager.get_policy_state("learner")
        assert state.auto_checkpoint_threshold == 50

    def test_record_outcome(self, manager: PolicyManager) -> None:
        """Test recording outcome."""
        manager.record_outcome("learner", success=True, quality_score=0.8)

        assert "learner" in manager._recent_outcomes
        assert len(manager._recent_outcomes["learner"]) == 1

    def test_record_outcome_updates_metrics(self, manager: PolicyManager) -> None:
        """Test recording outcome updates running metrics."""
        for _ in range(10):
            manager.record_outcome("learner", success=True, quality_score=0.9)
        for _ in range(5):
            manager.record_outcome("learner", success=False, quality_score=0.4)

        state = manager.get_policy_state("learner")
        assert state.recent_success_rate == pytest.approx(0.667, abs=0.01)

    def test_record_outcome_increments_counter(self, manager: PolicyManager) -> None:
        """Test recording outcome increments checkpoint counter."""
        manager.record_outcome("learner", success=True)
        manager.record_outcome("learner", success=True)

        state = manager.get_policy_state("learner")
        assert state.last_checkpoint_outcomes == 2

    def test_auto_checkpoint_triggered(
        self,
        manager: PolicyManager,
        mock_coordinator: MagicMock,
        mock_learner: MagicMock,
    ) -> None:
        """Test auto-checkpoint is triggered at threshold."""
        mock_coordinator.get_learner.return_value = mock_learner
        manager.enable_auto_checkpoint("tool_selector", threshold=10)

        # Record outcomes up to threshold
        for _ in range(10):
            manager.record_outcome("tool_selector", success=True, quality_score=0.8)

        # Checkpoint should have been created
        checkpoints = manager._checkpoint_store.list_checkpoints("tool_selector")
        assert len(checkpoints) >= 1

    def test_auto_checkpoint_resets_counter(
        self,
        manager: PolicyManager,
        mock_coordinator: MagicMock,
        mock_learner: MagicMock,
    ) -> None:
        """Test auto-checkpoint resets outcome counter."""
        mock_coordinator.get_learner.return_value = mock_learner
        manager.enable_auto_checkpoint("tool_selector", threshold=10)

        for _ in range(10):
            manager.record_outcome("tool_selector", success=True)

        state = manager.get_policy_state("tool_selector")
        assert state.last_checkpoint_outcomes == 0

    def test_should_rollback_insufficient_samples(self, manager: PolicyManager) -> None:
        """Test no rollback with insufficient samples."""
        for _ in range(10):
            manager.record_outcome("learner", success=False)

        assert manager.should_rollback("learner") is False

    def test_should_rollback_with_baseline_degradation(self, manager: PolicyManager) -> None:
        """Test rollback triggered by baseline degradation."""
        manager.set_performance_baseline("learner", success_rate=0.8, quality_score=0.8)

        # Record poor outcomes
        for _ in range(50):
            manager.record_outcome("learner", success=False)

        assert manager.should_rollback("learner") is True

    def test_should_rollback_with_recent_degradation(self, manager: PolicyManager) -> None:
        """Test rollback triggered by recent degradation."""
        # Good early performance
        for _ in range(50):
            manager.record_outcome("learner", success=True)
        # Poor recent performance
        for _ in range(50):
            manager.record_outcome("learner", success=False)

        assert manager.should_rollback("learner") is True

    def test_should_rollback_no_degradation(self, manager: PolicyManager) -> None:
        """Test no rollback when performance is stable."""
        for _ in range(100):
            manager.record_outcome("learner", success=True)

        assert manager.should_rollback("learner") is False

    def test_rollback(
        self,
        manager: PolicyManager,
        mock_coordinator: MagicMock,
        mock_learner: MagicMock,
    ) -> None:
        """Test manual rollback."""
        mock_coordinator.get_learner.return_value = mock_learner

        # Create checkpoints
        manager._checkpoint_store.create_checkpoint(
            "tool_selector", "v1.0.0", {"q_values": {"ctx1": 0.5}}
        )
        manager._checkpoint_store.create_checkpoint(
            "tool_selector", "v2.0.0", {"q_values": {"ctx1": 0.7}}
        )

        state = manager.get_policy_state("tool_selector")
        state.current_version = "v2.0.0"

        # Rollback
        result = manager.rollback("tool_selector", to_version="v1.0.0")

        assert result is True
        assert state.current_version == "v1.0.0"
        assert len(manager._rollback_history) == 1

    def test_rollback_to_previous(
        self,
        manager: PolicyManager,
        mock_coordinator: MagicMock,
        mock_learner: MagicMock,
    ) -> None:
        """Test rollback to previous version."""
        mock_coordinator.get_learner.return_value = mock_learner

        manager._checkpoint_store.create_checkpoint("tool_selector", "v1.0.0", {})
        manager._checkpoint_store.create_checkpoint("tool_selector", "v2.0.0", {})

        state = manager.get_policy_state("tool_selector")
        state.current_version = "v2.0.0"

        result = manager.rollback("tool_selector")

        assert result is True
        assert state.current_version == "v1.0.0"

    def test_rollback_no_previous_checkpoint(self, manager: PolicyManager) -> None:
        """Test rollback fails when no previous checkpoint."""
        result = manager.rollback("unknown_learner")
        assert result is False

    def test_rollback_records_event(
        self,
        manager: PolicyManager,
        mock_coordinator: MagicMock,
        mock_learner: MagicMock,
    ) -> None:
        """Test rollback records event."""
        mock_coordinator.get_learner.return_value = mock_learner

        manager._checkpoint_store.create_checkpoint("tool_selector", "v1.0.0", {})
        manager._checkpoint_store.create_checkpoint("tool_selector", "v2.0.0", {})

        state = manager.get_policy_state("tool_selector")
        state.current_version = "v2.0.0"
        state.recent_success_rate = 0.4

        manager.rollback("tool_selector", reason="Test rollback")

        events = manager.get_rollback_history("tool_selector")
        assert len(events) == 1
        assert events[0].reason == "Test rollback"
        assert events[0].metrics_before["success_rate"] == 0.4

    def test_auto_rollback_on_degradation(
        self,
        manager: PolicyManager,
        mock_coordinator: MagicMock,
        mock_learner: MagicMock,
    ) -> None:
        """Test auto-rollback when degradation detected."""
        mock_coordinator.get_learner.return_value = mock_learner

        # Create checkpoint first
        manager._checkpoint_store.create_checkpoint("tool_selector", "v1.0.0", {})
        manager._checkpoint_store.create_checkpoint("tool_selector", "v2.0.0", {})

        state = manager.get_policy_state("tool_selector")
        state.current_version = "v2.0.0"

        # Record good early performance
        for _ in range(50):
            manager.record_outcome("tool_selector", success=True)

        # Record poor recent performance (triggers degradation)
        for _ in range(50):
            manager.record_outcome("tool_selector", success=False)

        # Should have auto-rolled back
        assert len(manager._rollback_history) > 0

    def test_start_shadow_mode(
        self, manager: PolicyManager, checkpoint_store: CheckpointStore
    ) -> None:
        """Test starting shadow mode."""
        checkpoint_store.create_checkpoint("tool_selector", "v2.0.0", {"q": 0.8})

        result = manager.start_shadow_mode("tool_selector", "v2.0.0")

        assert result is True
        state = manager.get_policy_state("tool_selector")
        assert state.shadow_version == "v2.0.0"
        assert state.stage == PolicyStage.STAGING
        assert "tool_selector" in manager._shadow_states

    def test_start_shadow_mode_version_not_found(self, manager: PolicyManager) -> None:
        """Test starting shadow mode with non-existent version."""
        result = manager.start_shadow_mode("tool_selector", "v999")
        assert result is False

    def test_stop_shadow_mode(
        self, manager: PolicyManager, checkpoint_store: CheckpointStore
    ) -> None:
        """Test stopping shadow mode."""
        checkpoint_store.create_checkpoint("tool_selector", "v2.0.0", {})
        manager.start_shadow_mode("tool_selector", "v2.0.0")

        # Record some outcomes
        for _ in range(10):
            manager.record_outcome("tool_selector", success=True)

        comparison = manager.stop_shadow_mode("tool_selector")

        assert comparison["shadow_version"] == "v2.0.0"
        assert "tool_selector" not in manager._shadow_states
        state = manager.get_policy_state("tool_selector")
        assert state.shadow_version is None

    def test_promote_shadow(
        self,
        manager: PolicyManager,
        checkpoint_store: CheckpointStore,
        mock_coordinator: MagicMock,
        mock_learner: MagicMock,
    ) -> None:
        """Test promoting shadow version."""
        mock_coordinator.get_learner.return_value = mock_learner
        checkpoint_store.create_checkpoint("tool_selector", "v2.0.0", {"q": 0.9})

        manager.start_shadow_mode("tool_selector", "v2.0.0")
        result = manager.promote_shadow("tool_selector")

        assert result is True
        state = manager.get_policy_state("tool_selector")
        assert state.current_version == "v2.0.0"
        assert state.stage == PolicyStage.PRODUCTION

    def test_promote_shadow_not_in_shadow_mode(self, manager: PolicyManager) -> None:
        """Test promoting when not in shadow mode."""
        result = manager.promote_shadow("tool_selector")
        assert result is False

    def test_set_canary_traffic(self, manager: PolicyManager) -> None:
        """Test setting canary traffic."""
        manager.set_canary_traffic("tool_selector", 10)

        state = manager.get_policy_state("tool_selector")
        assert state.canary_traffic == 10
        assert state.stage == PolicyStage.CANARY

    def test_set_canary_traffic_zero(self, manager: PolicyManager) -> None:
        """Test setting canary traffic to zero."""
        manager.set_canary_traffic("tool_selector", 10)
        manager.set_canary_traffic("tool_selector", 0)

        state = manager.get_policy_state("tool_selector")
        assert state.canary_traffic == 0
        assert state.stage == PolicyStage.PRODUCTION

    def test_set_canary_traffic_bounds(self, manager: PolicyManager) -> None:
        """Test canary traffic is bounded 0-100."""
        manager.set_canary_traffic("tool_selector", -10)
        state = manager.get_policy_state("tool_selector")
        assert state.canary_traffic == 0

        manager.set_canary_traffic("tool_selector", 150)
        assert state.canary_traffic == 100

    def test_set_performance_baseline(self, manager: PolicyManager) -> None:
        """Test setting performance baseline."""
        manager.set_performance_baseline("learner", success_rate=0.85, quality_score=0.9)

        state = manager.get_policy_state("learner")
        assert state.performance_baseline["success_rate"] == 0.85
        assert state.performance_baseline["quality_score"] == 0.9

    def test_get_rollback_history_all(self, manager: PolicyManager) -> None:
        """Test getting all rollback history."""
        event1 = RollbackEvent("l1", "v2", "v1", "test", {})
        event2 = RollbackEvent("l2", "v2", "v1", "test", {})
        manager._rollback_history = [event1, event2]

        history = manager.get_rollback_history()
        assert len(history) == 2

    def test_get_rollback_history_filtered(self, manager: PolicyManager) -> None:
        """Test getting filtered rollback history."""
        event1 = RollbackEvent("l1", "v2", "v1", "test", {})
        event2 = RollbackEvent("l2", "v2", "v1", "test", {})
        manager._rollback_history = [event1, event2]

        history = manager.get_rollback_history("l1")
        assert len(history) == 1
        assert history[0].learner_name == "l1"

    def test_export_metrics(
        self, manager: PolicyManager, checkpoint_store: CheckpointStore
    ) -> None:
        """Test metrics export."""
        state = manager.get_policy_state("tool_selector")
        state.stage = PolicyStage.PRODUCTION

        manager.get_policy_state("model_selector")  # DEVELOPMENT by default

        metrics = manager.export_metrics()

        assert metrics["managed_policies"] == 2
        assert metrics["policies_by_stage"]["production"] == 1
        assert metrics["policies_by_stage"]["development"] == 1
        assert "checkpoint_store" in metrics

    def test_export_metrics_with_rollbacks(self, manager: PolicyManager) -> None:
        """Test metrics include rollback count."""
        manager._rollback_history = [
            RollbackEvent("l1", "v2", "v1", "test", {}),
            RollbackEvent("l2", "v2", "v1", "test", {}),
        ]

        metrics = manager.export_metrics()
        assert metrics["total_rollbacks"] == 2

    def test_export_metrics_with_shadow_mode(
        self, manager: PolicyManager, checkpoint_store: CheckpointStore
    ) -> None:
        """Test metrics include shadow mode count."""
        checkpoint_store.create_checkpoint("l1", "v2", {})
        checkpoint_store.create_checkpoint("l2", "v2", {})

        manager.start_shadow_mode("l1", "v2")
        manager.start_shadow_mode("l2", "v2")

        metrics = manager.export_metrics()
        assert metrics["shadow_mode_active"] == 2


class TestGlobalSingleton:
    """Tests for global singleton."""

    def test_get_policy_manager(self) -> None:
        """Test getting global singleton."""
        import victor.framework.rl.policy_manager as module

        module._policy_manager = None

        manager1 = get_policy_manager()
        manager2 = get_policy_manager()

        assert manager1 is manager2

    def test_singleton_with_coordinator(self, mock_coordinator: MagicMock) -> None:
        """Test singleton initialization with coordinator."""
        import victor.framework.rl.policy_manager as module

        module._policy_manager = None

        manager = get_policy_manager(coordinator=mock_coordinator)
        assert manager._coordinator == mock_coordinator
