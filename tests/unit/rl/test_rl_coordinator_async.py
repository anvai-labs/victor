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

"""Unit tests for RLCoordinator async wrapper methods.

Tests the async versions of RL coordinator methods that use asyncio.to_thread()
to avoid blocking the event loop during SQLite operations.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import patch, MagicMock

import pytest

from victor.framework.rl.coordinator import (
    RLCoordinator,
    get_rl_coordinator,
    get_rl_coordinator_async,
)
from victor.framework.rl.base import RLOutcome, RLRecommendation
from victor.core.database import reset_database, get_database


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_rl.db"


@pytest.fixture
def temp_storage_path() -> Generator[Path, None, None]:
    """Create a temporary storage path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def coordinator(
    temp_storage_path: Path, temp_db_path: Path
) -> Generator[RLCoordinator, None, None]:
    """Create a test RLCoordinator with temp database."""
    reset_database()
    get_database(temp_db_path)
    coord = RLCoordinator(storage_path=temp_storage_path, db_path=temp_db_path)
    yield coord
    reset_database()


@pytest.fixture
def sample_outcome() -> RLOutcome:
    """Create a sample outcome for testing."""
    return RLOutcome(
        provider="anthropic",
        model="claude-3-opus",
        task_type="analysis",
        success=True,
        quality_score=0.85,
        metadata={"test": True},
    )


class TestAsyncWrappers:
    """Tests for async wrapper methods."""

    @pytest.mark.asyncio
    async def test_record_outcome_async(
        self, coordinator: RLCoordinator, sample_outcome: RLOutcome
    ) -> None:
        """Test record_outcome_async offloads to thread pool."""
        # Should not block event loop
        await coordinator.record_outcome_async("continuation_patience", sample_outcome, "coding")

        # Verify outcome was recorded by checking stats
        stats = coordinator.get_stats()
        assert stats["total_outcomes"] >= 1

    @pytest.mark.asyncio
    async def test_get_recommendation_async(
        self, coordinator: RLCoordinator, sample_outcome: RLOutcome
    ) -> None:
        """Test get_recommendation_async offloads to thread pool."""
        # Record an outcome first
        coordinator.record_outcome("continuation_patience", sample_outcome, "coding")

        # Get recommendation asynchronously
        rec = await coordinator.get_recommendation_async(
            "continuation_patience",
            sample_outcome.provider,
            sample_outcome.model,
            sample_outcome.task_type,
        )

        # May return None if not enough data for confidence
        # Just verify it doesn't raise and completes
        assert rec is None or isinstance(rec, RLRecommendation)

    @pytest.mark.asyncio
    async def test_get_all_recommendations_async(self, coordinator: RLCoordinator) -> None:
        """Test get_all_recommendations_async offloads to thread pool."""
        recommendations = await coordinator.get_all_recommendations_async(
            "anthropic", "claude-3-opus", "analysis"
        )

        # Returns dict (may be empty if no learners initialized)
        assert isinstance(recommendations, dict)

    @pytest.mark.asyncio
    async def test_export_metrics_async(self, coordinator: RLCoordinator) -> None:
        """Test export_metrics_async offloads to thread pool."""
        metrics = await coordinator.export_metrics_async()

        assert isinstance(metrics, dict)
        assert "coordinator" in metrics

    @pytest.mark.asyncio
    async def test_get_stats_async(self, coordinator: RLCoordinator) -> None:
        """Test get_stats_async offloads to thread pool."""
        stats = await coordinator.get_stats_async()

        assert isinstance(stats, dict)
        assert "total_outcomes" in stats
        assert "learner_count" in stats

    @pytest.mark.asyncio
    async def test_create_prompt_rollout_experiment_async(self, coordinator: RLCoordinator) -> None:
        """Test async prompt rollout creation offloads to thread pool."""
        with patch.object(
            coordinator,
            "create_prompt_rollout_experiment",
            return_value="prompt_exp_123",
        ) as create_mock:
            experiment_id = await coordinator.create_prompt_rollout_experiment_async(
                section_name="GROUNDING_RULES",
                provider="ollama",
                treatment_hash="candidate123",
                traffic_split=0.2,
                min_samples_per_variant=25,
            )

        assert experiment_id == "prompt_exp_123"
        create_mock.assert_called_once_with(
            section_name="GROUNDING_RULES",
            provider="ollama",
            treatment_hash="candidate123",
            control_hash=None,
            traffic_split=0.2,
            min_samples_per_variant=25,
        )

    @pytest.mark.asyncio
    async def test_analyze_prompt_rollout_experiment_async(
        self, coordinator: RLCoordinator
    ) -> None:
        """Test async prompt rollout analysis offloads to thread pool."""
        with patch.object(
            coordinator,
            "analyze_prompt_rollout_experiment",
            return_value={"auto_action": "rollout"},
        ) as analyze_mock:
            report = await coordinator.analyze_prompt_rollout_experiment_async(
                section_name="GROUNDING_RULES",
                provider="ollama",
                treatment_hash="candidate123",
            )

        assert report == {"auto_action": "rollout"}
        analyze_mock.assert_called_once_with(
            section_name="GROUNDING_RULES",
            provider="ollama",
            treatment_hash="candidate123",
        )

    @pytest.mark.asyncio
    async def test_apply_prompt_rollout_recommendation_async(
        self, coordinator: RLCoordinator
    ) -> None:
        """Test async prompt rollout decision application offloads to thread pool."""
        with patch.object(
            coordinator,
            "apply_prompt_rollout_recommendation",
            return_value={"action": "rollout", "applied": True},
        ) as apply_mock:
            decision = await coordinator.apply_prompt_rollout_recommendation_async(
                section_name="GROUNDING_RULES",
                provider="ollama",
                treatment_hash="candidate123",
                dry_run=True,
            )

        assert decision == {"action": "rollout", "applied": True}
        apply_mock.assert_called_once_with(
            section_name="GROUNDING_RULES",
            provider="ollama",
            treatment_hash="candidate123",
            dry_run=True,
        )

    @pytest.mark.asyncio
    async def test_process_prompt_candidate_evaluation_suite_async(
        self, coordinator: RLCoordinator
    ) -> None:
        """Test async suite workflow helper offloads to thread pool."""
        with patch.object(
            coordinator,
            "process_prompt_candidate_evaluation_suite",
            return_value={"prompt_rollout": {"created": True}},
        ) as workflow_mock:
            workflow = await coordinator.process_prompt_candidate_evaluation_suite_async(
                {"runs": []},
                create_rollout=True,
            )

        assert workflow == {"prompt_rollout": {"created": True}}
        workflow_mock.assert_called_once_with(
            {"runs": []},
            min_pass_rate=0.5,
            promote_best=False,
            create_rollout=True,
            rollout_control_hash=None,
            rollout_traffic_split=0.1,
            rollout_min_samples_per_variant=100,
            analyze_rollout=False,
            apply_rollout_decision=False,
            rollout_decision_dry_run=False,
        )


class TestAsyncSingleton:
    """Tests for async singleton getter."""

    @pytest.mark.asyncio
    async def test_get_rl_coordinator_async_returns_coordinator(self) -> None:
        """Test get_rl_coordinator_async returns RLCoordinator instance."""
        # Reset global singleton
        import victor.framework.rl.coordinator as coord_module

        original = coord_module._rl_coordinator
        coord_module._rl_coordinator = None

        try:
            coordinator = await get_rl_coordinator_async()
            assert isinstance(coordinator, RLCoordinator)
        finally:
            # Restore original
            coord_module._rl_coordinator = original

    @pytest.mark.asyncio
    async def test_get_rl_coordinator_async_returns_same_instance(self) -> None:
        """Test get_rl_coordinator_async returns same singleton."""
        # Reset global singleton
        import victor.framework.rl.coordinator as coord_module

        original = coord_module._rl_coordinator
        coord_module._rl_coordinator = None

        try:
            coordinator1 = await get_rl_coordinator_async()
            coordinator2 = await get_rl_coordinator_async()
            assert coordinator1 is coordinator2
        finally:
            coord_module._rl_coordinator = original

    @pytest.mark.asyncio
    async def test_async_and_sync_return_same_instance(self) -> None:
        """Test async and sync getters share the same singleton."""
        # Reset global singleton
        import victor.framework.rl.coordinator as coord_module

        original = coord_module._rl_coordinator
        coord_module._rl_coordinator = None

        try:
            # Get via async first
            coordinator_async = await get_rl_coordinator_async()
            # Then get via sync
            coordinator_sync = get_rl_coordinator()

            assert coordinator_async is coordinator_sync
        finally:
            coord_module._rl_coordinator = original


class TestConcurrentAccess:
    """Tests for concurrent async access."""

    @pytest.mark.asyncio
    async def test_sequential_record_outcomes(self, coordinator: RLCoordinator) -> None:
        """Test sequential record_outcome_async calls complete without error."""
        outcomes = [
            RLOutcome(
                provider="anthropic",
                model="claude-3-opus",
                task_type="analysis",
                success=True,
                quality_score=0.8 + i * 0.02,
            )
            for i in range(3)
        ]

        # Run sequentially (SQLite doesn't handle concurrent writes well)
        for outcome in outcomes:
            await coordinator.record_outcome_async("continuation_patience", outcome)

        # Verify all were recorded
        stats = coordinator.get_stats()
        assert stats["total_outcomes"] >= 3

    @pytest.mark.asyncio
    async def test_concurrent_get_recommendations(
        self, coordinator: RLCoordinator, sample_outcome: RLOutcome
    ) -> None:
        """Test multiple concurrent get_recommendation_async calls."""
        # Record some data first
        coordinator.record_outcome("continuation_patience", sample_outcome)

        # Run concurrent gets
        results = await asyncio.gather(
            *[
                coordinator.get_recommendation_async(
                    "continuation_patience",
                    sample_outcome.provider,
                    sample_outcome.model,
                    sample_outcome.task_type,
                )
                for _ in range(5)
            ]
        )

        # All should complete (may be None due to low sample count)
        assert len(results) == 5


class TestEventLoopNonBlocking:
    """Tests verifying async methods don't block event loop."""

    @pytest.mark.asyncio
    async def test_record_outcome_allows_concurrent_tasks(
        self, coordinator: RLCoordinator, sample_outcome: RLOutcome
    ) -> None:
        """Test record_outcome_async allows other tasks to run."""
        other_task_ran = False

        async def other_task():
            nonlocal other_task_ran
            await asyncio.sleep(0)  # Yield to event loop
            other_task_ran = True

        # Run both concurrently
        await asyncio.gather(
            coordinator.record_outcome_async("continuation_patience", sample_outcome),
            other_task(),
        )

        assert other_task_ran

    @pytest.mark.asyncio
    async def test_export_metrics_allows_concurrent_tasks(self, coordinator: RLCoordinator) -> None:
        """Test export_metrics_async allows other tasks to run."""
        counter = 0

        async def increment_task():
            nonlocal counter
            await asyncio.sleep(0)
            counter += 1

        # Run multiple increments alongside export
        await asyncio.gather(
            coordinator.export_metrics_async(),
            increment_task(),
            increment_task(),
            increment_task(),
        )

        assert counter == 3


class TestActiveLearners:
    """Tests for active learners enforcement."""

    def test_no_restriction_by_default(self, coordinator: RLCoordinator) -> None:
        """Test that get_active_learner_names() returns None by default."""
        assert coordinator.get_active_learner_names() is None

    def test_set_active_learners_restricts_creation(self, coordinator: RLCoordinator) -> None:
        """Test that unlisted learner returns None after set_active_learners."""
        coordinator.set_active_learners(["continuation_patience"])

        # Listed learner should be creatable
        learner = coordinator.get_learner("continuation_patience")
        assert learner is not None

        # Unlisted learner should return None (not created)
        learner = coordinator.get_learner("model_selector")
        assert learner is None

    def test_already_created_learners_still_accessible(self, coordinator: RLCoordinator) -> None:
        """Test that pre-existing learners survive restriction."""
        # Create learner before setting restriction
        learner = coordinator.get_learner("continuation_patience")
        assert learner is not None

        # Now restrict to a different set
        coordinator.set_active_learners(["model_selector"])

        # Pre-existing learner should still be accessible
        same_learner = coordinator.get_learner("continuation_patience")
        assert same_learner is learner

    def test_record_outcome_skips_inactive(
        self, coordinator: RLCoordinator, sample_outcome: RLOutcome
    ) -> None:
        """Test that recording for inactive learner doesn't error."""
        coordinator.set_active_learners(["continuation_patience"])

        # Recording for inactive learner should not raise
        coordinator.record_outcome("model_selector", sample_outcome, "coding")
