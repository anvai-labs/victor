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

"""Unit tests for MultiTaskLearner.

Tests the multi-task learning coordinator for cross-vertical transfer.
"""

import sqlite3
import pytest
from unittest.mock import MagicMock, patch

from victor.framework.rl.multi_task_learner import (
    MultiTaskLearner,
    VerticalHead,
)
from victor.framework.rl.base import RLOutcome


@pytest.fixture
def db_connection():
    """Create in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def learner(db_connection):
    """Create MultiTaskLearner instance."""
    return MultiTaskLearner(
        name="multi_task",
        db_connection=db_connection,
        learning_rate=0.1,
        transfer_rate=0.3,
    )


class TestVerticalHead:
    """Tests for VerticalHead."""

    def test_vertical_head_creation(self) -> None:
        """Test creating vertical head."""
        head = VerticalHead(vertical="coding")

        assert head.vertical == "coding"
        assert head.q_values == {}
        assert head.sample_counts == {}
        assert head.success_rates == {}

    def test_get_q_value_default(self) -> None:
        """Test getting Q-value with default."""
        head = VerticalHead(vertical="coding")

        assert head.get_q_value("test_key") == 0.5
        assert head.get_q_value("test_key", default=0.7) == 0.7

    def test_get_q_value_existing(self) -> None:
        """Test getting existing Q-value."""
        head = VerticalHead(vertical="coding")
        head.q_values["test_key"] = 0.8

        assert head.get_q_value("test_key") == 0.8

    def test_update_q_value(self) -> None:
        """Test updating Q-value."""
        head = VerticalHead(vertical="coding")

        head.update("test_key", reward=1.0, learning_rate=0.1)

        assert "test_key" in head.q_values
        assert head.sample_counts["test_key"] == 1
        assert head.success_rates["test_key"] == 1.0

    def test_update_q_value_multiple_times(self) -> None:
        """Test multiple Q-value updates."""
        head = VerticalHead(vertical="coding")

        head.update("test_key", reward=1.0, learning_rate=0.1)
        head.update("test_key", reward=0.0, learning_rate=0.1)

        assert head.sample_counts["test_key"] == 2
        # Success rate should be 0.5 (1 success, 1 failure)
        assert head.success_rates["test_key"] == 0.5


class TestMultiTaskLearner:
    """Tests for MultiTaskLearner."""

    def test_initialization(self, learner: MultiTaskLearner) -> None:
        """Test learner initialization."""
        assert learner.name == "multi_task"
        assert learner.learning_rate == 0.1
        assert learner.transfer_rate == 0.3
        assert learner._heads == {}
        assert learner._global_q_values == {}

    def test_record_outcome_creates_head(self, learner: MultiTaskLearner) -> None:
        """Test recording outcome creates vertical head."""
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-3",
            task_type="code_generation",
            success=True,
            quality_score=0.8,
            vertical="coding",
        )

        learner.record_outcome(outcome)

        assert "coding" in learner._heads
        head = learner._heads["coding"]
        assert len(head.q_values) > 0

    def test_record_outcome_updates_global(self, learner: MultiTaskLearner) -> None:
        """Test recording outcome updates global Q-values."""
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-3",
            task_type="code_generation",
            success=True,
            quality_score=0.8,
            vertical="coding",
        )

        learner.record_outcome(outcome)

        assert len(learner._global_q_values) > 0
        assert len(learner._global_sample_counts) > 0

    def test_compute_reward_success(self, learner: MultiTaskLearner) -> None:
        """Test reward computation for success."""
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-3",
            task_type="code_generation",
            success=True,
            quality_score=1.0,
        )

        reward = learner._compute_reward(outcome)

        assert reward > 0

    def test_compute_reward_failure(self, learner: MultiTaskLearner) -> None:
        """Test reward computation for failure."""
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-3",
            task_type="code_generation",
            success=False,
            quality_score=0.0,
        )

        reward = learner._compute_reward(outcome)

        assert reward < 0

    def test_get_recommendation_no_data(self, learner: MultiTaskLearner) -> None:
        """Test recommendation with no data."""
        rec = learner.get_recommendation(
            provider="anthropic",
            model="claude-3",
            task_type="code_generation",
            vertical="coding",
        )

        assert rec is not None
        assert rec.value == 0.5
        assert rec.is_baseline is True
        assert rec.sample_size == 0

    def test_get_recommendation_with_data(self, learner: MultiTaskLearner) -> None:
        """Test recommendation with learned data."""
        # Record some outcomes
        for _ in range(5):
            outcome = RLOutcome(
                provider="anthropic",
                model="claude-3",
                task_type="code_generation",
                success=True,
                quality_score=0.9,
                vertical="coding",
            )
            learner.record_outcome(outcome)

        rec = learner.get_recommendation(
            provider="anthropic",
            model="claude-3",
            task_type="code_generation",
            vertical="coding",
        )

        assert rec is not None
        assert rec.value > 0.5  # Should be positive from successes
        assert rec.sample_size > 0
        assert rec.is_baseline is False

    def test_transfer_learning_between_verticals(self, learner: MultiTaskLearner) -> None:
        """Test transfer learning updates across verticals."""
        # Record outcomes in coding vertical
        for _ in range(10):
            outcome = RLOutcome(
                provider="anthropic",
                model="claude-3",
                task_type="code_generation",
                success=True,
                quality_score=0.9,
                vertical="coding",
            )
            learner.record_outcome(outcome)

        # Now record in devops vertical (should transfer knowledge)
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-3",
            task_type="code_generation",
            success=True,
            quality_score=0.8,
            vertical="devops",
        )
        learner.record_outcome(outcome)

        # Both verticals should exist
        assert "coding" in learner._heads
        assert "devops" in learner._heads

    def test_get_transferred_q_no_data(self, learner: MultiTaskLearner) -> None:
        """Test transferred Q-value with no data."""
        q_value, samples = learner._get_transferred_q(
            provider="anthropic",
            model="claude-3",
            task_type="code_generation",
            target_vertical="coding",
        )

        assert q_value is None
        assert samples == 0

    def test_compute_transfer_weight_high_similarity(self, learner: MultiTaskLearner) -> None:
        """Test transfer weight for high similarity."""
        from victor.framework.rl.shared_encoder import ContextEmbedding

        # Create embeddings with same vector (identical contexts)
        source = ContextEmbedding(
            task_type="code_generation",
            provider="anthropic",
            model="claude-3",
            vertical="coding",
            vector=[0.9, 0.8, 0.7, 0.6],
        )
        target = ContextEmbedding(
            task_type="code_generation",
            provider="anthropic",
            model="claude-3",
            vertical="coding",
            vector=[0.9, 0.8, 0.7, 0.6],
        )

        weight = learner._compute_transfer_weight(source, target)

        # Same embeddings should have high transfer (similarity = 1.0)
        assert weight > 0

    def test_compute_confidence(self, learner: MultiTaskLearner) -> None:
        """Test confidence computation."""
        # Zero samples
        assert learner._compute_confidence(0) == 0.3

        # Few samples
        conf_5 = learner._compute_confidence(5)
        assert 0.3 < conf_5 < 0.95

        # Many samples
        conf_100 = learner._compute_confidence(100)
        assert conf_100 > conf_5
        assert conf_100 <= 0.95

    def test_get_vertical_stats_empty(self, learner: MultiTaskLearner) -> None:
        """Test vertical stats for empty vertical."""
        stats = learner.get_vertical_stats("coding")

        assert stats["vertical"] == "coding"
        assert stats["contexts"] == 0
        assert stats["total_samples"] == 0

    def test_get_vertical_stats_with_data(self, learner: MultiTaskLearner) -> None:
        """Test vertical stats with data."""
        # Record some outcomes
        for _ in range(5):
            outcome = RLOutcome(
                provider="anthropic",
                model="claude-3",
                task_type="code_generation",
                success=True,
                quality_score=0.9,
                vertical="coding",
            )
            learner.record_outcome(outcome)

        stats = learner.get_vertical_stats("coding")

        assert stats["vertical"] == "coding"
        assert stats["contexts"] > 0
        assert stats["total_samples"] == 5

    def test_get_transfer_stats(self, learner: MultiTaskLearner) -> None:
        """Test transfer statistics."""
        stats = learner.get_transfer_stats()

        assert "verticals" in stats
        assert "transfer_rate" in stats
        assert stats["transfer_rate"] == 0.3

    def test_export_metrics(self, learner: MultiTaskLearner) -> None:
        """Test exporting metrics."""
        # Record some data
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-3",
            task_type="code_generation",
            success=True,
            quality_score=0.8,
            vertical="coding",
        )
        learner.record_outcome(outcome)

        metrics = learner.export_metrics()

        assert metrics["learner"] == "multi_task"
        assert metrics["verticals"] >= 1
        assert "vertical_stats" in metrics
        assert "transfer_stats" in metrics


class TestMultiTaskLearnerPersistence:
    """Tests for database persistence."""

    def test_state_persists_to_database(self, learner: MultiTaskLearner) -> None:
        """Test state is saved to database."""
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-3",
            task_type="code_generation",
            success=True,
            quality_score=0.8,
            vertical="coding",
        )
        learner.record_outcome(outcome)

        # Check database has data
        cursor = learner.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM multi_task_vertical_q")
        count = cursor.fetchone()[0]
        assert count > 0

    def test_state_loads_from_database(self, db_connection) -> None:
        """Test state loads correctly on initialization."""
        # Create first learner and record outcome
        learner1 = MultiTaskLearner(
            name="multi_task",
            db_connection=db_connection,
            learning_rate=0.1,
        )
        outcome = RLOutcome(
            provider="anthropic",
            model="claude-3",
            task_type="code_generation",
            success=True,
            quality_score=0.8,
            vertical="coding",
        )
        learner1.record_outcome(outcome)

        # Create second learner with same connection
        learner2 = MultiTaskLearner(
            name="multi_task",
            db_connection=db_connection,
            learning_rate=0.1,
        )

        # Should have loaded state
        assert "coding" in learner2._heads


class TestVerticalHeadInteraction:
    """Tests for vertical head interaction."""

    def test_multiple_verticals(self, learner: MultiTaskLearner) -> None:
        """Test recording to multiple verticals."""
        verticals = ["coding", "devops", "data_science"]

        for vertical in verticals:
            outcome = RLOutcome(
                provider="anthropic",
                model="claude-3",
                task_type="code_generation",
                success=True,
                quality_score=0.8,
                vertical=vertical,
            )
            learner.record_outcome(outcome)

        assert len(learner._heads) == 3
        for vertical in verticals:
            assert vertical in learner._heads

    def test_different_task_types_same_vertical(self, learner: MultiTaskLearner) -> None:
        """Test different task types in same vertical."""
        task_types = ["code_generation", "code_review", "debugging"]

        for task_type in task_types:
            outcome = RLOutcome(
                provider="anthropic",
                model="claude-3",
                task_type=task_type,
                success=True,
                quality_score=0.8,
                vertical="coding",
            )
            learner.record_outcome(outcome)

        head = learner._heads["coding"]
        assert len(head.q_values) == 3

    def test_recommendation_blends_verticals(self, learner: MultiTaskLearner) -> None:
        """Test recommendation blends vertical and transferred knowledge."""
        # Build up coding vertical
        for _ in range(20):
            outcome = RLOutcome(
                provider="anthropic",
                model="claude-3",
                task_type="code_generation",
                success=True,
                quality_score=0.9,
                vertical="coding",
            )
            learner.record_outcome(outcome)

        # Get recommendation for devops (should use transfer)
        rec = learner.get_recommendation(
            provider="anthropic",
            model="claude-3",
            task_type="code_generation",
            vertical="devops",
        )

        # Should have some data from transfer
        assert rec is not None
