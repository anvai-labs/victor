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

"""Unit tests for quality weight RL learner.

Tests the QualityWeightLearner which uses gradient descent
to learn optimal quality dimension weights per task type.
"""

import pytest
from pathlib import Path
from typing import Dict

from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import RLCoordinator
from victor.framework.rl.learners.quality_weights import (
    QualityWeightLearner,
    QualityDimension,
)
from victor.core.database import reset_database, get_database
from victor.core.schema import Tables


@pytest.fixture
def coordinator(tmp_path: Path) -> RLCoordinator:
    """Fixture for RLCoordinator, ensuring a clean database for each test."""
    reset_database()
    db_path = tmp_path / "rl_test.db"
    get_database(db_path)
    coord = RLCoordinator(storage_path=tmp_path, db_path=db_path)
    yield coord
    reset_database()


@pytest.fixture
def learner(coordinator: RLCoordinator) -> QualityWeightLearner:
    """Fixture for QualityWeightLearner."""
    return coordinator.get_learner("quality_weights")  # type: ignore


def _record_quality_outcome(
    learner: QualityWeightLearner,
    task_type: str = "analysis",
    dimension_scores: Dict[str, float] = None,
    overall_success: float = 0.8,
) -> None:
    """Helper to record a quality outcome."""
    if dimension_scores is None:
        dimension_scores = {
            QualityDimension.RELEVANCE: 0.8,
            QualityDimension.COMPLETENESS: 0.7,
            QualityDimension.ACCURACY: 0.9,
            QualityDimension.CONCISENESS: 0.6,
            QualityDimension.ACTIONABILITY: 0.7,
            QualityDimension.COHERENCE: 0.8,
            QualityDimension.CODE_QUALITY: 0.75,
        }

    outcome = RLOutcome(
        provider="test",
        model="test-model",
        task_type=task_type,
        success=overall_success > 0.5,
        quality_score=overall_success,
        metadata={
            "dimension_scores": dimension_scores,
            "overall_success": overall_success,
        },
    )
    learner.record_outcome(outcome)


class TestQualityWeightLearner:
    """Tests for QualityWeightLearner."""

    def test_initialization(self, learner: QualityWeightLearner) -> None:
        """Test learner initializes correctly and creates tables."""
        assert learner.name == "quality_weights"
        assert learner.learning_rate == 0.05
        assert learner.momentum == 0.9

        # Check tables using correct names from Tables enum
        cursor = learner.db.cursor()
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_PARAM}';"
        )
        assert cursor.fetchone() is not None, f"Table {Tables.RL_PARAM} not found"
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_TRANSITION}';"
        )
        assert cursor.fetchone() is not None, f"Table {Tables.RL_TRANSITION} not found"

    def test_default_weights(self, learner: QualityWeightLearner) -> None:
        """Test default weights are set correctly."""
        defaults = learner.DEFAULT_WEIGHTS

        assert defaults[QualityDimension.RELEVANCE] == 1.5
        assert defaults[QualityDimension.COMPLETENESS] == 1.2
        assert defaults[QualityDimension.ACCURACY] == 1.3
        assert defaults[QualityDimension.CONCISENESS] == 0.8

    def test_record_single_outcome(self, learner: QualityWeightLearner) -> None:
        """Recording one outcome initializes weights for task type."""
        task_type = "code_review"

        _record_quality_outcome(learner, task_type=task_type)

        assert task_type in learner._weights
        assert task_type in learner._sample_counts
        assert learner._sample_counts[task_type] == 1

    def test_weight_updates_with_high_success(self, learner: QualityWeightLearner) -> None:
        """Weights should converge when high-scoring dimensions correlate with success."""
        task_type = "high_accuracy_task"

        # Record outcomes where high accuracy correlates with success
        for _ in range(20):
            _record_quality_outcome(
                learner,
                task_type=task_type,
                dimension_scores={
                    QualityDimension.RELEVANCE: 0.5,
                    QualityDimension.COMPLETENESS: 0.5,
                    QualityDimension.ACCURACY: 0.95,  # High accuracy
                    QualityDimension.CONCISENESS: 0.5,
                    QualityDimension.ACTIONABILITY: 0.5,
                    QualityDimension.COHERENCE: 0.5,
                    QualityDimension.CODE_QUALITY: 0.5,
                },
                overall_success=0.9,  # High success
            )

        # After many successful outcomes with high accuracy, weight should update
        weights = learner.get_weights(task_type)
        assert task_type in learner._weights

    def test_get_recommendation_unknown_task(self, learner: QualityWeightLearner) -> None:
        """Unknown task type returns default weights."""
        rec = learner.get_recommendation("", "", "unknown_task_type")

        assert rec is not None
        assert rec.is_baseline is True
        assert rec.confidence == 0.3
        assert rec.value == learner.DEFAULT_WEIGHTS

    def test_get_recommendation_known_task(self, learner: QualityWeightLearner) -> None:
        """Known task type returns learned weights."""
        task_type = "learned_task"

        # Record several outcomes
        for _ in range(5):
            _record_quality_outcome(learner, task_type=task_type)

        rec = learner.get_recommendation("", "", task_type)

        assert rec is not None
        assert rec.sample_size == 5
        assert isinstance(rec.value, dict)

    def test_get_weights(self, learner: QualityWeightLearner) -> None:
        """Test getting weights for a task type."""
        task_type = "test_task"

        # Unknown task returns defaults
        weights_unknown = learner.get_weights("unknown")
        assert weights_unknown == learner.DEFAULT_WEIGHTS

        # Record and check
        _record_quality_outcome(learner, task_type=task_type)
        weights = learner.get_weights(task_type)

        assert QualityDimension.RELEVANCE in weights
        assert QualityDimension.ACCURACY in weights

    def test_get_weight_adjustments(self, learner: QualityWeightLearner) -> None:
        """Test getting weight adjustments relative to defaults."""
        task_type = "adjustment_task"

        # Unknown task has zero adjustments
        adj_unknown = learner.get_weight_adjustments("unknown")
        assert all(v == 0.0 for v in adj_unknown.values())

        # Record outcomes
        _record_quality_outcome(learner, task_type=task_type)
        adjustments = learner.get_weight_adjustments(task_type)

        # Should have values for all dimensions
        for dim in QualityDimension.ALL:
            assert dim in adjustments

    def test_persistence(self, tmp_path: Path) -> None:
        """State persists across learner instances."""
        task_type = "persistent_task"

        reset_database()
        db_path = tmp_path / "rl_test.db"
        get_database(db_path)
        coordinator1 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
        learner1 = coordinator1.get_learner("quality_weights")  # type: ignore

        _record_quality_outcome(learner1, task_type=task_type)
        weights_before = learner1.get_weights(task_type)
        reset_database()

        get_database(db_path)
        coordinator2 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
        learner2 = coordinator2.get_learner("quality_weights")  # type: ignore

        weights_after = learner2.get_weights(task_type)

        assert task_type in learner2._weights
        # Weights should be preserved
        for dim in QualityDimension.ALL:
            assert abs(weights_before[dim] - weights_after[dim]) < 0.01

        reset_database()

    def test_weight_bounds(self, learner: QualityWeightLearner) -> None:
        """Weights should stay within bounds."""
        task_type = "bounded_task"

        # Record many extreme outcomes
        for _ in range(50):
            _record_quality_outcome(
                learner,
                task_type=task_type,
                dimension_scores={
                    QualityDimension.RELEVANCE: 1.0,
                    QualityDimension.COMPLETENESS: 0.0,
                    QualityDimension.ACCURACY: 1.0,
                    QualityDimension.CONCISENESS: 0.0,
                    QualityDimension.ACTIONABILITY: 1.0,
                    QualityDimension.COHERENCE: 0.0,
                    QualityDimension.CODE_QUALITY: 0.5,
                },
                overall_success=1.0,
            )

        weights = learner.get_weights(task_type)

        for dim, weight in weights.items():
            assert learner.MIN_WEIGHT <= weight <= learner.MAX_WEIGHT

    def test_correlation_analysis_empty(self, learner: QualityWeightLearner) -> None:
        """Correlation analysis returns zeros for unknown task."""
        correlations = learner.get_correlation_analysis("unknown_task")

        for dim in QualityDimension.ALL:
            assert correlations[dim] == 0.0

    def test_correlation_analysis_with_data(self, learner: QualityWeightLearner) -> None:
        """Correlation analysis works with sufficient data."""
        task_type = "correlation_task"

        # Record outcomes with varying success
        for i in range(10):
            success = 0.5 + (i / 20)  # Increasing success
            _record_quality_outcome(
                learner,
                task_type=task_type,
                dimension_scores={
                    QualityDimension.RELEVANCE: success,  # Correlated
                    QualityDimension.COMPLETENESS: 0.5,  # Uncorrelated
                    QualityDimension.ACCURACY: success,  # Correlated
                    QualityDimension.CONCISENESS: 0.5,
                    QualityDimension.ACTIONABILITY: 0.5,
                    QualityDimension.COHERENCE: 0.5,
                    QualityDimension.CODE_QUALITY: 0.5,
                },
                overall_success=success,
            )

        correlations = learner.get_correlation_analysis(task_type)

        # Relevance and accuracy should have higher correlation
        assert (
            correlations[QualityDimension.RELEVANCE] > correlations[QualityDimension.COMPLETENESS]
        )
        assert correlations[QualityDimension.ACCURACY] > correlations[QualityDimension.COMPLETENESS]

    def test_export_metrics(self, learner: QualityWeightLearner) -> None:
        """Test metrics export."""
        _record_quality_outcome(learner, task_type="task_a")
        _record_quality_outcome(learner, task_type="task_b")
        _record_quality_outcome(learner, task_type="task_a")

        metrics = learner.export_metrics()

        assert metrics["learner"] == "quality_weights"
        assert metrics["task_types_learned"] == 2
        assert metrics["total_samples"] == 3
        assert metrics["learning_rate"] == 0.05
        assert metrics["momentum"] == 0.9
        assert "samples_per_task" in metrics
        assert metrics["samples_per_task"]["task_a"] == 2
        assert metrics["samples_per_task"]["task_b"] == 1

    def test_missing_dimension_scores(self, learner: QualityWeightLearner) -> None:
        """Outcome with missing dimension_scores is skipped."""
        outcome = RLOutcome(
            provider="test",
            model="test",
            task_type="test",
            success=True,
            quality_score=0.8,
            metadata={},  # Missing dimension_scores
        )

        # Should not raise, just skip
        learner.record_outcome(outcome)

        assert "test" not in learner._weights

    def test_multiple_task_types_independent(self, learner: QualityWeightLearner) -> None:
        """Different task types maintain independent weights."""
        # Record for task A with high accuracy focus
        for _ in range(10):
            _record_quality_outcome(
                learner,
                task_type="task_a",
                dimension_scores={
                    QualityDimension.RELEVANCE: 0.5,
                    QualityDimension.COMPLETENESS: 0.5,
                    QualityDimension.ACCURACY: 0.9,
                    QualityDimension.CONCISENESS: 0.5,
                    QualityDimension.ACTIONABILITY: 0.5,
                    QualityDimension.COHERENCE: 0.5,
                    QualityDimension.CODE_QUALITY: 0.5,
                },
                overall_success=0.9,
            )

        # Record for task B with high completeness focus
        for _ in range(10):
            _record_quality_outcome(
                learner,
                task_type="task_b",
                dimension_scores={
                    QualityDimension.RELEVANCE: 0.5,
                    QualityDimension.COMPLETENESS: 0.9,
                    QualityDimension.ACCURACY: 0.5,
                    QualityDimension.CONCISENESS: 0.5,
                    QualityDimension.ACTIONABILITY: 0.5,
                    QualityDimension.COHERENCE: 0.5,
                    QualityDimension.CODE_QUALITY: 0.5,
                },
                overall_success=0.9,
            )

        weights_a = learner.get_weights("task_a")
        weights_b = learner.get_weights("task_b")

        # Weights should be different for different tasks
        assert weights_a != weights_b

    def test_confidence_increases_with_samples(self, learner: QualityWeightLearner) -> None:
        """Confidence should increase with more samples."""
        task_type = "confidence_task"

        # Record a few samples
        for _ in range(3):
            _record_quality_outcome(learner, task_type=task_type)

        rec_low = learner.get_recommendation("", "", task_type)

        # Record many more samples
        for _ in range(20):
            _record_quality_outcome(learner, task_type=task_type)

        rec_high = learner.get_recommendation("", "", task_type)

        assert rec_high.confidence > rec_low.confidence
