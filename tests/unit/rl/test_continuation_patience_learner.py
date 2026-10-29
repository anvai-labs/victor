"""Unit tests for continuation patience RL learner."""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import RLCoordinator
from victor.framework.rl.learners.continuation_patience import (
    ContinuationPatienceLearner,
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
def learner(coordinator: RLCoordinator) -> ContinuationPatienceLearner:
    """Fixture for ContinuationPatienceLearner."""
    return coordinator.get_learner("continuation_patience")  # type: ignore


def _record_patience_outcome(
    learner: ContinuationPatienceLearner,
    provider: str = "ollama",
    model: str = "test-model",
    task_type: str = "analysis",
    *,
    continuation_prompts: int = 3,
    patience_threshold: int = 3,
    flagged_as_stuck: bool = False,
    actually_stuck: bool = False,
    eventually_made_progress: bool = False,
    success: bool = True,
    quality_score: float = 0.8,
) -> None:
    """Helper to record a single continuation patience outcome."""
    outcome = RLOutcome(
        provider=provider,
        model=model,
        task_type=task_type,
        success=success,
        quality_score=quality_score,
        metadata={
            "continuation_prompts": continuation_prompts,
            "patience_threshold": patience_threshold,
            "flagged_as_stuck": flagged_as_stuck,
            "actually_stuck": actually_stuck,
            "eventually_made_progress": eventually_made_progress,
        },
    )
    learner.record_outcome(outcome)


def _get_patience_stats(
    coordinator: RLCoordinator,
    provider: str,
    model: str,
    task_type: str,
):
    """Helper to retrieve raw stats from the database."""
    cursor = coordinator.db.cursor()
    context_key = f"{provider}:{model}:{task_type}"
    rows = cursor.execute(
        f"SELECT stat_key, stat_value, sample_count FROM {Tables.RL_TASK_STAT} "
        f"WHERE learner_id = 'continuation_patience' AND task_type = ?",
        (context_key,),
    ).fetchall()
    if not rows:
        return None
    result: dict = {"context_key": context_key}
    for stat_key, stat_value, sample_count in rows:
        result[stat_key] = stat_value
    return result


class TestContinuationPatienceLearner:
    def test_initialization(self, learner: ContinuationPatienceLearner) -> None:
        """Test learner initializes correctly and creates tables."""
        assert learner.name == "continuation_patience"
        cursor = learner.db.cursor()
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_TASK_STAT}';"
        )
        assert cursor.fetchone() is not None

    def test_record_outcome_false_positive(
        self, coordinator: RLCoordinator, learner: ContinuationPatienceLearner
    ) -> None:
        """Test recording an outcome with a false positive stuck flag."""
        _record_patience_outcome(
            learner,
            flagged_as_stuck=True,
            actually_stuck=False,
            eventually_made_progress=True,
        )

        stats = _get_patience_stats(coordinator, "ollama", "test-model", "analysis")
        assert stats is not None
        assert stats["total_sessions"] == 1
        assert stats["false_positives"] == 1
        assert stats["true_positives"] == 0
        assert stats["missed_stuck_loops"] == 0

    def test_record_outcome_true_positive(
        self, coordinator: RLCoordinator, learner: ContinuationPatienceLearner
    ) -> None:
        """Test recording an outcome with a true positive stuck flag."""
        _record_patience_outcome(
            learner,
            flagged_as_stuck=True,
            actually_stuck=True,
            eventually_made_progress=False,
        )

        stats = _get_patience_stats(coordinator, "ollama", "test-model", "analysis")
        assert stats is not None
        assert stats["total_sessions"] == 1
        assert stats["false_positives"] == 0
        assert stats["true_positives"] == 1
        assert stats["missed_stuck_loops"] == 0

    def test_record_outcome_missed_stuck_loop(
        self, coordinator: RLCoordinator, learner: ContinuationPatienceLearner
    ) -> None:
        """Test recording an outcome where a stuck loop was missed."""
        _record_patience_outcome(
            learner,
            flagged_as_stuck=False,
            actually_stuck=True,
            eventually_made_progress=False,
        )

        stats = _get_patience_stats(coordinator, "ollama", "test-model", "analysis")
        assert stats is not None
        assert stats["total_sessions"] == 1
        assert stats["false_positives"] == 0
        assert stats["true_positives"] == 0
        assert stats["missed_stuck_loops"] == 1

    def test_get_recommendation_insufficient_data(
        self, learner: ContinuationPatienceLearner
    ) -> None:
        """Should return a baseline recommendation with insufficient data."""
        _record_patience_outcome(learner)  # Only one session
        rec = learner.get_recommendation("ollama", "test-model", "analysis")
        assert rec is not None
        assert rec.is_baseline is True
        assert rec.reason.startswith("Insufficient data")
        assert rec.sample_size == 1

    def test_get_recommendation_with_baseline(self, learner: ContinuationPatienceLearner) -> None:
        """Should return baseline if no data, and provider adapter is mocked."""
        # Mock provider adapter to return a baseline patience
        mock_adapter = MagicMock()
        mock_adapter.capabilities.continuation_patience = 5
        learner.provider_adapter = mock_adapter

        rec = learner.get_recommendation("ollama", "test-model", "analysis")
        assert rec is not None
        assert rec.value == 5
        assert rec.is_baseline is True
        assert rec.confidence == 0.0

    def test_patience_increases_with_false_positives(
        self, coordinator: RLCoordinator, learner: ContinuationPatienceLearner
    ) -> None:
        """Patience should increase if many false positives."""
        initial_patience = 3
        # Ensure the baseline is set for consistency in this test
        mock_adapter = MagicMock()
        mock_adapter.capabilities.continuation_patience = initial_patience
        learner.provider_adapter = mock_adapter

        # Record enough false positives to trigger an increase
        for i in range(10):
            _record_patience_outcome(
                learner,
                flagged_as_stuck=True,
                actually_stuck=False,
                eventually_made_progress=True,
                patience_threshold=initial_patience,  # Ensure patience threshold is consistent
            )

        # After 5 sessions, update_patience would have been called
        stats = _get_patience_stats(coordinator, "ollama", "test-model", "analysis")
        assert stats["false_positives"] == 10
        assert stats["total_sessions"] == 10

        rec = learner.get_recommendation("ollama", "test-model", "analysis")
        assert rec is not None
        assert rec.value > initial_patience  # Expect patience to increase
        assert rec.confidence > 0.0

    def test_patience_decreases_with_missed_stuck_loops(
        self, coordinator: RLCoordinator, learner: ContinuationPatienceLearner
    ) -> None:
        """Patience should decrease if many stuck loops are missed."""
        initial_patience = 5
        # Ensure the baseline is set for consistency in this test
        mock_adapter = MagicMock()
        mock_adapter.capabilities.continuation_patience = initial_patience
        learner.provider_adapter = mock_adapter

        # Record enough missed stuck loops
        for i in range(10):
            _record_patience_outcome(
                learner,
                flagged_as_stuck=False,
                actually_stuck=True,
                eventually_made_progress=False,
                patience_threshold=initial_patience,  # Ensure patience threshold is consistent
            )

        # After 5 sessions, update_patience would have been called
        stats = _get_patience_stats(coordinator, "ollama", "test-model", "analysis")
        assert stats["missed_stuck_loops"] == 10
        assert stats["total_sessions"] == 10

        rec = learner.get_recommendation("ollama", "test-model", "analysis")
        assert rec is not None
        assert rec.value < initial_patience  # Expect patience to decrease
        assert rec.confidence > 0.0

    def test_patience_stays_within_bounds(
        self, coordinator: RLCoordinator, learner: ContinuationPatienceLearner
    ) -> None:
        """Patience recommendation should stay within [1, 15] bounds."""
        mock_adapter = MagicMock()
        mock_adapter.capabilities.continuation_patience = 1
        learner.provider_adapter = mock_adapter

        # Force a decrease below 1
        for _ in range(10):
            _record_patience_outcome(
                learner,
                flagged_as_stuck=False,
                actually_stuck=True,
                eventually_made_progress=False,
                patience_threshold=1,
            )
        rec = learner.get_recommendation("ollama", "test-model", "analysis")
        assert rec.value == 1  # Should not go below 1

        mock_adapter.capabilities.continuation_patience = 15
        learner.provider_adapter = mock_adapter

        # Force an increase above 15
        for _ in range(10):
            _record_patience_outcome(
                learner,
                flagged_as_stuck=True,
                actually_stuck=False,
                eventually_made_progress=True,
                patience_threshold=15,
            )
        rec = learner.get_recommendation("ollama", "test-model", "analysis")
        assert rec.value == 15  # Should not go above 15
