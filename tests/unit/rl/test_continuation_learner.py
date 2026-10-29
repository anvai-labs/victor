"""Unit tests for continuation prompt RL learner (framework-based)."""

from pathlib import Path

import pytest

from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import RLCoordinator
from victor.framework.rl.learners.continuation_prompts import ContinuationPromptLearner
from victor.core.database import reset_database, get_database
from victor.core.schema import Tables


def _record_outcome(
    learner: ContinuationPromptLearner,
    provider: str = "ollama",
    model: str = "test-model",
    task_type: str = "analysis",
    *,
    continuation_prompts_used: int = 2,
    max_prompts_configured: int = 6,
    success: bool = True,
    quality_score: float = 0.8,
    stuck_loop_detected: bool = False,
    forced_completion: bool = False,
    tool_calls_total: int = 10,
) -> None:
    """Helper to record a single outcome."""
    outcome = RLOutcome(
        provider=provider,
        model=model,
        task_type=task_type,
        success=success,
        quality_score=quality_score,
        metadata={
            "continuation_prompts_used": continuation_prompts_used,
            "max_prompts_configured": max_prompts_configured,
            "stuck_loop_detected": stuck_loop_detected,
            "forced_completion": forced_completion,
            "tool_calls_total": tool_calls_total,
        },
    )
    learner.record_outcome(outcome)


def _get_stats(
    coordinator: RLCoordinator,
    provider: str,
    model: str,
    task_type: str,
):
    cursor = coordinator.db.cursor()
    context_key = f"{provider}:{model}:{task_type}"
    rows = cursor.execute(
        f"SELECT stat_key, stat_value FROM {Tables.RL_TASK_STAT} "
        f"WHERE learner_id = 'continuation_prompts' AND task_type = ?",
        (context_key,),
    ).fetchall()
    result: dict = {"context_key": context_key}
    for stat_key, stat_value in rows:
        result[stat_key] = stat_value
    # current_max_prompts stored in rl_param is the post-adjustment recommendation
    param_row = cursor.execute(
        f"SELECT param_value FROM {Tables.RL_PARAM} "
        f"WHERE learner_id = 'continuation_prompts' AND param_key = 'current_max_prompts' "
        f"AND context = ?",
        (context_key,),
    ).fetchone()
    if param_row is not None:
        result["recommended_max_prompts"] = param_row[0]
    return result


@pytest.fixture
def coordinator(tmp_path: Path) -> RLCoordinator:
    reset_database()
    db_path = tmp_path / "rl_test.db"
    get_database(db_path)
    coord = RLCoordinator(storage_path=tmp_path, db_path=db_path)
    yield coord
    reset_database()


@pytest.fixture
def learner(coordinator: RLCoordinator) -> ContinuationPromptLearner:
    return coordinator.get_learner("continuation_prompts")  # type: ignore[return-value]


def test_learner_initialization(
    coordinator: RLCoordinator, learner: ContinuationPromptLearner
) -> None:
    """Learner starts with empty tables and configured learning rate."""
    assert learner.learning_rate == 0.1
    cursor = coordinator.db.cursor()
    cursor.execute(
        f"SELECT count(*) FROM {Tables.RL_TASK_STAT} WHERE learner_id = 'continuation_prompts'"
    )
    assert cursor.fetchone()[0] == 0


def test_record_single_outcome(
    coordinator: RLCoordinator, learner: ContinuationPromptLearner
) -> None:
    """Recording one outcome persists stats."""
    _record_outcome(
        learner,
        provider="ollama",
        model="qwen3-coder-tools:30b",
        task_type="analysis",
        continuation_prompts_used=2,
        max_prompts_configured=6,
        quality_score=0.8,
    )

    stats = _get_stats(coordinator, "ollama", "qwen3-coder-tools:30b", "analysis")
    assert stats["total_sessions"] == 1
    assert stats["successful_sessions"] == 1
    assert stats["stuck_loop_count"] == 0
    assert pytest.approx(stats["avg_quality_score"], rel=1e-3) == 0.8


def test_high_stuck_rate_decreases_recommendation(
    coordinator: RLCoordinator, learner: ContinuationPromptLearner
) -> None:
    """High stuck rate should reduce recommended max prompts."""
    for i in range(5):
        _record_outcome(
            learner,
            stuck_loop_detected=i < 3,  # 60% stuck
            max_prompts_configured=6,
            continuation_prompts_used=2,
            quality_score=0.7,
        )

    stats = _get_stats(coordinator, "ollama", "test-model", "analysis")
    assert stats["stuck_loop_count"] == 3
    assert stats["recommended_max_prompts"] is not None
    assert stats["recommended_max_prompts"] < 6


def test_low_quality_increases_recommendation(
    coordinator: RLCoordinator, learner: ContinuationPromptLearner
) -> None:
    """Low quality with few prompts should increase limit."""
    for _ in range(5):
        _record_outcome(
            learner,
            continuation_prompts_used=2,
            max_prompts_configured=6,
            quality_score=0.4,
            tool_calls_total=5,
        )

    stats = _get_stats(coordinator, "ollama", "test-model", "analysis")
    assert stats["recommended_max_prompts"] is not None
    assert stats["recommended_max_prompts"] > 6


def test_high_quality_decreases_recommendation(
    coordinator: RLCoordinator, learner: ContinuationPromptLearner
) -> None:
    """High quality with many prompts should lower the limit."""
    for _ in range(5):
        _record_outcome(
            learner,
            continuation_prompts_used=5,
            max_prompts_configured=6,
            quality_score=0.9,
            tool_calls_total=20,
        )

    stats = _get_stats(coordinator, "ollama", "test-model", "analysis")
    assert stats["recommended_max_prompts"] is not None
    assert stats["recommended_max_prompts"] <= 6


def test_persistence(tmp_path: Path) -> None:
    """Stats persist across learner instances sharing the same DB."""
    reset_database()
    db_path = tmp_path / "rl_test.db"
    get_database(db_path)
    coordinator1 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
    learner1 = coordinator1.get_learner("continuation_prompts")  # type: ignore[return-value]
    _record_outcome(
        learner1,
        provider="ollama",
        model="persist-model",
        task_type="analysis",
        continuation_prompts_used=2,
        max_prompts_configured=6,
        quality_score=0.8,
    )
    reset_database()

    get_database(db_path)
    coordinator2 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
    learner2 = coordinator2.get_learner("continuation_prompts")  # type: ignore[return-value]
    stats = _get_stats(coordinator2, "ollama", "persist-model", "analysis")
    assert stats["total_sessions"] == 1
    assert pytest.approx(stats["avg_quality_score"], rel=1e-3) == 0.8

    reset_database()


def test_no_recommendation_with_insufficient_data(
    coordinator: RLCoordinator, learner: ContinuationPromptLearner
) -> None:
    """Require at least 3 sessions before recommending."""
    for _ in range(2):
        _record_outcome(learner, max_prompts_configured=6, quality_score=0.7)

    assert learner.get_recommendation("ollama", "test-model", "analysis") is None


def test_recommendation_bounds(
    coordinator: RLCoordinator, learner: ContinuationPromptLearner
) -> None:
    """Recommendations stay within [1, 20]."""
    for _ in range(10):
        _record_outcome(
            learner,
            continuation_prompts_used=1,
            max_prompts_configured=2,
            success=False,
            quality_score=0.1,
            forced_completion=True,
            tool_calls_total=2,
        )

    stats = _get_stats(coordinator, "ollama", "test-model", "analysis")
    assert stats["recommended_max_prompts"] is not None
    assert 1 <= stats["recommended_max_prompts"] <= 20


def test_weighted_moving_average(
    coordinator: RLCoordinator, learner: ContinuationPromptLearner
) -> None:
    """Recent high-quality outcomes should dominate the average."""
    _record_outcome(
        learner,
        quality_score=0.3,
        continuation_prompts_used=2,
        max_prompts_configured=6,
    )
    for _ in range(5):
        _record_outcome(
            learner,
            quality_score=0.9,
            continuation_prompts_used=2,
            max_prompts_configured=6,
        )

    stats = _get_stats(coordinator, "ollama", "test-model", "analysis")
    assert stats["avg_quality_score"] > 0.7


def test_recommendation_confidence_and_reason(
    learner: ContinuationPromptLearner,
) -> None:
    """Recommendation returns RLRecommendation with metadata."""
    for _ in range(3):
        _record_outcome(
            learner,
            quality_score=0.8,
            continuation_prompts_used=2,
            max_prompts_configured=6,
        )

    rec = learner.get_recommendation("ollama", "test-model", "analysis")
    assert rec is not None
    assert rec.value is not None
    assert rec.sample_size >= 3
    assert rec.reason.startswith("Learned from")
