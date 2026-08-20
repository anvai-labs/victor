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

"""Unit tests for tool selector RL learner.

Tests the ToolSelectorLearner which uses contextual bandits with
conservative exploration (ε=0.1) for adaptive tool ranking.
"""

import pytest
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import patch

from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import RLCoordinator
from victor.framework.rl.learners.tool_selector import ToolSelectorLearner
from victor.core.database import reset_database, get_database
from victor.core.schema import Tables


@pytest.fixture
def coordinator(tmp_path: Path) -> RLCoordinator:
    """Fixture for RLCoordinator, ensuring a clean database for each test."""
    # Reset the global singleton to ensure fresh database for each test
    reset_database()
    db_path = tmp_path / "rl_test.db"
    # Initialize the database singleton with temp path BEFORE creating coordinator
    # This ensures RLCoordinator uses the temp database, not ~/.victor/victor.db
    get_database(db_path)
    coord = RLCoordinator(storage_path=tmp_path, db_path=db_path)
    yield coord
    # Reset again after the test to clean up
    reset_database()


@pytest.fixture
def learner(coordinator: RLCoordinator) -> ToolSelectorLearner:
    """Fixture for ToolSelectorLearner."""
    return coordinator.get_learner("tool_selector")  # type: ignore


def _record_tool_outcome(
    learner: ToolSelectorLearner,
    tool_name: str = "read",
    task_type: str = "analysis",
    *,
    success: bool = True,
    quality_score: float = 0.8,
    tool_success: bool = True,
    task_completed: bool = True,
    grounding_score: float = 0.8,
    efficiency_score: float = 0.5,
) -> None:
    """Helper to record a single tool execution outcome."""
    outcome = RLOutcome(
        provider="tool_selector",  # Not used by tool selector
        model="hybrid",
        task_type=task_type,
        success=success,
        quality_score=quality_score,
        metadata={
            "tool_name": tool_name,
            "tool_success": tool_success,
            "task_completed": task_completed,
            "grounding_score": grounding_score,
            "efficiency_score": efficiency_score,
        },
    )
    learner.record_outcome(outcome)


def _get_q_value_from_db(
    coordinator: RLCoordinator,
    tool_name: str,
    task_type: Optional[str] = None,
) -> Tuple[float, int]:
    """Helper to retrieve Q-value and selection count from the database."""
    cursor = coordinator.db.cursor()
    if task_type:
        cursor.execute(
            f"SELECT q_value, visit_count FROM {Tables.RL_Q_VALUE} "
            f"WHERE learner_id = 'tool_selector' AND state_key = ? AND action_key = ?",
            (tool_name, task_type),
        )
    else:
        cursor.execute(
            f"SELECT q_value, visit_count FROM {Tables.RL_Q_VALUE} "
            f"WHERE learner_id = 'tool_selector' AND state_key = ? AND action_key = 'select'",
            (tool_name,),
        )
    row = cursor.fetchone()
    return (row[0], row[1]) if row else (0.5, 0)  # Default values


class TestToolSelectorLearner:
    """Tests for ToolSelectorLearner."""

    def test_initialization(self, learner: ToolSelectorLearner) -> None:
        """Test learner initializes correctly and creates tables."""
        assert learner.name == "tool_selector"
        assert learner.epsilon == 0.1  # Conservative exploration
        assert learner.learning_rate == 0.05  # Slow, stable learning

        cursor = learner.db.cursor()
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_Q_VALUE}';"
        )
        assert cursor.fetchone() is not None
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_TASK_STAT}';"
        )
        assert cursor.fetchone() is not None
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{Tables.RL_TRANSITION}';"
        )
        assert cursor.fetchone() is not None

    def test_record_single_outcome(
        self, coordinator: RLCoordinator, learner: ToolSelectorLearner
    ) -> None:
        """Recording one outcome updates Q-values and counts."""
        _record_tool_outcome(
            learner,
            tool_name="read",
            task_type="analysis",
            success=True,
            quality_score=0.9,
        )

        q_value, count = _get_q_value_from_db(coordinator, "read")
        assert count == 1
        assert q_value != 0.5  # Should have been updated from default

        task_q_value, task_count = _get_q_value_from_db(coordinator, "read", "analysis")
        assert task_count == 1
        assert task_q_value != 0.5

    def test_multiple_outcomes_converge(
        self, coordinator: RLCoordinator, learner: ToolSelectorLearner
    ) -> None:
        """Multiple positive outcomes increase Q-value."""
        # Record several successful outcomes
        for _ in range(10):
            _record_tool_outcome(
                learner,
                tool_name="code_search",
                task_type="search",
                success=True,
                quality_score=0.9,
                tool_success=True,
                task_completed=True,
            )

        q_value, count = _get_q_value_from_db(coordinator, "code_search")
        assert count == 10
        assert q_value > 0.5  # Should increase towards 1.0

    def test_failure_decreases_q_value(
        self, coordinator: RLCoordinator, learner: ToolSelectorLearner
    ) -> None:
        """Failed outcomes decrease Q-value."""
        # Record several failed outcomes
        for _ in range(10):
            _record_tool_outcome(
                learner,
                tool_name="web_search",
                task_type="search",
                success=False,
                quality_score=0.1,
                tool_success=False,
                task_completed=False,
            )

        q_value, count = _get_q_value_from_db(coordinator, "web_search")
        assert count == 10
        assert q_value < 0.5  # Should decrease towards 0.0

    def test_persistence(self, tmp_path: Path) -> None:
        """State persists across learner instances."""
        # Reset to ensure clean state
        reset_database()
        db_path = tmp_path / "rl_test.db"

        # Initialize DB singleton with temp path
        get_database(db_path)
        coordinator1 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
        learner1 = coordinator1.get_learner("tool_selector")  # type: ignore

        _record_tool_outcome(learner1, tool_name="edit")

        # Reset the singleton to simulate new session (don't call close directly)
        reset_database()

        # Re-initialize with same temp path to test persistence
        get_database(db_path)
        coordinator2 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
        learner2 = coordinator2.get_learner("tool_selector")  # type: ignore

        q_value, count = _get_q_value_from_db(coordinator2, "edit")
        assert count == 1
        assert q_value != 0.5

        # Check state was loaded correctly
        assert learner2._tool_selection_counts.get("edit", 0) == 1

        # Clean up
        reset_database()

    def test_get_recommendation(self, learner: ToolSelectorLearner) -> None:
        """Test get_recommendation returns correct values."""
        # Record some outcomes
        _record_tool_outcome(learner, tool_name="read", task_type="analysis", success=True)

        # Get recommendation (provider param overloaded as tool_name)
        rec = learner.get_recommendation(
            provider="read",  # tool_name
            model="hybrid",  # ignored
            task_type="analysis",
        )

        assert rec is not None
        assert rec.value > 0  # Q-value
        assert 0 < rec.confidence < 1  # Confidence
        assert rec.sample_size == 1

    def test_get_tool_rankings(self, learner: ToolSelectorLearner) -> None:
        """Test get_tool_rankings returns sorted tools."""
        # Record outcomes for multiple tools with different success rates
        # Use same task_type to ensure Q-values are comparable
        for _ in range(5):
            _record_tool_outcome(
                learner,
                tool_name="good_tool",
                task_type="analysis",  # Same task type
                success=True,
                quality_score=0.9,
                tool_success=True,
                task_completed=True,
                grounding_score=0.9,
                efficiency_score=0.9,
            )
        for _ in range(5):
            _record_tool_outcome(
                learner,
                tool_name="bad_tool",
                task_type="analysis",  # Same task type
                success=False,
                quality_score=0.1,
                tool_success=False,
                task_completed=False,
                grounding_score=0.1,
                efficiency_score=0.1,
            )

        rankings = learner.get_tool_rankings(["good_tool", "bad_tool"], "analysis")

        assert len(rankings) == 2
        # Good tool should be ranked first (higher Q-value)
        assert rankings[0][0] == "good_tool"
        assert rankings[1][0] == "bad_tool"
        assert rankings[0][1] > rankings[1][1]  # good_tool Q-value > bad_tool Q-value

    def test_should_explore(self, learner: ToolSelectorLearner) -> None:
        """Test exploration probability."""
        # With ε=0.1, should explore ~10% of the time. Control the learner's
        # injectable RNG (self._rng) directly rather than patching the stdlib
        # global random module.
        with patch.object(learner._rng, "random", return_value=0.05):
            assert learner.should_explore() is True  # 0.05 < 0.1

        with patch.object(learner._rng, "random", return_value=0.15):
            assert learner.should_explore() is False  # 0.15 > 0.1

    def test_blended_q_value(self, learner: ToolSelectorLearner) -> None:
        """Test blended Q-value calculation (70% task-specific + 30% global)."""
        # Record outcomes for same tool with different task types
        _record_tool_outcome(
            learner,
            tool_name="edit",
            task_type="action",
            success=True,
            quality_score=1.0,
        )
        _record_tool_outcome(
            learner,
            tool_name="edit",
            task_type="analysis",
            success=False,
            quality_score=0.0,
        )

        # Blended value should be weighted mix
        blended = learner._get_blended_q_value("edit", "action")
        global_q = learner._tool_q_values.get("edit", 0.5)
        task_q = learner._tool_task_q_values.get("edit", {}).get("action", 0.5)

        expected = 0.7 * task_q + 0.3 * global_q
        assert abs(blended - expected) < 0.01

    def test_compute_reward(self, learner: ToolSelectorLearner) -> None:
        """Test reward computation from implicit signals."""
        # Perfect outcome
        outcome = RLOutcome(
            provider="tool_selector",
            model="hybrid",
            task_type="default",
            success=True,
            quality_score=1.0,
            metadata={
                "tool_name": "test",
                "tool_success": True,
                "task_completed": True,
                "grounding_score": 1.0,
                "efficiency_score": 1.0,
            },
        )
        reward = learner._compute_reward(outcome)
        # 0.4*1.0 + 0.3*1.0 + 0.2*1.0 + 0.1*1.0 = 1.0
        assert abs(reward - 1.0) < 0.001  # Allow floating point tolerance

        # Failed outcome
        outcome_fail = RLOutcome(
            provider="tool_selector",
            model="hybrid",
            task_type="default",
            success=False,
            quality_score=0.0,
            metadata={
                "tool_name": "test",
                "tool_success": False,
                "task_completed": False,
                "grounding_score": 0.0,
                "efficiency_score": 0.0,
            },
        )
        reward_fail = learner._compute_reward(outcome_fail)
        assert abs(reward_fail - 0.0) < 0.001  # Allow floating point tolerance

    def test_q_value_clamping(self, learner: ToolSelectorLearner) -> None:
        """Test Q-values are clamped to [0.0, 1.0]."""
        assert learner._clamp_q_value(-0.5) == 0.0
        assert learner._clamp_q_value(1.5) == 1.0
        assert learner._clamp_q_value(0.7) == 0.7

    def test_get_tool_stats(self, learner: ToolSelectorLearner) -> None:
        """Test tool statistics retrieval."""
        _record_tool_outcome(learner, tool_name="git", success=True)
        _record_tool_outcome(learner, tool_name="git", success=False)

        stats = learner.get_tool_stats("git")

        assert stats["tool_name"] == "git"
        assert stats["selection_count"] == 2
        assert stats["success_count"] == 1
        assert stats["success_rate"] == 0.5

    def test_export_metrics(self, learner: ToolSelectorLearner) -> None:
        """Test metrics export."""
        _record_tool_outcome(learner, tool_name="read")
        _record_tool_outcome(learner, tool_name="edit")

        metrics = learner.export_metrics()

        assert metrics["learner"] == "tool_selector"
        assert metrics["total_tools_tracked"] == 2
        assert metrics["total_selections"] == 2
        assert metrics["epsilon"] == 0.1
        assert metrics["learning_rate"] == 0.05

    def test_low_confidence_for_few_samples(self, learner: ToolSelectorLearner) -> None:
        """Test confidence is low when sample size is below threshold."""
        _record_tool_outcome(learner, tool_name="new_tool")

        rec = learner.get_recommendation("new_tool", "hybrid", "default")

        assert rec is not None
        assert rec.is_baseline is True  # Low confidence
        assert rec.confidence < 0.5  # Below high-confidence threshold
        assert "Low confidence" in rec.reason

    def test_high_confidence_after_many_samples(self, learner: ToolSelectorLearner) -> None:
        """Test confidence increases with more samples."""
        for _ in range(25):  # Above MIN_SAMPLES_FOR_CONFIDENCE (20)
            _record_tool_outcome(learner, tool_name="mature_tool")

        rec = learner.get_recommendation("mature_tool", "hybrid", "default")

        assert rec is not None
        assert rec.is_baseline is False  # High confidence
        assert rec.confidence > 0.5
