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

"""Unit tests for mode transition RL learner.

Tests the ModeTransitionLearner which unifies Q-learning from
AdaptiveModeController with the RLCoordinator framework.
"""

import pytest
from pathlib import Path
from typing import Dict, Tuple
from unittest.mock import patch

from victor.framework.rl.base import RLOutcome
from victor.framework.rl.coordinator import RLCoordinator
from victor.framework.rl.learners.mode_transition import (
    ModeTransitionLearner,
    AgentMode,
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
def learner(coordinator: RLCoordinator) -> ModeTransitionLearner:
    """Fixture for ModeTransitionLearner."""
    return coordinator.get_learner("mode_transition")  # type: ignore


def _record_transition_outcome(
    learner: ModeTransitionLearner,
    from_mode: str = "explore",
    to_mode: str = "plan",
    task_type: str = "analysis",
    state_key: str = "explore:analysis:low:mid_low:fair:fair",
    action_key: str = "plan:0",
    *,
    success: bool = True,
    quality_score: float = 0.8,
    tool_budget_used: int = 5,
    tool_budget_total: int = 10,
) -> None:
    """Helper to record a single mode transition outcome."""
    outcome = RLOutcome(
        provider="mode_controller",
        model="default",
        task_type=task_type,
        success=success,
        quality_score=quality_score,
        metadata={
            "from_mode": from_mode,
            "to_mode": to_mode,
            "state_key": state_key,
            "action_key": action_key,
            "task_completed": success,
            "tool_budget_used": tool_budget_used,
            "tool_budget_total": tool_budget_total,
        },
    )
    learner.record_outcome(outcome)


def _get_q_value_from_db(
    coordinator: RLCoordinator,
    state_key: str,
    action_key: str,
) -> Tuple[float, int]:
    """Helper to retrieve Q-value and visit count from the database."""
    cursor = coordinator.db.cursor()
    cursor.execute(
        f"SELECT q_value, visit_count FROM {Tables.RL_Q_VALUE} "
        f"WHERE learner_id = 'mode_transition' AND state_key = ? AND action_key = ?",
        (state_key, action_key),
    )
    row = cursor.fetchone()
    return (row[0], row[1]) if row else (0.0, 0)


class TestModeTransitionLearner:
    """Tests for ModeTransitionLearner."""

    def test_initialization(self, learner: ModeTransitionLearner) -> None:
        """Test learner initializes correctly and creates tables."""
        assert learner.name == "mode_transition"
        assert learner.learning_rate == 0.1
        assert learner.discount_factor == 0.9
        assert learner.epsilon == 0.1

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
        self, coordinator: RLCoordinator, learner: ModeTransitionLearner
    ) -> None:
        """Recording one outcome updates Q-values and counts."""
        state_key = "explore:analysis:low:low:fair:fair"
        action_key = "plan:0"

        _record_transition_outcome(
            learner,
            from_mode="explore",
            to_mode="plan",
            task_type="analysis",
            state_key=state_key,
            action_key=action_key,
            success=True,
            quality_score=0.9,
        )

        q_value, count = _get_q_value_from_db(coordinator, state_key, action_key)
        assert count == 1
        assert q_value != 0.0  # Should have been updated from default

    def test_q_learning_update(
        self, coordinator: RLCoordinator, learner: ModeTransitionLearner
    ) -> None:
        """Q-values update correctly with repeated outcomes."""
        state_key = "plan:create:mid_low:low:good:good"
        action_key = "build:0"

        # Record multiple successful transitions
        for _ in range(5):
            _record_transition_outcome(
                learner,
                from_mode="plan",
                to_mode="build",
                state_key=state_key,
                action_key=action_key,
                success=True,
                quality_score=0.9,
            )

        q_value, count = _get_q_value_from_db(coordinator, state_key, action_key)
        assert count == 5
        assert q_value > 0  # Positive reward should increase Q-value

    def test_failure_decreases_q_value(
        self, coordinator: RLCoordinator, learner: ModeTransitionLearner
    ) -> None:
        """Failed transitions decrease Q-value."""
        state_key = "build:action:high:high:poor:poor"
        action_key = "complete:0"

        # Record failed transitions
        for _ in range(5):
            _record_transition_outcome(
                learner,
                from_mode="build",
                to_mode="complete",
                state_key=state_key,
                action_key=action_key,
                success=False,
                quality_score=0.2,
            )

        q_value, count = _get_q_value_from_db(coordinator, state_key, action_key)
        assert count == 5
        assert q_value < 0  # Negative reward should decrease Q-value

    def test_persistence(self, tmp_path: Path) -> None:
        """State persists across learner instances."""
        state_key = "explore:search:low:low:fair:fair"
        action_key = "plan:0"

        reset_database()
        db_path = tmp_path / "rl_test.db"
        get_database(db_path)
        coordinator1 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
        learner1 = coordinator1.get_learner("mode_transition")  # type: ignore

        _record_transition_outcome(learner1, state_key=state_key, action_key=action_key)
        reset_database()

        get_database(db_path)
        coordinator2 = RLCoordinator(storage_path=tmp_path, db_path=db_path)
        learner2 = coordinator2.get_learner("mode_transition")  # type: ignore

        q_value, count = _get_q_value_from_db(coordinator2, state_key, action_key)
        assert count == 1
        assert q_value != 0.0

        # Check state was loaded correctly
        assert state_key in learner2._q_values
        assert action_key in learner2._q_values[state_key]

        reset_database()

    def test_get_recommendation_exploitation(self, learner: ModeTransitionLearner) -> None:
        """Test get_recommendation returns best action in exploitation mode."""
        state_key = "explore:analysis:low:low:fair:fair"

        # Record outcomes for different actions
        _record_transition_outcome(
            learner,
            state_key=state_key,
            action_key="plan:0",
            to_mode="plan",
            success=True,
            quality_score=0.9,
        )
        _record_transition_outcome(
            learner,
            state_key=state_key,
            action_key="build:0",
            to_mode="build",
            success=False,
            quality_score=0.2,
        )

        # Force exploitation (epsilon=0)
        learner.epsilon = 0.0
        rec = learner.get_recommendation(state_key, "", "analysis")

        assert rec is not None
        assert rec.value == "plan:0"  # Higher Q-value action

    def test_get_recommendation_exploration(self, learner: ModeTransitionLearner) -> None:
        """Test get_recommendation can explore with high epsilon."""
        state_key = "explore:analysis:low:low:fair:fair"

        _record_transition_outcome(learner, state_key=state_key, action_key="plan:0")
        _record_transition_outcome(learner, state_key=state_key, action_key="build:0")

        # Force exploration (epsilon=1.0 always explores; control the learner's
        # injectable RNG rather than the stdlib global).
        learner.epsilon = 1.0
        with patch.object(learner._rng, "random", return_value=0.5):
            rec = learner.get_recommendation(state_key, "", "analysis")

        assert rec is not None
        assert rec.is_baseline is True
        assert "Exploration" in rec.reason

    def test_get_optimal_budget(self, learner: ModeTransitionLearner) -> None:
        """Test optimal budget learning."""
        task_type = "create"

        # Record successful outcomes with different budgets
        for _ in range(5):
            _record_transition_outcome(
                learner,
                task_type=task_type,
                success=True,
                quality_score=0.9,
                tool_budget_used=7,
                tool_budget_total=15,
            )

        budget = learner.get_optimal_budget(task_type)
        assert budget >= 5  # Should have learned from outcomes
        assert budget <= 20  # Should be reasonable

    def test_get_task_stats(self, learner: ModeTransitionLearner) -> None:
        """Test task statistics retrieval."""
        task_type = "edit"

        _record_transition_outcome(learner, task_type=task_type, success=True, quality_score=0.8)
        _record_transition_outcome(learner, task_type=task_type, success=False, quality_score=0.3)

        stats = learner.get_task_stats(task_type)

        assert stats["task_type"] == task_type
        assert stats["sample_count"] == 2
        assert 0 < stats["avg_quality_score"] < 1
        assert 0 < stats["avg_completion_rate"] < 1

    def test_compute_reward(self, learner: ModeTransitionLearner) -> None:
        """Test reward computation."""
        # Perfect outcome
        outcome = RLOutcome(
            provider="mode_controller",
            model="default",
            task_type="analysis",
            success=True,
            quality_score=1.0,
            metadata={
                "from_mode": "explore",
                "to_mode": "plan",
                "state_key": "test",
                "action_key": "plan:0",
                "task_completed": True,
                "tool_budget_used": 3,
                "tool_budget_total": 10,
            },
        )
        reward = learner._compute_reward(outcome)
        assert reward > 0.5  # Should be positive for success

        # Failed outcome
        outcome_fail = RLOutcome(
            provider="mode_controller",
            model="default",
            task_type="analysis",
            success=False,
            quality_score=0.2,
            metadata={
                "from_mode": "build",
                "to_mode": "complete",
                "state_key": "test",
                "action_key": "complete:0",
                "task_completed": False,
                "tool_budget_used": 10,
                "tool_budget_total": 10,
            },
        )
        reward_fail = learner._compute_reward(outcome_fail)
        assert reward_fail < 0  # Should be negative for failure

    def test_valid_transitions(self, learner: ModeTransitionLearner) -> None:
        """Test valid transition definitions."""
        assert AgentMode.PLAN in learner.VALID_TRANSITIONS[AgentMode.EXPLORE]
        assert AgentMode.BUILD in learner.VALID_TRANSITIONS[AgentMode.PLAN]
        assert AgentMode.REVIEW in learner.VALID_TRANSITIONS[AgentMode.BUILD]
        assert AgentMode.COMPLETE in learner.VALID_TRANSITIONS[AgentMode.REVIEW]
        assert len(learner.VALID_TRANSITIONS[AgentMode.COMPLETE]) == 0

    def test_export_metrics(self, learner: ModeTransitionLearner) -> None:
        """Test metrics export."""
        _record_transition_outcome(learner, from_mode="explore", to_mode="plan")
        _record_transition_outcome(learner, from_mode="plan", to_mode="build")

        metrics = learner.export_metrics()

        assert metrics["learner"] == "mode_transition"
        assert metrics["total_transitions"] == 2
        assert metrics["epsilon"] == 0.1
        assert metrics["learning_rate"] == 0.1
        assert metrics["discount_factor"] == 0.9

    def test_transition_history_recorded(
        self, coordinator: RLCoordinator, learner: ModeTransitionLearner
    ) -> None:
        """Test that transition history is recorded to database."""
        _record_transition_outcome(
            learner,
            from_mode="explore",
            to_mode="plan",
            task_type="analysis",
        )

        cursor = coordinator.db.cursor()
        cursor.execute(f"SELECT * FROM {Tables.RL_TRANSITION} WHERE learner_id = 'mode_transition'")
        rows = cursor.fetchall()

        assert len(rows) == 1
        row = dict(rows[0])
        assert row["from_state"] == "explore"
        assert row["to_state"] == "plan"

    def test_no_data_returns_baseline(self, learner: ModeTransitionLearner) -> None:
        """Test that unknown state returns baseline recommendation."""
        rec = learner.get_recommendation("unknown:state:key", "", "default")

        assert rec is not None
        assert rec.is_baseline is True
        assert rec.confidence == 0.3
        assert rec.sample_size == 0
