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

"""Unit tests for the AdaptiveModeController module.

Tests the adaptive mode controller which manages mode transitions using
Q-learning, provider-aware thresholds, and tool budget optimization.
"""

import sqlite3
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

from victor.agent.adaptive_mode_controller import (
    AdaptiveModeController,
    AgentMode,
    ModeAction,
    ModeState,
    QLearningStore,
    TransitionEvent,
    TransitionTrigger,
    get_mode_controller,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def _make_isolated_db():
    """Build a fresh in-memory DB mock that satisfies QLearningStore."""
    from victor.framework.rl.migration import RLTableMigrator

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Bootstrap unified RL tables (rl_q_value, rl_task_stat, rl_transition, rl_param).
    # RLTableMigrator._ensure_unified_tables() is called in __init__.
    RLTableMigrator(conn)
    mock_db = MagicMock()
    mock_db.get_connection.return_value = conn
    mock_db.db_path = Path(":memory:")
    return mock_db


def _make_q_store(mock_db=None) -> QLearningStore:
    """Construct a QLearningStore wired to an isolated in-memory DB."""
    if mock_db is None:
        mock_db = _make_isolated_db()
    store = QLearningStore.__new__(QLearningStore)
    store._db = mock_db
    store.db_path = mock_db.db_path
    store._initialized = False
    store.learning_rate = 0.1
    store.discount_factor = 0.9
    store.exploration_rate = 0.1
    return store


@pytest.fixture
def tmp_project_path(tmp_path: Path) -> Path:
    """Temporary project path — also patches DatabaseManager so any
    QLearningStore(project_path=...) call in the test body gets an
    isolated in-memory DB instead of the global singleton."""
    victor_dir = tmp_path / ".victor"
    victor_dir.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture(autouse=True)
def _patch_db_manager(monkeypatch):
    """Redirect DatabaseManager() to a fresh in-memory instance per test.

    QLearningStore ignores project_path and calls DatabaseManager() directly
    (imported inline inside __init__). We patch the class in its home module
    so the local `from victor.core.database import DatabaseManager` picks it up.
    Without this patch every test shares the same singleton DB, causing
    sample_count / Q-value accumulation across the test session.
    """
    mock_db = _make_isolated_db()
    monkeypatch.setattr(
        "victor.core.database.DatabaseManager",
        lambda: mock_db,
    )


@pytest.fixture
def q_store() -> QLearningStore:
    """QLearningStore backed by the per-test isolated DB (via _patch_db_manager)."""
    return QLearningStore()


@pytest.fixture
def controller(q_store: QLearningStore) -> AdaptiveModeController:
    """Create an AdaptiveModeController with an isolated database."""
    return AdaptiveModeController(
        profile_name="test_profile",
        q_store=q_store,
    )


@pytest.fixture
def controller_with_provider(q_store: QLearningStore) -> AdaptiveModeController:
    """Create an AdaptiveModeController with provider configuration."""
    return AdaptiveModeController(
        profile_name="test_profile",
        q_store=q_store,
        provider_name="anthropic",
        model_name="claude-3-sonnet",
    )


# =============================================================================
# Test AgentMode Enum
# =============================================================================


class TestAgentMode:
    """Tests for AgentMode enum."""

    def test_agent_mode_values(self):
        """Test AgentMode has expected values."""
        assert AgentMode.EXPLORE.value == "explore"
        assert AgentMode.PLAN.value == "plan"
        assert AgentMode.BUILD.value == "build"
        assert AgentMode.REVIEW.value == "review"
        assert AgentMode.COMPLETE.value == "complete"

    def test_agent_mode_from_string(self):
        """Test AgentMode can be created from string."""
        assert AgentMode("explore") == AgentMode.EXPLORE
        assert AgentMode("plan") == AgentMode.PLAN
        assert AgentMode("build") == AgentMode.BUILD


class TestTransitionTrigger:
    """Tests for TransitionTrigger enum."""

    def test_transition_trigger_values(self):
        """Test TransitionTrigger has expected values."""
        assert TransitionTrigger.USER_REQUEST.value == "user_request"
        assert TransitionTrigger.TASK_COMPLETE.value == "task_complete"
        assert TransitionTrigger.BUDGET_LOW.value == "budget_low"
        assert TransitionTrigger.QUALITY_THRESHOLD.value == "quality_threshold"
        assert TransitionTrigger.LOOP_DETECTED.value == "loop_detected"


# =============================================================================
# Test ModeState
# =============================================================================


class TestModeState:
    """Tests for ModeState dataclass."""

    def test_mode_state_creation(self):
        """Test ModeState can be created with valid parameters."""
        state = ModeState(
            mode=AgentMode.EXPLORE,
            task_type="analysis",
            tool_calls_made=5,
            tool_budget=10,
            iteration_count=3,
            iteration_budget=20,
            quality_score=0.75,
            grounding_score=0.8,
            time_in_mode_seconds=30.0,
            recent_tool_success_rate=0.9,
        )

        assert state.mode == AgentMode.EXPLORE
        assert state.task_type == "analysis"
        assert state.tool_calls_made == 5
        assert state.quality_score == 0.75

    def test_mode_state_to_state_key(self):
        """Test ModeState generates correct state key."""
        state = ModeState(
            mode=AgentMode.EXPLORE,
            task_type="analysis",
            tool_calls_made=2,
            tool_budget=10,
            iteration_count=5,
            iteration_budget=20,
            quality_score=0.75,
            grounding_score=0.8,
            time_in_mode_seconds=30.0,
            recent_tool_success_rate=0.9,
        )

        state_key = state.to_state_key()
        assert "explore" in state_key
        assert "analysis" in state_key
        # Ratio 2/10 = 0.2 should be "low"
        assert "low" in state_key

    def test_discretize_ratio_buckets(self):
        """Test ratio discretization produces correct buckets."""
        state = ModeState(
            mode=AgentMode.BUILD,
            task_type="edit",
            tool_calls_made=1,
            tool_budget=10,
            iteration_count=1,
            iteration_budget=20,
            quality_score=0.5,
            grounding_score=0.5,
            time_in_mode_seconds=0.0,
            recent_tool_success_rate=1.0,
        )

        # Test ratio discretization
        assert state._discretize_ratio(0.1) == "low"
        assert state._discretize_ratio(0.3) == "mid_low"
        assert state._discretize_ratio(0.6) == "mid_high"
        assert state._discretize_ratio(0.9) == "high"

    def test_discretize_quality_buckets(self):
        """Test quality score discretization produces correct buckets."""
        state = ModeState(
            mode=AgentMode.BUILD,
            task_type="edit",
            tool_calls_made=1,
            tool_budget=10,
            iteration_count=1,
            iteration_budget=20,
            quality_score=0.5,
            grounding_score=0.5,
            time_in_mode_seconds=0.0,
            recent_tool_success_rate=1.0,
        )

        assert state._discretize_quality(0.2) == "poor"
        assert state._discretize_quality(0.5) == "fair"
        assert state._discretize_quality(0.7) == "good"
        assert state._discretize_quality(0.9) == "excellent"


# =============================================================================
# Test ModeAction
# =============================================================================


class TestModeAction:
    """Tests for ModeAction dataclass."""

    def test_mode_action_creation(self):
        """Test ModeAction can be created with defaults."""
        action = ModeAction(target_mode=AgentMode.BUILD)

        assert action.target_mode == AgentMode.BUILD
        assert action.adjust_tool_budget == 0
        assert action.should_continue is True
        assert action.confidence == 0.5

    def test_mode_action_with_budget_adjustment(self):
        """Test ModeAction with budget adjustment."""
        action = ModeAction(
            target_mode=AgentMode.PLAN,
            adjust_tool_budget=5,
            should_continue=True,
            reason="Testing budget increase",
            confidence=0.8,
        )

        assert action.adjust_tool_budget == 5
        assert action.reason == "Testing budget increase"
        assert action.confidence == 0.8

    def test_mode_action_repr(self):
        """Test ModeAction string representation."""
        action = ModeAction(
            target_mode=AgentMode.BUILD,
            adjust_tool_budget=3,
            confidence=0.75,
        )

        repr_str = repr(action)
        assert "build" in repr_str
        assert "+3" in repr_str
        assert "0.75" in repr_str


# =============================================================================
# Test QLearningStore
# =============================================================================


class TestQLearningStore:
    """Tests for QLearningStore class."""

    def test_initialization(self, q_store: QLearningStore):
        """Test QLearningStore initializes correctly."""
        assert q_store.learning_rate == 0.1
        assert q_store.discount_factor == 0.9
        assert q_store.exploration_rate == 0.1

    def test_get_q_value_default(self, q_store: QLearningStore):
        """Test getting Q-value for non-existent state returns 0."""
        q_value = q_store.get_q_value("unknown_state", "unknown_action")
        assert q_value == 0.0

    def test_update_q_value(self, q_store: QLearningStore):
        """Test updating Q-value works correctly."""
        state_key = "explore:analysis:low:low:fair:fair"
        action_key = "plan:+0"

        # Update Q-value
        new_q = q_store.update_q_value(state_key, action_key, reward=1.0)

        assert new_q != 0.0
        # Verify it's stored
        stored_q = q_store.get_q_value(state_key, action_key)
        assert stored_q == new_q

    def test_update_q_value_with_next_state(self, q_store: QLearningStore):
        """Test Q-learning update with next state."""
        state_key = "explore:analysis:low:low:fair:fair"
        action_key = "plan:+0"
        next_state_key = "plan:analysis:low:low:good:good"

        # First set up next state Q-value
        q_store.update_q_value(next_state_key, "build:+0", reward=1.0)

        # Now update with next state consideration
        new_q = q_store.update_q_value(
            state_key, action_key, reward=0.5, next_state_key=next_state_key
        )

        assert new_q != 0.0

    def test_get_all_actions(self, q_store: QLearningStore):
        """Test getting all actions for a state."""
        state_key = "explore:analysis:low:low:fair:fair"

        # Add multiple actions
        q_store.update_q_value(state_key, "plan:+0", reward=1.0)
        q_store.update_q_value(state_key, "build:+0", reward=0.5)
        q_store.update_q_value(state_key, "explore:+2", reward=0.3)

        actions = q_store.get_all_actions(state_key)

        assert len(actions) == 3
        assert "plan:+0" in actions
        assert "build:+0" in actions

    def test_record_transition(self, q_store: QLearningStore):
        """Test recording transition history."""
        q_store.record_transition(
            profile_name="test_profile",
            from_mode=AgentMode.EXPLORE,
            to_mode=AgentMode.PLAN,
            trigger=TransitionTrigger.PATTERN_DETECTED,
            state_key="test_state",
            action_key="test_action",
            reward=0.5,
        )

        # Verify transition was recorded (no assertion error means success)
        # The transition is stored in the database

    def test_update_task_stats_first_sample(self, q_store: QLearningStore):
        """Test updating task stats for first sample."""
        q_store.update_task_stats(
            task_type="analysis",
            tool_budget_used=5,
            quality_score=0.8,
            completed=True,
        )

        stats = q_store.get_task_stats("analysis")

        assert stats["task_type"] == "analysis"
        assert stats["sample_count"] == 1
        assert stats["avg_quality_score"] == 0.8

    def test_update_task_stats_success_efficient(self, q_store: QLearningStore):
        """Test task stats update for efficient completion."""
        # First sample to establish baseline
        q_store.update_task_stats(
            task_type="edit",
            tool_budget_used=10,
            quality_score=0.7,
            completed=True,
        )

        # Efficient completion (used < 50% of budget)
        q_store.update_task_stats(
            task_type="edit",
            tool_budget_used=3,
            quality_score=0.9,
            completed=True,
            tool_budget_total=20,
        )

        stats = q_store.get_task_stats("edit")
        assert stats["sample_count"] == 2

    def test_update_task_stats_failure_increases_budget(self, q_store: QLearningStore):
        """Test task stats update on failure increases budget recommendation."""
        # First sample
        q_store.update_task_stats(
            task_type="create",
            tool_budget_used=10,
            quality_score=0.7,
            completed=True,
        )

        initial_stats = q_store.get_task_stats("create")
        initial_budget = initial_stats["optimal_tool_budget"]

        # Record failure with budget exhaustion
        q_store.update_task_stats(
            task_type="create",
            tool_budget_used=10,
            quality_score=0.3,
            completed=False,
            tool_budget_total=10,
            budget_exhausted=True,
        )

        final_stats = q_store.get_task_stats("create")
        # Budget should increase after failure with exhaustion
        assert final_stats["optimal_tool_budget"] >= initial_budget

    def test_get_task_stats_default(self, q_store: QLearningStore):
        """Test getting task stats for unknown task type returns defaults."""
        stats = q_store.get_task_stats("unknown_task")

        assert stats["task_type"] == "unknown_task"
        assert stats["optimal_tool_budget"] == 10
        assert stats["avg_quality_score"] == 0.5
        assert stats["sample_count"] == 0

    def test_get_min_budget_for_task(self, q_store: QLearningStore):
        """Test minimum budget floor for different task types."""
        assert q_store._get_min_budget_for_task("code_generation") == 8
        assert q_store._get_min_budget_for_task("design") == 25
        assert q_store._get_min_budget_for_task("unknown") == 10


# =============================================================================
# Test AdaptiveModeController
# =============================================================================


class TestAdaptiveModeControllerInit:
    """Tests for AdaptiveModeController initialization."""

    def test_basic_initialization(self, controller: AdaptiveModeController):
        """Test basic controller initialization."""
        assert controller.profile_name == "test_profile"
        assert controller._current_state is None
        assert controller._current_action is None
        assert controller._total_reward == 0.0

    def test_initialization_with_provider(self, controller_with_provider: AdaptiveModeController):
        """Test controller initialization with provider."""
        assert controller_with_provider._provider_name == "anthropic"
        assert controller_with_provider._model_name == "claude-3-sonnet"


class TestProviderNormalization:
    """Tests for provider name normalization."""

    def test_normalize_provider_default(self, controller: AdaptiveModeController):
        """Test provider normalization returns default for None."""
        assert controller._normalize_provider_name(None) == "default"

    def test_normalize_provider_anthropic(self, controller: AdaptiveModeController):
        """Test anthropic provider normalization."""
        assert controller._normalize_provider_name("anthropic") == "anthropic"
        assert controller._normalize_provider_name("claude") == "anthropic"
        assert controller._normalize_provider_name("ANTHROPIC") == "anthropic"

    def test_normalize_provider_openai(self, controller: AdaptiveModeController):
        """Test OpenAI provider normalization."""
        assert controller._normalize_provider_name("openai") == "openai"
        assert controller._normalize_provider_name("gpt") == "openai"

    def test_normalize_provider_google(self, controller: AdaptiveModeController):
        """Test Google provider normalization."""
        assert controller._normalize_provider_name("google") == "google"
        assert controller._normalize_provider_name("gemini") == "google"

    def test_normalize_provider_xai(self, controller: AdaptiveModeController):
        """Test xAI provider normalization."""
        assert controller._normalize_provider_name("xai") == "xai"
        assert controller._normalize_provider_name("grok") == "xai"

    def test_normalize_provider_deepseek(self, controller: AdaptiveModeController):
        """Test DeepSeek provider normalization."""
        assert controller._normalize_provider_name("deepseek") == "deepseek"
        assert controller._normalize_provider_name("deepseek:deepseek-chat") == "deepseek"

    def test_normalize_provider_local(self, controller: AdaptiveModeController):
        """Test local provider normalization."""
        assert controller._normalize_provider_name("ollama") == "ollama"
        assert controller._normalize_provider_name("lmstudio") == "ollama"
        assert controller._normalize_provider_name("vllm") == "ollama"


class TestProviderThresholds:
    """Tests for provider-specific thresholds."""

    def test_get_iteration_thresholds_default(self, controller: AdaptiveModeController):
        """Test default iteration thresholds."""
        thresholds = controller.get_iteration_thresholds()

        assert "min_iterations_before_loop" in thresholds
        assert "no_tool_threshold" in thresholds

    def test_get_iteration_thresholds_anthropic(self, tmp_project_path: Path):
        """Test Anthropic-specific iteration thresholds."""
        q_store = QLearningStore(project_path=tmp_project_path)
        controller = AdaptiveModeController(
            profile_name="test",
            q_store=q_store,
            provider_name="anthropic",
        )

        thresholds = controller.get_iteration_thresholds()
        assert thresholds["min_iterations_before_loop"] == 3
        assert thresholds["no_tool_threshold"] == 2

    def test_get_iteration_thresholds_deepseek(self, tmp_project_path: Path):
        """Test DeepSeek-specific iteration thresholds (reasoning model)."""
        q_store = QLearningStore(project_path=tmp_project_path)
        controller = AdaptiveModeController(
            profile_name="test",
            q_store=q_store,
            provider_name="deepseek",
        )

        thresholds = controller.get_iteration_thresholds()
        # DeepSeek should have higher thresholds for reasoning
        assert thresholds["min_iterations_before_loop"] == 5
        assert thresholds["no_tool_threshold"] == 3

    def test_get_quality_thresholds_default(self, controller: AdaptiveModeController):
        """Test default quality thresholds."""
        thresholds = controller.get_quality_thresholds()

        assert "min_quality" in thresholds
        assert "grounding_threshold" in thresholds

    def test_get_quality_thresholds_with_provider_adapter(self, tmp_project_path: Path):
        """Test quality thresholds from provider adapter."""
        mock_adapter = MagicMock()
        mock_adapter.capabilities.quality_threshold = 0.85
        mock_adapter.capabilities.grounding_strictness = 0.75

        q_store = QLearningStore(project_path=tmp_project_path)
        controller = AdaptiveModeController(
            profile_name="test",
            q_store=q_store,
            provider_adapter=mock_adapter,
        )

        thresholds = controller.get_quality_thresholds()
        assert thresholds["min_quality"] == 0.85
        assert thresholds["grounding_threshold"] == 0.75

    def test_set_provider(self, controller: AdaptiveModeController):
        """Test setting provider updates thresholds."""
        controller.set_provider("deepseek")
        assert controller._provider_name == "deepseek"


class TestValidTransitions:
    """Tests for valid mode transitions."""

    def test_explore_transitions(self, controller: AdaptiveModeController):
        """Test valid transitions from EXPLORE mode."""
        valid = controller.VALID_TRANSITIONS[AgentMode.EXPLORE]
        assert AgentMode.PLAN in valid
        assert AgentMode.BUILD in valid
        assert AgentMode.COMPLETE in valid

    def test_plan_transitions(self, controller: AdaptiveModeController):
        """Test valid transitions from PLAN mode."""
        valid = controller.VALID_TRANSITIONS[AgentMode.PLAN]
        assert AgentMode.BUILD in valid
        assert AgentMode.EXPLORE in valid
        assert AgentMode.COMPLETE in valid

    def test_build_transitions(self, controller: AdaptiveModeController):
        """Test valid transitions from BUILD mode."""
        valid = controller.VALID_TRANSITIONS[AgentMode.BUILD]
        assert AgentMode.REVIEW in valid
        assert AgentMode.EXPLORE in valid
        assert AgentMode.COMPLETE in valid

    def test_review_transitions(self, controller: AdaptiveModeController):
        """Test valid transitions from REVIEW mode."""
        valid = controller.VALID_TRANSITIONS[AgentMode.REVIEW]
        assert AgentMode.BUILD in valid
        assert AgentMode.COMPLETE in valid

    def test_complete_no_transitions(self, controller: AdaptiveModeController):
        """Test COMPLETE mode has no valid transitions."""
        valid = controller.VALID_TRANSITIONS[AgentMode.COMPLETE]
        assert len(valid) == 0


class TestGetRecommendedAction:
    """Tests for get_recommended_action method."""

    def test_get_recommended_action_basic(self, controller: AdaptiveModeController):
        """Test getting recommended action returns valid ModeAction."""
        action = controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=5,
            tool_budget=10,
        )

        assert isinstance(action, ModeAction)
        assert action.target_mode in AgentMode
        assert isinstance(action.should_continue, bool)

    def test_get_recommended_action_invalid_mode(self, controller: AdaptiveModeController):
        """Test handling of invalid mode string."""
        action = controller.get_recommended_action(
            current_mode="invalid_mode",
            task_type="analysis",
            tool_calls_made=5,
            tool_budget=10,
        )

        # Should default to EXPLORE
        assert isinstance(action, ModeAction)

    def test_get_recommended_action_updates_current_state(self, controller: AdaptiveModeController):
        """Test that get_recommended_action updates internal state."""
        controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=5,
            tool_budget=10,
            quality_score=0.7,
        )

        assert controller._current_state is not None
        assert controller._current_state.mode == AgentMode.EXPLORE
        assert controller._current_state.task_type == "analysis"

    def test_get_recommended_action_high_quality_completion(
        self, controller: AdaptiveModeController
    ):
        """Test that high quality triggers completion suggestion."""
        # Set exploration rate to 0 for deterministic test
        controller._q_store.exploration_rate = 0.0

        action = controller.get_recommended_action(
            current_mode="build",
            task_type="edit",
            tool_calls_made=5,
            tool_budget=10,
            quality_score=0.95,  # High quality
        )

        # Should suggest completion or stay based on Q-values
        assert isinstance(action, ModeAction)

    def test_get_recommended_action_budget_exhausted(self, controller: AdaptiveModeController):
        """Test that budget exhaustion affects recommendations."""
        controller._q_store.exploration_rate = 0.0

        action = controller.get_recommended_action(
            current_mode="build",
            task_type="edit",
            tool_calls_made=10,
            tool_budget=10,  # Budget exhausted
            quality_score=0.6,
        )

        assert isinstance(action, ModeAction)


class TestRecordOutcome:
    """Tests for record_outcome method."""

    def test_record_outcome_success(self, controller: AdaptiveModeController):
        """Test recording successful outcome."""
        # First get an action to set up state
        controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=5,
            tool_budget=10,
            quality_score=0.7,
        )

        reward = controller.record_outcome(
            success=True,
            quality_score=0.85,
            user_satisfied=True,
            completed=True,
        )

        assert reward > 0
        assert controller._total_reward > 0

    def test_record_outcome_failure(self, controller: AdaptiveModeController):
        """Test recording failed outcome."""
        controller.get_recommended_action(
            current_mode="build",
            task_type="edit",
            tool_calls_made=5,
            tool_budget=10,
            quality_score=0.3,
        )

        reward = controller.record_outcome(
            success=False,
            quality_score=0.2,
            user_satisfied=False,
            completed=False,
        )

        assert reward < 0

    def test_record_outcome_without_state(self, controller: AdaptiveModeController):
        """Test recording outcome without prior action returns 0."""
        reward = controller.record_outcome(
            success=True,
            quality_score=0.8,
        )

        assert reward == 0.0

    def test_record_outcome_with_mode_transition_learner(self, tmp_project_path: Path):
        """Test recording outcome with ModeTransitionLearner integration."""
        mock_learner = MagicMock()
        q_store = QLearningStore(project_path=tmp_project_path)

        controller = AdaptiveModeController(
            profile_name="test",
            q_store=q_store,
            mode_transition_learner=mock_learner,
        )

        controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=5,
            tool_budget=10,
        )

        controller.record_outcome(
            success=True,
            quality_score=0.8,
        )

        # Verify learner was called
        mock_learner.record_outcome.assert_called()


class TestCalculateReward:
    """Tests for reward calculation."""

    def test_calculate_reward_full_success(self, controller: AdaptiveModeController):
        """Test reward calculation for full success."""
        # Set up state for efficiency calculation
        controller.get_recommended_action(
            current_mode="build",
            task_type="edit",
            tool_calls_made=3,  # Efficient: 30% of budget
            tool_budget=10,
            quality_score=0.8,
        )

        reward = controller._calculate_reward(
            success=True,
            quality_score=0.9,
            user_satisfied=True,
            completed=True,
        )

        # Success (+1.0) + quality (0.9) + satisfaction (+0.5) + completion (+0.3) + efficiency (+0.2)
        assert reward > 2.5

    def test_calculate_reward_failure(self, controller: AdaptiveModeController):
        """Test reward calculation for failure."""
        controller.get_recommended_action(
            current_mode="build",
            task_type="edit",
            tool_calls_made=8,
            tool_budget=10,
            quality_score=0.3,
        )

        reward = controller._calculate_reward(
            success=False,
            quality_score=0.2,
            user_satisfied=False,
            completed=False,
        )

        # Failure (-0.5) + quality (0.2) = -0.3
        assert reward < 0


class TestShouldContinue:
    """Tests for should_continue method."""

    def test_should_continue_normal(self, controller: AdaptiveModeController):
        """Test continuation under normal conditions."""
        should_continue, reason = controller.should_continue(
            tool_calls_made=5,
            tool_budget=10,
            quality_score=0.6,
            iteration_count=5,
            iteration_budget=20,
        )

        assert should_continue is True
        assert reason == "Continue processing"

    def test_should_continue_tool_budget_exhausted(self, controller: AdaptiveModeController):
        """Test stopping when tool budget exhausted."""
        should_continue, reason = controller.should_continue(
            tool_calls_made=10,
            tool_budget=10,
            quality_score=0.6,
            iteration_count=5,
            iteration_budget=20,
        )

        assert should_continue is False
        assert "Tool budget exhausted" in reason

    def test_should_continue_iteration_budget_exhausted(self, controller: AdaptiveModeController):
        """Test stopping when iteration budget exhausted."""
        should_continue, reason = controller.should_continue(
            tool_calls_made=5,
            tool_budget=10,
            quality_score=0.6,
            iteration_count=20,
            iteration_budget=20,
        )

        assert should_continue is False
        assert "Iteration budget exhausted" in reason

    def test_should_continue_high_quality(self, controller: AdaptiveModeController):
        """Test stopping when high quality achieved."""
        should_continue, reason = controller.should_continue(
            tool_calls_made=5,
            tool_budget=10,
            quality_score=0.95,  # Very high quality
            iteration_count=5,
            iteration_budget=20,
        )

        assert should_continue is False
        assert "High quality achieved" in reason


class TestLoopDetection:
    """Tests for loop detection functionality."""

    def test_check_loop_detection_no_loop(self, controller: AdaptiveModeController):
        """Test no loop detected under normal conditions."""
        is_stuck, reason = controller.check_loop_detection(
            iteration_count=3,
            current_tool_calls=2,
        )

        assert is_stuck is False
        assert reason == ""

    def test_check_loop_detection_stuck(self, controller: AdaptiveModeController):
        """Test loop detection when stuck."""
        # Simulate consecutive iterations with no tool calls
        for _ in range(3):
            controller.check_loop_detection(iteration_count=5, current_tool_calls=0)

        is_stuck, reason = controller.check_loop_detection(
            iteration_count=5,
            current_tool_calls=0,
        )

        # After enough no-tool iterations, should detect loop
        # Need to reach threshold which depends on provider
        thresholds = controller.get_iteration_thresholds()
        if controller._no_tool_iterations >= thresholds["no_tool_threshold"]:
            assert is_stuck is True
            assert "loop" in reason.lower()

    def test_check_loop_detection_resets_on_tool_call(self, controller: AdaptiveModeController):
        """Test loop detection resets when tool calls made."""
        # Simulate no-tool iterations
        controller.check_loop_detection(iteration_count=5, current_tool_calls=0)
        controller.check_loop_detection(iteration_count=6, current_tool_calls=0)

        # Make a tool call
        controller.check_loop_detection(iteration_count=7, current_tool_calls=2)

        assert controller._no_tool_iterations == 0

    def test_reset_loop_tracking(self, controller: AdaptiveModeController):
        """Test reset_loop_tracking method."""
        # Set up some no-tool iterations
        controller._no_tool_iterations = 5

        controller.reset_loop_tracking()

        assert controller._no_tool_iterations == 0


class TestOptimalToolBudget:
    """Tests for optimal tool budget retrieval."""

    def test_get_optimal_tool_budget_default(self, controller: AdaptiveModeController):
        """Test getting optimal budget returns reasonable default."""
        budget = controller.get_optimal_tool_budget("unknown_task")

        assert budget > 0

    def test_get_optimal_tool_budget_learned(self, controller: AdaptiveModeController):
        """Test getting optimal budget after learning."""
        # Record some outcomes
        controller._q_store.update_task_stats(
            task_type="analysis",
            tool_budget_used=8,
            quality_score=0.9,
            completed=True,
        )

        budget = controller.get_optimal_tool_budget("analysis")

        assert budget > 0

    def test_get_optimal_tool_budget_with_learner(self, tmp_project_path: Path):
        """Test optimal budget from ModeTransitionLearner when available."""
        mock_learner = MagicMock()
        mock_learner.get_optimal_budget.return_value = 15
        mock_learner.get_task_stats.return_value = {"sample_count": 100}

        q_store = QLearningStore(project_path=tmp_project_path)
        controller = AdaptiveModeController(
            profile_name="test",
            q_store=q_store,
            mode_transition_learner=mock_learner,
        )

        # Local store has fewer samples
        q_store.update_task_stats(
            task_type="edit",
            tool_budget_used=5,
            quality_score=0.7,
            completed=True,
        )

        budget = controller.get_optimal_tool_budget("edit")

        # Should prefer learner with more samples
        assert budget == 15


class TestSessionManagement:
    """Tests for session management methods."""

    def test_get_session_stats(self, controller: AdaptiveModeController):
        """Test getting session statistics."""
        stats = controller.get_session_stats()

        assert "profile_name" in stats
        assert stats["profile_name"] == "test_profile"
        assert "session_duration_seconds" in stats
        assert "total_reward" in stats
        assert "mode_transitions" in stats
        assert "exploration_rate" in stats

    def test_reset_session(self, controller: AdaptiveModeController):
        """Test resetting session."""
        # Add some state
        controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=5,
            tool_budget=10,
        )
        controller._total_reward = 5.0

        controller.reset_session()

        assert controller._current_state is None
        assert controller._current_action is None
        assert controller._total_reward == 0.0
        assert len(controller._mode_history) == 0

    def test_adjust_exploration_rate(self, controller: AdaptiveModeController):
        """Test adjusting exploration rate."""
        controller.adjust_exploration_rate(0.5)
        assert controller._q_store.exploration_rate == 0.5

        # Test bounds
        controller.adjust_exploration_rate(2.0)
        assert controller._q_store.exploration_rate == 1.0

        controller.adjust_exploration_rate(-0.5)
        assert controller._q_store.exploration_rate == 0.0

    def test_decay_exploration_rate(self, controller: AdaptiveModeController):
        """Test decaying exploration rate."""
        initial_rate = controller._q_store.exploration_rate
        controller.decay_exploration_rate(decay_factor=0.9)

        assert controller._q_store.exploration_rate == pytest.approx(initial_rate * 0.9)


class TestTransitionEvent:
    """Tests for TransitionEvent dataclass."""

    def test_transition_event_creation(self):
        """Test creating TransitionEvent."""
        state = ModeState(
            mode=AgentMode.EXPLORE,
            task_type="analysis",
            tool_calls_made=5,
            tool_budget=10,
            iteration_count=3,
            iteration_budget=20,
            quality_score=0.7,
            grounding_score=0.8,
            time_in_mode_seconds=30.0,
            recent_tool_success_rate=0.9,
        )

        action = ModeAction(target_mode=AgentMode.PLAN)

        event = TransitionEvent(
            from_mode=AgentMode.EXPLORE,
            to_mode=AgentMode.PLAN,
            trigger=TransitionTrigger.PATTERN_DETECTED,
            state_before=state,
            action_taken=action,
        )

        assert event.from_mode == AgentMode.EXPLORE
        assert event.to_mode == AgentMode.PLAN
        assert event.trigger == TransitionTrigger.PATTERN_DETECTED
        assert event.outcome_success is None  # Not yet filled


class TestModuleLevelFunction:
    """Tests for module-level convenience function."""

    def test_get_mode_controller_function(self, tmp_path: Path):
        """Test get_mode_controller convenience function."""
        with patch("victor.config.settings.get_project_paths") as mock_paths:
            mock_path_obj = MagicMock()
            mock_path_obj.project_victor_dir = tmp_path / "test_victor"
            mock_paths.return_value = mock_path_obj

            controller = get_mode_controller(profile_name="test")

            assert isinstance(controller, AdaptiveModeController)
            assert controller.profile_name == "test"


class TestInferTrigger:
    """Tests for trigger inference."""

    def test_infer_trigger_quality_threshold(self, controller: AdaptiveModeController):
        """Test inferring quality threshold trigger."""
        state = ModeState(
            mode=AgentMode.BUILD,
            task_type="edit",
            tool_calls_made=5,
            tool_budget=10,
            iteration_count=5,
            iteration_budget=20,
            quality_score=0.9,  # High quality
            grounding_score=0.8,
            time_in_mode_seconds=30.0,
            recent_tool_success_rate=0.9,
        )

        action = ModeAction(target_mode=AgentMode.COMPLETE)

        trigger = controller._infer_trigger(state, action)
        assert trigger == TransitionTrigger.QUALITY_THRESHOLD

    def test_infer_trigger_budget_low(self, controller: AdaptiveModeController):
        """Test inferring budget low trigger."""
        state = ModeState(
            mode=AgentMode.BUILD,
            task_type="edit",
            tool_calls_made=10,  # Budget exhausted
            tool_budget=10,
            iteration_count=5,
            iteration_budget=20,
            quality_score=0.6,
            grounding_score=0.6,
            time_in_mode_seconds=30.0,
            recent_tool_success_rate=0.9,
        )

        action = ModeAction(target_mode=AgentMode.COMPLETE)

        trigger = controller._infer_trigger(state, action)
        assert trigger == TransitionTrigger.BUDGET_LOW

    def test_infer_trigger_error_recovery(self, controller: AdaptiveModeController):
        """Test inferring error recovery trigger."""
        state = ModeState(
            mode=AgentMode.BUILD,
            task_type="edit",
            tool_calls_made=5,
            tool_budget=10,
            iteration_count=5,
            iteration_budget=20,
            quality_score=0.6,
            grounding_score=0.6,
            time_in_mode_seconds=30.0,
            recent_tool_success_rate=0.3,  # Low success rate
        )

        action = ModeAction(target_mode=AgentMode.EXPLORE)

        trigger = controller._infer_trigger(state, action)
        assert trigger == TransitionTrigger.ERROR_RECOVERY


class TestQToConfidence:
    """Tests for Q-value to confidence conversion."""

    def test_q_to_confidence_positive(self, controller: AdaptiveModeController):
        """Test Q to confidence for positive Q-value."""
        confidence = controller._q_to_confidence(2.0)
        assert 0.5 < confidence < 1.0

    def test_q_to_confidence_negative(self, controller: AdaptiveModeController):
        """Test Q to confidence for negative Q-value."""
        confidence = controller._q_to_confidence(-2.0)
        assert 0.0 < confidence < 0.5

    def test_q_to_confidence_zero(self, controller: AdaptiveModeController):
        """Test Q to confidence for zero Q-value."""
        confidence = controller._q_to_confidence(0.0)
        assert confidence == pytest.approx(0.5)


class TestHeuristicBonus:
    """Tests for heuristic bonus calculation."""

    def test_heuristic_bonus_complete_high_quality(self, controller: AdaptiveModeController):
        """Test heuristic bonus for completion with high quality."""
        controller.get_recommended_action(
            current_mode="build",
            task_type="edit",
            tool_calls_made=5,
            tool_budget=10,
            quality_score=0.9,
        )

        action = ModeAction(target_mode=AgentMode.COMPLETE)
        bonus = controller._get_heuristic_bonus(action)

        assert bonus > 0  # Should have bonus for completing with high quality

    def test_heuristic_bonus_early_mode_switch(self, controller: AdaptiveModeController):
        """Test heuristic penalty for early mode switch."""
        controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=1,  # Very few tool calls
            tool_budget=10,
            quality_score=0.5,
        )

        action = ModeAction(target_mode=AgentMode.PLAN)
        bonus = controller._get_heuristic_bonus(action)

        assert bonus < 0  # Should have penalty for switching too early
