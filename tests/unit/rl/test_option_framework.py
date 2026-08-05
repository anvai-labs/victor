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

"""Unit tests for option_framework.

Tests the Option abstraction for hierarchical RL.
"""

import pytest
from unittest.mock import MagicMock

from victor.framework.rl.option_framework import (
    OptionStatus,
    OptionState,
    OptionResult,
    Option,
    ExploreOption,
    ImplementOption,
    DebugOption,
    ReviewOption,
    OptionRegistry,
)


class TestOptionStatus:
    """Tests for OptionStatus enum."""

    def test_status_values(self) -> None:
        """Test status enum values."""
        assert OptionStatus.INACTIVE.value == "inactive"
        assert OptionStatus.RUNNING.value == "running"
        assert OptionStatus.COMPLETED.value == "completed"
        assert OptionStatus.TERMINATED.value == "terminated"
        assert OptionStatus.FAILED.value == "failed"


class TestOptionState:
    """Tests for OptionState dataclass."""

    def test_default_state(self) -> None:
        """Test default state values."""
        state = OptionState()

        assert state.current_mode == "build"
        assert state.tools_used == []
        assert state.iterations == 0
        assert state.context_size == 0
        assert state.task_progress == 0.0
        assert state.last_tool_success is True

    def test_custom_state(self) -> None:
        """Test custom state values."""
        state = OptionState(
            current_mode="explore",
            tools_used=["read", "code_search"],
            iterations=5,
            context_size=10000,
            task_progress=0.5,
            last_tool_success=False,
        )

        assert state.current_mode == "explore"
        assert len(state.tools_used) == 2
        assert state.iterations == 5
        assert state.context_size == 10000
        assert state.task_progress == 0.5
        assert state.last_tool_success is False

    def test_to_tuple(self) -> None:
        """Test state conversion to tuple."""
        state = OptionState(
            current_mode="build",
            tools_used=["a", "b", "c"],
            iterations=15,
            context_size=25000,
            task_progress=0.75,
            last_tool_success=True,
        )

        tuple_repr = state.to_tuple()

        assert tuple_repr[0] == "build"  # mode
        assert tuple_repr[1] == 3  # tool count
        assert tuple_repr[2] == 3  # bucketed iterations (15 // 5)
        assert tuple_repr[3] == 2  # bucketed context (25000 // 10000)
        assert tuple_repr[4] == 7  # discretized progress (0.75 * 10)
        assert tuple_repr[5] is True  # last success

    def test_to_tuple_bucketing(self) -> None:
        """Test bucketing in to_tuple."""
        # Test iteration bucketing max
        state = OptionState(iterations=100)
        assert state.to_tuple()[2] == 10  # Capped at 10

        # Test context bucketing max
        state = OptionState(context_size=200000)
        assert state.to_tuple()[3] == 10  # Capped at 10


class TestOptionResult:
    """Tests for OptionResult dataclass."""

    def test_result_creation(self) -> None:
        """Test result creation."""
        result = OptionResult(
            status=OptionStatus.COMPLETED,
            reward=0.8,
            steps=10,
        )

        assert result.status == OptionStatus.COMPLETED
        assert result.reward == 0.8
        assert result.steps == 10
        assert result.final_state is None
        assert result.metadata == {}

    def test_result_with_metadata(self) -> None:
        """Test result with metadata."""
        result = OptionResult(
            status=OptionStatus.COMPLETED,
            metadata={"name": "explore", "duration": 5.5},
        )

        assert result.metadata["name"] == "explore"
        assert result.metadata["duration"] == 5.5


class TestExploreOption:
    """Tests for ExploreOption."""

    @pytest.fixture
    def option(self) -> ExploreOption:
        """Create ExploreOption fixture."""
        return ExploreOption()

    def test_initialization(self, option: ExploreOption) -> None:
        """Test explore option initialization."""
        assert option.name == "explore_codebase"
        assert option.status == OptionStatus.INACTIVE
        assert not option.is_active

    def test_can_initiate_low_context(self, option: ExploreOption) -> None:
        """Test initiation with low context."""
        state = OptionState(context_size=1000)
        assert option.can_initiate(state) is True

    def test_can_initiate_explore_mode(self, option: ExploreOption) -> None:
        """Test initiation in explore mode."""
        state = OptionState(current_mode="explore", context_size=50000)
        assert option.can_initiate(state) is True

    def test_can_initiate_low_progress(self, option: ExploreOption) -> None:
        """Test initiation with low progress."""
        state = OptionState(task_progress=0.1, context_size=50000)
        assert option.can_initiate(state) is True

    def test_should_terminate_max_steps(self, option: ExploreOption) -> None:
        """Test termination at max steps."""
        option.start(OptionState(context_size=1000))
        option._steps = 15
        assert option.should_terminate(OptionState()) is True

    def test_should_terminate_enough_context(self, option: ExploreOption) -> None:
        """Test termination with enough context."""
        option.start(OptionState(context_size=1000))
        state = OptionState(context_size=15000)
        assert option.should_terminate(state) is True

    def test_get_action_prioritizes_search(self, option: ExploreOption) -> None:
        """Test action prioritizes search."""
        state = OptionState(tools_used=[])
        action = option.get_action(state)
        assert action == "semantic_code_search"

    def test_get_action_then_read(self, option: ExploreOption) -> None:
        """Test action sequence."""
        state = OptionState(tools_used=["semantic_code_search"])
        action = option.get_action(state)
        assert action == "read"

    def test_get_action_normalizes_legacy_history(self, option: ExploreOption) -> None:
        """Test legacy history still yields canonical actions."""
        state = OptionState(tools_used=["semantic_code_search", "read_file"])
        action = option.get_action(state)
        assert action == "ls"

    def test_start_and_step(self, option: ExploreOption) -> None:
        """Test option lifecycle."""
        state = OptionState(context_size=1000)

        # Start
        assert option.start(state) is True
        assert option.is_active

        # Step
        action = option.step(state, reward=0.1)
        assert action is not None
        assert option._steps == 1
        assert option._cumulative_reward == 0.1


class TestImplementOption:
    """Tests for ImplementOption."""

    @pytest.fixture
    def option(self) -> ImplementOption:
        """Create ImplementOption fixture."""
        return ImplementOption()

    def test_initialization(self, option: ImplementOption) -> None:
        """Test implement option initialization."""
        assert option.name == "implement_feature"

    def test_can_initiate_with_context(self, option: ImplementOption) -> None:
        """Test initiation with sufficient context."""
        state = OptionState(context_size=5000)
        assert option.can_initiate(state) is True

    def test_can_initiate_with_progress(self, option: ImplementOption) -> None:
        """Test initiation with progress."""
        state = OptionState(task_progress=0.4)
        assert option.can_initiate(state) is True

    def test_can_initiate_build_mode(self, option: ImplementOption) -> None:
        """Test initiation in build mode."""
        state = OptionState(current_mode="build")
        assert option.can_initiate(state) is True

    def test_should_terminate_max_steps(self, option: ImplementOption) -> None:
        """Test termination at max steps."""
        option.start(OptionState(context_size=5000))
        option._steps = 20
        assert option.should_terminate(OptionState()) is True

    def test_should_terminate_task_complete(self, option: ImplementOption) -> None:
        """Test termination when task complete."""
        option.start(OptionState(context_size=5000))
        state = OptionState(task_progress=0.9)
        assert option.should_terminate(state) is True

    def test_get_action_after_write(self, option: ImplementOption) -> None:
        """Test action after writing."""
        state = OptionState(tools_used=["write"])
        action = option.get_action(state)
        assert action == "run_tests"

    def test_get_action_after_failed_tests(self, option: ImplementOption) -> None:
        """Test action after test failure."""
        state = OptionState(
            tools_used=["run_tests"],
            last_tool_success=False,
        )
        action = option.get_action(state)
        assert action == "edit"


class TestDebugOption:
    """Tests for DebugOption."""

    @pytest.fixture
    def option(self) -> DebugOption:
        """Create DebugOption fixture."""
        return DebugOption()

    def test_initialization(self, option: DebugOption) -> None:
        """Test debug option initialization."""
        assert option.name == "debug_issue"

    def test_can_initiate_after_failure(self, option: DebugOption) -> None:
        """Test initiation after failure."""
        state = OptionState(last_tool_success=False)
        assert option.can_initiate(state) is True

    def test_can_initiate_debug_mode(self, option: DebugOption) -> None:
        """Test initiation in debug mode."""
        state = OptionState(current_mode="debug")
        assert option.can_initiate(state) is True

    def test_should_terminate_max_steps(self, option: DebugOption) -> None:
        """Test termination at max steps."""
        option.start(OptionState(last_tool_success=False))
        option._steps = 10
        assert option.should_terminate(OptionState()) is True

    def test_should_terminate_fixed(self, option: DebugOption) -> None:
        """Test termination when fixed."""
        option.start(OptionState(last_tool_success=False))
        option._steps = 3
        state = OptionState(last_tool_success=True)
        assert option.should_terminate(state) is True

    def test_get_action_first_step(self, option: DebugOption) -> None:
        """Test first action is read."""
        option._steps = 0
        state = OptionState()
        action = option.get_action(state)
        assert action == "read"

    def test_get_action_after_read(self, option: DebugOption) -> None:
        """Test action after reading."""
        option._steps = 1
        state = OptionState(tools_used=["read"])
        action = option.get_action(state)
        assert action == "edit"


class TestReviewOption:
    """Tests for ReviewOption."""

    @pytest.fixture
    def option(self) -> ReviewOption:
        """Create ReviewOption fixture."""
        return ReviewOption()

    def test_initialization(self, option: ReviewOption) -> None:
        """Test review option initialization."""
        assert option.name == "review_work"

    def test_can_initiate_high_progress(self, option: ReviewOption) -> None:
        """Test initiation with high progress."""
        state = OptionState(task_progress=0.8)
        assert option.can_initiate(state) is True

    def test_can_initiate_review_mode(self, option: ReviewOption) -> None:
        """Test initiation in review mode."""
        state = OptionState(current_mode="review")
        assert option.can_initiate(state) is True

    def test_should_terminate_max_steps(self, option: ReviewOption) -> None:
        """Test termination at max steps."""
        option.start(OptionState(task_progress=0.8))
        option._steps = 8
        assert option.should_terminate(OptionState()) is True

    def test_get_action_sequence(self, option: ReviewOption) -> None:
        """Test action sequence."""
        state = OptionState()

        option._steps = 0
        assert option.get_action(state) == "run_tests"

        option._steps = 1
        assert option.get_action(state) == "read"

        option._steps = 2
        assert option.get_action(state) == "code_search"


class TestOptionLifecycle:
    """Tests for option lifecycle management."""

    def test_start_success(self) -> None:
        """Test successful start."""
        option = ExploreOption()
        state = OptionState(context_size=1000)

        result = option.start(state)

        assert result is True
        assert option.status == OptionStatus.RUNNING
        assert option.is_active
        assert option._steps == 0

    def test_start_failure(self) -> None:
        """Test failed start (cannot initiate)."""
        option = ReviewOption()
        state = OptionState(task_progress=0.1)  # Too low for review

        result = option.start(state)

        assert result is False
        assert option.status == OptionStatus.INACTIVE

    def test_step_accumulates_reward(self) -> None:
        """Test reward accumulation."""
        option = ExploreOption()
        state = OptionState(context_size=1000)
        option.start(state)

        option.step(state, reward=0.5)
        option.step(state, reward=0.3)
        option.step(state, reward=0.2)

        assert option._cumulative_reward == 1.0
        assert option._steps == 3

    def test_step_returns_none_when_terminated(self) -> None:
        """Test step returns None when terminated."""
        option = ExploreOption()
        state = OptionState(context_size=1000)
        option.start(state)

        # Force termination condition
        option._steps = 15

        action = option.step(state)

        assert action is None
        assert option.status == OptionStatus.COMPLETED

    def test_terminate_success(self) -> None:
        """Test successful termination."""
        option = ExploreOption()
        state = OptionState(context_size=1000)
        option.start(state)
        option._cumulative_reward = 0.8
        option._steps = 5

        result = option.terminate(success=True)

        assert result.status == OptionStatus.COMPLETED
        assert result.reward == 0.8
        assert result.steps == 5
        assert "name" in result.metadata

    def test_terminate_failure(self) -> None:
        """Test failed termination."""
        option = ExploreOption()
        state = OptionState(context_size=1000)
        option.start(state)

        result = option.terminate(success=False)

        assert result.status == OptionStatus.TERMINATED


class TestOptionRegistry:
    """Tests for OptionRegistry."""

    @pytest.fixture
    def registry(self) -> OptionRegistry:
        """Create OptionRegistry fixture."""
        return OptionRegistry()

    def test_initialization(self, registry: OptionRegistry) -> None:
        """Test registry initialization."""
        assert len(registry._options) == 4  # Default options
        assert registry.active_option is None

    def test_default_options_registered(self, registry: OptionRegistry) -> None:
        """Test default options are registered."""
        assert registry.get_option("explore_codebase") is not None
        assert registry.get_option("implement_feature") is not None
        assert registry.get_option("debug_issue") is not None
        assert registry.get_option("review_work") is not None

    def test_register_custom_option(self, registry: OptionRegistry) -> None:
        """Test registering custom option."""
        custom = ExploreOption()
        custom.name = "custom_explore"
        registry.register(custom)

        assert registry.get_option("custom_explore") is not None

    def test_get_available_options(self, registry: OptionRegistry) -> None:
        """Test getting available options."""
        state = OptionState(context_size=1000, task_progress=0.1)

        available = registry.get_available_options(state)

        # ExploreOption should be available (low context)
        # ImplementOption may be available (low progress but build mode default)
        assert len(available) >= 1
        option_names = [o.name for o in available]
        assert "explore_codebase" in option_names

    def test_get_option_not_found(self, registry: OptionRegistry) -> None:
        """Test getting non-existent option."""
        result = registry.get_option("nonexistent")
        assert result is None

    def test_start_option(self, registry: OptionRegistry) -> None:
        """Test starting an option."""
        state = OptionState(context_size=1000)

        result = registry.start_option("explore_codebase", state)

        assert result is True
        assert registry.active_option is not None
        assert registry.active_option.name == "explore_codebase"

    def test_start_option_not_found(self, registry: OptionRegistry) -> None:
        """Test starting non-existent option."""
        state = OptionState()

        result = registry.start_option("nonexistent", state)

        assert result is False
        assert registry.active_option is None

    def test_start_option_cannot_initiate(self, registry: OptionRegistry) -> None:
        """Test starting option that cannot initiate."""
        state = OptionState(task_progress=0.1)  # Too low for review

        result = registry.start_option("review_work", state)

        assert result is False
        assert registry.active_option is None

    def test_step_active_option(self, registry: OptionRegistry) -> None:
        """Test stepping active option."""
        state = OptionState(context_size=1000)
        registry.start_option("explore_codebase", state)

        action = registry.step_active_option(state, reward=0.5)

        assert action is not None

    def test_step_active_option_none(self, registry: OptionRegistry) -> None:
        """Test stepping when no active option."""
        state = OptionState()

        action = registry.step_active_option(state)

        assert action is None

    def test_step_clears_on_termination(self, registry: OptionRegistry) -> None:
        """Test stepping clears active option on termination."""
        state = OptionState(context_size=1000)
        registry.start_option("explore_codebase", state)

        # Force termination
        registry.active_option._steps = 15

        action = registry.step_active_option(state)

        assert action is None
        assert registry.active_option is None

    def test_terminate_active_option(self, registry: OptionRegistry) -> None:
        """Test terminating active option."""
        state = OptionState(context_size=1000)
        registry.start_option("explore_codebase", state)

        result = registry.terminate_active_option(success=True)

        assert result is not None
        assert result.status == OptionStatus.COMPLETED
        assert registry.active_option is None

    def test_terminate_no_active_option(self, registry: OptionRegistry) -> None:
        """Test terminating when no active option."""
        result = registry.terminate_active_option()

        assert result is None

    def test_export_metrics(self, registry: OptionRegistry) -> None:
        """Test exporting metrics."""
        metrics = registry.export_metrics()

        assert metrics["total_options"] == 4
        assert "explore_codebase" in metrics["option_names"]
        assert metrics["active_option"] is None

    def test_export_metrics_with_active(self, registry: OptionRegistry) -> None:
        """Test exporting metrics with active option."""
        state = OptionState(context_size=1000)
        registry.start_option("explore_codebase", state)

        metrics = registry.export_metrics()

        assert metrics["active_option"] == "explore_codebase"
