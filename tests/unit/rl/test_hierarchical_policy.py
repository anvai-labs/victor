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

"""Unit tests for HierarchicalPolicy and Option Framework.

Tests the hierarchical RL components for multi-step task planning.
"""

import pytest
from unittest.mock import MagicMock

from victor.framework.rl.option_framework import (
    Option,
    OptionRegistry,
    OptionResult,
    OptionState,
    OptionStatus,
    ExploreOption,
    ImplementOption,
    DebugOption,
    ReviewOption,
)
from victor.framework.rl.hierarchical_policy import (
    HierarchicalPolicy,
    HierarchicalState,
    PolicyDecision,
    get_hierarchical_policy,
)
from victor.framework.rl.base import RLOutcome

# =============================================================================
# Option Framework Tests
# =============================================================================


class TestOptionState:
    """Tests for OptionState."""

    def test_option_state_creation(self) -> None:
        """Test creating option state."""
        state = OptionState(
            current_mode="explore",
            tools_used=["read", "code_search"],
            iterations=5,
            context_size=10000,
            task_progress=0.3,
        )

        assert state.current_mode == "explore"
        assert len(state.tools_used) == 2
        assert state.iterations == 5

    def test_option_state_to_tuple(self) -> None:
        """Test converting state to hashable tuple."""
        state = OptionState(
            current_mode="build",
            iterations=7,
            context_size=15000,
        )

        t = state.to_tuple()
        assert isinstance(t, tuple)
        assert t[0] == "build"

    def test_option_state_defaults(self) -> None:
        """Test default values for option state."""
        state = OptionState()

        assert state.current_mode == "build"
        assert state.tools_used == []
        assert state.iterations == 0
        assert state.last_tool_success is True


class TestExploreOption:
    """Tests for ExploreOption."""

    @pytest.fixture
    def option(self) -> ExploreOption:
        return ExploreOption()

    def test_can_initiate_explore_mode(self, option: ExploreOption) -> None:
        """Test initiation in explore mode."""
        state = OptionState(current_mode="explore", context_size=20000)
        assert option.can_initiate(state) is True

    def test_should_terminate_max_steps(self, option: ExploreOption) -> None:
        """Test termination after max steps."""
        state = OptionState()
        option.start(state)

        # Simulate max steps
        for _ in range(option.MAX_EXPLORE_STEPS):
            option.step(state)

        assert option.should_terminate(state) is True

    def test_should_terminate_sufficient_context(self, option: ExploreOption) -> None:
        """Test termination with sufficient context."""
        state = OptionState(context_size=option.MIN_CONTEXT_FOR_COMPLETION * 2)
        option.start(state)
        option._steps = 1  # At least one step

        assert option.should_terminate(state) is True

    def test_get_action_then_read(self, option: ExploreOption) -> None:
        """Test action progression to reading."""
        state = OptionState(tools_used=["semantic_code_search", "code_search"])

        action = option.get_action(state)
        assert action == "read"


class TestImplementOption:
    """Tests for ImplementOption."""

    @pytest.fixture
    def option(self) -> ImplementOption:
        return ImplementOption()

    def test_can_initiate_sufficient_context(self, option: ImplementOption) -> None:
        """Test initiation with sufficient context."""
        state = OptionState(context_size=5000)
        assert option.can_initiate(state) is True

    def test_can_initiate_build_mode(self, option: ImplementOption) -> None:
        """Test initiation in build mode."""
        state = OptionState(current_mode="build", context_size=0)
        assert option.can_initiate(state) is True

    def test_should_terminate_task_complete(self, option: ImplementOption) -> None:
        """Test termination when task complete."""
        state = OptionState(task_progress=0.9)
        option.start(state)

        assert option.should_terminate(state) is True

    def test_get_action_after_write_runs_tests(self, option: ImplementOption) -> None:
        """Test running tests after writing."""
        state = OptionState(tools_used=["read", "edit"])

        action = option.get_action(state)
        assert action == "run_tests"


class TestDebugOption:
    """Tests for DebugOption."""

    @pytest.fixture
    def option(self) -> DebugOption:
        return DebugOption()

    def test_should_terminate_after_fix(self, option: DebugOption) -> None:
        """Test termination after successful fix."""
        state = OptionState(last_tool_success=True)
        option.start(state)
        option._steps = 3

        assert option.should_terminate(state) is True


class TestReviewOption:
    """Tests for ReviewOption."""

    @pytest.fixture
    def option(self) -> ReviewOption:
        return ReviewOption()

    def test_can_initiate_near_completion(self, option: ReviewOption) -> None:
        """Test initiation when task nearly complete."""
        state = OptionState(task_progress=0.8)
        assert option.can_initiate(state) is True


class TestOptionLifecycle:
    """Tests for option lifecycle management."""

    def test_option_start(self) -> None:
        """Test starting an option."""
        option = ExploreOption()
        state = OptionState(context_size=1000)

        result = option.start(state)

        assert result is True
        assert option.status == OptionStatus.RUNNING
        assert option.is_active is True

    def test_option_start_fails_when_not_initiable(self) -> None:
        """Test starting option that can't initiate."""
        option = ReviewOption()
        state = OptionState(task_progress=0.2)  # Too early for review

        result = option.start(state)

        assert result is False
        assert option.status == OptionStatus.INACTIVE

    def test_option_step_returns_action(self) -> None:
        """Test stepping returns action."""
        option = ExploreOption()
        state = OptionState(context_size=1000)
        option.start(state)

        action = option.step(state)

        assert action is not None
        assert option._steps == 1

    def test_option_step_accumulates_reward(self) -> None:
        """Test stepping accumulates reward."""
        option = ExploreOption()
        state = OptionState(context_size=1000)
        option.start(state)

        option.step(state, reward=0.5)
        option.step(state, reward=0.3)

        assert option._cumulative_reward == pytest.approx(0.8)

    def test_option_terminate(self) -> None:
        """Test terminating an option."""
        option = ExploreOption()
        state = OptionState(context_size=1000)
        option.start(state)
        option.step(state, reward=0.5)

        result = option.terminate(success=True)

        assert result.status == OptionStatus.COMPLETED
        assert result.reward == 0.5
        assert result.steps == 1


class TestOptionRegistry:
    """Tests for OptionRegistry."""

    @pytest.fixture
    def registry(self) -> OptionRegistry:
        return OptionRegistry()

    def test_registry_has_default_options(self, registry: OptionRegistry) -> None:
        """Test registry initialized with default options."""
        assert registry.get_option("explore_codebase") is not None
        assert registry.get_option("implement_feature") is not None
        assert registry.get_option("debug_issue") is not None
        assert registry.get_option("review_work") is not None

    def test_get_available_options(self, registry: OptionRegistry) -> None:
        """Test getting available options."""
        state = OptionState(context_size=1000, last_tool_success=False)

        available = registry.get_available_options(state)

        assert len(available) > 0
        # Explore and Debug should be available
        names = [opt.name for opt in available]
        assert "explore_codebase" in names
        assert "debug_issue" in names

    def test_step_active_option(self, registry: OptionRegistry) -> None:
        """Test stepping active option."""
        state = OptionState(context_size=1000)
        registry.start_option("explore_codebase", state)

        action = registry.step_active_option(state)

        assert action is not None

    def test_terminate_active_option(self, registry: OptionRegistry) -> None:
        """Test terminating active option."""
        state = OptionState(context_size=1000)
        registry.start_option("explore_codebase", state)
        registry.step_active_option(state, reward=0.5)

        result = registry.terminate_active_option(success=True)

        assert result is not None
        assert result.status == OptionStatus.COMPLETED
        assert registry.active_option is None

    def test_export_metrics(self, registry: OptionRegistry) -> None:
        """Test exporting registry metrics."""
        metrics = registry.export_metrics()

        assert metrics["total_options"] == 4
        assert "explore_codebase" in metrics["option_names"]


# =============================================================================
# Hierarchical Policy Tests
# =============================================================================


class TestHierarchicalState:
    """Tests for HierarchicalState."""

    def test_hierarchical_state_creation(self) -> None:
        """Test creating hierarchical state."""
        state = HierarchicalState(
            task_type="action",
            task_complexity=0.7,
            current_stage="planning",
            context_gathered=0.5,
        )

        assert state.task_type == "action"
        assert state.task_complexity == 0.7

    def test_hierarchical_state_to_key(self) -> None:
        """Test converting to hashable key."""
        state = HierarchicalState(
            task_type="analysis",
            task_complexity=0.3,  # low
            current_stage="initial",
            context_gathered=0.5,  # medium
        )

        key = state.to_key()

        assert "analysis" in key
        assert "low" in key
        assert "initial" in key
        assert "medium" in key

    def test_hierarchical_state_complexity_buckets(self) -> None:
        """Test complexity bucketing."""
        low = HierarchicalState(task_complexity=0.2).to_key()
        medium = HierarchicalState(task_complexity=0.5).to_key()
        high = HierarchicalState(task_complexity=0.8).to_key()

        assert "low" in low
        assert "medium" in medium
        assert "high" in high


class TestPolicyDecision:
    """Tests for PolicyDecision."""

    def test_policy_decision_option(self) -> None:
        """Test decision for option selection."""
        decision = PolicyDecision(
            option_name="explore_codebase",
            confidence=0.8,
            reason="Exploitation",
            is_option=True,
        )

        assert decision.option_name == "explore_codebase"
        assert decision.is_option is True

    def test_policy_decision_primitive(self) -> None:
        """Test decision for primitive action."""
        decision = PolicyDecision(
            primitive_action="continue",
            confidence=0.5,
            is_option=False,
        )

        assert decision.primitive_action == "continue"
        assert decision.is_option is False


class TestHierarchicalPolicy:
    """Tests for HierarchicalPolicy."""

    @pytest.fixture
    def policy(self) -> HierarchicalPolicy:
        return HierarchicalPolicy(epsilon=0.0)  # No exploration for deterministic tests

    def test_initialization(self, policy: HierarchicalPolicy) -> None:
        """Test policy initialization."""
        assert policy.learning_rate == HierarchicalPolicy.DEFAULT_LEARNING_RATE
        assert policy.discount_factor == HierarchicalPolicy.DEFAULT_DISCOUNT_FACTOR
        assert len(policy.option_names) == 4

    def test_get_decision_returns_option(self, policy: HierarchicalPolicy) -> None:
        """Test getting decision returns an option."""
        state = HierarchicalState(
            task_type="action",
            context_gathered=0.1,  # Low context
        )

        decision = policy.get_decision(state)

        assert decision.is_option is True
        assert decision.option_name in policy.option_names

    def test_get_decision_increments_count(self, policy: HierarchicalPolicy) -> None:
        """Test decision increments total count."""
        state = HierarchicalState()

        policy.get_decision(state)
        policy.get_decision(state)

        assert policy._total_decisions == 2

    def test_start_option(self, policy: HierarchicalPolicy) -> None:
        """Test starting an option."""
        state = HierarchicalState(context_gathered=0.1)

        result = policy.start_option("explore_codebase", state)

        assert result is True
        assert policy.has_active_option() is True

    def test_step_option(self, policy: HierarchicalPolicy) -> None:
        """Test stepping an option."""
        state = HierarchicalState(context_gathered=0.1)
        policy.start_option("explore_codebase", state)

        action = policy.step_option(state, reward=0.5)

        assert action is not None

    def test_terminate_option(self, policy: HierarchicalPolicy) -> None:
        """Test terminating an option."""
        state = HierarchicalState(context_gathered=0.1)
        policy.start_option("explore_codebase", state)
        policy.step_option(state, reward=0.5)

        result = policy.terminate_current_option(success=True)

        assert result is not None
        assert policy.has_active_option() is False

    def test_record_option_outcome(self, policy: HierarchicalPolicy) -> None:
        """Test recording option outcome."""
        state = HierarchicalState(task_type="action")

        policy.record_option_outcome(
            option_name="implement_feature",
            success=True,
            reward=0.8,
            start_state=state,
        )

        assert policy._option_completions["implement_feature"] == 1
        assert len(policy._option_success_rate["implement_feature"]) == 1

    def test_q_value_update(self, policy: HierarchicalPolicy) -> None:
        """Test Q-value is updated after outcome."""
        state = HierarchicalState(task_type="action")
        state_key = state.to_key()

        initial_q = policy._q_table[state_key]["implement_feature"]

        policy.record_option_outcome(
            option_name="implement_feature",
            success=True,
            reward=1.0,
            start_state=state,
        )

        new_q = policy._q_table[state_key]["implement_feature"]
        assert new_q != initial_q  # Q-value should change

    def test_exploration(self) -> None:
        """Test exploration with high epsilon."""
        policy = HierarchicalPolicy(epsilon=1.0)  # Always explore
        state = HierarchicalState()

        decisions = set()
        for _ in range(20):
            decision = policy.get_decision(state)
            decisions.add(decision.option_name)

        # Should see multiple options due to exploration
        assert len(decisions) > 1

    def test_get_recommendation(self, policy: HierarchicalPolicy) -> None:
        """Test BaseLearner interface."""
        rec = policy.get_recommendation("ollama", "qwen", "action")

        assert rec is not None
        assert rec.value in policy.option_names or rec.value == "continue"

    def test_record_outcome(self, policy: HierarchicalPolicy) -> None:
        """Test BaseLearner outcome recording."""
        outcome = RLOutcome(
            provider="ollama",
            model="qwen",
            task_type="action",
            success=True,
            quality_score=0.8,
            metadata={"option_name": "implement_feature"},
        )

        policy.record_outcome(outcome)

        assert policy._option_completions["implement_feature"] == 1

    def test_export_metrics(self, policy: HierarchicalPolicy) -> None:
        """Test exporting metrics."""
        state = HierarchicalState()
        policy.get_decision(state)
        policy.record_option_outcome("explore_codebase", True, 0.8, state)

        metrics = policy.export_metrics()

        assert metrics["total_decisions"] == 1
        assert "explore_codebase" in metrics["option_completions"]
        assert "epsilon" in metrics


class TestGlobalSingleton:
    """Tests for global singleton."""

    def test_get_hierarchical_policy(self) -> None:
        """Test getting global singleton."""
        import victor.framework.rl.hierarchical_policy as module

        module._hierarchical_policy = None

        policy1 = get_hierarchical_policy()
        policy2 = get_hierarchical_policy()

        assert policy1 is policy2

    def test_singleton_preserves_state(self) -> None:
        """Test singleton preserves state."""
        import victor.framework.rl.hierarchical_policy as module

        module._hierarchical_policy = None

        policy = get_hierarchical_policy()
        state = HierarchicalState()
        policy.get_decision(state)

        policy2 = get_hierarchical_policy()
        assert policy2._total_decisions == 1
