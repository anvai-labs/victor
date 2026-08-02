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

"""
Unit tests for Credit Assignment Implementation.

Tests cover:
1. Token-level credit assignment for reasoning chains
2. Segment-level credit assignment
3. Episode-level credit with GAE
4. Turn-level credit with bifurcation detection
5. Hindsight credit assignment
6. Multi-agent Shapley value attribution
7. Critical action identification (CARL)
8. Integration with StateGraph
"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from victor.framework.rl.credit_assignment import (
    ActionMetadata,
    BaseCreditAssigner,
    CreditAssignmentConfig,
    CreditAssignmentIntegration,
    CreditGranularity,
    CreditMethodology,
    CreditSignal,
    CriticalActionIdentifier,
    EpisodeLevelCreditAssigner,
    HindsightCreditAssigner,
    MultiAgentCreditAssigner,
    SegmentLevelCreditAssigner,
    StateGraphCreditMixin,
    TokenLevelCreditAssigner,
    TrajectorySegment,
    TurnLevelCreditAssigner,
    compute_credit_metrics,
    visualize_credit_assignment,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def basic_config():
    """Standard credit assignment configuration."""
    return CreditAssignmentConfig(
        methodology=CreditMethodology.GAE,
        granularity=CreditGranularity.STEP,
        gamma=0.99,
        lambda_gae=0.95,
    )


@pytest.fixture
def sample_trajectory():
    """Sample trajectory for testing."""
    return [
        ActionMetadata(
            agent_id="agent_1",
            action_id=f"action_{i}",
            turn_index=i // 3,
            step_index=i,
            tool_name="test_tool",
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_rewards():
    """Sample reward sequence."""
    return [0.1, 0.2, -0.1, 0.3, 0.5, -0.2, 0.4, 0.6, -0.1, 0.8]


@pytest.fixture
def sample_multi_agent_trajectory():
    """Sample multi-agent trajectory."""
    trajectory = []
    agents = ["agent_A", "agent_B", "agent_C"]
    for i in range(12):
        trajectory.append(
            ActionMetadata(
                agent_id=agents[i % 3],
                action_id=f"action_{i}",
                turn_index=i // 3,
                step_index=i,
                team_id="team_1",
            )
        )
    return trajectory


# ============================================================================
# Data Structure Tests
# ============================================================================


class TestCreditGranularity:
    """Tests for CreditGranularity enum."""

    def test_all_granularities_exist(self):
        """All expected granularity levels should exist."""
        expected = {
            CreditGranularity.TOKEN,
            CreditGranularity.SEGMENT,
            CreditGranularity.STEP,
            CreditGranularity.TURN,
            CreditGranularity.AGENT,
            CreditGranularity.EPISODE,
        }
        assert set(CreditGranularity) == expected

    def test_granularity_values(self):
        """Granularity values should be consistent."""
        assert CreditGranularity.TOKEN.value == "token"
        assert CreditGranularity.SEGMENT.value == "segment"
        assert CreditGranularity.STEP.value == "step"
        assert CreditGranularity.TURN.value == "turn"
        assert CreditGranularity.AGENT.value == "agent"
        assert CreditGranularity.EPISODE.value == "episode"


class TestCreditMethodology:
    """Tests for CreditMethodology enum."""

    def test_all_methodologies_exist(self):
        """All expected methodologies should exist."""
        expected = {
            CreditMethodology.MONTE_CARLO,
            CreditMethodology.TEMPORAL_DIFFERENCE,
            CreditMethodology.GAE,
            CreditMethodology.N_STEP_RETURNS,
            CreditMethodology.ACTOR_CRITIC,
            CreditMethodology.LLM_AS_CRITIC,
            CreditMethodology.C3,
            CreditMethodology.CCPO,
            CreditMethodology.SHAPLEY,
            CreditMethodology.CARL,
            CreditMethodology.HCAPO,
            CreditMethodology.HINDSIGHT,
        }
        assert set(CreditMethodology) == expected


class TestActionMetadata:
    """Tests for ActionMetadata dataclass."""

    def test_create_basic_metadata(self):
        """Should create metadata with required fields."""
        metadata = ActionMetadata(agent_id="agent_1")

        assert metadata.agent_id == "agent_1"
        assert metadata.team_id is None
        assert metadata.turn_index == 0
        assert metadata.step_index == 0
        assert metadata.tool_name is None

    def test_create_full_metadata(self):
        """Should create metadata with all fields."""
        timestamp = datetime.now().timestamp()
        metadata = ActionMetadata(
            agent_id="agent_1",
            team_id="team_1",
            turn_index=5,
            step_index=10,
            tool_name="bash",
            method_name="execute",
            timestamp=timestamp,
            duration_ms=150,
            parent_action_id="parent_1",
            action_id="custom_id",
        )

        assert metadata.agent_id == "agent_1"
        assert metadata.team_id == "team_1"
        assert metadata.turn_index == 5
        assert metadata.step_index == 10
        assert metadata.tool_name == "bash"
        assert metadata.method_name == "execute"
        assert metadata.timestamp == timestamp
        assert metadata.duration_ms == 150
        assert metadata.parent_action_id == "parent_1"
        assert metadata.action_id == "custom_id"

    def test_action_id_auto_generation(self):
        """Should generate action_id if not provided."""
        metadata = ActionMetadata(agent_id="agent_1")

        assert metadata.action_id is not None
        assert metadata.action_id.startswith("action_")


class TestCreditSignal:
    """Tests for CreditSignal dataclass."""

    def test_create_basic_signal(self):
        """Should create signal with required fields."""
        signal = CreditSignal(
            action_id="action_1",
            raw_reward=0.5,
            credit=0.6,
        )

        assert signal.action_id == "action_1"
        assert signal.raw_reward == 0.5
        assert signal.credit == 0.6
        assert signal.confidence == 0.0
        assert signal.methodology is None
        assert signal.granularity == CreditGranularity.STEP

    def test_to_dict(self):
        """Should serialize to dictionary correctly."""
        metadata = ActionMetadata(agent_id="agent_1", tool_name="bash")
        signal = CreditSignal(
            action_id="action_1",
            raw_reward=0.5,
            credit=0.6,
            confidence=0.9,
            methodology=CreditMethodology.GAE,
            granularity=CreditGranularity.TURN,
            metadata=metadata,
            attribution={"agent_1": 0.6},
        )

        result = signal.to_dict()

        assert result["action_id"] == "action_1"
        assert result["raw_reward"] == 0.5
        assert result["credit"] == 0.6
        assert result["confidence"] == 0.9
        assert result["methodology"] == "gae"


class TestTrajectorySegment:
    """Tests for TrajectorySegment dataclass."""

    def test_create_segment(self):
        """Should create segment with required fields."""
        segment = TrajectorySegment(
            segment_id="seg_1",
            action_ids=["action_1", "action_2", "action_3"],
            start_time=100.0,
            end_time=105.0,
            agent_id="agent_1",
        )

        assert segment.segment_id == "seg_1"
        assert segment.action_count == 3
        assert segment.start_time == 100.0
        assert segment.end_time == 105.0

    def test_duration_property(self):
        """Duration should be end_time - start_time."""
        segment = TrajectorySegment(
            segment_id="seg_1",
            action_ids=["action_1"],
            start_time=100.0,
            end_time=105.5,
            agent_id="agent_1",
        )

        assert segment.duration == 5.5


class TestCreditAssignmentConfig:
    """Tests for CreditAssignmentConfig."""

    def test_default_config(self):
        """Should create config with sensible defaults."""
        config = CreditAssignmentConfig()

        assert config.methodology == CreditMethodology.GAE
        assert config.granularity == CreditGranularity.STEP
        assert config.gamma == 0.99
        assert config.lambda_gae == 0.95
        assert config.n_step == 5
        assert config.credit_confidence_threshold == 0.5


# ============================================================================
# Token-Level Credit Assignment Tests
# ============================================================================


class TestTokenLevelCreditAssigner:
    """Tests for TokenLevelCreditAssigner."""

    def test_assign_credit_basic(self):
        """Should assign credit to tokens in reasoning chain."""
        assigner = TokenLevelCreditAssigner()
        tokens = ["token1", "token2", "token3", "token4", "token5"]
        rewards = [0.1, 0.2, 0.0, -0.1, 0.3]

        signals = assigner.assign_credit(tokens, rewards)

        assert len(signals) == len(tokens)
        for signal in signals:
            assert signal.granularity == CreditGranularity.TOKEN
            assert signal.methodology == CreditMethodology.TEMPORAL_DIFFERENCE

    def test_assign_credit_mismatched_lengths(self):
        """Should raise error when trajectory and rewards lengths differ."""
        assigner = TokenLevelCreditAssigner()
        tokens = ["token1", "token2", "token3"]
        rewards = [0.1, 0.2]

        with pytest.raises(ValueError, match="same length"):
            assigner.assign_credit(tokens, rewards)


# ============================================================================
# Segment-Level Credit Assignment Tests
# ============================================================================


class TestSegmentLevelCreditAssigner:
    """Tests for SegmentLevelCreditAssigner."""

    def test_assign_credit_basic(self):
        """Should assign credit to segments using Monte Carlo."""
        assigner = SegmentLevelCreditAssigner()
        segments = [
            TrajectorySegment(
                segment_id=f"seg_{i}",
                action_ids=[f"action_{i}_{j}" for j in range(3)],
                start_time=float(i * 10),
                end_time=float(i * 10 + 5),
                agent_id="agent_1",
            )
            for i in range(5)
        ]
        rewards = [0.1, 0.2, -0.1, 0.3, 0.4]

        signals = assigner.assign_credit(segments, rewards)

        assert len(signals) == len(segments)
        for signal in signals:
            assert signal.granularity == CreditGranularity.SEGMENT
            assert signal.methodology == CreditMethodology.MONTE_CARLO

    def test_attribution_distributes_within_segment(self):
        """Should distribute credit evenly within segment."""
        assigner = SegmentLevelCreditAssigner()
        segment = TrajectorySegment(
            segment_id="seg_1",
            action_ids=["a1", "a2", "a3", "a4"],
            start_time=0.0,
            end_time=10.0,
            agent_id="agent_1",
        )
        rewards = [1.0]

        signals = assigner.assign_credit([segment], rewards)

        assert len(signals) == 1
        signal = signals[0]
        assert len(signal.attribution) == 4


# ============================================================================
# Episode-Level Credit Assignment Tests
# ============================================================================


class TestEpisodeLevelCreditAssigner:
    """Tests for EpisodeLevelCreditAssigner."""

    def test_assign_credit_gae(self):
        """Should assign credit using GAE."""
        config = CreditAssignmentConfig(methodology=CreditMethodology.GAE)
        assigner = EpisodeLevelCreditAssigner(config)

        trajectory = [
            ActionMetadata(agent_id="agent_1", action_id=f"action_{i}") for i in range(10)
        ]
        rewards = [0.1] * 10

        signals = assigner.assign_credit(trajectory, rewards)

        assert len(signals) == len(trajectory)
        for signal in signals:
            assert signal.methodology == CreditMethodology.GAE

    def test_compute_returns(self):
        """Should compute discounted returns correctly."""
        assigner = EpisodeLevelCreditAssigner()
        rewards = [1.0, 1.0, 1.0]

        returns = assigner._compute_returns(rewards, gamma=0.9)

        # With gamma=0.9
        assert abs(returns[0] - 2.71) < 0.01
        assert abs(returns[1] - 1.9) < 0.01
        assert abs(returns[2] - 1.0) < 0.01


# ============================================================================
# Turn-Level Credit Assignment Tests
# ============================================================================


class TestTurnLevelCreditAssigner:
    """Tests for TurnLevelCreditAssigner."""

    def test_group_by_turn(self):
        """Should group actions by turn index."""
        assigner = TurnLevelCreditAssigner()

        trajectory = [
            ActionMetadata(agent_id="agent_1", action_id=f"a_{i}", turn_index=i // 3)
            for i in range(9)
        ]
        rewards = [0.1] * 9

        signals = assigner.assign_credit(trajectory, rewards)

        assert len(signals) == 9
        for signal in signals:
            assert signal.granularity == CreditGranularity.TURN


# ============================================================================
# Hindsight Credit Assignment Tests
# ============================================================================


class TestHindsightCreditAssigner:
    """Tests for HindsightCreditAssigner."""

    def test_successful_trajectory_normal_assignment(self):
        """Should use normal credit assignment for successful trajectories."""
        assigner = HindsightCreditAssigner()

        trajectory = [ActionMetadata(agent_id="agent_1", action_id=f"a_{i}") for i in range(5)]
        rewards = [0.1, 0.2, 0.3, 0.4, 0.5]  # Positive final reward

        signals = assigner.assign_credit(trajectory, rewards)

        assert len(signals) > 0

    def test_failed_trajectory_uses_hindsight(self):
        """Should apply hindsight for failed trajectories."""
        assigner = HindsightCreditAssigner()

        trajectory = [ActionMetadata(agent_id="agent_1", action_id=f"a_{i}") for i in range(5)]
        rewards = [0.1, 0.1, 0.1, 0.1, -1.0]  # Negative final reward

        signals = assigner.assign_credit(trajectory, rewards)

        # All signals should have HINDSIGHT methodology
        for signal in signals:
            assert signal.methodology == CreditMethodology.HINDSIGHT


# ============================================================================
# Multi-Agent Credit Assignment Tests
# ============================================================================


class TestMultiAgentCreditAssigner:
    """Tests for MultiAgentCreditAssigner."""

    def test_shapley_value_attribution(self):
        """Should compute Shapley values for fair attribution."""
        config = CreditAssignmentConfig(
            methodology=CreditMethodology.SHAPLEY,
            shapley_sampling_count=5,
        )
        assigner = MultiAgentCreditAssigner(config)

        trajectory = [
            ActionMetadata(agent_id=f"agent_{i % 3}", action_id=f"a_{i}") for i in range(12)
        ]
        rewards = [0.5] * 12

        signals = assigner.assign_credit(trajectory, rewards)

        assert len(signals) == 12
        for signal in signals:
            assert signal.methodology == CreditMethodology.SHAPLEY
            assert signal.granularity == CreditGranularity.AGENT

    def test_credit_distributes_among_agents(self):
        """Should distribute credit fairly among agents."""
        assigner = MultiAgentCreditAssigner()

        # Agent A contributes more
        trajectory = []
        for i in range(10):
            agent_id = "agent_A" if i < 7 else "agent_B"
            trajectory.append(ActionMetadata(agent_id=agent_id, action_id=f"a_{i}"))
        rewards = [1.0] * 10

        signals = assigner.assign_credit(trajectory, rewards)

        agent_a_signals = [s for s in signals if s.metadata and s.metadata.agent_id == "agent_A"]
        agent_b_signals = [s for s in signals if s.metadata and s.metadata.agent_id == "agent_B"]

        assert len(agent_a_signals) == 7
        assert len(agent_b_signals) == 3


# ============================================================================
# Critical Action Identifier Tests
# ============================================================================


class TestCriticalActionIdentifier:
    """Tests for CriticalActionIdentifier."""

    def test_identify_variance_bifurcations(self):
        """Should identify points with high variance."""
        identifier = CriticalActionIdentifier(threshold=0.2, window_size=2)

        rewards = [0.1, 0.1, 0.5, 0.1, 0.1, 0.1]
        trajectory = ["a" for _ in rewards]

        critical = identifier.identify(trajectory, rewards)

        # Index 2 has high variance
        assert 2 in critical

    def test_empty_trajectory(self):
        """Should handle empty trajectory."""
        identifier = CriticalActionIdentifier()

        critical = identifier.identify([], [])

        assert critical == []


# ============================================================================
# Credit Assignment Integration Tests
# ============================================================================


class TestCreditAssignmentIntegration:
    """Tests for CreditAssignmentIntegration."""

    def test_assign_credit_default_method(self):
        """Should use default methodology when none specified."""
        config = CreditAssignmentConfig(methodology=CreditMethodology.GAE)
        integration = CreditAssignmentIntegration(default_config=config)

        trajectory = [ActionMetadata(agent_id="agent_1", action_id=f"a_{i}") for i in range(5)]
        rewards = [0.1] * 5

        signals = integration.assign_credit(trajectory, rewards)

        assert len(signals) == 5

    def test_get_credit(self):
        """Should retrieve credit for specific action."""
        integration = CreditAssignmentIntegration()

        trajectory = [ActionMetadata(agent_id="agent_1", action_id=f"a_{i}") for i in range(3)]
        rewards = [0.1, 0.2, 0.3]

        signals = integration.assign_credit(trajectory, rewards)
        first_action_id = signals[0].action_id

        retrieved = integration.get_credit(first_action_id)

        assert retrieved is not None
        assert retrieved.action_id == first_action_id

    def test_identify_critical_actions(self):
        """Should identify critical actions in trajectory."""
        integration = CreditAssignmentIntegration()

        trajectory = ["a"] * 10
        rewards = [0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1]

        critical = integration.identify_critical_actions(trajectory, rewards)

        assert len(critical) > 0

    def test_trajectory_summary(self):
        """Should provide trajectory summary."""
        integration = CreditAssignmentIntegration()

        trajectory = [ActionMetadata(agent_id="agent_1", action_id=f"a_{i}") for i in range(5)]
        rewards = [0.1] * 5

        integration.assign_credit(trajectory, rewards)

        summary = integration.get_trajectory_summary()

        assert summary["count"] == 1
        assert summary["total_reward"] == 0.5

    def test_reset(self):
        """Should reset all state."""
        integration = CreditAssignmentIntegration()

        trajectory = [ActionMetadata(agent_id="agent_1", action_id=f"a_{i}") for i in range(3)]
        rewards = [0.1, 0.2, 0.3]

        integration.assign_credit(trajectory, rewards)
        integration.reset()

        summary = integration.get_trajectory_summary()
        assert summary["count"] == 0


# ============================================================================
# StateGraph Integration Tests
# ============================================================================


class TestStateGraphCreditMixin:
    """Tests for StateGraphCreditMixin."""

    def test_mixin_provides_credit_assigner(self):
        """Mixin should provide credit assigner."""

        class MockGraph:
            pass

        class GraphWithCredit(MockGraph, StateGraphCreditMixin):
            pass

        graph = GraphWithCredit()

        assert hasattr(graph, "credit_assigner")
        assert graph.credit_assigner is not None

    def test_assign_transition_credit(self):
        """Should assign credit for state transitions."""

        class MockGraph:
            pass

        class GraphWithCredit(MockGraph, StateGraphCreditMixin):
            pass

        graph = GraphWithCredit()

        transitions = [
            {"from": "state_a", "to": "state_b", "action": "move"},
            {"from": "state_b", "to": "state_c", "action": "move"},
            {"from": "state_c", "to": "state_d", "action": "finish"},
        ]
        rewards = [0.1, 0.2, 0.5]

        signals = graph.assign_transition_credit(transitions, rewards)

        assert len(signals) == 3


# ============================================================================
# Utility Tests
# ============================================================================


class TestUtilities:
    """Tests for utility functions."""

    def test_compute_credit_metrics_empty(self):
        """Should handle empty signals list."""
        metrics = compute_credit_metrics([])

        assert metrics["count"] == 0
        assert metrics["total_credit"] == 0.0

    def test_compute_credit_metrics(self):
        """Should compute metrics from signals."""
        signals = [
            CreditSignal(action_id=f"a_{i}", raw_reward=0.1, credit=float(i)) for i in range(1, 6)
        ]

        metrics = compute_credit_metrics(signals)

        assert metrics["count"] == 5
        assert metrics["total_credit"] == 15.0
        assert metrics["avg_credit"] == 3.0

    def test_visualize_credit_assignment_empty(self):
        """Should handle empty signals."""
        result = visualize_credit_assignment([])

        assert "No credit signals" in result

    def test_visualize_credit_assignment(self):
        """Should generate ASCII visualization."""
        signals = [
            CreditSignal(
                action_id=f"action_{i}",
                raw_reward=0.1,
                credit=float(i) * 0.1,
                confidence=0.5 + i * 0.05,
                methodology=CreditMethodology.GAE,
            )
            for i in range(5)
        ]

        result = visualize_credit_assignment(signals)

        assert "Credit Assignment Visualization" in result
        assert len(result.split("\n")) > 5


# ============================================================================
# Integration Test Scenarios
# ============================================================================


class TestCreditAssignmentScenarios:
    """Integration test scenarios for credit assignment."""

    def test_reasoning_chain_scenario(self):
        """Test credit assignment for long reasoning chain."""
        trajectory = [f"token_{i}" for i in range(100)]
        rewards = [0.01] * 99 + [1.0]  # Success at the end

        assigner = TokenLevelCreditAssigner()
        signals = assigner.assign_credit(trajectory, rewards)

        assert len(signals) == 100

    def test_multi_turn_workflow_scenario(self):
        """Test credit assignment for multi-turn workflow."""
        trajectory = []
        for turn in range(5):
            for step in range(3):
                trajectory.append(
                    ActionMetadata(
                        agent_id="agent_1",
                        action_id=f"turn{turn}_step{step}",
                        turn_index=turn,
                        step_index=step,
                    )
                )

        rewards = []
        for turn in range(5):
            turn_reward = 0.5 if turn < 4 else 1.0
            rewards.extend([turn_reward / 3] * 3)

        assigner = TurnLevelCreditAssigner()
        signals = assigner.assign_credit(trajectory, rewards)

        assert len(signals) == 15

    def test_multi_agent_team_scenario(self):
        """Test credit assignment for multi-agent team."""
        trajectory = []
        for i in range(20):
            agent = f"agent_{i % 3}"
            trajectory.append(
                ActionMetadata(agent_id=agent, action_id=f"action_{i}", team_id="team_1")
            )

        rewards = [1.0] * 20

        assigner = MultiAgentCreditAssigner()
        signals = assigner.assign_credit(trajectory, rewards)

        agent_credits = {}
        for signal in signals:
            if signal.metadata:
                agent = signal.metadata.agent_id
                agent_credits[agent] = agent_credits.get(agent, 0.0) + signal.credit

        assert len(agent_credits) == 3

    def test_failed_trajectory_with_hindsight(self):
        """Test hindsight credit assignment for failed trajectory."""
        trajectory = [
            ActionMetadata(agent_id="agent_1", action_id=f"attempt_{i}", turn_index=i)
            for i in range(5)
        ]
        rewards = [0.1, 0.1, -0.5, -0.5, -1.0]

        assigner = HindsightCreditAssigner()
        signals = assigner.assign_credit(trajectory, rewards)

        assert len(signals) > 0
        for signal in signals:
            assert signal.methodology == CreditMethodology.HINDSIGHT
