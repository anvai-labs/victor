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

"""Tests for transition strategies."""

from unittest.mock import MagicMock

import pytest

from victor.agent.coordinators.transition_strategies import (
    EdgeModelTransitionStrategy,
    HeuristicOnlyTransitionStrategy,
    HybridTransitionStrategy,
    TransitionStrategyProtocol,
    create_transition_strategy,
)
from victor.core.shared_types import ConversationStage


class TestHeuristicOnlyTransitionStrategy:
    """Test suite for HeuristicOnlyTransitionStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return HeuristicOnlyTransitionStrategy()

    @pytest.fixture
    def mock_state_machine(self):
        """Create mock state machine."""
        sm = MagicMock()
        sm.get_stage.return_value = ConversationStage.INITIAL
        sm._get_tools_for_stage.return_value = {"read", "edit", "write"}
        return sm

    @pytest.fixture
    def mock_coordinator(self):
        """Create mock coordinator."""
        coordinator = MagicMock()
        coordinator._min_tools_for_transition = 5
        return coordinator

    def test_detect_transition_no_detection(self, strategy, mock_state_machine, mock_coordinator):
        """Test detection when heuristic returns None."""
        mock_state_machine._detect_stage_from_tools.return_value = None

        result = strategy.detect_transition(
            current_stage=ConversationStage.INITIAL,
            tools_executed=[("read", {"path": "test.py"})],
            state_machine=mock_state_machine,
            coordinator=mock_coordinator,
        )

        assert result.decision == "no_transition"
        assert result.new_stage is None
        assert result.confidence == 0.0
        assert result.edge_model_called is False

    def test_detect_transition_with_detection(self, strategy, mock_state_machine, mock_coordinator):
        """Test detection when heuristic detects a stage."""
        mock_state_machine._detect_stage_from_tools.return_value = ConversationStage.EXECUTION
        mock_state_machine._get_tools_for_stage.return_value = {"read", "edit", "write"}

        result = strategy.detect_transition(
            current_stage=ConversationStage.INITIAL,
            tools_executed=[
                ("read", {"path": "test.py"}),
                ("edit", {"path": "test.py"}),
            ],
            state_machine=mock_state_machine,
            coordinator=mock_coordinator,
        )

        assert result.decision == "heuristic"
        assert result.new_stage == ConversationStage.EXECUTION
        assert result.confidence > 0.6
        assert result.edge_model_called is False
        assert "overlap" in result.reason

    def test_requires_edge_model(self, strategy):
        """Test that heuristic strategy doesn't require edge model."""
        assert strategy.requires_edge_model() is False


class TestEdgeModelTransitionStrategy:
    """Test suite for EdgeModelTransitionStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return EdgeModelTransitionStrategy(edge_model_enabled=True)

    @pytest.fixture
    def mock_state_machine(self):
        """Create mock state machine."""
        sm = MagicMock()
        sm.get_stage.return_value = ConversationStage.INITIAL
        sm._get_tools_for_stage.return_value = {"read", "edit", "write"}
        return sm

    @pytest.fixture
    def mock_coordinator(self):
        """Create mock coordinator."""
        coordinator = MagicMock()
        coordinator._min_tools_for_transition = 5
        return coordinator

    def test_detect_transition_with_edge_model(
        self, strategy, mock_state_machine, mock_coordinator
    ):
        """Test detection using edge model."""
        mock_state_machine._try_edge_model_transition.return_value = (
            ConversationStage.EXECUTION,
            0.85,
        )

        result = strategy.detect_transition(
            current_stage=ConversationStage.INITIAL,
            tools_executed=[("read", {"path": "test.py"})],
            state_machine=mock_state_machine,
            coordinator=mock_coordinator,
        )

        assert result.decision == "edge_model"
        assert result.new_stage == ConversationStage.EXECUTION
        assert result.confidence == 0.85
        assert result.edge_model_called is True

    def test_detect_transition_edge_model_disabled(self, mock_state_machine, mock_coordinator):
        """Test detection when edge model is disabled."""
        strategy = EdgeModelTransitionStrategy(edge_model_enabled=False)

        # Should fall back to heuristic
        mock_state_machine._detect_stage_from_tools.return_value = ConversationStage.EXECUTION
        mock_state_machine._get_tools_for_stage.return_value = {"read", "edit", "write"}

        result = strategy.detect_transition(
            current_stage=ConversationStage.INITIAL,
            tools_executed=[("read", {"path": "test.py"})],
            state_machine=mock_state_machine,
            coordinator=mock_coordinator,
        )

        assert result.decision == "heuristic"
        assert result.edge_model_called is False

    def test_requires_edge_model(self, strategy):
        """Test that edge model strategy requires edge model."""
        assert strategy.requires_edge_model() is True


class TestHybridTransitionStrategy:
    """Test suite for HybridTransitionStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return HybridTransitionStrategy(edge_model_enabled=True)

    @pytest.fixture
    def mock_state_machine(self):
        """Create mock state machine."""
        sm = MagicMock()
        sm.get_stage.return_value = ConversationStage.INITIAL
        sm._get_tools_for_stage.return_value = {
            "read",
            "edit",
            "write",
            "shell",
            "git",
            "test",
        }
        sm.state.observed_files = set()
        sm.state.modified_files = set()
        return sm

    @pytest.fixture
    def mock_coordinator(self):
        """Create mock coordinator."""
        coordinator = MagicMock()
        coordinator._min_tools_for_transition = 5
        coordinator.should_skip_edge_model.return_value = False
        return coordinator

    def test_high_confidence_skip(self, strategy, mock_state_machine, mock_coordinator):
        """Test high confidence skip (tool overlap >= threshold)."""
        mock_state_machine._detect_stage_from_tools.return_value = ConversationStage.EXECUTION
        mock_state_machine._get_tools_for_stage.return_value = {
            "read",
            "edit",
            "write",
            "shell",
            "git",
            "test",
        }

        # 6 unique tools, all in EXECUTION stage
        tools_executed = [
            ("read", {"path": "test.py"}),
            ("edit", {"path": "test.py"}),
            ("write", {"path": "test.py"}),
            ("shell", {"cmd": "ls"}),
            ("git", {"args": "status"}),
            ("test", {"path": "test.py"}),
        ]

        result = strategy.detect_transition(
            current_stage=ConversationStage.INITIAL,
            tools_executed=tools_executed,
            state_machine=mock_state_machine,
            coordinator=mock_coordinator,
        )

        assert result.decision == "high_confidence_skip"
        assert result.new_stage == ConversationStage.EXECUTION
        assert result.edge_model_called is False
        assert "overlap=6" in result.reason

    def test_edge_model_fallback(self, strategy, mock_state_machine, mock_coordinator):
        """Test edge model fallback when confidence is low."""
        mock_state_machine._detect_stage_from_tools.return_value = ConversationStage.ANALYSIS
        mock_state_machine._get_tools_for_stage.return_value = {"read", "edit", "write"}

        # Only 1 unique tool (low overlap)
        tools_executed = [("read", {"path": "test.py"})]

        mock_coordinator.should_skip_edge_model.return_value = False
        mock_state_machine._try_edge_model_transition.return_value = (
            ConversationStage.ANALYSIS,
            0.85,
        )

        result = strategy.detect_transition(
            current_stage=ConversationStage.INITIAL,
            tools_executed=tools_executed,
            state_machine=mock_state_machine,
            coordinator=mock_coordinator,
        )

        assert result.decision == "edge_model"
        assert result.new_stage == ConversationStage.ANALYSIS
        assert result.confidence == 0.85
        assert result.edge_model_called is True

    def test_cooldown_skip(self, strategy, mock_state_machine, mock_coordinator):
        """Test skip when coordinator says to skip edge model."""
        mock_state_machine._detect_stage_from_tools.return_value = ConversationStage.EXECUTION
        mock_state_machine._get_tools_for_stage.return_value = {"read", "edit", "write"}

        # Low overlap, but coordinator says skip
        mock_coordinator.should_skip_edge_model.return_value = True

        tools_executed = [("read", {"path": "test.py"})]

        result = strategy.detect_transition(
            current_stage=ConversationStage.INITIAL,
            tools_executed=tools_executed,
            state_machine=mock_state_machine,
            coordinator=mock_coordinator,
        )

        assert result.decision == "cooldown_skip"
        assert result.edge_model_called is False
        assert "skipped" in result.reason

    def test_heuristic_fallback(self, mock_state_machine, mock_coordinator):
        """Test heuristic fallback when edge model unavailable."""
        strategy = HybridTransitionStrategy(edge_model_enabled=False)

        mock_state_machine._detect_stage_from_tools.return_value = ConversationStage.EXECUTION
        mock_state_machine._get_tools_for_stage.return_value = {"read", "edit", "write"}

        tools_executed = [("read", {"path": "test.py"})]

        result = strategy.detect_transition(
            current_stage=ConversationStage.INITIAL,
            tools_executed=tools_executed,
            state_machine=mock_state_machine,
            coordinator=mock_coordinator,
        )

        assert result.decision == "heuristic"
        assert result.edge_model_called is False

    def test_requires_edge_model(self, strategy):
        """Test that hybrid strategy requires edge model."""
        assert strategy.requires_edge_model() is True


class TestCreateTransitionStrategy:
    """Test suite for create_transition_strategy factory."""

    def test_create_heuristic_strategy(self):
        """Test creating heuristic strategy."""
        strategy = create_transition_strategy("heuristic")

        assert isinstance(strategy, HeuristicOnlyTransitionStrategy)
        assert strategy.requires_edge_model() is False

    def test_create_edge_model_strategy(self):
        """Test creating edge model strategy."""
        strategy = create_transition_strategy("edge_model", edge_model_enabled=True)

        assert isinstance(strategy, EdgeModelTransitionStrategy)
        assert strategy.requires_edge_model() is True

    def test_create_hybrid_strategy(self):
        """Test creating hybrid strategy."""
        strategy = create_transition_strategy("hybrid", edge_model_enabled=True)

        assert isinstance(strategy, HybridTransitionStrategy)
        assert strategy.requires_edge_model() is True

    def test_invalid_strategy_type(self):
        """Test that invalid strategy type raises error."""
        with pytest.raises(ValueError, match="Unknown strategy type"):
            create_transition_strategy("invalid_type")


class TestStrategyProtocol:
    """Test suite for strategy protocol compliance."""

    def test_heuristic_strategy_protocol_compliance(self):
        """Test that HeuristicOnlyTransitionStrategy implements protocol."""
        strategy = HeuristicOnlyTransitionStrategy()

        # Check required methods exist and are callable
        assert hasattr(strategy, "detect_transition")
        assert hasattr(strategy, "requires_edge_model")
        assert callable(strategy.detect_transition)
        assert callable(strategy.requires_edge_model)
        # Check it has the right signature
        import inspect

        sig = inspect.signature(strategy.detect_transition)
        assert len(sig.parameters) == 4  # current_stage, tools_executed, state_machine, coordinator

    def test_edge_model_strategy_protocol_compliance(self):
        """Test that EdgeModelTransitionStrategy implements protocol."""
        strategy = EdgeModelTransitionStrategy()

        assert hasattr(strategy, "detect_transition")
        assert hasattr(strategy, "requires_edge_model")
        assert callable(strategy.detect_transition)
        assert callable(strategy.requires_edge_model)

    def test_hybrid_strategy_protocol_compliance(self):
        """Test that HybridTransitionStrategy implements protocol."""
        strategy = HybridTransitionStrategy()

        assert hasattr(strategy, "detect_transition")
        assert hasattr(strategy, "requires_edge_model")
        assert callable(strategy.detect_transition)
        assert callable(strategy.requires_edge_model)
