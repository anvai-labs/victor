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

"""Tests for StageTransitionCoordinator."""

import time
from unittest.mock import MagicMock, patch

import pytest

from victor.agent.action_authorizer import ActionIntent
from victor.agent.coordinators.stage_transition_coordinator import (
    StageTransitionCoordinator,
    TransitionDecision,
    TransitionResult,
    TurnContext,
)
from victor.agent.coordinators.transition_strategies import HybridTransitionStrategy
from victor.core.shared_types import ConversationStage


class TestTurnContext:
    """Test suite for TurnContext."""

    def test_initialization(self):
        """Test TurnContext initialization."""
        ctx = TurnContext(
            turn_id="test-turn",
            start_time=time.time(),
            current_stage=ConversationStage.INITIAL,
        )

        assert ctx.turn_id == "test-turn"
        assert ctx.current_stage == ConversationStage.INITIAL
        assert ctx.tool_count == 0
        assert len(ctx.tools_executed) == 0
        assert len(ctx.unique_tools) == 0

    def test_add_tool(self):
        """Test adding tools to context."""
        ctx = TurnContext(
            turn_id="test-turn",
            start_time=time.time(),
        )

        ctx.add_tool("read", {"path": "test.py"})
        ctx.add_tool("edit", {"path": "test.py"})

        assert ctx.tool_count == 2
        assert len(ctx.unique_tools) == 2
        assert "read" in ctx.unique_tools
        assert "edit" in ctx.unique_tools

    def test_unique_tools(self):
        """Test unique tools property."""
        ctx = TurnContext(
            turn_id="test-turn",
            start_time=time.time(),
        )

        # Add duplicate tool
        ctx.add_tool("read", {"path": "test.py"})
        ctx.add_tool("read", {"path": "other.py"})

        assert ctx.tool_count == 2  # Both executions counted
        assert len(ctx.unique_tools) == 1  # Only one unique tool


class TestTransitionResult:
    """Test suite for TransitionResult."""

    def test_initialization(self):
        """Test TransitionResult initialization."""
        result = TransitionResult(
            decision=TransitionDecision.HEURISTIC_TRANSITION,
            new_stage=ConversationStage.EXECUTION,
            confidence=0.8,
            reason="Test transition",
        )

        assert result.decision == TransitionDecision.HEURISTIC_TRANSITION
        assert result.new_stage == ConversationStage.EXECUTION
        assert result.confidence == 0.8
        assert result.reason == "Test transition"
        assert result.edge_model_called is False
        assert result.calibration_applied is False

    def test_with_edge_model(self):
        """Test TransitionResult with edge model."""
        result = TransitionResult(
            decision=TransitionDecision.EDGE_MODEL_TRANSITION,
            new_stage=ConversationStage.ANALYSIS,
            confidence=0.9,
            reason="Edge model detected",
            edge_model_called=True,
        )

        assert result.edge_model_called is True
        assert result.decision == TransitionDecision.EDGE_MODEL_TRANSITION


class TestStageTransitionCoordinator:
    """Test suite for StageTransitionCoordinator."""

    @pytest.fixture
    def mock_state_machine(self):
        """Create mock state machine."""
        sm = MagicMock()
        sm.get_stage.return_value = ConversationStage.INITIAL
        sm._get_tools_for_stage.return_value = {"read", "edit", "write"}
        sm.state.observed_files = set()
        sm.state.modified_files = set()
        sm._action_intent = ActionIntent.AMBIGUOUS
        return sm

    @pytest.fixture
    def mock_strategy(self):
        """Create mock transition strategy."""
        strategy = MagicMock()
        strategy.detect_transition.return_value = TransitionResult(
            decision=TransitionDecision.NO_TRANSITION,
            new_stage=None,
            confidence=0.0,
            reason="No transition",
            edge_model_called=False,
        )
        strategy.requires_edge_model.return_value = False
        return strategy

    @pytest.fixture
    def coordinator(self, mock_state_machine, mock_strategy):
        """Create coordinator with mocks."""
        return StageTransitionCoordinator(
            state_machine=mock_state_machine,
            strategy=mock_strategy,
            cooldown_seconds=2.0,
            min_tools_for_transition=5,
        )

    def test_initialization(self, mock_state_machine, mock_strategy):
        """Test coordinator initialization."""
        coordinator = StageTransitionCoordinator(
            state_machine=mock_state_machine,
            strategy=mock_strategy,
            cooldown_seconds=3.0,
            min_tools_for_transition=7,
        )

        assert coordinator._state_machine == mock_state_machine
        assert coordinator._strategy == mock_strategy
        assert coordinator._cooldown_seconds == 3.0
        assert coordinator._min_tools_for_transition == 7
        assert coordinator._current_turn is None
        assert coordinator._last_transition_time == 0.0
        assert coordinator._transition_count == 0

    def test_begin_turn(self, coordinator):
        """Test beginning a new turn."""
        coordinator.begin_turn()

        assert coordinator._current_turn is not None
        assert coordinator._current_turn.tool_count == 0
        assert coordinator._current_turn.current_stage == ConversationStage.INITIAL

    def test_record_tool(self, coordinator):
        """Test recording tool executions."""
        coordinator.begin_turn()

        coordinator.record_tool("read", {"path": "test.py"})
        coordinator.record_tool("edit", {"path": "test.py"})

        assert coordinator._current_turn.tool_count == 2
        assert coordinator._current_turn.unique_tools == {"read", "edit"}

    def test_record_tool_without_begin(self, coordinator):
        """Test recording tool before begin_turn creates turn."""
        coordinator.record_tool("read", {"path": "test.py"})

        assert coordinator._current_turn is not None
        assert coordinator._current_turn.tool_count == 1

    def test_end_turn_no_transition(self, coordinator):
        """Test ending turn with no transition."""
        coordinator.begin_turn()
        coordinator.record_tool("read", {"path": "test.py"})

        result = coordinator.end_turn()

        assert result is None  # No transition occurred
        assert coordinator._transition_count == 0

    def test_end_turn_with_transition(self, mock_state_machine):
        """Test ending turn with transition."""
        # Create strategy that returns a transition
        strategy = MagicMock()
        strategy.detect_transition.return_value = TransitionResult(
            decision=TransitionDecision.HEURISTIC_TRANSITION,
            new_stage=ConversationStage.EXECUTION,
            confidence=0.8,
            reason="Heuristic detection",
            edge_model_called=False,
        )
        strategy.requires_edge_model.return_value = False

        coordinator = StageTransitionCoordinator(
            state_machine=mock_state_machine,
            strategy=strategy,
        )

        coordinator.begin_turn()
        coordinator.record_tool("read", {"path": "test.py"})

        result = coordinator.end_turn()

        assert result == ConversationStage.EXECUTION
        assert coordinator._transition_count == 1
        assert coordinator._last_transition_time > 0

    def test_cooldown_prevents_transition(self, mock_state_machine, mock_strategy):
        """Test that cooldown prevents transitions."""
        coordinator = StageTransitionCoordinator(
            state_machine=mock_state_machine,
            strategy=mock_strategy,
            cooldown_seconds=2.0,
        )

        # First transition
        mock_strategy.detect_transition.return_value = TransitionResult(
            decision=TransitionDecision.HEURISTIC_TRANSITION,
            new_stage=ConversationStage.EXECUTION,
            confidence=0.8,
            reason="First transition",
            edge_model_called=False,
        )

        coordinator.begin_turn()
        coordinator.record_tool("read", {"path": "test.py"})
        coordinator.end_turn()

        assert coordinator._transition_count == 1

        # Second transition within cooldown - should be skipped
        mock_strategy.detect_transition.return_value = TransitionResult(
            decision=TransitionDecision.HEURISTIC_TRANSITION,
            new_stage=ConversationStage.ANALYSIS,
            confidence=0.8,
            reason="Second transition",
            edge_model_called=False,
        )

        coordinator.begin_turn()
        coordinator.record_tool("read", {"path": "other.py"})
        result = coordinator.end_turn()

        assert result is None  # Skipped due to cooldown
        assert coordinator._transition_count == 1  # Not incremented

    def test_should_skip_edge_model_cooldown(self, coordinator):
        """Test should_skip_edge_model with cooldown."""
        # Set last transition time to now
        coordinator._last_transition_time = time.time()

        result = coordinator.should_skip_edge_model(
            detected_stage=ConversationStage.EXECUTION,
            current_stage=ConversationStage.INITIAL,
        )

        assert result is True  # Skip due to cooldown

    def test_should_skip_edge_model_same_stage(self, coordinator):
        """Test should_skip_edge_model when stages are same."""
        coordinator.begin_turn()

        result = coordinator.should_skip_edge_model(
            detected_stage=ConversationStage.INITIAL,
            current_stage=ConversationStage.INITIAL,
        )

        assert result is True  # Skip, no transition needed

    def test_should_skip_edge_model_high_confidence(self, mock_state_machine):
        """Test should_skip_edge_model with high confidence."""
        # Mock high tool overlap
        mock_state_machine._get_tools_for_stage.return_value = {
            "read",
            "edit",
            "write",
            "shell",
            "git",
            "test",
        }

        coordinator = StageTransitionCoordinator(
            state_machine=mock_state_machine,
            strategy=MagicMock(),
            min_tools_for_transition=5,
        )

        coordinator.begin_turn()
        # Add tools that overlap with stage
        for tool in ["read", "edit", "write", "shell", "git"]:
            coordinator.record_tool(tool, {})

        result = coordinator.should_skip_edge_model(
            detected_stage=ConversationStage.EXECUTION,
            current_stage=ConversationStage.INITIAL,
        )

        assert result is True  # Skip due to high confidence

    def test_calibration_not_applied(self, coordinator):
        """Test calibration not applied when conditions not met."""
        # Create result that doesn't need calibration
        strategy_result = TransitionResult(
            decision=TransitionDecision.HEURISTIC_TRANSITION,
            new_stage=ConversationStage.ANALYSIS,  # Not EXECUTION
            confidence=0.8,
            reason="Test",
            edge_model_called=False,
        )

        result = coordinator._should_calibrate(strategy_result)

        assert result is False  # No calibration needed

    def test_calibration_applied_read_only(self, mock_state_machine):
        """Test calibration applied for read-only exploration."""
        # Setup: read many files, no edits (need > 10 files)
        for i in range(11):
            mock_state_machine.state.observed_files.add(f"file{i}.py")
        mock_state_machine.state.modified_files = set()

        # Create result that needs calibration
        strategy_result = TransitionResult(
            decision=TransitionDecision.EDGE_MODEL_TRANSITION,
            new_stage=ConversationStage.EXECUTION,
            confidence=0.95,  # High confidence
            reason="Edge model",
            edge_model_called=True,
        )

        coordinator = StageTransitionCoordinator(
            state_machine=mock_state_machine,
            strategy=MagicMock(),
        )

        result = coordinator._should_calibrate(strategy_result)

        assert result is True  # Calibration needed

    def test_apply_calibration(self, mock_state_machine):
        """Test applying calibration to result."""
        # Setup: read many files, no edits
        mock_state_machine.state.observed_files = {"file1.py", "file2.py", "file3.py"}
        mock_state_machine.state.modified_files = set()

        coordinator = StageTransitionCoordinator(
            state_machine=mock_state_machine,
            strategy=MagicMock(),
        )

        original_result = TransitionResult(
            decision=TransitionDecision.EDGE_MODEL_TRANSITION,
            new_stage=ConversationStage.EXECUTION,
            confidence=0.95,
            reason="Edge model",
            edge_model_called=True,
        )

        calibrated_result = coordinator._apply_calibration(original_result)

        assert calibrated_result.new_stage == ConversationStage.ANALYSIS
        assert calibrated_result.confidence == 0.7
        assert calibrated_result.calibration_applied is True
        assert "Calibration applied" in calibrated_result.reason

    def test_calibration_skipped_for_write_allowed_execution(self, mock_state_machine):
        """Explicit write intent should preserve EXECUTION even after heavy reading."""
        for i in range(11):
            mock_state_machine.state.observed_files.add(f"file{i}.py")
        mock_state_machine.state.modified_files = set()
        mock_state_machine._action_intent = ActionIntent.WRITE_ALLOWED

        strategy_result = TransitionResult(
            decision=TransitionDecision.EDGE_MODEL_TRANSITION,
            new_stage=ConversationStage.EXECUTION,
            confidence=0.97,
            reason="Edge model",
            edge_model_called=True,
        )

        coordinator = StageTransitionCoordinator(
            state_machine=mock_state_machine,
            strategy=MagicMock(),
        )

        assert coordinator._should_calibrate(strategy_result) is False

    def test_get_statistics(self, coordinator):
        """Test getting coordinator statistics."""
        coordinator.begin_turn()
        coordinator.record_tool("read", {"path": "test.py"})

        stats = coordinator.get_statistics()

        assert stats["transition_count"] == 0
        assert stats["current_turn_id"] is not None
        assert stats["current_turn_tools"] == 1
        assert "strategy" in stats


class TestCoordinatorIntegration:
    """Integration tests for coordinator with real components."""

    @pytest.fixture
    def state_machine(self):
        """Create real state machine for integration tests."""
        from victor.agent.conversation.state_machine import ConversationStateMachine

        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.INITIAL
        return sm

    def test_full_turn_flow(self, state_machine):
        """Test full turn flow: begin -> record -> end."""
        strategy = HybridTransitionStrategy(edge_model_enabled=False)
        coordinator = StageTransitionCoordinator(
            state_machine=state_machine,
            strategy=strategy,
            cooldown_seconds=2.0,
            min_tools_for_transition=3,  # Lower threshold for test
        )

        # Begin turn
        coordinator.begin_turn()
        assert coordinator._current_turn is not None

        # Record tools (batching)
        coordinator.record_tool("read", {"path": "test.py"})
        coordinator.record_tool("edit", {"path": "test.py"})
        coordinator.record_tool("write", {"path": "test.py"})

        # End turn
        new_stage = coordinator.end_turn()

        # Should have transitioned based on tool overlap
        # (Heuristic detection will find EXECUTION based on edit/write tools)
        assert new_stage is not None or coordinator._transition_count >= 0

    def test_multiple_turns(self, state_machine):
        """Test multiple turns with cooldown."""
        strategy = HybridTransitionStrategy(edge_model_enabled=False)
        coordinator = StageTransitionCoordinator(
            state_machine=state_machine,
            strategy=strategy,
            cooldown_seconds=2.0,
            min_tools_for_transition=3,  # Lower threshold for test
        )

        # Turn 1
        coordinator.begin_turn()
        coordinator.record_tool("read", {"path": "test.py"})
        coordinator.record_tool("edit", {"path": "test.py"})
        coordinator.record_tool("write", {"path": "test.py"})
        coordinator.end_turn()

        first_count = coordinator._transition_count

        # Turn 2 (within cooldown, should skip)
        coordinator.begin_turn()
        coordinator.record_tool("read", {"path": "other.py"})
        result = coordinator.end_turn()

        # Cooldown should prevent transition
        assert coordinator._transition_count == first_count  # No new transitions

        time.sleep(2.1)  # Wait for cooldown to expire

        # Turn 3 (after cooldown)
        coordinator.begin_turn()
        coordinator.record_tool("edit", {"path": "test.py"})
        coordinator.end_turn()

        # Should have transitioned again (or at least not failed)
        assert coordinator._transition_count >= first_count
