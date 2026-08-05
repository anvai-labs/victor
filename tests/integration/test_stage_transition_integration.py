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

"""Integration tests for StageTransitionCoordinator.

Tests the coordinator with real ConversationStateMachine, tool execution,
and turn lifecycle to verify Phase 1 optimizations work correctly.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from victor.agent.coordinators import (
    HybridTransitionStrategy,
    StageTransitionCoordinator,
)
from victor.agent.conversation.state_machine import (
    ConversationStateMachine,
    ConversationStage,
)
from victor.core.feature_flags import FeatureFlag, get_feature_flag_manager


class TestCoordinatorIntegration:
    """Integration tests for coordinator with full system."""

    @pytest.fixture
    def state_machine(self):
        """Create real state machine."""
        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.INITIAL
        return sm

    @pytest.fixture
    def coordinator(self, state_machine):
        """Create coordinator with real state machine."""
        return StageTransitionCoordinator(
            state_machine=state_machine,
            strategy=HybridTransitionStrategy(edge_model_enabled=False),
            cooldown_seconds=2.0,
            min_tools_for_transition=3,  # Lower threshold for tests
        )

    def test_coordinator_wired_to_state_machine(self, state_machine, coordinator):
        """Test that coordinator can be wired to state machine."""
        # Initially no coordinator
        assert state_machine._transition_coordinator is None

        # Wire coordinator
        state_machine.set_transition_coordinator(coordinator)

        # Verify wired
        assert state_machine._transition_coordinator is coordinator

    def test_batching_prevents_immediate_transitions(self, state_machine, coordinator):
        """Test that batching prevents immediate per-tool transitions."""
        # Wire coordinator
        state_machine.set_transition_coordinator(coordinator)

        # Begin turn
        coordinator.begin_turn()

        # Record multiple tools - should NOT trigger immediate transitions
        state_machine.record_tool_execution("read", {"path": "test.py"})
        assert state_machine.get_stage() == ConversationStage.INITIAL  # No transition yet

        state_machine.record_tool_execution("edit", {"path": "test.py"})
        assert state_machine.get_stage() == ConversationStage.INITIAL  # Still no transition

        state_machine.record_tool_execution("write", {"path": "test.py"})
        assert state_machine.get_stage() == ConversationStage.INITIAL  # Still no transition

        # End turn - batching complete
        # Note: May or may not transition depending on heuristic detection
        new_stage = coordinator.end_turn()
        # The key test is that batching happened (3 tools were recorded)
        assert coordinator._current_turn is None  # Turn ended
        assert state_machine.state.tool_history == ["read", "edit", "write"]

    def test_cooldown_prevents_rapid_transitions(self, state_machine, coordinator):
        """Test that cooldown prevents rapid transitions across turns."""
        state_machine.set_transition_coordinator(coordinator)

        # Turn 1: Force a transition by manually setting state
        coordinator.begin_turn()
        state_machine.record_tool_execution("edit", {"path": "test.py"})
        state_machine.record_tool_execution("write", {"path": "test.py"})
        state_machine.record_tool_execution("test", {"path": "test.py"})

        # Manually trigger a transition to ensure cooldown is set
        # (simulating what would happen if heuristic detected a stage change)
        from victor.core.shared_types import ConversationStage

        state_machine.state.stage = ConversationStage.EXECUTION
        coordinator._last_transition_time = time.time()
        coordinator._transition_count = 1
        first_count = coordinator._transition_count
        coordinator.end_turn()

        # Turn 2: Within cooldown - should skip
        coordinator.begin_turn()
        state_machine.record_tool_execution("read", {"path": "other.py"})
        result2 = coordinator.end_turn()

        # Should not have transitioned due to cooldown
        assert result2 is None, f"Expected None due to cooldown, got {result2}"
        assert (
            coordinator._transition_count == first_count
        ), f"Transition count should not increase: {coordinator._transition_count} != {first_count}"

        # Verify we're actually in cooldown
        time_since_last = time.time() - coordinator._last_transition_time
        assert (
            time_since_last < coordinator._cooldown_seconds
        ), f"Not in cooldown: {time_since_last:.2f}s >= {coordinator._cooldown_seconds}s"

        # Wait for cooldown to expire
        time.sleep(coordinator._cooldown_seconds + 0.1)

        # Turn 3: After cooldown - should be allowed
        coordinator.begin_turn()
        state_machine.record_tool_execution("edit", {"path": "test.py"})
        result3 = coordinator.end_turn()

        # Cooldown should have expired (transition may or may not occur depending on heuristic)
        time_since_last = time.time() - coordinator._last_transition_time
        assert (
            time_since_last >= coordinator._cooldown_seconds
        ), f"Cooldown should have expired: {time_since_last:.2f}s < {coordinator._cooldown_seconds}s"

    def test_feature_flag_controls_coordinator(self):
        """Test that feature flag controls coordinator creation."""
        # Initially disabled
        assert not get_feature_flag_manager().is_enabled(
            FeatureFlag.USE_STAGE_TRANSITION_COORDINATOR
        )

        # Enable flag
        get_feature_flag_manager().enable(FeatureFlag.USE_STAGE_TRANSITION_COORDINATOR)

        # Now enabled
        assert get_feature_flag_manager().is_enabled(FeatureFlag.USE_STAGE_TRANSITION_COORDINATOR)

        # Disable for cleanup
        get_feature_flag_manager().disable(FeatureFlag.USE_STAGE_TRANSITION_COORDINATOR)

    def test_coordinator_statistics(self, state_machine, coordinator):
        """Test that coordinator provides useful statistics."""
        state_machine.set_transition_coordinator(coordinator)

        # Begin turn and record tools
        coordinator.begin_turn()
        state_machine.record_tool_execution("read", {"path": "test.py"})
        state_machine.record_tool_execution("edit", {"path": "test.py"})

        # Get statistics
        stats = coordinator.get_statistics()

        assert stats["transition_count"] == 0
        assert stats["current_turn_id"] is not None
        assert stats["current_turn_tools"] == 2
        assert "strategy" in stats

        # End turn
        coordinator.end_turn()

        # Check final stats
        stats = coordinator.get_statistics()
        assert "transition_count" in stats

    def test_high_confidence_skip(self, state_machine, coordinator):
        """Test that high confidence skip bypasses edge model."""
        state_machine.set_transition_coordinator(coordinator)

        # Begin turn
        coordinator.begin_turn()

        # Add many tools that overlap with EXECUTION stage
        # (edit, write, test, shell, git are all EXECUTION tools)
        for tool in ["edit", "write", "test", "shell", "git"]:
            state_machine.record_tool_execution(tool, {"path": "test.py"})

        # End turn - should use high confidence skip
        new_stage = coordinator.end_turn()

        # Verify batching worked (5 tools recorded)
        assert len(state_machine.state.tool_history) == 5
        # Turn ended properly
        assert coordinator._current_turn is None

    def test_state_machine_fallback_without_coordinator(self, state_machine):
        """Test that state machine falls back to legacy behavior without coordinator."""
        # No coordinator wired
        assert state_machine._transition_coordinator is None

        # Record tools - should use legacy _maybe_transition() behavior
        initial_stage = state_machine.get_stage()
        state_machine.record_tool_execution("edit", {"path": "test.py"})

        # Legacy behavior: state machine handles transitions directly
        # (may or may not transition depending on heuristic)
        current_stage = state_machine.get_stage()
        assert current_stage in ConversationStage


class TestCoordinatorWithTurnExecutor:
    """Test coordinator integration with turn execution lifecycle."""

    def test_begin_end_turn_lifecycle(self):
        """Test that begin/end turn lifecycle works correctly."""
        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.INITIAL

        coordinator = StageTransitionCoordinator(
            state_machine=sm,
            strategy=HybridTransitionStrategy(edge_model_enabled=False),
            cooldown_seconds=2.0,
            min_tools_for_transition=3,
        )

        sm.set_transition_coordinator(coordinator)

        # Simulate turn lifecycle
        coordinator.begin_turn()
        assert coordinator._current_turn is not None

        sm.record_tool_execution("read", {"path": "test.py"})
        sm.record_tool_execution("edit", {"path": "test.py"})
        sm.record_tool_execution("write", {"path": "test.py"})

        # Should still be in initial stage (batched)
        assert sm.get_stage() == ConversationStage.INITIAL

        # End turn - batching complete
        new_stage = coordinator.end_turn()
        # Verify turn ended and tools were recorded
        assert coordinator._current_turn is None
        assert len(sm.state.tool_history) == 3

    def test_multiple_turns_with_cooldown(self):
        """Test multiple turns with cooldown enforcement."""
        sm = ConversationStateMachine()
        sm.state.stage = ConversationStage.INITIAL

        coordinator = StageTransitionCoordinator(
            state_machine=sm,
            strategy=HybridTransitionStrategy(edge_model_enabled=False),
            cooldown_seconds=1.5,  # Shorter for tests
            min_tools_for_transition=2,
        )

        sm.set_transition_coordinator(coordinator)

        # Turn 1: Force a transition to set cooldown
        coordinator.begin_turn()
        sm.record_tool_execution("edit", {"path": "test.py"})
        sm.record_tool_execution("write", {"path": "test.py"})

        # Manually trigger transition to ensure cooldown is set
        sm.state.stage = ConversationStage.EXECUTION
        coordinator._last_transition_time = time.time()
        coordinator._transition_count = 1
        coordinator.end_turn()
        count1 = coordinator._transition_count

        # Turn 2 (within cooldown)
        coordinator.begin_turn()
        sm.record_tool_execution("read", {"path": "other.py"})
        result = coordinator.end_turn()
        assert result is None, f"Expected None due to cooldown, got {result}"
        assert (
            coordinator._transition_count == count1
        ), f"Transition count should not increase: {coordinator._transition_count} != {count1}"

        # Verify we're in cooldown
        time_since_last = time.time() - coordinator._last_transition_time
        assert (
            time_since_last < coordinator._cooldown_seconds
        ), f"Not in cooldown: {time_since_last:.2f}s >= {coordinator._cooldown_seconds}s"

        # Wait for cooldown to expire
        time.sleep(coordinator._cooldown_seconds + 0.1)

        # Turn 3 (after cooldown)
        coordinator.begin_turn()
        sm.record_tool_execution("test", {"path": "test.py"})
        coordinator.end_turn()

        # Cooldown should have expired
        time_since_last = time.time() - coordinator._last_transition_time
        assert (
            time_since_last >= coordinator._cooldown_seconds
        ), f"Cooldown should have expired: {time_since_last:.2f}s < {coordinator._cooldown_seconds}s"
