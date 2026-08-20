# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""TDD tests for live state runtime migration to the canonical API.

Tests that:
1. StateRuntimeAdapter preserves the legacy state-coordination contract shape
2. ConversationController + ConversationStateMachine provide the underlying state
3. Migration path is clear for callers without a concrete StateCoordinator shim
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, PropertyMock

from victor.agent.conversation.state_machine import ConversationStage
from victor.agent.services.state_runtime import StateRuntimeAdapter


class TestStateRuntimeAdapterParity:
    """Verify StateRuntimeAdapter preserves the legacy state runtime contract."""

    @pytest.fixture
    def mock_controller(self):
        """Create mock conversation controller."""
        controller = MagicMock()
        controller.messages = []
        controller.message_count = 0
        type(controller).stage = PropertyMock(return_value=ConversationStage.INITIAL)
        return controller

    @pytest.fixture
    def mock_state_machine(self):
        """Create mock state machine."""
        machine = MagicMock()
        machine.get_stage.return_value = ConversationStage.INITIAL
        return machine

    @pytest.fixture
    def adapter(self, mock_controller, mock_state_machine):
        """Create adapter with mocks."""
        return StateRuntimeAdapter(
            conversation_controller=mock_controller,
            state_machine=mock_state_machine,
        )

    def test_get_current_stage(self, adapter, mock_state_machine):
        """Legacy get_current_stage contract maps to StateRuntimeAdapter."""
        mock_state_machine.get_stage.return_value = ConversationStage.READING
        assert adapter.get_current_stage() == ConversationStage.READING

    def test_get_current_stage_fallback_to_controller(self):
        """Fallback to controller.stage when state_machine is None."""
        mock_controller = MagicMock()
        # Explicitly set _state_machine to None to test fallback
        mock_controller._state_machine = None
        type(mock_controller).stage = PropertyMock(return_value=ConversationStage.EXECUTION)
        adapter = StateRuntimeAdapter(conversation_controller=mock_controller)
        # Verify it accesses controller.stage when no state_machine
        assert adapter.get_current_stage().value == "execution"

    def test_transition_to_success(self, adapter, mock_state_machine):
        """Legacy transition_to contract maps to StateRuntimeAdapter."""
        mock_state_machine.get_stage.return_value = ConversationStage.PLANNING
        result = adapter.transition_to(ConversationStage.PLANNING)
        assert result is True

    def test_transition_to_same_stage(self, adapter, mock_state_machine):
        """Transition to same stage returns True (no-op)."""
        mock_state_machine.get_stage.return_value = ConversationStage.INITIAL
        result = adapter.transition_to(ConversationStage.INITIAL)
        assert result is True

    def test_get_message_history(self, adapter, mock_controller):
        """Legacy get_message_history contract maps to StateRuntimeAdapter."""
        mock_messages = [MagicMock(), MagicMock()]
        mock_controller.messages = mock_messages
        result = adapter.get_message_history()
        assert result == mock_messages

    def test_get_recent_messages(self, adapter, mock_controller):
        """Legacy get_recent_messages contract maps to StateRuntimeAdapter."""
        mock_messages = [
            MagicMock(role="system"),
            MagicMock(role="user"),
            MagicMock(role="assistant"),
            MagicMock(role="user"),
        ]
        mock_controller.messages = mock_messages
        result = adapter.get_recent_messages(limit=2, include_system=False)
        assert len(result) == 2

    def test_get_recent_messages_with_system(self, adapter, mock_controller):
        """get_recent_messages includes system messages when requested."""
        mock_messages = [
            MagicMock(role="system"),
            MagicMock(role="user"),
        ]
        mock_controller.messages = mock_messages
        result = adapter.get_recent_messages(limit=10, include_system=True)
        assert len(result) == 2

    def test_is_in_exploration_phase(self, adapter, mock_state_machine):
        """Legacy exploration-phase contract maps to StateRuntimeAdapter."""
        mock_state_machine.get_stage.return_value = ConversationStage.READING
        assert adapter.is_in_exploration_phase() is True

        mock_state_machine.get_stage.return_value = ConversationStage.EXECUTION
        assert adapter.is_in_exploration_phase() is False

    def test_is_in_execution_phase(self, adapter, mock_state_machine):
        """Legacy execution-phase contract maps to StateRuntimeAdapter."""
        mock_state_machine.get_stage.return_value = ConversationStage.EXECUTION
        assert adapter.is_in_execution_phase() is True

        mock_state_machine.get_stage.return_value = ConversationStage.READING
        assert adapter.is_in_execution_phase() is False


class TestStageTransitionData:
    """Verify ConversationStage enum has all expected stages."""

    def test_exploration_stages(self):
        """All exploration stages are defined."""
        assert ConversationStage.INITIAL in ConversationStage
        assert ConversationStage.PLANNING in ConversationStage
        assert ConversationStage.READING in ConversationStage
        assert ConversationStage.ANALYSIS in ConversationStage

    def test_execution_stage(self):
        """Execution stage is defined."""
        assert ConversationStage.EXECUTION in ConversationStage

    def test_completion_stages(self):
        """All completion stages are defined."""
        assert ConversationStage.VERIFICATION in ConversationStage
        assert ConversationStage.COMPLETION in ConversationStage


class TestMigrationPath:
    """Tests for the actual migration path."""

    def test_state_runtime_adapter_uses_conversation_controller(self):
        """StateRuntimeAdapter wraps ConversationController for state."""
        mock_controller = MagicMock()
        mock_msg = MagicMock(role="user")
        mock_controller.messages = [mock_msg]
        type(mock_controller).stage = PropertyMock(return_value=ConversationStage.READING)

        adapter = StateRuntimeAdapter(conversation_controller=mock_controller)

        # Messages come from controller
        history = adapter.get_message_history()
        assert len(history) == 1
        assert history[0].role == "user"

    def test_state_runtime_adapter_uses_state_machine(self):
        """StateRuntimeAdapter uses ConversationStateMachine for stage."""
        mock_controller = MagicMock()
        mock_machine = MagicMock()
        mock_machine.get_stage.return_value = ConversationStage.EXECUTION

        adapter = StateRuntimeAdapter(
            conversation_controller=mock_controller,
            state_machine=mock_machine,
        )

        # Stage comes from state machine
        assert adapter.get_current_stage() == ConversationStage.EXECUTION

    def test_state_runtime_adapter_auto_discovers_state_machine(self):
        """StateRuntimeAdapter auto-discovers state_machine from controller."""
        mock_controller = MagicMock()
        mock_machine = MagicMock()
        mock_machine.get_stage.return_value = ConversationStage.PLANNING
        mock_controller._state_machine = mock_machine

        adapter = StateRuntimeAdapter(conversation_controller=mock_controller)

        # Should discover and use the state machine
        assert adapter.get_current_stage() == ConversationStage.PLANNING
