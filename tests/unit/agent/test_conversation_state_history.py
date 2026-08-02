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

"""Tests for ConversationStateMachine transition history tracking.

Tests the new history tracking capabilities added to the state machine
for Phase 3 observability integration.
"""

import pytest
import time
from unittest.mock import patch

from victor.agent.conversation.state_machine import (
    ConversationStateMachine,
    ConversationStage,
    ConversationState,
)


class TestTransitionHistoryTracking:
    """Tests for transition history tracking."""

    def test_history_tracking_enabled_by_default(self):
        """Test that history tracking is enabled by default."""
        machine = ConversationStateMachine()
        assert machine._track_history is True

    def test_history_tracking_can_be_disabled(self):
        """Test that history tracking can be disabled."""
        machine = ConversationStateMachine(track_history=False)
        assert machine._track_history is False

    def test_max_history_size_default(self):
        """Test default max history size."""
        machine = ConversationStateMachine()
        assert machine._max_history_size == 100

    def test_max_history_size_custom(self):
        """Test custom max history size."""
        machine = ConversationStateMachine(max_history_size=50)
        assert machine._max_history_size == 50

    def test_empty_history_on_creation(self):
        """Test that history is empty on creation."""
        machine = ConversationStateMachine()
        assert len(machine.transition_history) == 0

    def test_transition_count_starts_at_zero(self):
        """Test that transition count starts at zero."""
        machine = ConversationStateMachine()
        assert machine.transition_count == 0

    @patch("time.time")
    def test_transition_records_history(self, mock_time):
        """Test that transitions are recorded in history."""
        mock_time.return_value = 1000.0
        machine = ConversationStateMachine()

        # Force a transition (bypass cooldown)
        machine._last_transition_time = 0.0
        machine._transition_to(ConversationStage.PLANNING, confidence=0.8)

        history = machine.transition_history
        assert len(history) == 1

        record = history[0]
        assert record["from_stage"] == "INITIAL"
        assert record["to_stage"] == "PLANNING"
        assert record["confidence"] == 0.8
        assert record["timestamp"] == 1000.0
        assert "datetime" in record
        assert record["transition_number"] == 1

    @patch("time.time")
    def test_multiple_transitions(self, mock_time):
        """Test recording multiple transitions."""
        machine = ConversationStateMachine()

        # Simulate transitions with time passing
        mock_time.return_value = 1000.0
        machine._last_transition_time = 0.0
        machine._transition_to(ConversationStage.PLANNING, confidence=0.7)

        mock_time.return_value = 1010.0
        machine._transition_to(ConversationStage.READING, confidence=0.8)

        mock_time.return_value = 1020.0
        machine._transition_to(ConversationStage.EXECUTION, confidence=0.9)

        assert len(machine.transition_history) == 3
        assert machine.transition_count == 3

        # Check order
        assert machine.transition_history[0]["to_stage"] == "PLANNING"
        assert machine.transition_history[1]["to_stage"] == "READING"
        assert machine.transition_history[2]["to_stage"] == "EXECUTION"

    @patch("time.time")
    def test_history_enforces_max_size(self, mock_time):
        """Test that history enforces max size."""
        machine = ConversationStateMachine(max_history_size=3)
        stages = [
            ConversationStage.PLANNING,
            ConversationStage.READING,
            ConversationStage.ANALYSIS,
            ConversationStage.EXECUTION,
            ConversationStage.VERIFICATION,
        ]

        base_time = 1000.0
        for i, stage in enumerate(stages):
            mock_time.return_value = base_time + (i * 10)
            machine._last_transition_time = base_time + (i * 10) - 10
            machine._transition_to(stage, confidence=0.8)

        # Should only keep last 3
        assert len(machine.transition_history) == 3

        # First entries should have been dropped
        assert machine.transition_history[0]["to_stage"] == "ANALYSIS"
        assert machine.transition_history[1]["to_stage"] == "EXECUTION"
        assert machine.transition_history[2]["to_stage"] == "VERIFICATION"

    def test_disabled_history_tracking(self):
        """Test that disabled history tracking doesn't record."""
        machine = ConversationStateMachine(track_history=False)

        # Force a transition
        machine._last_transition_time = 0.0
        machine._transition_to(ConversationStage.PLANNING, confidence=0.8)

        assert len(machine.transition_history) == 0
        # But transition count should still be tracked
        assert machine.transition_count == 1

    def test_reset_clears_history(self):
        """Test that reset clears transition history."""
        machine = ConversationStateMachine()

        # Force a transition
        machine._last_transition_time = 0.0
        machine._transition_to(ConversationStage.PLANNING, confidence=0.8)

        assert len(machine.transition_history) == 1
        assert machine.transition_count == 1

        machine.reset()

        assert len(machine.transition_history) == 0
        assert machine.transition_count == 0

    def test_transition_history_returns_copy(self):
        """Test that transition_history returns a copy."""
        machine = ConversationStateMachine()

        # Force a transition
        machine._last_transition_time = 0.0
        machine._transition_to(ConversationStage.PLANNING, confidence=0.8)

        history1 = machine.transition_history
        history2 = machine.transition_history

        # Should be different objects
        assert history1 is not history2
        # But equal content
        assert history1 == history2


class TestTransitionsSummary:
    """Tests for get_transitions_summary method."""

    def test_empty_summary(self):
        """Test summary with no transitions."""
        machine = ConversationStateMachine()
        summary = machine.get_transitions_summary()

        assert summary["total_transitions"] == 0
        assert summary["unique_paths"] == 0
        assert summary["transitions_by_stage"] == {}
        assert summary["average_confidence"] == 0.0

    @patch("time.time")
    def test_summary_with_transitions(self, mock_time):
        """Test summary with multiple transitions."""
        machine = ConversationStateMachine()

        # Simulate transitions
        mock_time.return_value = 1000.0
        machine._last_transition_time = 0.0
        machine._transition_to(ConversationStage.PLANNING, confidence=0.7)

        mock_time.return_value = 1010.0
        machine._transition_to(ConversationStage.READING, confidence=0.8)

        mock_time.return_value = 1020.0
        machine._transition_to(ConversationStage.PLANNING, confidence=0.9)  # Back to planning

        summary = machine.get_transitions_summary()

        assert summary["total_transitions"] == 3
        assert (
            summary["unique_paths"] >= 2
        )  # INITIAL->PLANNING, PLANNING->READING, READING->PLANNING
        assert "transitions_by_path" in summary
        assert "transitions_by_stage" in summary

        # Average confidence
        expected_avg = (0.7 + 0.8 + 0.9) / 3
        assert abs(summary["average_confidence"] - expected_avg) < 0.01

    @patch("time.time")
    def test_summary_stage_counts(self, mock_time):
        """Test that summary counts stage entries correctly."""
        machine = ConversationStateMachine()

        mock_time.return_value = 1000.0
        machine._last_transition_time = 0.0
        machine._transition_to(ConversationStage.PLANNING, confidence=0.8)

        mock_time.return_value = 1010.0
        machine._transition_to(ConversationStage.READING, confidence=0.8)

        mock_time.return_value = 1020.0
        machine._transition_to(ConversationStage.PLANNING, confidence=0.9)

        summary = machine.get_transitions_summary()

        # PLANNING was entered twice
        assert summary["transitions_by_stage"]["PLANNING"] == 2
        # READING was entered once
        assert summary["transitions_by_stage"]["READING"] == 1


class TestHistoryRecordContents:
    """Tests for the contents of history records."""

    @patch("time.time")
    def test_record_contains_all_fields(self, mock_time):
        """Test that history records contain all expected fields."""
        mock_time.return_value = 1000.0
        machine = ConversationStateMachine()

        machine._last_transition_time = 0.0
        machine._transition_to(ConversationStage.PLANNING, confidence=0.75)

        record = machine.transition_history[0]

        required_fields = [
            "from_stage",
            "to_stage",
            "confidence",
            "timestamp",
            "datetime",
            "transition_number",
            "message_count",
            "tool_count",
        ]

        for field in required_fields:
            assert field in record, f"Missing field: {field}"

    @patch("time.time")
    def test_record_message_count(self, mock_time):
        """Test that record includes message count."""
        mock_time.return_value = 1000.0
        machine = ConversationStateMachine()

        # Add some messages
        machine.state.message_count = 5

        machine._last_transition_time = 0.0
        machine._transition_to(ConversationStage.PLANNING, confidence=0.8)

        record = machine.transition_history[0]
        assert record["message_count"] == 5

    @patch("time.time")
    def test_record_tool_count(self, mock_time):
        """Test that record includes tool count."""
        mock_time.return_value = 1000.0
        machine = ConversationStateMachine()

        # Add some tools to history
        machine.state.tool_history = ["read", "write", "edit"]

        machine._last_transition_time = 0.0
        machine._transition_to(ConversationStage.PLANNING, confidence=0.8)

        record = machine.transition_history[0]
        assert record["tool_count"] == 3

    @patch("time.time")
    def test_record_datetime_format(self, mock_time):
        """Test that datetime is in ISO format."""
        mock_time.return_value = 1000.0
        machine = ConversationStateMachine()

        machine._last_transition_time = 0.0
        machine._transition_to(ConversationStage.PLANNING, confidence=0.8)

        record = machine.transition_history[0]
        datetime_str = record["datetime"]

        # Should be parseable as ISO format
        from datetime import datetime as dt

        parsed = dt.fromisoformat(datetime_str)
        assert parsed is not None


class TestIntegrationWithHooks:
    """Tests for history tracking integration with hooks."""

    @patch("time.time")
    def test_history_recorded_with_hooks(self, mock_time):
        """Test that history is recorded even with hooks."""
        from victor.observability import StateHookManager

        mock_time.return_value = 1000.0
        hooks = StateHookManager()
        hook_calls = []

        @hooks.on_transition
        def track_transition(old, new, ctx):
            hook_calls.append((old, new))

        machine = ConversationStateMachine(hooks=hooks)

        machine._last_transition_time = 0.0
        machine._transition_to(ConversationStage.PLANNING, confidence=0.8)

        # Both hook and history should be called
        assert len(hook_calls) == 1
        assert len(machine.transition_history) == 1
