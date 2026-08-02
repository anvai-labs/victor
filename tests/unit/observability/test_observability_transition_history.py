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

"""Tests for the TransitionHistory and history-aware hooks.

Tests the Observer pattern implementation with history tracking for state
machine transitions, including analytics and query capabilities.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from victor.observability import (
    LoggingHook,
    MetricsHook,
    StateHookManager,
    StateTransitionHook,
    TransitionHistory,
    TransitionRecord,
)


class TestTransitionRecord:
    """Tests for TransitionRecord dataclass."""

    def test_create_record(self):
        """Test creating a transition record."""
        record = TransitionRecord(
            old_stage="INITIAL",
            new_stage="PLANNING",
            timestamp=datetime.now(timezone.utc),
            context={"tool": "read"},
        )

        assert record.old_stage == "INITIAL"
        assert record.new_stage == "PLANNING"
        assert record.context == {"tool": "read"}
        assert record.duration_ms is None
        assert record.sequence_number == 0

    def test_record_immutability(self):
        """Test that records are immutable (frozen dataclass)."""
        record = TransitionRecord(
            old_stage="INITIAL",
            new_stage="PLANNING",
            timestamp=datetime.now(timezone.utc),
            context={},
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            record.old_stage = "MODIFIED"

    def test_record_string_representation(self):
        """Test string representation includes key info."""
        record = TransitionRecord(
            old_stage="INITIAL",
            new_stage="PLANNING",
            timestamp=datetime.now(timezone.utc),
            context={},
            duration_ms=150.5,
        )

        str_repr = str(record)
        assert "INITIAL" in str_repr
        assert "PLANNING" in str_repr
        assert "150.5ms" in str_repr

    def test_record_with_duration(self):
        """Test record with duration tracking."""
        record = TransitionRecord(
            old_stage="READING",
            new_stage="EXECUTION",
            timestamp=datetime.now(timezone.utc),
            context={},
            duration_ms=250.0,
            sequence_number=5,
        )

        assert record.duration_ms == 250.0
        assert record.sequence_number == 5


class TestTransitionHistory:
    """Tests for TransitionHistory analytics."""

    def test_create_empty_history(self):
        """Test creating empty history."""
        history = TransitionHistory()

        assert len(history) == 0
        assert history.current_stage is None
        assert history.previous_stage is None
        assert not bool(history)

    def test_create_record_via_history(self):
        """Test creating records through history."""
        history = TransitionHistory()

        record = history.create_record("INITIAL", "PLANNING", {"tool": "read"})

        assert record.old_stage == "INITIAL"
        assert record.new_stage == "PLANNING"
        assert record.sequence_number == 1
        assert len(history) == 1

    def test_current_and_previous_stage(self):
        """Test tracking current and previous stages."""
        history = TransitionHistory()

        history.create_record("INITIAL", "PLANNING", {})
        assert history.current_stage == "PLANNING"
        assert history.previous_stage == "INITIAL"

        history.create_record("PLANNING", "EXECUTION", {})
        assert history.current_stage == "EXECUTION"
        assert history.previous_stage == "PLANNING"

    def test_duration_calculation(self):
        """Test that duration is calculated between transitions."""
        history = TransitionHistory()

        # First transition - no duration (no prior enter time)
        history.create_record("INITIAL", "PLANNING", {})

        # Small delay to get measurable duration
        time.sleep(0.01)

        # Second transition - should have duration
        record2 = history.create_record("PLANNING", "EXECUTION", {})

        # Duration should be tracked for PLANNING stage
        assert record2.duration_ms is not None
        assert record2.duration_ms > 0

    def test_get_last_n_transitions(self):
        """Test retrieving last N transitions."""
        history = TransitionHistory()

        history.create_record("INITIAL", "PLANNING", {})
        history.create_record("PLANNING", "READING", {})
        history.create_record("READING", "EXECUTION", {})

        last_2 = history.get_last(2)
        assert len(last_2) == 2
        assert last_2[0].new_stage == "READING"
        assert last_2[1].new_stage == "EXECUTION"

    def test_get_transitions_to_stage(self):
        """Test finding all transitions into a specific stage."""
        history = TransitionHistory()

        history.create_record("INITIAL", "PLANNING", {})
        history.create_record("PLANNING", "EXECUTION", {})
        history.create_record("EXECUTION", "PLANNING", {})  # Re-enter PLANNING
        history.create_record("PLANNING", "COMPLETION", {})

        to_planning = history.get_transitions_to("PLANNING")
        assert len(to_planning) == 2

    def test_get_transitions_from_stage(self):
        """Test finding all transitions from a specific stage."""
        history = TransitionHistory()

        history.create_record("PLANNING", "READING", {})
        history.create_record("READING", "EXECUTION", {})
        history.create_record("PLANNING", "EXECUTION", {})

        from_planning = history.get_transitions_from("PLANNING")
        assert len(from_planning) == 2

    def test_get_transitions_between_stages(self):
        """Test finding transitions between specific stages."""
        history = TransitionHistory()

        history.create_record("PLANNING", "EXECUTION", {})
        history.create_record("EXECUTION", "VERIFICATION", {})
        history.create_record("VERIFICATION", "EXECUTION", {})
        history.create_record("EXECUTION", "VERIFICATION", {})

        between = history.get_transitions_between("EXECUTION", "VERIFICATION")
        assert len(between) == 2

    def test_stage_visit_count(self):
        """Test counting stage visits."""
        history = TransitionHistory()

        history.create_record("INITIAL", "PLANNING", {})
        history.create_record("PLANNING", "EXECUTION", {})
        history.create_record("EXECUTION", "PLANNING", {})
        history.create_record("PLANNING", "COMPLETION", {})

        assert history.get_stage_visit_count("PLANNING") == 2
        assert history.get_stage_visit_count("EXECUTION") == 1
        assert history.get_stage_visit_count("UNKNOWN") == 0

    def test_has_visited(self):
        """Test checking if a stage was visited."""
        history = TransitionHistory()

        history.create_record("INITIAL", "PLANNING", {})

        assert history.has_visited("PLANNING")
        assert not history.has_visited("EXECUTION")

    def test_has_cycle_detection(self):
        """Test cycle detection."""
        history = TransitionHistory()

        history.create_record("INITIAL", "PLANNING", {})
        history.create_record("PLANNING", "EXECUTION", {})
        assert not history.has_cycle()

        history.create_record("EXECUTION", "PLANNING", {})  # Cycle!
        assert history.has_cycle()

    def test_get_stage_sequence(self):
        """Test getting full stage sequence."""
        history = TransitionHistory()

        history.create_record("INITIAL", "PLANNING", {})
        history.create_record("PLANNING", "EXECUTION", {})
        history.create_record("EXECUTION", "COMPLETION", {})

        sequence = history.get_stage_sequence()
        assert sequence == ["INITIAL", "PLANNING", "EXECUTION", "COMPLETION"]

    def test_get_transition_pattern(self):
        """Test getting transition pattern as tuples."""
        history = TransitionHistory()

        history.create_record("A", "B", {})
        history.create_record("B", "C", {})

        pattern = history.get_transition_pattern()
        assert pattern == [("A", "B"), ("B", "C")]

    def test_max_size_circular_buffer(self):
        """Test that history respects max_size."""
        history = TransitionHistory(max_size=3)

        history.create_record("A", "B", {})
        history.create_record("B", "C", {})
        history.create_record("C", "D", {})
        history.create_record("D", "E", {})  # Should evict first

        assert len(history) == 3
        records = history.get_last(10)
        assert records[0].old_stage == "B"  # First record evicted

    def test_clear_history(self):
        """Test clearing history."""
        history = TransitionHistory()

        history.create_record("A", "B", {})
        history.create_record("B", "C", {})

        assert len(history) == 2

        history.clear()

        assert len(history) == 0
        assert history.current_stage is None

    def test_average_duration(self):
        """Test calculating average stage duration."""
        history = TransitionHistory()

        # Manually add records with known durations
        history.records.append(
            TransitionRecord(
                old_stage="PLANNING",
                new_stage="EXECUTION",
                timestamp=datetime.now(timezone.utc),
                context={},
                duration_ms=100.0,
            )
        )
        history.records.append(
            TransitionRecord(
                old_stage="PLANNING",
                new_stage="READING",
                timestamp=datetime.now(timezone.utc),
                context={},
                duration_ms=200.0,
            )
        )

        avg = history.get_average_duration("PLANNING")
        assert avg == 150.0

    def test_average_duration_no_data(self):
        """Test average duration returns None when no data."""
        history = TransitionHistory()

        assert history.get_average_duration("UNKNOWN") is None


class TestStateHookManagerHistory:
    """Tests for StateHookManager with history tracking."""

    def test_manager_has_history_by_default(self):
        """Test that manager has history enabled by default."""
        manager = StateHookManager()

        assert manager.history_enabled
        assert manager.history is not None

    def test_disable_history(self):
        """Test disabling history tracking."""
        manager = StateHookManager(enable_history=False)

        assert not manager.history_enabled
        assert manager.history is None

    def test_custom_history_size(self):
        """Test custom history size."""
        manager = StateHookManager(history_max_size=5)

        assert manager.history.max_size == 5

    def test_fire_transition_records_history(self):
        """Test that fire_transition records to history."""
        manager = StateHookManager()

        record = manager.fire_transition("INITIAL", "PLANNING", {"tool": "read"})

        assert record is not None
        assert record.old_stage == "INITIAL"
        assert record.new_stage == "PLANNING"
        assert len(manager.history) == 1

    def test_fire_transition_returns_none_when_disabled(self):
        """Test fire_transition returns None when history disabled."""
        manager = StateHookManager(enable_history=False)

        record = manager.fire_transition("INITIAL", "PLANNING", {})

        assert record is None

    def test_get_last_transition(self):
        """Test getting last transition record."""
        manager = StateHookManager()

        manager.fire_transition("INITIAL", "PLANNING", {})
        manager.fire_transition("PLANNING", "EXECUTION", {})

        last = manager.get_last_transition()
        assert last.new_stage == "EXECUTION"

    def test_clear_history(self):
        """Test clearing history via manager."""
        manager = StateHookManager()

        manager.fire_transition("INITIAL", "PLANNING", {})
        assert len(manager.history) == 1

        manager.clear_history()
        assert len(manager.history) == 0


class TestHistoryAwareHooks:
    """Tests for history-aware transition hooks."""

    def test_on_transition_with_history_decorator(self):
        """Test the history-aware decorator."""
        manager = StateHookManager()
        received: List[Dict[str, Any]] = []

        @manager.on_transition_with_history
        def analyze(old: str, new: str, ctx: Dict, history: TransitionHistory) -> None:
            received.append(
                {
                    "old": old,
                    "new": new,
                    "history_len": len(history),
                    "has_cycle": history.has_cycle(),
                }
            )

        manager.fire_transition("INITIAL", "PLANNING", {})
        manager.fire_transition("PLANNING", "EXECUTION", {})
        manager.fire_transition("EXECUTION", "PLANNING", {})  # Cycle!

        assert len(received) == 3
        assert received[0]["history_len"] == 1
        assert received[0]["has_cycle"] is False
        assert received[2]["history_len"] == 3
        assert received[2]["has_cycle"] is True

    def test_on_transition_with_history_priority(self):
        """Test history-aware hooks respect priority."""
        manager = StateHookManager()
        call_order: List[str] = []

        @manager.on_transition_with_history(priority=10)
        def high_priority(old, new, ctx, history):
            call_order.append("high")

        @manager.on_transition_with_history(priority=1)
        def low_priority(old, new, ctx, history):
            call_order.append("low")

        manager.fire_transition("A", "B", {})

        assert call_order == ["high", "low"]

    def test_history_available_in_hook(self):
        """Test that history is available for queries in hook."""
        manager = StateHookManager()

        manager.fire_transition("INITIAL", "PLANNING", {})
        manager.fire_transition("PLANNING", "READING", {})

        detected_pattern = []

        @manager.on_transition_with_history
        def check_pattern(old, new, ctx, history):
            detected_pattern.extend(history.get_stage_sequence())

        manager.fire_transition("READING", "EXECUTION", {})

        # After third transition, sequence should be full
        assert detected_pattern == [
            "INITIAL",
            "PLANNING",
            "READING",
            "EXECUTION",
        ]

    def test_both_hook_types_fire(self):
        """Test that regular and history-aware hooks both fire."""
        manager = StateHookManager()
        regular_called = []
        history_called = []

        @manager.on_transition
        def regular_hook(old, new, ctx):
            regular_called.append(new)

        @manager.on_transition_with_history
        def history_hook(old, new, ctx, history):
            history_called.append((new, len(history)))

        manager.fire_transition("A", "B", {})

        assert regular_called == ["B"]
        assert history_called == [("B", 1)]

    def test_history_disabled_provides_empty_history(self):
        """Test that history-aware hooks get empty history when disabled."""
        manager = StateHookManager(enable_history=False)
        received_len = []

        @manager.on_transition_with_history
        def hook(old, new, ctx, history):
            received_len.append(len(history))

        manager.fire_transition("A", "B", {})
        manager.fire_transition("B", "C", {})

        # Should get empty history each time
        assert received_len == [0, 0]

    def test_hook_error_isolation(self):
        """Test that one hook error doesn't affect others."""
        manager = StateHookManager()
        calls = []

        @manager.on_transition_with_history
        def failing_hook(old, new, ctx, history):
            raise ValueError("Test error")

        @manager.on_transition_with_history(priority=-1)  # Lower priority, called after
        def succeeding_hook(old, new, ctx, history):
            calls.append(new)

        # Should not raise
        manager.fire_transition("A", "B", {})

        # Second hook should still execute
        assert calls == ["B"]


class TestLoggingAndMetricsHooks:
    """Tests for pre-built hooks with history."""

    def test_logging_hook_with_history(self):
        """Test LoggingHook works with history-enabled manager."""
        manager = StateHookManager()
        logging_hook = LoggingHook(include_context=True)
        manager.add_hook(logging_hook.create_hook())

        # Should not raise
        manager.fire_transition("INITIAL", "PLANNING", {"tool": "read"})

        assert len(manager.history) == 1

    def test_metrics_hook_with_history(self):
        """Test MetricsHook works alongside history."""
        manager = StateHookManager()
        metrics = MetricsHook()
        manager.add_hook(metrics.create_hook())

        manager.fire_transition("INITIAL", "PLANNING", {})
        manager.fire_transition("PLANNING", "EXECUTION", {})
        manager.fire_transition("EXECUTION", "PLANNING", {})

        stats = metrics.get_stats()
        assert stats["total_transitions"] == 3
        assert stats["stage_entries"]["PLANNING"] == 2

        # History should track same transitions
        assert len(manager.history) == 3
        assert manager.history.get_stage_visit_count("PLANNING") == 2


class TestHistoryIntegrationPatterns:
    """Tests for advanced history usage patterns."""

    def test_cycle_detection_hook(self):
        """Test implementing cycle detection with history."""
        manager = StateHookManager()
        cycle_warnings: List[str] = []

        @manager.on_transition_with_history
        def detect_cycles(old, new, ctx, history):
            if history.get_stage_visit_count(new) > 1:
                cycle_warnings.append(f"Revisiting {new} (cycle detected)")

        manager.fire_transition("INITIAL", "PLANNING", {})
        manager.fire_transition("PLANNING", "EXECUTION", {})
        manager.fire_transition("EXECUTION", "PLANNING", {})  # Cycle

        assert len(cycle_warnings) == 1
        assert "Revisiting PLANNING" in cycle_warnings[0]

    def test_stage_duration_tracking(self):
        """Test tracking time spent in each stage."""
        manager = StateHookManager()
        durations: Dict[str, List[float]] = {}

        @manager.on_transition_with_history
        def track_durations(old, new, ctx, history):
            last = history.get_last(1)
            if last and last[0].duration_ms:
                durations.setdefault(old, []).append(last[0].duration_ms)

        manager.fire_transition("INITIAL", "PLANNING", {})
        time.sleep(0.01)
        manager.fire_transition("PLANNING", "EXECUTION", {})
        time.sleep(0.02)
        manager.fire_transition("EXECUTION", "COMPLETION", {})

        assert "PLANNING" in durations
        assert "EXECUTION" in durations
        assert durations["EXECUTION"][0] > durations["PLANNING"][0]

    def test_conditional_behavior_based_on_history(self):
        """Test implementing conditional behavior based on history."""
        manager = StateHookManager()
        actions: List[str] = []

        @manager.on_transition_with_history
        def conditional_action(old, new, ctx, history):
            # If we've been in PLANNING more than twice, skip to COMPLETION
            if new == "PLANNING" and history.get_stage_visit_count("PLANNING") > 2:
                actions.append("TOO_MANY_PLANNING_CYCLES")
            elif new == "EXECUTION" and not history.has_visited("READING"):
                actions.append("SKIPPED_READING")

        manager.fire_transition("INITIAL", "PLANNING", {})
        manager.fire_transition("PLANNING", "EXECUTION", {})  # Skipped READING

        assert "SKIPPED_READING" in actions
