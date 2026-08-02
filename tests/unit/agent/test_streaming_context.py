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

"""Unit tests for streaming context module."""

import time
import pytest

from victor.agent.streaming.context import (
    StreamingChatContext,
    create_stream_context,
)
from victor.agent.unified_classifier import ClassifierTaskType


class TestStreamingChatContext:
    """Tests for StreamingChatContext dataclass."""

    def test_creation_with_defaults(self):
        """Context can be created with just user message."""
        ctx = StreamingChatContext(user_message="test message")
        assert ctx.user_message == "test message"
        assert ctx.total_iterations == 0
        assert ctx.force_completion is False
        assert ctx.is_analysis_task is False

    def test_creation_with_custom_values(self):
        """Context can be created with custom values."""
        ctx = StreamingChatContext(
            user_message="test",
            max_total_iterations=50,
            max_exploration_iterations=20,
            is_analysis_task=True,
        )
        assert ctx.max_total_iterations == 50
        assert ctx.max_exploration_iterations == 20
        assert ctx.is_analysis_task is True

    def test_elapsed_time(self):
        """elapsed_time calculates correctly."""
        ctx = StreamingChatContext(user_message="test")
        time.sleep(0.01)  # Small delay
        elapsed = ctx.elapsed_time()
        assert elapsed >= 0.01
        assert elapsed < 1.0

    def test_increment_iteration(self):
        """increment_iteration updates and returns count."""
        ctx = StreamingChatContext(user_message="test")
        assert ctx.total_iterations == 0

        result = ctx.increment_iteration()
        assert result == 1
        assert ctx.total_iterations == 1

        result = ctx.increment_iteration()
        assert result == 2
        assert ctx.total_iterations == 2

    def test_is_over_time_limit_false(self):
        """is_over_time_limit returns False when within limit."""
        ctx = StreamingChatContext(user_message="test")
        assert ctx.is_over_time_limit(100.0) is False

    def test_is_over_time_limit_true(self):
        """is_over_time_limit returns True when exceeded."""
        ctx = StreamingChatContext(user_message="test")
        # Set last_activity_time in the past (10 seconds ago)
        ctx.last_activity_time = time.time() - 10
        assert ctx.is_over_time_limit(5.0) is True

    def test_is_over_iteration_limit_false(self):
        """is_over_iteration_limit returns False when within limit."""
        ctx = StreamingChatContext(
            user_message="test",
            max_total_iterations=10,
            total_iterations=5,
        )
        assert ctx.is_over_iteration_limit() is False

    def test_is_over_iteration_limit_true(self):
        """is_over_iteration_limit returns True when at limit."""
        ctx = StreamingChatContext(
            user_message="test",
            max_total_iterations=10,
            total_iterations=10,
        )
        assert ctx.is_over_iteration_limit() is True

    def test_should_force_completion_from_flag(self):
        """should_force_completion returns True when force_completion is True."""
        ctx = StreamingChatContext(user_message="test", force_completion=True)
        assert ctx.should_force_completion() is True

    def test_should_force_completion_from_blocked(self):
        """should_force_completion returns True when blocked attempts exceed threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=4,
            max_blocked_before_force=4,
        )
        assert ctx.should_force_completion() is True

    def test_should_not_force_completion_after_three_blocked_attempts_by_default(self):
        """Default blocked threshold should still allow recovery after three attempts."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=3,
        )
        assert ctx.should_force_completion() is False

    def test_should_force_completion_false(self):
        """should_force_completion returns False when no conditions met."""
        ctx = StreamingChatContext(user_message="test")
        assert ctx.should_force_completion() is False

    def test_record_blocked_attempt_below_threshold(self):
        """record_blocked_attempt returns False when below threshold."""
        ctx = StreamingChatContext(user_message="test")
        assert ctx.record_blocked_attempt() is False
        assert ctx.consecutive_blocked_attempts == 1

    def test_record_blocked_attempt_three_times_still_below_default_threshold(self):
        """Default blocked threshold should trigger on the fourth attempt, not the third."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=2,
        )
        assert ctx.record_blocked_attempt() is False
        assert ctx.consecutive_blocked_attempts == 3

    def test_record_blocked_attempt_at_threshold(self):
        """record_blocked_attempt returns True when reaching threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=3,
            max_blocked_before_force=4,
        )
        assert ctx.record_blocked_attempt() is True
        assert ctx.consecutive_blocked_attempts == 4

    def test_reset_blocked_attempts(self):
        """reset_blocked_attempts clears the counter."""
        ctx = StreamingChatContext(user_message="test", consecutive_blocked_attempts=5)
        ctx.reset_blocked_attempts()
        assert ctx.consecutive_blocked_attempts == 0

    def test_accumulate_content(self):
        """accumulate_content adds to total."""
        ctx = StreamingChatContext(user_message="test")
        ctx.accumulate_content("hello")
        assert ctx.total_accumulated_chars == 5
        ctx.accumulate_content(" world")
        assert ctx.total_accumulated_chars == 11

    def test_update_context_message(self):
        """update_context_message updates correctly."""
        ctx = StreamingChatContext(user_message="original")
        ctx.update_context_message("new content")
        assert ctx.context_msg == "new content"

    def test_update_context_message_empty_uses_user_message(self):
        """update_context_message falls back to user_message when empty."""
        ctx = StreamingChatContext(user_message="original")
        ctx.update_context_message("")
        assert ctx.context_msg == "original"

    def test_cumulative_usage_defaults(self):
        """cumulative_usage has expected default keys."""
        ctx = StreamingChatContext(user_message="test")
        assert "prompt_tokens" in ctx.cumulative_usage
        assert "completion_tokens" in ctx.cumulative_usage
        assert "cache_read_input_tokens" in ctx.cumulative_usage
        assert ctx.cumulative_usage["prompt_tokens"] == 0

    def test_topology_fields_default_to_empty_or_none(self):
        """Topology-related streaming state starts with neutral defaults."""
        ctx = StreamingChatContext(user_message="test")

        assert ctx.runtime_context_overrides == {}
        assert ctx.provider_kwargs == {}
        assert ctx.topology_input is None
        assert ctx.topology_decision is None
        assert ctx.topology_plan is None
        assert ctx.structured_routing_policy is None
        assert ctx.topology_events == []
        assert ctx.degradation_events == []
        assert ctx.recovery_events == []
        assert ctx.runtime_override_snapshot is None

    def test_continuation_ledger_renders_intent_plan_and_resume_context(self):
        """Continuation ledger should preserve task, plan, and resume context."""
        ctx = StreamingChatContext(user_message="inspect runtime")
        ctx.set_task_intent("Investigate the runtime convergence path")
        ctx.extend_plan_steps(["Read orchestrator", "Check compaction behavior"])
        ctx.resume_summary = "2 tool call(s) used; previous turn ended before completion"
        ctx.resume_recent_tools = ["read", "edit"]
        ctx.resume_recent_resources = ["victor/agent/orchestrator.py"]
        ctx.record_intent_event("tool_intent", "planned read (path=victor/agent/orchestrator.py)")

        ledger = ctx.build_continuation_ledger()

        assert "Intent:" in ledger
        assert "Plan:" in ledger
        assert "Resume:" in ledger
        assert "Recent tools:" in ledger
        assert "Recent files:" in ledger
        assert "Recent actions:" in ledger

    def test_record_compaction_event_updates_context_state(self):
        """Compaction event helper should update recovery state and ledger."""
        ctx = StreamingChatContext(user_message="inspect runtime", total_iterations=3)

        ctx.record_compaction_event(
            summary="Removed stale tool chatter",
            messages_removed=12,
            strategy="hybrid",
            reason="pre_tool_output",
            policy_reason="tool_output_exceeds_remaining_budget",
        )

        assert ctx.compaction_occurred is True
        assert ctx.last_compaction_turn == 3
        assert ctx.compaction_message_removed_count == 12
        assert ctx.last_compaction_strategy == "hybrid"
        assert ctx.last_compaction_reason == "pre_tool_output"
        assert ctx.last_compaction_policy_reason == "tool_output_exceeds_remaining_budget"
        assert ctx.intent_log[-1]["kind"] == "compaction"


class TestCreateStreamContext:
    """Tests for create_stream_context factory function."""

    def test_basic_creation(self):
        """Factory creates context with message."""
        ctx = create_stream_context("test message")
        assert ctx.user_message == "test message"
        assert ctx.context_msg == "test message"

    def test_custom_iterations(self):
        """Factory accepts custom iteration limits."""
        ctx = create_stream_context(
            "test",
            max_iterations=50,
            max_exploration=15,
        )
        assert ctx.max_total_iterations == 50
        assert ctx.max_exploration_iterations == 15

    def test_tool_budget(self):
        """Factory accepts tool budget."""
        ctx = create_stream_context("test", tool_budget=100)
        assert ctx.complexity_tool_budget == 100

    def test_tool_budget_none(self):
        """Factory handles None tool budget."""
        ctx = create_stream_context("test", tool_budget=None)
        assert ctx.complexity_tool_budget is None

    def test_blocked_threshold_override(self):
        """Factory accepts blocked-attempt threshold override."""
        ctx = create_stream_context("test", max_blocked_before_force=7)
        assert ctx.max_blocked_before_force == 7


class TestStreamingChatContextTaskTypes:
    """Tests for task type handling in context."""

    def test_default_task_type(self):
        """Default task type is DEFAULT."""
        ctx = StreamingChatContext(user_message="test")
        assert ctx.unified_task_type == ClassifierTaskType.DEFAULT

    def test_analysis_task_type(self):
        """Can set ANALYSIS task type."""
        ctx = StreamingChatContext(
            user_message="test",
            unified_task_type=ClassifierTaskType.ANALYSIS,
            is_analysis_task=True,
        )
        assert ctx.unified_task_type == ClassifierTaskType.ANALYSIS
        assert ctx.is_analysis_task is True

    def test_action_task_type(self):
        """Can set ACTION task type."""
        ctx = StreamingChatContext(
            user_message="test",
            unified_task_type=ClassifierTaskType.ACTION,
            is_action_task=True,
            needs_execution=True,
        )
        assert ctx.unified_task_type == ClassifierTaskType.ACTION
        assert ctx.is_action_task is True
        assert ctx.needs_execution is True

    def test_generation_task_type(self):
        """Can set GENERATION task type."""
        ctx = StreamingChatContext(
            user_message="test",
            unified_task_type=ClassifierTaskType.GENERATION,
        )
        assert ctx.unified_task_type == ClassifierTaskType.GENERATION


class TestStreamingChatContextGoals:
    """Tests for goals handling in context."""

    def test_default_empty_goals(self):
        """Goals default to empty list."""
        ctx = StreamingChatContext(user_message="test")
        assert ctx.goals == []

    def test_goals_can_be_set(self):
        """Goals can be set at creation."""
        goals = ["find files", "analyze code"]
        ctx = StreamingChatContext(user_message="test", goals=goals)
        assert ctx.goals == goals

    def test_goals_are_mutable(self):
        """Goals list can be modified."""
        ctx = StreamingChatContext(user_message="test")
        ctx.goals.append("new goal")
        assert "new goal" in ctx.goals
