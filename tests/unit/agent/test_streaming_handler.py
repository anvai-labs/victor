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

"""Unit tests for streaming handler module."""

import time
import pytest
from unittest.mock import MagicMock, call

from victor.agent.streaming.context import StreamingChatContext
from victor.agent.streaming.handler import StreamingChatHandler
from victor.agent.streaming.iteration import (
    IterationAction,
    IterationResult,
    ProviderResponseResult,
    ToolExecutionResult,
)
from victor.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    return Settings(
        analytics_enabled=False,
        use_semantic_tool_selection=False,
    )


@pytest.fixture
def mock_message_adder():
    """Create mock message adder."""
    adder = MagicMock()
    adder.add_message = MagicMock()
    return adder


@pytest.fixture
def handler(mock_settings, mock_message_adder):
    """Create a StreamingChatHandler for testing."""
    return StreamingChatHandler(
        settings=mock_settings,
        message_adder=mock_message_adder,
        session_idle_timeout=60.0,
    )


@pytest.fixture
def basic_context():
    """Create a basic StreamingChatContext."""
    return StreamingChatContext(user_message="test message")


class TestStreamingChatHandlerCreation:
    """Tests for StreamingChatHandler creation."""

    def test_handler_creation(self, mock_settings, mock_message_adder):
        """Handler can be created with required args."""
        handler = StreamingChatHandler(
            settings=mock_settings,
            message_adder=mock_message_adder,
        )
        assert handler.settings == mock_settings
        assert handler.message_adder == mock_message_adder

    def test_handler_custom_time_limit(self, mock_settings, mock_message_adder):
        """Handler accepts custom time limit."""
        handler = StreamingChatHandler(
            settings=mock_settings,
            message_adder=mock_message_adder,
            session_idle_timeout=120.0,
        )
        assert handler.session_idle_timeout == 120.0


class TestCheckTimeLimit:
    """Tests for check_time_limit method."""

    def test_within_limit_returns_none(self, handler, basic_context):
        """Returns None when within time limit."""
        result = handler.check_time_limit(basic_context)
        assert result is None

    def test_over_limit_returns_result(self, handler, mock_message_adder):
        """Returns result when over time limit."""
        ctx = StreamingChatContext(user_message="test")
        ctx.last_activity_time = time.time() - 120  # 2 minutes ago
        handler.session_idle_timeout = 60.0  # 1 minute limit

        result = handler.check_time_limit(ctx)

        assert result is not None
        assert result.action == IterationAction.FORCE_COMPLETION
        assert ctx.force_completion is True
        mock_message_adder.add_message.assert_called_once()


class TestCheckIterationLimit:
    """Tests for check_iteration_limit method."""

    def test_within_limit_returns_none(self, handler):
        """Returns None when within iteration limit."""
        ctx = StreamingChatContext(
            user_message="test",
            max_total_iterations=30,
            total_iterations=10,
        )
        result = handler.check_iteration_limit(ctx)
        assert result is None

    def test_at_limit_returns_result(self, handler):
        """Returns result when at iteration limit."""
        ctx = StreamingChatContext(
            user_message="test",
            max_total_iterations=30,
            total_iterations=30,
        )
        result = handler.check_iteration_limit(ctx)
        assert result is not None
        assert result.action == IterationAction.BREAK


class TestCheckForceCompletion:
    """Tests for check_force_completion method."""

    def test_not_forced_returns_none(self, handler, basic_context):
        """Returns None when no force condition."""
        result = handler.check_force_completion(basic_context)
        assert result is None

    def test_forced_returns_result(self, handler):
        """Returns result when force_completion is True."""
        ctx = StreamingChatContext(user_message="test", force_completion=True)
        result = handler.check_force_completion(ctx)
        assert result is not None
        assert result.action == IterationAction.FORCE_COMPLETION

    def test_blocked_attempts_triggers_force(self, handler):
        """Returns result when blocked attempts exceed threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=4,
            max_blocked_before_force=4,
        )
        result = handler.check_force_completion(ctx)
        assert result is not None


class TestHandleBlockedAttempts:
    """Tests for handle_blocked_attempts method."""

    def test_below_threshold_returns_none(self, handler):
        """Returns None when below threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=1,
            max_blocked_before_force=4,
        )
        result = handler.handle_blocked_attempts(ctx)
        assert result is None
        assert ctx.consecutive_blocked_attempts == 2

    def test_at_threshold_returns_result(self, handler, mock_message_adder):
        """Returns result and resets counter at threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=3,
            max_blocked_before_force=4,
        )
        result = handler.handle_blocked_attempts(ctx)
        assert result is not None
        assert result.action == IterationAction.YIELD_AND_CONTINUE
        assert ctx.consecutive_blocked_attempts == 0
        mock_message_adder.add_message.assert_called_once()


class TestProcessProviderResponse:
    """Tests for process_provider_response method."""

    def test_response_with_tool_calls(self, handler, basic_context):
        """Processes response with tool calls."""
        response = ProviderResponseResult(
            content="Let me check that",
            tool_calls=[{"name": "read_file", "arguments": {"path": "/test"}}],
            tokens_used=50.0,
        )
        result = handler.process_provider_response(response, basic_context)

        assert result.action == IterationAction.CONTINUE
        assert result.has_tool_calls is True
        assert basic_context.total_accumulated_chars > 0

    def test_response_with_content_only(self, handler, basic_context):
        """Processes response with content but no tool calls."""
        response = ProviderResponseResult(
            content="Here is the answer",
            tokens_used=25.0,
        )
        result = handler.process_provider_response(response, basic_context)

        assert result.action == IterationAction.YIELD_AND_CONTINUE
        assert result.has_content is True
        assert len(result.chunks) == 1

    def test_response_with_garbage(self, handler, basic_context):
        """Processes response with garbage detection."""
        response = ProviderResponseResult(
            content="some content",
            garbage_detected=True,
        )
        result = handler.process_provider_response(response, basic_context)

        assert basic_context.force_completion is True

    def test_empty_response(self, handler, basic_context):
        """Processes empty response."""
        response = ProviderResponseResult()
        result = handler.process_provider_response(response, basic_context)

        assert result.action == IterationAction.CONTINUE


class TestProcessToolResults:
    """Tests for process_tool_results method."""

    def test_successful_tool_results(self, handler, basic_context):
        """Processes successful tool results."""
        execution = ToolExecutionResult()
        execution.add_result(
            "read_file",
            success=True,
            args={"path": "/test.txt"},
            elapsed=0.5,
        )

        chunks = handler.process_tool_results(execution, basic_context)

        assert len(chunks) >= 1
        # Find the tool result chunk
        tool_result_chunks = [c for c in chunks if c.metadata and "tool_result" in c.metadata]
        assert len(tool_result_chunks) == 1
        assert tool_result_chunks[0].metadata["tool_result"]["success"] is True

    def test_failed_tool_results(self, handler, basic_context):
        """Processes failed tool results."""
        execution = ToolExecutionResult()
        execution.add_result(
            "read_file",
            success=False,
            error="file not found",
            args={"path": "/missing.txt"},
        )

        chunks = handler.process_tool_results(execution, basic_context)

        tool_result_chunks = [c for c in chunks if c.metadata and "tool_result" in c.metadata]
        assert len(tool_result_chunks) == 1
        assert tool_result_chunks[0].metadata["tool_result"]["success"] is False
        assert tool_result_chunks[0].metadata["tool_result"]["error"] == "file not found"

    def test_no_thinking_status_chunk(self, handler, basic_context):
        """process_tool_results does not emit a separate Thinking status chunk.

        The thinking indicator is handled by the pipeline/provider layer to
        avoid duplication (providers like z.ai yield reasoning content directly).
        """
        execution = ToolExecutionResult()
        execution.add_result("test_tool", success=True)

        chunks = handler.process_tool_results(execution, basic_context)

        status_chunks = [c for c in chunks if c.metadata and c.metadata.get("status")]
        assert len(status_chunks) == 0

    def test_preserves_follow_up_suggestions(self, handler, basic_context):
        """Preserves tool follow-up suggestions in tool_result metadata."""
        execution = ToolExecutionResult()
        execution.add_result(
            "code_search",
            success=True,
            args={"query": "main entry point"},
            follow_up_suggestions=[
                {
                    "command": 'graph(mode="trace", node="main", depth=3)',
                    "description": "Trace execution starting from main.",
                }
            ],
        )

        chunks = handler.process_tool_results(execution, basic_context)

        tool_result_chunks = [c for c in chunks if c.metadata and "tool_result" in c.metadata]
        assert len(tool_result_chunks) == 1
        assert tool_result_chunks[0].metadata["tool_result"]["follow_up_suggestions"] == [
            {
                "command": 'graph(mode="trace", node="main", depth=3)',
                "description": "Trace execution starting from main.",
            }
        ]


class TestGenerateToolStartChunk:
    """Tests for generate_tool_start_chunk method."""

    def test_generates_correct_chunk(self, handler):
        """Generates chunk with tool start metadata."""
        chunk = handler.generate_tool_start_chunk(
            tool_name="read_file",
            tool_args={"path": "/test.txt"},
            status_msg="Reading file...",
        )

        assert chunk.content == ""
        assert chunk.metadata is not None
        assert "tool_start" in chunk.metadata
        assert chunk.metadata["tool_start"]["name"] == "read_file"
        assert chunk.metadata["tool_start"]["arguments"]["path"] == "/test.txt"
        assert chunk.metadata["tool_start"]["status_msg"] == "Reading file..."


class TestShouldContinueLoop:
    """Tests for should_continue_loop method."""

    def test_continues_on_continue_action(self, handler, basic_context):
        """Returns True for CONTINUE action."""
        result = IterationResult(action=IterationAction.CONTINUE)
        assert handler.should_continue_loop(result, basic_context) is True

    def test_stops_on_break_action(self, handler, basic_context):
        """Returns False for BREAK action."""
        result = IterationResult(action=IterationAction.BREAK)
        assert handler.should_continue_loop(result, basic_context) is False

    def test_stops_on_iteration_limit(self, handler):
        """Returns False when iteration limit reached."""
        ctx = StreamingChatContext(
            user_message="test",
            max_total_iterations=10,
            total_iterations=10,
        )
        result = IterationResult(action=IterationAction.CONTINUE)
        assert handler.should_continue_loop(result, ctx) is False

    def test_stops_on_force_completion_without_tools(self, handler):
        """Returns False when force completion and no tool calls."""
        ctx = StreamingChatContext(user_message="test", force_completion=True)
        result = IterationResult(action=IterationAction.CONTINUE)
        assert handler.should_continue_loop(result, ctx) is False

    def test_continues_on_force_completion_with_tools(self, handler):
        """Returns True when force completion but has tool calls."""
        ctx = StreamingChatContext(user_message="test", force_completion=True)
        result = IterationResult(
            action=IterationAction.CONTINUE,
            tool_calls=[{"name": "test", "arguments": {}}],
        )
        # Force completion with tool calls should still process tools
        # Implementation may vary, but typically we let tool calls complete
        # This test documents expected behavior
        assert handler.should_continue_loop(result, ctx) is True


class TestCheckNaturalCompletion:
    """Tests for check_natural_completion method."""

    def test_returns_none_with_tool_calls(self, handler):
        """Returns None when there are tool calls."""
        ctx = StreamingChatContext(
            user_message="test",
            total_accumulated_chars=1000,
            substantial_content_threshold=500,
        )
        result = handler.check_natural_completion(ctx, has_tool_calls=True, content_length=0)
        assert result is None

    def test_returns_none_without_substantial_content(self, handler):
        """Returns None when content below threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            total_accumulated_chars=100,
            substantial_content_threshold=500,
        )
        result = handler.check_natural_completion(ctx, has_tool_calls=False, content_length=0)
        assert result is None

    def test_returns_break_with_substantial_content(self, handler):
        """Returns None - signal-based completion handles completion, not buffer/size heuristics.

        After Phase 5 cleanup, buffer/size heuristics were removed in favor of
        TaskCompletionDetector's explicit rare-marker completion signaling.
        """
        ctx = StreamingChatContext(
            user_message="test",
            total_accumulated_chars=600,
            substantial_content_threshold=500,
        )
        result = handler.check_natural_completion(ctx, has_tool_calls=False, content_length=0)
        # Signal-based completion: handler returns None, TaskCompletionDetector decides
        assert result is None


class TestHandleEmptyResponse:
    """Tests for handle_empty_response method."""

    def test_returns_none_below_threshold(self, handler):
        """Returns None when empty responses below threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_empty_responses=1,
        )
        result = handler.handle_empty_response(ctx)
        assert result is None
        assert ctx.consecutive_empty_responses == 2

    def test_returns_result_at_threshold(self, handler, mock_message_adder):
        """Returns result and forces completion at threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_empty_responses=2,  # Will become 3 which is threshold
        )
        result = handler.handle_empty_response(ctx)
        assert result is not None
        assert result.action == IterationAction.YIELD_AND_CONTINUE
        assert ctx.force_completion is True
        assert ctx.consecutive_empty_responses == 0
        mock_message_adder.add_message.assert_called_once()


class TestHandleBlockedToolCall:
    """Tests for handle_blocked_tool_call method."""

    def test_records_blocked_and_returns_chunk(self, handler, mock_message_adder):
        """Records blocked attempt and returns notification chunk."""
        ctx = StreamingChatContext(user_message="test")
        initial_blocked = ctx.total_blocked_attempts

        chunk = handler.handle_blocked_tool_call(
            ctx,
            tool_name="read_file",
            tool_args={"path": "/blocked"},
            block_reason="Already tried this 3 times",
        )

        assert ctx.total_blocked_attempts == initial_blocked + 1
        # Accept both emoji (⛔) and text ([!]) versions
        assert "⛔" in chunk.content or "[!]" in chunk.content
        mock_message_adder.add_message.assert_called_once()
        assert "TOOL BLOCKED" in mock_message_adder.add_message.call_args[0][1]


class TestCheckBlockedThreshold:
    """Tests for check_blocked_threshold method."""

    def test_returns_none_below_threshold(self, handler):
        """Returns None when below thresholds."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=1,
            total_blocked_attempts=2,
        )
        result = handler.check_blocked_threshold(
            ctx, all_blocked=False, consecutive_limit=4, total_limit=6
        )
        assert result is None

    def test_returns_result_at_consecutive_threshold(self, handler, mock_message_adder):
        """Returns result when consecutive blocked reaches limit."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=3,
            total_blocked_attempts=3,
        )
        result = handler.check_blocked_threshold(
            ctx, all_blocked=True, consecutive_limit=4, total_limit=10
        )
        assert result is not None
        assert result.clear_tool_calls is True
        mock_message_adder.add_message.assert_called()

    def test_returns_result_at_total_threshold(self, handler, mock_message_adder):
        """Returns result when total blocked reaches limit."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=0,
            total_blocked_attempts=6,
        )
        result = handler.check_blocked_threshold(
            ctx, all_blocked=False, consecutive_limit=4, total_limit=6
        )
        assert result is not None
        assert result.clear_tool_calls is True

    def test_resets_consecutive_when_not_all_blocked(self, handler):
        """Resets consecutive counter when some calls succeed."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=2,
            total_blocked_attempts=2,
        )
        result = handler.check_blocked_threshold(
            ctx, all_blocked=False, consecutive_limit=4, total_limit=10
        )
        assert result is None
        assert ctx.consecutive_blocked_attempts == 0


class TestHandleForceToolExecution:
    """Tests for handle_force_tool_execution method."""

    def test_first_attempt_prompts_tool_call(self, handler, mock_message_adder):
        """First attempt prompts for actual tool call."""
        ctx = StreamingChatContext(user_message="test")
        result = handler.handle_force_tool_execution(ctx, mentioned_tools=["read_file"])

        assert result is not None
        assert ctx.force_tool_execution_attempts == 1
        mock_message_adder.add_message.assert_called_once()
        call_content = mock_message_adder.add_message.call_args[0][1]
        assert "read_file" in call_content

    def test_third_attempt_gives_up(self, handler, mock_message_adder):
        """After 3 attempts, gives up and requests summary."""
        ctx = StreamingChatContext(
            user_message="test",
            force_tool_execution_attempts=2,  # Will become 3
        )
        result = handler.handle_force_tool_execution(ctx, mentioned_tools=["search"])

        assert result is not None
        assert ctx.force_tool_execution_attempts == 0  # Reset
        # Should have called add_message for "unable to make tool calls"
        call_content = mock_message_adder.add_message.call_args[0][1]
        assert "unable to make tool calls" in call_content


class TestToolBudgetMethods:
    """Tests for tool budget and progress checking methods."""

    def test_check_tool_budget_below_threshold(self, handler):
        """No warning when below threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=300,
            tool_calls_used=100,
        )
        result = handler.check_tool_budget(ctx, warning_threshold=250)
        assert result is None

    def test_check_tool_budget_at_threshold(self, handler):
        """Warning when at threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=300,
            tool_calls_used=250,
        )
        result = handler.check_tool_budget(ctx, warning_threshold=250)
        assert result is not None
        assert result.chunks
        assert "Approaching tool budget limit" in result.chunks[0].content

    def test_check_tool_budget_warns_on_ratio_threshold(self, handler):
        """Warn when the used/total ratio reaches the configured threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=10,
            tool_calls_used=8,
        )
        result = handler.check_tool_budget(
            ctx,
            warning_threshold=250,
            warning_pct=0.8,
            warning_remaining=1,
        )
        assert result is not None
        assert "2 remaining" in result.chunks[0].content

    def test_check_tool_budget_warns_on_low_remaining(self, handler):
        """Warn when only a few tool calls remain even if the absolute threshold is low."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=40,
            tool_calls_used=36,
        )
        result = handler.check_tool_budget(
            ctx,
            warning_threshold=250,
            warning_pct=0.95,
            warning_remaining=4,
        )
        assert result is not None
        assert "4 remaining" in result.chunks[0].content

    def test_check_tool_budget_warns_only_once_per_turn(self, handler):
        """Repeated checks should not spam duplicate warnings."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=10,
            tool_calls_used=8,
        )
        first_result = handler.check_tool_budget(
            ctx,
            warning_threshold=250,
            warning_pct=0.8,
            warning_remaining=2,
        )
        second_result = handler.check_tool_budget(
            ctx,
            warning_threshold=250,
            warning_pct=0.8,
            warning_remaining=2,
        )
        assert first_result is not None
        assert second_result is None
        assert ctx.budget_warning_shown is True

    def test_check_tool_budget_exhausted_no_warning(self, handler):
        """No warning when budget exhausted (0 remaining)."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=200,
            tool_calls_used=200,
        )
        result = handler.check_tool_budget(ctx, warning_threshold=250)
        assert result is None  # No warning because remaining is 0

    def test_check_budget_exhausted_true(self, handler):
        """Budget exhausted returns True when no budget left."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=100,
            tool_calls_used=100,
        )
        assert handler.check_budget_exhausted(ctx) is True

    def test_check_budget_exhausted_false(self, handler):
        """Budget exhausted returns False when budget remains."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=100,
            tool_calls_used=50,
        )
        assert handler.check_budget_exhausted(ctx) is False

    def test_get_budget_exhausted_chunks(self, handler):
        """Budget exhausted generates correct chunks."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=200,
        )
        chunks = handler.get_budget_exhausted_chunks(ctx)
        assert len(chunks) == 2
        assert "Tool budget reached (200)" in chunks[0].content
        assert "Generating final summary" in chunks[1].content

    def test_truncate_tool_calls_within_budget(self, handler):
        """Tool calls not truncated when within budget."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=10,
            tool_calls_used=5,
        )
        tool_calls = [{"name": "tool1"}, {"name": "tool2"}, {"name": "tool3"}]
        result = handler.truncate_tool_calls(tool_calls, ctx)
        assert len(result) == 3

    def test_truncate_tool_calls_over_budget(self, handler):
        """Tool calls truncated when over budget."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=10,
            tool_calls_used=8,  # Only 2 remaining
        )
        tool_calls = [{"name": "tool1"}, {"name": "tool2"}, {"name": "tool3"}]
        result = handler.truncate_tool_calls(tool_calls, ctx)
        assert len(result) == 2

    def test_truncate_empty_tool_calls(self, handler):
        """Empty tool calls returns empty."""
        ctx = StreamingChatContext(user_message="test")
        result = handler.truncate_tool_calls([], ctx)
        assert result == []


class TestProgressChecking:
    """Tests for progress checking methods."""

    def test_check_progress_below_limit(self, handler):
        """No force when below consecutive limit."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_calls_used=5,  # Below default 8
        )
        result = handler.check_progress_and_force(ctx, base_max_consecutive=8)
        assert result is False
        assert ctx.force_completion is False

    def test_check_progress_good_progress(self, handler):
        """No force when making good progress."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_calls_used=10,
        )
        # Add enough unique resources (10/2 = 5 needed)
        for i in range(5):
            ctx.add_unique_resource(f"file{i}.py")

        result = handler.check_progress_and_force(ctx, base_max_consecutive=8)
        assert result is False
        assert ctx.force_completion is False

    def test_check_progress_not_enough_progress(self, handler):
        """Force completion when not enough progress."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_calls_used=10,
        )
        # Only 2 resources (need 5 = 10/2)
        ctx.add_unique_resource("file1.py")
        ctx.add_unique_resource("file2.py")

        result = handler.check_progress_and_force(ctx, base_max_consecutive=8)
        assert result is True
        assert ctx.force_completion is True

    def test_check_progress_analysis_task_lenient(self, handler):
        """Analysis tasks use lenient threshold (1/4 instead of 1/2)."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_calls_used=20,
            is_analysis_task=True,
        )
        # Need 20/4 = 5 resources for analysis task
        for i in range(5):
            ctx.add_unique_resource(f"file{i}.py")

        result = handler.check_progress_and_force(ctx, base_max_consecutive=8)
        # Analysis task limit is 50, so 20 calls won't trigger
        assert result is False

    def test_check_progress_already_forcing(self, handler):
        """No change when already forcing completion."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_calls_used=10,
            force_completion=True,  # Already set
        )
        result = handler.check_progress_and_force(ctx, base_max_consecutive=8)
        assert result is False  # Returns False because it didn't set it


class TestResearchLoopDetection:
    """Tests for research loop detection methods."""

    def test_is_research_loop_true(self, handler):
        """Detects research loop correctly."""
        assert handler.is_research_loop("loop_detected", "research pattern detected") is True
        assert handler.is_research_loop("loop_detected", "Research loop: web_search") is True
        assert handler.is_research_loop("loop_detected", "RESEARCH pattern") is True

    def test_is_research_loop_false_wrong_reason(self, handler):
        """Not a research loop with different reason."""
        assert handler.is_research_loop("tool_budget", "research loop") is False
        assert handler.is_research_loop("max_iterations", "research") is False
        assert handler.is_research_loop("none", "research") is False

    def test_is_research_loop_false_no_research(self, handler):
        """Not a research loop without research in hint."""
        assert handler.is_research_loop("loop_detected", "tool budget exceeded") is False
        assert handler.is_research_loop("loop_detected", "max iterations") is False
        assert handler.is_research_loop("loop_detected", "") is False


class TestForceCompletionMessages:
    """Tests for force completion message generation."""

    def test_get_force_completion_chunks_research_loop(self, handler):
        """Research loop generates research-specific message."""
        ctx = StreamingChatContext(user_message="test")
        chunk, message = handler.get_force_completion_chunks(ctx, is_research_loop=True)

        assert "Research loop detected" in chunk.content
        assert "SYNTHESIZE" in message
        assert "search results" in message

    def test_get_force_completion_chunks_exploration_limit(self, handler):
        """Non-research generates exploration limit message."""
        ctx = StreamingChatContext(user_message="test")
        chunk, message = handler.get_force_completion_chunks(ctx, is_research_loop=False)

        assert "exploration limit" in chunk.content
        assert "FINAL COMPREHENSIVE ANSWER" in message
        assert "STOP using tools" in message

    def test_handle_force_completion_not_forcing(self, handler):
        """Returns None when force_completion is False."""
        ctx = StreamingChatContext(
            user_message="test",
            force_completion=False,
        )
        result = handler.handle_force_completion(ctx, "loop_detected", "research")
        assert result is None

    def test_handle_force_completion_research_loop(self, handler, mock_message_adder):
        """Handles research loop force completion."""
        ctx = StreamingChatContext(
            user_message="test",
            force_completion=True,
        )
        result = handler.handle_force_completion(ctx, "loop_detected", "research loop detected")

        assert result is not None
        assert len(result.chunks) == 1
        assert "Research loop detected" in result.chunks[0].content
        # Check user message was added (force completion uses "user" role)
        mock_message_adder.add_message.assert_called_once()
        call_args = mock_message_adder.add_message.call_args
        assert call_args[0][0] == "user"
        assert "SYNTHESIZE" in call_args[0][1]

    def test_handle_force_completion_exploration_limit(self, handler, mock_message_adder):
        """Handles exploration limit force completion."""
        ctx = StreamingChatContext(
            user_message="test",
            force_completion=True,
        )
        result = handler.handle_force_completion(ctx, "max_iterations", "too many iterations")

        assert result is not None
        assert len(result.chunks) == 1
        assert "exploration limit" in result.chunks[0].content
        # Check user message was added (force completion uses "user" role)
        mock_message_adder.add_message.assert_called_once()
        call_args = mock_message_adder.add_message.call_args
        assert call_args[0][0] == "user"
        assert "FINAL COMPREHENSIVE ANSWER" in call_args[0][1]


class TestRecoveryPrompts:
    """Tests for recovery prompt generation methods."""

    def test_get_recovery_prompts_analysis_task_with_budget(self, handler):
        """Analysis task with budget returns task continuation prompts."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=10,
            tool_budget=100,  # 80% threshold = 80, so 10 < 80 = has budget
        )
        prompts = handler.get_recovery_prompts(ctx, 0.5, has_thinking_mode=False)

        assert len(prompts) == 3
        # First prompt should be about continuing exploration
        assert "discovery tools" in prompts[0][0].lower() or "tool call" in prompts[0][0].lower()
        # Temperatures should increase
        assert prompts[1][1] > prompts[0][1]
        assert prompts[2][1] > prompts[1][1]

    def test_get_recovery_prompts_analysis_task_exhausted(self, handler):
        """Analysis task with exhausted budget returns summary prompts."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=90,  # Above 80% of 100
            tool_budget=100,
        )
        prompts = handler.get_recovery_prompts(ctx, 0.5, has_thinking_mode=False)

        assert len(prompts) == 3
        # Should be summary prompts since budget exhausted
        assert "summarize" in prompts[0][0].lower() or "findings" in prompts[0][0].lower()

    def test_get_recovery_prompts_thinking_mode(self, handler):
        """Thinking mode returns simpler prompts with lower temps."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=False,
            tool_calls_used=90,
            tool_budget=100,
        )
        prompts = handler.get_recovery_prompts(ctx, 0.5, has_thinking_mode=True)

        assert len(prompts) == 3
        # Should have lower temperature caps
        for _, temp in prompts:
            assert temp <= 0.9

    def test_get_recovery_prompts_with_thinking_prefix(self, handler):
        """Thinking prefix is added when provided."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=10,
            tool_budget=100,
        )
        prompts = handler.get_recovery_prompts(
            ctx, 0.5, has_thinking_mode=True, thinking_disable_prefix="/no_think"
        )

        assert len(prompts) == 3
        # First prompt should have the prefix
        assert "/no_think" in prompts[0][0]

    def test_get_recovery_prompts_action_task(self, handler):
        """Action task with budget returns task continuation prompts."""
        ctx = StreamingChatContext(
            user_message="test",
            is_action_task=True,
            is_analysis_task=False,
            tool_calls_used=10,
            tool_budget=100,
        )
        prompts = handler.get_recovery_prompts(ctx, 0.5, has_thinking_mode=False)

        assert len(prompts) == 3
        # Should be task continuation prompts
        assert "discovery tools" in prompts[0][0].lower() or "tool call" in prompts[0][0].lower()

    def test_get_recovery_prompts_standard_task(self, handler):
        """Standard task returns summary prompts."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=False,
            is_action_task=False,
            tool_calls_used=10,
            tool_budget=100,
        )
        prompts = handler.get_recovery_prompts(ctx, 0.5, has_thinking_mode=False)

        assert len(prompts) == 3
        # Should be summary prompts
        assert "summarize" in prompts[0][0].lower() or "findings" in prompts[0][0].lower()


class TestShouldUseToolsForRecovery:
    """Tests for should_use_tools_for_recovery method."""

    def test_uses_tools_for_analysis_task_first_attempts(self, handler):
        """Analysis task enables tools for first 2 attempts."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=10,
            tool_budget=100,
        )

        assert handler.should_use_tools_for_recovery(ctx, 1) is True
        assert handler.should_use_tools_for_recovery(ctx, 2) is True
        assert handler.should_use_tools_for_recovery(ctx, 3) is False

    def test_no_tools_for_exhausted_budget(self, handler):
        """No tools when budget is exhausted."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=90,  # Above 80% threshold
            tool_budget=100,
        )

        assert handler.should_use_tools_for_recovery(ctx, 1) is False

    def test_no_tools_for_standard_task(self, handler):
        """Standard tasks don't use tools for recovery."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=False,
            is_action_task=False,
            tool_calls_used=10,
            tool_budget=100,
        )

        assert handler.should_use_tools_for_recovery(ctx, 1) is False


class TestGetRecoveryFallbackMessage:
    """Tests for get_recovery_fallback_message method."""

    def test_analysis_task_with_tool_calls(self, handler):
        """Analysis task with tool calls returns file summary."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=5,
        )
        unique_resources = ["file1.py", "file2.py", "file3.py"]

        message = handler.get_recovery_fallback_message(ctx, unique_resources)

        assert "Analysis Summary" in message
        assert "3 files" in message
        assert "file1.py" in message

    def test_analysis_task_truncates_files(self, handler):
        """Analysis task truncates file list to 10."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=5,
        )
        unique_resources = [f"file{i}.py" for i in range(20)]

        message = handler.get_recovery_fallback_message(ctx, unique_resources)

        assert "20 files" in message
        # Should only show first 10
        assert "file9.py" in message
        assert "file10.py" not in message

    def test_non_analysis_task(self, handler):
        """Non-analysis task returns generic message."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=False,
            tool_calls_used=0,
        )
        unique_resources = []

        message = handler.get_recovery_fallback_message(ctx, unique_resources)

        assert "No tool calls were returned" in message
        assert "retry" in message.lower()

    def test_analysis_task_no_tool_calls(self, handler):
        """Analysis task with no tool calls returns generic message."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=0,
        )
        unique_resources = []

        message = handler.get_recovery_fallback_message(ctx, unique_resources)

        assert "No tool calls were returned" in message


class TestFormatCompletionMetrics:
    """Tests for format_completion_metrics method."""

    def test_with_provider_reported_tokens(self, handler):
        """Uses provider-reported tokens when available."""
        ctx = StreamingChatContext(
            user_message="test",
            cumulative_usage={
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            estimated_content_tokens=100,  # Should be ignored
        )

        result = handler.format_completion_metrics(ctx, elapsed_time=10.0)

        assert "in=1,000" in result
        assert "out=500" in result
        assert "10.0s" in result
        assert "50.0 tok/s" in result
        assert "cached" not in result  # No cache tokens

    def test_with_cache_read_tokens(self, handler):
        """Includes cache read info when available."""
        ctx = StreamingChatContext(
            user_message="test",
            cumulative_usage={
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 0,
            },
        )

        result = handler.format_completion_metrics(ctx, elapsed_time=10.0)

        assert "cached=800" in result

    def test_with_cache_creation_tokens(self, handler):
        """Includes cache creation info when available."""
        ctx = StreamingChatContext(
            user_message="test",
            cumulative_usage={
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 200,
            },
        )

        result = handler.format_completion_metrics(ctx, elapsed_time=10.0)

        assert "cache_new=200" in result

    def test_with_both_cache_types(self, handler):
        """Includes both cache types when available."""
        ctx = StreamingChatContext(
            user_message="test",
            cumulative_usage={
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 200,
            },
        )

        result = handler.format_completion_metrics(ctx, elapsed_time=10.0)

        assert "cached=800" in result
        assert "cache_new=200" in result

    def test_fallback_to_estimated_tokens(self, handler):
        """Falls back to estimated tokens when no provider tokens."""
        ctx = StreamingChatContext(
            user_message="test",
            cumulative_usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            estimated_content_tokens=1500.0,
        )

        result = handler.format_completion_metrics(ctx, elapsed_time=10.0)

        assert "~1500 tokens (est.)" in result
        assert "10.0s" in result
        assert "150.0 tok/s" in result

    def test_handles_zero_elapsed_time(self, handler):
        """Handles zero elapsed time without division error."""
        ctx = StreamingChatContext(
            user_message="test",
            cumulative_usage={
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
            },
        )

        result = handler.format_completion_metrics(ctx, elapsed_time=0.0)

        assert "0.0 tok/s" in result


class TestFormatBudgetExhaustedMetrics:
    """Tests for format_budget_exhausted_metrics method."""

    def test_basic_format(self, handler):
        """Basic metrics format without TTFT."""
        ctx = StreamingChatContext(
            user_message="test",
            estimated_content_tokens=2000.0,
        )

        result = handler.format_budget_exhausted_metrics(ctx, elapsed_time=20.0)

        assert "2000 tokens" in result
        assert "20.0s" in result
        assert "100.0 tok/s" in result
        assert "TTFT" not in result

    def test_with_ttft(self, handler):
        """Includes TTFT when provided."""
        ctx = StreamingChatContext(
            user_message="test",
            estimated_content_tokens=2000.0,
        )

        result = handler.format_budget_exhausted_metrics(
            ctx, elapsed_time=20.0, time_to_first_token=1.5
        )

        assert "2000 tokens" in result
        assert "TTFT: 1.50s" in result

    def test_handles_zero_elapsed_time(self, handler):
        """Handles zero elapsed time without division error."""
        ctx = StreamingChatContext(
            user_message="test",
            estimated_content_tokens=1000.0,
        )

        result = handler.format_budget_exhausted_metrics(ctx, elapsed_time=0.0)

        assert "0.0 tok/s" in result

    def test_ttft_zero_not_included(self, handler):
        """TTFT of 0 or None is not included."""
        ctx = StreamingChatContext(
            user_message="test",
            estimated_content_tokens=1000.0,
        )

        result = handler.format_budget_exhausted_metrics(
            ctx, elapsed_time=10.0, time_to_first_token=0.0
        )

        assert "TTFT" not in result


class TestGenerateToolResultChunk:
    """Tests for generate_tool_result_chunk method."""

    def test_successful_result(self, handler):
        """Generates correct chunk for successful tool."""
        chunk = handler.generate_tool_result_chunk(
            tool_name="read_file",
            tool_args={"path": "/test.py"},
            elapsed=0.5,
            success=True,
        )

        assert chunk.content == ""
        assert chunk.metadata["tool_result"]["name"] == "read_file"
        assert chunk.metadata["tool_result"]["success"] is True
        assert chunk.metadata["tool_result"]["elapsed"] == 0.5
        assert chunk.metadata["tool_result"]["arguments"] == {"path": "/test.py"}
        assert "error" not in chunk.metadata["tool_result"]

    def test_failed_result_with_error(self, handler):
        """Generates correct chunk for failed tool with error."""
        chunk = handler.generate_tool_result_chunk(
            tool_name="write_file",
            tool_args={"path": "/test.py"},
            elapsed=0.1,
            success=False,
            error="Permission denied",
        )

        assert chunk.metadata["tool_result"]["success"] is False
        assert chunk.metadata["tool_result"]["error"] == "Permission denied"

    def test_failed_result_no_error(self, handler):
        """Failed result without error message."""
        chunk = handler.generate_tool_result_chunk(
            tool_name="bash",
            tool_args={"command": "ls"},
            elapsed=0.2,
            success=False,
        )

        assert chunk.metadata["tool_result"]["success"] is False
        assert "error" not in chunk.metadata["tool_result"]


class TestGenerateFilePreviewChunk:
    """Tests for generate_file_preview_chunk method."""

    def test_short_content(self, handler):
        """Short content shows in full."""
        content = "line1\nline2\nline3"
        chunk = handler.generate_file_preview_chunk(content, "/test.py")

        assert chunk is not None
        assert chunk.metadata["file_preview"] == content
        assert chunk.metadata["path"] == "/test.py"

    def test_long_content_truncated(self, handler):
        """Long content is truncated with line count."""
        content = "\n".join([f"line{i}" for i in range(20)])
        chunk = handler.generate_file_preview_chunk(content, "/test.py", preview_lines=8)

        assert chunk is not None
        assert "... (12 more lines)" in chunk.metadata["file_preview"]
        assert "line7" in chunk.metadata["file_preview"]
        assert "line8" not in chunk.metadata["file_preview"]

    def test_empty_content_returns_none(self, handler):
        """Empty content returns None."""
        chunk = handler.generate_file_preview_chunk("", "/test.py")
        assert chunk is None


class TestGenerateEditPreviewChunk:
    """Tests for generate_edit_preview_chunk method.

    Diff format is unified-diff (proper `-`/`+` lines + `---`/`+++` headers)
    so renderer can show real colored diffs to the user.
    """

    def test_short_strings_produce_unified_diff(self, handler):
        """Short strings produce a unified diff with file headers + +/- lines."""
        chunk = handler.generate_edit_preview_chunk(
            old_string="old text",
            new_string="new text",
            path="/test.py",
        )

        assert chunk is not None
        preview = chunk.metadata["edit_preview"]
        assert "--- a/test.py" in preview or "--- a//test.py" in preview
        assert "+++ b/test.py" in preview or "+++ b//test.py" in preview
        assert "-old text" in preview
        assert "+new text" in preview
        assert chunk.metadata["path"] == "/test.py"

    def test_long_strings_capped(self, handler):
        """Very long diffs are capped to max_diff_lines."""
        old_string = "\n".join(f"old line {i}" for i in range(100))
        new_string = "\n".join(f"new line {i}" for i in range(100))
        chunk = handler.generate_edit_preview_chunk(
            old_string, new_string, "/test.py", max_diff_lines=20
        )

        assert chunk is not None
        preview = chunk.metadata["edit_preview"]
        # Should be capped — 20 lines + 1 truncation marker
        assert len(preview.split("\n")) <= 21
        assert "more lines" in preview

    def test_empty_old_produces_creation_diff(self, handler):
        """Empty old_string (file creation) produces a full-add diff."""
        chunk = handler.generate_edit_preview_chunk("", "new content", "/test.py")
        assert chunk is not None
        assert "+new content" in chunk.metadata["edit_preview"]

    def test_empty_new_produces_deletion_diff(self, handler):
        """Empty new_string (content removal) produces a full-delete diff."""
        chunk = handler.generate_edit_preview_chunk("old content", "", "/test.py")
        assert chunk is not None
        assert "-old content" in chunk.metadata["edit_preview"]

    def test_both_empty_returns_none(self, handler):
        """Both empty: no edit happened, no preview."""
        chunk = handler.generate_edit_preview_chunk("", "", "/test.py")
        assert chunk is None


class TestGenerateToolResultChunks:
    """Tests for generate_tool_result_chunks method."""

    def test_simple_success(self, handler):
        """Simple successful result generates one chunk."""
        result = {
            "name": "read",
            "elapsed": 0.5,
            "args": {"path": "/test.py"},
            "success": True,
        }

        chunks = handler.generate_tool_result_chunks(result)

        assert len(chunks) == 1
        assert chunks[0].metadata["tool_result"]["success"] is True

    def test_write_with_preview(self, handler):
        """Canonical write generates result and preview chunks."""
        result = {
            "name": "write",
            "elapsed": 0.3,
            "args": {"path": "/test.py", "content": "line1\nline2\nline3"},
            "success": True,
        }

        chunks = handler.generate_tool_result_chunks(result)

        assert len(chunks) == 2
        assert chunks[0].metadata["tool_result"]["name"] == "write"
        assert "file_preview" in chunks[1].metadata
        handler.message_adder.add_message.assert_called_once_with(
            "system",
            "File preview: /test.py",
            preview_body="line1\nline2\nline3",
            preview_kind="file",
            preview_language="py",
            preview_path="/test.py",
        )

    def test_pruned_read_chunk_preserves_full_result_for_renderer(self, handler):
        """Pruned read results expose preview and full output separately."""
        result = {
            "name": "read",
            "elapsed": 0.3,
            "args": {"path": "/test.py"},
            "success": True,
            "was_pruned": True,
            "result": "preview line 1\npreview line 2\n[PRUNED PREVIEW: omitted 200 lines]",
            "full_result": "preview line 1\npreview line 2\nfull line 3\nfull line 4",
        }

        chunks = handler.generate_tool_result_chunks(result)

        assert len(chunks) == 1
        tool_metadata = chunks[0].metadata["tool_result"]
        # The real (pruned) preview passes through — how much to show and the
        # /expand affordance are the rendering layer's decisions.
        assert tool_metadata["result"] == result["result"]
        assert tool_metadata["original_result"] == result["full_result"]
        assert tool_metadata["was_pruned"] is True

    def test_edit_with_preview(self, handler):
        """Canonical edit generates result and edit preview chunks."""
        result = {
            "name": "edit",
            "elapsed": 0.4,
            "args": {
                "ops": [
                    {
                        "type": "replace",
                        "path": "/test.py",
                        "old_str": "old1",
                        "new_str": "new1",
                    },
                    {
                        "type": "replace",
                        "path": "/test.py",
                        "old_str": "old2",
                        "new_str": "new2",
                    },
                ]
            },
            "success": True,
        }

        chunks = handler.generate_tool_result_chunks(result)

        assert len(chunks) == 3  # 1 result + 2 edit previews
        assert chunks[0].metadata["tool_result"]["name"] == "edit"
        assert "edit_preview" in chunks[1].metadata
        assert "edit_preview" in chunks[2].metadata
        # Verify two preview messages were recorded with unified-diff bodies.
        # Exact diff text is checked via substring rather than full equality
        # because difflib output formatting may vary slightly across versions.
        recorded_calls = handler.message_adder.add_message.call_args_list
        assert len(recorded_calls) == 2
        for idx, expected_old, expected_new in (
            (0, "-old1", "+new1"),
            (1, "-old2", "+new2"),
        ):
            args, kwargs = recorded_calls[idx]
            assert args == ("system", "Edit preview: /test.py")
            assert kwargs["preview_kind"] == "edit"
            assert kwargs["preview_language"] == "diff"
            assert kwargs["preview_path"] == "/test.py"
            assert expected_old in kwargs["preview_body"]
            assert expected_new in kwargs["preview_body"]

    def test_failed_result_no_preview(self, handler):
        """Failed result doesn't generate preview chunks."""
        result = {
            "name": "write_file",
            "elapsed": 0.1,
            "args": {"path": "/test.py", "content": "content"},
            "success": False,
            "error": "Permission denied",
        }

        chunks = handler.generate_tool_result_chunks(result)

        assert len(chunks) == 1
        assert chunks[0].metadata["tool_result"]["success"] is False

    def test_preview_message_falls_back_when_adder_rejects_kwargs(self, mock_settings):
        """Preview recording should retry without kwargs for legacy adders."""

        class RejectingAdder:
            def __init__(self) -> None:
                self.calls = []

            def add_message(self, role: str, content: str, **kwargs):
                self.calls.append((role, content, kwargs))
                if kwargs:
                    raise TypeError("legacy adder")

        adder = RejectingAdder()
        handler = StreamingChatHandler(
            settings=mock_settings,
            message_adder=adder,
            session_idle_timeout=60.0,
        )
        result = {
            "name": "write",
            "elapsed": 0.3,
            "args": {"path": "/test.py", "content": "line1\nline2\nline3"},
            "success": True,
        }

        chunks = handler.generate_tool_result_chunks(result)

        assert len(chunks) == 2
        assert adder.calls == [
            (
                "system",
                "File preview: /test.py",
                {
                    "preview_body": "line1\nline2\nline3",
                    "preview_kind": "file",
                    "preview_language": "py",
                    "preview_path": "/test.py",
                },
            ),
            ("system", "File preview: /test.py", {}),
        ]

    def test_max_files_limit(self, handler):
        """Respects max_files limit for edit_files."""
        result = {
            "name": "edit_files",
            "elapsed": 0.5,
            "args": {
                "files": [
                    {
                        "path": f"/file{i}.py",
                        "edits": [{"old_string": "old", "new_string": "new"}],
                    }
                    for i in range(10)
                ]
            },
            "success": True,
        }

        chunks = handler.generate_tool_result_chunks(result, max_files=2)

        # 1 result + 2 edit previews (max_files=2)
        assert len(chunks) == 3


class TestLoopWarningChunks:
    """Tests for loop warning chunk generation."""

    def test_generates_warning_chunk_and_message(self, handler):
        """Generates both warning chunk and system message."""
        warning_msg = "repeated tool call detected"

        chunk, system_msg = handler.get_loop_warning_chunks(warning_msg)

        # Accept both emoji (⚠) and text (!) versions of warning
        assert "Warning: Approaching loop limit" in chunk.content
        assert warning_msg in chunk.content
        assert "WARNING" in system_msg
        assert "loop detection" in system_msg
        assert "DIFFERENT" in system_msg

    def test_system_message_contains_guidance(self, handler):
        """System message contains helpful guidance."""
        warning_msg = "test warning"

        _, system_msg = handler.get_loop_warning_chunks(warning_msg)

        assert "writing a file repeatedly" in system_msg
        assert "summary and finish" in system_msg
        assert "force the conversation to end" in system_msg


class TestHandleLoopWarning:
    """Tests for handle_loop_warning method."""

    def test_returns_chunk_when_warning_present(self, handler, mock_message_adder):
        """Returns warning chunk when warning message is present."""
        ctx = StreamingChatContext(user_message="test")
        ctx.force_completion = False

        chunk = handler.handle_loop_warning(ctx, "repeated pattern detected")

        assert chunk is not None
        assert "repeated pattern detected" in chunk.content
        mock_message_adder.add_message.assert_called_once()
        call_args = mock_message_adder.add_message.call_args
        assert call_args[0][0] == "user"

    def test_returns_none_when_no_warning(self, handler, mock_message_adder):
        """Returns None when warning message is empty."""
        ctx = StreamingChatContext(user_message="test")
        ctx.force_completion = False

        chunk = handler.handle_loop_warning(ctx, "")

        assert chunk is None
        mock_message_adder.add_message.assert_not_called()

    def test_returns_none_when_force_completion_set(self, handler, mock_message_adder):
        """Returns None when force_completion is already set."""
        ctx = StreamingChatContext(user_message="test")
        ctx.force_completion = True

        chunk = handler.handle_loop_warning(ctx, "some warning")

        assert chunk is None
        mock_message_adder.add_message.assert_not_called()

    def test_returns_none_when_none_warning(self, handler, mock_message_adder):
        """Returns None when warning message is None."""
        ctx = StreamingChatContext(user_message="test")
        ctx.force_completion = False

        chunk = handler.handle_loop_warning(ctx, None)

        assert chunk is None
        mock_message_adder.add_message.assert_not_called()


class TestGenerateThinkingStatusChunk:
    """Tests for generate_thinking_status_chunk method."""

    def test_generates_thinking_status_chunk(self, handler):
        """Generates a thinking status chunk with correct metadata."""
        chunk = handler.generate_thinking_status_chunk()

        assert chunk.content == ""
        # Accept both emoji (💭) and text (...) versions
        status = chunk.metadata.get("status", "")
        assert "Thinking" in status
        assert "💭" in status or "..." in status

    def test_chunk_has_no_content(self, handler):
        """Generated chunk has empty content."""
        chunk = handler.generate_thinking_status_chunk()

        assert chunk.content == ""
        assert len(chunk.content) == 0


class TestGenerateBudgetErrorChunk:
    """Tests for generate_budget_error_chunk method."""

    def test_generates_budget_error_chunk(self, handler):
        """Generates a budget error chunk with correct message."""
        chunk = handler.generate_budget_error_chunk()

        assert "Unable to generate summary due to budget limit" in chunk.content

    def test_chunk_ends_with_newline(self, handler):
        """Budget error chunk ends with newline."""
        chunk = handler.generate_budget_error_chunk()

        assert chunk.content.endswith("\n")


class TestGenerateForceResponseErrorChunk:
    """Tests for generate_force_response_error_chunk method."""

    def test_generates_force_response_error_chunk(self, handler):
        """Generates a force response error chunk with correct message."""
        chunk = handler.generate_force_response_error_chunk()

        assert "Unable to generate final summary" in chunk.content
        assert "simpler query" in chunk.content


class TestGenerateFinalMarkerChunk:
    """Tests for generate_final_marker_chunk method."""

    def test_generates_final_marker_chunk(self, handler):
        """Generates a final marker chunk with is_final=True."""
        chunk = handler.generate_final_marker_chunk()

        assert chunk.content == ""
        assert chunk.is_final is True

    def test_chunk_has_no_content(self, handler):
        """Final marker chunk has empty content."""
        chunk = handler.generate_final_marker_chunk()

        assert chunk.content == ""
        assert len(chunk.content) == 0


class TestGenerateMetricsChunk:
    """Tests for generate_metrics_chunk method."""

    def test_generates_metrics_chunk_with_default_prefix(self, handler):
        """Generates a metrics chunk with double newline prefix."""
        metrics_line = "⏱ 1.5s | 💰 1000 tokens"
        chunk = handler.generate_metrics_chunk(metrics_line)

        assert chunk.content == f"\n\n{metrics_line}\n"
        assert chunk.is_final is False

    def test_generates_metrics_chunk_with_custom_prefix(self, handler):
        """Generates a metrics chunk with custom prefix."""
        metrics_line = "⏱ 2.0s | 💰 2000 tokens"
        chunk = handler.generate_metrics_chunk(metrics_line, prefix="\n")

        assert chunk.content == f"\n{metrics_line}\n"
        assert chunk.is_final is False

    def test_generates_final_metrics_chunk(self, handler):
        """Generates a final metrics chunk."""
        metrics_line = "⏱ 3.0s | 💰 3000 tokens"
        chunk = handler.generate_metrics_chunk(metrics_line, is_final=True)

        assert chunk.content == f"\n\n{metrics_line}\n"
        assert chunk.is_final is True

    def test_generates_metrics_chunk_with_empty_prefix(self, handler):
        """Generates a metrics chunk with empty prefix."""
        metrics_line = "⏱ 4.0s"
        chunk = handler.generate_metrics_chunk(metrics_line, prefix="")

        assert chunk.content == f"{metrics_line}\n"


class TestGenerateContentChunk:
    """Tests for generate_content_chunk method."""

    def test_generates_content_chunk(self, handler):
        """Generates a basic content chunk."""
        content = "Hello, world!"
        chunk = handler.generate_content_chunk(content)

        assert chunk.content == content
        assert chunk.is_final is False

    def test_generates_content_chunk_with_suffix(self, handler):
        """Generates a content chunk with suffix."""
        content = "Hello"
        chunk = handler.generate_content_chunk(content, suffix="\n")

        assert chunk.content == "Hello\n"
        assert chunk.is_final is False

    def test_generates_final_content_chunk(self, handler):
        """Generates a final content chunk."""
        content = "Goodbye!"
        chunk = handler.generate_content_chunk(content, is_final=True)

        assert chunk.content == content
        assert chunk.is_final is True

    def test_generates_final_content_chunk_with_suffix(self, handler):
        """Generates a final content chunk with suffix."""
        content = "Final answer"
        chunk = handler.generate_content_chunk(content, is_final=True, suffix="\n\n")

        assert chunk.content == "Final answer\n\n"
        assert chunk.is_final is True


class TestCheckForceAction:
    """Tests for check_force_action method."""

    def test_triggers_force_completion_when_checker_returns_true(self, handler, basic_context):
        """Sets force_completion when force checker returns True."""
        assert basic_context.force_completion is False

        def force_checker():
            return True, "Budget exhausted"

        was_triggered, hint = handler.check_force_action(basic_context, force_checker)

        assert was_triggered is True
        assert hint == "Budget exhausted"
        assert basic_context.force_completion is True

    def test_does_not_trigger_when_checker_returns_false(self, handler, basic_context):
        """Does not set force_completion when force checker returns False."""
        assert basic_context.force_completion is False

        def force_checker():
            return False, None

        was_triggered, hint = handler.check_force_action(basic_context, force_checker)

        assert was_triggered is False
        assert hint is None
        assert basic_context.force_completion is False

    def test_does_not_trigger_when_already_forced(self, handler, basic_context):
        """Does not re-trigger when force_completion is already True."""
        basic_context.force_completion = True

        def force_checker():
            return True, "Should not trigger again"

        was_triggered, hint = handler.check_force_action(basic_context, force_checker)

        assert was_triggered is False
        assert hint is None
        # Still True from before
        assert basic_context.force_completion is True

    def test_returns_hint_from_checker(self, handler, basic_context):
        """Returns the hint string from force checker."""

        def force_checker():
            return True, "Max iterations reached"

        was_triggered, hint = handler.check_force_action(basic_context, force_checker)

        assert hint == "Max iterations reached"

    def test_handles_none_hint(self, handler, basic_context):
        """Handles None hint from force checker gracefully."""

        def force_checker():
            return True, None

        was_triggered, hint = handler.check_force_action(basic_context, force_checker)

        assert was_triggered is True
        assert hint is None
        assert basic_context.force_completion is True


class TestFilterBlockedToolCalls:
    """Tests for filter_blocked_tool_calls method."""

    def test_no_blocked_tools_returns_all(self, handler, basic_context):
        """Returns all tools when none are blocked."""
        tool_calls = [
            {"name": "read_file", "arguments": {"path": "/test.txt"}},
            {
                "name": "write_file",
                "arguments": {"path": "/out.txt", "content": "hello"},
            },
        ]

        # Block checker that blocks nothing
        def block_checker(name, args):
            return None

        filtered, blocked_chunks, count = handler.filter_blocked_tool_calls(
            basic_context, tool_calls, block_checker
        )

        assert filtered == tool_calls
        assert blocked_chunks == []
        assert count == 0

    def test_all_blocked_returns_empty(self, handler, basic_context):
        """Returns empty list when all tools are blocked."""
        tool_calls = [
            {"name": "read_file", "arguments": {"path": "/test.txt"}},
            {
                "name": "write_file",
                "arguments": {"path": "/out.txt", "content": "hello"},
            },
        ]

        # Block checker that blocks everything
        def block_checker(name, args):
            return f"Tool {name} is blocked"

        filtered, blocked_chunks, count = handler.filter_blocked_tool_calls(
            basic_context, tool_calls, block_checker
        )

        assert filtered == []
        assert len(blocked_chunks) == 2
        assert count == 2

    def test_partial_blocking(self, handler, basic_context):
        """Returns only non-blocked tools when some are blocked."""
        tool_calls = [
            {"name": "read_file", "arguments": {"path": "/test.txt"}},
            {
                "name": "write_file",
                "arguments": {"path": "/out.txt", "content": "hello"},
            },
            {"name": "list_dir", "arguments": {"path": "/"}},
        ]

        # Block only write_file
        def block_checker(name, args):
            if name == "write_file":
                return "Write blocked for testing"
            return None

        filtered, blocked_chunks, count = handler.filter_blocked_tool_calls(
            basic_context, tool_calls, block_checker
        )

        assert len(filtered) == 2
        assert filtered[0]["name"] == "read_file"
        assert filtered[1]["name"] == "list_dir"
        assert len(blocked_chunks) == 1
        assert count == 1

    def test_blocked_chunk_content(self, handler, basic_context):
        """Blocked chunks contain expected content."""
        tool_calls = [
            {"name": "dangerous_tool", "arguments": {"target": "system"}},
        ]

        def block_checker(name, args):
            return "This is dangerous"

        filtered, blocked_chunks, count = handler.filter_blocked_tool_calls(
            basic_context, tool_calls, block_checker
        )

        assert len(blocked_chunks) == 1
        assert "This is dangerous" in blocked_chunks[0].content

    def test_records_blocked_count_in_context(self, handler, basic_context):
        """Records blocked tool count in context."""
        initial_blocked = basic_context.total_blocked_attempts
        tool_calls = [
            {"name": "tool1", "arguments": {}},
            {"name": "tool2", "arguments": {}},
        ]

        def block_checker(name, args):
            return "blocked"

        handler.filter_blocked_tool_calls(basic_context, tool_calls, block_checker)

        # handle_blocked_tool_call calls ctx.record_tool_blocked() for each
        assert basic_context.total_blocked_attempts == initial_blocked + 2

    def test_empty_tool_calls_list(self, handler, basic_context):
        """Handles empty tool_calls list gracefully."""

        def block_checker(name, args):
            return None

        filtered, blocked_chunks, count = handler.filter_blocked_tool_calls(
            basic_context, [], block_checker
        )

        assert filtered == []
        assert blocked_chunks == []
        assert count == 0

    def test_block_checker_receives_correct_args(self, handler, basic_context):
        """Block checker receives tool name and arguments correctly."""
        received_calls = []
        tool_calls = [
            {"name": "my_tool", "arguments": {"key": "value", "num": 42}},
        ]

        def block_checker(name, args):
            received_calls.append((name, args))
            return None

        handler.filter_blocked_tool_calls(basic_context, tool_calls, block_checker)

        assert len(received_calls) == 1
        assert received_calls[0][0] == "my_tool"
        assert received_calls[0][1] == {"key": "value", "num": 42}
