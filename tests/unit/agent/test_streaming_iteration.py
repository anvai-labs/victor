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

"""Unit tests for streaming iteration module."""

import pytest

from victor.agent.streaming.iteration import (
    IterationAction,
    IterationResult,
    ProviderResponseResult,
    ToolExecutionResult,
    create_break_result,
    create_continue_result,
    create_force_completion_result,
)
from victor.providers.base import StreamChunk


class TestIterationAction:
    """Tests for IterationAction enum."""

    def test_continue_action(self):
        """CONTINUE action exists."""
        assert IterationAction.CONTINUE is not None

    def test_break_action(self):
        """BREAK action exists."""
        assert IterationAction.BREAK is not None

    def test_yield_and_continue(self):
        """YIELD_AND_CONTINUE action exists."""
        assert IterationAction.YIELD_AND_CONTINUE is not None

    def test_yield_and_break(self):
        """YIELD_AND_BREAK action exists."""
        assert IterationAction.YIELD_AND_BREAK is not None

    def test_force_completion(self):
        """FORCE_COMPLETION action exists."""
        assert IterationAction.FORCE_COMPLETION is not None


class TestIterationResult:
    """Tests for IterationResult dataclass."""

    def test_default_creation(self):
        """Result can be created with defaults."""
        result = IterationResult(action=IterationAction.CONTINUE)
        assert result.action == IterationAction.CONTINUE
        assert result.chunks == []
        assert result.content == ""
        assert result.tool_calls == []

    def test_should_break_true(self):
        """should_break returns True for break actions."""
        result = IterationResult(action=IterationAction.BREAK)
        assert result.should_break is True

        result = IterationResult(action=IterationAction.YIELD_AND_BREAK)
        assert result.should_break is True

    def test_should_break_false(self):
        """should_break returns False for continue actions."""
        result = IterationResult(action=IterationAction.CONTINUE)
        assert result.should_break is False

        result = IterationResult(action=IterationAction.YIELD_AND_CONTINUE)
        assert result.should_break is False

    def test_should_yield_with_chunks(self):
        """should_yield returns True when chunks present."""
        result = IterationResult(action=IterationAction.CONTINUE)
        result.add_chunk(StreamChunk(content="test"))
        assert result.should_yield is True

    def test_should_yield_from_action(self):
        """should_yield returns True for yield actions."""
        result = IterationResult(action=IterationAction.YIELD_AND_CONTINUE)
        assert result.should_yield is True

        result = IterationResult(action=IterationAction.YIELD_AND_BREAK)
        assert result.should_yield is True

    def test_has_tool_calls_true(self):
        """has_tool_calls returns True when tool calls present."""
        result = IterationResult(
            action=IterationAction.CONTINUE,
            tool_calls=[{"name": "read_file", "arguments": {}}],
        )
        assert result.has_tool_calls is True

    def test_has_tool_calls_false(self):
        """has_tool_calls returns False when no tool calls."""
        result = IterationResult(action=IterationAction.CONTINUE)
        assert result.has_tool_calls is False

    def test_has_content_true(self):
        """has_content returns True when content present."""
        result = IterationResult(action=IterationAction.CONTINUE, content="some text")
        assert result.has_content is True

    def test_has_content_false(self):
        """has_content returns False when no content."""
        result = IterationResult(action=IterationAction.CONTINUE)
        assert result.has_content is False

    def test_add_chunk(self):
        """add_chunk appends to chunks list."""
        result = IterationResult(action=IterationAction.CONTINUE)
        chunk = StreamChunk(content="test")
        result.add_chunk(chunk)
        assert len(result.chunks) == 1
        assert result.chunks[0] == chunk

    def test_set_break_without_chunks(self):
        """set_break sets BREAK when no chunks."""
        result = IterationResult(action=IterationAction.CONTINUE)
        result.set_break()
        assert result.action == IterationAction.BREAK

    def test_set_break_with_chunks(self):
        """set_break sets YIELD_AND_BREAK when chunks present."""
        result = IterationResult(action=IterationAction.CONTINUE)
        result.add_chunk(StreamChunk(content="test"))
        result.set_break()
        assert result.action == IterationAction.YIELD_AND_BREAK

    def test_set_continue_without_chunks(self):
        """set_continue sets CONTINUE when no chunks."""
        result = IterationResult(action=IterationAction.BREAK)
        result.set_continue()
        assert result.action == IterationAction.CONTINUE

    def test_set_continue_with_chunks(self):
        """set_continue sets YIELD_AND_CONTINUE when chunks present."""
        result = IterationResult(action=IterationAction.BREAK)
        result.add_chunk(StreamChunk(content="test"))
        result.set_continue()
        assert result.action == IterationAction.YIELD_AND_CONTINUE


class TestProviderResponseResult:
    """Tests for ProviderResponseResult dataclass."""

    def test_default_creation(self):
        """Result can be created with defaults."""
        result = ProviderResponseResult()
        assert result.content == ""
        assert result.tool_calls == []
        assert result.garbage_detected is False

    def test_has_tool_calls(self):
        """has_tool_calls property works."""
        result = ProviderResponseResult(tool_calls=[{"name": "test", "arguments": {}}])
        assert result.has_tool_calls is True

        result = ProviderResponseResult()
        assert result.has_tool_calls is False

    def test_has_content(self):
        """has_content property works."""
        result = ProviderResponseResult(content="hello")
        assert result.has_content is True

        result = ProviderResponseResult()
        assert result.has_content is False


class TestToolExecutionResult:
    """Tests for ToolExecutionResult dataclass."""

    def test_default_creation(self):
        """Result can be created with defaults."""
        result = ToolExecutionResult()
        assert result.results == []
        assert result.all_succeeded is True

    def test_has_results(self):
        """has_results property works."""
        result = ToolExecutionResult()
        assert result.has_results is False

        result.add_result("test", success=True)
        assert result.has_results is True

    def test_add_result_success(self):
        """add_result adds successful result."""
        result = ToolExecutionResult()
        result.add_result("read_file", success=True, result="content")
        assert len(result.results) == 1
        assert result.results[0]["name"] == "read_file"
        assert result.results[0]["success"] is True
        assert result.all_succeeded is True

    def test_add_result_failure(self):
        """add_result adds failed result and updates all_succeeded."""
        result = ToolExecutionResult()
        result.add_result("read_file", success=False, error="file not found")
        assert len(result.results) == 1
        assert result.results[0]["success"] is False
        assert result.results[0]["error"] == "file not found"
        assert result.all_succeeded is False

    def test_add_multiple_results(self):
        """Multiple results can be added."""
        result = ToolExecutionResult()
        result.add_result("tool1", success=True)
        result.add_result("tool2", success=True)
        assert len(result.results) == 2
        assert result.all_succeeded is True

    def test_add_result_with_args(self):
        """add_result includes arguments."""
        result = ToolExecutionResult()
        result.add_result(
            "read_file",
            success=True,
            args={"path": "/test.txt"},
            elapsed=0.5,
        )
        assert result.results[0]["args"]["path"] == "/test.txt"
        assert result.results[0]["elapsed"] == 0.5


class TestCreateBreakResult:
    """Tests for create_break_result factory."""

    def test_creates_break_action(self):
        """Factory creates BREAK action."""
        result = create_break_result()
        assert result.action == IterationAction.BREAK

    def test_with_content(self):
        """Factory adds content and chunk."""
        result = create_break_result(content="Done!")
        assert result.content == "Done!"
        assert len(result.chunks) == 1
        assert result.chunks[0].content == "Done!"

    def test_with_error(self):
        """Factory includes error."""
        result = create_break_result(error="Something failed")
        assert result.error == "Something failed"


class TestCreateContinueResult:
    """Tests for create_continue_result factory."""

    def test_creates_continue_action(self):
        """Factory creates CONTINUE action."""
        result = create_continue_result()
        assert result.action == IterationAction.CONTINUE

    def test_with_content(self):
        """Factory includes content."""
        result = create_continue_result(content="Processing...")
        assert result.content == "Processing..."

    def test_with_tool_calls(self):
        """Factory includes tool calls."""
        tool_calls = [{"name": "read_file", "arguments": {"path": "/test"}}]
        result = create_continue_result(tool_calls=tool_calls)
        assert result.tool_calls == tool_calls

    def test_with_tokens(self):
        """Factory includes token count."""
        result = create_continue_result(tokens=100.5)
        assert result.tokens_used == 100.5


class TestCreateForceCompletionResult:
    """Tests for create_force_completion_result factory."""

    def test_creates_force_completion_action(self):
        """Factory creates FORCE_COMPLETION action."""
        result = create_force_completion_result()
        assert result.action == IterationAction.FORCE_COMPLETION

    def test_with_reason(self):
        """Factory includes reason in chunk."""
        result = create_force_completion_result(reason="Time limit reached")
        assert len(result.chunks) == 1
        assert "Time limit reached" in result.chunks[0].content

    def test_without_reason(self):
        """Factory works without reason."""
        result = create_force_completion_result()
        assert len(result.chunks) == 0
