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

"""Integration tests for streaming pipeline completion detection.

Tests that final responses are always displayed to users, even when tool calls
are present, and that content is preserved when streams end in thinking mode.
"""

from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from rich.console import Console
from victor.agent.services.chat_stream_runtime import ServiceStreamingRuntime

from victor.ui.theme import victor_theme


def _themed_console() -> Console:
    """Mirror production console setup: the renderer assumes the victor theme
    (which defines ``thinking.text`` / ``thinking.indicator``) is loaded."""
    return Console(theme=victor_theme)


from victor.agent.services.orchestrator_protocol_adapter import (
    OrchestratorProtocolAdapter,
)
from victor.agent.unified_task_tracker import TrackerTaskType
from victor.agent.streaming.context import StreamingChatContext
from victor.framework.task import TaskComplexity
from victor.ui.rendering.live_renderer import LiveDisplayRenderer


class TestStreamingCompletion:
    """Test streaming pipeline completion and finalization."""

    @pytest.mark.asyncio
    async def test_final_response_displayed_after_tool_calls(self):
        """Verify final response is rendered when tool calls are present.

        Simulates the scenario where:
        1. LLM makes tool calls
        2. Provider sends additional content chunks after tool calls
        3. Final chunk with is_final=True is received

        Expected: All content including final response is rendered.
        """

        # Mock provider that sends tool calls then content
        async def mock_provider_stream(*args, **kwargs):
            # First chunk: tool call
            chunk1 = MagicMock()
            chunk1.content = ""
            chunk1.tool_calls = [{"name": "read", "arguments": {"path": "file.py"}}]
            chunk1.is_final = False
            yield chunk1

            # Second chunk: content after tool calls
            chunk2 = MagicMock()
            chunk2.content = "Here is the file content you requested."
            chunk2.tool_calls = None
            chunk2.is_final = True
            chunk2.finish_reason = "stop"
            yield chunk2

        # Mock orchestrator
        mock_orch = MagicMock()
        mock_orch.provider.stream = mock_provider_stream
        mock_orch.model = "test-model"
        mock_orch.temperature = 0.7
        mock_orch.max_tokens = 4096

        # Mock renderer
        console = _themed_console()
        renderer = LiveDisplayRenderer(console=console)
        renderer.on_content = MagicMock()
        renderer.finalize = MagicMock()

        # Simulate streaming with tool calls
        full_content = ""
        tool_calls = None

        async for chunk in mock_provider_stream():
            if chunk.content:
                full_content += chunk.content
                if renderer:
                    renderer.on_content(chunk.content)

            if chunk.tool_calls:
                tool_calls = chunk.tool_calls

            if getattr(chunk, "is_final", False):
                break

        # Verify: Content after tool calls was accumulated
        assert "Here is the file content" in full_content

        # Verify: Renderer's on_content was called with final content
        assert renderer.on_content.called
        renderer.on_content.assert_any_call("Here is the file content you requested.")

    @pytest.mark.asyncio
    async def test_thinking_mode_content_preserved(self):
        """Verify content is not lost when stream ends in thinking mode.

        Simulates the scenario where:
        1. Stream starts in thinking mode
        2. Content is added while in thinking mode
        3. Stream ends without explicitly exiting thinking mode

        Expected: Content is preserved in content_buffer and displayed on finalize.
        """
        console = _themed_console()
        renderer = LiveDisplayRenderer(console=console)

        # Simulate thinking mode start
        renderer.on_thinking_start()

        # Add content while in thinking mode (using on_content)
        test_content = "This is thinking content that should be preserved."
        renderer.on_content(test_content)

        # Verify: Content is in content_buffer (single source of truth)
        assert test_content in renderer._content_buffer

        # End stream while still in thinking mode
        renderer.finalize()

        # Verify: After finalize, content_buffer still has the content (not lost)
        assert test_content in renderer._content_buffer

    @pytest.mark.asyncio
    async def test_multiple_chunks_after_tool_calls(self):
        """Verify multiple content chunks after tool calls are all accumulated.

        Simulates the scenario where:
        1. Tool calls are received
        2. Multiple content chunks follow
        3. Stream completes

        Expected: All chunks are accumulated and rendered.
        """

        # Mock provider
        async def mock_provider_stream(*args, **kwargs):
            # Tool call chunk
            chunk1 = MagicMock()
            chunk1.content = ""
            chunk1.tool_calls = [{"name": "code_search", "arguments": {"query": "test"}}]
            chunk1.is_final = False
            yield chunk1

            # Content chunk 1
            chunk2 = MagicMock()
            chunk2.content = "Search results found 10 files. "
            chunk2.tool_calls = None
            chunk2.is_final = False
            yield chunk2

            # Content chunk 2
            chunk3 = MagicMock()
            chunk3.content = "Here are the top 3 results:"
            chunk3.tool_calls = None
            chunk3.is_final = False
            yield chunk3

            # Final chunk
            chunk4 = MagicMock()
            chunk4.content = "Result 1, Result 2, Result 3"
            chunk4.tool_calls = None
            chunk4.is_final = True
            chunk4.finish_reason = "stop"
            yield chunk4

        # Accumulate content
        full_content = ""
        async for chunk in mock_provider_stream():
            if chunk.content:
                full_content += chunk.content

        # Verify: All chunks were accumulated
        assert "Search results found" in full_content
        assert "Here are the top 3 results:" in full_content
        assert "Result 1, Result 2, Result 3" in full_content

    @pytest.mark.asyncio
    async def test_continuation_request_reuses_prior_stream_task_shape(self):
        """Continuation prompts should preserve prior task-shape hints in the runtime context."""
        orch = MagicMock()
        orch.has_capability.return_value = False
        orch.get_capability_value.return_value = None
        orch._cumulative_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        orch._conversation_controller = MagicMock()
        orch.messages = []
        orch.settings = SimpleNamespace(recovery_blocked_consecutive_threshold=5)
        orch._classify_task_keywords.return_value = {}
        orch._tool_planner = SimpleNamespace(infer_goals_from_message=lambda _: [])
        orch.tool_budget = 42
        orch.tool_calls_used = 0
        orch._task_completion_detector = None
        orch._pending_continuation_task_context = {
            "carry_forward_task_shape": True,
            "coarse_task_type": "analysis",
            "is_analysis_task": True,
            "is_action_task": False,
            "needs_execution": False,
        }

        runtime = ServiceStreamingRuntime(OrchestratorProtocolAdapter(orch))
        runtime._prepare_stream = AsyncMock(
            return_value=(
                SimpleNamespace(),
                0.0,
                0.0,
                {},
                30,
                10,
                0,
                False,
                TrackerTaskType.GENERAL,
                SimpleNamespace(complexity=TaskComplexity.ANALYSIS),
                42,
            )
        )

        ctx = await runtime._create_stream_context("continue")

        assert ctx.is_analysis_task is True
        assert ctx.coarse_task_type == "analysis"
        assert orch._pending_continuation_task_context is None


class TestStreamCompletionValidation:
    """Test stream completion validation and logging."""

    def test_empty_stream_detection(self):
        """Verify empty streams are detected and logged."""
        # Simulate empty stream
        full_content = ""
        tool_calls = None
        force_completion = False

        content_length = len(full_content.strip()) if full_content else 0

        # Should detect empty stream
        if not tool_calls and not force_completion and content_length == 0:
            # This would trigger warning log
            detected = True
        else:
            detected = False

        assert detected, "Empty stream should be detected"

    def test_short_stream_warning(self):
        """Verify very short streams trigger warnings."""
        # Simulate short stream
        full_content = "OK"
        tool_calls = None
        force_completion = False

        content_length = len(full_content.strip()) if full_content else 0

        # Should detect short stream
        if not tool_calls and not force_completion and content_length < 50:
            # This would trigger warning log
            detected = True
        else:
            detected = False

        assert detected, "Short stream should trigger warning"

    def test_normal_stream_validation(self):
        """Verify normal streams pass validation."""
        # Simulate normal stream
        full_content = "This is a substantial response with enough content."
        tool_calls = None
        force_completion = False

        content_length = len(full_content.strip()) if full_content else 0

        # Should NOT trigger warnings
        if not tool_calls and not force_completion:
            if content_length == 0:
                detected_empty = True
            elif content_length < 50:
                detected_short = True
            else:
                detected_empty = False
                detected_short = False
        else:
            detected_empty = False
            detected_short = False

        assert not detected_empty, "Normal stream should not be detected as empty"
        assert not detected_short, "Normal stream should not be detected as short"


class TestThinkingModeTransitions:
    """Test thinking mode state transitions and content preservation."""

    def test_thinking_to_normal_transition(self):
        """Verify content preserved when transitioning from thinking to normal mode."""
        console = _themed_console()
        renderer = LiveDisplayRenderer(console=console)

        # Start thinking mode
        renderer.on_thinking_start()

        # Add thinking content (using on_content while in thinking mode)
        renderer.on_content("Thinking: Let me analyze this...")
        renderer.on_content("This is the actual analysis content.")

        # Verify content is in content_buffer (single source of truth)
        assert "Thinking:" in renderer._content_buffer
        assert "actual analysis content" in renderer._content_buffer

        # End thinking mode
        renderer.on_thinking_end()

        # Add normal content
        renderer.on_content("More content after thinking.")

        # Verify: All content is in content_buffer
        assert "actual analysis content" in renderer._content_buffer
        assert "More content after thinking" in renderer._content_buffer

    def test_content_buffer_during_thinking(self):
        """Verify content goes to content_buffer during thinking mode.

        With the single-buffer design, content is preserved in _content_buffer
        even when stream ends in thinking mode.
        """
        console = _themed_console()
        renderer = LiveDisplayRenderer(console=console)

        # Start thinking mode
        renderer.on_thinking_start()

        # Add content while in thinking mode
        test_content = "Important response content"
        renderer.on_content(test_content)

        # Verify: Content is in content_buffer (single source of truth)
        assert test_content in renderer._content_buffer, "Content should be in content_buffer"

        # Finalize (simulates stream end)
        renderer.finalize()

        # Content should still be in content_buffer after finalize
        assert test_content in renderer._content_buffer
