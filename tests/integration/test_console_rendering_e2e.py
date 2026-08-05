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

"""Integration tests for console rendering fixes.

These tests verify end-to-end rendering flows to ensure content is
always displayed to users, even during thinking mode transitions and tool
boundaries.
"""

import pytest
from victor.ui.rendering import LiveDisplayRenderer, stream_response
from rich.console import Console
from victor.ui.theme import victor_theme


def _themed_console() -> Console:
    """Mirror production console setup: the renderer assumes the victor theme
    (which defines ``thinking.text`` / ``thinking.indicator``) is loaded."""
    return Console(theme=victor_theme)


class TestConsoleRenderingE2E:
    """End-to-end tests for console rendering."""

    @pytest.mark.asyncio
    async def test_deepseek_thinking_mode_full_response_visible(self):
        """Full response should be visible with DeepSeek thinking markers.

        This test verifies that content is not lost during thinking mode
        transitions. DeepSeek uses a special reasoning_content field that needs
        to be properly handled.
        """
        # Note: This test requires mocking or a real provider with thinking support
        # For CI/CD, we mock the provider. For local testing, use real provider.

        # Mock agent that simulates DeepSeek thinking mode
        from unittest.mock import MagicMock, AsyncMock
        from victor.providers.base import StreamChunk

        mock_agent = MagicMock()
        mock_renderer = LiveDisplayRenderer(_themed_console())

        async def mock_stream():
            # Simulate DeepSeek thinking pattern
            yield StreamChunk(
                content="",
                metadata={"reasoning_content": "Let me think about this step by step..."},
            )
            yield StreamChunk(content="Here's my analysis:", metadata=None)
            yield StreamChunk(content="**Step 1:** Analyze the problem", metadata=None)
            yield StreamChunk(content="**Step 2:** Provide solution", metadata=None)
            # Stream ends without explicit thinking_end marker

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        # Stream and verify
        content_buffer = await stream_response(
            mock_agent,
            "Explain quantum computing",
            mock_renderer,
            suppress_thinking=False,
        )

        # Verify all content was captured
        assert "Let me think" in content_buffer or "analysis" in content_buffer.lower()
        assert len(content_buffer) > 0, "Content buffer should not be empty"
        assert "Step 1" in content_buffer or "Step 2" in content_buffer

    @pytest.mark.asyncio
    async def test_content_visible_when_stream_ends_in_thinking(self):
        """Content should be visible even if stream ends during thinking mode.

        This is a critical regression test for the bug where content buffered
        during thinking mode was lost if the stream ended without proper
        thinking_end signal.
        """
        from unittest.mock import MagicMock
        from victor.providers.base import StreamChunk

        mock_agent = MagicMock()
        mock_renderer = LiveDisplayRenderer(_themed_console())

        async def broken_stream():
            # Stream starts thinking but never calls on_thinking_end
            yield StreamChunk(
                content="Let me think...", metadata={"reasoning_content": "Thinking..."}
            )
            yield StreamChunk(content="Here's my answer:", metadata=None)
            # Stream ends abruptly without thinking_end marker
            # (this was causing content to be lost)

        mock_agent.stream_chat = MagicMock(return_value=broken_stream())

        # Stream should handle this gracefully
        content_buffer = await stream_response(
            mock_agent,
            "What is machine learning?",
            mock_renderer,
            suppress_thinking=False,
        )

        # Content should still be visible
        assert len(content_buffer) > 0
        assert "answer" in content_buffer.lower() or "machine learning" in content_buffer.lower()

    @pytest.mark.asyncio
    async def test_content_not_lost_at_tool_boundaries(self):
        """Content should not be lost when tool calls occur.

        This verifies the fix to tool call break logic that ensures content
        at tool boundaries is always forwarded to the renderer.
        """
        from unittest.mock import MagicMock
        from victor.providers.base import StreamChunk

        mock_agent = MagicMock()
        mock_renderer = LiveDisplayRenderer(_themed_console())

        # Track content calls without interfering with the renderer
        original_on_content = mock_renderer.on_content
        content_calls = []

        def tracking_on_content(text):
            content_calls.append(text)
            return original_on_content(text)

        mock_renderer.on_content = tracking_on_content

        async def stream_with_tools():
            # Content before tool call
            yield StreamChunk(content="Before tool call", metadata=None)

            # Tool call
            yield StreamChunk(
                content="",
                metadata={
                    "tool_start": {
                        "name": "read",
                        "arguments": {"path": "/tmp/test.txt"},
                    }
                },
            )

            # Content after tool call (THIS WAS BEING LOST)
            yield StreamChunk(content="After tool call", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=stream_with_tools())

        # Stream the response
        content_buffer = await stream_response(
            mock_agent,
            "Read /tmp/test.txt and analyze",
            mock_renderer,
            suppress_thinking=False,
        )

        # Verify ALL content was forwarded to renderer
        # Check that both before and after content were forwarded
        assert any("Before tool call" in call for call in content_calls)
        assert any("After tool call" in call for call in content_calls)

        # Verify final content buffer
        assert "Before tool call" in content_buffer
        assert "After tool call" in content_buffer

    @pytest.mark.asyncio
    async def test_multiple_thinking_blocks_handled_correctly(self):
        """Multiple thinking blocks should all be visible.

        Some models (like Qwen) use inline thinking markers that can
        appear multiple times. This test verifies all thinking blocks are
        captured and displayed.
        """
        from unittest.mock import MagicMock
        from victor.providers.base import StreamChunk

        mock_agent = MagicMock()
        mock_renderer = LiveDisplayRenderer(_themed_console())

        async def stream_with_multiple_thinking():
            # First thinking block
            yield StreamChunk(
                content="First thinking: ",
                metadata={"reasoning_content": "Analyzing requirements..."},
            )
            yield StreamChunk(content="Requirements understood", metadata=None)

            # Normal content
            yield StreamChunk(content="Now I'll proceed", metadata=None)

            # Second thinking block
            yield StreamChunk(
                content="Second thinking: ",
                metadata={"reasoning_content": "Considering options..."},
            )
            yield StreamChunk(content="Option chosen", metadata=None)

            # Final answer
            yield StreamChunk(content="Here's the solution", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=stream_with_multiple_thinking())

        content_buffer = await stream_response(
            mock_agent,
            "Design a system architecture",
            mock_renderer,
            suppress_thinking=False,
        )

        # All content should be present
        assert "First thinking" in content_buffer or "requirements" in content_buffer.lower()
        assert "Second thinking" in content_buffer or "option" in content_buffer.lower()
        assert "solution" in content_buffer.lower()

    @pytest.mark.asyncio
    async def test_empty_stream_handled_gracefully(self):
        """Empty stream should be handled without errors."""
        from unittest.mock import MagicMock
        from victor.providers.base import StreamChunk

        mock_agent = MagicMock()
        mock_renderer = LiveDisplayRenderer(_themed_console())

        async def empty_stream():
            # Stream with no content at all
            return
            yield  # Empty generator

        mock_agent.stream_chat = MagicMock(return_value=empty_stream())

        # Should not raise error
        content_buffer = await stream_response(
            mock_agent, "Hello", mock_renderer, suppress_thinking=False
        )

        # Should return empty string
        assert content_buffer == ""


class TestThinkingModeTransitions:
    """Tests for thinking mode state transitions."""

    def test_thinking_mode_toggle(self):
        """Thinking mode should toggle correctly."""
        console = _themed_console()
        renderer = LiveDisplayRenderer(console)

        renderer.start()

        # Enter thinking mode
        assert not renderer._in_thinking_mode
        renderer.on_thinking_start()
        assert renderer._in_thinking_mode

        # Exit thinking mode
        renderer.on_thinking_end()
        assert not renderer._in_thinking_mode

    @pytest.mark.asyncio
    async def test_thinking_mode_always_closed_before_finalize(self):
        """Thinking state should be closed before finalize is called.

        This is a critical fix - if thinking mode is still active when finalize()
        is called, content can be lost.
        """
        from unittest.mock import MagicMock
        from victor.providers.base import StreamChunk

        mock_agent = MagicMock()
        mock_renderer = LiveDisplayRenderer(_themed_console())

        async def stream_ending_in_thinking():
            # Start thinking
            yield StreamChunk(content="", metadata={"reasoning_content": "Thinking..."})
            # Stream ends WITHOUT calling on_thinking_end
            # (this was the bug)

        mock_agent.stream_chat = MagicMock(return_value=stream_ending_in_thinking())

        # Stream should handle this and force thinking mode closed
        content_buffer = await stream_response(
            mock_agent, "Test question", mock_renderer, suppress_thinking=False
        )

        # Verify thinking mode was closed
        assert not mock_renderer._in_thinking_mode

    def test_rapid_thinking_transitions(self):
        """Should handle rapid enter/exit thinking cycles."""
        console = _themed_console()
        renderer = LiveDisplayRenderer(console)

        renderer.start()

        # Rapid cycles
        for i in range(5):
            renderer.on_thinking_start()
            assert renderer._in_thinking_mode
            renderer.on_thinking_end()
            assert not renderer._in_thinking_mode

        # Should be stable
        assert not renderer._in_thinking_mode


class TestContentBufferSize:
    """Tests for content buffer management."""

    @pytest.mark.asyncio
    async def test_large_content_buffer_handled_correctly(self):
        """Large responses (>10KB) should be handled without issues."""
        from unittest.mock import MagicMock
        from victor.providers.base import StreamChunk

        mock_agent = MagicMock()
        mock_renderer = LiveDisplayRenderer(_themed_console())

        large_content = "X" * 15000  # 15KB of content

        async def large_stream():
            yield StreamChunk(content=large_content, metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=large_stream())

        content_buffer = await stream_response(
            mock_agent,
            "Generate large response",
            mock_renderer,
            suppress_thinking=False,
        )

        # Should handle large content
        assert len(content_buffer) == 15000
        assert content_buffer == large_content
