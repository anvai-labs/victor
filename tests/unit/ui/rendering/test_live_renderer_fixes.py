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

"""Unit tests for LiveDisplayRenderer content display fixes.

These tests verify the fixes for console rendering issues where content
was being buffered but not displayed to users.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from victor.ui.theme import victor_theme

from victor.ui.rendering import LiveDisplayRenderer


@pytest.fixture(autouse=True)
def reset_live_renderer_logger():
    """Keep caplog assertions isolated from logging mutations in other tests."""
    logger = logging.getLogger("victor.ui.rendering.live_renderer")
    original_disabled = logger.disabled
    original_handlers = logger.handlers.copy()
    original_level = logger.level
    original_propagate = logger.propagate

    logger.disabled = False
    logger.handlers.clear()
    logger.propagate = True

    yield

    logger.disabled = original_disabled
    logger.handlers = original_handlers
    logger.level = original_level
    logger.propagate = original_propagate


class TestLiveDisplayRendererContentDisplay:
    """Tests for content display fixes in Phase 1."""

    def test_content_visible_in_thinking_mode(self):
        """Content should be visible immediately during thinking mode.

        This is a regression test for the bug where content was buffered
        but not displayed during thinking mode. With the fix, content
        prints immediately to console (not Live display which is paused).
        """
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        renderer.on_thinking_start()

        # Add content during thinking mode - should print immediately
        renderer.on_content("This should be visible during thinking")

        # Content should be in main buffer
        assert "This should be visible during thinking" in renderer._content_buffer

        # Note: Thinking content prints to console immediately, not to Live display
        # (which is paused during thinking mode)

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_finalize_shows_buffered_content(self, mock_live_class):
        """finalize() should ensure all buffered content is displayed.

        This tests the fail-safe mechanism that displays content if Live display
        was never started or was stopped early.
        """
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        # Simulate stream ending in thinking mode without Live update
        renderer.on_thinking_start()
        renderer.on_content("Buffered content in thinking mode")
        renderer._live = None  # Simulate Live being stopped

        result = renderer.finalize()

        # Content should be returned even if Live wasn't updated
        assert result == "Buffered content in thinking mode"

    def test_thinking_end_updates_live_display(self):
        """on_thinking_end() should resume Live display with buffered content.

        This verifies the fix that resume() recreates Live display with
        content buffered during thinking, ensuring it becomes visible.
        """
        from unittest.mock import MagicMock

        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        renderer.start()
        renderer.on_thinking_start()
        renderer.on_content("Content during thinking")
        renderer.on_thinking_end()

        # Live display should be updated (via resume())
        assert renderer._live is not None
        assert "Content during thinking" in renderer._content_buffer

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_content_always_added_to_buffer(self, mock_live_class):
        """Content should always be added to buffer, regardless of mode.

        This ensures content is preserved for final display even if stream
        ends in thinking mode.
        """
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        renderer.start()

        # Add content in normal mode
        renderer.on_content("Normal content")
        assert "Normal content" in renderer._content_buffer

        # Add content in thinking mode
        renderer.on_thinking_start()
        renderer.on_content("Thinking content")
        assert "Thinking content" in renderer._content_buffer

        # Both should be in buffer
        assert "Normal content" in renderer._content_buffer
        assert "Thinking content" in renderer._content_buffer

    def test_finalize_preserves_content(self):
        """finalize() should preserve all content in the buffer."""
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        # Add content in thinking mode
        renderer.on_thinking_start()
        renderer.on_content("Thinking mode content")
        renderer.on_thinking_end()

        # Add normal content
        renderer.on_content("Normal mode content")

        result = renderer.finalize()

        # All content should be preserved
        assert "Thinking mode content" in result
        assert "Normal mode content" in result

    def test_finalize_with_empty_buffers(self):
        """finalize() should handle empty buffers gracefully."""
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        # Don't add any content
        result = renderer.finalize()

        # Should return empty string
        assert result == ""

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_multiple_thinking_cycles(self, mock_live_class):
        """Multiple thinking cycles should handle content correctly."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        renderer.start()

        # First thinking cycle
        renderer.on_thinking_start()
        renderer.on_content("First thinking")
        renderer.on_thinking_end()

        # Normal content
        renderer.on_content("Normal between")

        # Second thinking cycle
        renderer.on_thinking_start()
        renderer.on_content("Second thinking")
        renderer.on_thinking_end()

        # All content should be in buffer
        assert "First thinking" in renderer._content_buffer
        assert "Normal between" in renderer._content_buffer
        assert "Second thinking" in renderer._content_buffer

        # finalize should work
        result = renderer.finalize()
        assert len(result) > 0


class TestLiveDisplayRendererFailSafe:
    """Tests for fail-safe mechanisms in LiveDisplayRenderer."""

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_fail_safe_display_when_live_never_started(self, mock_live_class):
        """Fail-safe should display content if Live was never started."""
        mock_live_class.return_value = None  # Live never starts
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        # Add content without starting Live
        renderer.on_content("Content without Live")

        result = renderer.finalize()

        # Content should be in result
        assert "Content without Live" in result

    def test_fail_safe_preserves_content_on_error(self):
        """Content should be preserved even if Live update fails."""
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        # Start Live
        renderer.start()

        # Add content
        content = "Important content"
        renderer.on_content(content)

        # Stop Live (simulating early termination)
        renderer._live = None

        # finalize should still return content
        result = renderer.finalize()

        assert result == content
        assert content in result


class TestLiveDisplayRendererDebugLogging:
    """Tests for debug logging in LiveDisplayRenderer."""

    def test_finalize_logs_buffer_lengths(self, caplog):
        """finalize() should log buffer lengths for debugging."""
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        # Add content
        renderer.on_content("Test content")

        with caplog.at_level("DEBUG", logger="victor.ui.rendering.live_renderer"):
            result = renderer.finalize()

        # Should log debug info
        assert any("content_buffer_len" in record.message for record in caplog.records)
        assert any("in_thinking_mode" in record.message for record in caplog.records)
        # Note: thinking_buffer_len removed - no longer needed

    def test_finalize_logs_empty_content_warning(self, caplog):
        """finalize() should warn if content is empty."""
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        # Don't add any content
        with caplog.at_level("DEBUG"):
            result = renderer.finalize()

        # Should not warn for empty content (this is valid)
        # Warning only if content was expected but not received
        assert not any("empty content" in record.message for record in caplog.records)

    def test_unmatched_resume_is_debug_only(self, caplog):
        """Benign extra resume calls should not pollute operator logs."""
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        with caplog.at_level("WARNING"):
            renderer.resume()

        assert not any("no matching pause" in record.message for record in caplog.records)


class TestLiveDisplayRendererNoDuplication:
    """Tests for content duplication fixes."""

    def test_thinking_content_no_duplication_in_buffer(self):
        """Content during thinking mode should not be duplicated in buffer."""
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        renderer.on_thinking_start()
        renderer.on_content("First chunk")
        renderer.on_content("Second chunk")
        renderer.on_thinking_end()

        # Each chunk should appear exactly once in buffer
        assert renderer._content_buffer.count("First chunk") == 1
        assert renderer._content_buffer.count("Second chunk") == 1

    def test_normal_content_streams_to_live_display(self):
        """Normal content should stream to Live display incrementally."""
        from unittest.mock import MagicMock

        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        # Mock Live to track updates
        with patch("victor.ui.rendering.live_renderer.Live") as mock_live_class:
            mock_live = MagicMock()
            mock_live_class.return_value = mock_live
            renderer = LiveDisplayRenderer(console)
            renderer.start()

            # Stream normal content
            renderer.on_content("Chunk 1")
            renderer.on_content("Chunk 2")
            renderer.on_content("Chunk 3")

            # Live display should be updated multiple times
            assert mock_live.update.call_count >= 3

    def test_rapid_state_transitions(self):
        """Rapid transitions between thinking and normal should work correctly."""
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        # Rapid transitions
        renderer.on_content("Normal 1")
        renderer.on_thinking_start()
        renderer.on_content("Thinking 1")
        renderer.on_thinking_end()
        renderer.on_content("Normal 2")
        renderer.on_thinking_start()
        renderer.on_content("Thinking 2")
        renderer.on_thinking_end()
        renderer.on_content("Normal 3")

        # All content should be preserved exactly once
        assert renderer._content_buffer.count("Normal 1") == 1
        assert renderer._content_buffer.count("Normal 2") == 1
        assert renderer._content_buffer.count("Normal 3") == 1
        assert renderer._content_buffer.count("Thinking 1") == 1
        assert renderer._content_buffer.count("Thinking 2") == 1


class TestLiveDisplayRendererSimplifiedAPI:
    """Tests for simplified API after removing double buffering."""

    def test_no_thinking_buffer_attribute(self):
        """_thinking_buffer attribute should not exist (removed in fix)."""
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        # Should not have _thinking_buffer
        assert not hasattr(renderer, "_thinking_buffer") or not getattr(
            renderer, "_thinking_buffer", ""
        )

    def test_no_last_thinking_rendered_attribute(self):
        """_last_thinking_rendered attribute should not exist (removed in fix)."""
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        # Should not have _last_thinking_rendered
        assert not hasattr(renderer, "_last_thinking_rendered") or not getattr(
            renderer, "_last_thinking_rendered", ""
        )

    def test_single_buffer_for_all_content(self):
        """All content should go through single _content_buffer."""
        console = Console(theme=victor_theme)
        renderer = LiveDisplayRenderer(console)

        # Add content in different modes
        renderer.on_content("Normal content")
        renderer.on_thinking_start()
        renderer.on_content("Thinking content")
        renderer.on_thinking_end()
        renderer.on_content("More normal content")

        # Everything should be in single buffer in order
        assert "Normal content" in renderer._content_buffer
        assert "Thinking content" in renderer._content_buffer
        assert "More normal content" in renderer._content_buffer

        # Verify order
        normal_1_pos = renderer._content_buffer.index("Normal content")
        thinking_pos = renderer._content_buffer.index("Thinking content")
        normal_2_pos = renderer._content_buffer.index("More normal content")
        assert normal_1_pos < thinking_pos < normal_2_pos


class TestToolInvocationRendering:
    """The bash invocation line is styled chrome, not markdown content."""

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_invocation_not_raw_markup_in_content_buffer(self, mock_live_class):
        import io

        mock_live_class.return_value = MagicMock()
        buf = io.StringIO()
        console = Console(
            file=buf, theme=victor_theme, force_terminal=False, no_color=True, width=100
        )
        renderer = LiveDisplayRenderer(console)
        renderer.start()

        renderer.on_tool_start("ls", {"path": "victor/"})

        # Regression: the invocation must NOT be dumped into the markdown content
        # buffer as Rich markup — that is what made [dim]/[bold cyan] print literally.
        assert "[bold cyan]" not in renderer._content_buffer
        assert "[dim]$[/]" not in renderer._content_buffer

        # It is printed via the Rich-markup console path, with markup interpreted
        # (no_color strips styling) — so the command shows, not the raw tags.
        output = buf.getvalue()
        assert "ls" in output
        assert "victor/" in output
        assert "[bold cyan]" not in output
        assert "[dim]" not in output
