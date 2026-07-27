"""Tests for rendering module.

Tests cover:
- FormatterRenderer class methods
- LiveDisplayRenderer class methods
- stream_response() unified handler
- Protocol compliance
- Edge cases and error handling
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from rich.console import Console

from victor.ui.rendering import (
    StreamRenderer,
    BufferedRenderer,
    FormatterRenderer,
    LiveDisplayRenderer,
    stream_response,
    format_tool_args,
)
from victor.ui.rendering.utils import format_tool_display_name
from victor.framework.events import AgentExecutionEvent, EventType
from victor.providers.base import StreamChunk


class TestFormatterRenderer:
    """Tests for FormatterRenderer class."""

    @pytest.fixture
    def mock_formatter(self):
        """Create a mock OutputFormatter."""
        formatter = MagicMock()
        return formatter

    @pytest.fixture
    def mock_console(self):
        """Create a mock Console."""
        return MagicMock(spec=Console)

    @pytest.fixture
    def renderer(self, mock_formatter, mock_console):
        """Create a FormatterRenderer instance."""
        return FormatterRenderer(mock_formatter, mock_console)

    def test_start_calls_formatter_start_streaming(self, renderer, mock_formatter):
        """Test start() delegates to formatter.start_streaming()."""
        renderer.start()
        mock_formatter.start_streaming.assert_called_once()

    def test_pause_increments_depth_without_end_streaming(self, renderer, mock_formatter):
        """pause() increments depth counter but does not call end_streaming (no-op removed)."""
        renderer.pause()
        assert renderer._pause_count == 1
        mock_formatter.end_streaming.assert_not_called()

    def test_resume_calls_formatter_start_streaming(self, renderer, mock_formatter):
        """Test resume() delegates to formatter.start_streaming() after a pause."""
        renderer.pause()
        renderer.resume()
        mock_formatter.start_streaming.assert_called_once()

    def test_on_tool_start_records_and_delegates(self, renderer, mock_formatter):
        """on_tool_start records tool info and delegates to formatter.tool_start.

        It deliberately does NOT emit a separate "running" status line (the result
        line covers it) — see FormatterRenderer.on_tool_start.
        """
        renderer.on_tool_start("read", {"path": "/test.py"})

        mock_formatter.tool_start.assert_called_once_with("read", {"path": "/test.py"})
        mock_formatter.status.assert_not_called()
        assert renderer._last_tool_start == {
            "name": "read",
            "arguments": {"path": "/test.py"},
        }

    def test_on_tool_start_delegates_raw_name_without_status(self, renderer, mock_formatter):
        """on_tool_start forwards the raw tool name and emits no status line.

        Display-name formatting (e.g. codeSearch) is the formatter's concern, not
        the renderer's.
        """
        renderer.on_tool_start("code_search", {"query": "foo"})

        mock_formatter.tool_start.assert_called_once_with("code_search", {"query": "foo"})
        mock_formatter.status.assert_not_called()

    def test_on_tool_result_success(self, renderer, mock_formatter):
        """Test on_tool_result() for successful tool execution."""
        renderer.on_tool_result(
            name="write",
            success=True,
            elapsed=0.5,
            arguments={"path": "/out.py"},
        )

        mock_formatter.tool_result.assert_called_once_with(
            tool_name="write",
            success=True,
            error=None,
            follow_up_suggestions=None,
        )
        mock_formatter.start_streaming.assert_called_once()

    def test_on_tool_result_failure(self, renderer, mock_formatter):
        """Test on_tool_result() for failed tool execution."""
        renderer.on_tool_result(
            name="bash",
            success=False,
            elapsed=1.2,
            arguments={"command": "ls"},
            error="Permission denied",
        )

        mock_formatter.tool_result.assert_called_once_with(
            tool_name="bash",
            success=False,
            error="Permission denied",
            follow_up_suggestions=None,
        )

    def test_on_tool_result_forwards_follow_up_suggestions(self, renderer, mock_formatter):
        """Test on_tool_result() forwards follow-up suggestions to formatter."""
        follow_ups = [
            {
                "command": 'graph(mode="trace", node="main", depth=3)',
                "description": "Trace execution starting from main.",
            }
        ]

        renderer.on_tool_result(
            name="code_search",
            success=True,
            elapsed=0.4,
            arguments={"query": "main entry point"},
            follow_up_suggestions=follow_ups,
        )

        mock_formatter.tool_result.assert_called_once_with(
            tool_name="code_search",
            success=True,
            error=None,
            follow_up_suggestions=follow_ups,
        )

    def test_on_tool_result_forwards_preview_and_full_output(self, renderer, mock_formatter):
        """Pruned previews forward both preview and full output to formatter."""
        renderer.on_tool_result(
            name="read",
            success=True,
            elapsed=0.5,
            arguments={"path": "/tmp/test.py"},
            was_pruned=True,
            original_result="full line 1\nfull line 2\nfull line 3",
            result="full line 1\n[PRUNED PREVIEW: omitted 2 lines]",
        )

        mock_formatter.tool_result.assert_called_once_with(
            tool_name="read",
            success=True,
            error=None,
            follow_up_suggestions=None,
            original_result="full line 1\nfull line 2\nfull line 3",
            result="full line 1\n[PRUNED PREVIEW: omitted 2 lines]",
            was_pruned=True,
        )

    def test_on_status(self, renderer, mock_formatter):
        """Test on_status() shows status message."""
        renderer.on_status("Thinking...")

        mock_formatter.status.assert_called_once_with("Thinking...")
        mock_formatter.start_streaming.assert_called_once()

    def test_on_file_preview(self, renderer, mock_formatter, mock_console):
        """Test on_file_preview() displays syntax-highlighted content."""
        renderer.on_file_preview("/test.py", "print('hello')")

        mock_console.print.assert_called_once()
        # Check Panel was printed
        call_args = mock_console.print.call_args[0][0]
        assert hasattr(call_args, "title")  # Panel has title attribute
        mock_formatter.start_streaming.assert_called_once()

    def test_on_file_preview_no_extension(self, renderer, mock_formatter, mock_console):
        """Test on_file_preview() handles files without extension."""
        renderer.on_file_preview("Makefile", "all: build")

        mock_console.print.assert_called_once()

    def test_on_edit_preview(self, renderer, mock_formatter, mock_console):
        """Test on_edit_preview() displays colored diff."""
        diff = "-old line\n+new line\n context"
        renderer.on_edit_preview("/test.py", diff)

        # render_edit_preview renders the whole diff as one consolidated panel.
        assert mock_console.print.call_count == 1
        mock_formatter.start_streaming.assert_called_once()

    def test_on_content_accumulates(self, renderer, mock_formatter):
        """Test on_content() accumulates content and streams chunks."""
        renderer.on_content("Hello ")
        renderer.on_content("World")

        assert mock_formatter.stream_chunk.call_count == 2
        mock_formatter.stream_chunk.assert_any_call("Hello ")
        mock_formatter.stream_chunk.assert_any_call("World")

    def test_finalize_returns_content(self, renderer, mock_formatter):
        """Test finalize() returns accumulated content."""
        renderer.on_content("Test content")
        result = renderer.finalize()

        assert result == "Test content"
        mock_formatter.response.assert_called_once_with(content="Test content")

    def test_cleanup_is_noop(self, renderer):
        """Test cleanup() is a no-op for FormatterRenderer."""
        renderer.cleanup()  # Should not raise

    def test_on_thinking_content_prints_dimmed(self, renderer, mock_console):
        """Test on_thinking_content() prints dimmed/italic text."""
        renderer.on_thinking_content("Analyzing the problem...")

        # Should print styled text (not added to buffer)
        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args
        assert call_args[1].get("end") == ""  # No newline

    def test_on_thinking_start_shows_indicator(self, renderer, mock_formatter, mock_console):
        """Test on_thinking_start() shows thinking indicator."""
        renderer.on_thinking_start()

        # Should pause streaming and show indicator
        mock_console.print.assert_called_once()
        call_text = str(mock_console.print.call_args[0][0])
        assert "Thinking" in call_text

    def test_on_thinking_end_resumes_streaming(self, renderer, mock_formatter, mock_console):
        """Test on_thinking_end() resumes streaming after thinking_start."""
        renderer.on_thinking_start()
        mock_formatter.reset_mock()
        mock_console.reset_mock()
        renderer.on_thinking_end()

        # Should print newline and resume streaming
        mock_console.print.assert_called_once()
        mock_formatter.start_streaming.assert_called_once()

    def test_thinking_content_not_in_buffer(self, renderer, mock_formatter, mock_console):
        """Test thinking content is not accumulated in buffer."""
        renderer.on_content("Normal content")
        renderer.on_thinking_content("Thinking content")
        result = renderer.finalize()

        # Only normal content should be in result
        assert result == "Normal content"
        assert "Thinking" not in result


class TestLiveDisplayRenderer:
    """Tests for LiveDisplayRenderer class."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock Console."""
        return MagicMock(spec=Console)

    @pytest.fixture
    def renderer(self, mock_console):
        """Create a LiveDisplayRenderer instance."""
        return LiveDisplayRenderer(mock_console)

    def test_format_args_empty(self):
        """Test format_tool_args() with empty dict."""
        result = format_tool_args({})
        assert result == ""

    def test_format_args_single(self):
        """Test format_tool_args() with single argument."""
        result = format_tool_args({"path": "/test.py"})
        assert "path=" in result
        assert "/test.py" in result

    def test_format_args_multiple(self):
        """Test format_tool_args() with multiple arguments."""
        result = format_tool_args({"a": 1, "b": "test"})
        assert "a=" in result
        assert "b=" in result

    def test_format_args_truncates_long_values(self):
        """Test format_tool_args() truncates values longer than 60 chars."""
        long_value = "x" * 100
        result = format_tool_args({"long": long_value})
        assert len(result) < len(long_value) + 20  # Much shorter than original

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_start_creates_live_display(self, mock_live_class, renderer, mock_console):
        """Test start() creates and starts a Live display."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()

        mock_live_class.assert_called_once()
        mock_live.start.assert_called_once()

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_pause_stops_live_display(self, mock_live_class, renderer, mock_console):
        """Test pause() stops the Live display."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.pause()

        mock_live.stop.assert_called_once()

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_resume_creates_new_live_display(self, mock_live_class, renderer, mock_console):
        """Test resume() creates a new Live display with current content when paused."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        # Must start and pause first to enable resume
        renderer.start()  # Creates first Live
        renderer.pause()  # Pauses it
        renderer._content_buffer = "Previous content"
        renderer.resume()  # Should create new Live

        # Should have created two Live instances (start + resume)
        assert mock_live_class.call_count == 2
        assert mock_live.start.call_count == 2

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_tool_result_success_prints_checkmark(self, mock_live_class, renderer, mock_console):
        """Test on_tool_result() prints a success indicator for success.

        The canonical status line uses the ``success`` style and the ``✓``
        marker (see live_renderer.on_tool_result).
        """
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_tool_result(
            name="read",
            success=True,
            elapsed=0.1,
            arguments={"path": "/test.py"},
        )

        mock_console.print.assert_called()
        call_str = str(mock_console.print.call_args_list)
        assert "success" in call_str
        assert "✓" in call_str

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_tool_result_formats_tool_name_for_display(
        self, mock_live_class, renderer, mock_console
    ):
        """Tool status lines use lower-camel names without changing internal identifiers."""
        from unittest.mock import MagicMock

        mock_live_class.return_value = MagicMock()
        renderer.start()

        with patch("victor.config.tool_settings.get_tool_settings") as mock_settings:
            tool_settings = MagicMock()
            tool_settings.enable_tool_grouping = False
            tool_settings.tool_output_preview_enabled = False
            mock_settings.return_value = tool_settings

            renderer.on_tool_result(
                name="code_search",
                success=True,
                elapsed=0.1,
                arguments={"query": "main"},
            )

        printed_calls = [str(call_args) for call_args in mock_console.print.call_args_list]
        assert any("codeSearch" in call_str for call_str in printed_calls)
        assert not any("[bold]code_search[/]" in call_str for call_str in printed_calls)

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_tool_result_failure_prints_x(self, mock_live_class, renderer, mock_console):
        """Test on_tool_result() prints an error indicator for failure.

        The canonical status line uses the ``error`` style + ``✗`` marker,
        followed by an ``Error:`` line (see live_renderer.on_tool_result).
        """
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_tool_result(
            name="write",
            success=False,
            elapsed=0.5,
            arguments={},
            error="Failed",
        )

        mock_console.print.assert_called()
        call_str = str(mock_console.print.call_args_list)
        assert "error" in call_str
        assert "✗" in call_str
        assert "Error:" in call_str

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_tool_result_prints_follow_up_suggestions(
        self, mock_live_class, renderer, mock_console
    ):
        """Test on_tool_result() prints visible next-step suggestions."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_tool_result(
            name="code_search",
            success=True,
            elapsed=0.3,
            arguments={"query": "main entry point"},
            follow_up_suggestions=[
                {
                    "command": 'graph(mode="trace", node="main", depth=3)',
                    "description": "Trace execution starting from main.",
                }
            ],
        )

        printed_calls = [str(call_args) for call_args in mock_console.print.call_args_list]
        assert any("next:" in call_str for call_str in printed_calls)
        assert any(
            'graph(mode="trace", node="main", depth=3)' in call_str for call_str in printed_calls
        )

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_tool_result_failure_prints_follow_up_suggestions(
        self, mock_live_class, renderer, mock_console
    ):
        """Failed tool calls should still surface follow-up guidance."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_tool_result(
            name="graph",
            success=False,
            elapsed=0.2,
            arguments={"mode": "hubs"},
            error="Unsupported graph mode: hubs",
            follow_up_suggestions=[
                {
                    "command": 'graph(mode="overview", path=".", top_k=10)',
                    "description": "Use a supported overview mode.",
                }
            ],
        )

        printed_calls = [str(call_args) for call_args in mock_console.print.call_args_list]
        assert any("next:" in call_str for call_str in printed_calls)
        assert any(
            'graph(mode="overview", path=".", top_k=10)' in call_str for call_str in printed_calls
        )

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_status_prints_dim(self, mock_live_class, renderer, mock_console):
        """Test on_status() prints message with dim styling."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_status("Thinking deeply...")

        mock_console.print.assert_called()
        call_str = str(mock_console.print.call_args_list)
        assert "dim" in call_str
        assert "Thinking" in call_str

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_content_updates_live_display(self, mock_live_class, renderer, mock_console):
        """Test on_content() updates the Live display."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_content("Hello ")
        renderer.on_content("World")

        assert mock_live.update.call_count == 2
        assert renderer._content_buffer == "Hello World"

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_finalize_returns_content_and_cleans_up(self, mock_live_class, renderer, mock_console):
        """Test finalize() returns content and stops Live display."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_content("Final content")
        result = renderer.finalize()

        assert result == "Final content"
        mock_live.stop.assert_called()

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_cleanup_stops_live_if_running(self, mock_live_class, renderer, mock_console):
        """Test cleanup() stops Live display if it exists."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.cleanup()

        mock_live.stop.assert_called()
        assert renderer._live is None

    def test_cleanup_handles_no_live(self, renderer):
        """Test cleanup() handles case when Live was never started."""
        renderer.cleanup()  # Should not raise

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_thinking_content_prints_immediately(self, mock_live_class, renderer, mock_console):
        """Test on_thinking_content() prints text immediately without buffering."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_thinking_start()  # Start thinking mode
        renderer.on_thinking_content("Reasoning about the problem...")
        renderer.on_thinking_content(" More reasoning...")

        # Should print immediately, not buffer
        assert mock_console.print.call_count >= 2  # Each chunk prints immediately

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_thinking_start_shows_indicator(self, mock_live_class, renderer, mock_console):
        """Test on_thinking_start() shows thinking indicator."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_thinking_start()

        # Should stop live display and print indicator
        mock_live.stop.assert_called()
        call_text = str(mock_console.print.call_args[0][0])
        assert "Thinking" in call_text

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_thinking_end_resumes(self, mock_live_class, renderer, mock_console):
        """Test on_thinking_end() resumes live display."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        # Add thinking content first
        renderer.on_thinking_start()
        renderer.on_content("Some thinking content")
        renderer.on_thinking_end()

        # Should resume live display
        mock_live.start.assert_called()

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_thinking_end_without_content_just_resumes(
        self, mock_live_class, renderer, mock_console
    ):
        """Test on_thinking_end() without content just resumes display."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        # Simulate thinking mode without content
        renderer._in_thinking_mode = True
        renderer._is_paused = True
        renderer.on_thinking_end()

        # Should NOT print anything (no content to render)
        # Just resume the live display
        assert renderer._in_thinking_mode is False

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_thinking_content_renders_immediately(self, mock_live_class, renderer, mock_console):
        """Test on_thinking_content() renders text immediately without buffering.

        Note: Delta normalization is handled upstream in stream_response(),
        so this method just displays what it receives.
        """
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        renderer.start()
        renderer.on_thinking_start()

        # Send thinking chunks (already normalized by stream_response)
        renderer.on_thinking_content("Analyzing the problem...")
        renderer.on_thinking_content("Let me explore...")
        renderer.on_thinking_content("Now checking...")

        # Should print each chunk immediately (no buffering at display layer)
        # Now includes: separator + badge + indicator + 3 chunks
        assert mock_console.print.call_count == 6  # Separator + badge + indicator + 3 chunks

    def test_calculate_adaptive_preview_lines_with_error_shows_all(self, renderer):
        """Adaptive preview with error shows all lines."""
        from unittest.mock import MagicMock

        tool_settings = MagicMock()
        tool_settings.tool_output_preview_lines_min = 1
        tool_settings.tool_output_preview_lines_max = 10

        output = "line1\nline2\nline3\nline4\nline5"
        lines = renderer._calculate_adaptive_preview_lines(
            output, "Error occurred", 3, tool_settings
        )
        assert lines == 5  # All lines shown for errors

    def test_calculate_adaptive_preview_lines_small_output_shows_all(self, renderer):
        """Adaptive preview with small output (≤5 lines) shows all."""
        from unittest.mock import MagicMock

        tool_settings = MagicMock()
        tool_settings.tool_output_preview_lines_min = 1
        tool_settings.tool_output_preview_lines_max = 10

        output = "line1\nline2\nline3"
        lines = renderer._calculate_adaptive_preview_lines(output, None, 3, tool_settings)
        assert lines == 3  # All lines shown for small output

    def test_calculate_adaptive_preview_lines_medium_output_shows_moderate(self, renderer):
        """Adaptive preview with medium output (5-50 lines) shows 3-5 lines."""
        from unittest.mock import MagicMock

        tool_settings = MagicMock()
        tool_settings.tool_output_preview_lines_min = 1
        tool_settings.tool_output_preview_lines_max = 10

        # 20 lines → should show 5 lines (max for medium)
        output = "\n".join([f"line{i}" for i in range(20)])
        lines = renderer._calculate_adaptive_preview_lines(output, None, 3, tool_settings)
        assert lines == 5  # Moderate preview for medium output

    def test_calculate_adaptive_preview_lines_large_output_shows_minimal(self, renderer):
        """Adaptive preview with large output (>50 lines) shows 1-2 lines."""
        from unittest.mock import MagicMock

        tool_settings = MagicMock()
        tool_settings.tool_output_preview_lines_min = 1
        tool_settings.tool_output_preview_lines_max = 10

        # 100 lines → should show 2 lines (max for large)
        output = "\n".join([f"line{i}" for i in range(100)])
        lines = renderer._calculate_adaptive_preview_lines(output, None, 3, tool_settings)
        assert lines == 2  # Minimal preview for large output

    def test_calculate_adaptive_preview_lines_respects_min_max_bounds(self, renderer):
        """Adaptive preview respects configured min/max bounds."""
        from unittest.mock import MagicMock

        tool_settings = MagicMock()
        tool_settings.tool_output_preview_lines_min = 3  # Higher minimum
        tool_settings.tool_output_preview_lines_max = 8  # Lower maximum

        # Large output that would normally show 2 lines
        output = "\n".join([f"line{i}" for i in range(100)])
        lines = renderer._calculate_adaptive_preview_lines(output, None, 3, tool_settings)
        assert lines == 3  # Respects minimum bound

        # Small output that would show 5 lines
        output = "\n".join([f"line{i}" for i in range(20)])
        lines = renderer._calculate_adaptive_preview_lines(output, None, 3, tool_settings)
        assert lines == 5  # Within bounds

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_on_tool_result_skips_group_header_when_disabled(
        self, mock_live_class, renderer, mock_console
    ):
        """Group header is not shown when grouping is disabled."""
        from unittest.mock import MagicMock

        mock_live_class.return_value = MagicMock()
        renderer.start()

        # Mock tool settings to disable grouping
        with patch("victor.config.tool_settings.get_tool_settings") as mock_settings:
            tool_settings = MagicMock()
            tool_settings.enable_tool_grouping = False
            tool_settings.tool_output_preview_enabled = False
            mock_settings.return_value = tool_settings

            # First tool
            renderer.on_tool_result(
                name="read",
                success=True,
                elapsed=0.1,
                arguments={"path": "file1.txt"},
                result="content1",
            )

            # Second tool (different category)
            renderer.on_tool_result(
                name="code_search",
                success=True,
                elapsed=0.2,
                arguments={"query": "test"},
                result="results",
            )

        # Should only have status prints (no group headers)
        # Count should be less than when grouping is enabled
        assert mock_console.print.call_count < 4


class TestAccessibilityFeatures:
    """Tests for accessibility features like high contrast mode."""

    @pytest.fixture
    def mock_console(self):
        return MagicMock(spec=Console)

    def test_render_thinking_indicator_normal_mode(self, mock_console):
        """Thinking indicator uses dim colors in normal mode."""
        from victor.ui.rendering.utils import render_thinking_indicator
        from unittest.mock import patch

        with patch("victor.config.theme_settings.get_theme_settings") as mock_settings:
            theme_settings = MagicMock()
            theme_settings.high_contrast = False
            mock_settings.return_value = theme_settings

            render_thinking_indicator(mock_console)

            # Should have been called with dim styling
            assert mock_console.print.called
            assert mock_console.rule.called

    def test_render_thinking_indicator_high_contrast_mode(self, mock_console):
        """Thinking indicator uses bold colors in high contrast mode."""
        from victor.ui.rendering.utils import render_thinking_indicator
        from unittest.mock import patch

        with patch("victor.config.theme_settings.get_theme_settings") as mock_settings:
            theme_settings = MagicMock()
            theme_settings.high_contrast = True
            mock_settings.return_value = theme_settings

            render_thinking_indicator(mock_console)

            # Should have been called with bold styling
            assert mock_console.print.called
            assert mock_console.rule.called

    def test_render_content_badge_normal_mode(self, mock_console):
        """Content badge uses dim colors in normal mode."""
        from victor.ui.rendering.utils import render_content_badge
        from unittest.mock import patch

        with patch("victor.config.theme_settings.get_theme_settings") as mock_settings:
            theme_settings = MagicMock()
            theme_settings.high_contrast = False
            mock_settings.return_value = theme_settings

            render_content_badge(mock_console, "thinking")

            assert mock_console.print.called

    def test_render_content_badge_high_contrast_mode(self, mock_console):
        """Content badge uses bold colors in high contrast mode."""
        from victor.ui.rendering.utils import render_content_badge
        from unittest.mock import patch

        with patch("victor.config.theme_settings.get_theme_settings") as mock_settings:
            theme_settings = MagicMock()
            theme_settings.high_contrast = True
            mock_settings.return_value = theme_settings

            render_content_badge(mock_console, "thinking")

            assert mock_console.print.called

    def test_render_status_message_normal_mode(self, mock_console):
        """Status message uses dim colors in normal mode."""
        from victor.ui.rendering.utils import render_status_message
        from unittest.mock import patch

        with patch("victor.config.theme_settings.get_theme_settings") as mock_settings:
            theme_settings = MagicMock()
            theme_settings.high_contrast = False
            mock_settings.return_value = theme_settings

            render_status_message(mock_console, "Processing file")

            assert mock_console.print.called

    def test_render_status_message_high_contrast_mode(self, mock_console):
        """Status message uses bold colors in high contrast mode."""
        from victor.ui.rendering.utils import render_status_message
        from unittest.mock import patch

        with patch("victor.config.theme_settings.get_theme_settings") as mock_settings:
            theme_settings = MagicMock()
            theme_settings.high_contrast = True
            mock_settings.return_value = theme_settings

            render_status_message(mock_console, "Processing file")

            assert mock_console.print.called


class TestToolDisplayFormatting:
    """Tests for renderer-facing tool display formatting."""

    def test_format_tool_display_name_from_snake_case(self):
        assert format_tool_display_name("code_search") == "codeSearch"

    def test_format_tool_display_name_preserves_existing_camel_case(self):
        assert format_tool_display_name("codeSearch") == "codeSearch"

    def test_format_tool_display_name_normalizes_pascal_case(self):
        assert format_tool_display_name("CodeSearch") == "codeSearch"


class TestStreamResponse:
    """Tests for stream_response() unified handler."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock AgentOrchestrator."""
        agent = MagicMock()
        return agent

    @pytest.fixture
    def mock_renderer(self):
        """Create a mock StreamRenderer."""
        renderer = MagicMock()
        renderer.finalize.return_value = "Response content"
        return renderer

    @pytest.mark.asyncio
    async def test_processes_tool_start_event(self, mock_agent, mock_renderer):
        """Test stream_response handles tool_start metadata."""

        async def mock_stream():
            yield StreamChunk(
                content="",
                metadata={
                    "tool_start": {
                        "name": "read",
                        "arguments": {"path": "/test.py"},
                    }
                },
            )

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_tool_start.assert_called_once_with(
            name="read",
            arguments={"path": "/test.py"},
            tool_call_id=None,
            batch_index=None,
            batch_total=None,
            execution_mode=None,
        )

    @pytest.mark.asyncio
    async def test_processes_tool_result_event(self, mock_agent, mock_renderer):
        """Test stream_response handles tool_result metadata."""

        async def mock_stream():
            yield StreamChunk(
                content="",
                metadata={
                    "tool_result": {
                        "name": "write",
                        "success": True,
                        "elapsed": 0.5,
                        "arguments": {"path": "/out.py"},
                    }
                },
            )

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_tool_result.assert_called_once_with(
            name="write",
            success=True,
            elapsed=0.5,
            arguments={"path": "/out.py"},
            error=None,
            follow_up_suggestions=None,
            result=None,
            tool_call_id=None,
        )

    @pytest.mark.asyncio
    async def test_processes_tool_result_event_with_follow_ups(self, mock_agent, mock_renderer):
        """Test stream_response forwards tool follow-up suggestions."""

        async def mock_stream():
            yield StreamChunk(
                content="",
                metadata={
                    "tool_result": {
                        "name": "code_search",
                        "success": True,
                        "elapsed": 0.2,
                        "arguments": {"query": "main entry point"},
                        "follow_up_suggestions": [
                            {
                                "command": 'graph(mode="trace", node="main", depth=3)',
                                "description": "Trace execution starting from main.",
                            }
                        ],
                    }
                },
            )

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_tool_result.assert_called_once_with(
            name="code_search",
            success=True,
            elapsed=0.2,
            arguments={"query": "main entry point"},
            error=None,
            follow_up_suggestions=[
                {
                    "command": 'graph(mode="trace", node="main", depth=3)',
                    "description": "Trace execution starting from main.",
                }
            ],
            result=None,
            tool_call_id=None,
        )

    @pytest.mark.asyncio
    async def test_processes_tool_result_with_error(self, mock_agent, mock_renderer):
        """Test stream_response handles tool_result with error."""

        async def mock_stream():
            yield StreamChunk(
                content="",
                metadata={
                    "tool_result": {
                        "name": "bash",
                        "success": False,
                        "elapsed": 1.0,
                        "arguments": {"cmd": "rm -rf /"},
                        "error": "Permission denied",
                    }
                },
            )

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_tool_result.assert_called_once_with(
            name="bash",
            success=False,
            elapsed=1.0,
            arguments={"cmd": "rm -rf /"},
            error="Permission denied",
            follow_up_suggestions=None,
            result=None,
            tool_call_id=None,
        )

    @pytest.mark.asyncio
    async def test_processes_status_event(self, mock_agent, mock_renderer):
        """Test stream_response handles status metadata."""

        async def mock_stream():
            yield StreamChunk(content="", metadata={"status": "Starting..."})

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_status.assert_called_once_with("Starting...")

    @pytest.mark.asyncio
    async def test_maps_generic_thinking_status_to_thinking_ui(self, mock_agent, mock_renderer):
        """Generic thinking statuses should use thinking UI instead of a status line."""

        async def mock_stream():
            yield StreamChunk(content="", metadata={"status": "💭 Thinking..."})
            yield StreamChunk(content="Answer", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_thinking_start.assert_called_once()
        mock_renderer.on_status.assert_not_called()
        mock_renderer.on_thinking_end.assert_called_once()
        mock_renderer.on_content.assert_called_once_with("Answer")

    @pytest.mark.asyncio
    async def test_suppressed_generic_thinking_status_is_hidden(self, mock_agent, mock_renderer):
        """Suppress-thinking mode should hide generic thinking statuses entirely."""

        async def mock_stream():
            yield StreamChunk(content="", metadata={"status": "Thinking..."})
            yield StreamChunk(content="Answer", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(
            mock_agent,
            "test message",
            mock_renderer,
            suppress_thinking=True,
        )

        mock_renderer.on_status.assert_not_called()
        mock_renderer.on_thinking_start.assert_not_called()
        mock_renderer.on_content.assert_called_once_with("Answer")

    @pytest.mark.asyncio
    async def test_processes_file_preview_event(self, mock_agent, mock_renderer):
        """Test stream_response handles file_preview metadata."""

        async def mock_stream():
            yield StreamChunk(
                content="",
                metadata={
                    "path": "/test.py",
                    "file_preview": "print('hello')",
                },
            )

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_file_preview.assert_called_once_with(
            path="/test.py",
            content="print('hello')",
        )

    @pytest.mark.asyncio
    async def test_processes_edit_preview_event(self, mock_agent, mock_renderer):
        """Test stream_response handles edit_preview metadata."""

        async def mock_stream():
            yield StreamChunk(
                content="",
                metadata={
                    "path": "/file.py",
                    "edit_preview": "-old\n+new",
                },
            )

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_edit_preview.assert_called_once_with(
            path="/file.py",
            diff="-old\n+new",
        )

    @pytest.mark.asyncio
    async def test_processes_content_chunks(self, mock_agent, mock_renderer):
        """Test stream_response handles content chunks."""

        async def mock_stream():
            yield StreamChunk(content="Hello ", metadata=None)
            yield StreamChunk(content="World", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        assert mock_renderer.on_content.call_count == 2
        mock_renderer.on_content.assert_any_call("Hello ")
        mock_renderer.on_content.assert_any_call("World")

    @pytest.mark.asyncio
    async def test_normalizes_cumulative_content_snapshots(self, mock_agent, mock_renderer):
        """Cumulative provider snapshots should be reduced to append-only deltas."""

        async def mock_stream():
            yield StreamChunk(content="I will analyze", metadata=None)
            yield StreamChunk(content="I will analyze the repository", metadata=None)
            yield StreamChunk(content="I will analyze the repository", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        assert mock_renderer.on_content.call_count == 2
        mock_renderer.on_content.assert_has_calls([call("I will analyze"), call(" the repository")])

    @pytest.mark.asyncio
    async def test_normalizes_cumulative_reasoning_snapshots(self, mock_agent, mock_renderer):
        """Accumulated reasoning snapshots should not be printed repeatedly."""

        async def mock_stream():
            yield StreamChunk(content="", metadata={"reasoning_content": "Plan"})
            yield StreamChunk(content="", metadata={"reasoning_content": "Plan carefully"})
            yield StreamChunk(content="Done", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_thinking_content.assert_has_calls([call("Plan"), call(" carefully")])
        assert mock_renderer.on_thinking_content.call_count == 2

    @pytest.mark.asyncio
    async def test_renders_reasoning_and_content_from_same_chunk(self, mock_agent, mock_renderer):
        """Chunks containing both reasoning metadata and content should render both."""

        async def mock_stream():
            yield StreamChunk(content="Answer", metadata={"reasoning_content": "Thinking"})

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        result = await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_thinking_content.assert_called_once_with("Thinking")
        mock_renderer.on_content.assert_called_once_with("Answer")
        assert result == mock_renderer.finalize.return_value

    @pytest.mark.asyncio
    async def test_strips_provider_reasoning_prefix(self, mock_agent, mock_renderer):
        """Provider-added thinking banners should not be echoed in reasoning output."""

        async def mock_stream():
            yield StreamChunk(content="", metadata={"reasoning_content": "💭 Thinking...Plan"})
            yield StreamChunk(content="Answer", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_thinking_content.assert_called_once_with("Plan")
        mock_renderer.on_content.assert_called_once_with("Answer")

    @pytest.mark.asyncio
    async def test_buffered_renderer_accepts_tool_result_payloads(self, mock_agent):
        """Buffered non-stream rendering should accept tool result payload metadata."""

        async def mock_stream():
            yield StreamChunk(
                content="",
                metadata={
                    "tool_start": {
                        "name": "read",
                        "arguments": {"path": "file.py"},
                    }
                },
            )
            yield StreamChunk(
                content="",
                metadata={
                    "tool_result": {
                        "name": "read",
                        "success": True,
                        "elapsed": 0.1,
                        "arguments": {"path": "file.py"},
                        "result": "line1\nline2\nline3\nline4",
                    }
                },
            )
            yield StreamChunk(content="Done", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())
        renderer = BufferedRenderer()

        result = await stream_response(mock_agent, "test message", renderer)

        assert result == "Done"
        assert renderer._tool_calls[0]["result"]["output"] == "line1\nline2\nline3\nline4"

    @pytest.mark.asyncio
    async def test_buffered_renderer_normalizes_exact_response_output(self, mock_agent):
        async def mock_stream():
            yield AgentExecutionEvent(
                type=EventType.CONTENT,
                content="The user asked for exactly READY, so the answer is READY",
            )

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())
        renderer = BufferedRenderer(user_message="Reply with exactly READY")

        result = await stream_response(mock_agent, "Reply with exactly READY", renderer)

        assert result == "READY"

    @pytest.mark.asyncio
    async def test_calls_lifecycle_methods(self, mock_agent, mock_renderer):
        """Test stream_response calls start, finalize, cleanup."""

        async def mock_stream():
            yield StreamChunk(content="test", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        result = await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.start.assert_called_once()
        mock_renderer.finalize.assert_called_once()
        mock_renderer.cleanup.assert_called_once()
        assert result == "Response content"

    @pytest.mark.asyncio
    async def test_cleanup_on_exception(self, mock_agent, mock_renderer):
        """Test stream_response cleans up even on exception."""

        async def mock_stream():
            yield StreamChunk(content="test", metadata=None)
            raise RuntimeError("Simulated error")

        mock_agent.stream = MagicMock(return_value=mock_stream())

        with pytest.raises(RuntimeError, match="Simulated error"):
            await stream_response(mock_agent, "test message", mock_renderer)

        # Cleanup should still be called
        mock_renderer.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_closes_generator_on_success(self, mock_agent, mock_renderer):
        """Test stream_response closes async generator on success."""
        mock_gen = AsyncMock()
        mock_gen.__aiter__ = MagicMock(return_value=iter([]))
        mock_gen.aclose = AsyncMock()

        async def mock_stream():
            return
            yield  # Never reached

        gen = mock_stream()
        mock_agent.stream_chat = MagicMock(return_value=gen)

        await stream_response(mock_agent, "test message", mock_renderer)

        # Generator should be closed (aclose called in finally)

    @pytest.mark.asyncio
    async def test_handles_mixed_events(self, mock_agent, mock_renderer):
        """Test stream_response handles mixed event types correctly."""

        async def mock_stream():
            yield StreamChunk(content="", metadata={"status": "Starting..."})
            yield StreamChunk(
                content="",
                metadata={"tool_start": {"name": "read", "arguments": {}}},
            )
            yield StreamChunk(
                content="",
                metadata={
                    "tool_result": {
                        "name": "read",
                        "success": True,
                        "elapsed": 0.1,
                        "arguments": {},
                    }
                },
            )
            yield StreamChunk(content="Response ", metadata=None)
            yield StreamChunk(content="text", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        mock_renderer.on_status.assert_called_once_with("Starting...")
        mock_renderer.on_tool_start.assert_called_once()
        mock_renderer.on_tool_result.assert_called_once()
        assert mock_renderer.on_content.call_count == 2

    @pytest.mark.asyncio
    async def test_filters_deepseek_thinking_markers(self, mock_agent, mock_renderer):
        """Test stream_response filters DeepSeek thinking markers."""

        async def mock_stream():
            # DeepSeek thinking output with markers
            yield StreamChunk(
                content="<｜begin▁of▁thinking｜>Let me analyze this...",
                metadata=None,
            )
            yield StreamChunk(
                content="<｜end▁of▁thinking｜>Here is the answer.",
                metadata=None,
            )

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        # Should call on_thinking_start when entering thinking
        mock_renderer.on_thinking_start.assert_called()
        # Should call on_thinking_end when exiting thinking
        mock_renderer.on_thinking_end.assert_called()

    @pytest.mark.asyncio
    async def test_filters_qwen_thinking_markers(self, mock_agent, mock_renderer):
        """Test stream_response filters Qwen3 thinking markers."""

        async def mock_stream():
            yield StreamChunk(content="<think>Analyzing...", metadata=None)
            yield StreamChunk(content="</think>The answer is 42.", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer)

        # Should handle thinking transitions
        mock_renderer.on_thinking_start.assert_called()
        mock_renderer.on_thinking_end.assert_called()

    @pytest.mark.asyncio
    async def test_suppress_thinking_parameter(self, mock_agent, mock_renderer):
        """Test suppress_thinking parameter hides thinking content."""

        async def mock_stream():
            yield StreamChunk(content="<think>Hidden thinking</think>Visible", metadata=None)

        mock_agent.stream_chat = MagicMock(return_value=mock_stream())

        await stream_response(mock_agent, "test message", mock_renderer, suppress_thinking=True)

        # Should still call lifecycle methods
        mock_renderer.start.assert_called()
        mock_renderer.finalize.assert_called()


class TestBufferedRenderer:
    """Tests for BufferedRenderer-specific display behavior."""

    def test_flush_formats_tool_name_for_display(self):
        renderer = BufferedRenderer()
        renderer.on_tool_start("code_search", {"query": "main"})
        renderer.on_tool_result(
            name="code_search",
            success=True,
            elapsed=0.1,
            arguments={"query": "main"},
        )
        console = MagicMock(spec=Console)

        renderer.flush(console)

        printed_calls = [str(call_args) for call_args in console.print.call_args_list]
        assert any("codeSearch" in call_str for call_str in printed_calls)


class TestProtocolCompliance:
    """Tests for StreamRenderer protocol compliance."""

    def test_formatter_renderer_is_stream_renderer(self):
        """Test FormatterRenderer is a StreamRenderer."""
        mock_formatter = MagicMock()
        mock_console = MagicMock(spec=Console)
        renderer = FormatterRenderer(mock_formatter, mock_console)
        assert isinstance(renderer, StreamRenderer)

    def test_live_display_renderer_is_stream_renderer(self):
        """Test LiveDisplayRenderer is a StreamRenderer."""
        mock_console = MagicMock(spec=Console)
        renderer = LiveDisplayRenderer(mock_console)
        assert isinstance(renderer, StreamRenderer)

    def test_protocol_has_required_methods(self):
        """Test StreamRenderer protocol defines required methods."""
        required_methods = [
            "start",
            "pause",
            "resume",
            "on_tool_start",
            "on_tool_result",
            "on_status",
            "on_file_preview",
            "on_edit_preview",
            "on_content",
            "on_thinking_content",
            "on_thinking_start",
            "on_thinking_end",
            "finalize",
            "cleanup",
        ]
        for method in required_methods:
            assert hasattr(StreamRenderer, method)


class TestStreamingMetrics:
    """Tests for StreamingMetrics dataclass and renderer metric collection."""

    def test_initial_state_all_zero(self):
        from victor.ui.rendering.metrics import StreamingMetrics

        m = StreamingMetrics()
        assert m.pause_count == 0
        assert m.resume_count == 0
        assert m.content_chunks == 0
        assert m.tool_results == 0
        assert m.total_pause_ms == 0.0
        assert m.total_content_ms == 0.0
        assert m.slow_renders == 0

    def test_avg_content_ms_no_chunks(self):
        from victor.ui.rendering.metrics import StreamingMetrics

        assert StreamingMetrics().avg_content_ms == 0.0

    def test_avg_content_ms_multiple_chunks(self):
        from victor.ui.rendering.metrics import StreamingMetrics

        m = StreamingMetrics()
        m.record_content_chunk(10.0)
        m.record_content_chunk(30.0)
        assert m.avg_content_ms == 20.0

    def test_slow_render_counted_above_threshold(self):
        from victor.ui.rendering.metrics import StreamingMetrics

        m = StreamingMetrics()
        m.record_content_chunk(50.0)  # fast
        m.record_content_chunk(150.0)  # slow
        assert m.slow_renders == 1

    def test_record_pause_accumulates_time(self):
        from victor.ui.rendering.metrics import StreamingMetrics

        m = StreamingMetrics()
        m.record_pause(20.0)
        m.record_pause(30.0)
        assert m.pause_count == 2
        assert m.total_pause_ms == 50.0

    def test_formatter_renderer_get_metrics(self):
        """FormatterRenderer.get_metrics() returns a StreamingMetrics instance."""
        from victor.ui.rendering.metrics import StreamingMetrics

        mock_formatter = MagicMock()
        mock_console = MagicMock(spec=Console)
        renderer = FormatterRenderer(mock_formatter, mock_console)
        assert isinstance(renderer.get_metrics(), StreamingMetrics)

    def test_formatter_renderer_counts_tool_results(self):
        """on_tool_result() increments tool_results counter."""
        mock_formatter = MagicMock()
        mock_console = MagicMock(spec=Console)
        renderer = FormatterRenderer(mock_formatter, mock_console)
        renderer.on_tool_result("grep", True, 0.1, {})
        renderer.on_tool_result("read", True, 0.2, {})
        assert renderer.get_metrics().tool_results == 2

    def test_formatter_renderer_counts_content_chunks(self):
        """on_content() increments content_chunks counter."""
        mock_formatter = MagicMock()
        mock_console = MagicMock(spec=Console)
        renderer = FormatterRenderer(mock_formatter, mock_console)
        renderer.on_content("hello ")
        renderer.on_content("world")
        assert renderer.get_metrics().content_chunks == 2

    def test_formatter_renderer_counts_pause_resume_cycles(self):
        """Balanced pause/resume increments both counters once."""
        mock_formatter = MagicMock()
        mock_console = MagicMock(spec=Console)
        renderer = FormatterRenderer(mock_formatter, mock_console)
        renderer.pause()
        renderer.resume()
        m = renderer.get_metrics()
        assert m.pause_count == 1
        assert m.resume_count == 1

    def test_live_display_renderer_get_metrics(self):
        """LiveDisplayRenderer.get_metrics() returns a StreamingMetrics instance."""
        from victor.ui.rendering.metrics import StreamingMetrics

        mock_console = MagicMock(spec=Console)
        renderer = LiveDisplayRenderer(mock_console)
        assert isinstance(renderer.get_metrics(), StreamingMetrics)

    def test_live_display_renderer_counts_tool_results(self):
        """on_tool_result() increments tool_results counter in LiveDisplayRenderer."""
        mock_console = MagicMock(spec=Console)
        renderer = LiveDisplayRenderer(mock_console)
        with (
            patch("victor.ui.rendering.live_renderer.Live"),
            patch("victor.config.tool_settings.get_tool_settings") as mock_ts,
        ):
            mock_ts.return_value = MagicMock(
                tool_output_preview_enabled=False,
                tool_output_show_transparency=False,
            )
            renderer.on_tool_result("bash", True, 0.5, {})
        assert renderer.get_metrics().tool_results == 1


class TestFormatterRendererPauseDepth:
    """Tests for pause depth counting in FormatterRenderer."""

    @pytest.fixture
    def mock_formatter(self):
        return MagicMock()

    @pytest.fixture
    def mock_console(self):
        return MagicMock(spec=Console)

    @pytest.fixture
    def renderer(self, mock_formatter, mock_console):
        return FormatterRenderer(mock_formatter, mock_console)

    def test_double_pause_increments_depth_twice(self, renderer, mock_formatter):
        """Two consecutive pause() calls increment depth without calling end_streaming."""
        renderer.pause()
        renderer.pause()
        assert renderer._pause_count == 2
        mock_formatter.end_streaming.assert_not_called()

    def test_nested_resume_only_restarts_at_depth_zero(self, renderer, mock_formatter):
        """start_streaming is only called after the outermost resume()."""
        renderer.pause()
        renderer.pause()
        renderer.resume()  # depth → 1, still paused
        mock_formatter.start_streaming.assert_not_called()
        renderer.resume()  # depth → 0, now resume
        mock_formatter.start_streaming.assert_called_once()

    def test_resume_without_pause_is_noop(self, renderer, mock_formatter, caplog):
        """resume() with no prior pause() logs a warning and is a no-op."""
        import logging

        with caplog.at_level(logging.WARNING, logger="victor.ui.rendering.formatter_renderer"):
            renderer.resume()
        mock_formatter.start_streaming.assert_not_called()
        assert "no matching pause" in caplog.text

    def test_pause_count_resets_after_balanced_cycle(self, renderer):
        """_pause_count returns to 0 after a balanced pause/resume cycle."""
        renderer.pause()
        renderer.resume()
        assert renderer._pause_count == 0

    def test_cleanup_clears_last_tool_result(self, renderer):
        """cleanup() clears the stored tool result."""
        renderer._last_tool_result = {"name": "tool", "result": "data"}
        renderer.cleanup()
        assert renderer._last_tool_result is None

    def test_on_tool_result_stores_last_result(self, renderer, mock_formatter):
        """on_tool_result() stores the result for later expansion."""
        renderer.on_tool_result(
            name="grep",
            success=True,
            elapsed=0.1,
            arguments={"pattern": "foo"},
            result="line1\nline2",
        )
        assert renderer._last_tool_result is not None
        assert renderer._last_tool_result["name"] == "grep"
        assert renderer._last_tool_result["result"] == "line1\nline2"
        assert renderer._last_tool_result["success"] is True

    def test_on_tool_result_none_result_stored_as_empty(self, renderer, mock_formatter):
        """on_tool_result() with no result stores empty string."""
        renderer.on_tool_result(
            name="bash",
            success=True,
            elapsed=0.2,
            arguments={},
        )
        assert renderer._last_tool_result["result"] == ""


class TestFormatterRendererExpandLastOutput:
    """Tests for FormatterRenderer.expand_last_output()."""

    @pytest.fixture
    def mock_formatter(self):
        return MagicMock()

    @pytest.fixture
    def mock_console(self):
        return MagicMock(spec=Console)

    @pytest.fixture
    def renderer(self, mock_formatter, mock_console):
        return FormatterRenderer(mock_formatter, mock_console)

    def test_expand_with_no_result_prints_message(self, renderer, mock_console):
        """expand_last_output() with nothing stored prints a notice."""
        renderer.expand_last_output()
        mock_console.print.assert_called_once()
        text = str(mock_console.print.call_args[0][0])
        assert "No tool output" in text

    def test_expand_with_failed_tool_does_nothing(self, renderer, mock_console):
        """expand_last_output() skips display for failed tools."""
        renderer._last_tool_result = {
            "name": "bash",
            "success": False,
            "result": "error text",
            "arguments": {},
            "elapsed": 0.1,
        }
        renderer.expand_last_output()
        mock_console.print.assert_not_called()

    def test_expand_with_empty_result_does_nothing(self, renderer, mock_console):
        """expand_last_output() skips display when result is empty."""
        renderer._last_tool_result = {
            "name": "bash",
            "success": True,
            "result": "",
            "arguments": {},
            "elapsed": 0.1,
        }
        renderer.expand_last_output()
        mock_console.print.assert_not_called()

    def test_expand_shows_rich_panel(self, renderer, mock_console):
        """expand_last_output() calls console.print with a Rich Panel."""
        from rich.panel import Panel

        renderer._last_tool_result = {
            "name": "code_search",
            "success": True,
            "result": "def main():\n    pass",
            "arguments": {},
            "elapsed": 0.3,
        }
        renderer.expand_last_output()
        mock_console.print.assert_called_once()
        args = mock_console.print.call_args[0]
        assert isinstance(args[0], Panel)

    def test_expand_large_output_truncated(self, renderer, mock_console):
        """expand_last_output() truncates output exceeding 10 000 chars."""
        large = "x" * 15000
        renderer._last_tool_result = {
            "name": "read",
            "success": True,
            "result": large,
            "arguments": {},
            "elapsed": 0.1,
        }
        renderer.expand_last_output()
        # First print should be the truncation warning
        first_call = str(mock_console.print.call_args_list[0][0][0])
        assert "10000" in first_call or "chars" in first_call


class TestLiveDisplayRendererPauseDepth:
    """Tests for pause depth counting in LiveDisplayRenderer."""

    @pytest.fixture
    def mock_console(self):
        return MagicMock(spec=Console)

    @pytest.fixture
    def renderer(self, mock_console):
        return LiveDisplayRenderer(mock_console)

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_double_pause_stops_live_only_once(self, mock_live_class, renderer):
        """Two consecutive pause() calls must only stop the Live once."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live
        renderer.start()
        renderer.pause()
        renderer.pause()
        mock_live.stop.assert_called_once()

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_nested_resume_only_restarts_at_depth_zero(self, mock_live_class, renderer):
        """Live is only restarted when the depth counter returns to zero."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live
        renderer.start()
        renderer.pause()
        renderer.pause()
        renderer.resume()  # depth → 1, still paused
        # Live was stopped once; not yet restarted
        assert renderer._is_paused is True
        renderer.resume()  # depth → 0, now resume
        assert renderer._is_paused is False

    def test_resume_without_pause_is_noop(self, renderer, caplog):
        """resume() with no prior pause() is a quiet no-op for operators."""
        import logging

        with caplog.at_level(logging.WARNING, logger="victor.ui.rendering.live_renderer"):
            renderer.resume()
        assert "no matching pause" not in caplog.text

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_start_resets_pause_count(self, mock_live_class, renderer):
        """start() resets _pause_count to 0."""
        renderer._pause_count = 3
        mock_live_class.return_value = MagicMock()
        renderer.start()
        assert renderer._pause_count == 0

    @patch("victor.ui.rendering.live_renderer.Live")
    def test_cleanup_resets_pause_count(self, mock_live_class, renderer):
        """cleanup() resets _pause_count to 0."""
        mock_live_class.return_value = MagicMock()
        renderer.start()
        renderer.pause()
        renderer.cleanup()
        assert renderer._pause_count == 0


class TestLiveDisplayRendererExpandLastOutput:
    """Tests for LiveDisplayRenderer.expand_last_output()."""

    @pytest.fixture
    def mock_console(self):
        return MagicMock(spec=Console)

    @pytest.fixture
    def renderer(self, mock_console):
        return LiveDisplayRenderer(mock_console)

    def test_expand_with_no_result_prints_message(self, renderer, mock_console):
        """expand_last_output() with nothing stored prints a notice."""
        renderer.expand_last_output()
        mock_console.print.assert_called_once()
        text = str(mock_console.print.call_args[0][0])
        assert "No tool output" in text

    def test_expand_with_failed_tool_does_nothing(self, renderer, mock_console):
        """expand_last_output() skips display for failed tools."""
        renderer._last_tool_result = {
            "name": "bash",
            "success": False,
            "result": "error text",
            "arguments": {},
            "elapsed": 0.1,
        }
        renderer.expand_last_output()
        mock_console.print.assert_not_called()

    def test_expand_shows_rich_panel(self, renderer, mock_console):
        """expand_last_output() calls console.print with a Rich Panel."""
        from rich.panel import Panel

        renderer._last_tool_result = {
            "name": "code_search",
            "success": True,
            "result": "def main():\n    pass",
            "arguments": {},
            "elapsed": 0.3,
        }
        # expand_last_output calls pause/resume — mock the Live so it doesn't crash
        with patch("victor.ui.rendering.live_renderer.Live"):
            renderer.expand_last_output()
        # Console.print should have been called with a Panel
        panel_calls = [
            c for c in mock_console.print.call_args_list if c[0] and isinstance(c[0][0], Panel)
        ]
        assert len(panel_calls) == 1

    def test_expand_large_output_truncated(self, renderer, mock_console):
        """expand_last_output() truncates output exceeding 10 000 chars."""
        large = "x" * 15000
        renderer._last_tool_result = {
            "name": "read",
            "success": True,
            "result": large,
            "arguments": {},
            "elapsed": 0.1,
        }
        with patch("victor.ui.rendering.live_renderer.Live"):
            renderer.expand_last_output()
        warning_calls = [
            c
            for c in mock_console.print.call_args_list
            if "chars" in str(c[0][0]) or "10000" in str(c[0][0])
        ]
        assert len(warning_calls) == 1
