"""Unit tests for BufferedRenderer.

Tests verify that the buffered renderer correctly collects streaming events
and renders them at completion, including tool calls, reasoning, content,
and the flush() output path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from victor.ui.rendering.buffered import BufferedRenderer


@pytest.fixture
def renderer() -> BufferedRenderer:
    return BufferedRenderer()


@pytest.fixture
def renderer_with_reasoning() -> BufferedRenderer:
    return BufferedRenderer(show_reasoning=True)


@pytest.fixture
def renderer_plain() -> BufferedRenderer:
    return BufferedRenderer(plain=True)


@pytest.fixture
def renderer_with_user_msg() -> BufferedRenderer:
    return BufferedRenderer(user_message="Hello, what is 2+2?")


class TestLifecycle:
    """Lifecycle methods are no-ops for buffered renderer."""

    def test_start(self, renderer: BufferedRenderer) -> None:
        renderer.start()

    def test_pause(self, renderer: BufferedRenderer) -> None:
        renderer.pause()

    def test_resume(self, renderer: BufferedRenderer) -> None:
        renderer.resume()

    def test_cleanup(self, renderer: BufferedRenderer) -> None:
        renderer.cleanup()


class TestToolCalls:
    """Tool call recording."""

    def test_on_tool_start_records_call(self, renderer: BufferedRenderer) -> None:
        renderer.on_tool_start("search", {"query": "test"})
        assert len(renderer._tool_calls) == 1
        assert renderer._tool_calls[0]["name"] == "search"
        assert renderer._tool_calls[0]["arguments"] == {"query": "test"}
        assert renderer._tool_calls[0]["result"] is None

    def test_on_tool_start_multiple_calls(self, renderer: BufferedRenderer) -> None:
        renderer.on_tool_start("search", {"q": "1"})
        renderer.on_tool_start("read", {"path": "f.py"})
        assert len(renderer._tool_calls) == 2

    def test_had_tool_calls_false(self, renderer: BufferedRenderer) -> None:
        assert renderer.had_tool_calls() is False

    def test_had_tool_calls_true(self, renderer: BufferedRenderer) -> None:
        renderer.on_tool_start("search", {})
        assert renderer.had_tool_calls() is True

    def test_on_tool_progress_is_noop(self, renderer: BufferedRenderer) -> None:
        renderer.on_tool_progress("search", stdout="progress")
        assert len(renderer._tool_calls) == 0


class TestToolResults:
    """Tool result recording."""

    def test_on_tool_result_matches_last_call(self, renderer: BufferedRenderer) -> None:
        renderer.on_tool_start("search", {"q": "1"})
        renderer.on_tool_result("search", True, 1.5, {"q": "1"}, result="found")
        assert renderer._tool_calls[0]["result"] is not None
        assert renderer._tool_calls[0]["result"]["success"] is True
        assert renderer._tool_calls[0]["result"]["output"] == "found"

    def test_on_tool_result_with_error(self, renderer: BufferedRenderer) -> None:
        renderer.on_tool_start("search", {"q": "x"})
        renderer.on_tool_result("search", False, 0.5, {"q": "x"}, error="not found")
        result = renderer._tool_calls[0]["result"]
        assert result["success"] is False
        assert result["error"] == "not found"

    def test_on_tool_result_with_pruned(self, renderer: BufferedRenderer) -> None:
        renderer.on_tool_start("read", {"path": "big.txt"})
        renderer.on_tool_result(
            "read", True, 0.1, {"path": "big.txt"}, was_pruned=True, result="small"
        )
        assert renderer._tool_calls[0]["result"]["was_pruned"] is True

    def test_on_tool_result_with_original(self, renderer: BufferedRenderer) -> None:
        renderer.on_tool_start("read", {"path": "f.py"})
        renderer.on_tool_result(
            "read",
            True,
            0.1,
            {"path": "f.py"},
            original_result="full",
            result="truncated",
        )
        assert renderer._tool_calls[0]["result"]["full_output"] == "full"
        assert renderer._tool_calls[0]["result"]["output"] == "truncated"

    def test_on_tool_result_no_result(self, renderer: BufferedRenderer) -> None:
        renderer.on_tool_start("search", {})
        renderer.on_tool_result("search", True, 0.1, {})
        assert renderer._tool_calls[0]["result"]["output"] == ""

    def test_on_tool_result_matches_reversed(self, renderer: BufferedRenderer) -> None:
        renderer.on_tool_start("search", {"q": "1"})
        renderer.on_tool_start("search", {"q": "2"})
        renderer.on_tool_result("search", True, 0.1, {"q": "2"}, result="second")
        assert renderer._tool_calls[1]["result"]["output"] == "second"
        assert renderer._tool_calls[0]["result"] is None

    def test_on_tool_result_no_matching_call(self, renderer: BufferedRenderer) -> None:
        # on_tool_result intentionally records orphan results (no prior
        # on_tool_start) as synthetic tool entries via its for...else fallback,
        # so they still appear in the flush summary.
        renderer.on_tool_result("nonexistent", True, 0.1, {})
        assert len(renderer._tool_calls) == 1
        assert renderer._tool_calls[0]["name"] == "nonexistent"
        assert renderer._tool_calls[0]["result"] is not None
        assert renderer._tool_calls[0]["result"]["success"] is True


class TestContent:
    """Content buffering."""

    def test_on_content_appends(self, renderer: BufferedRenderer) -> None:
        renderer.on_content("Hello")
        renderer.on_content(" World")
        assert "".join(renderer._content_chunks) == "Hello World"

    def test_on_thinking_content_no_reasoning(self, renderer: BufferedRenderer) -> None:
        renderer.on_thinking_content("thinking...")
        assert len(renderer._reasoning_chunks) == 0

    def test_on_thinking_content_with_reasoning(
        self, renderer_with_reasoning: BufferedRenderer
    ) -> None:
        renderer_with_reasoning.on_thinking_content("thinking...")
        assert "".join(renderer_with_reasoning._reasoning_chunks) == "thinking..."

    def test_on_thinking_start_end(self, renderer: BufferedRenderer) -> None:
        renderer.on_thinking_start()
        renderer.on_thinking_end()

    def test_on_file_preview(self, renderer: BufferedRenderer) -> None:
        renderer.on_file_preview("/path/to/file.py", "content")
        assert len(renderer._content_chunks) == 0

    def test_on_edit_preview(self, renderer: BufferedRenderer) -> None:
        renderer.on_edit_preview("/path/to/file.py", "diff")
        assert len(renderer._content_chunks) == 0

    def test_on_status(self, renderer: BufferedRenderer) -> None:
        renderer.on_status("Processing...")
        assert len(renderer._statuses) == 1
        assert renderer._statuses[0] == "Processing..."


class TestFinalize:
    """finalize() returns accumulated content."""

    def test_finalize_empty(self, renderer: BufferedRenderer) -> None:
        assert renderer.finalize() == ""

    def test_finalize_with_content(self, renderer: BufferedRenderer) -> None:
        renderer.on_content("Hello")
        renderer.on_content(" World")
        assert renderer.finalize() == "Hello World"

    def test_finalize_does_not_clear(self, renderer: BufferedRenderer) -> None:
        renderer.on_content("Hello")
        renderer.finalize()
        assert renderer.finalize() == "Hello"

    def test_finalize_with_user_message(self, renderer_with_user_msg: BufferedRenderer) -> None:
        renderer_with_user_msg.on_content("The answer is 4")
        result = renderer_with_user_msg.finalize()
        assert "4" in result


class TestFlush:
    """flush() prints collected output to console."""

    def test_flush_empty(self, renderer: BufferedRenderer) -> None:
        console = MagicMock()
        renderer.flush(console)
        console.print.assert_not_called()

    def test_flush_with_content_plain(self, renderer_plain: BufferedRenderer) -> None:
        console = MagicMock()
        renderer_plain.on_content("Hello World")
        renderer_plain.flush(console)
        console.print.assert_called_once_with("Hello World")

    def test_flush_with_content_markdown(self, renderer: BufferedRenderer) -> None:
        console = MagicMock()
        renderer.on_content("Hello World")
        # flush() does a function-local `from rich.markdown import Markdown`,
        # so patch the source module rather than the buffered namespace.
        with patch("rich.markdown.Markdown") as mock_md:
            renderer.flush(console)
            mock_md.assert_called_once_with("Hello World")
            console.print.assert_called_once()

    def test_flush_with_tool_calls(self, renderer: BufferedRenderer) -> None:
        console = MagicMock()
        renderer.on_tool_start("search", {"q": "test"})
        renderer.on_tool_result("search", True, 1.5, {"q": "test"}, result="found")
        with patch(
            "victor.ui.rendering.buffered.format_tool_display_name",
            return_value="Search",
        ):
            with patch("victor.ui.rendering.buffered.format_duration", return_value="1.5s"):
                with patch(
                    "victor.ui.rendering.buffered.format_tool_args",
                    return_value="q=test",
                ):
                    with patch("victor.ui.rendering.buffered.render_tool_preview"):
                        renderer.flush(console)
        assert console.print.call_count >= 1

    def test_flush_with_pending_tool(self, renderer: BufferedRenderer) -> None:
        console = MagicMock()
        renderer.on_tool_start("search", {"q": "test"})
        with patch(
            "victor.ui.rendering.buffered.format_tool_display_name",
            return_value="Search",
        ):
            renderer.flush(console)
        pending_calls = [c for c in console.print.call_args_list if "pending" in str(c)]
        assert len(pending_calls) >= 1

    def test_flush_with_reasoning(self, renderer_with_reasoning: BufferedRenderer) -> None:
        console = MagicMock()
        renderer_with_reasoning.on_thinking_content("step 1")
        renderer_with_reasoning.flush(console)
        console.print.assert_called_once()

    def test_flush_tool_with_error(self, renderer: BufferedRenderer) -> None:
        console = MagicMock()
        renderer.on_tool_start("search", {"q": "x"})
        renderer.on_tool_result("search", False, 0.5, {"q": "x"}, error="timeout")
        with patch(
            "victor.ui.rendering.buffered.format_tool_display_name",
            return_value="Search",
        ):
            renderer.flush(console)
        assert console.print.call_count >= 1


class TestEdgeCases:
    """Edge cases."""

    def test_no_user_message(self, renderer: BufferedRenderer) -> None:
        renderer.on_content("Hello")
        assert renderer.finalize() == "Hello"

    def test_multiple_tool_calls_mixed_results(self, renderer: BufferedRenderer) -> None:
        renderer.on_tool_start("search", {"q": "1"})
        renderer.on_tool_start("read", {"path": "f.py"})
        renderer.on_tool_result("search", True, 0.1, {"q": "1"}, result="found")
        renderer.on_tool_result("read", False, 0.2, {"path": "f.py"}, error="missing")
        assert renderer._tool_calls[0]["result"]["success"] is True
        assert renderer._tool_calls[1]["result"]["success"] is False

    def test_on_status_multiple(self, renderer: BufferedRenderer) -> None:
        renderer.on_status("Step 1")
        renderer.on_status("Step 2")
        assert renderer._statuses == ["Step 1", "Step 2"]

    def test_finalize_with_empty_user_message(self) -> None:
        r = BufferedRenderer(user_message="")
        r.on_content("Hello")
        assert r.finalize() == "Hello"


class TestStatusFlush:
    """Statuses (surfaced provider errors) must appear in flush output."""

    def test_flush_prints_statuses(self, renderer: BufferedRenderer) -> None:
        renderer.on_status("❌ upstream status 400")
        console = MagicMock()
        renderer.flush(console)
        printed = " ".join(str(call.args[0]) for call in console.print.call_args_list)
        assert "upstream status 400" in printed

    def test_statuses_accessor_returns_copy(self, renderer: BufferedRenderer) -> None:
        renderer.on_status("one")
        statuses = renderer.statuses()
        statuses.append("two")
        assert renderer.statuses() == ["one"]


class TestStringArgumentsRegression:
    """flush() must not crash when tool arguments arrived as a raw string.

    Regression: session modality-doc-review-fixes-b4e87728 — a recovery-produced
    tool call carried arguments as a JSON string; format_tool_args crashed with
    AttributeError ('str' object has no attribute 'items'), replacing the real
    provider error in `victor run` output.
    """

    def test_flush_with_string_arguments(self, renderer: BufferedRenderer) -> None:
        renderer.on_tool_start(name="shell", arguments='{"cmd": "ls docs/"}')
        renderer.on_tool_result(
            name="shell",
            success=True,
            elapsed=0.1,
            arguments='{"cmd": "ls docs/"}',
            result="file1\nfile2",
        )
        console = MagicMock()
        renderer.flush(console)  # must not raise
        printed = " ".join(str(call.args[0]) for call in console.print.call_args_list)
        assert "shell" in printed.lower()
