"""Tests for progressive tool-output rendering (Track B)."""

from __future__ import annotations

import asyncio
import io
from typing import Optional, Any

from rich.console import Console

import victor.framework.tool_progress as tp
from victor.framework.events import AgentExecutionEvent, EventType, tool_progress_event
from victor.ui.rendering.buffered import BufferedRenderer
from victor.ui.rendering.formatter_renderer import FormatterRenderer
from victor.ui.rendering.handler import stream_response
from victor.ui.rendering.live_renderer import LiveDisplayRenderer
from victor.ui.rendering.utils import Severity
from victor.ui.theme import victor_theme


class _ProgressRecorder:
    """Minimal StreamRenderer recording progress calls."""

    def __init__(self) -> None:
        self.progress: list[dict[str, Any]] = []
        self.content: list[str] = []
        self.tool_calls: list[str] = []

    def start(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        self.tool_calls.append(name)

    def on_tool_progress(self, name, stdout="", stderr="", progress=0.0, is_final=False) -> None:
        self.progress.append(
            {"name": name, "stdout": stdout, "stderr": stderr, "is_final": is_final}
        )

    def on_tool_result(self, **kwargs: Any) -> None: ...
    def on_status(self, message: str, severity: Optional["Severity"] = None) -> None: ...
    def on_file_preview(self, path: str, content: str) -> None: ...
    def on_edit_preview(self, path: str, diff: str) -> None: ...
    def on_content(self, text: str) -> None:
        self.content.append(text)

    def on_thinking_content(self, text: str) -> None: ...
    def on_thinking_start(self) -> None: ...
    def on_thinking_end(self) -> None: ...
    def had_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def finalize(self) -> str:
        return "".join(self.content)

    def cleanup(self) -> None: ...


class _AgentStreamSource:
    def __init__(self, events: list[AgentExecutionEvent]) -> None:
        self._events = events

    async def stream(self, _message: str):
        for event in self._events:
            yield event


def teardown_function(_fn):
    tp.clear_progress_sink()


def test_handler_dispatches_tool_progress_to_renderer():
    recorder = _ProgressRecorder()
    events = [
        tool_progress_event("shell", stdout="building...\n", progress=0.3),
        AgentExecutionEvent(type=EventType.CONTENT, content="done"),
    ]
    asyncio.run(stream_response(_AgentStreamSource(events), "go", recorder))

    assert len(recorder.progress) == 1
    assert recorder.progress[0]["name"] == "shell"
    assert "building" in recorder.progress[0]["stdout"]


def test_handler_registers_and_clears_sink_around_turn():
    recorder = _ProgressRecorder()
    events = [AgentExecutionEvent(type=EventType.CONTENT, content="hi")]
    # No sink before the turn.
    assert tp.has_progress_sink() is False
    asyncio.run(stream_response(_AgentStreamSource(events), "go", recorder))
    # Sink is cleared after the turn regardless of outcome.
    assert tp.has_progress_sink() is False


def test_live_renderer_progress_does_not_raise_and_clears_panel():
    console = Console(theme=victor_theme, file=io.StringIO(), force_terminal=True, width=80)
    renderer = LiveDisplayRenderer(console)
    renderer.start()
    try:
        renderer.on_tool_start("shell", {"cmd": "make test"})
        renderer.on_tool_progress("shell", stdout="line 1\n")
        renderer.on_tool_progress("shell", stdout="line 2\n", is_final=False)
        assert renderer._tool_progress_active is True
        # Result tears the live panel down so it doesn't freeze into scrollback.
        renderer.on_tool_result(
            name="shell",
            success=True,
            elapsed=0.4,
            arguments={},
            result="line 1\nline 2\n",
        )
        assert renderer._tool_progress_active is False
        assert len(renderer._tool_progress_lines) == 0
    finally:
        renderer.cleanup()


def test_live_renderer_progress_noop_when_not_live():
    console = Console(theme=victor_theme, file=io.StringIO(), force_terminal=True, width=80)
    renderer = LiveDisplayRenderer(console)
    # Never started -> no Live display -> must be a safe no-op.
    renderer.on_tool_progress("shell", stdout="x")
    assert renderer._tool_progress_active is False


def test_shell_streams_progress_when_sink_active(monkeypatch):
    """The shell tool emits live progress yet still returns full stdout.

    Forces the async (non-cached) execution path so streaming applies; read-only
    commands otherwise go through the synchronous cache and need no streaming.
    """
    from victor.tools import bash
    from victor.tools.bash import shell

    monkeypatch.setattr(bash, "_is_readonly_command", lambda _cmd: False)

    seen: list[str] = []
    tp.set_progress_sink(lambda **kw: seen.append(kw.get("stdout", "")))
    try:
        result = asyncio.run(shell(cmd="printf 'alpha\\nbeta\\n'", readonly=False))
    finally:
        tp.clear_progress_sink()

    # Final result contract is unchanged...
    assert result["success"] is True
    assert "alpha" in result["stdout"] and "beta" in result["stdout"]
    # ...and progress was streamed to the sink.
    assert "".join(seen).count("alpha") >= 1


def test_shell_does_not_stream_without_sink(monkeypatch):
    """Without a sink the shell tool uses the plain path and emits nothing."""
    from victor.tools import bash
    from victor.tools.bash import shell

    monkeypatch.setattr(bash, "_is_readonly_command", lambda _cmd: False)
    tp.clear_progress_sink()
    result = asyncio.run(shell(cmd="printf 'plain\\n'", readonly=False))
    assert result["success"] is True
    assert "plain" in result["stdout"]


def test_buffered_and_formatter_renderers_have_noop_progress():
    buffered = BufferedRenderer()
    buffered.on_tool_progress("shell", stdout="x")  # no raise

    # on_tool_progress is a pure no-op (no self access), so an uninitialized
    # instance is enough to exercise the method without OutputFormatter deps.
    formatter = object.__new__(FormatterRenderer)
    formatter.on_tool_progress("shell", stdout="x")  # no raise
