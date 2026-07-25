"""Renderer conformance: every StreamRenderer survives the same event script.

The interactive (LiveDisplayRenderer), plain (FormatterRenderer), and
non-interactive (BufferedRenderer) paths share stream_response() and
EventDispatcher; only the renderer differs. That makes the renderers the one
place behavior can silently diverge — a defect in the least-exercised one
(BufferedRenderer's str-arguments crash, session
modality-doc-review-fixes-b4e87728) surfaces only in `victor run`.

This suite drives an identical, deliberately hostile event script through
every renderer implementation so protocol regressions fail everywhere at once.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from victor.ui.rendering.buffered import BufferedRenderer


def _renderers():
    """Instantiate every production StreamRenderer implementation."""
    from rich.console import Console

    from victor.ui.output_formatter import create_formatter
    from victor.ui.rendering import FormatterRenderer, LiveDisplayRenderer

    console = Console(file=MagicMock(), force_terminal=False, width=100)
    return [
        ("buffered", BufferedRenderer()),
        ("formatter", FormatterRenderer(create_formatter(), console)),
        ("live", LiveDisplayRenderer(console)),
    ]


def _drive_hostile_script(renderer) -> None:
    """The shared event script: valid events plus known-hostile shapes."""
    renderer.start()
    try:
        renderer.on_status("Starting")
        # Hostile: arguments as raw JSON string (models do this)
        renderer.on_tool_start(name="shell", arguments='{"cmd": "ls docs/"}')
        renderer.on_tool_result(
            name="shell",
            success=True,
            elapsed=0.42,
            arguments='{"cmd": "ls docs/"}',
            result="a\nb\nc",
        )
        # Hostile: arguments as non-JSON string
        renderer.on_tool_start(name="read", arguments="path=weird raw string")
        renderer.on_tool_result(
            name="read",
            success=False,
            elapsed=0.1,
            arguments="path=weird raw string",
            error="File not found: nope.txt",
        )
        renderer.on_thinking_content("thinking...")
        renderer.on_content("Hello ")
        renderer.on_content("world")
        # Error surfacing (EventDispatcher routes ERROR events here)
        renderer.on_status("❌ upstream status 400")
        final = renderer.finalize()
        assert isinstance(final, str)
        assert renderer.had_tool_calls() is True
    finally:
        renderer.cleanup()


@pytest.mark.parametrize("name_renderer", _renderers(), ids=lambda nr: nr[0])
def test_renderer_survives_hostile_event_script(name_renderer):
    name, renderer = name_renderer
    _drive_hostile_script(renderer)


def test_buffered_flush_survives_hostile_script_and_shows_error():
    renderer = BufferedRenderer()
    _drive_hostile_script(renderer)
    console = MagicMock()
    renderer.flush(console)
    printed = " ".join(str(call.args[0]) for call in console.print.call_args_list)
    assert "upstream status 400" in printed
