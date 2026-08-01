"""Tests for the TUI diff pane — pure extractor + Textual widget + app wiring."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict

from textual.app import App, ComposeResult

from victor.ui.chat_app.event_mapping import RenderAction, RenderKind
from victor.ui.tui.app import VictorTUIApp
from victor.ui.tui.diff_pane import DiffPane, EditDiff, extract_edit_diff, is_edit_tool

# ── pure extractor ────────────────────────────────────────────────


def test_is_edit_tool_recognizes_edit_family() -> None:
    assert is_edit_tool("edit")
    assert is_edit_tool("edit_file")
    assert is_edit_tool("patch")
    assert is_edit_tool("replace_in_file")
    assert not is_edit_tool("read")
    assert not is_edit_tool("shell")


def test_extract_uses_result_diff() -> None:
    result = json.dumps({"diff": "--- a.py\n+++ a.py\n@@ -1,2 +1,2 @@\n-old\n+new"})
    edit = extract_edit_diff("edit", {"path": "a.py"}, result)
    assert edit is not None
    assert edit.path == "a.py"
    assert any("+new" in line for line in edit.lines)
    assert any("-old" in line for line in edit.lines)


def test_extract_falls_back_to_difflib_over_args() -> None:
    edit = extract_edit_diff(
        "edit",
        {"old_str": "a\nb\n", "new_str": "a\nc\n", "path": "x.py"},
        "",
    )
    assert edit is not None
    assert edit.path == "x.py"
    assert any(line.startswith("+") for line in edit.lines)
    assert any(line.startswith("-") for line in edit.lines)


def test_extract_returns_none_for_non_edit_tool() -> None:
    assert extract_edit_diff("read", {"path": "a.py"}, "file contents") is None


def test_extract_returns_none_when_no_diff_available() -> None:
    # An edit result with only a summary (no diff, no replace args) → nothing to show.
    result = json.dumps({"operations_applied": 3})
    assert extract_edit_diff("edit", {}, result) is None


# ── DiffPane widget ───────────────────────────────────────────────


class _PaneHost(App[None]):
    def compose(self) -> ComposeResult:
        yield DiffPane(id="dp")


async def test_diffpane_add_reveals_and_cycles() -> None:
    app = _PaneHost()
    async with app.run_test() as pilot:
        pane = app.query_one("#dp", DiffPane)
        assert pane.has_edits is False

        pane.add_edit(EditDiff(path="a.py", stats="+1 -1", lines=["-old", "+new"]))
        await pilot.pause()
        assert pane.has_edits is True
        assert pane.display is True

        pane.add_edit(EditDiff(path="b.py", lines=["+x"]))
        await pilot.pause()
        assert pane._index == 1  # newest focused

        pane.cycle()  # wraps back to the first
        await pilot.pause()
        assert pane._index == 0

        pane.clear_edits()
        await pilot.pause()
        assert pane.has_edits is False
        assert pane.display is False


# ── app integration ───────────────────────────────────────────────


class _FakeClient:
    model = "m"
    provider_name = "p"

    def set_approval_handler(self, handler: Any) -> None:
        pass

    def get_last_turn_cost(self) -> Dict[str, Any]:
        return {}

    async def stream(self, message: str) -> AsyncIterator[Any]:
        return
        yield  # pragma: no cover - makes this an async generator


class _FakeAgent:
    active_session_id = "sess"


def _app() -> VictorTUIApp:
    return VictorTUIApp(client=_FakeClient(), agent=_FakeAgent(), settings=None)


async def test_app_captures_edit_diff_and_reveals_pane() -> None:
    app = _app()
    async with app.run_test() as pilot:
        pane = app.query_one("#diff-pane", DiffPane)
        assert pane.display is False  # hidden until an edit occurs

        action = RenderAction(
            RenderKind.TOOL_END,
            tool_name="edit",
            text=json.dumps({"diff": "--- a\n+++ a\n@@ -1 +1 @@\n-x\n+y"}),
            metadata={"arguments": {"path": "a.py"}},
        )
        app._maybe_capture_diff(action)
        await pilot.pause()
        assert pane.has_edits is True
        assert pane.display is True


async def test_toggle_diff_is_noop_without_edits() -> None:
    app = _app()
    async with app.run_test() as pilot:
        pane = app.query_one("#diff-pane", DiffPane)
        app.action_toggle_diff()
        await pilot.pause()
        assert pane.display is False  # nothing to show, stays hidden


async def test_non_edit_tool_result_does_not_reveal_pane() -> None:
    app = _app()
    async with app.run_test() as pilot:
        pane = app.query_one("#diff-pane", DiffPane)
        action = RenderAction(
            RenderKind.TOOL_END,
            tool_name="read",
            text="file contents",
            metadata={"arguments": {"path": "a.py"}},
        )
        app._maybe_capture_diff(action)
        await pilot.pause()
        assert pane.has_edits is False
        assert pane.display is False
