"""Interaction tests for PromptInput paste (Textual run_test harness)."""

from __future__ import annotations

from textual import events
from textual.app import App, ComposeResult

import victor.ui.tui.prompt_input as pi_mod
from victor.ui.tui.prompt_input import PromptInput


class _Host(App[None]):
    """Minimal app hosting a single PromptInput for paste tests."""

    def compose(self) -> ComposeResult:
        yield PromptInput(id="prompt")


async def test_bracketed_paste_keeps_all_lines() -> None:
    app = _Host()
    async with app.run_test():
        prompt = app.query_one("#prompt", PromptInput)
        prompt.focus()
        prompt._on_paste(events.Paste("line1\nline2\nline3"))
        assert prompt.value == "line1\nline2\nline3"


async def test_ctrl_v_reads_os_clipboard(monkeypatch) -> None:
    monkeypatch.setattr(pi_mod, "read_clipboard", lambda: "multi\nline\npaste")
    app = _Host()
    async with app.run_test():
        prompt = app.query_one("#prompt", PromptInput)
        prompt.focus()
        await prompt.action_paste()
        assert prompt.value == "multi\nline\npaste"


async def test_ctrl_v_falls_back_to_in_app_clipboard(monkeypatch) -> None:
    monkeypatch.setattr(pi_mod, "read_clipboard", lambda: None)
    app = _Host()
    async with app.run_test():
        app.copy_to_clipboard("copied-inside")
        prompt = app.query_one("#prompt", PromptInput)
        prompt.focus()
        await prompt.action_paste()
        assert prompt.value == "copied-inside"


async def test_ctrl_v_noop_when_nothing_to_paste(monkeypatch) -> None:
    monkeypatch.setattr(pi_mod, "read_clipboard", lambda: None)
    app = _Host()
    async with app.run_test():
        prompt = app.query_one("#prompt", PromptInput)
        prompt.focus()
        await prompt.action_paste()
        assert prompt.value == ""
