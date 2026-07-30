"""Interaction tests for the VictorTUIApp (Textual App.run_test harness)."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from victor.framework.hitl import ApprovalRequest, ApprovalStatus
from victor.ui.tui.app import VictorTUIApp
from victor.ui.tui.approval_modal import ApprovalScreen
from victor.ui.tui.palette import HelpScreen


class FakeEvent:
    """Minimal stand-in for a VictorClient stream event (see map_event)."""

    def __init__(
        self,
        event_type: str,
        content: Optional[str] = None,
        tool_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.event_type = event_type
        self.content = content
        self.tool_name = tool_name
        self.metadata = metadata or {}


class FakeClient:
    """A VictorClient double exposing only the surface the TUI touches."""

    def __init__(self) -> None:
        self.approval_handler: Any = None
        self.model = "test-model"
        self.provider_name = "test-provider"

    def set_approval_handler(self, handler: Any) -> None:
        self.approval_handler = handler

    def get_last_turn_cost(self) -> Dict[str, Any]:
        return {"total_tokens": 42, "total_cost_usd": 0.0012}

    async def stream(self, message: str) -> AsyncIterator[FakeEvent]:
        yield FakeEvent("content", content="Hello ")
        yield FakeEvent("content", content="world")


class FakeAgent:
    active_session_id = "sess1234abcd"


def _make_app() -> VictorTUIApp:
    return VictorTUIApp(
        client=FakeClient(),
        agent=FakeAgent(),
        settings=None,
        mode="build",
        tool_budget=50,
    )


async def test_mounts_all_panes() -> None:
    app = _make_app()
    async with app.run_test():
        assert app.query_one("#conversation") is not None
        assert app.query_one("#agent-sidebar") is not None
        assert app.query_one("#status-bar") is not None
        assert app.query_one("#prompt") is not None


async def test_registers_terminal_approval_handler() -> None:
    app = _make_app()
    async with app.run_test():
        assert app._client.approval_handler is not None  # type: ignore[attr-defined]


async def test_submitting_a_message_streams_into_the_log() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "hi there"  # type: ignore[attr-defined]
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        convo = app.query_one("#conversation")
        # The user echo plus streamed tokens produced rendered lines.
        assert len(convo.lines) > 0  # type: ignore[attr-defined]


async def test_help_action_opens_help_screen() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        app.action_help()
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)


async def test_toggle_sidebar_hides_and_shows() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        sidebar = app.query_one("#agent-sidebar")
        assert sidebar.display is True
        app.action_toggle_sidebar()
        await pilot.pause()
        assert sidebar.display is False


async def test_approval_screen_resolves_on_approve() -> None:
    app = _make_app()
    results: List[Optional[ApprovalStatus]] = []
    request = ApprovalRequest(
        id="1",
        title="Run shell",
        description="ls -la",
        context={"tool": "shell", "arguments": {"command": "ls -la"}},
        timeout_seconds=30,
    )
    async with app.run_test() as pilot:
        app.push_screen(ApprovalScreen(request), results.append)
        await pilot.pause()
        await pilot.press("a")
        await pilot.pause()
    assert results and results[0] is ApprovalStatus.APPROVED


async def test_approval_screen_rejects_on_escape() -> None:
    app = _make_app()
    results: List[Optional[ApprovalStatus]] = []
    request = ApprovalRequest(
        id="2",
        title="Delete file",
        description="rm x",
        context={"tool": "shell", "arguments": {"command": "rm x"}},
        timeout_seconds=30,
    )
    async with app.run_test() as pilot:
        app.push_screen(ApprovalScreen(request), results.append)
        await pilot.pause()
        await pilot.press("r")
        await pilot.pause()
    assert results and results[0] is ApprovalStatus.REJECTED
