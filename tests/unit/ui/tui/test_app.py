"""Interaction tests for the VictorTUIApp (Textual App.run_test harness)."""

from __future__ import annotations

import asyncio
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


class BlockingFakeClient(FakeClient):
    """Streams that hold a turn open on a gate, recording submission order."""

    def __init__(self, fail_first: bool = False) -> None:
        super().__init__()
        self.calls: List[str] = []
        self.gate = asyncio.Event()
        self._fail_first = fail_first

    async def stream(self, message: str) -> AsyncIterator[FakeEvent]:
        self.calls.append(message)
        yield FakeEvent("content", content=f"start:{message}")
        await self.gate.wait()
        if self._fail_first and message == "first":
            self._fail_first = False
            raise RuntimeError("boom")
        yield FakeEvent("content", content=f"end:{message}")


class FakeAgent:
    active_session_id = "sess1234abcd"


def _make_app(client: Optional[FakeClient] = None) -> VictorTUIApp:
    return VictorTUIApp(
        client=client or FakeClient(),
        agent=FakeAgent(),
        settings=None,
        mode="build",
        tool_budget=50,
    )


async def _submit(pilot, app: VictorTUIApp, text: str) -> None:
    prompt = app.query_one("#prompt")
    prompt.value = text  # type: ignore[attr-defined]
    await pilot.press("enter")
    await pilot.pause()


async def _settle(app: VictorTUIApp, rounds: int = 4) -> None:
    """Let workers finish and the message loop drain (pump may chain turns)."""
    for _ in range(rounds):
        await app.workers.wait_for_complete()
        await app.workers.wait_for_complete()
        await asyncio.sleep(0)


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


async def test_copy_action_copies_last_response_when_no_selection() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "hi there"  # type: ignore[attr-defined]
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.action_copy()
        await pilot.pause()
        # With no active selection, ctrl+c copies the whole assistant reply.
        assert app.clipboard == "Hello world"


async def test_copy_action_is_a_noop_when_nothing_to_copy() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        app.action_copy()  # no selection, no response yet
        await pilot.pause()
        assert app.clipboard == ""


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


# ── queued prompts: input stays live while a turn runs ────────────────


async def test_prompt_stays_enabled_and_focusable_during_turn() -> None:
    client = BlockingFakeClient()
    app = _make_app(client)
    async with app.run_test() as pilot:
        await _submit(pilot, app, "first")
        await pilot.pause()
        assert client.calls == ["first"]  # turn in flight, gated mid-stream
        prompt = app.query_one("#prompt")
        assert prompt.disabled is False  # type: ignore[attr-defined]
        prompt.focus()
        assert prompt.has_focus
        client.gate.set()
        await _settle(app)


async def test_second_submit_while_running_is_queued_not_started() -> None:
    client = BlockingFakeClient()
    app = _make_app(client)
    async with app.run_test() as pilot:
        await _submit(pilot, app, "first")
        await _submit(pilot, app, "second")
        assert client.calls == ["first"]  # second must not start mid-run
        assert app._queue.snapshot() == ("second",)
        client.gate.set()
        await _settle(app)
        assert client.calls == ["first", "second"]  # drained FIFO after the run
        assert not app._queue


async def test_queued_prompts_run_in_fifo_order() -> None:
    client = BlockingFakeClient()
    app = _make_app(client)
    async with app.run_test() as pilot:
        await _submit(pilot, app, "first")
        await _submit(pilot, app, "second")
        await _submit(pilot, app, "third")
        assert client.calls == ["first"]
        client.gate.set()
        await _settle(app, rounds=6)
        assert client.calls == ["first", "second", "third"]
        assert not app._queue


async def test_status_bar_shows_queued_count_mid_run() -> None:
    client = BlockingFakeClient()
    app = _make_app(client)
    async with app.run_test() as pilot:
        await _submit(pilot, app, "first")
        await _submit(pilot, app, "second")
        await pilot.pause()
        status = str(app.query_one("#status-bar").render())  # type: ignore[attr-defined]
        assert "1 queued" in status
        client.gate.set()
        await _settle(app)


async def test_interrupt_preserves_queue_and_starts_next() -> None:
    client = BlockingFakeClient()
    app = _make_app(client)
    async with app.run_test() as pilot:
        await _submit(pilot, app, "first")
        await _submit(pilot, app, "second")
        await pilot.press("escape")  # interrupt the running turn
        client.gate.set()  # release whatever is in flight; queued prompt proceeds
        await _settle(app)
        assert client.calls == ["first", "second"]  # queued prompt auto-ran
        assert not app._queue
        convo = app.query_one("#conversation")
        text = "\n".join(line.text for line in convo.lines)  # type: ignore[attr-defined]
        assert "⏹ interrupted" in text  # the first turn was cancelled, not completed


async def test_queue_clear_slash_drops_pending_prompts() -> None:
    client = BlockingFakeClient()
    app = _make_app(client)
    async with app.run_test() as pilot:
        await _submit(pilot, app, "first")
        await _submit(pilot, app, "second")
        client.gate.set()
        await _settle(app)
        assert client.calls == ["first", "second"]
        client.gate = asyncio.Event()  # fresh closed gate for phase two
        await _submit(pilot, app, "third")  # runs, blocks mid-stream
        await _submit(pilot, app, "fourth")  # queued
        await pilot.pause()
        assert app._queue.snapshot() == ("fourth",)
        app.run_worker(app._dispatch_slash("/queue clear"), group="slash")
        await asyncio.sleep(0)
        await pilot.pause()
        assert not app._queue  # fourth dropped
        client.gate.set()
        await _settle(app)
        assert client.calls == ["first", "second", "third"]  # fourth never ran


async def test_queue_slash_lists_pending_prompts() -> None:
    client = BlockingFakeClient()
    app = _make_app(client)
    async with app.run_test() as pilot:
        await _submit(pilot, app, "first")
        await _submit(pilot, app, "second")
        convo = app.query_one("#conversation")
        lines_before = len(convo.lines)  # type: ignore[attr-defined]
        app.run_worker(app._dispatch_slash("/queue"), group="slash")
        await asyncio.sleep(0)
        await pilot.pause()
        assert len(convo.lines) > lines_before  # listing rendered into the log
        assert app._queue.snapshot() == ("second",)  # listing does not consume
        client.gate.set()
        await _settle(app)


async def test_stream_failure_does_not_drop_queue() -> None:
    client = BlockingFakeClient(fail_first=True)
    app = _make_app(client)
    async with app.run_test() as pilot:
        await _submit(pilot, app, "first")
        await _submit(pilot, app, "second")
        client.gate.set()  # first raises now; second must still run
        await _settle(app)
        assert client.calls == ["first", "second"]
        assert not app._queue


# ── shell-first bootstrap: UI up first, session init in-app ───────────


class FakeBootstrap:
    """Async bootstrap callable the app runs after mount (like chat.py builds)."""

    def __init__(self, fail_with: Optional[Exception] = None) -> None:
        self.release = asyncio.Event()
        self.fail_with = fail_with
        self.calls = 0

    async def __call__(self) -> Any:
        self.calls += 1
        await self.release.wait()
        if self.fail_with is not None:
            raise self.fail_with
        return FakeAgent()


async def test_prompt_disabled_until_bootstrap_ready() -> None:
    boot = FakeBootstrap()
    client = BlockingFakeClient()
    app = VictorTUIApp(client=client, agent=None, settings=None, bootstrap=boot)
    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        assert prompt.disabled is True  # type: ignore[attr-defined]
        await _submit(pilot, app, "early")  # must be swallowed pre-ready
        await pilot.pause()
        assert client_calls(app) == []  # type: ignore[attr-defined]
        boot.release.set()
        await _settle(app)
        assert prompt.disabled is False  # type: ignore[attr-defined]
        assert prompt.has_focus
        convo = app.query_one("#conversation")
        text = "\n".join(line.text for line in convo.lines)  # type: ignore[attr-defined]
        assert "ready in" in text


def client_calls(app: VictorTUIApp) -> list:
    return app._client.calls  # type: ignore[attr-defined]


async def test_ready_enables_submit_and_uses_bootstrapped_agent() -> None:
    boot = FakeBootstrap()
    client = BlockingFakeClient()
    app = VictorTUIApp(client=client, agent=None, settings=None, bootstrap=boot)
    async with app.run_test() as pilot:
        boot.release.set()
        await _settle(app)
        await _submit(pilot, app, "hello")
        client.gate.set()
        await _settle(app)
        assert client.calls == ["hello"]  # turn ran against the bootstrapped session
        assert app._agent is not None  # agent adopted from the bootstrap result


async def test_bootstrap_failure_shows_error_and_exits() -> None:
    boot = FakeBootstrap(fail_with=RuntimeError("provider api key missing"))
    app = VictorTUIApp(client=FakeClient(), agent=None, settings=None, bootstrap=boot)
    async with app.run_test() as pilot:
        boot.release.set()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._bootstrap_error is not None  # type: ignore[attr-defined]
        assert "provider api key missing" in app._bootstrap_error  # type: ignore[attr-defined]
