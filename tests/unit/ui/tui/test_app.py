"""Interaction tests for the VictorTUIApp (Textual App.run_test harness)."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

from victor.framework.hitl import ApprovalRequest, ApprovalStatus
from victor.ui.tui.app import VictorTUIApp
from victor.ui.tui.themes import resolve_theme
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


async def test_terminal_theme_mounts_and_cycles_all_four() -> None:
    """The victor-terminal theme must mount (ANSI mode, no Color.parse crash)
    and F6 cycling must survive every theme in the cycle — the adversarial
    review caught the empty-color-string variant crashing on mount."""
    app = _make_app()
    app._initial_theme = resolve_theme("terminal")
    async with app.run_test():
        assert app.theme == "victor-terminal"
        seen: list[str] = []
        for _ in range(4):
            app.action_cycle_theme()
            await asyncio.sleep(0)
            seen.append(app.theme)
        assert seen[0] == "victor-dark"  # cycle wraps terminal -> dark
        assert seen[3] == "victor-terminal"


async def test_mounts_all_panes() -> None:
    app = _make_app()
    async with app.run_test():
        assert app.query_one("#conversation") is not None
        assert app.query_one("#token-usage-bar") is not None
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


async def test_bootstrap_rebinds_agent_into_slash_handler() -> None:
    # The handler is constructed during on_mount against the then-None agent
    # (shell-first bootstrap). Without a rebind, every agent-requiring slash
    # command reports "requires an active session" forever (audit B1).
    boot = FakeBootstrap()
    app = VictorTUIApp(client=FakeClient(), agent=None, settings=None, bootstrap=boot)
    async with app.run_test() as pilot:
        assert app._slash is not None  # type: ignore[attr-defined]
        assert app._slash.agent is None  # type: ignore[attr-defined]
        boot.release.set()
        await _settle(app)
        assert app._slash.agent is app._agent  # type: ignore[attr-defined]


async def test_slash_command_requires_session_before_bootstrap_completes() -> None:
    # Negative control for the rebind: pre-bootstrap, agent-bound commands
    # must degrade to the "requires an active session" message, not crash.
    boot = FakeBootstrap()
    app = VictorTUIApp(client=FakeClient(), agent=None, settings=None, bootstrap=boot)
    async with app.run_test() as pilot:
        await app._dispatch_slash("/model")  # type: ignore[attr-defined]
        await pilot.pause()
        convo = app.query_one("#conversation")
        text = "\n".join(line.text for line in convo.lines)  # type: ignore[attr-defined]
        assert "requires an active session" in text


async def test_dispatch_slash_tolerates_bare_separator_input() -> None:
    # "/ " passed len(text) > 1 but split() is [] -> IndexError killed the
    # slash worker silently (audit B2). It must dispatch as a no-op instead.
    app = _make_app()
    async with app.run_test() as pilot:
        # Before the fix this raised IndexError out of the worker.
        await app._dispatch_slash("/ ")  # type: ignore[attr-defined]
        await app._dispatch_slash("/")  # type: ignore[attr-defined]
        await pilot.pause()
        convo = app.query_one("#conversation")
        text = "\n".join(line.text for line in convo.lines)  # type: ignore[attr-defined]
        assert "requires an active session" not in text


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


async def test_sidebar_uses_current_context_and_never_cumulative_usage():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    client = FakeClient()
    client.context_window = lambda: 10000
    client.get_session_metrics = AsyncMock(return_value={"context_tokens": 1200})
    app = _make_app(client)
    cumulative = MagicMock(return_value=SimpleNamespace(total_tokens=150000))
    app._agent = SimpleNamespace(_orchestrator=SimpleNamespace(get_token_usage=cumulative))
    assert await app._compute_context_usage() == (1200, 10000, 12.0)
    cumulative.assert_not_called()
    # Compaction can shrink current context even as billed session tokens rise.
    client.get_session_metrics.return_value = {"context_tokens": 0}
    assert await app._compute_context_usage() == (0, 10000, 0.0)
    # Aggregate turn cost is not a valid estimate of the current request.
    client.get_session_metrics.return_value = {}
    assert await app._compute_context_usage() == (None, 10000, None)


# ── cancellation hygiene: modal + generator lifecycle (audit B6 + B8) ──


class ApprovalBlockingClient(FakeClient):
    """Stream that asks for approval and never yields content on its own."""

    async def stream(self, message: str) -> AsyncIterator[FakeEvent]:
        request = ApprovalRequest(
            id="blocked-1",
            title="run_command",
            description="",
            context={"tool_name": "run_command"},
            timeout_seconds=30,
        )
        assert self.approval_handler is not None
        await self.approval_handler(request)
        yield FakeEvent("content", content="never reached")


async def test_interrupt_dismisses_pending_approval_modal() -> None:
    # ESC during a turn blocked on the approval modal: the worker is cancelled,
    # the modal's awaiting handler gets CancelledError, and the modal must pop
    # immediately — not keep stealing keys until its own timeout (audit B6).
    app = VictorTUIApp(client=ApprovalBlockingClient(), agent=FakeAgent(), settings=None)
    async with app.run_test() as pilot:
        await _submit(pilot, app, "do a thing")
        deadline = asyncio.get_running_loop().time() + 5
        from victor.ui.tui.approval_modal import ApprovalScreen as AS

        while not isinstance(app.screen, AS):
            await pilot.pause()
            assert asyncio.get_running_loop().time() < deadline, "modal never appeared"
        app.action_interrupt()
        try:  # the turn worker is intentionally cancelled
            await app.workers.wait_for_complete()
        except Exception:  # noqa: BLE001 - WorkerCancelled is the expected path
            pass
        await pilot.pause()
        assert not isinstance(app.screen, AS)
        convo = app.query_one("#conversation")
        text = "\n".join(line.text for line in convo.lines)  # type: ignore[attr-defined]
        assert "⏹ interrupted" in text


class ClosingAwareClient(FakeClient):
    """Records when its stream generator is closed (aclosing verification)."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def stream(self, message: str) -> AsyncIterator[FakeEvent]:
        try:
            yield FakeEvent("content", content="partial ")
            yield FakeEvent("content", content="output")
        except GeneratorExit:
            self.closed = True
            raise


async def test_early_exit_closes_the_stream_generator() -> None:
    # aclosing parity with the Chainlit surface: an interrupt (or any early
    # exit from the turn loop) must close the generator chain deterministically
    # instead of waiting for GC finalization.
    from contextlib import aclosing

    client = ClosingAwareClient()
    app = _make_app(client)
    async with app.run_test() as pilot:
        async with aclosing(app._stream_actions("hello")) as actions:
            count = 0
            async for _action in actions:
                count += 1
                if count == 1:
                    break  # early exit mid-stream
        await pilot.pause()
        assert client.closed


# ── slash dispatch serialization (audit B9) ───────────────────────────


async def test_overlapping_slash_dispatches_render_fully_and_in_order() -> None:
    # Two rapid dispatches used to share one seek/truncate StringIO: the
    # second could wipe the first's output mid-render. Each now renders
    # against its own console under a lock, so both outputs land, whole.
    app = _make_app()
    seen: List[str] = []

    async def fake_captured(self, text: str) -> str:
        seen.append(f"start:{text}")
        await asyncio.sleep(0.01)
        seen.append(f"end:{text}")
        return f"out:{text}"

    from unittest.mock import patch

    async with app.run_test() as pilot:
        with patch.object(type(app._slash), "execute_captured", fake_captured):  # type: ignore[attr-defined]
            await asyncio.gather(
                app._run_captured_slash("/cost"),
                app._run_captured_slash("/tools"),
            )
        await pilot.pause()

        # No interleaving: each command's start/end pair is contiguous.
        assert seen == ["start:/cost", "end:/cost", "start:/tools", "end:/tools"] or seen == [
            "start:/tools",
            "end:/tools",
            "start:/cost",
            "end:/cost",
        ]
        convo = app.query_one("#conversation")
        text = "\n".join(line.text for line in convo.lines)  # type: ignore[attr-defined]
        assert "out:/cost" in text and "out:/tools" in text


async def test_execute_captured_restores_handler_console() -> None:
    from victor.ui.slash.handler import SlashCommandHandler
    from victor.ui.slash.protocol import CommandMetadata
    from victor.ui.slash.registry import CommandRegistry

    class Echo:
        @property
        def metadata(self) -> CommandMetadata:
            return CommandMetadata(name="echo", description="", usage="/echo", category="test")

        async def execute(self, ctx) -> None:
            ctx.console.print("captured-line")

    registry = CommandRegistry()
    registry.register(Echo())
    outer_console = object()
    handler = SlashCommandHandler.__new__(SlashCommandHandler)
    handler.console = outer_console  # type: ignore[assignment]
    handler.agent = None
    handler.settings = None
    handler._registry = registry
    handler.exit_callback = None

    output = await handler.execute_captured("/echo")
    assert output == "captured-line"
    assert handler.console is outer_console  # type: ignore[comparison-overlap]
