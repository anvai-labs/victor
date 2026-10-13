"""FEP-0029 Phase 1: durable single-agent chat pause mechanism.

When durable pause is armed (``governance.durable``), a policy ASK raises :class:`ApprovalPause`
(the shared BaseException pause signal), which the turn boundary (``execute_message``) catches and
converts into an ``awaiting_approval`` :class:`TaskResult` with a resumable ``run_id`` + the pending
approval request, recording the pause in the process-local store. Disarmed, the approval handler
delegates inline — byte-identical.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from victor.agent.factory.coordination_builders import resolve_policy_approval_handler
from victor.agent.member_approval_context import MemberApprovalPause
from victor.agent.paused_run_store import (
    InMemoryPausedRunStore,
    get_paused_run_store,
    reset_paused_run_store,
    set_paused_run_store,
)
from victor.framework import message_execution as me
from victor.framework.approval_pause import ApprovalPause, current_durable_pause_enabled
from victor.framework.hitl import ApprovalRequest, ApprovalStatus


@pytest.fixture(autouse=True)
def _reset_paused_run_store():
    """Keep an injected/overridden paused-run store from leaking across tests."""
    yield
    reset_paused_run_store()


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        id="req-1",
        title="Approve tool: run_command",
        description="rm -rf /tmp/x",
        context={"tool_name": "run_command", "arguments": {"cmd": "rm -rf /tmp/x"}},
    )


def _governance(*, durable: bool, interactive: bool = False) -> Any:
    return SimpleNamespace(durable=durable, interactive_approval=interactive)


# ── the pause signal ──────────────────────────────────────────────


def test_approval_pause_is_baseexception_and_member_subclass() -> None:
    # Rides through `except Exception` (it's a BaseException, not Exception).
    assert issubclass(ApprovalPause, BaseException)
    assert not issubclass(ApprovalPause, Exception)
    # Team pause is now a subclass — team code catching MemberApprovalPause is unaffected.
    assert issubclass(MemberApprovalPause, ApprovalPause)
    p = ApprovalPause(_request())
    assert p.request.title == "Approve tool: run_command"


# ── the handler wrapper (arming) ──────────────────────────────────


async def test_handler_raises_when_armed_delegates_when_disarmed() -> None:
    calls = []

    async def inner(req: ApprovalRequest) -> Any:
        calls.append(req)
        return (ApprovalStatus.APPROVED, "ok", "tester")

    container = SimpleNamespace(get_optional=lambda _t: SimpleNamespace(handler=inner))
    handler = resolve_policy_approval_handler(_governance(durable=True), container)
    assert handler is not None

    # Disarmed → delegates to the inner handler (inline path, byte-identical).
    token = current_durable_pause_enabled.set(False)
    try:
        result = await handler(_request())
        assert result == (ApprovalStatus.APPROVED, "ok", "tester")
        assert len(calls) == 1
    finally:
        current_durable_pause_enabled.reset(token)

    # Armed → raises ApprovalPause instead of calling inner.
    token = current_durable_pause_enabled.set(True)
    try:
        with pytest.raises(ApprovalPause):
            await handler(_request())
        assert len(calls) == 1  # inner not called again
    finally:
        current_durable_pause_enabled.reset(token)


async def test_handler_none_when_not_durable_and_no_inner() -> None:
    # No container handler + durable off → unchanged (None → ASK falls back).
    assert resolve_policy_approval_handler(_governance(durable=False), None) is None


async def test_handler_wraps_when_durable_without_inner() -> None:
    handler = resolve_policy_approval_handler(_governance(durable=True), None)
    assert handler is not None
    # Armed → pauses even with no base handler.
    token = current_durable_pause_enabled.set(True)
    try:
        with pytest.raises(ApprovalPause):
            await handler(_request())
    finally:
        current_durable_pause_enabled.reset(token)
    # Disarmed + no base handler → fail safe reject (not a hang, not an approve).
    token = current_durable_pause_enabled.set(False)
    try:
        status, _resp, responder = await handler(_request())
        assert status == ApprovalStatus.REJECTED and responder == "no-handler"
    finally:
        current_durable_pause_enabled.reset(token)


# ── the turn boundary (catch + surface + record) ──────────────────


def _orchestrator(*, durable: bool) -> Any:
    return SimpleNamespace(
        settings=SimpleNamespace(governance=SimpleNamespace(durable=durable)),
        model="test-model",
        active_session_id="sess-42",
        agent_id=None,
    )


async def test_execute_message_pauses_and_surfaces_run_id(monkeypatch: Any) -> None:
    # Isolate from the real project.db-backed store: inject an in-memory store for this turn.
    set_paused_run_store(InMemoryPausedRunStore())
    monkeypatch.setattr(me, "_resolve_chat_runtime", lambda *a, **k: object())

    async def _raise(*a: Any, **k: Any) -> Any:
        raise ApprovalPause(_request())

    monkeypatch.setattr(me, "_invoke_chat", _raise)

    result = await me.execute_message(
        orchestrator=_orchestrator(durable=True), user_message="do it"
    )

    assert result.status == "awaiting_approval"
    assert result.success is False
    assert result.run_id
    assert result.approval_request["title"] == "Approve tool: run_command"

    # The pause was recorded, with the gated tool captured for a faithful resume.
    stored = get_paused_run_store().get(result.run_id)
    assert stored is not None
    assert stored.session_id == "sess-42"
    assert stored.pending_tool == {
        "tool_name": "run_command",
        "arguments": {"cmd": "rm -rf /tmp/x"},
    }
    assert stored.status == "awaiting_approval"


async def test_durable_pause_enabled_reads_governance() -> None:
    assert me._durable_pause_enabled(_orchestrator(durable=True)) is True
    assert me._durable_pause_enabled(_orchestrator(durable=False)) is False
    assert me._durable_pause_enabled(SimpleNamespace()) is False  # no settings → off


# ── config threading ──────────────────────────────────────────────


def test_tool_approval_config_threads_durable_to_governance() -> None:
    from victor.core.feature_flags import FeatureFlag, disable_feature, is_feature_enabled
    from victor.framework.session_config import SessionConfig, ToolApprovalConfig

    was_enabled = is_feature_enabled(FeatureFlag.USE_POLICY_ENGINE)
    governance = SimpleNamespace(enabled=False, ask_fallback="deny", ask_on_tools=[], durable=False)
    settings = SimpleNamespace(governance=governance)
    try:
        SessionConfig(
            tool_approval=ToolApprovalConfig(enabled=True, durable=True)
        ).apply_to_settings(settings)
        assert governance.durable is True
        assert governance.enabled is True
    finally:
        if not was_enabled:
            disable_feature(FeatureFlag.USE_POLICY_ENGINE)

    # Default is off (byte-identical when not requested).
    assert ToolApprovalConfig().durable is False


# ── the store ─────────────────────────────────────────────────────


def test_paused_run_store_roundtrip() -> None:
    store = InMemoryPausedRunStore()
    run_id = store.save(
        session_id="s1",
        agent_id="a1",
        approval_request={"id": "r", "title": "t"},
        pending_tool={"tool_name": "run_command", "arguments": {}},
        created_at=1.0,
    )
    run = store.get(run_id)
    assert run is not None and run.session_id == "s1" and run.status == "awaiting_approval"
    assert [r.run_id for r in store.list_pending()] == [run_id]

    assert store.mark_resumed(run_id) is True
    assert store.get(run_id).status == "resumed"
    assert store.list_pending() == []
    assert store.mark_resumed(run_id) is False  # single-use


async def _run_pausing_boundary(monkeypatch, request, streaming, orchestrator=None):
    """Exercise real turn-boundary persistence with only model execution replaced."""
    from victor.framework import _internal

    orchestrator = orchestrator or _orchestrator(durable=True)

    monkeypatch.setattr(me, "_resolve_chat_runtime", lambda *a, **k: object())

    async def chat(*args, **kwargs):
        raise ApprovalPause(request)

    async def stream(*args, **kwargs):
        if False:
            yield
        raise ApprovalPause(request)

    monkeypatch.setattr(me, "_invoke_chat", chat)
    monkeypatch.setattr(_internal, "stream_with_events", stream)
    try:
        if streaming:
            events = [
                event
                async for event in me.stream_message_events(
                    orchestrator=orchestrator, user_message="do it"
                )
            ]
            assert len(events) == 1
            return events[0].metadata
        result = await me.execute_message(orchestrator=orchestrator, user_message="do it")
        assert result.status == "awaiting_approval" and not result.success
        return {"run_id": result.run_id, "approval_request": result.approval_request}
    finally:
        assert not current_durable_pause_enabled.get()


@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streamed"])
@pytest.mark.parametrize("mode", ["commit", "cancel", "session-switch", "agent-switch"])
async def test_pause_sqlite_contention_keeps_loop_live_and_publishes_only_after_commit(
    tmp_path, monkeypatch, streaming, mode
):
    import asyncio
    import sqlite3
    import threading
    from victor.agent.paused_run_store import ProjectDbPausedRunStore

    store = ProjectDbPausedRunStore(tmp_path / "pause.db")
    set_paused_run_store(store)
    store.get("initialize-schema")
    entered, finished, release = threading.Event(), threading.Event(), threading.Event()
    released_by_loop = []
    save = store.save

    def observed_save(**kwargs):
        entered.set()
        try:
            return save(**kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(store, "save", observed_save)
    blocker = sqlite3.connect(store.db_path, check_same_thread=False)
    blocker.execute("BEGIN IMMEDIATE")

    def unlock():
        # Watchdog also releases the pre-fix blocking call; no speed threshold assertion.
        released_by_loop.append(release.wait(5))
        blocker.rollback()
        blocker.close()

    watchdog = threading.Thread(target=unlock)
    watchdog.start()
    orchestrator = _orchestrator(durable=True)
    task = asyncio.create_task(
        _run_pausing_boundary(monkeypatch, _request(), streaming, orchestrator)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert not released_by_loop, "pause persistence blocked loop until watchdog release"
        assert not task.done(), "pause published before persistence finished"
        if mode == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        if mode == "session-switch":
            orchestrator.active_session_id = "other-session"
        elif mode == "agent-switch":
            orchestrator.agent_id = "other-agent"
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        result = None
        if mode.endswith("switch"):
            with pytest.raises(PermissionError, match="ownership changed"):
                await task
        elif mode == "commit":
            result = await task
        reopened = ProjectDbPausedRunStore(store.db_path)
        pending = reopened.list_pending()
        assert len(pending) == 1
        assert pending[0].status == "awaiting_approval" and pending[0].action is None
        assert pending[0].session_id == "sess-42"
        if result:
            assert result["run_id"] == pending[0].run_id
            assert result["approval_request"] == pending[0].approval_request
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(watchdog.join, 5)
        assert not watchdog.is_alive()
    assert released_by_loop == [True]


@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streamed"])
@pytest.mark.parametrize("committed", [False, True], ids=["before-commit", "lost-ack"])
async def test_pause_save_failure_never_publishes_or_retries(
    tmp_path, monkeypatch, streaming, committed
):
    from unittest.mock import Mock
    from victor.agent.paused_run_store import ProjectDbPausedRunStore

    store = ProjectDbPausedRunStore(tmp_path / "pause.db")
    set_paused_run_store(store)
    save = store.save

    def fail(**kwargs):
        if committed:
            save(**kwargs)
        raise OSError("pause storage unavailable")

    observed = Mock(side_effect=fail)
    monkeypatch.setattr(store, "save", observed)
    with pytest.raises(OSError, match="pause storage unavailable"):
        await _run_pausing_boundary(monkeypatch, _request(), streaming)
    observed.assert_called_once()
    assert len(ProjectDbPausedRunStore(store.db_path).list_pending()) == int(committed)
