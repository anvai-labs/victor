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
