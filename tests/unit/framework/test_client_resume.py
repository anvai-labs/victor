"""FEP-0029 Phase 3a: VictorClient.resume glue — load, single-use guard, TaskResult shape.

The heavy replay logic is covered by tests/unit/agent/test_durable_resume.py; here we verify the thin
public wiring: unknown/already-resumed run_ids raise, and a successful resume returns an "ok"
TaskResult carrying the resume metadata. `resume_paused_run` is stubbed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from victor.agent.paused_run_store import (
    InMemoryPausedRunStore,
    reset_paused_run_store,
    set_paused_run_store,
)
from victor.framework.approval_pause import ApprovalDecision
from victor.framework.client import VictorClient


@pytest.fixture(autouse=True)
def _store():
    store = InMemoryPausedRunStore()
    set_paused_run_store(store)
    yield store
    reset_paused_run_store()


def _client() -> VictorClient:
    # Bypass full initialization — resume() only touches _initialized/_context/_agent.
    client = VictorClient.__new__(VictorClient)
    client._initialized = True
    client._context = object()
    client._agent = SimpleNamespace(_orchestrator=object())
    return client


def _save(store: Any) -> str:
    # session_id=None so resume() skips resume_session (no live session to hydrate in this glue test).
    return store.save(
        session_id=None,
        agent_id="a",
        approval_request={"id": "r", "title": "t"},
        pending_tool={"tool_name": "run_command", "arguments": {}},
    )


async def test_resume_unknown_run_raises() -> None:
    with pytest.raises(ValueError):
        await _client().resume("does-not-exist", ApprovalDecision(approved=True))


async def test_resume_returns_ok_taskresult_and_is_single_use(
    monkeypatch: Any, _store: Any
) -> None:
    from victor.agent import durable_resume

    async def _stub(orchestrator: Any, paused: Any, decision: Any) -> Any:
        return durable_resume.ResumeResult(
            final_content="continued answer",
            tool_calls=[],
            approved=True,
            gated_tool="run_command",
            continuation_turns=1,
        )

    monkeypatch.setattr(durable_resume, "resume_paused_run", _stub)

    run_id = _save(_store)
    result = await _client().resume(run_id, ApprovalDecision(approved=True))

    assert result.status == "ok" and result.success is True
    assert result.content == "continued answer"
    assert result.metadata["resumed_run_id"] == run_id
    assert result.metadata["gated_tool"] == "run_command"

    # Single-use: the run is now marked resumed, so a second resume raises.
    assert _store.get(run_id).status == "resumed"
    with pytest.raises(ValueError):
        await _client().resume(run_id, ApprovalDecision(approved=True))


async def test_resume_surfaces_a_chained_pause(monkeypatch: Any, _store: Any) -> None:
    from victor.agent import durable_resume

    async def _stub(orchestrator: Any, paused: Any, decision: Any) -> Any:
        return durable_resume.ResumeResult(
            final_content="",
            approved=True,
            gated_tool="run_command",
            continuation_turns=1,
            awaiting_run_id="chained-run-2",
            awaiting_approval_request={"id": "r2", "title": "Approve tool: deploy"},
        )

    monkeypatch.setattr(durable_resume, "resume_paused_run", _stub)

    run_id = _save(_store)
    result = await _client().resume(run_id, ApprovalDecision(approved=True))

    # A chained pause is surfaced exactly like a first pause: awaiting + a fresh run_id.
    assert result.status == "awaiting_approval" and result.success is False
    assert result.run_id == "chained-run-2"
    assert result.approval_request["title"] == "Approve tool: deploy"
    assert result.metadata["resumed_run_id"] == run_id


async def test_resume_of_a_stale_pause_is_expired(_store: Any) -> None:
    # A pause with an ancient created_at is expired by the resume seam's opportunistic GC.
    run_id = _store.save(
        session_id=None,
        agent_id="a",
        approval_request={"id": "r", "title": "t"},
        created_at=1.0,  # ancient → older than the TTL
    )
    with pytest.raises(ValueError, match="expired"):
        await _client().resume(run_id, ApprovalDecision(approved=True))
    # It was retired (not left pending) so a retry keeps failing cleanly.
    assert _store.get(run_id).status == "expired"


async def test_failed_session_hydration_never_reaches_resume(monkeypatch, _store):
    from unittest.mock import AsyncMock
    from victor.agent import durable_resume

    run_id = _store.save(session_id="missing", agent_id="a", approval_request={})
    client = _client()
    client.resume_session = AsyncMock(return_value=None)
    replay = AsyncMock()
    monkeypatch.setattr(durable_resume, "resume_paused_run", replay)
    with pytest.raises(ValueError, match="restored"):
        await client.resume(run_id, ApprovalDecision(True))
    replay.assert_not_awaited()
    assert _store.get(run_id).status == "resumed"  # no automatic reopening of claims


async def test_non_boolean_decision_does_not_consume_pause(_store):
    run_id = _save(_store)
    with pytest.raises(ValueError, match="boolean"):
        await _client().resume(run_id, ApprovalDecision("false"))
    assert _store.get(run_id).status == "awaiting_approval"


async def test_durable_action_mode_rejects_memory_store_before_claim(_store):
    run_id = _save(_store)
    with pytest.raises(ValueError, match="persistent"):
        await _client().resume(run_id, ApprovalDecision(True), durable_actions=True)
    assert _store.get(run_id).status == "awaiting_approval"


async def test_durable_action_mode_passes_same_store_to_canonical_resume(tmp_path, monkeypatch):
    from victor.agent import durable_resume
    from victor.agent.paused_run_store import ProjectDbPausedRunStore

    persistent = ProjectDbPausedRunStore(tmp_path / "state.db")
    set_paused_run_store(persistent)
    run_id = _save(persistent)

    async def resume(orchestrator, paused, decision, *, action_store):
        assert action_store is persistent
        assert paused.run_id == run_id
        return durable_resume.ResumeResult(final_content="done")

    monkeypatch.setattr(durable_resume, "resume_paused_run", resume)
    out = await _client().resume(run_id, ApprovalDecision(True), durable_actions=True)
    assert out.content == "done"


async def test_default_off_writes_no_action_record_on_persistent_store(tmp_path, monkeypatch):
    """Pin the opt-in default: without durable_actions=True, canonical resume
    receives no action journal and the persistent store never gains durable
    action evidence. If someone flips the default, this fails."""
    from victor.agent import durable_resume
    from victor.agent.paused_run_store import ProjectDbPausedRunStore

    persistent = ProjectDbPausedRunStore(tmp_path / "state.db")
    set_paused_run_store(persistent)
    run_id = _save(persistent)

    seen: dict[str, Any] = {}

    async def resume(orchestrator, paused, decision, *, action_store=None):
        seen["action_store"] = action_store
        return durable_resume.ResumeResult(final_content="done")

    monkeypatch.setattr(durable_resume, "resume_paused_run", resume)
    out = await _client().resume(run_id, ApprovalDecision(True))
    assert out.content == "done"
    assert seen["action_store"] is None
    assert persistent.get(run_id).action is None


def test_action_status_requires_original_restored_session(_store):
    run_id = _store.save(session_id="original", agent_id="a", approval_request={})
    client = _client()
    client._agent._orchestrator = SimpleNamespace(active_session_id="other")
    with pytest.raises(ValueError, match="original session"):
        client.get_action_status(run_id)
    client._agent._orchestrator.active_session_id = "original"
    status = client.get_action_status(run_id)
    assert status == {
        "version": 1,
        "run_id": run_id,
        "approval_status": "awaiting_approval",
        "action": None,
    }


@pytest.mark.parametrize("cancel", [False, True], ids=["continue", "cancel"])
async def test_sqlite_claim_contention_keeps_loop_responsive_and_cancel_never_dispatches(
    tmp_path, monkeypatch, cancel
):
    import asyncio
    import sqlite3
    import threading
    from unittest.mock import AsyncMock

    from victor.agent import durable_resume
    from victor.agent.paused_run_store import ProjectDbPausedRunStore

    persistent = ProjectDbPausedRunStore(tmp_path / "state.db")
    set_paused_run_store(persistent)
    run_id = _save(persistent)
    entered = threading.Event()
    claimed = threading.Event()
    release = threading.Event()
    released_by_loop = []
    expire = persistent.expire_pending
    mark = persistent.mark_resumed

    def observed_expire(**kwargs):
        entered.set()
        return expire(**kwargs)

    def observed_claim(run_id):
        try:
            return mark(run_id)
        finally:
            claimed.set()

    monkeypatch.setattr(persistent, "expire_pending", observed_expire)
    monkeypatch.setattr(persistent, "mark_resumed", observed_claim)
    replay = AsyncMock(return_value=durable_resume.ResumeResult(final_content="done"))
    monkeypatch.setattr(durable_resume, "resume_paused_run", replay)
    blocker = sqlite3.connect(persistent.db_path, check_same_thread=False)
    blocker.execute("BEGIN IMMEDIATE")

    def unlock():
        # A separate watchdog releases even the pre-fix blocking implementation.
        # The invariant is who releases it, not a wall-clock speed assertion.
        released_by_loop.append(release.wait(timeout=5))
        blocker.rollback()
        blocker.close()

    watchdog = threading.Thread(target=unlock)
    watchdog.start()
    task = asyncio.create_task(_client().resume(run_id, ApprovalDecision(True)))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert (
            not released_by_loop
        ), "SQLite admission blocked the event loop until watchdog release"
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            replay.assert_not_awaited()
        release.set()
        assert await asyncio.to_thread(claimed.wait, 5)
        if not cancel:
            assert (await task).content == "done"
            replay.assert_awaited_once()
        else:
            replay.assert_not_awaited()
        assert ProjectDbPausedRunStore(persistent.db_path).get(run_id).status == "resumed"
        # Cancellation may consume admission, but can never reopen it for replay.
        with pytest.raises(ValueError, match="already resumed"):
            await _client().resume(run_id, ApprovalDecision(True))
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(watchdog.join, 5)
        assert not watchdog.is_alive()
    assert released_by_loop == [True]


async def test_competing_persistent_resumes_dispatch_once(tmp_path, monkeypatch):
    import asyncio
    from unittest.mock import AsyncMock

    from victor.agent import durable_resume
    from victor.agent.paused_run_store import ProjectDbPausedRunStore

    persistent = ProjectDbPausedRunStore(tmp_path / "state.db")
    set_paused_run_store(persistent)
    run_id = _save(persistent)
    # Force both workers to observe pending before either attempts the atomic claim.
    import threading

    barrier = threading.Barrier(2)
    get = persistent.get

    def pending_snapshot(run_id):
        paused = get(run_id)
        assert paused.status == "awaiting_approval"
        barrier.wait(timeout=5)
        return paused

    monkeypatch.setattr(persistent, "get", pending_snapshot)
    replay = AsyncMock(return_value=durable_resume.ResumeResult(final_content="done"))
    monkeypatch.setattr(durable_resume, "resume_paused_run", replay)
    outcomes = await asyncio.gather(
        _client().resume(run_id, ApprovalDecision(True)),
        _client().resume(run_id, ApprovalDecision(True)),
        return_exceptions=True,
    )
    assert sum(isinstance(outcome, ValueError) for outcome in outcomes) == 1
    assert sum(getattr(outcome, "content", None) == "done" for outcome in outcomes) == 1
    replay.assert_awaited_once()
    assert ProjectDbPausedRunStore(persistent.db_path).get(run_id).status == "resumed"


async def test_persistent_claim_commits_then_errors_without_dispatch_or_reopening(
    tmp_path, monkeypatch
):
    import sqlite3
    from unittest.mock import AsyncMock

    from victor.agent import durable_resume
    from victor.agent.paused_run_store import ProjectDbPausedRunStore

    persistent = ProjectDbPausedRunStore(tmp_path / "state.db")
    set_paused_run_store(persistent)
    run_id = _save(persistent)
    mark = persistent.mark_resumed

    def lost_ack(run_id):
        assert mark(run_id)
        raise sqlite3.OperationalError("claim acknowledgement lost")

    monkeypatch.setattr(persistent, "mark_resumed", lost_ack)
    replay = AsyncMock()
    monkeypatch.setattr(durable_resume, "resume_paused_run", replay)
    with pytest.raises(sqlite3.OperationalError, match="acknowledgement lost"):
        await _client().resume(run_id, ApprovalDecision(True))
    replay.assert_not_awaited()
    reopened = ProjectDbPausedRunStore(persistent.db_path)
    assert reopened.get(run_id).status == "resumed"
    assert not reopened.mark_resumed(run_id)


async def test_injected_store_retains_caller_thread_affinity(monkeypatch, _store):
    import threading

    from victor.agent import durable_resume

    owner = threading.get_ident()
    calls = []
    for name in ("expire_pending", "get", "mark_resumed"):
        method = getattr(_store, name)

        def checked(*args, _name=name, _method=method, **kwargs):
            assert threading.get_ident() == owner
            calls.append(_name)
            return _method(*args, **kwargs)

        monkeypatch.setattr(_store, name, checked)

    async def resume(*args):
        return durable_resume.ResumeResult(final_content="done")

    monkeypatch.setattr(durable_resume, "resume_paused_run", resume)
    result = await _client().resume(_save(_store), ApprovalDecision(True))
    assert result.content == "done"
    assert calls == ["expire_pending", "get", "mark_resumed"]


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf"), 61, "10"])
async def test_receipt_lookup_rejects_invalid_timeout_before_storage(timeout):
    with pytest.raises(ValueError, match="timeout"):
        await _client().reconcile_action("unused", timeout_seconds=timeout)


@pytest.mark.parametrize(
    "kind",
    [
        "identity_version",
        "empty_backend",
        "receipt_version",
        "action_hash",
        "receipt_url",
        "outcome",
    ],
)
def test_receipt_contract_rejects_malformed_evidence(kind):
    from victor.framework.action_recovery import BackendReceipt, RecoveryIdentity

    with pytest.raises(ValueError):
        if kind == "identity_version":
            RecoveryIdentity("adapter-v1", "tenant:prod", version=True)
        elif kind == "empty_backend":
            RecoveryIdentity("adapter-v1", "")
        else:
            overrides = {
                "receipt_version": {"version": True},
                "action_hash": {"action_id": "wrong"},
                "receipt_url": {"receipt_id": "https://credential@example.invalid"},
                "outcome": {"outcome": "not_found"},
            }[kind]
            BackendReceipt(
                **{
                    "action_id": "a" * 64,
                    "binding_digest": "b" * 64,
                    "identity": RecoveryIdentity("adapter-v1", "tenant:prod"),
                    "receipt_id": "receipt-1",
                    **overrides,
                }
            )
