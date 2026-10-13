"""FEP-0029 Phase 2: durable project.db persistence for paused runs.

A pause recorded by :class:`ProjectDbPausedRunStore` survives a process restart — a *fresh* store
instance pointed at the same database reads it back, with the approval request / pending tool
round-tripped through JSON. This is what lets a single-agent turn be parked and resumed later (or on
another process). Exercised against a temporary database file (no real project needed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from victor.agent.paused_run_store import (
    PausedRunStoreProtocol,
    ProjectDbPausedRunStore,
)


def _store(tmp_path: Path) -> ProjectDbPausedRunStore:
    return ProjectDbPausedRunStore(db_path=tmp_path / ".victor" / "project.db")


_APPROVAL = {"id": "req-1", "title": "Approve tool: run_command", "context": {"x": 1}}
_TOOL = {"tool_name": "run_command", "arguments": {"cmd": "rm -rf /tmp/x"}}


def test_conforms_to_protocol(tmp_path: Path) -> None:
    assert isinstance(_store(tmp_path), PausedRunStoreProtocol)


def test_save_survives_a_fresh_store_instance(tmp_path: Path) -> None:
    db = tmp_path / ".victor" / "project.db"
    run_id = ProjectDbPausedRunStore(db_path=db).save(
        session_id="sess-1",
        agent_id="agent-1",
        approval_request=_APPROVAL,
        pending_tool=_TOOL,
        created_at=123.0,
        metadata={"stage": "chat"},
    )

    # A brand-new store instance (simulating a restart) reads the pause back verbatim.
    reopened = ProjectDbPausedRunStore(db_path=db)
    run = reopened.get(run_id)
    assert run is not None
    assert run.session_id == "sess-1"
    assert run.agent_id == "agent-1"
    assert run.approval_request == _APPROVAL  # JSON round-trip
    assert run.pending_tool == _TOOL
    assert run.metadata == {"stage": "chat"}
    assert run.created_at == 123.0
    assert run.status == "awaiting_approval"


def test_mark_resumed_is_durable_and_single_use(tmp_path: Path) -> None:
    db = tmp_path / ".victor" / "project.db"
    store = ProjectDbPausedRunStore(db_path=db)
    run_id = store.save(session_id="s", agent_id=None, approval_request=_APPROVAL)

    assert store.mark_resumed(run_id) is True
    # Durable: a fresh instance sees it resumed and won't resume it again (single-use).
    reopened = ProjectDbPausedRunStore(db_path=db)
    assert reopened.get(run_id).status == "resumed"
    assert reopened.mark_resumed(run_id) is False


def test_list_pending_excludes_resumed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = store.save(session_id="s", agent_id=None, approval_request=_APPROVAL)
    b = store.save(session_id="s", agent_id=None, approval_request=_APPROVAL)
    store.mark_resumed(a)

    pending = [r.run_id for r in store.list_pending()]
    assert pending == [b]


def test_get_unknown_returns_none(tmp_path: Path) -> None:
    assert _store(tmp_path).get("nope") is None


def test_null_pending_tool_and_metadata_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run_id = store.save(session_id=None, agent_id=None, approval_request=_APPROVAL)
    run = store.get(run_id)
    assert run is not None
    assert run.pending_tool is None
    assert run.metadata == {}
    assert run.session_id is None


def test_in_memory_store_snapshots_nested_approval_data():
    from victor.agent.paused_run_store import InMemoryPausedRunStore

    store = InMemoryPausedRunStore()
    request = {"context": {"arguments": {"payload": "approved"}}}
    pending = {"binding": {"payload": "hash"}}
    run_id = store.save(
        session_id="s", agent_id="a", approval_request=request, pending_tool=pending
    )
    request["context"]["arguments"]["payload"] = "changed"
    pending["binding"]["payload"] = "changed"
    run = store.get(run_id)
    assert run.approval_request["context"]["arguments"]["payload"] == "approved"
    assert run.pending_tool["binding"]["payload"] == "hash"
    run.pending_tool["binding"]["payload"] = "changed again"
    store.list_pending()[0].approval_request["context"].clear()
    assert store.get(run_id).pending_tool["binding"]["payload"] == "hash"
    assert store.get(run_id).approval_request["context"]["arguments"]["payload"] == "approved"


def _claimed_action(store):
    binding = {"version": 1, "session_id": "s", "tool_name": "submit", "call_id": "c1"}
    run_id = store.save(
        session_id="s",
        agent_id="a",
        approval_request={"id": "approval"},
        pending_tool={"tool_name": "submit", "binding": binding},
        created_at=1,
    )
    assert store.mark_resumed(run_id)
    return run_id, binding


@pytest.mark.parametrize("observation", [None, "unknown"])
def test_action_intent_survives_restart_and_is_never_purged(tmp_path, observation):
    import pytest
    from victor.agent.action_observation import ActionStateError

    store = _store(tmp_path)
    run_id, binding = _claimed_action(store)
    action = store.begin_action(run_id, binding)
    if observation:
        action = store.observe_action(run_id, action["action_id"], observation, None)
    reopened = _store(tmp_path)
    assert reopened.get(run_id).action == action
    assert action["state"] == (observation or "pending")
    assert reopened.purge(before=100) == 0
    with pytest.raises(ActionStateError):
        reopened.begin_action(run_id, binding)
    assert reopened.get(run_id).action["state"] == (observation or "pending")


def test_action_observation_is_compare_and_set_and_retained(tmp_path):
    import pytest
    from victor.agent.action_observation import ActionStateError

    store = _store(tmp_path)
    run_id, binding = _claimed_action(store)
    action = store.begin_action(run_id, binding)
    with pytest.raises(ActionStateError):
        store.observe_action(run_id, "wrong-key", "returned", True)
    store.observe_action(run_id, action["action_id"], "returned", True)
    with pytest.raises(ActionStateError):
        store.observe_action(run_id, action["action_id"], "unknown", None)
    reopened = _store(tmp_path)
    assert reopened.get(run_id).action["reported_success"] is True
    assert reopened.get(run_id).action["state"] == "returned"
    assert reopened.get(run_id).action["backend_receipt"] is None
    assert reopened.purge(before=100) == 0


def test_action_intent_rejects_changed_binding_and_competing_process_owners(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import pytest
    from victor.agent.action_observation import ActionStateError

    store = _store(tmp_path)
    run_id, binding = _claimed_action(store)
    with pytest.raises(ActionStateError):
        store.begin_action(run_id, {**binding, "session_id": "other"})
    assert store.get(run_id).action is None

    def claim(_):
        try:
            _store(tmp_path).begin_action(run_id, binding)
            return True
        except ActionStateError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(claim, range(2))) == [False, True]


def test_legacy_schema_migrates_without_changing_approval_or_single_use_claim(tmp_path):
    import sqlite3

    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as connection:
        connection.execute("""CREATE TABLE paused_run (
            run_id TEXT PRIMARY KEY, session_id TEXT, agent_id TEXT,
            approval_request TEXT NOT NULL, pending_tool TEXT,
            status TEXT NOT NULL DEFAULT 'awaiting_approval', created_at REAL,
            resumed_at REAL, metadata TEXT)""")
        connection.execute(
            "INSERT INTO paused_run (run_id, session_id, approval_request) VALUES (?, ?, ?)",
            ("legacy", "s", '{"id":"req"}'),
        )
    store = ProjectDbPausedRunStore(db)
    run = store.get("legacy")
    assert run.approval_request == {"id": "req"} and run.action is None
    assert store.mark_resumed("legacy")
    reopened = ProjectDbPausedRunStore(db)
    assert not reopened.mark_resumed("legacy")
    assert reopened.get("legacy").action is None


def test_schema_failure_releases_connection_and_allows_retry(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from victor.agent import paused_run_store
    import sqlite3

    store = _store(tmp_path)
    connection = MagicMock()
    connection.execute.side_effect = sqlite3.OperationalError("schema unavailable")
    with monkeypatch.context() as patch:
        patch.setattr(paused_run_store.sqlite3, "connect", lambda *a, **kw: connection)
        with pytest.raises(sqlite3.OperationalError):
            store.get("missing")
    connection.close.assert_called_once()
    assert store.get("missing") is None


@pytest.mark.parametrize("receipt_first", [False, True])
def test_receipt_and_invocation_observation_preserve_each_others_evidence(tmp_path, receipt_first):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from victor.framework.action_recovery import ActionLookup, BackendReceipt, RecoveryIdentity

    store = _store(tmp_path)
    run_id, binding = _claimed_action(store)
    identity = RecoveryIdentity("test-adapter-v1", "tenant:test-ledger")
    action = store.begin_action(run_id, binding, recovery=identity.to_dict())
    request = ActionLookup(action["action_id"], action["binding_digest"], identity)
    receipt = BackendReceipt(request.action_id, request.binding_digest, identity, "receipt-1")
    first_committed = threading.Event()

    def write_receipt():
        if not receipt_first:
            assert first_committed.wait(3)
        try:
            return _store(tmp_path).retain_receipt(run_id, request, receipt)
        finally:
            if receipt_first:
                first_committed.set()

    def write_observation():
        if receipt_first:
            assert first_committed.wait(3)
        try:
            return _store(tmp_path).observe_action(run_id, request.action_id, "returned", True)
        finally:
            if not receipt_first:
                first_committed.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(write_receipt), pool.submit(write_observation)
        a.result(timeout=5)
        b.result(timeout=5)
    saved = _store(tmp_path).get(run_id)
    assert saved.action["state"] == "returned" and saved.action["reported_success"] is True
    assert saved.action["backend_receipt"] == receipt.to_dict()
    assert saved.status == "resumed" and not store.mark_resumed(run_id)


async def test_async_pause_snapshots_inputs_and_store_before_worker_yields(tmp_path, monkeypatch):
    import asyncio
    import threading
    from victor.agent.paused_run_store import (
        InMemoryPausedRunStore,
        record_pause_from_approval_async,
        reset_paused_run_store,
        set_paused_run_store,
    )
    from victor.framework.hitl import ApprovalRequest

    store = _store(tmp_path)
    replacement = InMemoryPausedRunStore()
    request = ApprovalRequest(
        id="bound",
        title="submit",
        description="",
        context={"tool_name": "submit", "arguments": {}, "action_binding": {"payload": "hash"}},
    )
    metadata = {"nested": {"owner": "original"}}
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    save = store.save

    def delayed(**kwargs):
        entered.set()
        try:
            assert release.wait(5), "pause worker release timed out"
            return save(**kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(store, "save", delayed)
    set_paused_run_store(store)
    task = asyncio.create_task(
        record_pause_from_approval_async(
            request,
            session_id="s",
            agent_id="a",
            metadata=metadata,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        request.context["arguments"]["unexpected"] = True
        request.context["action_binding"]["payload"] = "changed"
        metadata["nested"]["owner"] = "changed"
        set_paused_run_store(replacement)
        release.set()
        run_id, approval = await task
        run = ProjectDbPausedRunStore(store.db_path).get(run_id)
        assert run.approval_request == approval
        assert run.pending_tool["arguments"] == {}
        assert run.pending_tool["binding"] == {
            "payload": "hash",
            "session_id": "s",
            "agent_id": "a",
            "request_id": "bound",
            "expires_at": request.created_at + request.timeout_seconds,
        }
        assert run.metadata == {"nested": {"owner": "original"}}
        assert not replacement.list_pending()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await asyncio.to_thread(finished.wait, 5)
        reset_paused_run_store()


async def test_custom_pause_store_stays_on_caller_thread_and_matches_sync_contract():
    import threading
    from victor.agent.paused_run_store import (
        InMemoryPausedRunStore,
        record_pause_from_approval,
        record_pause_from_approval_async,
        reset_paused_run_store,
        set_paused_run_store,
    )
    from victor.framework.hitl import ApprovalRequest

    caller = threading.get_ident()

    class CallerThreadStore(InMemoryPausedRunStore):
        def save(self, **kwargs):
            assert threading.get_ident() == caller
            return super().save(**kwargs)

    store = CallerThreadStore()
    set_paused_run_store(store)
    request = ApprovalRequest(
        id="r",
        title="legacy aliases",
        description="",
        context={"tool": "submit", "args": {"value": "original"}},
    )
    try:
        sync_id, sync_request = record_pause_from_approval(request, session_id="s", agent_id="a")
        async_id, async_request = await record_pause_from_approval_async(
            request, session_id="s", agent_id="a"
        )
        assert sync_request == async_request == request.to_dict()
        assert (
            store.get(sync_id).pending_tool
            == store.get(async_id).pending_tool
            == {"tool_name": "submit", "arguments": {"value": "original"}}
        )
        assert sync_id != async_id
    finally:
        reset_paused_run_store()
