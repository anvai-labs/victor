"""FEP-0029 Phase 2: durable project.db persistence for paused runs.

A pause recorded by :class:`ProjectDbPausedRunStore` survives a process restart — a *fresh* store
instance pointed at the same database reads it back, with the approval request / pending tool
round-tripped through JSON. This is what lets a single-agent turn be parked and resumed later (or on
another process). Exercised against a temporary database file (no real project needed).
"""

from __future__ import annotations

from pathlib import Path

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
