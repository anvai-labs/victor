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
