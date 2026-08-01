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


async def test_resume_returns_ok_taskresult_and_is_single_use(monkeypatch: Any, _store: Any) -> None:
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
