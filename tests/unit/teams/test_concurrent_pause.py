"""ADR-023 pillar 2b: concurrent durable pause/resume for the PARALLEL formation.

A concurrent wave can have *several* members hit an approval gate at once. When a checkpointer
is configured (durable run), every awaiting member is collected after the wave into a single
multi-pause aggregate (``__awaiting_approvals__``) + one pause checkpoint recording all pending
approvals; the members that finished are checkpointed as completed. Re-running on the same
thread_id skips the completed members and re-runs exactly the paused set. Exercised with fake
members (no orchestrator core), mirroring the SEQUENTIAL pause tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from victor.framework.graph_checkpoint import MemoryCheckpointer
from victor.teams import TeamFormation, UnifiedTeamCoordinator


def _approval(member_id: str) -> Dict[str, Any]:
    return {"id": f"req-{member_id}", "tool_name": "run_command", "title": f"cmd-{member_id}"}


class _FakeMember:
    """PARALLEL member that can report awaiting-approval, recording how often it ran."""

    def __init__(self, member_id: str, *, pause: bool = False) -> None:
        self.id = member_id
        self._pause = pause
        self.calls = 0

    async def execute_task(self, *args: Any, **kwargs: Any) -> dict:
        self.calls += 1
        if self._pause:
            return {
                "output": "",
                "success": False,
                "metadata": {
                    "awaiting_approval": True,
                    "approval_request": _approval(self.id),
                },
            }
        return {"output": f"ok-{self.id}", "success": True}

    async def receive_message(self, *args: Any, **kwargs: Any) -> None:
        return None


def _coordinator(members: List[_FakeMember], checkpointer: Any = None) -> UnifiedTeamCoordinator:
    coord = UnifiedTeamCoordinator(lightweight_mode=True, checkpointer=checkpointer)
    for m in members:
        coord.add_member(m)
    coord.set_formation(TeamFormation.PARALLEL)
    return coord


def _pauses(checkpoints: List[Any]) -> List[Any]:
    return [c for c in checkpoints if c.metadata.get("awaiting_approval")]


# ── pause / save (multi-member aggregate) ─────────────────────────


async def test_parallel_pauses_on_multiple_awaiting_members() -> None:
    cp = MemoryCheckpointer()
    # a1 and a3 await approval; a0 and a2 complete.
    members = [
        _FakeMember("a0"),
        _FakeMember("a1", pause=True),
        _FakeMember("a2"),
        _FakeMember("a3", pause=True),
    ]
    result = await _coordinator(members, cp).execute_task("go", {"thread_id": "t1"})

    # All four ran once (concurrent wave — no early stop); two came back awaiting.
    assert all(m.calls == 1 for m in members)

    # Paused aggregate surfaces every pending approval.
    assert result["success"] is False
    assert result["status"] == "awaiting_approval"
    assert set(result["paused_member_ids"]) == {"a1", "a3"}
    approvals = {a["member_id"]: a["approval_request"] for a in result["awaiting_approvals"]}
    assert approvals == {"a1": _approval("a1"), "a3": _approval("a3")}
    assert result["thread_id"] == "t1"

    # Only the completed members are in the results (awaiting ones excluded → they re-run).
    assert set(result["member_results"].keys()) == {"a0", "a2"}

    # Exactly one pause checkpoint, recording all pending approvals and only completed members.
    pauses = _pauses(await cp.list("t1"))
    assert len(pauses) == 1
    state = pauses[0].state
    assert set(state["completed_member_ids"]) == {"a0", "a2"}
    assert {a["member_id"] for a in state["awaiting_approvals"]} == {"a1", "a3"}


async def test_single_awaiting_member_uses_the_plural_aggregate() -> None:
    # Even one awaiting member in a concurrent wave surfaces via the multi-pause aggregate
    # (not the sequential singular key), so clients read a uniform concurrent shape.
    cp = MemoryCheckpointer()
    members = [_FakeMember("a0"), _FakeMember("a1", pause=True)]
    result = await _coordinator(members, cp).execute_task("go", {"thread_id": "t1"})

    assert result["status"] == "awaiting_approval"
    assert result["paused_member_ids"] == ["a1"]
    assert result.get("paused_member_id") is None  # singular seq key not set on the concurrent path
    assert set(result["member_results"].keys()) == {"a0"}


# ── resume / re-run (only the paused set) ─────────────────────────


async def test_resume_reruns_only_the_paused_members() -> None:
    cp = MemoryCheckpointer()
    # First run: a1, a3 pause; a0, a2 complete.
    await _coordinator(
        [
            _FakeMember("a0"),
            _FakeMember("a1", pause=True),
            _FakeMember("a2"),
            _FakeMember("a3", pause=True),
        ],
        cp,
    ).execute_task("go", {"thread_id": "t1"})

    # Resume: a1, a3 now proceed (approval granted → fresh members no longer pause).
    resumed = [
        _FakeMember("a0"),
        _FakeMember("a1"),
        _FakeMember("a2"),
        _FakeMember("a3"),
    ]
    result = await _coordinator(resumed, cp).execute_task("go", {"thread_id": "t1"})

    assert resumed[0].calls == 0  # completed member skipped
    assert resumed[2].calls == 0  # completed member skipped
    assert resumed[1].calls == 1  # paused member re-runs
    assert resumed[3].calls == 1  # paused member re-runs
    assert result["success"] is True
    assert result.get("status") != "awaiting_approval"
    assert set(result["member_results"].keys()) == {"a0", "a1", "a2", "a3"}


# ── opt-out ───────────────────────────────────────────────────────


async def test_no_checkpointer_does_not_pause() -> None:
    # Without a checkpointer, durable pause is not armed: an awaiting member is just a
    # non-success result and the wave completes — no aggregate, byte-identical to today.
    members = [_FakeMember("a0"), _FakeMember("a1", pause=True), _FakeMember("a2")]
    result = await _coordinator(members).execute_task("go", {"thread_id": "t1"})

    assert all(m.calls == 1 for m in members)
    assert result.get("status") != "awaiting_approval"
    assert "awaiting_approvals" not in result
    assert set(result["member_results"].keys()) == {"a0", "a1", "a2"}
