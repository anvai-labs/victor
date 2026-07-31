"""ADR-023 pillar 2b: durable member pause/resume for SEQUENTIAL teams.

A member reporting ``metadata["awaiting_approval"]`` durably pauses the team (when a
checkpointer is configured): the formation stops, a pause checkpoint is persisted, and the
coordinator returns a paused aggregate. Re-running on the same thread_id skips completed
members and re-runs the paused one. Exercised with a fake member (no orchestrator core).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from victor.framework.graph_checkpoint import MemoryCheckpointer
from victor.teams import TeamFormation, UnifiedTeamCoordinator

_APPROVAL_REQUEST = {"id": "req-1", "tool_name": "run_command", "title": "run_command"}


class _FakeMember:
    """Team member that can report awaiting-approval, recording how often it ran."""

    def __init__(
        self, member_id: str, *, pause: bool = False, approval_request: Optional[Dict] = None
    ) -> None:
        self.id = member_id
        self._pause = pause
        self._approval_request = approval_request or _APPROVAL_REQUEST
        self.calls = 0

    async def execute_task(self, *args: Any, **kwargs: Any) -> dict:
        self.calls += 1
        if self._pause:
            return {
                "output": "",
                "success": False,
                "metadata": {
                    "awaiting_approval": True,
                    "approval_request": self._approval_request,
                },
            }
        return {"output": f"ok-{self.id}", "success": True}

    async def receive_message(self, *args: Any, **kwargs: Any) -> None:
        return None


def _coordinator(members: List[_FakeMember], checkpointer: Any = None) -> UnifiedTeamCoordinator:
    coord = UnifiedTeamCoordinator(lightweight_mode=True, checkpointer=checkpointer)
    for member in members:
        coord.add_member(member)
    coord.set_formation(TeamFormation.SEQUENTIAL)
    return coord


# ── pause / save ──────────────────────────────────────────────────


async def test_pauses_and_checkpoints_at_awaiting_member() -> None:
    cp = MemoryCheckpointer()
    members = [_FakeMember("m0"), _FakeMember("m1", pause=True), _FakeMember("m2")]
    result = await _coordinator(members, cp).execute_task("do it", {"thread_id": "t1"})

    # Formation stopped at the paused member; m2 never ran.
    assert members[0].calls == 1
    assert members[1].calls == 1
    assert members[2].calls == 0

    # Paused aggregate.
    assert result["success"] is False
    assert result["status"] == "awaiting_approval"
    assert result["paused_member_id"] == "m1"
    assert result["approval_request"] == _APPROVAL_REQUEST
    assert result["thread_id"] == "t1"
    # Only the completed member is in the aggregate.
    assert set(result["member_results"].keys()) == {"m0"}

    # A durable pause checkpoint was written (paused member excluded from completed).
    pauses = [c for c in await cp.list("t1") if c.metadata.get("awaiting_approval")]
    assert len(pauses) == 1
    state = pauses[0].state
    assert state["completed_member_ids"] == ["m0"]
    assert state["paused_member_id"] == "m1"
    assert state["approval_request"] == _APPROVAL_REQUEST


# ── resume / re-run ───────────────────────────────────────────────


async def test_resume_reruns_paused_member_and_continues() -> None:
    cp = MemoryCheckpointer()
    # First run pauses at m1.
    await _coordinator(
        [_FakeMember("m0"), _FakeMember("m1", pause=True), _FakeMember("m2")], cp
    ).execute_task("do it", {"thread_id": "t1"})

    # Resume: m1 now proceeds (approval granted). Fresh members + same checkpointer.
    resumed = [_FakeMember("m0"), _FakeMember("m1"), _FakeMember("m2")]
    result = await _coordinator(resumed, cp).execute_task(
        "do it", {"thread_id": "t1", "approval_decision": {"member_id": "m1", "approved": True}}
    )

    assert resumed[0].calls == 0  # completed member skipped
    assert resumed[1].calls == 1  # paused member re-runs
    assert resumed[2].calls == 1  # and the rest continue
    assert result["success"] is True
    assert result.get("status") != "awaiting_approval"
    assert set(result["member_results"].keys()) == {"m0", "m1", "m2"}
    # The human's decision was surfaced to the re-run.
    assert result["shared_context"]["approval_decision"] == {"member_id": "m1", "approved": True}


# ── opt-out ───────────────────────────────────────────────────────


async def test_no_checkpointer_does_not_pause() -> None:
    # Without a checkpointer the awaiting-approval flag is inert: the member flows through
    # like any non-success result and the run continues — byte-identical to today.
    members = [_FakeMember("m0"), _FakeMember("m1", pause=True), _FakeMember("m2")]
    result = await _coordinator(members).execute_task("do it", {"thread_id": "t1"})

    assert all(m.calls == 1 for m in members)  # all ran; no pause
    assert result.get("status") != "awaiting_approval"
    assert set(result["member_results"].keys()) == {"m0", "m1", "m2"}
