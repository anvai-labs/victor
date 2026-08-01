"""ADR-023 increment 1: member-granular checkpoint/resume for SEQUENTIAL teams."""

from __future__ import annotations

from typing import Any, List

from victor.framework.graph_checkpoint import MemoryCheckpointer, WorkflowCheckpoint
from victor.teams import TeamFormation, UnifiedTeamCoordinator
from victor.teams.types import MemberResult


class _FakeMember:
    """Minimal team member that records how many times it ran."""

    def __init__(self, member_id: str, output: str = "ok") -> None:
        self.id = member_id
        self._output = output
        self.calls = 0

    async def execute_task(self, *args: Any, **kwargs: Any) -> dict:
        self.calls += 1
        return {"output": self._output, "success": True}

    async def receive_message(self, *args: Any, **kwargs: Any) -> None:
        return None


def _coordinator(members: List[_FakeMember], checkpointer: Any = None) -> UnifiedTeamCoordinator:
    coord = UnifiedTeamCoordinator(lightweight_mode=True, checkpointer=checkpointer)
    for member in members:
        coord.add_member(member)
    coord.set_formation(TeamFormation.SEQUENTIAL)
    return coord


# ── save ──────────────────────────────────────────────────────────


async def test_sequential_checkpoints_after_each_member() -> None:
    cp = MemoryCheckpointer()
    members = [_FakeMember(f"m{i}") for i in range(3)]
    await _coordinator(members, cp).execute_task("do it", {"thread_id": "t1"})

    checkpoints = [c for c in await cp.list("t1") if c.metadata.get("team_node_id")]
    assert len(checkpoints) == 3  # one per member
    latest = max(checkpoints, key=lambda c: c.timestamp)
    assert latest.state["completed_member_ids"] == ["m0", "m1", "m2"]
    assert len(latest.state["member_results"]) == 3
    assert latest.metadata["formation"] == "sequential"


# ── resume ────────────────────────────────────────────────────────


async def test_resume_skips_completed_member() -> None:
    cp = MemoryCheckpointer()
    # Simulate a crash after member 0 completed: seed its checkpoint.
    done = MemberResult(member_id="m0", success=True, output="out0")
    await cp.save(
        WorkflowCheckpoint(
            checkpoint_id="t1:UnifiedTeam:member:0",
            thread_id="t1",
            node_id="UnifiedTeam:member:m0",
            state={
                "completed_member_ids": ["m0"],
                "member_results": [done.to_dict()],
                "shared_state": {},
                "last_output": "out0",
                "last_agent_id": "m0",
            },
            timestamp=1.0,
            metadata={"team_node_id": "UnifiedTeam", "member_id": "m0", "formation": "sequential"},
        )
    )

    members = [_FakeMember("m0"), _FakeMember("m1"), _FakeMember("m2")]
    result = await _coordinator(members, cp).execute_task("do it", {"thread_id": "t1"})

    assert members[0].calls == 0  # m0 skipped (resumed from checkpoint)
    assert members[1].calls == 1
    assert members[2].calls == 1
    # Final aggregate carries all three members (resumed + freshly run).
    assert set(result["member_results"].keys()) == {"m0", "m1", "m2"}


# ── opt-out ───────────────────────────────────────────────────────


async def test_no_checkpointer_is_unchanged() -> None:
    members = [_FakeMember(f"m{i}") for i in range(3)]
    result = await _coordinator(members).execute_task("do it", {"thread_id": "t1"})
    assert all(m.calls == 1 for m in members)  # all run, none skipped
    assert result["success"] is True
    assert set(result["member_results"].keys()) == {"m0", "m1", "m2"}


async def test_checkpointer_without_thread_id_is_inert() -> None:
    cp = MemoryCheckpointer()
    members = [_FakeMember(f"m{i}") for i in range(2)]
    await _coordinator(members, cp).execute_task("do it", {})  # no thread_id
    assert await cp.list("t1") == []  # nothing checkpointed without a thread_id
    assert all(m.calls == 1 for m in members)
