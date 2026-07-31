"""ADR-023: durable checkpoint/resume for the PARALLEL formation.

A PARALLEL wave checkpoints the cumulative completed set as each member finishes (the
completion handler is serialized under a lock; member execution stays concurrent). A resumed
run skips the members already in the checkpoint's completed set and re-runs the rest. Answers
the FEP's "well-defined completed set" open question for concurrent formations.
"""

from __future__ import annotations

import asyncio
from typing import Any, List

from victor.coordination.formations.base import TeamContext
from victor.coordination.formations.parallel import ParallelFormation
from victor.framework.graph_checkpoint import MemoryCheckpointer, WorkflowCheckpoint
from victor.teams import TeamFormation, UnifiedTeamCoordinator
from victor.teams.types import AgentMessage, MemberResult, MessageType


class _FakeMember:
    def __init__(self, member_id: str) -> None:
        self.id = member_id
        self.calls = 0

    async def execute_task(self, *args: Any, **kwargs: Any) -> dict:
        self.calls += 1
        return {"output": f"ok-{self.id}", "success": True}

    async def receive_message(self, *args: Any, **kwargs: Any) -> None:
        return None


def _coordinator(members: List[_FakeMember], checkpointer: Any = None) -> UnifiedTeamCoordinator:
    coord = UnifiedTeamCoordinator(lightweight_mode=True, checkpointer=checkpointer)
    for m in members:
        coord.add_member(m)
    coord.set_formation(TeamFormation.PARALLEL)
    return coord


# ── save ──────────────────────────────────────────────────────────


async def test_parallel_checkpoints_the_completed_set() -> None:
    cp = MemoryCheckpointer()
    members = [_FakeMember(f"a{i}") for i in range(3)]
    result = await _coordinator(members, cp).execute_task("go", {"thread_id": "t1"})
    assert result["success"] is True

    checkpoints = [c for c in await cp.list("t1") if c.metadata.get("team_node_id")]
    assert len(checkpoints) == 3  # one per member as it completed
    latest = max(checkpoints, key=lambda c: c.timestamp)
    # Order-independent: the final checkpoint holds every member.
    assert set(latest.state["completed_member_ids"]) == {"a0", "a1", "a2"}


# ── resume ────────────────────────────────────────────────────────


async def test_parallel_resume_skips_completed_reruns_rest() -> None:
    cp = MemoryCheckpointer()
    done = MemberResult(member_id="a0", success=True, output="ok-a0")
    await cp.save(
        WorkflowCheckpoint(
            checkpoint_id="t1:UnifiedTeam:member:0",
            thread_id="t1",
            node_id="UnifiedTeam:member:a0",
            state={
                "completed_member_ids": ["a0"],
                "member_results": [done.to_dict()],
                "shared_state": {},
            },
            timestamp=1.0,
            metadata={"team_node_id": "UnifiedTeam", "member_id": "a0", "formation": "parallel"},
        )
    )

    members = [_FakeMember("a0"), _FakeMember("a1"), _FakeMember("a2")]
    result = await _coordinator(members, cp).execute_task("go", {"thread_id": "t1"})

    assert members[0].calls == 0  # completed member skipped
    assert members[1].calls == 1 and members[2].calls == 1  # rest re-run
    assert set(result["member_results"].keys()) == {"a0", "a1", "a2"}


# ── concurrency preserved ─────────────────────────────────────────


async def test_execution_stays_concurrent_only_completion_is_serialized() -> None:
    # Two members must be able to execute simultaneously — the lock guards only the
    # completion checkpoint, not execution.
    both_running = asyncio.Event()
    entered = 0

    class _OverlapMember:
        def __init__(self, member_id: str) -> None:
            self.id = member_id

        async def execute(self, task: Any, context: Any) -> MemberResult:
            nonlocal entered
            entered += 1
            if entered >= 2:
                both_running.set()
            await asyncio.wait_for(both_running.wait(), timeout=2.0)  # deadlocks if serialized
            return MemberResult(member_id=self.id, success=True, output="ok")

    ctx = TeamContext(team_id="t", formation="parallel")
    agents = [_OverlapMember("a0"), _OverlapMember("a1")]
    results = await ParallelFormation().execute(agents, ctx, _task())
    assert {r.member_id for r in results} == {"a0", "a1"}
    assert both_running.is_set()  # both were in-flight at once


# ── opt-out ───────────────────────────────────────────────────────


async def test_no_checkpointer_is_unchanged() -> None:
    members = [_FakeMember(f"a{i}") for i in range(3)]
    result = await _coordinator(members).execute_task("go", {"thread_id": "t1"})
    assert all(m.calls == 1 for m in members)
    assert set(result["member_results"].keys()) == {"a0", "a1", "a2"}


def _task() -> AgentMessage:
    return AgentMessage(sender_id="coordinator", content="go", message_type=MessageType.TASK)
