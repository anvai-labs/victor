"""ADR-023 increment 1: member-granular checkpoint/resume for SEQUENTIAL teams."""

from __future__ import annotations

from typing import Any, List

import pytest

from victor.framework.graph_checkpoint import MemoryCheckpointer, WorkflowCheckpoint
from victor.teams import TeamFormation, UnifiedTeamCoordinator
from victor.teams.types import MemberResult


class _FakeMember:
    """Minimal team member that records how many times it ran."""

    def __init__(self, member_id: str, output: str = "ok", *, pause: bool = False) -> None:
        self.id = member_id
        self._output = output
        self.calls = 0
        self.pause = pause

    async def execute_task(self, *args: Any, **kwargs: Any) -> dict:
        self.calls += 1
        if self.pause:
            return {
                "output": "",
                "success": False,
                "metadata": {
                    "awaiting_approval": True,
                    "approval_request": {"id": "approval", "title": "submit"},
                },
            }
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


class _FailingCheckpointer(MemoryCheckpointer):
    """Fail one authoritative read/write, optionally after retaining its snapshot."""

    def __init__(self, operation, *, committed=False):
        super().__init__()
        self.operation = operation
        self.committed = committed
        self.attempts = []

    async def list(self, thread_id):
        if self.operation == "load":
            raise OSError("private backend detail")
        return await super().list(thread_id)

    async def save(self, checkpoint):
        operation = (
            "save_batch_pause"
            if "awaiting_approvals" in checkpoint.state
            else "save_pause" if checkpoint.metadata.get("awaiting_approval") else "save_member"
        )
        self.attempts.append(operation)
        if operation == self.operation:
            if self.committed:
                await super().save(checkpoint)
            raise OSError("private backend detail")
        await super().save(checkpoint)


@pytest.mark.parametrize(
    "formation", [TeamFormation.SEQUENTIAL, TeamFormation.PARALLEL, TeamFormation.PIPELINE]
)
async def test_checkpoint_read_failure_never_restarts_members(formation, monkeypatch):
    from unittest.mock import Mock

    cp = _FailingCheckpointer("load")
    # Persist prior completion first: treating the failed read as an empty history
    # would repeat this member's work.
    await MemoryCheckpointer.save(
        cp,
        WorkflowCheckpoint(
            checkpoint_id="completed",
            thread_id="t1",
            node_id="UnifiedTeam:member:m0",
            state={
                "completed_member_ids": ["m0"],
                "member_results": [
                    MemberResult(member_id="m0", success=True, output="already done").to_dict()
                ],
            },
            timestamp=1.0,
            metadata={"team_node_id": "UnifiedTeam"},
        ),
    )
    members = [_FakeMember("m0"), _FakeMember("m1")]
    coord = _coordinator(members, cp)
    coord.set_formation(formation)
    materialize = Mock(wraps=coord._materialize_worktree_plan_with_diagnostics)
    monkeypatch.setattr(coord, "_materialize_worktree_plan_with_diagnostics", materialize)
    result = await coord.execute_task("do it", {"thread_id": "t1"})
    assert [m.calls for m in members] == [0, 0]
    assert result["success"] is False and result["error_code"] == "member_checkpoint_failed"
    assert result["checkpoint_operation"] == "load" and result["reconciliation_required"] is True
    assert "private backend detail" not in str(result)
    materialize.assert_not_called()


@pytest.mark.parametrize(
    "formation", [TeamFormation.SEQUENTIAL, TeamFormation.PARALLEL, TeamFormation.PIPELINE]
)
@pytest.mark.parametrize("pause", [False, True], ids=["member", "pause"])
@pytest.mark.parametrize("committed", [False, True], ids=["before-save", "lost-ack"])
async def test_checkpoint_save_failure_stops_without_success_or_durable_pause(
    formation, pause, committed
):
    operation = (
        ("save_batch_pause" if formation == TeamFormation.PARALLEL else "save_pause")
        if pause
        else "save_member"
    )
    cp = _FailingCheckpointer(operation, committed=committed)
    members = [_FakeMember("m0", pause=pause), _FakeMember("m1")]
    coord = _coordinator(members, cp)
    coord.set_formation(formation)
    from victor.framework.member_event_sink import MemberEventSink, current_member_sink

    sink = MemberEventSink()
    token = current_member_sink.set(sink)
    try:
        result = await coord.execute_task(
            "do it", {"thread_id": "t1", "member_concurrency_limit": 1}
        )
    finally:
        current_member_sink.reset(token)
        await sink.close()
    events = [event async for event in sink.drain()]
    assert all(event.kind != "member_awaiting_approval" for event in events)

    assert result["success"] is False
    assert result.get("status") != "awaiting_approval"
    assert result["error_code"] == "member_checkpoint_failed"
    assert result["checkpoint_operation"] == operation
    assert result["reconciliation_required"] is True
    assert "private backend detail" not in str(result)
    assert members[0].calls == 1
    assert members[1].calls == int(pause and formation == TeamFormation.PARALLEL)
    assert cp.attempts.count(operation) == 1
    snapshots = await cp.list("t1")
    relevant = [c for c in snapshots if bool(c.metadata.get("awaiting_approval")) == pause]
    assert len(relevant) == int(committed)


@pytest.mark.parametrize(
    "formation", [TeamFormation.SEQUENTIAL, TeamFormation.PARALLEL, TeamFormation.PIPELINE]
)
async def test_durable_approval_event_follows_acknowledged_pause(formation):
    from victor.framework.member_event_sink import MemberEventSink, current_member_sink

    cp = MemoryCheckpointer()
    emitted = []

    class InspectingSink(MemberEventSink):
        async def emit(self, event):
            if event.kind == "member_awaiting_approval":
                snapshots = await cp.list("t1")
                assert snapshots[-1].metadata["awaiting_approval"] is True
                emitted.append(event.member_id)
            await super().emit(event)

    members = [_FakeMember("m0", pause=True), _FakeMember("m1", pause=True)]
    coord = _coordinator(members, cp)
    coord.set_formation(formation)
    token = current_member_sink.set(InspectingSink())
    try:
        result = await coord.execute_task("go", {"thread_id": "t1"})
    finally:
        current_member_sink.reset(token)
    assert result["status"] == "awaiting_approval" and result["success"] is False
    assert emitted == (["m0", "m1"] if formation == TeamFormation.PARALLEL else ["m0"])
