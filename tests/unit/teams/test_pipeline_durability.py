"""ADR-023: full durability for the PIPELINE formation (checkpoint / resume / pause / lanes).

PIPELINE is sequential (stages chain output → input, stop on failure), so it reuses the same
durability machinery as SEQUENTIAL — the shared `_execute_member_with_events` helper + the
coordinator's checkpoint/pause/event hooks. These tests mirror the SEQUENTIAL durability tests.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from victor.coordination.formations.base import TeamContext
from victor.coordination.formations.pipeline import PipelineFormation
from victor.framework.graph_checkpoint import MemoryCheckpointer
from victor.teams import TeamFormation, UnifiedTeamCoordinator
from victor.teams.types import AgentMessage, MemberResult, MessageType

_APPROVAL = {"id": "req-1", "title": "run_command", "tool_name": "run_command"}


class _FakeStage:
    def __init__(
        self, member_id: str, *, output: str = "ok", fail: bool = False, pause: bool = False
    ) -> None:
        self.id = member_id
        self._output = output
        self._fail = fail
        self._pause = pause
        self.calls = 0

    async def execute_task(self, *args: Any, **kwargs: Any) -> dict:
        self.calls += 1
        if self._pause:
            return {
                "output": "",
                "success": False,
                "metadata": {"awaiting_approval": True, "approval_request": _APPROVAL},
            }
        if self._fail:
            return {"output": "", "success": False, "error": "boom"}
        return {"output": f"{self._output}-{self.id}", "success": True}

    async def receive_message(self, *args: Any, **kwargs: Any) -> None:
        return None


def _coordinator(stages: List[_FakeStage], checkpointer: Any = None) -> UnifiedTeamCoordinator:
    coord = UnifiedTeamCoordinator(lightweight_mode=True, checkpointer=checkpointer)
    for s in stages:
        coord.add_member(s)
    coord.set_formation(TeamFormation.PIPELINE)
    return coord


# ── checkpoint + resume ────────────────────────────────────────────


async def test_pipeline_checkpoints_each_stage_and_resumes() -> None:
    cp = MemoryCheckpointer()
    stages = [_FakeStage("s0"), _FakeStage("s1"), _FakeStage("s2")]
    result = await _coordinator(stages, cp).execute_task("go", {"thread_id": "t1"})
    assert result["success"] is True
    checkpoints = [c for c in await cp.list("t1") if c.metadata.get("team_node_id")]
    assert len(checkpoints) == 3  # one per stage
    assert max(checkpoints, key=lambda c: c.timestamp).state["completed_member_ids"] == [
        "s0",
        "s1",
        "s2",
    ]

    # Resume with s0 pre-completed → s0 skipped, s1/s2 re-run.
    cp2 = MemoryCheckpointer()
    await cp2.save(_first_stage_checkpoint())
    resumed = [_FakeStage("s0"), _FakeStage("s1"), _FakeStage("s2")]
    await _coordinator(resumed, cp2).execute_task("go", {"thread_id": "t1"})
    assert resumed[0].calls == 0  # completed stage skipped
    assert resumed[1].calls == 1
    assert resumed[2].calls == 1


def _first_stage_checkpoint():
    from victor.framework.graph_checkpoint import WorkflowCheckpoint

    done = MemberResult(member_id="s0", success=True, output="ok-s0")
    return WorkflowCheckpoint(
        checkpoint_id="t1:UnifiedTeam:member:0",
        thread_id="t1",
        node_id="UnifiedTeam:member:s0",
        state={
            "completed_member_ids": ["s0"],
            "member_results": [done.to_dict()],
            "shared_state": {},
            "last_output": "ok-s0",
            "last_agent_id": "s0",
        },
        timestamp=1.0,
        metadata={"team_node_id": "UnifiedTeam", "member_id": "s0", "formation": "pipeline"},
    )


# ── durable pause + resume ─────────────────────────────────────────


async def test_pipeline_pauses_at_awaiting_stage_and_resumes() -> None:
    cp = MemoryCheckpointer()
    stages = [_FakeStage("s0"), _FakeStage("s1", pause=True), _FakeStage("s2")]
    result = await _coordinator(stages, cp).execute_task("go", {"thread_id": "t1"})

    assert stages[2].calls == 0  # pipeline stopped at the paused stage
    assert result["status"] == "awaiting_approval"
    assert result["paused_member_id"] == "s1"
    pauses = [c for c in await cp.list("t1") if c.metadata.get("awaiting_approval")]
    assert len(pauses) == 1 and pauses[0].state["completed_member_ids"] == ["s0"]

    # Resume: s1 now proceeds → s0 skipped, s1 re-runs, s2 runs.
    resumed = [_FakeStage("s0"), _FakeStage("s1"), _FakeStage("s2")]
    result2 = await _coordinator(resumed, cp).execute_task(
        "go", {"thread_id": "t1", "approval_decision": {"member_id": "s1", "approved": True}}
    )
    assert resumed[0].calls == 0 and resumed[1].calls == 1 and resumed[2].calls == 1
    assert result2["success"] is True


# ── stop-on-failure + opt-out ──────────────────────────────────────


async def test_pipeline_stops_on_failure_but_checkpoints_it() -> None:
    cp = MemoryCheckpointer()
    stages = [_FakeStage("s0"), _FakeStage("s1", fail=True), _FakeStage("s2")]
    result = await _coordinator(stages, cp).execute_task("go", {"thread_id": "t1"})
    assert stages[2].calls == 0  # stopped after the failed stage
    assert result["success"] is False
    assert set(result["member_results"].keys()) == {"s0", "s1"}  # s2 never ran
    checkpoints = [c for c in await cp.list("t1") if c.metadata.get("team_node_id")]
    assert len(checkpoints) == 2  # s0 + s1 checkpointed


async def test_pipeline_without_checkpointer_is_unchanged() -> None:
    stages = [_FakeStage("s0"), _FakeStage("s1"), _FakeStage("s2")]
    result = await _coordinator(stages).execute_task("go", {"thread_id": "t1"})
    assert all(s.calls == 1 for s in stages)
    assert result["success"] is True


# ── streaming lanes ────────────────────────────────────────────────


def _task() -> AgentMessage:
    return AgentMessage(sender_id="coordinator", content="go", message_type=MessageType.TASK)


class _FakeAgent:
    def __init__(self, member_id: str) -> None:
        self.id = member_id

    async def execute(self, task: Any, context: Any) -> MemberResult:
        return MemberResult(member_id=self.id, success=True, output=f"ok-{self.id}")


async def test_pipeline_emits_per_stage_lanes() -> None:
    ctx = TeamContext(team_id="t", formation="pipeline")
    events: List[Tuple[str, str]] = []

    async def _hook(kind: str, member_id: str, index: int, **kw: Any) -> None:
        events.append((kind, member_id))

    ctx.member_event_hook = _hook
    await PipelineFormation().execute([_FakeAgent("s0"), _FakeAgent("s1")], ctx, _task())
    assert ("member_start", "s0") in events and ("member_completed", "s0") in events
    assert ("member_start", "s1") in events and ("member_completed", "s1") in events
