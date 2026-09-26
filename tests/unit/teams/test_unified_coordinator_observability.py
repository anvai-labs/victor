"""Warnings and approval/usage metadata survive the public team adapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.core.shared_types import SubAgentRole
from victor.framework.member_event_sink import MemberEventSink, current_member_sink
from victor.framework.graph_checkpoint import MemoryCheckpointer
from victor.teams.types import TeamConfig, TeamFormation, TeamMember
from victor.teams.unified_coordinator import StateGraphNodeConfig, UnifiedTeamCoordinator


@pytest.mark.parametrize("invalid", [False, True])
async def test_strategy_failure_emits_warning_and_dispatches_default(invalid, caplog):
    def strategy(state):
        if invalid:
            return "not-a-formation"
        raise RuntimeError("selection unavailable")

    coord = UnifiedTeamCoordinator(lightweight_mode=True)
    coord.add_member(
        SimpleNamespace(
            id="a",
            execute_task=AsyncMock(return_value={"success": True, "output": "ok"}),
            receive_message=AsyncMock(),
        )
    )
    coord.with_state_graph_config(StateGraphNodeConfig(formation_strategy=strategy))
    sink = MemberEventSink()
    token = current_member_sink.set(sink)
    try:
        result = await coord({"task": "work"})
    finally:
        current_member_sink.reset(token)
    await sink.close()
    events = [event async for event in sink.drain()]
    warning = next(e for e in events if e.kind == "team_formation_warning")
    assert warning.metadata == {
        "reason": "ValueError" if invalid else "RuntimeError",
        "level": "warning",
    }
    assert result["team_output"]["formation"] == "sequential"
    assert result["team_output"]["success"]
    assert "Formation strategy failed" in caplog.text


async def test_public_pipeline_adapter_preserves_pause_and_usage(monkeypatch):
    calls = []

    async def spawn(**kwargs):
        calls.append(kwargs)
        paused = kwargs["member_id"] == "b" and len(calls) == 2
        return SimpleNamespace(
            success=not paused,
            summary="ok",
            details=(
                {"awaiting_approval": True, "approval_request": {"id": "gate"}}
                if paused
                else {"usage": {"total_tokens": 13}, "session_id": kwargs["member_id"]}
            ),
        )

    sub = MagicMock(spawn=AsyncMock(side_effect=spawn))
    monkeypatch.setattr(
        "victor.agent.subagents.orchestrator.SubAgentOrchestrator", lambda parent: sub
    )
    coord = UnifiedTeamCoordinator(
        MagicMock(), lightweight_mode=True, checkpointer=MemoryCheckpointer()
    )
    config = TeamConfig(
        name="pipeline",
        goal="work",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMember(id=x, name=x, role=SubAgentRole.EXECUTOR, goal="work") for x in ("a", "b")
        ],
        shared_context={"thread_id": "t", "capture_member_usage": True},
    )
    first = await coord.execute_team_config(config)
    assert not first.success
    assert first.status == "awaiting_approval"
    assert first.to_dict()["approval_request"] == {"id": "gate"}
    assert first.paused_member_id == "b"
    result = await coord.execute_team_config(config)
    assert result.success
    assert [c["member_id"] for c in calls] == ["a", "b", "b"]
    assert all(c["capture_usage"] for c in calls)
    assert sum(r.metadata["usage"]["total_tokens"] for r in result.member_results.values()) == 26
