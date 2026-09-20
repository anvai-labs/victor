"""Public reflection dispatch must retain opted-in member attribution across rounds."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.framework.teams import AgentTeam, TeamMemberSpec


@pytest.mark.parametrize(
    "capture,rounds,fault",
    [
        (False, 1, None),
        (True, 1, None),
        (True, 2, None),
        (True, 1, "failed"),
        (True, 1, "invalid"),
        (True, 1, "missing_usage"),
        (True, 2, "session_changed"),
        (True, 1, "inconsistent_total"),
        (True, 2, "recovered_failure"),
    ],
)
async def test_reflection_keeps_member_results_and_round_usage(
    tmp_path, monkeypatch, capture, rounds, fault
):
    counts = {}

    async def spawn(**kwargs):
        identity = kwargs["member_id"]
        counts[identity] = counts.get(identity, 0) + 1
        critic = kwargs["display_name"] == "critic"
        output = (
            json.dumps(
                {
                    "verdict": "satisfied" if counts[identity] == rounds else "needs_work",
                    "feedback": "check",
                }
            )
            if critic
            else "solution"
        )
        if critic and fault == "invalid":
            output = "invalid JSON"
        if critic and fault == "recovered_failure":
            output = "VERDICT: SATISFIED"
        details = {
            "session_id": "session-" + identity,
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }
        if critic and fault == "missing_usage":
            details.pop("usage")
        if critic and fault == "session_changed" and counts[identity] > 1:
            details["session_id"] = "changed"
        if critic and fault == "inconsistent_total":
            details["usage"]["total_tokens"] = 999
        failed = critic and (
            fault == "failed" or (fault == "recovered_failure" and counts[identity] == 1)
        )
        return SimpleNamespace(
            success=not failed,
            summary=output,
            error="member failed" if failed else None,
            details=details,
            tool_calls_used=1,
            duration_seconds=1.0,
        )

    monkeypatch.setattr(
        "victor.agent.subagents.orchestrator.SubAgentOrchestrator",
        lambda *args: SimpleNamespace(spawn=AsyncMock(side_effect=spawn)),
    )
    team = await AgentTeam.create_reflection_team(
        MagicMock(),
        "reflection",
        "produce solution",
        generator=TeamMemberSpec(role="executor", name="generator", goal="write"),
        critic=TeamMemberSpec(role="reviewer", name="critic", goal="review"),
        verdict_format="legacy" if fault == "recovered_failure" else "json",
        rounds=rounds,
        shared_context={"capture_member_usage": capture},
    )
    result = await team.run()
    assert result.success == (fault is None)
    if not capture:
        assert set(result.member_results) == {"reflection_formation"}
        assert result.final_output == "solution"
        return
    assert set(result.member_results) == {member.id for member in team._config.members}
    assert len({member.metadata["session_id"] for member in result.member_results.values()}) == 2
    for member in result.member_results.values():
        if (
            fault in {"missing_usage", "session_changed", "inconsistent_total"}
            and member.member_id == team._config.members[1].id
        ):
            assert member.metadata["reflection_capture_error"] is True
            assert "usage" not in member.metadata
        else:
            assert member.metadata["usage"] == {
                "input_tokens": 3 * rounds,
                "output_tokens": 2 * rounds,
                "total_tokens": 5 * rounds,
            }
        assert member.tool_calls_used == rounds
        assert member.duration_seconds == rounds
        assert len(member.metadata["reflection_attempts"]) == rounds
    if fault is None:
        assert result.final_output == "solution"
        assert result.total_tool_calls == 2 * rounds
    else:
        critic = team._config.members[1]
        assert result.member_results[critic.id].success is False


@pytest.mark.parametrize("resume_from", ["partial", "terminal"])
async def test_captured_reflection_resume_retains_usage_without_replaying_completed_iterations(
    monkeypatch, resume_from
):
    from victor.framework.graph_checkpoint import MemoryCheckpointer

    calls = []
    counts = {}

    async def spawn(**kwargs):
        identity = kwargs["member_id"]
        calls.append(identity)
        counts[identity] = counts.get(identity, 0) + 1
        output = "solution"
        if kwargs["display_name"] == "critic":
            output = json.dumps(
                {
                    "verdict": "satisfied" if counts[identity] >= 2 else "needs_work",
                    "feedback": "check",
                }
            )
        return SimpleNamespace(
            success=True,
            summary=output,
            details={
                "session_id": "session-" + identity,
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            },
            tool_calls_used=1,
            duration_seconds=1.0,
        )

    monkeypatch.setattr(
        "victor.agent.subagents.orchestrator.SubAgentOrchestrator",
        lambda *args: SimpleNamespace(spawn=AsyncMock(side_effect=spawn)),
    )
    team = await AgentTeam.create_reflection_team(
        MagicMock(),
        "reflection",
        "produce solution",
        generator=TeamMemberSpec(role="executor", name="generator", goal="write"),
        critic=TeamMemberSpec(role="reviewer", name="critic", goal="review"),
        verdict_format="json",
        rounds=3,
        shared_context={"capture_member_usage": True, "thread_id": "reflection-thread"},
    )
    checkpoint = MemoryCheckpointer()
    team._coordinator._checkpointer = checkpoint
    first = await team.run()
    assert first.success and len(calls) == 4
    if resume_from == "partial":
        snapshots = await checkpoint.list("reflection-thread")
        partial = next(
            item
            for item in snapshots
            if item.state.get("shared_state", {}).get("__reflection__", {}).get("iter_done") == 1
        )
        checkpoint = MemoryCheckpointer()
        await checkpoint.save(partial)
        team._coordinator._checkpointer = checkpoint
    calls.clear()
    resumed = await team.run()
    assert resumed.success
    assert len(calls) == (2 if resume_from == "partial" else 0)
    assert resumed.final_output == first.final_output
    for identity, member in resumed.member_results.items():
        assert member.metadata["usage"] == first.member_results[identity].metadata["usage"]
        assert len(member.metadata["reflection_attempts"]) == 2


def test_captured_history_is_independent_of_resume_and_checkpoint_data():
    from victor.coordination.formations.base import TeamContext
    from victor.coordination.formations.reflection_results import ReflectionResults
    from victor.teams.types import MemberResult

    member = MemberResult(
        "generator",
        True,
        "solution",
        metadata={
            "session_id": "session",
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        },
    )
    saved = {"capture_member_usage": True, "member_results": [member.to_dict()]}
    tracker = ReflectionResults(
        [], TeamContext("test", "reflection", {"capture_member_usage": True}), saved
    )
    checkpoint = tracker.checkpoint()
    result = tracker.finish(
        MemberResult("reflection_formation", True, "solution", metadata={"satisfied": True})
    )[0]
    result.metadata["reflection_attempts"][0]["metadata"]["usage"]["input_tokens"] = 777
    assert saved["member_results"][0]["metadata"]["usage"]["input_tokens"] == 3
    assert checkpoint["member_results"][0]["metadata"]["usage"]["input_tokens"] == 3
    assert tracker.attempts[0].metadata["usage"]["input_tokens"] == 3


def test_enabled_capture_never_returns_successful_synthetic_result_without_members():
    from victor.coordination.formations.base import TeamContext
    from victor.coordination.formations.reflection_results import ReflectionResults
    from victor.teams.types import MemberResult

    tracker = ReflectionResults(
        [], TeamContext("test", "reflection", {"capture_member_usage": True}), {}
    )
    with pytest.raises(ValueError, match="captured member results"):
        tracker.finish(
            MemberResult("reflection_formation", True, "solution", metadata={"satisfied": True})
        )


async def test_default_parallel_ignores_reflection_metadata(monkeypatch):
    from victor.teams.types import MemberResult, TeamFormation

    team = await AgentTeam.create(
        MagicMock(),
        "parallel",
        "task",
        [TeamMemberSpec(role="executor", name="first", goal="work")],
        formation=TeamFormation.PARALLEL,
    )
    member = team._config.members[0]
    monkeypatch.setattr(
        team._coordinator._formations[TeamFormation.PARALLEL],
        "execute",
        AsyncMock(
            return_value=[
                MemberResult(
                    member.id,
                    True,
                    "original output",
                    metadata={
                        "reflection_success": False,
                        "reflection_output": "unexpected override",
                    },
                )
            ]
        ),
    )
    result = await team.run()
    assert result.success
    assert result.final_output == "original output"
