"""Table-driven outcome contracts for consensus, reflection and parallel teams."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.coordination.formations.consensus import ConsensusFormation
from victor.teams.types import MemberResult, TeamFormation
from victor.teams.unified_coordinator import UnifiedTeamCoordinator


def member(name, responses):
    return SimpleNamespace(
        id=name,
        role="executor",
        receive_message=AsyncMock(),
        execute_task=AsyncMock(side_effect=responses),
    )


def coordinator(formation, members):
    coord = UnifiedTeamCoordinator(lightweight_mode=True)
    coord.set_formation(formation)
    for item in members:
        coord.add_member(item)
    return coord


@pytest.mark.parametrize(
    "outputs, threshold, expected",
    [
        (["a", "b", "c"], 0.7, False),
        (["a", "b", "b"], 0.6, True),
        (["a", "a", "b"], 0.7, False),
        (["a", "a", "a"], 1.0, True),
        (["A", "a"], 1.0, False),
        ([], 0.7, False),
    ],
)
def test_consensus_compares_agreement_not_execution_success(outputs, threshold, expected):
    results = [
        MemberResult(member_id=str(i), success=True, output=value)
        for i, value in enumerate(outputs)
    ]
    assert ConsensusFormation(agreement_threshold=threshold)._check_consensus(results) is expected


def test_consensus_explicit_key_and_failed_member_denominator():
    results = [
        MemberResult(
            member_id="a", success=True, output="one", metadata={"consensus_key": "decision"}
        ),
        MemberResult(
            member_id="b", success=True, output="two", metadata={"consensus_key": "decision"}
        ),
    ]
    assert ConsensusFormation()._check_consensus(results)
    results.append(MemberResult(member_id="c", success=False, output=""))
    assert not ConsensusFormation()._check_consensus(results)


async def test_consensus_default_rounds_can_converge_after_disagreement():
    a = member("a", ["first", "agreed"])
    b = member("b", ["different", "agreed"])
    result = await coordinator(TeamFormation.CONSENSUS, [a, b]).execute_task("task", {})
    assert result["success"] is True
    assert result["consensus_rounds"] == 2
    assert result["consensus_achieved"] is True
    assert ConsensusFormation().max_rounds == 3


@pytest.mark.parametrize(
    "tie_breaker, expected_success, expected_output",
    [
        (None, False, "a\n\nb"),
        ("supervisor", True, "a"),
    ],
)
async def test_consensus_exhaustion_and_supervisor_tie_break(
    tie_breaker, expected_success, expected_output
):
    members = [member("supervisor", ["a"]), member("b", ["b"])]
    shared = {"consensus_max_rounds": 1}
    if tie_breaker:
        shared["consensus_tie_breaker_id"] = tie_breaker
    result = await coordinator(TeamFormation.CONSENSUS, members).execute_task(
        "task", {"shared_state": shared}
    )
    assert result["success"] is expected_success
    assert result["consensus_achieved"] is False
    assert result["final_output"] == expected_output


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize(
    "outcome", ["converged", "exhausted", "missing_usage", "changed_session", "earlier_failure"]
)
async def test_consensus_retains_opted_in_whole_session_usage(monkeypatch, capture, outcome):
    from unittest.mock import MagicMock
    from victor.framework.teams import AgentTeam, TeamMemberSpec

    calls = {}

    async def spawn(**kwargs):
        identifier = kwargs["member_id"]
        count = calls.get(identifier, 0) + 1
        calls[identifier] = count
        output = "agreed" if count == 2 and outcome != "exhausted" else kwargs["display_name"]
        details = {
            "session_id": "session-" + identifier,
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5, "cached_tokens": 1},
        }
        if kwargs["display_name"] == "first" and count == 1:
            if outcome == "missing_usage":
                details.pop("usage")
            if outcome == "changed_session":
                details["session_id"] = "different-session"
        failed = outcome == "earlier_failure" and count == 1
        return SimpleNamespace(
            success=not failed,
            summary=output,
            details=details,
            error="failed attempt" if failed else None,
            tool_calls_used=1,
            duration_seconds=1.0,
        )

    monkeypatch.setattr(
        "victor.agent.subagents.orchestrator.SubAgentOrchestrator",
        lambda *args: SimpleNamespace(spawn=AsyncMock(side_effect=spawn)),
    )
    team = await AgentTeam.create(
        MagicMock(),
        "consensus",
        "agree",
        [TeamMemberSpec(role="executor", name=name, goal="agree") for name in ("first", "second")],
        formation=TeamFormation.CONSENSUS,
        shared_context={"capture_member_usage": capture, "consensus_max_rounds": 2},
    )
    result = await team.run()
    assert result.consensus_rounds == 2
    assert result.success == (outcome != "exhausted" and (not capture or outcome == "converged"))
    for member in result.member_results.values():
        bad = (
            capture
            and member.member_id == team._config.members[0].id
            and outcome in {"missing_usage", "changed_session"}
        )
        if bad:
            assert member.metadata["consensus_capture_error"] is True
            assert "usage" not in member.metadata
        else:
            multiplier = 2 if capture else 1
            assert member.metadata["usage"] == {
                "input_tokens": 3 * multiplier,
                "output_tokens": 2 * multiplier,
                "total_tokens": 5 * multiplier,
                "cached_tokens": multiplier,
            }
        assert member.tool_calls_used == (2 if capture else 1)
        assert member.duration_seconds == (2 if capture else 1)
        if capture:
            assert len(member.metadata["consensus_attempts"]) == 2
        else:
            assert "consensus_attempts" not in member.metadata
