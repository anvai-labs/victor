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
