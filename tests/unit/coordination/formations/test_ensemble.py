"""Ensemble contracts through the canonical coordinator dispatch."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.coordination.formations.base import TeamContext
from victor.coordination.formations.ensemble import execute_ensemble
from victor.coordination.formations.parallel import ParallelFormation
from victor.teams.types import AgentMessage, MessageType, TeamFormation
from victor.teams.unified_coordinator import UnifiedTeamCoordinator


async def run(outputs, mode="vote", formation=TeamFormation.PARALLEL):
    coordinator = UnifiedTeamCoordinator(lightweight_mode=True)
    coordinator.set_formation(formation)
    calls = []
    for index, output in enumerate(outputs):

        async def execute(task, context, index=index, output=output):
            calls.append((index, json.loads(task)))
            return output

        coordinator.add_member(
            SimpleNamespace(
                id=str(index), role="executor", execute_task=execute, receive_message=AsyncMock()
            )
        )
    context = {"ensemble_mode": mode}
    if mode != "vote":
        context["ensemble_aggregator_id"] = str(len(outputs) - 1)
    return await coordinator.execute_task("one task", context), calls


def proposal(key, answer=None):
    return json.dumps({"vote_key": key, "answer": answer or key})


@pytest.mark.parametrize("formation", [TeamFormation.PARALLEL, TeamFormation.CONSENSUS])
async def test_strict_majority_single_wave_and_identical_task(formation):
    result, calls = await run(
        [proposal("yes"), proposal("no"), proposal("yes")], formation=formation
    )
    assert result["success"]
    assert result["final_output"] == "yes"
    assert len(calls) == 3
    assert all(call[1]["task"] == "one task" for call in calls)
    assert len({json.dumps(call[1], sort_keys=True) for call in calls}) == 1
    assert len(result["member_results"]) == 3


@pytest.mark.parametrize(
    "outputs",
    [
        [proposal("a"), proposal("b")],
        [proposal("a"), "VERDICT: a"],
        [proposal("a"), '{"vote_key":"a","answer":"a","extra":1}'],
        [proposal("a"), '{"vote_key":7,"answer":"a"}'],
        [proposal("a"), proposal("a", "x" * 8193)],
    ],
)
async def test_ties_and_invalid_contracts_fail_explicitly(outputs, caplog):
    result, _ = await run(outputs)
    assert not result["success"]
    assert "Ensemble vote failed" in caplog.text


@pytest.mark.parametrize(
    "mode, response, expected",
    [
        ("judge", '{"selected_member_id":"1"}', "second"),
        ("synthesizer", '{"answer":"combined"}', "combined"),
    ],
)
async def test_one_shot_aggregation_preserves_candidates(mode, response, expected):
    result, calls = await run([proposal("a", "first"), proposal("b", "second"), response], mode)
    assert result["success"]
    assert result["final_output"] == expected
    assert len(result["member_results"]) == 3
    assert [item["member_id"] for item in calls[-1][1]["candidates"]] == ["0", "1"]


@pytest.mark.parametrize(
    "response", ['{"selected_member_id":"unknown"}', "prose", '{"selected_member_id":""}']
)
async def test_bad_judge_contract_does_not_fall_back_to_first_candidate(response):
    result, _ = await run([proposal("a"), proposal("b"), response], "judge")
    assert not result["success"]
    assert not result["member_results"]["2"].success


@pytest.mark.parametrize(
    "shared, checkpoint",
    [
        ({"ensemble_mode": "unknown"}, False),
        ({"ensemble_mode": "vote", "ensemble_aggregator_id": "a"}, False),
        ({"ensemble_mode": "judge"}, False),
        ({"ensemble_mode": "vote"}, True),
        ({"ensemble_mode": "vote"}, False),
    ],
)
async def test_invalid_configuration_rejected_before_members(shared, checkpoint):
    context = TeamContext(team_id="x", formation="parallel", shared_state=shared)
    if checkpoint:
        context.checkpoint_hook = AsyncMock()
    with pytest.raises(ValueError):
        await execute_ensemble(
            ParallelFormation(),
            [],
            context,
            AgentMessage(sender_id="c", content="task", message_type=MessageType.TASK),
        )
