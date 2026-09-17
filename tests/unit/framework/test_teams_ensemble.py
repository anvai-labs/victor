from unittest.mock import MagicMock

import pytest

from victor.framework.teams import AgentTeam, TeamMemberSpec, TeamFormation


def candidates():
    return [TeamMemberSpec(role="executor", name=name, goal="old") for name in ("a", "b")]


@pytest.mark.parametrize("mode", ["vote", "judge", "synthesizer"])
async def test_preset_uses_canonical_roles_without_mutating_specs(mode):
    specs = candidates()
    aggregator = (
        None if mode == "vote" else TeamMemberSpec(role="reviewer", name="aggregate", goal="judge")
    )
    team = await AgentTeam.create_ensemble_team(
        MagicMock(), "team", "same task", specs, mode=mode, aggregator=aggregator
    )
    assert team._config.formation == TeamFormation.PARALLEL
    assert [m.goal for m in team._config.members[:2]] == ["same task"] * 2
    assert [m.goal for m in specs] == ["old"] * 2
    if aggregator:
        assert team._config.members[-1].formation_role == mode
        assert team._config.shared_context["ensemble_aggregator_id"] == team._config.members[-1].id


async def test_consensus_vote_preset_and_default_agreement():
    vote = await AgentTeam.create_consensus_team(MagicMock(), "t", "g", candidates(), mode="vote")
    assert vote._config.formation == TeamFormation.CONSENSUS
    assert vote._config.shared_context["ensemble_mode"] == "vote"
    normal = await AgentTeam.create_consensus_team(MagicMock(), "t", "g", candidates())
    assert "ensemble_mode" not in normal._config.shared_context


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "wrong"},
        {"mode": "judge"},
        {"mode": "vote", "aggregator": TeamMemberSpec(role="reviewer", name="r", goal="x")},
    ],
)
async def test_invalid_preset(kwargs):
    with pytest.raises(ValueError):
        await AgentTeam.create_ensemble_team(MagicMock(), "t", "g", candidates(), **kwargs)


async def test_invalid_consensus_mode_or_supervisor():
    for kwargs in ({"mode": "bad"}, {"mode": "vote", "supervisor": candidates()[0]}):
        with pytest.raises(ValueError):
            await AgentTeam.create_consensus_team(MagicMock(), "t", "g", candidates(), **kwargs)
