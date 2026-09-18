from unittest.mock import MagicMock

import pytest

from victor.framework.teams import AgentTeam, TeamMemberSpec, TeamFormation


def members():
    return [TeamMemberSpec(role="executor", name=name, goal="speak") for name in ("a", "b")]


@pytest.mark.parametrize("mode", ["group_chat", "debate", "handoff"])
async def test_presets_use_canonical_registry_and_resolve_names_once(mode):
    special = TeamMemberSpec(role="reviewer", name="special", goal="route/judge")
    kwargs = (
        {"router": special}
        if mode == "group_chat"
        else {"judge": special} if mode == "debate" else {"start_member": "b"}
    )
    team = await getattr(AgentTeam, f"create_{mode}_team")(
        MagicMock(), "team", "task", members(), **kwargs
    )
    assert team._config.formation == TeamFormation(mode)
    shared = team._config.shared_context
    if mode == "handoff":
        assert shared["handoff_start_id"] == team._config.members[1].id
    else:
        role = "router" if mode == "group_chat" else "judge"
        assert shared[f"conversation_{role}_id"] == team._config.members[-1].id
        assert team._config.members[-1].formation_role == role
        assert special.formation_role is None


async def test_preset_callbacks_and_invalid_names():
    def selector(*_):
        return "a"

    def candidate(*_):
        return ["a"]

    team = await AgentTeam.create_group_chat_team(
        MagicMock(), "t", "g", members(), selector_func=selector, candidate_func=candidate
    )
    assert team._config.shared_context["selector_func"] is selector
    for specs, kwargs in [
        (members()[:1], {}),
        (members(), {"max_turns": 0}),
        (members(), {"start_member": "unknown"}),
        (members() * 2, {}),
    ]:
        with pytest.raises(ValueError):
            await AgentTeam.create_handoff_team(MagicMock(), "t", "g", specs, **kwargs)
    with pytest.raises(ValueError):
        await AgentTeam.create_group_chat_team(
            MagicMock(), "t", "g", members(), router=members()[0], selector_func=selector
        )
