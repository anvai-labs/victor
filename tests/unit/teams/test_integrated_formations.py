"""Public dispatch regressions for WS-A: real members, outcomes and run isolation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.framework.teams import AgentTeam, TeamMemberSpec
from victor.teams.types import TeamFormation
from victor.teams.unified_coordinator import UnifiedTeamCoordinator


def member(member_id, role="executor", *, success=True):
    return SimpleNamespace(
        id=member_id,
        role=role,
        execute_task=AsyncMock(
            return_value={
                "success": success,
                "output": f"output:{member_id}",
                "tool_calls_used": 2,
                "metadata": {"changed_files": [f"{member_id}.txt"]},
            }
        ),
        receive_message=AsyncMock(return_value=None),
    )


def coordinator(formation, members):
    coord = UnifiedTeamCoordinator(enable_observability=False, enable_rl=False)
    coord.set_formation(formation)
    for item in members:
        coord.add_member(item)
    return coord


@pytest.mark.parametrize(
    "preset, formation",
    [
        ("create_adaptive_team", TeamFormation.ADAPTIVE),
        ("create_router_team", TeamFormation.DYNAMIC_ROUTER),
        ("create_multi_level_hierarchy_team", TeamFormation.MULTI_LEVEL_HIERARCHY),
    ],
)
async def test_public_presets_reach_real_coordinator(preset, formation):
    specs = [
        TeamMemberSpec(role="executor", name="a", goal="task"),
        TeamMemberSpec(role="reviewer", name="b", goal="check"),
    ]
    team = await getattr(AgentTeam, preset)(SimpleNamespace(), "test", "Review task", specs)
    members = [member(config.id, config.role.value) for config in team._config.members]
    result = await team._coordinator.execute_team_config(team._config, members=members)
    assert team._config.formation == formation
    assert result.success is True
    assert sum(item.execute_task.await_count for item in members) >= 1


async def test_router_preset_resolves_names_to_existing_ids_once():
    team = await AgentTeam.create_router_team(
        SimpleNamespace(),
        "test",
        "custom task",
        [
            TeamMemberSpec(role="executor", name="first", goal="task"),
            TeamMemberSpec(role="executor", name="second", goal="task"),
        ],
        routes={"custom": "second"},
    )
    assert team._config.shared_context["router_routes"] == {"custom": team._config.members[1].id}


@pytest.mark.parametrize(
    "formation",
    [
        TeamFormation.ADAPTIVE,
        TeamFormation.DYNAMIC_ROUTER,
        TeamFormation.MULTI_LEVEL_HIERARCHY,
    ],
)
async def test_new_formations_keep_approval_inline_with_checkpointer(formation):
    from victor.agent.member_approval_context import current_member_durable_pause_enabled
    from victor.framework.graph_checkpoint import MemoryCheckpointer

    seen = []
    item = member("a")

    async def execute(task, context):
        seen.append(current_member_durable_pause_enabled.get())
        return {"success": True, "output": "ok"}

    item.execute_task = AsyncMock(side_effect=execute)
    coord = UnifiedTeamCoordinator(lightweight_mode=True, checkpointer=MemoryCheckpointer())
    coord.set_formation(formation)
    coord.add_member(item)
    result = await coord.execute_task("task", {"thread_id": "stable-thread"})
    assert result["success"] is True
    assert seen == [None]
