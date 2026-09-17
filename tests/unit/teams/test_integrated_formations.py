"""Public dispatch regressions for WS-A: real members, outcomes and run isolation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.coordination.formations import create_formation_registry
from victor.coordination.formations.multi_level_hierarchy import MultiLevelHierarchyFormation
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


def test_registry_covers_every_enum_and_declares_durability():
    registry = create_formation_registry()
    assert set(registry) == set(TeamFormation)
    for formation in (
        TeamFormation.ADAPTIVE,
        TeamFormation.DYNAMIC_ROUTER,
        TeamFormation.MULTI_LEVEL_HIERARCHY,
    ):
        assert registry[formation].supports_durable_pause() is False


@pytest.mark.parametrize("success", [True, False])
async def test_router_dispatch_preserves_selected_member_outcome(success):
    members = [member("builder"), member("review", "reviewer", success=success)]
    coord = coordinator(TeamFormation.DYNAMIC_ROUTER, members)
    result = await coord.execute_task("Review this change", {})
    assert result["success"] is success
    assert set(result["member_results"]) == {"review"}
    assert result["total_tool_calls"] == 2
    members[0].execute_task.assert_not_awaited()
    members[1].execute_task.assert_awaited_once()


async def test_router_explicit_route_and_warning_on_unmatched_task(caplog):
    members = [member("first"), member("chosen")]
    coord = coordinator(TeamFormation.DYNAMIC_ROUTER, members)
    result = await coord.execute_task(
        "Custom task", {"shared_state": {"router_routes": {"custom": "chosen"}}}
    )
    assert set(result["member_results"]) == {"chosen"}
    with caplog.at_level("WARNING"):
        result = await coord.execute_task("Unmatched task", {})
    assert set(result["member_results"]) == {"first"}
    assert "no matching route" in caplog.text


async def test_hierarchy_dispatch_runs_all_levels_and_preserves_failures():
    members = [
        member("supervisor"),
        member("subtree"),
        member("sibling"),
        member("leaf", success=False),
    ]
    coord = coordinator(TeamFormation.MULTI_LEVEL_HIERARCHY, members)
    result = await coord.execute_task("First task\nSecond task\nThird task\nFourth task", {})
    assert result["success"] is False
    assert set(result["member_results"]) == {item.id for item in members}
    assert result["total_tool_calls"] == 8
    assert result["member_results"]["leaf"].success is False
    assert result["member_results"]["supervisor"].metadata["hierarchy_root"] is True
    assert '"success": false' in members[1].execute_task.call_args.args[0]
    for item in members:
        item.execute_task.assert_awaited_once()


@pytest.mark.parametrize(
    "tree",
    [
        {"member_id": "missing"},
        {"member_id": "a", "children": [{"member_id": "a"}]},
        {"member_id": "a"},
    ],
)
async def test_invalid_hierarchy_executes_no_members(tree):
    members = [member("a"), member("b")]
    coord = coordinator(TeamFormation.MULTI_LEVEL_HIERARCHY, members)
    # Exercise dispatch directly so a configuration error cannot be hidden by
    # the outer coordinator's task-failure envelope.
    with pytest.raises(ValueError):
        await coord._execute_formation("task", {"shared_state": {"hierarchy": tree}})
    for item in members:
        item.execute_task.assert_not_awaited()


@pytest.mark.parametrize("strategy", ["line", "count", "auto"])
@pytest.mark.parametrize("task", ["", "x", "abcde", "one\ntwo\nthree\nfour\nfive"])
def test_hierarchy_partition_is_lossless(strategy, task):
    pieces = MultiLevelHierarchyFormation(split_strategy=strategy)._split_task(task, 3)
    assert len(pieces) == 3
    assert "".join(pieces) == task


async def test_adaptive_dispatch_keeps_member_failures_and_bounded_retries(caplog):
    members = [member("a", success=False), member("b", success=False)]
    coord = coordinator(TeamFormation.ADAPTIVE, members)
    with caplog.at_level("WARNING"):
        result = await coord.execute_task(
            "task",
            {
                "shared_state": {
                    "adaptive_options": {
                        "formation_cycle": ["sequential", "parallel"],
                        "max_switches": 1,
                        "adaptation_strategy": "error_rate",
                    }
                }
            },
        )
    assert result["success"] is False
    assert set(result["member_results"]) == {"a", "b"}
    assert result["total_tool_calls"] == 8
    assert "switching topology" in caplog.text
    for item in members:
        assert item.execute_task.await_count == 2
    for outcome in result["member_results"].values():
        assert outcome.metadata["formation_switches"] == 1
        assert [attempt["formation"] for attempt in outcome.metadata["formation_history"]] == [
            "sequential",
            "parallel",
        ]


async def test_adaptive_concurrent_calls_do_not_share_state():
    members = [member("a"), member("b")]
    coord = coordinator(TeamFormation.ADAPTIVE, members)

    async def run(initial):
        return await coord.execute_task(
            "task",
            {
                "shared_state": {
                    "initial_formation_hint": initial,
                    "adaptive_options": {
                        "formation_cycle": ["sequential", "parallel"],
                        "max_switches": 0,
                    },
                }
            },
        )

    outputs = await asyncio.gather(run("sequential"), run("parallel"))
    for output, expected in zip(outputs, ["sequential", "parallel"]):
        assert output["success"] is True
        metadata = output["member_results"]["a"].metadata
        assert metadata["current_formation"] == expected
        assert len(metadata["formation_history"]) == 1
        assert metadata["formation_switches"] == 0


@pytest.mark.parametrize(
    "options",
    [
        {"formation_cycle": []},
        {"formation_cycle": ["adaptive"]},
        {"formation_cycle": ["not_a_formation"]},
        {"max_switches": -1},
        {"max_duration_seconds": 0},
    ],
)
async def test_adaptive_rejects_invalid_options_before_execution(options):
    item = member("a")
    coord = coordinator(TeamFormation.ADAPTIVE, [item])
    with pytest.raises(ValueError):
        await coord._execute_formation("task", {"shared_state": {"adaptive_options": options}})
    item.execute_task.assert_not_awaited()


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
