"""Public dispatch regressions for WS-A: real members, outcomes and run isolation."""

import pytest

from victor.coordination.formations.multi_level_hierarchy import MultiLevelHierarchyFormation
from victor.teams.types import TeamFormation


from types import SimpleNamespace
from unittest.mock import AsyncMock
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
