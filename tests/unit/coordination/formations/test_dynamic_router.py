"""Public dispatch regressions for WS-A: real members, outcomes and run isolation."""

import pytest

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
