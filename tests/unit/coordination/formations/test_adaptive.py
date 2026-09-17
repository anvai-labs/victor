"""Public dispatch regressions for WS-A: real members, outcomes and run isolation."""

import asyncio

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
