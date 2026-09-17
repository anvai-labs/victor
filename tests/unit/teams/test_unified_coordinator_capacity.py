"""Provider-derived admission queues all members and emits throttle diagnostics."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.framework.member_event_sink import (
    MEMBER_THROTTLED,
    MemberEventSink,
    current_member_sink,
)
from victor.teams.types import TeamFormation
from victor.teams.unified_coordinator import UnifiedTeamCoordinator


async def run_team(capacity, *, enabled=True, max_workers=None):
    active = 0
    peak = 0
    completed = []

    async def execute(task, context):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        completed.append(context["member_name"])
        return "ok"

    provider = SimpleNamespace(get_parallel_capacity=AsyncMock(return_value=capacity))
    coord = UnifiedTeamCoordinator(
        SimpleNamespace(provider=provider, model="model"), lightweight_mode=True
    )
    coord.set_formation(TeamFormation.PARALLEL)
    for name in ["a", "b", "c"]:

        async def bound(task, context, name=name):
            return await execute(task, {**context, "member_name": name})

        coord.add_member(
            SimpleNamespace(
                id=name, role="executor", execute_task=bound, receive_message=AsyncMock()
            )
        )
    context = {"capacity_aware_parallelism": enabled}
    if max_workers is not None:
        context["max_workers"] = max_workers
    sink = MemberEventSink()
    token = current_member_sink.set(sink)
    try:
        result = await coord.execute_task("task", context)
    finally:
        current_member_sink.reset(token)
    await sink.close()
    events = [event async for event in sink.drain()]
    return result, completed, peak, events, provider


@pytest.mark.parametrize(
    "capacity, override, expected_peak", [(1, None, 1), (2, None, 2), (3, 2, 2)]
)
async def test_admission_runs_every_member_with_bounded_peak(capacity, override, expected_peak):
    result, completed, peak, events, provider = await run_team(capacity, max_workers=override)
    assert result["success"]
    assert set(completed) == {"a", "b", "c"}
    assert peak == expected_peak
    assert len(result["member_results"]) == 3
    throttles = [event for event in events if event.kind == MEMBER_THROTTLED]
    assert throttles and all(
        event.metadata["concurrency_limit"] == expected_peak for event in throttles
    )
    provider.get_parallel_capacity.assert_awaited_once_with("model")


async def test_admission_absent_preserves_legacy_member_limit():
    result, completed, peak, events, provider = await run_team(1, enabled=False, max_workers=2)
    assert result["success"]
    assert len(completed) == 2
    assert peak == 2
    assert not any(event.kind == MEMBER_THROTTLED for event in events)
    provider.get_parallel_capacity.assert_not_awaited()


@pytest.mark.parametrize("capacity", [0, -1, None, True, "2"])
async def test_bad_capacity_fails_without_executing_members(capacity):
    result, completed, peak, events, provider = await run_team(capacity)
    assert not result["success"]
    assert completed == []
