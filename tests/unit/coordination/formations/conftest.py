import json
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.framework.member_event_sink import MemberEventSink, current_member_sink
from victor.teams.unified_coordinator import UnifiedTeamCoordinator


@pytest.fixture
def conversation_run():
    async def run(formation, outputs, **context):
        coordinator = UnifiedTeamCoordinator(lightweight_mode=True)
        coordinator.set_formation(formation)
        calls = []
        counts = defaultdict(int)
        for identifier, values in outputs.items():

            async def execute(task, _context, identifier=identifier, values=values):
                index = counts[identifier]
                counts[identifier] += 1
                calls.append((identifier, json.loads(task)))
                value = values[min(index, len(values) - 1)]
                return {
                    "success": True,
                    "output": json.dumps(value) if isinstance(value, dict) else value,
                    "tool_calls_used": 2,
                    "duration_seconds": 0.5,
                }

            coordinator.add_member(
                SimpleNamespace(
                    id=identifier,
                    role="executor",
                    execute_task=execute,
                    receive_message=AsyncMock(),
                )
            )
        sink = MemberEventSink()
        token = current_member_sink.set(sink)
        try:
            result = await coordinator.execute_task("task", context)
        finally:
            current_member_sink.reset(token)
        await sink.close()
        return result, calls, [event async for event in sink.drain()]

    return run


@pytest.fixture
def say():
    return lambda content="message", done=False, target=None: {
        "content": content,
        "done": done,
        "handoff_to": target,
    }
