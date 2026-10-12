from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from victor.agent.services.planning_chat_runtime import PlanningChatRuntime
from victor.providers.base import CompletionResponse


@pytest.mark.asyncio
async def test_planning_chat_runtime_caches_planning_service_on_runtime_host():
    runtime_host = MagicMock()
    runtime_host._service_planning_coordinator = None
    runtime_host.task_analyzer.analyze.return_value = "task-analysis"
    runtime_host._get_conversation_message_count.side_effect = [0, 1]
    response = CompletionResponse(content="planned", role="assistant")

    with patch("victor.agent.services.planning_runtime.PlanningRuntimeService") as planning_cls:
        planning_instance = MagicMock()
        planning_instance.chat_with_planning = AsyncMock(return_value=response)
        planning_cls.return_value = planning_instance

        runtime = PlanningChatRuntime(runtime_host)
        result = await runtime.run("plan this")

    assert result is response
    assert runtime_host._service_planning_coordinator is planning_instance
    planning_cls.assert_called_once_with(
        runtime_host, direct_chat=runtime_host._chat_service.execute_direct_in_turn
    )
    planning_instance.chat_with_planning.assert_awaited_once_with(
        "plan this",
        task_analysis="task-analysis",
    )


@pytest.mark.asyncio
async def test_planning_chat_runtime_records_turn_when_planner_did_not():
    runtime_host = MagicMock()
    runtime_host._service_planning_coordinator = MagicMock()
    runtime_host._service_planning_coordinator.chat_with_planning = AsyncMock(
        return_value=CompletionResponse(content="planned", role="assistant")
    )
    runtime_host.task_analyzer.analyze.return_value = "task-analysis"
    runtime_host._get_conversation_message_count.side_effect = [0, 0]
    runtime_host._system_added = False

    runtime = PlanningChatRuntime(runtime_host)
    response = await runtime.run("plan this")

    assert response.content == "planned"
    runtime_host.conversation.ensure_system_prompt.assert_called_once_with()
    runtime_host.add_message.assert_any_call("user", "plan this", metadata=ANY)
    runtime_host.add_message.assert_any_call("assistant", "planned", metadata=ANY)
    assert runtime_host._system_added is True


@pytest.mark.asyncio
async def test_planning_chat_runtime_skips_duplicate_recording_when_planner_already_recorded():
    runtime_host = MagicMock()
    runtime_host._service_planning_coordinator = MagicMock()
    response = CompletionResponse(content="already recorded", role="assistant")
    runtime_host._service_planning_coordinator.chat_with_planning = AsyncMock(return_value=response)
    runtime_host.task_analyzer.analyze.return_value = "task-analysis"
    runtime_host._get_conversation_message_count.side_effect = [0, 2]
    runtime_host._system_added = False

    runtime = PlanningChatRuntime(runtime_host)
    result = await runtime.run("plan this")

    assert result is response
    runtime_host.conversation.ensure_system_prompt.assert_not_called()
    runtime_host.add_message.assert_not_called()
