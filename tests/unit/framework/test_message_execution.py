from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.agent import Agent
from victor.framework.client import VictorClient
from victor.framework.events import AgentExecutionEvent, EventType
from victor.framework.session_config import SessionConfig
from victor.framework.task import TaskResult
from victor.providers.base import CompletionResponse


class AgentOrchestratorMock(MagicMock):
    """MagicMock subclass whose type name satisfies Agent.__init__ validation."""


def _make_orchestrator() -> AgentOrchestratorMock:
    orchestrator = AgentOrchestratorMock()
    orchestrator.model = "test-model"
    orchestrator.provider = MagicMock()
    orchestrator.provider.name = "test-provider"
    orchestrator.get_stage = MagicMock(return_value=SimpleNamespace(value="INITIAL"))
    orchestrator._container = MagicMock()
    return orchestrator


def test_prepare_message_prepends_context_but_preserves_response_message() -> None:
    from victor.framework.message_execution import prepare_message

    prepared = prepare_message("Fix the bug", {"file": "auth.py"})

    assert "File: auth.py" in prepared.runtime_message
    assert prepared.runtime_message.endswith("Fix the bug")
    assert prepared.response_message == "Fix the bug"


def test_message_execution_does_not_expose_runtime_resolution_helpers() -> None:
    import victor.framework.message_execution as message_execution

    assert not hasattr(message_execution, "resolve_chat_runtime")
    assert not hasattr(message_execution, "resolve_chat_service")
    assert not hasattr(message_execution, "resolve_execution_context")
    assert not hasattr(message_execution, "resolve_runtime_services")


@pytest.mark.asyncio
async def test_execute_message_normalizes_direct_response_output() -> None:
    from victor.framework.message_execution import execute_message

    orchestrator = _make_orchestrator()
    chat_service = SimpleNamespace(chat=AsyncMock(return_value="READY"))
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=chat_service))

    result = await execute_message(
        orchestrator=orchestrator,
        execution_context=execution_context,
        user_message="Reply with exactly READY",
    )

    chat_service.chat.assert_awaited_once_with("Reply with exactly READY")
    assert result.success is True
    assert result.content == "READY"
    assert result.metadata["stage"] == "INITIAL"


@pytest.mark.asyncio
async def test_execute_message_can_forward_stream_flag_when_requested() -> None:
    from victor.framework.message_execution import execute_message

    orchestrator = _make_orchestrator()
    response = CompletionResponse(
        content="Streamed response",
        role="assistant",
        tool_calls=[],
        stop_reason="stop",
        usage=None,
        model="test-model",
    )
    chat_service = SimpleNamespace(chat=AsyncMock(return_value=response))
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=chat_service))

    await execute_message(
        orchestrator=orchestrator,
        execution_context=execution_context,
        user_message="hello",
        stream=True,
        forward_stream_option=True,
    )

    chat_service.chat.assert_awaited_once_with("hello", stream=True)


@pytest.mark.asyncio
async def test_execute_message_forwards_runtime_context_overrides_when_supported() -> None:
    from victor.framework.message_execution import execute_message

    orchestrator = _make_orchestrator()
    response = CompletionResponse(
        content="Overlay response",
        role="assistant",
        tool_calls=[],
        stop_reason="stop",
        usage=None,
        model="test-model",
    )
    chat_service = SimpleNamespace(chat=AsyncMock(return_value=response))
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=chat_service))
    runtime_context_overrides = {
        "prompt_overlays": [
            {
                "name": "decorator.doc_agent.system_prompt",
                "content": "You are a documentation writer.",
                "placement": "turn_prefix",
            }
        ]
    }

    await execute_message(
        orchestrator=orchestrator,
        execution_context=execution_context,
        user_message="write docs",
        runtime_context_overrides=runtime_context_overrides,
    )

    chat_service.chat.assert_awaited_once_with(
        "write docs",
        runtime_context_overrides=runtime_context_overrides,
    )


@pytest.mark.asyncio
async def test_execute_message_preserves_tool_calls_from_completion_like_objects() -> None:
    from victor.framework.message_execution import execute_message

    orchestrator = _make_orchestrator()
    tool_calls = [{"id": "call_1", "name": "read", "arguments": {"path": "a.py"}}]
    completion_like = SimpleNamespace(
        content="done",
        role="assistant",
        tool_calls=tool_calls,
        usage={"total_tokens": 1},
        stop_reason="stop",
        model="test-model",
    )
    chat_service = SimpleNamespace(chat=AsyncMock(return_value=completion_like))
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=chat_service))

    result = await execute_message(
        orchestrator=orchestrator,
        execution_context=execution_context,
        user_message="Read a.py",
    )

    assert result.success is True
    assert result.tool_calls == tool_calls


@pytest.mark.asyncio
async def test_execute_message_attaches_task_report_metrics() -> None:
    from victor.framework.message_execution import execute_message

    orchestrator = _make_orchestrator()
    orchestrator.get_last_task_report = MagicMock(
        return_value={
            "task_id": "task-123",
            "api_prompt_tokens": 32,
            "api_completion_tokens": 14,
            "api_total_tokens": 46,
            "request_count": 3,
            "cache_read_tokens": 8,
            "cache_hit_rate": 0.2,
            "tool_schema_tokens": 144,
            "compaction_saved_tokens": 21,
            "compaction_messages_removed": 2,
            "total_cost_usd": 0.0042,
            "metadata": {
                "continuation_ledger": "Intent: fix auth; Plan: inspect tests; patch parser",
            },
        }
    )
    response = CompletionResponse(
        content="done",
        role="assistant",
        tool_calls=[],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        stop_reason="stop",
        model="test-model",
    )
    chat_service = SimpleNamespace(chat=AsyncMock(return_value=response))
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=chat_service))

    result = await execute_message(
        orchestrator=orchestrator,
        execution_context=execution_context,
        user_message="Fix auth flow",
    )

    assert result.metadata["tokens_input"] == 32
    assert result.metadata["tokens_output"] == 14
    assert result.metadata["tokens_used"] == 46
    assert result.metadata["turns"] == 3
    assert result.metadata["cached_tokens"] == 8
    assert result.metadata["cache_hit_rate"] == pytest.approx(0.2)
    assert result.metadata["tool_schema_tokens"] == 144
    assert result.metadata["compaction_saved_tokens"] == 21
    assert result.metadata["compaction_messages_removed"] == 2
    assert result.metadata["cost_usd_micros"] == 4200
    assert result.metadata["task_report"]["task_id"] == "task-123"


@pytest.mark.asyncio
async def test_iter_runtime_stream_events_uses_stream_with_events_for_chat_runtime() -> None:
    from victor.framework.message_execution import iter_runtime_stream_events

    runtime = SimpleNamespace(stream_chat=MagicMock())

    async def fake_stream_with_events(runtime_obj, prompt, **kwargs):
        assert runtime_obj is runtime
        assert prompt == "ping"
        assert kwargs["response_prompt"] == "ping"
        yield AgentExecutionEvent(type=EventType.CONTENT, content="pong")

    with patch("victor.framework._internal.stream_with_events", fake_stream_with_events):
        events = [event async for event in iter_runtime_stream_events(runtime, "ping")]

    assert len(events) == 1
    assert events[0].content == "pong"


@pytest.mark.asyncio
async def test_agent_run_delegates_to_shared_message_executor() -> None:
    orchestrator = _make_orchestrator()
    agent = Agent(orchestrator)
    expected = TaskResult(content="READY", success=True)
    agent_module = importlib.import_module("victor.framework.agent")

    with patch.object(
        agent_module, "execute_message", new=AsyncMock(return_value=expected)
    ) as mock_execute:
        result = await agent.run("Reply with exactly READY", context={"file": "auth.py"})

    assert result is expected
    mock_execute.assert_awaited_once_with(
        orchestrator=orchestrator,
        execution_context=None,
        user_message="Reply with exactly READY",
        context={"file": "auth.py"},
    )


@pytest.mark.asyncio
async def test_victor_client_chat_delegates_to_shared_message_executor() -> None:
    config = SessionConfig.from_cli_flags(tool_budget=4, enable_smart_routing=True)
    client = VictorClient(config, container=object())
    orchestrator = _make_orchestrator()
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=MagicMock()))

    class _FakeAgent:
        def __init__(self):
            self.execution_context = execution_context

        def get_orchestrator(self):
            return orchestrator

    client._agent = _FakeAgent()
    client._context = execution_context

    expected = TaskResult(
        content="Service response",
        tool_calls=[{"name": "read"}],
        success=True,
        metadata={"stage": "INITIAL"},
    )

    with patch(
        "victor.framework.client.execute_message",
        new=AsyncMock(return_value=expected),
    ) as mock_execute:
        result = await client.chat("ping", stream=True)

    assert isinstance(result, TaskResult)
    assert result.content == "Service response"
    assert result.tool_count == 1
    assert result.metadata["tool_budget"] == 4
    assert result.metadata["smart_routing"] is True
    mock_execute.assert_awaited_once_with(
        orchestrator=orchestrator,
        execution_context=execution_context,
        user_message="ping",
        stream=True,
        forward_stream_option=True,
    )


def test_build_task_result_metadata_propagates_agentic_loop_iterations():
    """The real loop iteration count must reach TaskResult.metadata (A/B turn-count fix)."""
    from victor.framework.message_execution import _build_task_result_metadata

    response = SimpleNamespace(
        usage={}, model="m", stop_reason="stop", metadata={"agentic_loop_iterations": 6}
    )
    md = _build_task_result_metadata(SimpleNamespace(), response)
    assert md["agentic_loop_iterations"] == 6
    # absent on the response → 0 (harness then falls back to "turns"/proxy)
    md2 = _build_task_result_metadata(
        SimpleNamespace(),
        SimpleNamespace(usage={}, model="m", stop_reason="stop", metadata={}),
    )
    assert md2["agentic_loop_iterations"] == 0
