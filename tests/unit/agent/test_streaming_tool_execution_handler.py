import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.conversation.history_metadata import build_internal_history_metadata
from victor.agent.streaming.context import StreamingChatContext
from victor.agent.streaming.tool_execution import (
    ToolExecutionHandler,
    ToolExecutionResult,
    create_tool_execution_handler,
)
from victor.providers.base import StreamChunk


def test_factory_prefers_canonical_recovery_context_builder():
    def canonical(stream_ctx):
        return {"stream_ctx": stream_ctx}

    orchestrator = SimpleNamespace(
        create_recovery_context=canonical,
        _recovery_service=None,
        _recovery_coordinator=MagicMock(),
        _chunk_generator=MagicMock(),
        reminder_manager=MagicMock(),
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=MagicMock(),
        execute_tool_calls=MagicMock(),
        observed_files=set(),
    )

    handler = create_tool_execution_handler(orchestrator)

    assert handler._recovery_context_factory is canonical
    assert handler._recovery_context_factory("ctx") == {"stream_ctx": "ctx"}


@pytest.mark.asyncio
async def test_execute_tools_invokes_callback_without_signature_collision():
    recovery_runtime = SimpleNamespace(
        check_tool_budget=AsyncMock(return_value=None),
        truncate_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls, remaining: (tool_calls, remaining)
        ),
        filter_blocked_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls: (tool_calls, [], 0)
        ),
        check_blocked_threshold=MagicMock(return_value=None),
    )
    chunk_generator = SimpleNamespace(
        generate_tool_start_chunk=MagicMock(return_value=StreamChunk(content="start")),
        generate_tool_result_chunks=MagicMock(return_value=[StreamChunk(content="done")]),
    )
    reminder_manager = SimpleNamespace(
        update_state=MagicMock(),
        get_consolidated_reminder=MagicMock(return_value=None),
    )
    execute_tool_calls = AsyncMock(
        return_value=[{"name": "read", "success": True, "args": {}, "elapsed": 1.0}]
    )

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=chunk_generator,
        message_adder=SimpleNamespace(add_message=MagicMock()),
        reminder_manager=reminder_manager,
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=SimpleNamespace(tool_call_budget_warning_threshold=250),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=execute_tool_calls,
        get_tool_status_message=lambda tool_name, tool_args: f"{tool_name}: {tool_args}",
        observed_files=set(),
    )

    stream_ctx = StreamingChatContext(user_message="Run the read tool")
    tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/orchestrator.py"}}]

    result = await handler.execute_tools(
        stream_ctx=stream_ctx,
        tool_calls=tool_calls,
        user_message="Run the read tool",
        full_content="read the file",
        tool_calls_used=0,
        tool_budget=5,
    )

    execute_tool_calls.assert_awaited_once_with(tool_calls)
    assert result.tool_calls_executed == 1
    assert [chunk.content for chunk in result.chunks] == ["start", "done"]
    assert stream_ctx.executed_tool_names == {"read"}


@pytest.mark.asyncio
async def test_execute_tools_streaming_yields_tool_start_before_callback_awaits():
    recovery_runtime = SimpleNamespace(
        check_tool_budget=AsyncMock(return_value=None),
        truncate_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls, remaining: (tool_calls, remaining)
        ),
        filter_blocked_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls: (tool_calls, [], 0)
        ),
        check_blocked_threshold=MagicMock(return_value=None),
    )
    chunk_generator = SimpleNamespace(
        generate_tool_start_chunk=MagicMock(return_value=StreamChunk(content="start")),
        generate_tool_result_chunks=MagicMock(return_value=[StreamChunk(content="done")]),
    )
    reminder_manager = SimpleNamespace(
        update_state=MagicMock(),
        get_consolidated_reminder=MagicMock(return_value=None),
    )
    execution_started = False
    release_execution = asyncio.Event()

    async def execute_tool_calls(tool_calls):
        nonlocal execution_started
        execution_started = True
        await release_execution.wait()
        return [{"name": "graph", "success": True, "args": {}, "elapsed": 1.0}]

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=chunk_generator,
        message_adder=SimpleNamespace(add_message=MagicMock()),
        reminder_manager=reminder_manager,
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=SimpleNamespace(tool_call_budget_warning_threshold=250),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=execute_tool_calls,
        get_tool_status_message=lambda tool_name, tool_args: f"{tool_name}: {tool_args}",
        observed_files=set(),
    )

    stream_ctx = StreamingChatContext(user_message="Inspect the graph")
    result = ToolExecutionResult()
    stream = handler.execute_tools_streaming(
        stream_ctx=stream_ctx,
        tool_calls=[{"name": "graph", "arguments": {"mode": "overview"}}],
        user_message="Inspect the graph",
        full_content="",
        tool_calls_used=0,
        tool_budget=5,
        result=result,
    )

    first_chunk = await asyncio.wait_for(stream.__anext__(), timeout=0.1)
    assert first_chunk.content == "start"
    assert execution_started is False

    release_execution.set()
    remaining_chunks = [chunk async for chunk in stream]
    assert [chunk.content for chunk in remaining_chunks] == ["done"]
    assert result.tool_calls_executed == 1
    assert stream_ctx.executed_tool_names == {"graph"}


def test_tool_start_chunks_include_batch_metadata():
    recovery_runtime = SimpleNamespace()

    def start_chunk(tool_name, tool_args, status_msg, tool_call_id=None):
        return StreamChunk(
            content="",
            metadata={
                "tool_start": {
                    "name": tool_name,
                    "arguments": tool_args,
                    "status_msg": status_msg,
                }
            },
        )

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=SimpleNamespace(generate_tool_start_chunk=start_chunk),
        message_adder=SimpleNamespace(add_message=MagicMock()),
        reminder_manager=SimpleNamespace(),
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=SimpleNamespace(tool_call_budget_warning_threshold=250),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=AsyncMock(),
        get_tool_status_message=lambda tool_name, tool_args: f"{tool_name}: {tool_args}",
        observed_files=set(),
    )
    result = ToolExecutionResult()

    handler._add_tool_start_chunks(
        [
            {"name": "graph", "arguments": {"mode": "overview"}},
            {"name": "graph", "arguments": {"mode": "centrality"}},
        ],
        result,
    )

    first = result.chunks[0].metadata["tool_start"]
    second = result.chunks[1].metadata["tool_start"]
    assert first["batch_index"] == 1
    assert first["batch_total"] == 2
    assert first["execution_mode"] == "parallel_batch"
    assert second["batch_index"] == 2


def test_tool_start_chunks_survive_status_message_failure():
    """A raising status-message helper must not abort the streaming turn.

    Regression: glm-5.2 emitted `write` arguments as a JSON list; the dict-assuming
    status helper raised AttributeError and killed the whole turn before execution
    (session codingagent-363cca81, 2026-07-24).
    """
    recovery_runtime = SimpleNamespace()

    def start_chunk(tool_name, tool_args, status_msg, tool_call_id=None):
        return StreamChunk(
            content="",
            metadata={
                "tool_start": {
                    "name": tool_name,
                    "arguments": tool_args,
                    "status_msg": status_msg,
                }
            },
        )

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    def raising_status_message(tool_name, tool_args):
        return tool_args["path"]  # TypeError on list args, like dict-assuming helper

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=SimpleNamespace(generate_tool_start_chunk=start_chunk),
        message_adder=SimpleNamespace(add_message=MagicMock()),
        reminder_manager=SimpleNamespace(),
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=SimpleNamespace(tool_call_budget_warning_threshold=250),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=AsyncMock(),
        get_tool_status_message=raising_status_message,
        observed_files=set(),
    )
    result = ToolExecutionResult()

    last_tool_name = handler._add_tool_start_chunks(
        [{"name": "write", "arguments": [{"path": "a.py", "content": "x"}]}],
        result,
    )

    assert last_tool_name == "write"
    assert len(result.chunks) == 1
    assert result.chunks[0].metadata["tool_start"]["status_msg"] == "Running write..."


@pytest.mark.asyncio
async def test_execute_tools_marks_system_reminders_as_noninteractive_history():
    recovery_runtime = SimpleNamespace(
        check_tool_budget=AsyncMock(return_value=None),
        truncate_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls, remaining: (tool_calls, remaining)
        ),
        filter_blocked_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls: (tool_calls, [], 0)
        ),
        check_blocked_threshold=MagicMock(return_value=None),
    )
    chunk_generator = SimpleNamespace(
        generate_tool_start_chunk=MagicMock(return_value=StreamChunk(content="start")),
        generate_tool_result_chunks=MagicMock(return_value=[StreamChunk(content="done")]),
    )
    message_adder = SimpleNamespace(add_message=MagicMock())
    reminder_manager = SimpleNamespace(
        update_state=MagicMock(),
        get_consolidated_reminder=MagicMock(return_value="use more tools"),
    )
    execute_tool_calls = AsyncMock(
        return_value=[{"name": "read", "success": True, "args": {}, "elapsed": 1.0}]
    )

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=chunk_generator,
        message_adder=message_adder,
        reminder_manager=reminder_manager,
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=SimpleNamespace(tool_call_budget_warning_threshold=250),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=execute_tool_calls,
        get_tool_status_message=lambda tool_name, tool_args: f"{tool_name}: {tool_args}",
        observed_files=set(),
    )

    stream_ctx = StreamingChatContext(user_message="Run the read tool")
    tool_calls = [{"name": "read", "arguments": {"path": "victor/agent/orchestrator.py"}}]

    await handler.execute_tools(
        stream_ctx=stream_ctx,
        tool_calls=tool_calls,
        user_message="Run the read tool",
        full_content="read the file",
        tool_calls_used=0,
        tool_budget=5,
    )

    message_adder.add_message.assert_any_call(
        "user",
        "[SYSTEM-REMINDER: use more tools]",
        metadata=build_internal_history_metadata("system_reminder"),
    )


@pytest.mark.asyncio
async def test_execute_tools_forces_completion_after_terminal_skips():
    recovery_runtime = SimpleNamespace(
        check_tool_budget=AsyncMock(return_value=None),
        truncate_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls, remaining: (tool_calls, remaining)
        ),
        filter_blocked_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls: (tool_calls, [], 0)
        ),
        check_blocked_threshold=MagicMock(return_value=None),
    )
    chunk_generator = SimpleNamespace(
        generate_tool_start_chunk=MagicMock(return_value=StreamChunk(content="start")),
        generate_tool_result_chunks=MagicMock(return_value=[StreamChunk(content="blocked")]),
    )
    message_adder = SimpleNamespace(add_message=MagicMock())
    reminder_manager = SimpleNamespace(
        update_state=MagicMock(),
        get_consolidated_reminder=MagicMock(return_value=None),
    )
    execute_tool_calls = AsyncMock(
        return_value=[
            {
                "name": "read",
                "success": False,
                "skipped": True,
                "outcome_kind": "budget_exhausted",
            }
        ]
    )

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=chunk_generator,
        message_adder=message_adder,
        reminder_manager=reminder_manager,
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=SimpleNamespace(tool_call_budget_warning_threshold=250),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=execute_tool_calls,
        get_tool_status_message=lambda tool_name, tool_args: f"{tool_name}: {tool_args}",
        observed_files=set(),
    )

    stream_ctx = StreamingChatContext(user_message="Run the read tool")

    await handler.execute_tools(
        stream_ctx=stream_ctx,
        tool_calls=[{"name": "read", "arguments": {"path": "victor/agent/orchestrator.py"}}],
        user_message="Run the read tool",
        full_content="read the file",
        tool_calls_used=0,
        tool_budget=5,
    )

    assert stream_ctx.force_completion is True
    message_adder.add_message.assert_any_call(
        "user",
        (
            "[SYSTEM: Tool execution is no longer making progress. "
            "Do not request more blocked tools; summarize the current findings or explain "
            "the blocker directly.]"
        ),
        metadata=build_internal_history_metadata("force_completion"),
    )


@pytest.mark.asyncio
async def test_execute_tools_forces_completion_after_only_unknown_tool_skips():
    recovery_runtime = SimpleNamespace(
        check_tool_budget=AsyncMock(return_value=None),
        truncate_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls, remaining: (tool_calls, remaining)
        ),
        filter_blocked_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls: (tool_calls, [], 0)
        ),
        check_blocked_threshold=MagicMock(return_value=None),
    )
    chunk_generator = SimpleNamespace(
        generate_tool_start_chunk=MagicMock(return_value=StreamChunk(content="start")),
        generate_tool_result_chunks=MagicMock(return_value=[]),
    )
    message_adder = SimpleNamespace(add_message=MagicMock())
    reminder_manager = SimpleNamespace(
        update_state=MagicMock(),
        get_consolidated_reminder=MagicMock(return_value=None),
    )
    execute_tool_calls = AsyncMock(
        return_value=[
            {
                "name": "set_global_axis_manager",
                "success": False,
                "skipped": True,
                "outcome_kind": "tool_unavailable",
                "error": "Unknown or disabled tool",
            }
        ]
    )

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=chunk_generator,
        message_adder=message_adder,
        reminder_manager=reminder_manager,
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=SimpleNamespace(tool_call_budget_warning_threshold=250),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=execute_tool_calls,
        get_tool_status_message=lambda tool_name, tool_args: f"{tool_name}: {tool_args}",
        observed_files=set(),
    )

    stream_ctx = StreamingChatContext(user_message="summarize")
    await handler.execute_tools(
        stream_ctx=stream_ctx,
        tool_calls=[{"name": "set_global_axis_manager", "arguments": {}}],
        user_message="summarize",
        full_content="Final findings are already complete.",
        tool_calls_used=0,
        tool_budget=5,
    )

    assert stream_ctx.force_completion is True
    message_adder.add_message.assert_any_call(
        "user",
        (
            "[SYSTEM: Tool execution is no longer making progress. "
            "Do not request more blocked tools; summarize the current findings or explain "
            "the blocker directly.]"
        ),
        metadata=build_internal_history_metadata("force_completion"),
    )


@pytest.mark.asyncio
async def test_execute_tools_records_shell_alias_canonically():
    recovery_runtime = SimpleNamespace(
        check_tool_budget=AsyncMock(return_value=None),
        truncate_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls, remaining: (tool_calls, remaining)
        ),
        filter_blocked_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls: (tool_calls, [], 0)
        ),
        check_blocked_threshold=MagicMock(return_value=None),
    )
    chunk_generator = SimpleNamespace(
        generate_tool_start_chunk=MagicMock(return_value=StreamChunk(content="start")),
        generate_tool_result_chunks=MagicMock(return_value=[StreamChunk(content="done")]),
    )
    reminder_manager = SimpleNamespace(
        update_state=MagicMock(),
        get_consolidated_reminder=MagicMock(return_value=None),
    )
    execute_tool_calls = AsyncMock(
        return_value=[{"name": "bash", "success": True, "args": {}, "elapsed": 1.0}]
    )

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=chunk_generator,
        message_adder=SimpleNamespace(add_message=MagicMock()),
        reminder_manager=reminder_manager,
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=SimpleNamespace(tool_call_budget_warning_threshold=250),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=execute_tool_calls,
        get_tool_status_message=lambda tool_name, tool_args: f"{tool_name}: {tool_args}",
        observed_files=set(),
    )

    stream_ctx = StreamingChatContext(user_message="Run bash")

    await handler.execute_tools(
        stream_ctx=stream_ctx,
        tool_calls=[{"name": "bash", "arguments": {"cmd": 'sqlite3 data.db ".tables"'}}],
        user_message="Run bash",
        full_content="inspect the db",
        tool_calls_used=0,
        tool_budget=5,
    )

    assert stream_ctx.executed_tool_names == {"shell"}


@pytest.mark.asyncio
async def test_execute_tools_persists_skipped_responses_for_truncated_calls():
    recovery_runtime = SimpleNamespace(
        check_tool_budget=AsyncMock(return_value=None),
        truncate_tool_calls=MagicMock(
            return_value=(
                [
                    {
                        "id": "call_1",
                        "name": "read",
                        "arguments": {"path": "victor/framework/graph.py"},
                    }
                ],
                True,
            )
        ),
        filter_blocked_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls: (tool_calls, [], 0)
        ),
        check_blocked_threshold=MagicMock(return_value=None),
    )
    chunk_generator = SimpleNamespace(
        generate_tool_start_chunk=MagicMock(return_value=StreamChunk(content="start")),
        generate_tool_result_chunks=MagicMock(return_value=[StreamChunk(content="done")]),
    )
    message_adder = SimpleNamespace(add_message=MagicMock())
    reminder_manager = SimpleNamespace(
        update_state=MagicMock(),
        get_consolidated_reminder=MagicMock(return_value=None),
    )
    execute_tool_calls = AsyncMock(
        return_value=[
            {
                "name": "read",
                "success": True,
                "args": {"path": "victor/framework/graph.py"},
                "elapsed": 1.0,
                "tool_call_id": "call_1",
            }
        ]
    )

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=chunk_generator,
        message_adder=message_adder,
        reminder_manager=reminder_manager,
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=SimpleNamespace(tool_call_budget_warning_threshold=250),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=execute_tool_calls,
        get_tool_status_message=lambda tool_name, tool_args: f"{tool_name}: {tool_args}",
        observed_files=set(),
    )

    stream_ctx = StreamingChatContext(user_message="Review graph architecture")
    result = await handler.execute_tools(
        stream_ctx=stream_ctx,
        tool_calls=[
            {
                "id": "call_1",
                "name": "read",
                "arguments": {"path": "victor/framework/graph.py"},
            },
            {
                "id": "call_2",
                "name": "metrics",
                "arguments": {"path": "victor/framework/graph.py"},
            },
        ],
        user_message="Review graph architecture",
        full_content="inspect graph code",
        tool_calls_used=0,
        tool_budget=1,
    )

    message_adder.add_message.assert_called_once_with(
        "tool",
        (
            "Tool call skipped for 'metrics': Skipped because the remaining tool budget for "
            "this turn was exhausted. Use a different approach or continue with the "
            "available context."
        ),
        name="metrics",
        tool_call_id="call_2",
        persist_synchronously=True,
    )
    assert any(tool_result.get("tool_call_id") == "call_2" for tool_result in result.tool_results)
    assert any(tool_result.get("tool_call_id") == "call_1" for tool_result in result.tool_results)


@pytest.mark.asyncio
async def test_execute_tools_budget_exhaustion_emits_fallback_and_synthetic_tool_responses():
    recovery_runtime = SimpleNamespace(
        check_tool_budget=AsyncMock(return_value=None),
        truncate_tool_calls=MagicMock(),
        filter_blocked_tool_calls=MagicMock(),
        check_blocked_threshold=MagicMock(),
    )
    message_adder = SimpleNamespace(add_message=MagicMock())

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=SimpleNamespace(
            generate_tool_start_chunk=MagicMock(),
            generate_tool_result_chunks=MagicMock(return_value=[]),
        ),
        message_adder=message_adder,
        reminder_manager=SimpleNamespace(
            update_state=MagicMock(),
            get_consolidated_reminder=MagicMock(return_value=None),
        ),
        unified_tracker=SimpleNamespace(unique_resources=set()),
        settings=SimpleNamespace(tool_call_budget_warning_threshold=250),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=AsyncMock(),
        get_tool_status_message=lambda tool_name, tool_args: f"{tool_name}: {tool_args}",
        observed_files=set(),
    )

    stream_ctx = StreamingChatContext(
        user_message="Address the remaining findings",
        tool_calls_used=10,
        tool_budget=10,
    )
    result = await handler.execute_tools(
        stream_ctx=stream_ctx,
        tool_calls=[
            {
                "id": "call_1",
                "name": "read",
                "arguments": {"path": "victor/framework/graph.py"},
            },
            {
                "id": "call_2",
                "name": "edit",
                "arguments": {"path": "victor/framework/graph.py"},
            },
        ],
        user_message="Address the remaining findings",
        full_content="",
        tool_calls_used=10,
        tool_budget=10,
    )

    assert result.should_return is True
    assert [chunk.content for chunk in result.chunks] == [
        "[tool] Tool budget reached (10); skipped 2 queued tool call(s).\n",
        (
            "Unable to continue tool execution in this turn. Start a follow-up turn or "
            "increase the tool budget if more tool work is required.\n"
        ),
    ]
    assert any(tool_result.get("tool_call_id") == "call_1" for tool_result in result.tool_results)
    assert any(tool_result.get("tool_call_id") == "call_2" for tool_result in result.tool_results)
    assert message_adder.add_message.call_count == 2


@pytest.mark.asyncio
async def test_execute_tools_grants_progress_based_budget_relief_from_current_tracker_state():
    recovery_runtime = SimpleNamespace(
        check_tool_budget=AsyncMock(return_value=None),
        truncate_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls, remaining: (
                tool_calls[:remaining],
                len(tool_calls) > remaining,
            )
        ),
        filter_blocked_tool_calls=MagicMock(
            side_effect=lambda _ctx, tool_calls: (tool_calls, [], 0)
        ),
        check_blocked_threshold=MagicMock(return_value=None),
    )
    chunk_generator = SimpleNamespace(
        generate_tool_start_chunk=MagicMock(return_value=StreamChunk(content="start")),
        generate_tool_result_chunks=MagicMock(return_value=[StreamChunk(content="done")]),
    )
    execute_tool_calls = AsyncMock(
        return_value=[
            {"name": "read", "success": True, "args": {}, "elapsed": 1.0},
            {"name": "edit", "success": True, "args": {}, "elapsed": 1.0},
        ]
    )
    set_tool_budget_limit = MagicMock(side_effect=lambda budget: budget)

    async def _unused_async_generator(_stream_ctx):
        if False:
            yield None

    handler = ToolExecutionHandler(
        recovery_runtime=recovery_runtime,
        chunk_generator=chunk_generator,
        message_adder=SimpleNamespace(add_message=MagicMock()),
        reminder_manager=SimpleNamespace(
            update_state=MagicMock(),
            get_consolidated_reminder=MagicMock(return_value=None),
        ),
        unified_tracker=SimpleNamespace(unique_resources={"graph.py", "builder.py", "state.py"}),
        settings=SimpleNamespace(
            tool_call_budget_warning_threshold=250,
            tool_call_budget_warning_pct=0.8,
            tool_call_budget_warning_remaining=2,
            tool_budget_progress_relief_enabled=True,
            tool_budget_progress_relief_amount=2,
            tool_budget_progress_relief_max_uses=1,
        ),
        recovery_context_factory=lambda stream_ctx: {"stream_ctx": stream_ctx},
        check_progress_with_handler=lambda _stream_ctx: None,
        handle_force_completion_with_handler=lambda _stream_ctx: None,
        handle_budget_exhausted=_unused_async_generator,
        handle_force_final_response=_unused_async_generator,
        execute_tool_calls=execute_tool_calls,
        get_tool_status_message=lambda tool_name, tool_args: f"{tool_name}: {tool_args}",
        set_tool_budget_limit=set_tool_budget_limit,
        observed_files=set(),
    )

    stream_ctx = StreamingChatContext(
        user_message="Address the remaining findings comprehensively",
        tool_calls_used=4,
        tool_budget=5,
        is_action_task=True,
    )
    result = await handler.execute_tools(
        stream_ctx=stream_ctx,
        tool_calls=[
            {
                "id": "call_1",
                "name": "read",
                "arguments": {"path": "victor/framework/graph.py"},
            },
            {
                "id": "call_2",
                "name": "edit",
                "arguments": {"path": "victor/framework/graph.py"},
            },
        ],
        user_message="Address the remaining findings comprehensively",
        full_content="",
        tool_calls_used=4,
        tool_budget=5,
    )

    set_tool_budget_limit.assert_called_once_with(6)
    execute_tool_calls.assert_awaited_once()
    assert result.tool_calls_executed == 2
    assert stream_ctx.tool_budget == 6
    assert stream_ctx.budget_relief_uses == 1
    assert any(
        chunk.content
        == "[tool] Progress detected; extending tool budget to 6 calls for this turn.\n"
        for chunk in result.chunks
    )
