from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from victor.agent.services.orchestrator_protocol_adapter import (
    OrchestratorProtocolAdapter,
)
from victor.agent.services.tool_execution_runtime import ToolExecutionRuntime
from victor.agent.tool_output_formatter import FormattingContext
from victor.agent.streaming.context import StreamingChatContext


def _make_runtime_host(**overrides):
    values = {
        "_tool_pipeline": SimpleNamespace(
            execute_tool_calls=AsyncMock(return_value=[{"name": "read", "success": True}]),
            calls_used=4,
        ),
        "_tool_service": MagicMock(),
        "_get_tool_context": MagicMock(return_value={"provider": "mock"}),
        "executed_tools": ["read"],
        "observed_files": {"app.py"},
        "failed_tool_signatures": set(),
        "_shown_tool_errors": set(),
        "_continuation_prompts": 1,
        "_asking_input_prompts": 2,
        "tool_calls_used": 0,
        "_record_tool_execution": MagicMock(),
        "conversation_state": MagicMock(),
        "unified_tracker": MagicMock(),
        "usage_logger": MagicMock(),
        "add_message": MagicMock(),
        "_format_tool_output": MagicMock(),
        "_tool_output_formatter": MagicMock(),
        "console": MagicMock(),
        "_presentation": MagicMock(),
        "_current_stream_context": MagicMock(),
        "_current_task_type": "analysis",
        "_conversation_controller": SimpleNamespace(
            get_context_metrics=MagicMock(
                return_value=SimpleNamespace(remaining_tokens=1234, max_tokens=8192)
            )
        ),
        "provider": SimpleNamespace(name="mock-provider"),
        "settings": SimpleNamespace(model="mock-model", response_token_reserve=512),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_tool_execution_runtime_returns_empty_for_empty_calls():
    host = _make_runtime_host()
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    assert await runtime.execute_tool_calls([]) == []


@pytest.mark.asyncio
async def test_tool_execution_runtime_filters_nondict_calls():
    host = _make_runtime_host()
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    assert await runtime.execute_tool_calls(["bad-payload"]) == []


@pytest.mark.asyncio
async def test_tool_execution_runtime_executes_pipeline_and_syncs_mutable_state():
    host = _make_runtime_host()

    def _process_results(pipeline_result, ctx):
        assert pipeline_result == [{"name": "read", "success": True}]
        assert ctx.format_tool_output.__self__ is runtime
        ctx.continuation_prompts = 5
        ctx.asking_input_prompts = 6
        return [{"name": "read", "success": True, "elapsed": 0.1}]

    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))
    host._tool_service.process_tool_results.side_effect = _process_results

    result = await runtime.execute_tool_calls([{"name": "read", "arguments": {}}, "bad-payload"])

    assert result == [{"name": "read", "success": True, "elapsed": 0.1}]
    host._tool_pipeline.execute_tool_calls.assert_awaited_once_with(
        tool_calls=[{"name": "read", "arguments": {}}],
        context={"provider": "mock"},
        interrupted_results=ANY,
    )
    assert host.tool_calls_used == 4
    assert host._continuation_prompts == 5
    assert host._asking_input_prompts == 6


@pytest.mark.asyncio
async def test_tool_execution_runtime_creates_execution_checkpoint_before_write_batch():
    stream_ctx = StreamingChatContext(user_message="update app", total_iterations=1)
    host = _make_runtime_host(
        save_checkpoint=AsyncMock(return_value="conversation-ckpt-1"),
        active_session_id="session-1",
        _current_stream_context=stream_ctx,
    )
    host._tool_service.process_tool_results.return_value = [
        {"name": "write", "success": True, "elapsed": 0.1}
    ]

    async def execute_tool_calls(*args, **kwargs):
        checkpoint = host._last_execution_checkpoint
        assert checkpoint.session_id == "session-1"
        assert checkpoint.conversation_checkpoint_id == "conversation-ckpt-1"
        assert checkpoint.triggering_tool_call["name"] == "write"
        return [{"name": "write", "success": True}]

    host._tool_pipeline.execute_tool_calls = AsyncMock(side_effect=execute_tool_calls)
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    results = await runtime.execute_tool_calls(
        [
            {
                "id": "call_write_1",
                "name": "write",
                "arguments": {"path": "victor/app.py", "content": "print('hi')"},
            }
        ]
    )

    host.save_checkpoint.assert_awaited_once_with(
        description="Before tool write modifies files",
        tags=["execution", "pre_tool", "tool:write"],
    )
    assert len(host._execution_checkpoints) == 1
    assert host._execution_checkpoints[0] is host._last_execution_checkpoint
    assert stream_ctx.intent_log[0]["kind"] == "execution_checkpoint"
    assert stream_ctx.intent_log[0]["tool"] == "write"
    result_event = next(event for event in stream_ctx.intent_log if event["kind"] == "tool_result")
    assert result_event["execution_checkpoint_id"] == host._last_execution_checkpoint.id
    assert result_event["conversation_checkpoint_id"] == "conversation-ckpt-1"
    assert results[0]["execution_checkpoint_id"] == host._last_execution_checkpoint.id
    assert results[0]["conversation_checkpoint_id"] == "conversation-ckpt-1"
    assert results[0]["filesystem_checkpoint_id"] is None


@pytest.mark.asyncio
async def test_tool_execution_runtime_links_filesystem_checkpoint_owner():
    filesystem_manager = SimpleNamespace(
        create=MagicMock(return_value=SimpleNamespace(id="git-ckpt-1"))
    )
    host = _make_runtime_host(
        save_checkpoint=AsyncMock(return_value=None),
        git_checkpoint_manager=filesystem_manager,
        active_session_id="session-1",
    )
    host._tool_service.process_tool_results.return_value = [
        {"name": "edit", "success": True, "elapsed": 0.1}
    ]
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    await runtime.execute_tool_calls(
        [
            {
                "id": "call_edit_1",
                "name": "edit",
                "arguments": {"path": "victor/app.py", "old": "a", "new": "b"},
            }
        ]
    )

    filesystem_manager.create.assert_called_once_with(description="Before tool edit modifies files")
    assert host._last_execution_checkpoint.filesystem_checkpoint_id == "git-ckpt-1"


@pytest.mark.asyncio
async def test_tool_execution_runtime_skips_execution_checkpoint_for_read_only_batch():
    host = _make_runtime_host(save_checkpoint=AsyncMock())
    host._tool_service.process_tool_results.return_value = [
        {"name": "read", "success": True, "elapsed": 0.1}
    ]
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    await runtime.execute_tool_calls(
        [{"id": "call_read_1", "name": "read", "arguments": {"path": "victor/app.py"}}]
    )

    host.save_checkpoint.assert_not_awaited()
    assert not hasattr(host, "_last_execution_checkpoint")


@pytest.mark.asyncio
async def test_tool_execution_runtime_backfills_missing_tool_response_ids():
    host = _make_runtime_host(
        conversation=SimpleNamespace(_messages=[]),
    )
    host._tool_service.process_tool_results.return_value = [
        {"name": "read", "success": True, "elapsed": 0.1, "tool_call_id": "call_1"}
    ]
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.execute_tool_calls(
        [
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
        ]
    )

    assert len(result) == 2
    assert any(entry.get("tool_call_id") == "call_2" for entry in result)
    host.add_message.assert_called_once_with(
        "tool",
        (
            "Tool result unavailable for 'metrics'. Victor did not complete "
            "post-processing for this tool call, so treat it as failed and continue "
            "with the available context."
        ),
        name="metrics",
        tool_call_id="call_2",
        persist_synchronously=True,
    )


@pytest.mark.asyncio
async def test_tool_execution_runtime_backfill_survives_add_message_failure():
    host = _make_runtime_host(
        conversation=SimpleNamespace(_messages=[]),
    )
    host.add_message.side_effect = RuntimeError("conversation write failed")
    host._tool_service.process_tool_results.return_value = []
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    result = await runtime.execute_tool_calls(
        [
            {
                "id": "call_2",
                "name": "metrics",
                "arguments": {"path": "victor/framework/graph.py"},
            },
        ]
    )

    assert len(result) == 1
    assert result[0]["tool_call_id"] == "call_2"
    assert result[0]["outcome_kind"] == "tool_response_missing"


@pytest.mark.asyncio
async def test_tool_execution_runtime_compacts_before_large_tool_output_and_records_ledger():
    stream_ctx = StreamingChatContext(user_message="investigate runtime", total_iterations=2)
    pipeline_result = SimpleNamespace(
        results=[
            SimpleNamespace(
                result={"content": "x" * 9000},
                error=None,
            )
        ]
    )
    context_service = SimpleNamespace(
        prepare_for_tool_output_injection=AsyncMock(
            return_value={
                "should_compact": True,
                "compacted": True,
                "messages_removed": 3,
                "saved_tokens": 120,
                "strategy": "hybrid",
                "reason": "pre_tool_output",
                "policy_reason": "high_utilization_large_tool_output",
            }
        )
    )
    host = _make_runtime_host(
        _tool_pipeline=SimpleNamespace(
            execute_tool_calls=AsyncMock(return_value=pipeline_result),
            calls_used=1,
        ),
        _tool_service=MagicMock(),
        _context_service=context_service,
        settings=SimpleNamespace(context_compaction_strategy="tiered"),
        provider=SimpleNamespace(name="deepseek"),
        model="deepseek-chat",
        _conversation_controller=SimpleNamespace(
            get_compaction_summaries=MagicMock(return_value=["trimmed stale tool chatter"])
        ),
        _current_stream_context=stream_ctx,
    )
    host._tool_service.process_tool_results.return_value = [
        {"name": "read", "success": True, "elapsed": 0.1, "args": {"path": "app.py"}}
    ]

    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    await runtime.execute_tool_calls([{"name": "read", "arguments": {"path": "app.py"}}])

    context_service.prepare_for_tool_output_injection.assert_awaited_once()
    args, kwargs = context_service.prepare_for_tool_output_injection.await_args
    assert args and args[0] >= 2200
    assert kwargs == {
        "provider_name": "deepseek",
        "model_name": "deepseek-chat",
        "task_type": "analysis",
        "min_messages": 6,
        "default_strategy": "tiered",
    }
    assert stream_ctx.compaction_occurred is True
    assert stream_ctx.last_compaction_strategy == "hybrid"
    assert stream_ctx.last_compaction_reason == "pre_tool_output"
    assert stream_ctx.last_compaction_policy_reason == "high_utilization_large_tool_output"
    assert any(event["kind"] == "tool_intent" for event in stream_ctx.intent_log)
    assert any(event["kind"] == "tool_result" for event in stream_ctx.intent_log)


@pytest.mark.asyncio
async def test_tool_execution_runtime_prefers_context_lifecycle_for_large_tool_output():
    stream_ctx = StreamingChatContext(user_message="investigate runtime", total_iterations=2)
    pipeline_result = SimpleNamespace(
        results=[SimpleNamespace(result={"content": "x" * 9000}, error=None)]
    )
    context_lifecycle = SimpleNamespace(
        before_tool_output=AsyncMock(
            return_value={
                "should_compact": True,
                "compacted": True,
                "messages_removed": 2,
                "saved_tokens": 80,
                "strategy": "semantic",
                "reason": "pre_tool_output",
                "policy_reason": "high_utilization_large_tool_output",
                "compaction_event_id": "compact_1",
                "summary": "Lifecycle summary of removed tool context",
            }
        )
    )
    context_service = SimpleNamespace(prepare_for_tool_output_injection=AsyncMock())
    host = _make_runtime_host(
        _tool_pipeline=SimpleNamespace(
            execute_tool_calls=AsyncMock(return_value=pipeline_result),
            calls_used=1,
        ),
        _tool_service=MagicMock(),
        _context_lifecycle_service=context_lifecycle,
        _context_service=context_service,
        active_session_id="session_root",
        agent_id="root_agent",
        display_name="Root Agent",
        messages=[{"role": "user", "content": "hello"}],
        settings=SimpleNamespace(context_compaction_strategy="tiered"),
        provider=SimpleNamespace(name="deepseek"),
        model="deepseek-chat",
        _conversation_controller=SimpleNamespace(
            get_compaction_summaries=MagicMock(return_value=["trimmed stale tool chatter"])
        ),
        _current_stream_context=stream_ctx,
    )
    host._tool_service.process_tool_results.return_value = []

    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    await runtime.execute_tool_calls([{"name": "read", "arguments": {"path": "app.py"}}])

    context_lifecycle.before_tool_output.assert_awaited_once()
    runtime_context = context_lifecycle.before_tool_output.await_args.args[0]
    assert runtime_context.agent_id == "root_agent"
    assert runtime_context.session_id == "session_root"
    assert context_lifecycle.before_tool_output.await_args.kwargs["messages"] == [
        {"role": "user", "content": "hello"}
    ]
    context_service.prepare_for_tool_output_injection.assert_not_called()
    assert stream_ctx.compaction_occurred is True
    assert stream_ctx.last_compaction_strategy == "semantic"
    assert stream_ctx.compaction_summary == "Lifecycle summary of removed tool context"


def test_tool_execution_runtime_formats_output_with_runtime_context():
    formatter = MagicMock()
    formatter.format_tool_output.return_value = "formatted output"
    host = _make_runtime_host(
        _tool_output_formatter=formatter,
        _conversation_controller=SimpleNamespace(
            get_context_metrics=MagicMock(
                return_value=SimpleNamespace(remaining_tokens=6400, max_tokens=16000)
            )
        ),
        provider=SimpleNamespace(name="openai"),
        settings=SimpleNamespace(model="gpt-4.1", response_token_reserve=2048),
    )
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))

    result = runtime.format_tool_output("read", {"path": "app.py"}, {"content": "hello"})

    assert result == "formatted output"
    formatter.format_tool_output.assert_called_once()
    _, kwargs = formatter.format_tool_output.call_args
    assert kwargs["tool_name"] == "read"
    assert kwargs["args"] == {"path": "app.py"}
    assert kwargs["output"] == {"content": "hello"}
    context = kwargs["context"]
    assert isinstance(context, FormattingContext)
    assert context.provider_name == "openai"
    assert context.model == "gpt-4.1"
    assert context.remaining_tokens == 6400
    assert context.max_tokens == 16000
    assert context.response_token_reserve == 2048


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("withheld", [False, True])
async def test_interrupted_batch_publishes_only_known_results(cancel, withheld):
    import asyncio
    from victor.agent.tool_pipeline import ToolCallResult
    from victor.agent.services.tool_service import process_tool_results_with_context
    from victor.agent.durable_resume import _find_unresolved_calls
    from victor.framework.approval_pause import ApprovalPause
    from victor.framework.hitl import ApprovalRequest
    from victor.core.errors import ErrorInfo, ErrorCategory, ErrorSeverity

    control = (
        asyncio.CancelledError("stop")
        if cancel
        else ApprovalPause(ApprovalRequest(id="approval", title="Review", description="read"))
    )
    info = ErrorInfo(
        message="uncertain",
        category=ErrorCategory.TOOL_EXECUTION,
        severity=ErrorSeverity.ERROR,
        correlation_id="test",
        details={"execution_outcome": "unknown", "reconciliation_required": True},
    )
    calls = [
        {"id": name, "name": "write" if name != "pending" else "read", "arguments": {}}
        for name in ("done", "unknown", "pending")
    ]
    messages = [SimpleNamespace(role="assistant", tool_calls=calls)]
    host = _make_runtime_host()
    host._tool_service.process_tool_results.side_effect = process_tool_results_with_context
    host._tool_output_formatter.format_tool_output.return_value = "confirmed result"
    host.add_message.side_effect = lambda role, content, **kw: messages.append(
        SimpleNamespace(role=role, content=content, **kw)
    )

    async def interrupted(**kwargs):
        kwargs["interrupted_results"].extend(
            [
                ToolCallResult(
                    tool_name="write",
                    arguments={},
                    success=True,
                    result="done",
                    tool_call_id="done",
                ),
                ToolCallResult(
                    tool_name="write",
                    arguments={},
                    success=False,
                    tool_call_id="unknown",
                    error_info=info,
                    outcome_kind="result_withheld" if withheld else None,
                ),
            ]
        )
        raise control

    host._tool_pipeline.execute_tool_calls.side_effect = interrupted
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))
    runtime._compact_before_tool_result_injection = AsyncMock()
    with pytest.raises(type(control)) as caught:
        await runtime.execute_tool_calls(calls)
    assert caught.value is control
    assert [m.tool_call_id for m in messages if m.role == "tool"] == ["done"]
    assert [call_id for _, call_id in _find_unresolved_calls(messages)] == ["unknown", "pending"]
    assert host.tool_calls_used == 4
    runtime._compact_before_tool_result_injection.assert_not_awaited()
    assert host.add_message.call_args.kwargs["require_persistence"] is True


@pytest.mark.parametrize("failure", ["format", "persist"])
@pytest.mark.parametrize("readonly_context", [False, True])
async def test_interrupted_publication_failure_preserves_control_without_fallback(
    failure, readonly_context
):
    from victor.agent.tool_pipeline import ToolCallResult
    from victor.agent.services.tool_service import process_tool_results_with_context
    from victor.framework.approval_pause import ApprovalPause
    from victor.framework.hitl import ApprovalRequest

    pause = ApprovalPause(ApprovalRequest(id="a", title="Review", description="read"))
    if readonly_context:
        request = pause.request

        class ReadOnlyContextAdapter:
            @property
            def context(self):
                return request.context

        pause.request = ReadOnlyContextAdapter()
    host = _make_runtime_host()
    host._tool_service.process_tool_results.side_effect = process_tool_results_with_context
    host._tool_output_formatter.format_tool_output.return_value = "known result"
    messages = []

    def persist(role, content, **kwargs):
        messages.append((role, content, kwargs))
        raise TypeError("failure after append must not trigger another append")

    if failure == "persist":
        host.add_message.side_effect = persist
    else:
        host._tool_output_formatter.format_tool_output.side_effect = RuntimeError("format failed")

    async def interrupted(**kwargs):
        kwargs["interrupted_results"].append(
            ToolCallResult(
                tool_name="write", arguments={}, success=True, result="done", tool_call_id="done"
            )
        )
        raise pause

    host._tool_pipeline.execute_tool_calls.side_effect = interrupted
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))
    with pytest.raises(ApprovalPause) as caught:
        await runtime.execute_tool_calls([{"id": "pending", "name": "read", "arguments": {}}])
    assert caught.value is pause
    assert pause.request.context["batch_result_publication"] == {
        "schema_version": 1,
        "status": "incomplete",
    }
    assert host.add_message.call_count == (1 if failure == "persist" else 0)
    assert all(content == "known result" for _, content, _ in messages)


@pytest.mark.parametrize("member", [False, True])
async def test_completed_receipt_allows_exact_action_but_blocks_whole_member_replay(member):
    from victor.agent.member_approval_context import MemberApprovalPause
    from victor.agent.tool_pipeline import ToolCallResult
    from victor.agent.services.tool_service import process_tool_results_with_context
    from victor.framework.approval_pause import ApprovalPause
    from victor.framework.hitl import ApprovalRequest
    from victor.framework.graph_checkpoint import MemoryCheckpointer
    from victor.teams import UnifiedTeamCoordinator, TeamFormation
    from victor.teams.types import MemberResult
    from victor.framework.approval_binding import ApprovalBindingError

    pause_type = MemberApprovalPause if member else ApprovalPause
    pause = pause_type(ApprovalRequest(id="a", title="Review", description="read"))
    host = _make_runtime_host()
    host._tool_service.process_tool_results.side_effect = process_tool_results_with_context
    host._tool_output_formatter.format_tool_output.return_value = "receipt"

    async def interrupted(**kwargs):
        kwargs["interrupted_results"].append(
            ToolCallResult(
                tool_name="write", arguments={}, success=True, result="done", tool_call_id="done"
            )
        )
        raise pause

    host._tool_pipeline.execute_tool_calls.side_effect = interrupted
    runtime = ToolExecutionRuntime(OrchestratorProtocolAdapter(host))
    with pytest.raises(pause_type) as caught:
        await runtime.execute_tool_calls([{"id": "pending", "name": "read", "arguments": {}}])
    assert caught.value is pause
    host.add_message.assert_called_once()
    if member:
        assert (
            pause.request.context["batch_result_publication"]["reason"]
            == "member_continuation_required"
        )
        cp = MemoryCheckpointer()
        coord = UnifiedTeamCoordinator(lightweight_mode=True, checkpointer=cp)
        hook = coord._make_member_pause_hook(cp, "thread", "team", TeamFormation.SEQUENTIAL.value)
        # Persist through the coordinator's actual pause-hook envelope.
        await hook(
            0,
            MemberResult(
                member_id="member",
                success=False,
                output="",
                metadata={"approval_request": pause.request.to_dict()},
            ),
            [],
            {},
        )
        with pytest.raises(ApprovalBindingError, match="reconciliation"):
            await coord._load_member_resume(cp, "thread", "team")
    else:
        assert "batch_result_publication" not in pause.request.context
