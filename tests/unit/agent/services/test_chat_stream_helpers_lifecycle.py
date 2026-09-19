from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin
from victor.agent.services.chat_runtime_services import (
    ChatConversation,
    ChatFeedback,
    ChatStreamLifecycle,
)
from victor.agent.streaming.context import StreamingChatContext


class _Helper(ChatStreamHelperMixin):
    def __init__(self, orchestrator):
        self._orchestrator = orchestrator
        lifecycle = SimpleNamespace(
            begin=MagicMock(),
            is_cancelled=MagicMock(return_value=orchestrator.cancelled),
            finish=MagicMock(),
        )
        self.lifecycle = lifecycle
        self.services = SimpleNamespace(
            conversation=ChatConversation(),
            feedback=ChatFeedback(
                recorder=SimpleNamespace(
                    record_outcome=orchestrator._record_runtime_intelligence_outcome
                )
            ),
            stream_lifecycle=ChatStreamLifecycle(lifecycle),
        )


@pytest.mark.asyncio
async def test_pre_iteration_uses_context_lifecycle_before_legacy_compactor():
    lifecycle = SimpleNamespace(
        after_agent_turn=AsyncMock(
            return_value={
                "compacted": True,
                "messages_removed": 2,
                "tokens_freed": 80,
                "strategy": "tiered",
            }
        )
    )
    legacy_compactor = MagicMock()
    orch = SimpleNamespace(
        cancelled=False,
        _record_runtime_intelligence_outcome=MagicMock(),
        _context_lifecycle_service=lifecycle,
        _context_compactor=legacy_compactor,
        get_messages=MagicMock(return_value=[{"role": "user", "content": "hello"}]),
        active_session_id="session_root",
        agent_id="root_agent",
        display_name="Root Agent",
        settings=SimpleNamespace(
            context_compaction_strategy="tiered", stream_idle_timeout_seconds=300
        ),
        tool_calls_used=0,
    )
    stream_ctx = StreamingChatContext(user_message="investigate runtime", total_iterations=1)
    helper = _Helper(orch)

    chunks = [chunk async for chunk in helper._run_iteration_pre_checks(stream_ctx, "hello")]

    assert chunks == []
    lifecycle.after_agent_turn.assert_awaited_once()
    runtime_context = lifecycle.after_agent_turn.await_args.args[0]
    assert runtime_context.agent_id == "root_agent"
    assert runtime_context.session_id == "session_root"
    assert lifecycle.after_agent_turn.await_args.kwargs["messages"] == [
        {"role": "user", "content": "hello"}
    ]
    legacy_compactor.check_and_compact.assert_not_called()
    assert stream_ctx.compaction_occurred is True
    assert stream_ctx.last_compaction_reason == "pre_iteration"
    assert stream_ctx.last_compaction_policy_reason == "context_lifecycle"
    assert stream_ctx.total_iterations == 2


@pytest.mark.asyncio
async def test_pre_iteration_uses_context_service_before_legacy_compactor():
    context_service = SimpleNamespace(
        get_compaction_recommendation=MagicMock(return_value={"should_compact": True}),
        compact_context=AsyncMock(return_value=3),
    )
    legacy_compactor = MagicMock()
    orch = SimpleNamespace(
        cancelled=False,
        _record_runtime_intelligence_outcome=MagicMock(),
        _context_lifecycle_service=None,
        _context_service=context_service,
        _context_compactor=legacy_compactor,
        active_session_id="session_root",
        agent_id="root_agent",
        display_name="Root Agent",
        settings=SimpleNamespace(
            context_compaction_strategy="semantic", stream_idle_timeout_seconds=300
        ),
        tool_calls_used=0,
    )
    stream_ctx = StreamingChatContext(user_message="investigate runtime", total_iterations=1)
    helper = _Helper(orch)

    chunks = [chunk async for chunk in helper._run_iteration_pre_checks(stream_ctx, "hello")]

    assert chunks == []
    context_service.get_compaction_recommendation.assert_called_once()
    context_service.compact_context.assert_awaited_once_with(
        strategy="semantic",
        min_messages=6,
    )
    legacy_compactor.check_and_compact.assert_not_called()
    assert stream_ctx.compaction_occurred is True
    assert stream_ctx.last_compaction_policy_reason == "context_service"
    assert stream_ctx.total_iterations == 2


@pytest.mark.asyncio
async def test_pre_iteration_cancellation_records_outcome_through_feedback_capability():
    recorder = MagicMock()
    orch = SimpleNamespace(
        cancelled=True,
        _record_runtime_intelligence_outcome=MagicMock(
            side_effect=AssertionError("private facade hook must not be called directly")
        ),
    )
    helper = _Helper(orch)
    helper.services.feedback = ChatFeedback(recorder=SimpleNamespace(record_outcome=recorder))
    stream_ctx = StreamingChatContext(user_message="cancel", last_quality_score=0.42)

    chunks = [chunk async for chunk in helper._run_iteration_pre_checks(stream_ctx, "cancel")]

    assert [chunk.content for chunk in chunks] == ["\n\n[Cancelled by user]\n"]
    helper.lifecycle.finish.assert_called_once_with()
    recorder.assert_called_once_with(
        success=False,
        quality_score=0.42,
        user_satisfied=False,
        completed=False,
    )
