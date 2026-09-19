from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin
from victor.agent.services.chat_runtime_services import (
    ChatCompactionEvent,
    ChatContextLifecycle,
    ChatConversation,
    ChatRuntimeIntelligence,
    ChatStreamLifecycle,
)
from victor.agent.streaming.context import StreamingChatContext


class _Helper(ChatStreamHelperMixin):
    def __init__(self, orchestrator, *, context_lifecycle=None):
        self._orchestrator = orchestrator
        lifecycle = SimpleNamespace(
            begin=MagicMock(),
            is_cancelled=MagicMock(return_value=orchestrator.cancelled),
            finish=MagicMock(),
        )
        self.lifecycle = lifecycle
        self.services = SimpleNamespace(
            conversation=ChatConversation(),
            intelligence=ChatRuntimeIntelligence(
                runtime=SimpleNamespace(
                    record_outcome=orchestrator._record_runtime_intelligence_outcome
                )
            ),
            stream_lifecycle=ChatStreamLifecycle(lifecycle),
            context_lifecycle=context_lifecycle or ChatContextLifecycle(),
        )


@pytest.mark.asyncio
async def test_pre_iteration_records_context_lifecycle_compaction():
    runtime = SimpleNamespace(
        compact_before_iteration=AsyncMock(
            return_value=ChatCompactionEvent(
                messages_removed=2,
                tokens_freed=80,
                strategy="tiered",
                policy_reason="context_lifecycle",
            )
        ),
        start_background_compaction=AsyncMock(),
    )
    orch = SimpleNamespace(
        cancelled=False,
        _record_runtime_intelligence_outcome=MagicMock(),
        settings=SimpleNamespace(
            context_compaction_strategy="tiered", stream_idle_timeout_seconds=300
        ),
    )
    stream_ctx = StreamingChatContext(user_message="investigate runtime", total_iterations=1)
    helper = _Helper(orch, context_lifecycle=ChatContextLifecycle(runtime))

    chunks = [chunk async for chunk in helper._run_iteration_pre_checks(stream_ctx, "hello")]

    assert chunks == []
    runtime.compact_before_iteration.assert_awaited_once_with("hello")
    assert stream_ctx.compaction_occurred is True
    assert stream_ctx.last_compaction_reason == "pre_iteration"
    assert stream_ctx.last_compaction_policy_reason == "context_lifecycle"
    assert stream_ctx.total_iterations == 2


@pytest.mark.asyncio
async def test_pre_iteration_records_context_service_compaction():
    runtime = SimpleNamespace(
        compact_before_iteration=AsyncMock(
            return_value=ChatCompactionEvent(
                messages_removed=3,
                summary="Compacted 3 messages via ContextService",
                strategy="semantic",
                policy_reason="context_service",
            )
        ),
        start_background_compaction=AsyncMock(),
    )
    orch = SimpleNamespace(
        cancelled=False,
        _record_runtime_intelligence_outcome=MagicMock(),
        settings=SimpleNamespace(
            context_compaction_strategy="semantic", stream_idle_timeout_seconds=300
        ),
    )
    stream_ctx = StreamingChatContext(user_message="investigate runtime", total_iterations=1)
    helper = _Helper(orch, context_lifecycle=ChatContextLifecycle(runtime))

    chunks = [chunk async for chunk in helper._run_iteration_pre_checks(stream_ctx, "hello")]

    assert chunks == []
    runtime.compact_before_iteration.assert_awaited_once_with("hello")
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
    helper.services.intelligence = ChatRuntimeIntelligence(
        runtime=SimpleNamespace(record_outcome=recorder)
    )
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
