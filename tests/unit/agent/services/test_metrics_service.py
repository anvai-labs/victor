# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.session_cost_tracker import SessionCostTracker
from victor.agent.services.metrics_service import AgentMetricsService
from victor.providers.base import CompletionResponse


@pytest.mark.parametrize("streaming", [False, True])
async def test_invocation_accounting_includes_failures_without_inventing_usage(streaming):
    from unittest.mock import AsyncMock
    from contextlib import aclosing
    from victor.providers.base import StreamChunk

    usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}

    async def stream(**kwargs):
        yield StreamChunk(content="ok", usage=usage, is_final=True)

    provider = SimpleNamespace(name="unknown", chat=AsyncMock(), stream=stream)
    provider.chat.return_value = CompletionResponse(content="ok", role="assistant", usage=usage)
    tracker = SessionCostTracker()
    cumulative = {}
    metrics = AgentMetricsService(MagicMock(), tracker, cumulative)
    metrics.start_task_report("three invocations")
    for _ in range(2):
        if streaming:
            async with aclosing(metrics.stream(provider, messages=[], model="m")) as chunks:
                assert [chunk.content async for chunk in chunks] == ["ok"]
        else:
            await metrics.complete(provider, messages=[], model="m")

    async def failed_stream(**kwargs):
        raise RuntimeError("upstream failure")
        yield

    provider.stream = failed_stream
    provider.chat.side_effect = RuntimeError("upstream failure")
    with pytest.raises(RuntimeError, match="upstream failure"):
        if streaming:
            async with aclosing(metrics.stream(provider, messages=[], model="m")) as chunks:
                _ = [chunk async for chunk in chunks]
        else:
            await metrics.complete(provider, messages=[], model="m")
    report = metrics.finish_task_report(False)
    assert report["request_count"] == 3
    assert report["metadata"]["usage_coverage"] == {"unknown": 1, "partial": 0, "reported": 2}
    assert report["api_total_tokens"] == 26
    assert tracker.total_tokens == 26
    assert [r.outcome for r in tracker.requests] == ["success", "success", "error"]
    assert [r.usage_status for r in tracker.requests] == ["reported", "reported", "unknown"]


async def test_unused_and_early_closed_streams_have_honest_accounting():
    import asyncio
    from contextlib import aclosing
    from victor.providers.base import StreamChunk

    closed_in = []

    async def stream(**kwargs):
        try:
            yield StreamChunk(content="first", usage={"prompt_tokens": 10})
            pytest.fail("Must not drain an abandoned stream for metrics")
        finally:
            closed_in.append(asyncio.current_task())

    provider = SimpleNamespace(name="unknown", stream=stream)
    tracker = SessionCostTracker()
    metrics = AgentMetricsService(MagicMock(), tracker, {})
    unused = metrics.stream(provider, messages=[], model="m")
    await unused.aclose()
    assert not tracker.requests
    async with aclosing(metrics.stream(provider, messages=[], model="m")) as chunks:
        await anext(chunks)
    assert closed_in == [asyncio.current_task()]
    assert tracker.requests[0].outcome == "closed"
    assert tracker.requests[0].usage_status == "partial"
    assert tracker.total_prompt_tokens == 10


async def test_stream_usage_snapshots_and_transport_attempts_are_not_extra_calls():
    from contextlib import aclosing
    from victor.providers.base import StreamChunk

    usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}

    async def stream(**kwargs):
        yield StreamChunk(usage=usage)
        yield StreamChunk(usage=usage, is_final=True, metadata={"sandhi_usage": {"attempts": 2}})

    tracker = SessionCostTracker()
    cumulative = {}
    metrics = AgentMetricsService(MagicMock(), tracker, cumulative)
    async with aclosing(
        metrics.stream(SimpleNamespace(name="unknown", stream=stream), messages=[], model="m")
    ) as chunks:
        _ = [chunk async for chunk in chunks]
    assert len(tracker.requests) == 1
    assert tracker.requests[0].transport_attempts == 2
    assert cumulative["total_tokens"] == tracker.total_tokens == 13


@pytest.mark.parametrize("streaming", [False, True])
async def test_invocation_cancellation_is_preserved_and_settled(streaming):
    import asyncio
    from contextlib import aclosing

    entered = asyncio.Event()

    async def chat(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    async def stream(**kwargs):
        await chat(**kwargs)
        yield

    provider = SimpleNamespace(name="unknown", chat=chat, stream=stream)
    tracker = SessionCostTracker()
    service = AgentMetricsService(MagicMock(), tracker, {})

    async def consume():
        if streaming:
            async with aclosing(service.stream(provider, messages=[], model="m")) as chunks:
                _ = [chunk async for chunk in chunks]
        else:
            await service.complete(provider, messages=[], model="m")

    task = asyncio.create_task(consume())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(tracker.requests) == 1
    assert tracker.requests[0].outcome == "cancelled"
    assert tracker.requests[0].usage_status == "unknown"
    assert tracker.get_summary()["pending_requests"] == 0


@pytest.mark.parametrize("telemetry", ["opaque", {"prompt_tokens_details": "opaque"}])
async def test_optional_telemetry_cannot_turn_success_into_retryable_failure(telemetry, caplog):
    response = CompletionResponse(
        content="valid answer", role="assistant", raw_response={"usage": telemetry}
    )
    provider = SimpleNamespace(name="unknown", chat=AsyncMock(return_value=response))
    tracker = SessionCostTracker()
    metrics = AgentMetricsService(MagicMock(), tracker, {})
    assert await metrics.complete(provider, messages=[], model="m") is response
    assert tracker.requests[0].outcome == "success"
    assert tracker.requests[0].usage_status == "unknown"
    assert "malformed optional provider telemetry" in caplog.text


def _make_metrics_coordinator() -> AgentMetricsService:
    return AgentMetricsService(
        metrics_collector=MagicMock(),
        session_cost_tracker=MagicMock(),
        cumulative_token_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )


def test_emit_tool_strategy_event_uses_provider_capability_for_metrics_labels(caplog):
    coordinator = _make_metrics_coordinator()
    provider = MagicMock()
    provider.name = "openai"
    provider.supports_prompt_caching.return_value = True
    tools = [SimpleNamespace(name="read"), SimpleNamespace(name="git_diff")]

    with caplog.at_level("DEBUG"):
        coordinator.emit_tool_strategy_event(
            strategy="session_lock",
            tool_count=2,
            tool_tokens=512,
            context_window=32768,
            provider=provider,
            model="test-model",
            reason="cache_discount_or_large_context",
            tools=tools,
            v2_enabled=True,
        )

    assert "category=caching" in caplog.text
    assert 'victor_tool_strategy_v2_enabled{provider="openai"} 1' in caplog.text
    assert 'victor_tool_tier_count{provider="openai",tier="FULL"} 1' in caplog.text


def test_emit_tool_strategy_event_falls_back_to_provider_name_for_strings(caplog):
    coordinator = _make_metrics_coordinator()

    with caplog.at_level("INFO"):
        coordinator.emit_tool_strategy_event(
            strategy="semantic_selection",
            tool_count=1,
            tool_tokens=64,
            context_window=8192,
            provider="ollama",
            model="test-model",
            reason="small_context_window",
            tools=[SimpleNamespace(name="read")],
            v2_enabled=False,
        )

    assert "provider=ollama, category=local" in caplog.text


def test_emit_tool_strategy_event_records_last_event():
    coordinator = _make_metrics_coordinator()

    coordinator.emit_tool_strategy_event(
        strategy="semantic_selection",
        tool_count=1,
        tool_tokens=64,
        context_window=8192,
        provider="ollama",
        model="test-model",
        reason="small_context_window",
        tools=[SimpleNamespace(name="read")],
        v2_enabled=False,
    )

    assert coordinator.get_last_tool_strategy_event()["tool_tokens"] == 64


def test_task_report_captures_token_deltas_and_success_average():
    cumulative = {
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "total_tokens": 140,
        "cached_tokens": 10,
    }
    tracker = SessionCostTracker(provider="unknown", model="test-model")
    coordinator = AgentMetricsService(
        metrics_collector=MagicMock(),
        session_cost_tracker=tracker,
        cumulative_token_usage=cumulative,
    )

    coordinator.start_task_report(
        "Implement canonical runtime path",
        task_type="edit",
        metadata={"mode": "build"},
    )
    cumulative["prompt_tokens"] += 60
    cumulative["completion_tokens"] += 30
    cumulative["total_tokens"] += 90
    cumulative["cached_tokens"] += 20
    tracker.record_request(
        prompt_tokens=60,
        completion_tokens=30,
        cache_read_tokens=20,
        cache_write_tokens=5,
        duration_seconds=1.0,
    )

    report = coordinator.finish_task_report(
        True,
        compaction={
            "occurred": True,
            "saved_tokens": 45,
            "messages_removed": 3,
            "reason": "pre_tool_output",
            "policy_reason": "tool_output_exceeds_remaining_budget",
        },
        tool_schema_tokens=256,
    )

    assert report["task_type"] == "edit"
    assert report["api_total_tokens"] == 90
    assert report["api_prompt_tokens"] == 60
    assert report["api_completion_tokens"] == 30
    assert report["cache_read_tokens"] == 20
    assert report["cache_write_tokens"] == 5
    assert report["cache_hit_rate"] == pytest.approx(0.25)
    assert report["compaction_saved_tokens"] == 45
    assert report["compaction_messages_removed"] == 3
    assert report["tool_schema_tokens"] == 256
    assert report["tokens_per_successful_task"] == pytest.approx(90.0)
    assert report["metadata"]["compaction_reason"] == "pre_tool_output"
    assert report["metadata"]["compaction_policy_reason"] == "tool_output_exceeds_remaining_budget"
    assert coordinator.get_last_task_report()["task_id"] == report["task_id"]


@pytest.mark.parametrize("historical_stream", [False, True])
@pytest.mark.parametrize("mixed_window", [False, True])
async def test_buffered_task_reports_use_runtime_counters_across_task_boundaries(
    historical_stream, mixed_window
):
    cumulative = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    tracker = SessionCostTracker(provider="unknown", model="test-model")
    coordinator = AgentMetricsService(MagicMock(), tracker, cumulative)
    response = CompletionResponse(
        content="ok",
        role="assistant",
        usage={"prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26},
    )
    from contextlib import aclosing
    from victor.providers.base import StreamChunk

    provider = SimpleNamespace(name="unknown", chat=AsyncMock(return_value=response))

    async def stream(**kwargs):
        yield StreamChunk(
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, is_final=True
        )

    provider.stream = stream

    async def streamed_call():
        async with aclosing(
            coordinator.stream(provider, messages=[], model="test-model")
        ) as chunks:
            _ = [chunk async for chunk in chunks]

    if historical_stream:
        await streamed_call()
    # Usage before the task belongs to the session, not this task's delta.
    await coordinator.complete(provider, messages=[], model="test-model")
    for count, success in ((2, True), (1, False), (0, True)):
        coordinator.start_task_report("buffered task")
        for _ in range(count):
            await coordinator.complete(provider, messages=[], model="test-model")
        stream_count = int(mixed_window and count > 0)
        if stream_count:
            await streamed_call()
        report = coordinator.finish_task_report(success)
        assert report["api_prompt_tokens"] == 19 * count + 10 * stream_count
        assert report["api_completion_tokens"] == 7 * count + 5 * stream_count
        assert report["api_total_tokens"] == 26 * count + 15 * stream_count
        assert report["request_count"] == count + stream_count
    streamed_tokens = 15 * int(historical_stream) + 30 * int(mixed_window)
    assert coordinator.get_token_usage().total_tokens == 104 + streamed_tokens
    assert tracker.total_tokens == 104 + streamed_tokens


def test_task_report_promotes_workspace_policy_and_diagnostics():
    coordinator = AgentMetricsService(
        metrics_collector=MagicMock(),
        session_cost_tracker=SessionCostTracker(provider="unknown", model="test-model"),
        cumulative_token_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )

    coordinator.start_task_report("Delegate isolated work", metadata={"mode": "delegate"})

    report = coordinator.finish_task_report(
        False,
        metadata={
            "workspace_isolation_policy": {
                "mode": "delegate",
                "worktree_isolation": True,
                "materialize_worktrees": True,
                "dry_run_worktrees": False,
                "auto_merge_worktrees": False,
                "cleanup_worktrees": False,
            },
            "workspace_isolation_diagnostics": [
                {
                    "operation": "materialize",
                    "reason": "branch_exists",
                    "message": "branch already exists",
                    "details": {"member_id": "tester"},
                }
            ],
        },
    )

    metadata = report["metadata"]
    assert metadata["workspace_policy_mode"] == "delegate"
    assert metadata["workspace_policy_materialize_worktrees"] is True
    assert metadata["workspace_policy_cleanup_worktrees"] is False
    assert metadata["workspace_isolation_diagnostic_count"] == 1
    assert metadata["workspace_isolation_diagnostic_reasons"] == {"branch_exists": 1}
    assert metadata["workspace_isolation_diagnostic_operations"] == {"materialize": 1}


def test_task_report_history_respects_limit():
    tracker = SessionCostTracker(provider="unknown", model="test-model")
    coordinator = AgentMetricsService(
        metrics_collector=MagicMock(),
        session_cost_tracker=tracker,
        cumulative_token_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )

    for index in range(3):
        coordinator.start_task_report(f"task-{index}")
        coordinator.finish_task_report(True)

    history = coordinator.get_task_report_history(limit=2)

    assert len(history) == 2
    assert history[0]["description"] == "task-1"
    assert history[1]["description"] == "task-2"
