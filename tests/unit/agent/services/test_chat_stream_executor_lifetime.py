"""Streaming ownership at the real service/report boundary and async cleanup."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.chat_service import ChatService, ChatServiceConfig
from victor.agent.services.chat_stream_runtime import ServiceStreamingRuntime
from victor.agent.services.chat_turn_lifecycle import ChatTurnLifecycle
from victor.agent.services.metrics_service import AgentMetricsService
from victor.agent.services.streaming_act_adapter import StreamingActAdapter, StreamActSession
from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.session_state_manager import SessionStateManager
from victor.providers.base import StreamChunk


def make_services(executor_type):
    owner = MagicMock()
    owner._session_accessor = SessionStateAccessor(SessionStateManager())
    owner.has_capability.return_value = False
    owner.get_capability_value.return_value = None
    owner._cumulative_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    owner.messages = []
    events = []
    owner._metrics_coordinator = SimpleNamespace(
        finalize_stream_metrics=lambda usage, **_: events.append(("metrics", usage["total_tokens"]))
    )
    metrics = AgentMetricsService(None, None, owner._cumulative_token_usage)
    runtime = ServiceStreamingRuntime(owner)
    runtime._streaming_executor = executor_type(owner, events)
    service = ChatService(
        ChatServiceConfig(),
        None,
        SimpleNamespace(start_new_turn=lambda: events.append(("reset", None))),
        None,
        None,
        None,
        None,
    )

    def finish(success, **kwargs):
        events.append(("report", kwargs["user_message"]))
        return metrics.finish_task_report(success)

    service.bind_runtime_components(
        stream_turn_lock=runtime.services.stream_turn_lock,
        stream_chat_handler=runtime.stream_chat_under_turn_lock,
        task_report_start_handler=lambda message, **_: metrics.start_task_report(message),
        task_report_finish_handler=finish,
        turn_lifecycle=ChatTurnLifecycle(
            setup=lambda message, **_: events.append(("setup", message)),
            teardown=lambda message, **_: events.append(("teardown", message)),
        ),
    )
    return service, runtime, metrics, events


def set_usage(owner, message):
    prompt, completion = (2, 3) if message == "first" else (7, 11)
    owner._current_stream_context = SimpleNamespace(
        cumulative_usage={
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
        runtime_override_snapshot=None,
    )


async def consume(service, message):
    return [chunk async for chunk in service.stream_chat(message)]


@pytest.mark.asyncio
@pytest.mark.parametrize("aggregate_second", [False, True])
async def test_public_service_owns_preparation_reports_and_teardown(aggregate_second):
    started, release = asyncio.Event(), asyncio.Event()

    class Executor:
        def __init__(self, owner, events):
            self.owner = owner

        async def run_unified(self, message, **_):
            set_usage(self.owner, message)
            if message == "first":
                started.set()
                await release.wait()
            yield StreamChunk(content=message, is_final=True)

    service, runtime, metrics, events = make_services(Executor)
    first = asyncio.create_task(consume(service, "first"))
    await asyncio.wait_for(started.wait(), 2)
    second = asyncio.create_task(
        service.chat("second", stream=True) if aggregate_second else consume(service, "second")
    )
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert events == [("reset", None), ("setup", "first")]
        assert metrics._active_task_report.description == "first"
        assert runtime.services.stream_turn_lock.locked()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 2)
    reports = metrics.get_task_report_history()
    assert [(r["description"], r["api_total_tokens"]) for r in reports] == [
        ("first", 5),
        ("second", 18),
    ]
    assert events.index(("teardown", "first")) < events.index(("setup", "second"))
    assert not runtime.services.stream_turn_lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("public", ["runtime", "service", "orchestrator"])
@pytest.mark.parametrize("termination", ["close", "cancel"])
async def test_inner_cleanup_finishes_before_reports_unlock_and_next_turn(public, termination):
    cleanup_started, release_cleanup = asyncio.Event(), asyncio.Event()
    waiting = asyncio.Event()

    class Executor:
        def __init__(self, owner, events):
            self.owner, self.events = owner, events

        async def run_unified(self, message, **_):
            set_usage(self.owner, message)
            self.events.append(("execute", message))
            try:
                yield StreamChunk(content=message)
                if message == "first":
                    waiting.set()
                    await asyncio.Event().wait()
            finally:
                self.events.append(("cleanup-start", message))
                if message == "first":
                    cleanup_started.set()
                    await release_cleanup.wait()
                self.events.append(("cleanup-end", message))

    service, runtime, metrics, events = make_services(Executor)
    entry = runtime if public == "runtime" else service
    if public == "orchestrator":
        from victor.agent.orchestrator import AgentOrchestrator

        entry = object.__new__(AgentOrchestrator)
        entry._chat_service = service
        entry.active_session_id = None
    first = entry.stream_chat("first")
    await asyncio.wait_for(anext(first), 2)
    if termination == "close":
        closing = asyncio.create_task(first.aclose())
    else:
        closing = asyncio.create_task(anext(first))
        await asyncio.wait_for(waiting.wait(), 2)
        closing.cancel()
    await asyncio.wait_for(cleanup_started.wait(), 2)
    second = asyncio.create_task(consume(entry, "second"))
    try:
        await asyncio.sleep(0)
        assert not closing.done()
        assert runtime.services.stream_turn_lock.locked()
        assert ("execute", "second") not in events
        assert ("metrics", 5) not in events
        assert ("report", "first") not in events
    finally:
        release_cleanup.set()
        if termination == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(closing, 2)
        else:
            await asyncio.wait_for(closing, 2)
        await asyncio.wait_for(second, 2)
    assert events.index(("cleanup-end", "first")) < events.index(("metrics", 5))
    assert events.index(("metrics", 5)) < events.index(("execute", "second"))
    if public != "runtime":
        assert events.index(("metrics", 5)) < events.index(("report", "first"))
        assert events.index(("teardown", "first")) < events.index(("setup", "second"))
        assert [r["api_total_tokens"] for r in metrics.get_task_report_history()] == [5, 18]
    assert not runtime.services.stream_turn_lock.locked()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_prepare_or_release_active_turn():
    entered, release = asyncio.Event(), asyncio.Event()

    class Executor:
        def __init__(self, owner, events):
            self.owner = owner

        async def run_unified(self, message, **_):
            set_usage(self.owner, message)
            if message == "first":
                entered.set()
                await release.wait()
            yield StreamChunk(content=message, is_final=True)

    service, runtime, metrics, events = make_services(Executor)
    first = asyncio.create_task(consume(service, "first"))
    await asyncio.wait_for(entered.wait(), 2)
    waiter = asyncio.create_task(consume(service, "cancelled"))
    await asyncio.sleep(0)
    waiter.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, 2)
        assert events == [("reset", None), ("setup", "first")]
        assert runtime.services.stream_turn_lock.locked()
    finally:
        release.set()
        await asyncio.wait_for(first, 2)
    await asyncio.wait_for(consume(service, "second"), 2)
    assert [r["description"] for r in metrics.get_task_report_history()] == ["first", "second"]


@pytest.mark.asyncio
async def test_act_adapter_close_awaits_executor_act_cleanup():
    started, release = asyncio.Event(), asyncio.Event()

    class Executor:
        async def execute_turn_streaming(self, *args, **kwargs):
            try:
                yield StreamChunk(content="partial")
            finally:
                started.set()
                await release.wait()

    adapter = StreamingActAdapter(
        Executor(), StreamActSession(None, None, SimpleNamespace(), None, None, None)
    )
    stream = adapter.stream_turn_act(
        query="q", state={}, perception=None, plan=None, turn_index=1, outcome=SimpleNamespace()
    )
    await anext(stream)
    closing = asyncio.create_task(stream.aclose())
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert not closing.done()
    finally:
        release.set()
        await asyncio.wait_for(closing, 2)


@pytest.mark.asyncio
async def test_unified_executor_close_awaits_loop_cleanup(monkeypatch):
    from victor.agent.services.chat_stream_executor import StreamingChatExecutor

    started, release = asyncio.Event(), asyncio.Event()

    class Loop:
        async def run_streaming(self, *args, **kwargs):
            try:
                yield StreamChunk(content="partial")
            finally:
                started.set()
                await release.wait()

    owner = SimpleNamespace(_message_policy_gate=None, settings=None, turn_executor=None)
    executor = StreamingChatExecutor(SimpleNamespace(_orchestrator=owner))
    monkeypatch.setattr(executor, "_get_conversation_history", lambda *args: [])
    monkeypatch.setattr(
        StreamingActAdapter,
        "prepare",
        AsyncMock(
            return_value=SimpleNamespace(session=SimpleNamespace(stream_ctx=SimpleNamespace()))
        ),
    )
    monkeypatch.setattr("victor.framework.agentic_loop.AgenticLoop", lambda **kwargs: Loop())
    monkeypatch.setattr(
        "victor.agent.services.judge_calibration_gate.resolve_completion_strategy",
        lambda *args: "enhanced",
    )
    monkeypatch.setattr("victor.framework.effect_gate.resolve_effect_gate_enabled", lambda _: False)
    monkeypatch.setattr(
        "victor.framework.per_turn_auditor.resolve_per_turn_auditor_enabled", lambda _: False
    )
    stream = executor.run_unified("q")
    await anext(stream)
    closing = asyncio.create_task(stream.aclose())
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert not closing.done()
    finally:
        release.set()
        await asyncio.wait_for(closing, 2)
