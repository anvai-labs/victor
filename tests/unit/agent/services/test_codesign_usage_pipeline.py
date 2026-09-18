"""Reasoning accounting across the production stream consumer and finalizer."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.metrics_collector import MetricsCollector, MetricsCollectorConfig
from victor.agent.services.chat_delivery import ChatDelivery
from victor.agent.services.chat_runtime_services import (
    ChatRuntimeServices,
    SessionTaskRequirementState,
)
from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin
from victor.agent.services.chat_stream_runtime import ServiceStreamingRuntime
from victor.agent.services.metrics_service import AgentMetricsService
from victor.agent.session_cost_tracker import SessionCostTracker
from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.session_state_manager import SessionStateManager
from victor.agent.streaming.context import StreamingChatContext
from victor.config.metrics_capabilities import ProviderMetricsCapabilities
from victor.providers.base import StreamChunk
from victor.providers.usage_parsing import usage_dict_from_neutral


class Helper(ChatStreamHelperMixin):
    def __init__(self, orchestrator):
        self._orchestrator = orchestrator
        self.services = SimpleNamespace(delivery=ChatDelivery(sanitizer=orchestrator.sanitizer))


@pytest.mark.parametrize(
    "calls,expected_output",
    [
        ([(40, 25, False)], 65),
        ([(40, 40, False)], 80),
        ([(40, 90, True)], 40),
        ([(40, 25, False), (100, 90, True)], 165),
        ([(5, 12, None), (100, 5, None)], 117),
        ([(0, 25, False)], 25),
    ],
)
async def test_terminal_usage_reaches_session_cost_and_canonical_record(
    monkeypatch, calls, expected_output
):
    logger = MagicMock()
    collector = MetricsCollector(
        config=MetricsCollectorConfig(provider="test", model="m"), usage_logger=logger
    )
    capabilities = ProviderMetricsCapabilities(
        provider="test",
        model="m",
        cost_enabled=True,
        input_cost_per_mtok=1.0,
        output_cost_per_mtok=10.0,
    )
    monkeypatch.setattr(
        "victor.config.metrics_capabilities.get_metrics_capabilities", lambda *_args: capabilities
    )
    tracker = SessionCostTracker(_capabilities=capabilities)
    cumulative = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    service = AgentMetricsService(
        metrics_collector=collector, session_cost_tracker=tracker, cumulative_token_usage=cumulative
    )
    service.start_task_report("streamed task")
    ctx = StreamingChatContext(user_message="x", total_iterations=1)
    ctx.stream_metrics = collector.init_stream_metrics()
    orch = SimpleNamespace(
        model="m",
        temperature=0.7,
        max_tokens=100,
        settings=SimpleNamespace(),
        get_assembled_messages=lambda **_kwargs: [],
        sanitizer=SimpleNamespace(is_garbage_content=lambda _c: False, sanitize=lambda c: c),
        _metrics_collector=collector,
        _metrics_coordinator=service,
        _cumulative_token_usage=cumulative,
    )
    helper = Helper(orch)
    for completion, reasoning, included in calls:
        neutral = {"tokens_in": 10, "tokens_out": completion, "reasoning_tokens": reasoning}
        if included is not None:
            neutral["reasoning_included"] = included
        usage = usage_dict_from_neutral(neutral, None)

        async def stream(**_kwargs):
            yield StreamChunk(
                content="",
                usage=usage,
                metadata={
                    "sandhi_usage": {
                        "duration_ms": 120,
                        "duration_source": "origin",
                        "time_to_first_token_ms": 45,
                        "time_to_first_token_source": "boundary",
                        "attempts": 1,
                        "completeness": "complete",
                    }
                },
            )

        orch.provider = SimpleNamespace(stream=stream)
        await helper._stream_provider_response_inner({}, {}, ctx)

    raw_completion = sum(call[0] for call in calls)
    expected_prompt = 10 * len(calls)
    assert ctx.cumulative_usage["completion_tokens"] == raw_completion
    assert ctx.cumulative_usage["billable_completion_tokens"] == expected_output
    assert ctx.cumulative_usage["total_tokens"] == expected_prompt + expected_output

    runtime = ServiceStreamingRuntime(
        orch,
        services=ChatRuntimeServices(
            SessionTaskRequirementState(SessionStateAccessor(SessionStateManager()))
        ),
    )
    bindings = SimpleNamespace(
        state_host=orch,
        state_dict={},
        get_capability_value=lambda name, default=None: (
            ctx if name == "current_stream_context" else default
        ),
    )
    monkeypatch.setattr(runtime, "_get_runtime_bindings", lambda *a, **k: bindings)

    class Executor:
        async def run_unified(self, _message, **_kwargs):
            if False:
                yield None

    monkeypatch.setattr(runtime, "get_executor", lambda: Executor())
    async for _ in runtime.stream_chat("x"):
        pass

    expected_cost = (expected_prompt + expected_output * 10) / 1_000_000
    metrics = collector.get_last_stream_metrics()
    assert metrics.completion_tokens == raw_completion
    assert metrics.billable_completion_tokens == expected_output
    assert metrics.total_cost == pytest.approx(expected_cost)
    assert metrics.effective_total_tokens == expected_prompt + expected_output
    assert len(tracker.requests) == 1
    assert tracker.requests[0].completion_tokens == raw_completion
    assert tracker.requests[0].billable_completion_tokens == expected_output
    assert tracker.total_cost == pytest.approx(expected_cost)
    assert tracker.total_tokens == expected_prompt + expected_output
    assert orch._cumulative_token_usage["total_tokens"] == expected_prompt + expected_output
    report = service.finish_task_report(True)
    assert report["api_prompt_tokens"] == expected_prompt
    assert report["api_completion_tokens"] == raw_completion
    assert report["api_total_tokens"] == expected_prompt + expected_output
    assert report["total_cost_usd"] == pytest.approx(expected_cost)
    assert report["request_count"] == 1
    record = [
        call.args[1]
        for call in logger.log_event.call_args_list
        if call.args[0] == "stream_completed"
    ][-1]
    assert record["completion_tokens"] == raw_completion
    assert record["billable_completion_tokens"] == expected_output
    assert record["total_tokens"] == expected_prompt + expected_output
    assert record["duration_source"] == "origin"
    assert record["time_to_first_token_source"] == "boundary"
    assert record["sandhi_usage"]["provider_calls"] == len(calls)


def test_explicit_provider_total_is_preserved_through_finalization():
    usage = usage_dict_from_neutral(
        {"tokens_in": 10, "tokens_out": 40, "reasoning_tokens": 25, "reasoning_included": False},
        {"total_tokens": 90},
    )
    assert usage["total_tokens"] == 90
    collector = MetricsCollector(
        config=MetricsCollectorConfig(provider="test", model="m"), usage_logger=MagicMock()
    )
    tracker = SessionCostTracker()
    service = AgentMetricsService(
        metrics_collector=collector, session_cost_tracker=tracker, cumulative_token_usage={}
    )
    collector.init_stream_metrics()
    assert service.finalize_stream_metrics(usage).total_tokens == 90
    assert tracker.total_tokens == 90
