"""Typed disjoint units and source-rate authority at the actual metrics owner."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from victor.agent.services.metrics_service import AgentMetricsService
from victor.agent.session_cost_tracker import SessionCostTracker
from victor.providers.base import CompletionResponse
from victor.ui.rendering.markdown_presenters import turn_cost_footer


@pytest.mark.parametrize("gateway", [False, True])
@pytest.mark.parametrize("complete", [False, True])
async def test_metrics_prices_direct_fresh_once_and_refuses_gateway_alias_rate_authority(
    gateway, complete
):
    tracker = SessionCostTracker(provider="openai", model="server-alias")
    rates = SimpleNamespace(
        cost_enabled=True,
        calculate_cost=Mock(
            return_value={"input_cost": 2.0, "output_cost": 12.0, "cache_cost": 3.6}
        ),
    )
    tracker._capabilities = rates
    tracker._sandhi = Mock()
    units = {
        "tokens_in": 2,
        "tokens_out": 4,
        "cache_read_tokens": 6,
        "cache_creation_tokens": 3,
        "reasoning_tokens": 2,
        "reasoning_included": False,
        "completeness": "final" if complete else "partial",
    }
    # Compatibility prompt includes read and excludes write. Typed units own pricing.
    response = CompletionResponse(
        content="reply",
        role="assistant",
        usage={
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "cache_read_input_tokens": 6,
            "cache_creation_input_tokens": 3,
            "reasoning_tokens": 2,
            "reasoning_included": False,
        },
        metadata={
            "sandhi_usage": {
                "typed_usage": units,
                "usage_basis": "gateway_client_projection" if gateway else "direct_provider",
                "completeness": units["completeness"],
            }
        },
    )
    provider = SimpleNamespace(name="openai", chat=AsyncMock(return_value=response))
    metrics = AgentMetricsService(Mock(), tracker, {})
    metrics.start_task_report("typed accounting")
    await metrics.complete(provider, messages=[], model="server-alias")
    report = metrics.finish_task_report(success=True)
    request = tracker.requests[0]
    assert request.observed_units == units
    assert request.prompt_tokens == 8
    assert request.cache_read_tokens == 6
    assert request.cache_write_tokens == 3
    assert request.billable_completion_tokens == 6
    assert request.total_tokens == 17  # 2 fresh + 6 read + 3 write + 6 output
    assert report["api_total_tokens"] == 17
    if gateway:
        rates.calculate_cost.assert_not_called()
        tracker._sandhi.record.assert_not_called()  # The server owns the receipt.
        assert request.pricing_status == "unavailable"
        assert request.to_dict()["cost"] is None
        assert tracker.get_summary()["cost"]["total"] is None
        assert report["total_cost_usd"] is None
        assert report["session_cost_usd"] is None
        assert "cost unavailable" in turn_cost_footer(report)
        assert tracker.format_inline_cost(request) == "cost n/a"
        assert "Cost unavailable" in tracker.get_formatted_summary()
    else:
        rates.calculate_cost.assert_called_once_with(2, 6, 6, 3)
        assert request.pricing_status == ("estimated" if complete else "partial_estimate")
        assert request.total_cost == 17.6
        assert report["total_cost_usd"] == 17.6
        if not complete:
            tracker._sandhi.record.assert_not_called()


def test_mixed_known_estimate_and_unknown_gateway_price_export_qualified_subtotal(tmp_path):
    tracker = SessionCostTracker()
    tracker._capabilities = SimpleNamespace(
        cost_enabled=True,
        calculate_cost=lambda *_args: {"input_cost": 1.0, "output_cost": 2.0, "cache_cost": 3.0},
    )
    tracker.record_request(prompt_tokens=2, completion_tokens=1)
    request = tracker.begin_request()
    tracker.settle_request(
        request,
        outcome="success",
        usage={"prompt_tokens": 8, "completion_tokens": 4},
        typed_usage={
            "tokens_in": 2,
            "tokens_out": 4,
            "cache_read_tokens": 6,
            "cache_creation_tokens": 0,
        },
        usage_basis="gateway_client_projection",
    )
    summary = tracker.get_summary()
    assert summary["cost"]["total"] is None
    assert summary["cost"]["known_estimate"] == 6.0
    assert summary["pricing_coverage"]["unavailable"] == 1
    assert tracker.format_inline_cost() == "cost n/a"
    output = tmp_path / "costs.csv"
    tracker.export_csv(output)
    import csv

    with output.open() as source:
        rows = list(csv.DictReader(source))
    assert rows[-1]["total_cost"] == ""
    assert rows[-1]["pricing_status"] == "unavailable"
    assert request.observed_units["cache_creation_tokens"] == 0
    assert "reasoning_tokens" not in request.observed_units


def test_unknown_usage_cannot_export_a_priced_zero():
    tracker = SessionCostTracker(provider="openai", model="alias")
    request = tracker.begin_request()
    tracker.settle_request(request, outcome="error", usage=None)
    assert request.to_dict()["cost"] is None
    assert request.observed_units is None
    assert tracker.get_summary()["cost"]["total"] is None
    assert tracker.get_summary()["pricing_coverage"]["unavailable"] == 1


def test_stream_metrics_never_prices_gateway_alias_and_uses_direct_fresh_units():
    from victor.agent.stream_handler import StreamMetrics
    from victor.agent.streaming.context import StreamingChatContext

    capabilities = SimpleNamespace(
        cost_enabled=True,
        calculate_cost=Mock(
            return_value={
                "input_cost": 2.0,
                "output_cost": 12.0,
                "cache_cost": 3.6,
                "total_cost": 17.6,
            }
        ),
    )
    usage = {
        "prompt_tokens": 8,
        "completion_tokens": 4,
        "cache_read_input_tokens": 6,
        "cache_creation_input_tokens": 3,
        "reasoning_tokens": 2,
        "reasoning_included": False,
    }
    typed = {
        "tokens_in": 2,
        "tokens_out": 4,
        "cache_read_tokens": 6,
        "cache_creation_tokens": 3,
        "completeness": "final",
    }
    direct = StreamMetrics()
    direct.record_usage(usage, typed_usage=typed)
    assert direct.total_tokens == 17
    direct.calculate_cost(capabilities)
    capabilities.calculate_cost.assert_called_once_with(2, 6, 6, 3)
    capabilities.calculate_cost.reset_mock()
    context = StreamingChatContext(user_message="fixture")
    context.record_provider_diagnostics(
        {"typed_usage": typed, "usage_basis": "gateway_client_projection"}
    )
    assert context.provider_diagnostics["typed_usage_totals"]["tokens_in"] == 2
    gateway = StreamMetrics()
    gateway.record_usage(
        usage,
        typed_usage=context.provider_diagnostics["typed_usage_totals"],
        usage_basis=context.provider_diagnostics["usage_basis"],
    )
    gateway.calculate_cost(capabilities)
    capabilities.calculate_cost.assert_not_called()
    assert gateway.format_cost() == "cost n/a"
    assert not gateway.cost_calculated


@pytest.mark.parametrize("typed_first", [False, True])
async def test_mixed_usage_keeps_per_call_fresh_counts_at_stream_metrics_owner(
    monkeypatch, typed_first
):
    from victor.agent.metrics_collector import MetricsCollector, MetricsCollectorConfig
    from victor.agent.streaming.context import StreamingChatContext

    units = {
        "tokens_in": 2,
        "tokens_out": 4,
        "cache_read_tokens": 6,
        "cache_creation_tokens": 3,
        "completeness": "final",
    }
    typed = CompletionResponse(
        content="typed",
        role="assistant",
        usage={
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "reasoning_tokens": 2,
            "reasoning_included": False,
            "cache_read_input_tokens": 6,
            "cache_creation_input_tokens": 3,
        },
        metadata={"sandhi_usage": {"typed_usage": units}},
    )
    native = CompletionResponse(
        content="native", role="assistant", usage={"prompt_tokens": 10, "completion_tokens": 5}
    )
    rates = SimpleNamespace(
        cost_enabled=True,
        calculate_cost=Mock(
            return_value={"input_cost": 0, "output_cost": 0, "cache_cost": 0, "total_cost": 0}
        ),
    )
    monkeypatch.setattr(
        "victor.config.metrics_capabilities.get_metrics_capabilities", lambda *_: rates
    )
    tracker = SessionCostTracker()
    totals = {}
    service = AgentMetricsService(Mock(), tracker, totals)
    context = StreamingChatContext(user_message="mixed usage")
    context.record_provider_diagnostics(typed.metadata["sandhi_usage"])
    provider = SimpleNamespace(
        name="openai",
        chat=AsyncMock(side_effect=[typed, native] if typed_first else [native, typed]),
    )
    for _ in range(2):
        await service.complete(provider, messages=[], model="fixture")
    rates.calculate_cost.reset_mock()
    collector = MetricsCollector(MetricsCollectorConfig(provider="openai"), Mock())
    collector.init_stream_metrics()
    result = collector.finalize_stream_metrics(totals, context.provider_diagnostics)
    assert totals["fresh_input_tokens"] == 12
    assert result.total_tokens == 32
    assert result.prompt_tokens == 18
    rates.calculate_cost.assert_called_once_with(12, 11, 6, 3)


@pytest.mark.parametrize("usage", [None, {}])
def test_gateway_missing_usage_never_becomes_priced_zero(monkeypatch, usage):
    from victor.agent.metrics_collector import MetricsCollector, MetricsCollectorConfig

    rates = SimpleNamespace(cost_enabled=True, calculate_cost=Mock())
    monkeypatch.setattr(
        "victor.config.metrics_capabilities.get_metrics_capabilities", lambda *_: rates
    )
    usage_logger = Mock()
    collector = MetricsCollector(MetricsCollectorConfig(provider="openai"), usage_logger)
    collector.init_stream_metrics()
    result = collector.finalize_stream_metrics(usage, {"usage_basis": "gateway_client_projection"})
    rates.calculate_cost.assert_not_called()
    assert not result.cost_calculated
    assert result.format_cost() == "cost n/a"
    assert "total_cost" not in usage_logger.log_event.call_args.args[1]


@pytest.mark.parametrize("typed_first", [False, True])
def test_stream_metrics_per_call_mixed_usage_keeps_native_fresh_input(typed_first):
    from victor.agent.stream_handler import StreamMetrics

    metrics = StreamMetrics()
    calls = [
        ({"prompt_tokens": 8, "completion_tokens": 4}, {"tokens_in": 2, "tokens_out": 4}),
        ({"prompt_tokens": 10, "completion_tokens": 5}, None),
    ]
    for usage, typed in calls if typed_first else reversed(calls):
        metrics.record_usage(usage, typed_usage=typed)
    assert metrics.fresh_input_tokens == 12
