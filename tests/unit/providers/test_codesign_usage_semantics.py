"""Co-design fields must survive transport, accumulation, and pricing."""

from unittest.mock import Mock

import pytest

from victor.agent.stream_handler import StreamMetrics
from victor.providers.base import CompletionResponse, StreamChunk
from victor.providers.sandhi_transport import _latency_fields
from victor.providers.usage_parsing import usage_dict_from_neutral


@pytest.mark.parametrize("reasoning", [0, 25, 40, 90])
@pytest.mark.parametrize("included", [False, True])
@pytest.mark.parametrize("response_type", [CompletionResponse, StreamChunk])
def test_reasoning_inclusion_survives_to_cost(reasoning, included, response_type):
    usage = usage_dict_from_neutral(
        {
            "tokens_in": 100,
            "tokens_out": 40,
            "reasoning_tokens": reasoning,
            "reasoning_included": included,
        },
        None,
    )
    assert usage["reasoning_included"] is included
    # Pydantic's former Dict[str, int] silently coerced False/True to 0/1.
    response = response_type(content="", model="m", usage=usage)
    assert response.usage["reasoning_included"] is included
    metrics = StreamMetrics()
    metrics.record_usage(response.usage)
    pricing = Mock(cost_enabled=True)
    pricing.calculate_cost.return_value = dict(
        input_cost=0, output_cost=0, cache_cost=0, total_cost=0
    )
    metrics.calculate_cost(pricing)
    assert pricing.calculate_cost.call_args.args[1] == 40 + (0 if included else reasoning)


def test_reasoning_is_folded_per_call_before_accumulation():
    metrics = StreamMetrics()
    metrics.record_usage(
        {"completion_tokens": 40, "reasoning_tokens": 25, "reasoning_included": False}
    )
    metrics.record_usage(
        {"completion_tokens": 100, "reasoning_tokens": 90, "reasoning_included": True}
    )
    pricing = Mock(cost_enabled=True)
    pricing.calculate_cost.return_value = dict(
        input_cost=0, output_cost=0, cache_cost=0, total_cost=0
    )
    metrics.calculate_cost(pricing)
    assert pricing.calculate_cost.call_args.args[1] == 165


def test_latency_provenance_survives_transport_and_metrics():
    fields = _latency_fields(
        {
            "duration_ms": 120,
            "duration_source": "origin",
            "time_to_first_token_ms": 45,
            "time_to_first_token_source": "boundary",
        }
    )
    metrics = StreamMetrics()
    metrics.record_wire_latency(fields)
    assert metrics.wire_duration_ms == 120
    assert metrics.wire_ttft_ms == 45
    assert metrics.metadata["duration_source"] == "origin"
    assert metrics.metadata["time_to_first_token_source"] == "boundary"


@pytest.mark.parametrize("value", [True, float("inf"), float("nan"), -1, "bad"])
def test_invalid_latency_does_not_gain_origin_provenance(value):
    assert _latency_fields({"duration_ms": value, "duration_source": "origin"}) == {}
    metrics = StreamMetrics()
    metrics.record_wire_latency({"duration_ms": value, "duration_source": "origin"})
    assert metrics.wire_duration_ms is None
    assert "duration_source" not in metrics.metadata


def test_new_latency_without_source_does_not_inherit_previous_provenance():
    metrics = StreamMetrics()
    metrics.record_wire_latency({"duration_ms": 100, "duration_source": "origin"})
    metrics.record_wire_latency({"duration_ms": 200})
    assert metrics.wire_duration_ms == 200
    assert "duration_source" not in metrics.metadata
