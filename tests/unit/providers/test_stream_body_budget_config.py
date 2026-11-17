import json

import pytest
from pydantic import ValidationError

from victor.config.settings import ProviderConfig
from victor.providers import sandhi_transport as st
from victor.providers.sandhi_transport import SandhiTypedProviderMixin


def test_stream_body_budget_config_survives_profile_serialization():
    value = ProviderConfig(stream_body_timeout_secs=600).model_dump(exclude_none=True)
    assert value["stream_body_timeout_secs"] == 600


@pytest.mark.parametrize("value", [0, 0.0005, -1, float("nan"), float("inf"), 841, True])
def test_invalid_stream_body_budget_config_is_rejected(value):
    with pytest.raises(ValidationError):
        ProviderConfig(stream_body_timeout_secs=value)


def test_stream_body_budget_is_omitted_and_header_is_not_sent_by_default():
    provider = SandhiTypedProviderMixin()
    provider._gateway_overrides = lambda: None
    assert provider._sandhi_stream_body_timeout_secs() is None
    assert "x-sandhi-stream-body-budget-ms" not in provider._sandhi_stream_wire_headers()


def test_stream_body_budget_becomes_one_integer_gateway_header(monkeypatch):
    provider = SandhiTypedProviderMixin()
    provider._gateway_overrides = lambda: {"url": "https://gateway.test"}
    monkeypatch.setattr(st, "_wire_call_headers", lambda: {"x-sandhi-step-id": "step-1"})
    provider.extra_config = {"stream_body_timeout_secs": 600}
    headers = provider._sandhi_stream_wire_headers()
    assert headers["x-sandhi-step-id"] == "step-1"
    assert headers["x-sandhi-stream-body-budget-ms"] == "600000"
    assert json.dumps(headers).count("x-sandhi-stream-body-budget-ms") == 1


def test_invalid_stream_body_budget_runtime_is_rejected():
    provider = SandhiTypedProviderMixin()
    provider.extra_config = {"stream_body_timeout_secs": True}
    with pytest.raises(ValueError, match="stream body timeout"):
        provider._sandhi_stream_body_timeout_secs()


@pytest.mark.asyncio
async def test_stream_body_budget_reaches_gateway_stream_wire_request(monkeypatch):
    provider = SandhiTypedProviderMixin()
    provider.extra_config = {"stream_body_timeout_secs": 600}
    provider._validate_gateway_request = lambda request: None
    provider._typed_provider = lambda model: object()
    provider._gateway_overrides = lambda: {"url": "https://gateway.test"}
    monkeypatch.setattr(st, "_wire_call_headers", lambda: {"x-sandhi-step-id": "step-1"})
    provider._sandhi_timeout = lambda: 1.0
    provider._sandhi_stream_idle_timeout = lambda: 1.0
    captured = []

    async def stream_wrapper(_provider, _request, wire_headers, _idle_timeout):
        captured.append(wire_headers)
        yield '{"event":"finish","reason":"stop"}'

    provider._sandhi_stream_with_idle_timeout = stream_wrapper
    chunks = [chunk async for chunk in provider._sandhi_stream({"model": "test-model"})]

    assert len(chunks) == 1
    assert captured == [
        json.dumps(
            {
                "x-sandhi-step-id": "step-1",
                "x-sandhi-stream-body-budget-ms": "600000",
            }
        )
    ]


@pytest.mark.asyncio
async def test_stream_body_budget_is_not_sent_on_buffered_complete_request(monkeypatch):
    provider = SandhiTypedProviderMixin()
    provider.extra_config = {"stream_body_timeout_secs": 600}
    provider._validate_gateway_request = lambda request: None
    provider._gateway_overrides = lambda: {"url": "https://gateway.test"}
    monkeypatch.setattr(st, "_wire_call_headers", lambda: {"x-sandhi-step-id": "step-1"})
    provider._sandhi_timeout = lambda: 1.0
    captured = []

    class CompleteProvider:
        @staticmethod
        async def complete_json(_request, wire_headers):
            captured.append(wire_headers)
            return '{"ok":true}'

    provider._typed_provider = lambda model: CompleteProvider()
    result = await provider._sandhi_complete({"model": "test-model"})

    assert result == {"ok": True}
    assert captured == [json.dumps({"x-sandhi-step-id": "step-1"})]
