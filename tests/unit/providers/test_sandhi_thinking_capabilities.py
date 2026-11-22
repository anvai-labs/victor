"""Exact Sandhi model capability admission for reasoning effort."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import victor.providers.sandhi_transport as st
from victor.providers.sandhi_transport import SandhiOpenAIProvider


class FakeGateway:
    def __init__(self, descriptor=None):
        self.descriptor = descriptor

    def provider_descriptor_json(self, slug: str) -> str:
        if self.descriptor is None:
            raise RuntimeError("unknown provider")
        return json.dumps(self.descriptor)


def provider() -> SandhiOpenAIProvider:
    return SandhiOpenAIProvider.__new__(SandhiOpenAIProvider)


def static_descriptor(supported: bool) -> dict:
    thinking = (
        {"supported": True, "effort": {"values": ["low", "medium"]}}
        if supported
        else {"supported": False}
    )
    return {
        "schema_version": "1",
        "slug": "openai",
        "models": [
            {"id": "gpt-5", "thinking": thinking},
            {"id": "gpt-5-chat-latest", "thinking": {"supported": False}},
        ],
    }


def test_support_comes_from_exact_static_descriptor(monkeypatch):
    monkeypatch.setattr(st, "_sg", FakeGateway(static_descriptor(True)))
    model = provider()

    assert model.supports_reasoning_effort("gpt-5") is True
    assert model.supports_reasoning_effort("gpt-5-chat-latest") is False
    assert model.supports_reasoning_effort("gpt-5-similar") is False
    assert model.supports_reasoning_effort(None) is False


def test_missing_or_unsupported_static_metadata_fails_closed(monkeypatch):
    monkeypatch.setattr(st, "_sg", FakeGateway(static_descriptor(False)))
    assert provider().supports_reasoning_effort("gpt-5") is False

    monkeypatch.setattr(st, "_sg", None)
    assert provider().supports_reasoning_effort("gpt-5") is False


def test_malformed_static_catalog_fails_closed(monkeypatch):
    descriptor = static_descriptor(True)
    descriptor["models"][0]["thinking"]["allow_effort_and_budget"] = "corrupt"
    monkeypatch.setattr(st, "_sg", FakeGateway(descriptor))
    model = provider()

    assert model._static_sandhi_model_descriptor("gpt-5") is None
    assert model.supports_reasoning_effort("gpt-5") is False


def gateway_provider() -> SandhiOpenAIProvider:
    model = provider()
    model.extra_config = {
        "gateway": {"url": "http://127.0.0.1:8080", "virtual_key": "vk-synthetic"}
    }
    return model


def test_gateway_discovery_is_sanitized_contract_v1(monkeypatch):
    model = gateway_provider()
    captured = {}
    grants = []
    model.extra_config["gateway"]["grant"] = "coding"

    def get(url, **kwargs):
        captured["url"] = url
        captured["authorization"] = kwargs["headers"]["Authorization"]
        grants.append(kwargs["headers"]["x-sandhi-grant"])
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "object": "list",
                "sandhi_contract_version": 1,
                "data": [
                    {
                        "id": "qwen3.7-plus",
                        "object": "model",
                        "thinking": {
                            "supported": True,
                            "effort": {"values": ["low", "medium", "xhigh"]},
                        },
                    }
                ],
            },
        )

    monkeypatch.setattr(st.httpx, "get", get)
    assert model.supports_reasoning_effort("qwen3.7-plus") is True
    assert captured["url"] == "http://127.0.0.1:8080/v1/models"
    assert captured["authorization"] == "Bearer vk-synthetic"
    assert model.supports_reasoning_effort("qwen3.7-plus") is True
    assert grants == ["coding"]
    model.extra_config["gateway"]["grant"] = "general"
    assert model.supports_reasoning_effort("qwen3.7-plus") is True
    assert grants == ["coding", "general"]


def test_gateway_discovery_failure_fails_closed_and_is_cached(monkeypatch):
    model = gateway_provider()

    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(st.httpx, "get", fail)
    assert model.supports_reasoning_effort("qwen3.7-plus") is False
    assert model._gateway_model_capabilities == {}

    # The failed page must not be retried for every capability check.
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("offline")

    monkeypatch.setattr(st.httpx, "get", fail_once)
    assert model.supports_reasoning_effort("qwen3.7-plus") is False
    assert calls == 0


@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
def test_sdk_effort_maps_to_semantic_thinking(monkeypatch, effort):
    monkeypatch.setattr(st, "_wire_contract_checked", True)
    request = st._typed_request_from_openai_payload(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": effort}
    )
    assert request["thinking"] == {"enabled": True, "effort": effort}
    assert "reasoning_effort" not in request
    assert "reasoning_effort" not in request.get("extensions", {}).get("openai", {})


def test_native_thinking_argument_retains_effort(monkeypatch):
    monkeypatch.setattr(st, "_wire_contract_checked", True)
    request = st._typed_request_from_openai_payload(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"enabled": True, "effort": "medium", "budget_tokens": 2048},
        }
    )
    assert request["thinking"] == {
        "enabled": True,
        "effort": "medium",
        "budget_tokens": 2048,
    }


@pytest.mark.asyncio
async def test_sync_gateway_discovery_never_blocks_running_event_loop(monkeypatch):
    model = gateway_provider()

    def fail(*args, **kwargs):
        raise AssertionError("synchronous discovery must not run inside an event loop")

    monkeypatch.setattr(st.httpx, "get", fail)
    assert model.supports_reasoning_effort("qwen3.7-plus") is False
    assert model._gateway_model_capabilities is None


@pytest.mark.asyncio
async def test_async_discovery_refreshes_renewable_gateway_token(monkeypatch):
    model = gateway_provider()
    model._api_key = "gateway-pending"
    model._gateway_audience = "sandhi-gateway"
    model._gateway_credential = SimpleNamespace(
        get_token=AsyncMock(
            return_value=SimpleNamespace(token="vk-refreshed", expires_on=4102444800.0)
        )
    )
    model.extra_config["gateway"] = {
        "url": "http://127.0.0.1:8080",
        "credential": model._gateway_credential,
        "audience": model._gateway_audience,
        "grant": "coding",
    }
    discovery_headers = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, **kwargs):
            discovery_headers.append(kwargs["headers"])
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "object": "list",
                    "sandhi_contract_version": 1,
                    "data": [
                        {
                            "id": "qwen3.7-plus",
                            "object": "model",
                            "thinking": {
                                "supported": True,
                                "effort": {"values": ["low", "medium"]},
                            },
                        }
                    ],
                },
            )

    monkeypatch.setattr(st.httpx, "AsyncClient", Client)
    assert await model.supports_reasoning_effort_async("qwen3.7-plus") is True
    model._gateway_credential.get_token.assert_awaited_once_with("sandhi-gateway")
    assert discovery_headers == [
        {"Authorization": "Bearer vk-refreshed", "x-sandhi-grant": "coding"}
    ]
    assert model._gateway_model_capabilities_key == (
        "http://127.0.0.1:8080",
        "vk-refreshed",
        "coding",
        None,
    )


@pytest.mark.asyncio
async def test_failed_placeholder_page_does_not_poison_refreshed_token(monkeypatch):
    model = gateway_provider()
    model._api_key = "gateway-pending"
    model._gateway_audience = "sandhi-gateway"
    model._gateway_credential = SimpleNamespace(
        get_token=AsyncMock(
            return_value=SimpleNamespace(token="vk-refreshed", expires_on=4102444800.0)
        )
    )
    model.extra_config["gateway"] = {
        "url": "http://127.0.0.1:8080",
        "credential": model._gateway_credential,
        "audience": model._gateway_audience,
    }

    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(st.httpx, "get", fail)
    assert model._load_gateway_model_capabilities() == {}
    assert model._gateway_model_capabilities_key == (
        "http://127.0.0.1:8080",
        "gateway-pending",
        None,
        None,
    )

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, **kwargs):
            assert kwargs["headers"]["Authorization"] == "Bearer vk-refreshed"
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "object": "list",
                    "sandhi_contract_version": 1,
                    "data": [
                        {
                            "id": "qwen3.7-plus",
                            "object": "model",
                            "thinking": {
                                "supported": True,
                                "effort": {"values": ["low", "medium"]},
                            },
                        }
                    ],
                },
            )

    monkeypatch.setattr(st.httpx, "AsyncClient", Client)
    assert await model.supports_reasoning_effort_async("qwen3.7-plus") is True
    assert model._gateway_model_capabilities_key == (
        "http://127.0.0.1:8080",
        "vk-refreshed",
        None,
        None,
    )


@pytest.mark.parametrize(
    "body",
    [
        {"sandhi_contract_version": 1, "data": []},
        {"object": "list", "sandhi_contract_version": 2, "data": []},
        {"object": "list", "sandhi_contract_version": True, "data": []},
        {"object": "list", "sandhi_contract_version": 1.0, "data": []},
        {"object": "list", "sandhi_contract_version": 1, "data": {}},
        {
            "object": "list",
            "sandhi_contract_version": 1,
            "data": [{"id": "m", "object": "model"}],
        },
        {
            "object": "list",
            "sandhi_contract_version": 1,
            "data": [
                {
                    "id": "m",
                    "object": "other",
                    "thinking": {"supported": False},
                }
            ],
        },
        {
            "object": "list",
            "sandhi_contract_version": 1,
            "data": [
                {"id": "m", "object": "model", "thinking": {}},
                {"id": "m", "object": "model", "thinking": {}},
            ],
        },
        {
            "object": "list",
            "sandhi_contract_version": 1,
            "data": [
                {
                    "id": "m",
                    "object": "model",
                    "thinking": {"supported": True, "effort": {"values": ["low", 2]}},
                }
            ],
        },
        {
            "object": "list",
            "sandhi_contract_version": 1,
            "data": [
                {
                    "id": "m",
                    "object": "model",
                    "thinking": {
                        "supported": True,
                        "budget_tokens": {"min": 2048, "max": 1024},
                    },
                }
            ],
        },
    ],
)
def test_gateway_discovery_rejects_malformed_model_pages(body):
    with pytest.raises(ValueError, match="Sandhi"):
        SandhiOpenAIProvider._parse_gateway_model_capabilities(body)


@pytest.mark.asyncio
async def test_gateway_chat_leaves_explicit_effort_admission_to_sandhi():
    model = gateway_provider()
    model._store_gateway_model_capabilities(
        ("http://127.0.0.1:8080", "vk-synthetic", None, None, None),
        {
            "qwen3.7-plus": {
                "id": "qwen3.7-plus",
                "object": "model",
                "thinking": {"supported": False},
            }
        },
    )
    captured = {}

    async def complete(request):
        captured.update(request)
        return {
            "schema_version": "1",
            "model": "qwen3.7-plus",
            "output": {"content": "ok"},
            "usage": {
                "tokens_in": 1,
                "tokens_out": 1,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "completeness": "final",
                "attempts": 1,
                "outcome": "success",
            },
        }

    model._sandhi_complete = complete
    from victor.providers.base import Message

    assert model.supports_reasoning_effort("qwen3.7-plus") is False
    await model.chat(
        [Message(role="user", content="hi")], model="qwen3.7-plus", reasoning_effort="high"
    )
    assert captured["thinking"] == {"enabled": True, "effort": "high"}


def _cached_low_effort_gateway() -> SandhiOpenAIProvider:
    model = gateway_provider()
    model._store_gateway_model_capabilities(
        ("http://127.0.0.1:8080", "vk-synthetic", None, None, None),
        {
            "qwen3.7-plus": {
                "id": "qwen3.7-plus",
                "object": "model",
                "thinking": {"supported": True, "effort": {"values": ["low"]}},
            }
        },
    )
    return model


def _capture_typed_response() -> dict:
    return {
        "schema_version": "1",
        "model": "qwen3.7-plus",
        "output": {"content": "ok"},
        "usage": None,
    }


@pytest.mark.asyncio
async def test_gateway_omitted_or_disabled_effort_performs_no_discovery():
    from victor.providers.base import Message

    model = _cached_low_effort_gateway()
    model._gateway_model_capabilities = None
    model._refresh_gateway_model_capabilities = AsyncMock(
        side_effect=AssertionError("inactive control must not discover")
    )
    model._sandhi_complete = AsyncMock(return_value=_capture_typed_response())

    async def stream(request):
        yield {"event": "finish"}

    model._sandhi_stream = stream
    await model.chat([Message(role="user", content="hi")], model="qwen3.7-plus")
    await model.chat(
        [Message(role="user", content="hi")], model="qwen3.7-plus", reasoning_effort=None
    )
    async for _ in model.stream(
        [Message(role="user", content="hi")], model="qwen3.7-plus", thinking={"enabled": False}
    ):
        pass
    model._refresh_gateway_model_capabilities.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_buffer_and_stream_leave_explicit_effort_admission_to_sandhi():
    from victor.providers.base import Message

    model = _cached_low_effort_gateway()
    model._refresh_gateway_model_capabilities = AsyncMock()
    complete_capture = {}
    stream_capture = {}

    async def complete(request):
        complete_capture.update(request)
        return _capture_typed_response()

    async def stream(request):
        stream_capture.update(request)
        yield {"event": "finish"}

    model._sandhi_complete = complete
    model._sandhi_stream = stream
    await model.chat(
        [Message(role="user", content="hi")], model="qwen3.7-plus", reasoning_effort="high"
    )
    async for _ in model.stream(
        [Message(role="user", content="hi")], model="qwen3.7-plus", reasoning_effort="high"
    ):
        pass
    assert complete_capture["thinking"] == {"enabled": True, "effort": "high"}
    assert stream_capture["thinking"] == {"enabled": True, "effort": "high"}

    await model.chat(
        [Message(role="user", content="hi")], model="qwen3.7-plus", reasoning_effort="low"
    )
    assert complete_capture["thinking"] == {"enabled": True, "effort": "low"}
    model._refresh_gateway_model_capabilities.assert_not_awaited()


@pytest.mark.asyncio
async def test_anthropic_gateway_buffer_and_stream_send_portable_thinking():
    from victor.providers.base import Message

    model = st.SandhiAnthropicProvider.__new__(st.SandhiAnthropicProvider)
    model.extra_config = {
        "gateway": {
            "url": "http://127.0.0.1:8080",
            "virtual_key": "vk-synthetic",
            "protocol": "anthropic_messages",
        }
    }
    model._store_gateway_model_capabilities(
        ("http://127.0.0.1:8080", "vk-synthetic", None, None, None),
        {
            "claude-test": {
                "id": "claude-test",
                "object": "model",
                "thinking": {"supported": False},
            }
        },
    )
    complete_capture = {}
    stream_capture = {}

    async def complete(request):
        complete_capture.update(request)
        return _capture_typed_response()

    async def stream(request):
        stream_capture.update(request)
        yield {"event": "finish"}

    model._sandhi_complete = complete
    model._sandhi_stream = stream
    await model.chat(
        [Message(role="user", content="hi")], model="claude-test", reasoning_effort="high"
    )
    async for _ in model.stream(
        [Message(role="user", content="hi")],
        model="claude-test",
        thinking={"enabled": True, "budget_tokens": 2048},
    ):
        pass
    assert complete_capture["thinking"] == {"enabled": True, "effort": "high"}
    assert stream_capture["thinking"] == {"enabled": True, "budget_tokens": 2048}
