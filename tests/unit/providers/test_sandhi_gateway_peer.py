"""Explicit gateway client role, independent of direct backend policy."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from victor.config.settings import ProviderGatewayConfig
from victor.providers.base import Message
from victor.providers.sandhi_transport import resolve_transport_class
from victor.providers.anthropic_provider import AnthropicProvider
from victor.providers.google_provider import GoogleProvider
from victor.providers.openai_provider import OpenAIProvider
from victor.providers.zai_provider import ZAIProvider
from types import SimpleNamespace
import victor.providers.sandhi_transport as st


def install_runtime(monkeypatch):
    class Handle:
        requests = None

        def __init__(self):
            self.requests = []

        async def complete_json(self, request, headers=None):
            self.requests.append(json.loads(request))
            return json.dumps(
                {
                    "schema_version": "1",
                    "output": {"content": "hello"},
                    "finish_reason": "stop",
                    "usage": {
                        "tokens_in": 2,
                        "tokens_out": 1,
                        "attempts": 1,
                        "completeness": "final",
                    },
                }
            )

    handle = Handle()
    runtime = SimpleNamespace(handle=handle, calls=[])

    def provider(*args, **kwargs):
        runtime.calls.append((args, kwargs))
        return handle

    runtime.provider = provider
    monkeypatch.setattr(st, "_sg", SimpleNamespace(ProviderRuntime=lambda: runtime))
    monkeypatch.setattr(st, "_verify_wire_contract", lambda: None)
    monkeypatch.setattr(st, "_installed_contract_minor", 14)
    return runtime


PROTOCOLS = {
    "chat_completions": "/v1",
    "anthropic_messages": "",
    "responses": "/v1",
    "gemini_generate_content": "/v1beta",
}


@pytest.mark.parametrize("protocol,suffix", PROTOCOLS.items())
@pytest.mark.parametrize(
    "label,native",
    [
        ("anthropic", AnthropicProvider),
        ("google", GoogleProvider),
        ("openai", OpenAIProvider),
        ("zai", ZAIProvider),
    ],
)
async def test_gateway_ignores_backend_profile_and_selects_explicit_client_dialect(
    monkeypatch, protocol, suffix, label, native
):
    runtime = install_runtime(monkeypatch)
    cls = resolve_transport_class(label, native, {})
    provider = cls(
        gateway={
            "url": "https://gateway.example.test/v1beta",
            "virtual_key": "vk_fixture",
            "protocol": protocol,
        },
        endpoint_profile="openai_chat_terminal",
    )
    provider._sandhi_endpoint_profile = "openai_chat_terminal"
    provider._wire_headers = {
        "Authorization": "upstream-fixture",
        "CHATGPT-ACCOUNT-ID": "upstream-account",
        "Idempotency-Key": "logical",
        "x-sandhi-session": "session",
        "x-request-id": "correlation",
    }
    await provider.chat(
        [Message(role="user", content="hi")],
        model="server-logical-alias",
        cache_intent={"mode": "auto_stable_prefix", "lifetime": "5m"},
    )
    args, options = runtime.calls[0]
    assert args[2] == "vk_fixture"
    assert options["protocol"] == protocol
    assert options["peer_role"] == "sandhi_gateway_v1"
    assert options["base_url"] == "https://gateway.example.test" + suffix
    assert options["max_retries"] == 0
    assert "endpoint_profile" not in options
    assert json.loads(options["headers_json"]) == {
        "x-sandhi-session": "session",
        "x-request-id": "correlation",
    }
    request = runtime.handle.requests[0]
    assert request["model"] == "server-logical-alias"
    assert request["cache_intent"] == {"mode": "auto_stable_prefix", "lifetime": "5m"}
    assert "cache_control" not in json.dumps(request)
    assert "compiled_plan" not in json.dumps(request)
    assert not any(request.get("extensions", {}).values())
    await provider.close()


@pytest.mark.parametrize(
    "label,native,module",
    [
        ("anthropic", AnthropicProvider, "anthropic_provider"),
        ("google", GoogleProvider, "google_provider"),
        ("zai", ZAIProvider, "sandhi_openai_compat_policy"),
    ],
)
async def test_gateway_never_resolves_upstream_credentials_or_refreshes(
    monkeypatch, label, native, module
):
    install_runtime(monkeypatch)
    cls = resolve_transport_class(label, native, {})
    with patch(f"victor.providers.{module}.UnifiedApiKeyResolver") as resolver:
        kwargs = {"auth_mode": "oauth"} if label in {"anthropic", "google"} else {}
        with patch("victor.providers.oauth_manager.OAuthTokenManager") as oauth:
            provider = cls(
                gateway={"url": "https://gateway.example.test", "virtual_key": "vk_fixture"},
                **kwargs,
            )
            provider._ensure_valid_token = AsyncMock(
                side_effect=AssertionError("upstream refresh forbidden")
            )
            await provider.chat([Message(role="user", content="hi")], model="alias")
            resolver.assert_not_called()
            oauth.assert_not_called()
            provider._ensure_valid_token.assert_not_called()
            await provider.close()


@pytest.mark.parametrize("protocol", ["responses", "anthropic_messages", "gemini_generate_content"])
async def test_gateway_refuses_incompatible_native_options_without_dispatch(monkeypatch, protocol):
    runtime = install_runtime(monkeypatch)
    provider = resolve_transport_class("zai", ZAIProvider, {})(
        gateway={
            "url": "https://gateway.example.test",
            "virtual_key": "vk_fixture",
            "protocol": protocol,
        }
    )
    with pytest.raises(ValueError, match="native call options"):
        await provider.chat([Message(role="user", content="hi")], model="alias", top_p=0.5)
    assert not runtime.calls
    await provider.close()


def test_gateway_config_exposes_four_client_dialects():
    assert (
        ProviderGatewayConfig(url="https://gateway.example.test", protocol="responses").protocol
        == "responses"
    )
    with pytest.raises(ValueError):
        ProviderGatewayConfig(protocol="chatgpt_responses")


@pytest.mark.parametrize(
    "label,native,protocol,prompt",
    [
        ("anthropic", AnthropicProvider, "responses", 8),
        ("google", GoogleProvider, "anthropic_messages", 2),
        ("zai", ZAIProvider, "gemini_generate_content", 8),
    ],
)
async def test_consumer_preserves_client_usage_categories_independent_of_provider_label(
    monkeypatch, label, native, protocol, prompt
):
    from victor.agent.services.metrics_service import _InvocationObservation
    from victor.agent.stream_handler import StreamMetrics
    from victor.agent.session_cost_tracker import SessionCostTracker
    from victor.providers.usage_accounting import accumulate_usage
    from unittest.mock import Mock

    runtime = install_runtime(monkeypatch)
    neutral = {
        "tokens_in": 2,
        "tokens_out": 4,
        "cache_creation_tokens": 3,
        "cache_read_tokens": 6,
        "reasoning_tokens": 2,
        "reasoning_included": False,
        "completeness": "final",
        "attempts": 1,
        "cache_read_observation": {"availability": "reported", "source": "provider"},
    }
    runtime.handle.complete_json = AsyncMock(
        return_value=json.dumps(
            {
                "schema_version": "1",
                "output": {"content": "reply"},
                "usage": neutral,
                "finish_reason": "stop",
            }
        )
    )
    provider = resolve_transport_class(label, native, {})(
        gateway={
            "url": "https://gateway.example.test",
            "virtual_key": "vk_fixture",
            "protocol": protocol,
        }
    )
    result = await provider.chat([Message(role="user", content="hi")], model="alias")
    assert result.usage["prompt_tokens"] == prompt
    observation = _InvocationObservation()
    observation.observe(result)
    assert observation.usage["cache_creation_input_tokens"] == 3
    assert observation.usage["cache_read_input_tokens"] == 6
    assert observation.usage["reasoning_tokens"] == 2
    assert observation.usage["reasoning_included"] is False
    assert observation.diagnostics["gateway_client_protocol"] == protocol
    assert observation.diagnostics["gateway_client_usage"] == {
        key: value for key, value in neutral.items() if key != "attempts"
    }
    metrics = StreamMetrics()
    metrics.record_usage(observation.usage)
    assert metrics.prompt_tokens == prompt
    assert metrics.cache_write_tokens == 3
    assert metrics.cache_read_tokens == 6
    assert metrics.billable_completion_tokens == 6
    aggregate = {}
    accumulate_usage(aggregate, observation.usage)
    assert aggregate["cache_creation_input_tokens"] == 3
    assert aggregate["billable_completion_tokens"] == 6
    tracker = SessionCostTracker(provider=label, model="alias")
    tracker._sandhi = None
    tracker._capabilities = Mock(cost_enabled=False)
    request = tracker.begin_request()
    tracker.settle_request(request, outcome="success", usage=observation.usage)
    assert request.prompt_tokens == prompt
    assert request.cache_write_tokens == 3
    assert request.cache_read_tokens == 6
    assert request.billable_completion_tokens == 6
    await provider.close()


async def test_gateway_stream_cache_availability_correction_preserves_observed_counts(monkeypatch):
    runtime = install_runtime(monkeypatch)
    observation = {"availability": "unavailable", "source": "provider"}

    async def events(request, headers=None):
        for event in [
            {
                "event": "usage",
                "usage": {
                    "tokens_in": 2,
                    "tokens_out": 4,
                    "cache_read_tokens": 6,
                    "cache_creation_tokens": 0,
                    "reasoning_tokens": 0,
                    "completeness": "final",
                },
            },
            {
                "event": "usage",
                "usage": {
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cache_read_tokens": 0,
                    "completeness": "unavailable",
                    "cache_read_observation": observation,
                },
            },
            {"event": "finish", "reason": "stop"},
        ]:
            yield json.dumps(event)

    runtime.handle.stream_json = events
    provider = resolve_transport_class("anthropic", AnthropicProvider, {})(
        gateway={
            "url": "https://gateway.example.test",
            "virtual_key": "vk_fixture",
            "protocol": "responses",
        }
    )
    chunks = [
        chunk
        async for chunk in provider.stream([Message(role="user", content="hello")], model="alias")
    ]
    final = chunks[-1]
    assert final.usage["prompt_tokens"] == 8
    assert final.usage["cache_read_input_tokens"] == 6
    for key in ("typed_usage", "gateway_client_usage"):
        units = final.metadata["sandhi_usage"][key]
        assert units["tokens_in"] == 2
        assert units["tokens_out"] == 4
        assert units["cache_creation_tokens"] == 0
        assert units["reasoning_tokens"] == 0
        assert units["completeness"] == "final"
        assert units["cache_read_observation"] == observation
    await provider.close()


def test_provider_initialization_redacts_nested_gateway_credentials():
    from victor.providers.logging import ProviderLogger

    credential = object()
    sanitized = ProviderLogger("fixture", __name__)._sanitize_config(
        {
            "gateway": {
                "url": "https://gateway.example.test",
                "virtual_key": "private-gateway-fixture",
                "credential": credential,
            },
            "timeout": 10,
        }
    )
    assert sanitized == {"gateway": "***", "timeout": 10}
    assert "private-gateway-fixture" not in str(sanitized)
