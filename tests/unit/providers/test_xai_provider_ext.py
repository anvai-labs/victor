# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""xAI policy tests; transport and error behavior are covered in Sandhi."""

from victor.providers.base import Message, ToolDefinition
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy
from victor.providers.xai_provider import DEFAULT_BASE_URL, XAI_MODELS, XAIProvider


def test_xai_is_thin_typed_policy_without_python_http_client() -> None:
    assert issubclass(XAIProvider, SandhiOpenAICompatPolicy)
    provider = XAIProvider(api_key="test-key")
    assert provider.name == "xai"
    assert provider.base_url == DEFAULT_BASE_URL
    assert not hasattr(provider, "client")


def test_model_aliases_and_context_policy() -> None:
    provider = XAIProvider(api_key="test-key")
    assert provider._clean_model_name("grok-4.1-fast") == "grok-4-1-fast"
    assert provider.context_window("grok-4.1-fast") == 2_000_000
    assert provider.get_context_window("grok-4") == 262_144
    assert len(XAI_MODELS) == 8


def test_request_policy_keeps_roles_and_tools() -> None:
    provider = XAIProvider(api_key="test-key")
    payload = provider._build_request_payload(
        [Message(role="developer", content="rules"), Message(role="user", content="hi")],
        "grok-4",
        0.2,
        64,
        [ToolDefinition(name="lookup", description="Lookup", parameters={"type": "object"})],
        False,
    )
    assert [message["role"] for message in payload["messages"]] == ["developer", "user"]
    assert payload["tools"][0]["function"]["name"] == "lookup"


def test_cache_economics_remain_victor_policy() -> None:
    cache = XAIProvider(api_key="test-key").cache_cost_model()
    assert cache.supported and cache.read_discount == 0.625
