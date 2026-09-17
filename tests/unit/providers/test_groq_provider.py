# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Groq policy tests; provider wire behavior is covered in Sandhi."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import victor.providers.sandhi_transport as sandhi_transport
from victor.providers.base import Message
from victor.providers.groq_provider import DEFAULT_BASE_URL, GROQ_MODELS, GroqProvider
from victor.providers.payload_limiter import ProviderPayloadLimiter
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy


def test_groq_is_a_thin_typed_policy() -> None:
    assert issubclass(GroqProvider, SandhiOpenAICompatPolicy)
    provider = GroqProvider(api_key="test-key")
    assert provider.name == "groq"
    assert provider.base_url == DEFAULT_BASE_URL == "https://api.groq.com/openai/v1"
    assert not hasattr(provider, "client")


def test_model_policy_preserves_admitted_models_and_context() -> None:
    required = {"description", "context_window", "max_output", "supports_tools"}
    assert {
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    } <= GROQ_MODELS.keys()
    assert all(required <= metadata.keys() for metadata in GROQ_MODELS.values())
    provider = GroqProvider(api_key="test-key")
    assert provider.context_window("llama-3.3-70b-versatile") == 128_000
    assert provider.context_window("unknown") == 32_768


def test_groq_prompt_budget_is_victor_owned() -> None:
    provider = GroqProvider(api_key="test-key")
    assert isinstance(provider._payload_limiter, ProviderPayloadLimiter)
    assert provider._payload_limiter.provider_name == "groq"
    assert provider._payload_limiter.max_payload_bytes == 4 * 1024 * 1024


def test_cache_policy_is_agent_economics_not_wire_logic() -> None:
    cache = GroqProvider(api_key="test-key").cache_cost_model()
    assert cache.supported is True
    assert cache.read_discount == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_list_models_is_local_policy_and_does_not_bypass_sandhi(monkeypatch) -> None:
    provider = GroqProvider(api_key="test-key")
    # Listing is two-tier (TD-0004): sandhi catalog data first, Victor's admitted
    # policy as fallback — WHICH tier answers depends on the installed binding's
    # catalog coverage, so pin the fallback tier for a deterministic assertion.
    # The invariant under test is "local, no I/O", not "which tier".
    monkeypatch.setattr(type(provider), "_models_from_sandhi", lambda self: None)
    models = await provider.list_models()
    assert {entry["id"] for entry in models} == set(GROQ_MODELS)
    assert not hasattr(provider, "client")


@pytest.mark.asyncio
async def test_list_models_prefers_sandhi_catalog_tier(monkeypatch) -> None:
    provider = GroqProvider(api_key="test-key")
    catalog = [{"id": "catalog-model", "context_window": 8192}]
    monkeypatch.setattr(type(provider), "_models_from_sandhi", lambda self: list(catalog))
    assert await provider.list_models() == catalog


@pytest.mark.asyncio
async def test_chat_executes_through_typed_ffi(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[dict] = []

    class Handle:
        async def complete_json(self, request_json: str, wire_headers_json=None) -> str:
            request = json.loads(request_json)
            requests.append(request)
            return json.dumps(
                {
                    "schema_version": "1",
                    "model": request["model"],
                    "output": {"content": "ok"},
                    "finish_reason": "stop",
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
            )

    class Runtime:
        def provider(self, *args, **kwargs):
            assert args[0] == "groq"
            assert kwargs["base_url"] is None
            return Handle()

    monkeypatch.setattr(
        sandhi_transport,
        "_sg",
        SimpleNamespace(
            ProviderRuntime=Runtime,
            provider_spec=lambda _provider: {"base_url": DEFAULT_BASE_URL},
        ),
    )
    response = await GroqProvider(api_key="test-key", max_retries=0).chat(
        [Message(role="user", content="hello")], model="llama-3.3-70b-versatile"
    )
    assert response.content == "ok"
    assert requests[0]["messages"] == [{"role": "user", "content": "hello"}]
