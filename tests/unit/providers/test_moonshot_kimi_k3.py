"""Kimi K3 constraints are catalogued and enforced at Sandhi's typed boundary."""

from __future__ import annotations

import json

import pytest
import sandhi_gateway

import victor.providers.sandhi_transport as st
from victor.providers.base import Message
from victor.providers.moonshot_provider import (
    DEFAULT_BASE_URL,
    KIMI_K3_BASE_URL,
    MoonshotProvider,
)


def test_sandhi_catalog_owns_kimi_model_endpoint_and_temperature() -> None:
    assert MoonshotProvider.resolve_base_url_for_model("kimi-k3") == KIMI_K3_BASE_URL
    assert MoonshotProvider.resolve_base_url_for_model("kimi-k2-thinking") == DEFAULT_BASE_URL
    descriptor = json.loads(sandhi_gateway.provider_descriptor_json("moonshot"))
    kimi_k3 = next(model for model in descriptor["models"] if model["id"] == "kimi-k3")
    assert kimi_k3["default_temperature"] == 1.0
    assert kimi_k3["extensions"]["reasoning_effort_values"] == ["low", "high", "max"]


def test_compatibility_payload_reflects_k3_constraints() -> None:
    provider = MoonshotProvider(api_key="test-key")
    payload = provider._build_request_payload(
        [Message(role="user", content="hello")],
        "kimi-k3",
        0.2,
        32,
        None,
        False,
        reasoning_effort="high",
    )
    assert payload["temperature"] == 1.0
    assert payload["reasoning_effort"] == "high"
    with pytest.raises(ValueError, match="reasoning_effort"):
        provider._build_request_payload(
            [Message(role="user", content="hello")],
            "kimi-k3",
            0.2,
            32,
            None,
            False,
            reasoning_effort="medium",
        )


@pytest.mark.asyncio
async def test_default_endpoint_is_omitted_from_ffi_for_sandhi_model_routing(monkeypatch) -> None:
    calls: list[tuple] = []

    class Handle:
        async def complete_json(self, request_json: str, wire_headers_json=None) -> str:
            request = json.loads(request_json)
            return json.dumps(
                {
                    "schema_version": "1",
                    "model": request["model"],
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
            )

    class Runtime:
        def provider(self, *args, **kwargs):
            calls.append((args, kwargs))
            return Handle()

    monkeypatch.setattr(
        st,
        "_sg",
        type(
            "Binding",
            (),
            {
                "ProviderRuntime": Runtime,
                "provider_spec": staticmethod(lambda _provider: {"base_url": DEFAULT_BASE_URL}),
            },
        ),
    )
    provider = MoonshotProvider(api_key="test-key", max_retries=0)
    result = await provider.chat([Message(role="user", content="hello")], model="kimi-k3")
    assert result.content == "ok"
    assert calls[0][1]["base_url"] is None
