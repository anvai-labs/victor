# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Mistral policy tests; transport and codec behavior are covered in Sandhi."""

import pytest

from victor.providers.base import Message, ToolDefinition
from victor.providers.mistral_provider import DEFAULT_BASE_URL, MISTRAL_MODELS, MistralProvider
from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy


@pytest.fixture
def provider() -> MistralProvider:
    return MistralProvider(api_key="test-key")


def test_mistral_is_thin_typed_policy(provider: MistralProvider) -> None:
    assert issubclass(MistralProvider, SandhiOpenAICompatPolicy)
    assert provider.name == "mistral"
    assert provider.base_url == DEFAULT_BASE_URL == "https://api.mistral.ai/v1"
    assert not hasattr(provider, "client")


def test_model_policy_is_preserved(provider: MistralProvider) -> None:
    required = {"description", "context_window", "max_output", "supports_tools"}
    assert all(required <= metadata.keys() for metadata in MISTRAL_MODELS.values())
    assert MISTRAL_MODELS["mistral-large-latest"]["supports_parallel_tools"] is True
    assert provider.context_window("mistral-large-latest") == 262_144
    assert provider.context_window("mistral-small-latest") == 131_072


def test_request_and_response_compatibility_helpers(provider: MistralProvider) -> None:
    payload = provider._build_request_payload(
        [Message(role="user", content="2+2")],
        "mistral-large-latest",
        0.7,
        128,
        [ToolDefinition(name="calculator", description="Calculate", parameters={})],
        False,
    )
    assert payload["tools"][0]["function"]["name"] == "calculator"
    response = provider._parse_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "calculator",
                                    "arguments": '{"expression":"2+2"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        "mistral-large-latest",
    )
    assert response.tool_calls == [
        {"id": "c1", "name": "calculator", "arguments": {"expression": "2+2"}}
    ]


@pytest.mark.asyncio
async def test_model_listing_is_local_and_cannot_bypass_sandhi(
    provider: MistralProvider, monkeypatch
) -> None:
    # Pin the fallback tier (see the groq twin): the catalog tier's coverage
    # varies with the installed sandhi binding.
    monkeypatch.setattr(type(provider), "_models_from_sandhi", lambda self: None)
    assert {entry["id"] for entry in await provider.list_models()} == set(MISTRAL_MODELS)
