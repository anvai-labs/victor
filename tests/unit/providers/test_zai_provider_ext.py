# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Z.AI model-policy tests; wire behavior is covered in Sandhi."""

import pytest

from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy
from victor.providers.base import Message
from victor.providers.zai_provider import ZAI_MODELS, ZAIProvider
from victor.providers.sandhi_transport import _typed_request_from_openai_payload


def test_zai_is_thin_typed_policy_without_python_http_client() -> None:
    assert issubclass(ZAIProvider, SandhiOpenAICompatPolicy)
    provider = ZAIProvider(api_key="test-key")
    assert provider.name == "zai"
    assert not hasattr(provider, "client")


def test_context_policy_preserves_glm_limits_and_suffixes() -> None:
    provider = ZAIProvider(api_key="test-key")
    assert ZAI_MODELS["glm-5.2"]["context_window"] == 1_000_000
    assert ZAI_MODELS["glm-5.2"]["max_output"] == 131_072
    assert ZAI_MODELS["glm-5.1"]["max_output"] == 131_072
    assert provider.context_window("glm-5.3") == 1_000_000
    assert provider.context_window("glm-5.3:coding") == 1_000_000
    assert provider.context_window("glm-5.2") == 1_000_000
    assert provider.context_window("glm-5.2:coding") == 1_000_000
    assert provider.context_window("glm-4.7") == 200_000
    assert provider.context_window("glm-4.6") == 128_000
    assert provider.context_window("glm-unknown") == 200_000


def test_thinking_parameter_is_namespaced_for_sandhi_codec() -> None:
    provider = ZAIProvider(api_key="test-key")
    payload = provider._build_request_payload(
        [Message(role="user", content="hi")],
        "glm-4.6",
        0.7,
        32,
        None,
        False,
        thinking=True,
    )
    assert payload["thinking"] == {"type": "enabled"}


def test_compatibility_parser_keeps_reasoning_content() -> None:
    provider = ZAIProvider(api_key="test-key")
    response = provider._parse_response(
        {
            "choices": [
                {
                    "message": {"content": "Answer", "reasoning_content": "Step 1"},
                    "finish_reason": "stop",
                }
            ]
        },
        "glm-4.6",
    )
    assert response.metadata == {"reasoning_content": "Step 1"}


@pytest.mark.parametrize("model", ["glm-5.3", "glm-5.3:coding", "glm-5.3:china-coding"])
@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_direct_glm53_reasoning_effort_fails_closed_until_catalogued(
    model: str, effort: str
) -> None:
    provider = ZAIProvider(api_key="test-key")
    assert provider.supports_reasoning_effort(model) is False
    payload = provider._build_request_payload(
        [Message(role="user", content="hi")],
        model,
        0.2,
        64,
        None,
        False,
        reasoning_effort=effort,
    )
    assert "reasoning_effort" not in payload
    request = _typed_request_from_openai_payload(payload)
    assert "thinking" not in request


@pytest.mark.asyncio
async def test_gateway_discovery_can_admit_glm53_reasoning_effort(monkeypatch) -> None:
    provider = ZAIProvider.__new__(ZAIProvider)
    provider._spec = ZAIProvider.provider_spec()
    provider.extra_config = {
        "gateway": {"url": "http://127.0.0.1:8080", "virtual_key": "vk-synthetic"}
    }
    captured = {}

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, **kwargs):
            captured["url"] = url
            captured["authorization"] = kwargs["headers"]["Authorization"]
            from types import SimpleNamespace

            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "object": "list",
                    "sandhi_contract_version": 1,
                    "data": [
                        {
                            "id": "glm-5.3",
                            "object": "model",
                            "thinking": {
                                "supported": True,
                                "effort": {"values": ["low", "high", "max"]},
                            },
                        }
                    ],
                },
            )

    monkeypatch.setattr("victor.providers.sandhi_transport.httpx.AsyncClient", Client)
    assert await provider.supports_reasoning_effort_async("glm-5.3") is True
    assert captured["url"] == "http://127.0.0.1:8080/v1/models"
    assert captured["authorization"] == "Bearer vk-synthetic"


@pytest.mark.parametrize("effort", ["medium", "none", "ultra", "invalid", 1, [], {}])
def test_glm53_rejects_unsupported_effort_before_transport(effort: object) -> None:
    provider = ZAIProvider(api_key="test-key")
    with pytest.raises(ValueError, match="GLM-5.3 reasoning_effort"):
        provider._build_request_payload(
            [Message(role="user", content="hi")],
            "glm-5.3",
            0.2,
            64,
            None,
            False,
            reasoning_effort=effort,
        )


def test_glm53_omitted_effort_preserves_provider_default() -> None:
    provider = ZAIProvider(api_key="test-key")
    payload = provider._build_request_payload(
        [Message(role="user", content="hi")],
        "glm-5.3",
        0.2,
        64,
        None,
        False,
    )
    assert "reasoning_effort" not in payload
