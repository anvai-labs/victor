# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Z.AI model-policy tests; wire behavior is covered in Sandhi."""

from victor.providers.sandhi_openai_compat_policy import SandhiOpenAICompatPolicy
from victor.providers.base import Message
from victor.providers.zai_provider import ZAI_MODELS, ZAIProvider


def test_zai_is_thin_typed_policy_without_python_http_client() -> None:
    assert issubclass(ZAIProvider, SandhiOpenAICompatPolicy)
    provider = ZAIProvider(api_key="test-key")
    assert provider.name == "zai"
    assert not hasattr(provider, "client")


def test_context_policy_preserves_glm_limits_and_suffixes() -> None:
    provider = ZAIProvider(api_key="test-key")
    assert ZAI_MODELS["glm-5.2"]["context_window"] == 1_000_000
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
