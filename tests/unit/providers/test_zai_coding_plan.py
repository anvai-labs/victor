# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0
"""Z.AI named-endpoint policy tests."""

import pytest

from victor.providers.zai_provider import ZAI_BASE_URLS, ZAI_MODELS, ZAIProvider


def test_named_endpoints_are_loaded_from_sandhi_descriptor() -> None:
    assert ZAI_BASE_URLS == {
        "standard": "https://api.z.ai/api/paas/v4/",
        "coding": "https://api.z.ai/api/coding/paas/v4/",
        "china": "https://open.bigmodel.cn/api/paas/v4/",
        "china-coding": "https://open.bigmodel.cn/api/coding/paas/v4/",
    }


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, ZAI_BASE_URLS["standard"].rstrip("/")),
        ({"coding_plan": True}, ZAI_BASE_URLS["coding"].rstrip("/")),
        ({"endpoint": "china"}, ZAI_BASE_URLS["china"].rstrip("/")),
        ({"model": "glm-4.6:coding"}, ZAI_BASE_URLS["coding"].rstrip("/")),
    ],
)
def test_endpoint_selection(kwargs, expected) -> None:
    provider = ZAIProvider(api_key="test-key", **kwargs)
    assert provider.base_url == expected
    assert not hasattr(provider, "client")


def test_explicit_endpoint_wins_over_plan_policy() -> None:
    provider = ZAIProvider(
        api_key="test-key", base_url="https://private.example/v1", coding_plan=True
    )
    assert provider.base_url == "https://private.example/v1"


@pytest.mark.parametrize("selector", [{"endpoint": "anthropic"}, {"model": "glm:anthropic"}])
def test_anthropic_dialect_is_rejected_at_openai_boundary(selector) -> None:
    with pytest.raises(ValueError, match="different protocol"):
        ZAIProvider(api_key="test-key", **selector)


def test_models_and_tool_call_compatibility_helpers_remain_available() -> None:
    provider = ZAIProvider(api_key="test-key")
    assert {"glm-5.2", "glm-5", "glm-4.7", "glm-4.5-air"} <= ZAI_MODELS.keys()
    assert provider._normalize_tool_calls(
        [{"id": "c1", "function": {"name": "calc", "arguments": '{"x":1}'}}]
    ) == [{"id": "c1", "name": "calc", "arguments": {"x": 1}}]


def test_config_registry_keeps_zhipu_alias() -> None:
    from victor.config.provider_config_registry import get_provider_config_registry

    assert get_provider_config_registry()._aliases.get("zhipu") == "zai"
