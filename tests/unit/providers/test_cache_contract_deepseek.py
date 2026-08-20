# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""TDD RED-PHASE: DeepSeek prompt-caching contract tests.

These tests encode the desired caching contract for DeepSeek. They are
expected to FAIL until the implementation is completed (GREEN phase).

Contract under test:
1. DeepSeekProvider.supports_prompt_caching() == True
2. Victor keeps tool policy deterministic; Sandhi owns final wire serialization.
"""

from typing import Any, Dict, List, Optional

import pytest

from victor.providers.base import Message, ToolDefinition
from victor.providers.deepseek_provider import DeepSeekProvider

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_tools(n: int = 3) -> List[ToolDefinition]:
    return [
        ToolDefinition(
            name=f"tool_{i}",
            description=f"Tool number {i}",
            parameters={
                "type": "object",
                "properties": {"arg": {"type": "string"}},
            },
        )
        for i in range(n)
    ]


def _make_provider(api_key: str = "sk-test-fake") -> DeepSeekProvider:
    return DeepSeekProvider(api_key=api_key, base_url="https://api.deepseek.com/v1")


# ── 1. supports_prompt_caching ─────────────────────────────────────────────────


class TestDeepSeekPromptCachingFlag:
    """Contract: DeepSeek must declare API-level prompt caching support."""

    def test_supports_prompt_caching_returns_true(self):
        """DeepSeek offers cached-token billing discounts (context caching)."""
        provider = _make_provider()
        assert provider.supports_prompt_caching() is True

    def test_supports_kv_prefix_caching_returns_true(self):
        """DeepSeek reuses KV cache for stable prompt prefixes."""
        provider = _make_provider()
        assert provider.supports_kv_prefix_caching() is True


# ── 2. tools[] before messages[] for auto-prefix ───────────────────────────────


class TestDeepSeekDeterministicPromptPolicy:
    """Victor supplies stable tools/messages without asserting Rust JSON field order."""

    def test_tools_and_messages_are_preserved(self):
        provider = _make_provider()
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello"),
        ]
        tools = _make_tools(3)

        payload = provider._build_request_payload(
            messages=messages,
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=4096,
            tools=tools,
            stream=False,
        )

        assert [tool["function"]["name"] for tool in payload["tools"]] == [
            "tool_0",
            "tool_1",
            "tool_2",
        ]
        assert [message["role"] for message in payload["messages"]] == ["system", "user"]

    def test_tool_choice_is_stable(self):
        provider = _make_provider()
        messages = [Message(role="user", content="Hi")]
        tools = _make_tools(2)

        payload = provider._build_request_payload(
            messages=messages,
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=4096,
            tools=tools,
            stream=False,
        )

        assert payload["tool_choice"] == "auto"
