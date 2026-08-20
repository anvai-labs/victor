# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""Unit tests for the PayloadSizeLimiter."""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from victor.providers.payload_limiter import (
    DEFAULT_LIMITS,
    PayloadEstimate,
    ProviderPayloadLimiter,
    TruncationResult,
    TruncationStrategy,
    get_payload_limiter,
)


@dataclass
class MockMessage:
    """Mock message for testing."""

    role: str
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None


@dataclass
class MockToolDefinition:
    """Mock tool definition for testing."""

    name: str
    description: str
    parameters: Dict[str, Any]


class TestProviderPayloadLimiter:
    """Test ProviderPayloadLimiter functionality."""

    def test_default_limits_for_known_providers(self):
        """Test that known providers have default limits."""
        assert "groq" in DEFAULT_LIMITS
        assert "groqcloud" in DEFAULT_LIMITS
        assert "anthropic" in DEFAULT_LIMITS
        assert "openai" in DEFAULT_LIMITS
        assert "ollama" in DEFAULT_LIMITS

    def test_groq_limit_is_4mb(self):
        """Test that Groq limit is 4MB."""
        limiter = ProviderPayloadLimiter(provider_name="groq")
        assert limiter.max_payload_bytes == 4 * 1024 * 1024

    def test_estimate_size_basic(self):
        """Test basic payload size estimation."""
        limiter = ProviderPayloadLimiter(provider_name="groq")
        messages = [
            MockMessage(role="user", content="Hello, world!"),
            MockMessage(role="assistant", content="Hi there!"),
        ]

        estimate = limiter.estimate_size(messages, None)

        assert estimate.total_bytes > 0
        assert estimate.messages_bytes > 0
        assert estimate.user_bytes > 0
        assert estimate.assistant_bytes > 0
        assert not estimate.exceeds_limit

    def test_estimate_size_with_tools(self):
        """Test payload size estimation with tools."""
        limiter = ProviderPayloadLimiter(provider_name="groq")
        messages = [MockMessage(role="user", content="Test")]
        tools = [
            MockToolDefinition(
                name="test_tool",
                description="A test tool for testing",
                parameters={
                    "type": "object",
                    "properties": {"arg1": {"type": "string"}},
                },
            )
        ]

        estimate = limiter.estimate_size(messages, tools)

        assert estimate.tools_bytes > 0
        assert estimate.total_bytes > estimate.messages_bytes

    def test_estimate_size_tool_results(self):
        """Test that tool results are tracked separately."""
        limiter = ProviderPayloadLimiter(provider_name="groq")
        messages = [
            MockMessage(role="user", content="Test"),
            MockMessage(role="tool", content="Tool result content" * 100, tool_call_id="tc_1"),
        ]

        estimate = limiter.estimate_size(messages, None)

        assert estimate.tool_result_bytes > 0
        assert estimate.tool_result_bytes > estimate.user_bytes

    def test_check_limit_ok(self):
        """Test check_limit returns True for small payloads."""
        limiter = ProviderPayloadLimiter(provider_name="groq")
        messages = [MockMessage(role="user", content="Hello")]

        ok, warning = limiter.check_limit(messages, None)

        assert ok is True
        assert warning is None

    def test_check_limit_exceeds(self):
        """Test check_limit returns False for large payloads."""
        # Use 1KB limit for testing
        limiter = ProviderPayloadLimiter(provider_name="test", max_payload_bytes=1024)
        # Create message that exceeds 1KB
        messages = [MockMessage(role="user", content="x" * 2000)]

        ok, warning = limiter.check_limit(messages, None)

        assert ok is False
        assert warning is not None
        assert "exceeds limit" in warning.lower()

    def test_check_limit_warning_threshold(self):
        """Test check_limit warns when approaching limit."""
        # Use larger limit for more predictable behavior
        limiter = ProviderPayloadLimiter(
            provider_name="test",
            max_payload_bytes=2000,
            warning_threshold=0.80,
            safety_margin=1.0,  # Disable safety margin for predictable test
        )
        # Create message that puts us at ~85% utilization (above 80% warning)
        # JSON overhead: {"role": "user", "content": "..."} adds ~30 bytes
        # Plus overhead_bytes (200) puts us well above 80%
        messages = [MockMessage(role="user", content="x" * 1500)]

        ok, warning = limiter.check_limit(messages, None)

        assert ok is True  # Still OK, but warning
        assert warning is not None
        assert "%" in warning  # Contains percentage

    def test_truncate_if_needed_no_truncation(self):
        """Test truncate_if_needed does nothing for small payloads."""
        limiter = ProviderPayloadLimiter(provider_name="groq")
        messages = [MockMessage(role="user", content="Hello")]

        result = limiter.truncate_if_needed(messages, None)

        assert result.truncated is False
        assert result.messages == messages
        assert result.messages_removed == 0

    def test_truncate_if_needed_removes_oldest(self):
        """Test truncate_if_needed removes oldest messages."""
        # Use small limit
        limiter = ProviderPayloadLimiter(provider_name="test", max_payload_bytes=500)
        messages = [
            MockMessage(role="system", content="System prompt"),
            MockMessage(role="user", content="x" * 200),
            MockMessage(role="assistant", content="x" * 200),
            MockMessage(role="user", content="x" * 200),
            MockMessage(role="assistant", content="Final response"),
        ]

        result = limiter.truncate_if_needed(
            messages, None, strategy=TruncationStrategy.TRUNCATE_OLDEST
        )

        assert result.truncated is True
        assert result.messages_removed > 0
        assert len(result.messages) < len(messages)
        # System message should be preserved
        assert result.messages[0].role == "system"

    def test_truncate_tool_results(self):
        """Test truncating large tool results."""
        limiter = ProviderPayloadLimiter(provider_name="test", max_payload_bytes=1000)
        large_tool_result = "x" * 5000  # Much larger than 2000 char limit
        messages = [
            MockMessage(role="user", content="Test"),
            MockMessage(role="tool", content=large_tool_result, tool_call_id="tc_1"),
        ]

        result = limiter.truncate_if_needed(
            messages, None, strategy=TruncationStrategy.TRUNCATE_TOOL_RESULTS
        )

        assert result.truncated is True
        # Tool result should be truncated
        tool_msg = [m for m in result.messages if m.role == "tool"][0]
        assert len(tool_msg.content) < len(large_tool_result)
        assert "[truncated]" in tool_msg.content

    def test_dictionary_compression_deduplicates_repeated_tool_results(self):
        """Repeated tool payloads should collapse to a shared dictionary reference."""
        limiter = ProviderPayloadLimiter(
            provider_name="test",
            max_payload_bytes=9000,
            safety_margin=1.0,
            enable_dictionary_compression=True,
        )
        repeated_tool_result = "\n".join(f"line {i}: repeated payload content" for i in range(220))
        messages = [
            MockMessage(role="user", content="Analyze the duplicated tool outputs"),
            MockMessage(role="tool", content=repeated_tool_result, tool_call_id="tc_1"),
            MockMessage(role="tool", content=repeated_tool_result, tool_call_id="tc_2"),
        ]

        estimate = limiter.estimate_size(messages, None)
        assert estimate.exceeds_limit

        result = limiter.truncate_if_needed(
            messages, None, strategy=TruncationStrategy.TRUNCATE_TOOL_RESULTS
        )

        assert result.truncated is True
        assert result.strategy_used == TruncationStrategy.DICTIONARY_COMPRESS
        assert result.compression_applied is True
        assert result.compression_entries == 1
        tool_messages = [m for m in result.messages if m.role == "tool"]
        assert "[dictionary-ref:" in tool_messages[1].content
        assert "[truncated]" not in tool_messages[0].content

    def test_dictionary_compression_is_opt_in(self):
        """Without the flag, the limiter should fall back to normal truncation."""
        limiter = ProviderPayloadLimiter(
            provider_name="test",
            max_payload_bytes=4500,
            safety_margin=1.0,
            enable_dictionary_compression=False,
        )
        repeated_tool_result = "x" * 5000
        messages = [
            MockMessage(role="user", content="Analyze the duplicated tool outputs"),
            MockMessage(role="tool", content=repeated_tool_result, tool_call_id="tc_1"),
            MockMessage(role="tool", content=repeated_tool_result, tool_call_id="tc_2"),
        ]

        result = limiter.truncate_if_needed(
            messages, None, strategy=TruncationStrategy.TRUNCATE_TOOL_RESULTS
        )

        assert result.strategy_used == TruncationStrategy.TRUNCATE_TOOL_RESULTS
        assert result.compression_applied is False
        assert result.compression_entries == 0
        tool_messages = [m for m in result.messages if m.role == "tool"]
        assert all("[dictionary-ref:" not in m.content for m in tool_messages)
        assert all("[truncated]" in m.content for m in tool_messages)

    def test_reduce_tools_strategy(self):
        """Test reducing number of tools reduces tool count."""
        # Create a limiter with a limit that will be exceeded
        limiter = ProviderPayloadLimiter(
            provider_name="test",
            max_payload_bytes=2000,
            safety_margin=1.0,  # Disable safety margin for predictable test
        )
        # Need multiple messages so truncate_oldest can work as fallback
        messages = [
            MockMessage(role="system", content="System prompt"),
            MockMessage(role="user", content="First user message"),
            MockMessage(role="assistant", content="First response"),
            MockMessage(role="user", content="Second user message"),
        ]
        # Create 20 tools with large descriptions to ensure we exceed limit
        tools = [
            MockToolDefinition(
                name=f"tool_{i}",
                description="This is a test tool with a longer description " * 3,
                parameters={
                    "type": "object",
                    "properties": {"arg": {"type": "string"}},
                },
            )
            for i in range(20)
        ]

        # Check that tools exceed the limit
        estimate = limiter.estimate_size(messages, tools)
        assert (
            estimate.exceeds_limit
        ), f"Test setup issue: payload should exceed limit ({estimate.total_bytes} vs {estimate.limit_bytes})"

        result = limiter.truncate_if_needed(
            messages, tools, strategy=TruncationStrategy.REDUCE_TOOLS
        )

        # The strategy should have done something
        assert result.truncated is True
        # With 20 tools, reduce_tools should cut to 10, which may still exceed
        # so it may also truncate messages. Check that some action was taken.
        assert result.tools_removed > 0 or result.messages_removed > 0 or result.bytes_saved > 0

    def test_fail_strategy(self):
        """Test FAIL strategy returns error without truncation."""
        limiter = ProviderPayloadLimiter(provider_name="test", max_payload_bytes=100)
        messages = [MockMessage(role="user", content="x" * 500)]

        result = limiter.truncate_if_needed(messages, None, strategy=TruncationStrategy.FAIL)

        assert result.truncated is False
        assert result.warning is not None
        assert "exceeds limit" in result.warning.lower()

    def test_get_payload_limiter_factory(self):
        """Test factory function creates limiter correctly."""
        limiter = get_payload_limiter("groq")

        assert isinstance(limiter, ProviderPayloadLimiter)
        assert limiter.provider_name == "groq"
        assert limiter.max_payload_bytes == 4 * 1024 * 1024

    def test_get_payload_limiter_with_override(self):
        """Test factory function respects overrides."""
        limiter = get_payload_limiter("groq", max_payload_bytes=2 * 1024 * 1024)

        assert limiter.max_payload_bytes == 2 * 1024 * 1024

    def test_payload_estimate_to_dict(self):
        """Test PayloadEstimate serialization."""
        estimate = PayloadEstimate(
            total_bytes=1000,
            messages_bytes=800,
            tools_bytes=150,
            overhead_bytes=50,
            system_bytes=100,
            user_bytes=300,
            assistant_bytes=300,
            tool_result_bytes=100,
            exceeds_limit=False,
            limit_bytes=4000000,
            utilization_pct=0.25,
        )

        result = estimate.to_dict()

        assert result["total_bytes"] == 1000
        assert result["utilization_pct"] == 25.0  # Converted to percentage
        assert "exceeds_limit" in result


class TestGroqProviderIntegration:
    """Test Groq provider with payload limiter integration."""

    def test_groq_provider_has_payload_limiter(self):
        """Test that GroqProvider initializes payload limiter."""
        from victor.providers.groq_provider import GroqProvider

        # Create provider (won't connect without API key)
        provider = GroqProvider(api_key="test-key")

        assert hasattr(provider, "_payload_limiter")
        assert isinstance(provider._payload_limiter, ProviderPayloadLimiter)
        assert provider._payload_limiter.provider_name == "groq"
        assert provider._payload_limiter.max_payload_bytes == 4 * 1024 * 1024
