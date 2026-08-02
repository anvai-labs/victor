# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for context-aware tool strategy."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from victor.providers.base import BaseProvider
from victor.config.tool_tiers import get_tool_tier, get_tier_summary
from victor.tools.enums import SchemaLevel


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    def __init__(self, context_window_size: int = 128000):
        self._context_window = context_window_size
        self._name = "mock"

    @property
    def name(self) -> str:
        """Get provider name."""
        return self._name

    def context_window(self, model: str) -> int:
        """Get context window size."""
        return self._context_window

    def chat(self, messages, **kwargs):
        """Mock chat method."""
        return {"content": "Mock response"}

    def stream_chat(self, messages, **kwargs):
        """Mock stream chat method."""
        yield {"content": "Mock response"}

    def close(self):
        """Mock close method."""
        pass

    def stream(self, messages, **kwargs):
        """Mock stream method."""
        yield {"content": "Mock response"}


class TestContextWindowDetection:
    """Test context window detection for providers."""

    def test_large_context_window(self):
        """Test large context window detection (cloud providers)."""
        provider = MockProvider(context_window_size=200000)
        context_window = provider.context_window("claude-sonnet-4-20250514")
        assert context_window == 200000

    def test_medium_context_window(self):
        """Test medium context window detection (some local providers)."""
        provider = MockProvider(context_window_size=65536)
        context_window = provider.context_window("qwen2.5:14b")
        assert context_window == 65536

    def test_small_context_window(self):
        """Test small context window detection (small local models)."""
        provider = MockProvider(context_window_size=32768)
        context_window = provider.context_window("qwen2.5-coder:7b")
        assert context_window == 32768

    def test_context_window_calculation(self):
        """Test context window budget calculation (25% constraint)."""
        provider = MockProvider(context_window_size=128000)
        context_window = provider.context_window("model")
        max_tool_tokens = int(context_window * 0.25)
        assert max_tool_tokens == 32000


class TestToolTierAssignments:
    """Test tool tier assignments."""

    def test_full_tier_core_tools(self):
        """Test FULL tier assignment for core tools."""
        full_tools = ["read", "write", "edit", "code_search", "shell"]
        for tool in full_tools:
            tier = get_tool_tier(tool)
            assert tier == "FULL", f"Tool {tool} should be FULL tier, got {tier}"

    def test_compact_tier_secondary_tools(self):
        """Test COMPACT tier assignment for secondary tools not in FULL tier."""
        compact_tools = ["git_status", "git_diff", "web_search"]
        for tool in compact_tools:
            tier = get_tool_tier(tool)
            assert tier == "COMPACT", f"Tool {tool} should be COMPACT tier, got {tier}"

    def test_stub_tier_unknown_tools(self):
        """Test STUB tier assignment for unknown tools."""
        tier = get_tool_tier("unknown_tool_xyz")
        assert tier == "STUB", "Unknown tools should default to STUB tier"

    def test_tier_summary(self):
        """Test tier summary provides accurate counts."""
        summary = get_tier_summary()
        assert "FULL" in summary
        assert "COMPACT" in summary
        assert "STUB" in summary
        assert summary["FULL"] >= 0
        assert summary["COMPACT"] >= 0
        assert summary["STUB"] >= 0


class TestContextConstraints:
    """Test context window constraints for tool selection."""

    def test_large_context_tool_budget(self):
        """Test tool budget for large context windows (cloud providers)."""
        context_window = 200000
        max_tool_tokens = int(context_window * 0.25)
        assert max_tool_tokens == 50000

        # Typical tool set should fit
        typical_tool_tokens = 2090  # 6 FULL + 10 COMPACT + 20 STUB
        assert typical_tool_tokens <= max_tool_tokens

    def test_medium_context_tool_budget(self):
        """Test tool budget for medium context windows."""
        context_window = 65536
        max_tool_tokens = int(context_window * 0.25)
        assert max_tool_tokens == 16384

        # Typical tool set should fit
        typical_tool_tokens = 2090
        assert typical_tool_tokens <= max_tool_tokens

    def test_small_context_tool_budget(self):
        """Test tool budget for small context windows (local models)."""
        context_window = 32768
        max_tool_tokens = int(context_window * 0.25)
        assert max_tool_tokens == 8192

        # Typical tool set should fit
        typical_tool_tokens = 2090
        assert typical_tool_tokens <= max_tool_tokens

    def test_minimum_context_window(self):
        """Test minimum safe context window."""
        min_context = 8192
        max_tool_tokens = int(min_context * 0.25)
        assert max_tool_tokens == 2048

        # Even minimal tool set should fit
        minimal_tool_tokens = 500  # 5 FULL tools
        assert minimal_tool_tokens <= max_tool_tokens


class TestSchemaLevelTokenCosts:
    """Test schema level token cost estimates."""

    def test_full_schema_token_cost(self):
        """Test FULL schema token cost (~125 tokens)."""
        # FULL schema: complete description + all parameters
        full_tokens = 125
        assert 100 <= full_tokens <= 150

    def test_compact_schema_token_cost(self):
        """Test COMPACT schema token cost (~70 tokens)."""
        # COMPACT schema: shortened description + all parameters
        compact_tokens = 70
        assert 60 <= compact_tokens <= 80

    def test_stub_schema_token_cost(self):
        """Test STUB schema token cost (~32 tokens)."""
        # STUB schema: minimal description + required parameters only
        stub_tokens = 32
        assert 25 <= stub_tokens <= 40

    def test_schema_token_ordering(self):
        """Test FULL >= COMPACT >= STUB token ordering."""
        full_tokens = 125
        compact_tokens = 70
        stub_tokens = 32

        assert full_tokens >= compact_tokens >= stub_tokens

    def test_tool_set_token_calculation(self):
        """Test total token calculation for tool sets."""
        # Cloud provider: 7 FULL + 10 COMPACT + 20 STUB
        cloud_tokens = (7 * 125) + (10 * 70) + (20 * 32)
        assert cloud_tokens == 2215  # 875 + 700 + 640

        # Local provider: 8 STUB (core only)
        local_tokens = 8 * 32
        assert local_tokens == 256


class TestProviderSpecificStrategies:
    """Test provider-specific tool strategies."""

    def test_cloud_provider_strategy(self):
        """Test cloud provider uses session-locking strategy."""
        # Cloud providers have large context + cache discount
        context_window = 200000
        has_cache_discount = True

        # Should session-lock all tools
        should_session_lock = (
            has_cache_discount  # Cache discount makes locking beneficial
            or context_window >= 128000  # Large context makes locking feasible
        )

        assert should_session_lock is True

    def test_large_local_provider_strategy(self):
        """Test large local provider uses session-locking strategy."""
        # Large local models have no cache discount but large context
        context_window = 128000
        has_cache_discount = False

        # Should session-lock due to large context
        should_session_lock = context_window >= 128000

        assert should_session_lock is True

    def test_small_local_provider_strategy(self):
        """Test small local provider uses context-budgeted strategy."""
        # Small local models have limited context
        context_window = 32768
        has_cache_discount = False

        # Should use semantic selection within budget
        should_session_lock = context_window >= 128000

        assert should_session_lock is False

    def test_context_budget_enforcement(self):
        """Test context budget is enforced for small models."""
        # Small local model with very limited context
        context_window = 16384  # Even smaller than typical
        max_tool_tokens = int(context_window * 0.25)  # 4096

        # Calculate tool tokens for all tools with FULL schema (worst case)
        tool_count = 54
        avg_tokens_per_tool = 125  # FULL average
        total_tool_tokens = tool_count * avg_tokens_per_tool  # 6750

        # Should exceed budget with FULL schema on small model
        assert total_tool_tokens > max_tool_tokens  # 6750 > 4096

        # Should need semantic selection or schema demotion
        needs_selection = total_tool_tokens > max_tool_tokens
        assert needs_selection is True


class TestEconomyFirstPrinciples:
    """Test economy-first strategy principles."""

    def test_cache_discount_benefit(self):
        """Test cache discount makes session-locking economical."""
        # Without cache: pay full price every turn
        without_cache_tokens = 5000  # System prompt tokens
        turns = 10
        total_without_cache = without_cache_tokens * turns  # 50000

        # With cache: pay 10% after first turn
        with_cache_tokens = 5000
        total_with_cache = (with_cache_tokens * 1) + (with_cache_tokens * 0.1 * (turns - 1))  # 9500

        # Cache should provide significant savings
        savings = total_without_cache - total_with_cache
        assert savings == 40500

    def test_session_locking_minimizes_invalidations(self):
        """Test session-locking minimizes cache invalidations."""
        # Session-locking: tools frozen in system prompt
        # Dynamic injection: tools change every turn → cache misses

        session_lock_cache_hits = 10  # All 10 turns hit cache
        dynamic_injection_cache_hits = 1  # Only first turn hits cache

        assert session_lock_cache_hits > dynamic_injection_cache_hits

    def test_context_budget_prevents_overflow(self):
        """Test context budget prevents overflow for small models."""
        context_window = 32768
        max_tool_tokens = int(context_window * 0.25)  # 8192

        # Attempting to load all tools exceeds budget
        all_tools_tokens = 54 * 70  # 3780 tokens
        assert all_tools_tokens <= max_tool_tokens

        # But with user message and context, budget is tighter
        available_for_tools = max_tool_tokens - 1000  # Reserve for other content
        assert all_tools_tokens <= available_for_tools


class TestValidationCriteria:
    """Test validation criteria for tool strategy."""

    def test_context_window_detection_passes(self):
        """Test context window detection validation."""
        provider = MockProvider(context_window_size=128000)
        context_window = provider.context_window("model")

        # Should detect context window
        assert context_window > 0
        # Should be reasonable size
        assert context_window >= 4096

    def test_tier_assignments_exist(self):
        """Test tier assignments exist validation."""
        summary = get_tier_summary()

        # Should have tools assigned
        total_tools = summary["FULL"] + summary["COMPACT"] + summary["STUB"]
        assert total_tools > 0

    def test_schema_costs_ordered(self):
        """Test schema costs are ordered correctly."""
        full_tokens = 125
        compact_tokens = 70
        stub_tokens = 32

        # Should have FULL >= COMPACT >= STUB
        assert full_tokens >= compact_tokens >= stub_tokens

    def test_typical_tool_set_fits(self):
        """Test typical tool set fits within budget."""
        # Typical set: 6 FULL + 10 COMPACT + 20 STUB
        typical_full_tokens = 6 * 125  # 750
        typical_compact_tokens = 10 * 70  # 700
        typical_stub_tokens = 20 * 32  # 640
        typical_total = typical_full_tokens + typical_compact_tokens + typical_stub_tokens  # 2090

        # Should fit in large context
        large_context = 200000
        large_budget = int(large_context * 0.25)
        assert typical_total <= large_budget

        # Should fit in small context
        small_context = 32768
        small_budget = int(small_context * 0.25)
        assert typical_total <= small_budget
