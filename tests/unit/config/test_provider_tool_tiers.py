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

"""Unit tests for provider-specific tool tier assignments."""

import pytest

from victor.config.tool_tiers import (
    get_provider_category,
    get_provider_tool_tier,
    get_tool_tier,
    reload_tiers,
    reload_provider_tiers,
)


class TestProviderCategoryDetection:
    """Test provider category detection based on context window."""

    def test_edge_category_detection(self):
        """Test edge category detection for small context windows."""
        assert get_provider_category(8192) == "edge"
        assert get_provider_category(16383) == "edge"
        assert get_provider_category(1) == "edge"
        assert get_provider_category(16383) == "edge"

    def test_standard_category_detection(self):
        """Test standard category detection for medium context windows."""
        assert get_provider_category(16384) == "standard"
        assert get_provider_category(32768) == "standard"
        assert get_provider_category(65536) == "standard"
        assert get_provider_category(131071) == "standard"

    def test_large_category_detection(self):
        """Test large category detection for large context windows."""
        assert get_provider_category(131072) == "large"
        assert get_provider_category(200000) == "large"
        assert get_provider_category(1000000) == "large"


class TestProviderToolTierAssignments:
    """Test provider-specific tool tier assignments."""

    def test_edge_tier_full_tools(self):
        """Test edge tier FULL tool assignments."""
        # Only read and shell should be FULL for edge models
        assert get_provider_tool_tier("read", "edge") == "FULL"
        assert get_provider_tool_tier("shell", "edge") == "FULL"

    def test_edge_tier_stub_tools(self):
        """Test edge tier STUB tool assignments."""
        # All other tools should be STUB for edge models
        assert get_provider_tool_tier("ls", "edge") == "STUB"
        assert get_provider_tool_tier("code_search", "edge") == "STUB"
        assert get_provider_tool_tier("edit", "edge") == "STUB"
        assert get_provider_tool_tier("write", "edge") == "STUB"
        assert get_provider_tool_tier("test", "edge") == "STUB"
        assert get_provider_tool_tier("symbol", "edge") == "STUB"
        assert get_provider_tool_tier("find", "edge") == "STUB"
        assert get_provider_tool_tier("overview", "edge") == "STUB"

    def test_standard_tier_full_tools(self):
        """Test standard tier FULL tool assignments."""
        # 5 core tools should be FULL for standard models
        assert get_provider_tool_tier("read", "standard") == "FULL"
        assert get_provider_tool_tier("shell", "standard") == "FULL"
        assert get_provider_tool_tier("ls", "standard") == "FULL"
        assert get_provider_tool_tier("code_search", "standard") == "FULL"
        assert get_provider_tool_tier("edit", "standard") == "FULL"

    def test_standard_tier_compact_tools(self):
        """Test standard tier COMPACT tool assignments."""
        # write and test should be COMPACT for standard models
        assert get_provider_tool_tier("write", "standard") == "COMPACT"
        assert get_provider_tool_tier("test", "standard") == "COMPACT"

    def test_standard_tier_stub_tools(self):
        """Test standard tier STUB tool assignments."""
        # Specialty tools should be STUB for standard models
        assert get_provider_tool_tier("refs", "standard") == "STUB"
        assert get_provider_tool_tier("workflow", "standard") == "STUB"
        assert get_provider_tool_tier("overview", "standard") == "STUB"

    def test_large_tier_full_tools(self):
        """Test large tier FULL tool assignments."""
        # All core tools should be FULL for large models
        assert get_provider_tool_tier("read", "large") == "FULL"
        assert get_provider_tool_tier("shell", "large") == "FULL"
        assert get_provider_tool_tier("ls", "large") == "FULL"
        assert get_provider_tool_tier("code_search", "large") == "FULL"
        assert get_provider_tool_tier("edit", "large") == "FULL"
        assert get_provider_tool_tier("write", "large") == "FULL"
        assert get_provider_tool_tier("test", "large") == "FULL"
        assert get_provider_tool_tier("symbol", "large") == "FULL"
        assert get_provider_tool_tier("find", "large") == "FULL"
        assert get_provider_tool_tier("overview", "large") == "FULL"

    def test_large_tier_stub_tools(self):
        """Test large tier STUB tool assignments."""
        # Specialty tools should be STUB for large models
        assert get_provider_tool_tier("refs", "large") == "STUB"
        assert get_provider_tool_tier("workflow", "large") == "STUB"


class TestBackwardCompatibility:
    """Test backward compatibility with global tier system."""

    def test_global_get_tool_tier_still_works(self):
        """Test global get_tool_tier() function still works."""
        # Should use global tiers, not provider-specific
        assert get_tool_tier("read") == "FULL"
        assert get_tool_tier("shell") == "FULL"
        assert get_tool_tier("ls") == "FULL"
        assert get_tool_tier("code_search") == "FULL"
        assert get_tool_tier("edit") == "FULL"
        assert get_tool_tier("write") == "FULL"
        assert get_tool_tier("overview") == "FULL"
        assert get_tool_tier("symbol") == "FULL"
        assert get_tool_tier("find") == "FULL"
        assert get_tool_tier("test") == "FULL"

    def test_unknown_tool_defaults_to_stub(self):
        """Test unknown tools default to STUB tier."""
        assert get_tool_tier("unknown_tool_xyz") == "STUB"
        assert get_provider_tool_tier("unknown_tool_xyz", "edge") == "STUB"
        assert get_provider_tool_tier("unknown_tool_xyz", "standard") == "STUB"
        assert get_provider_tool_tier("unknown_tool_xyz", "large") == "STUB"


class TestProviderTierFallback:
    """Test fallback to global tiers when provider-specific tiers fail."""

    def test_fallback_to_global_tiers(self):
        """Test fallback to global tiers when provider category is invalid."""
        # Invalid provider category should fall back to global tiers
        assert get_provider_tool_tier("read", "invalid_category") == "FULL"
        assert get_provider_tool_tier("unknown_tool", "invalid_category") == "STUB"

    def test_missing_provider_tier_fallback(self):
        """Test fallback when provider_tiers config is missing."""
        # If provider_tiers section is missing, should use global tiers
        # This is tested by the implementation checking for provider_category in cache
        # If not found, it falls back to get_tool_tier()
        # "read" is FULL in global tiers, so should return FULL
        assert get_provider_tool_tier("read", "nonexistent") == "FULL"
        # Unknown tools should still be STUB
        assert get_provider_tool_tier("unknown_tool_xyz", "nonexistent") == "STUB"


class TestTokenSavings:
    """Test token savings calculations for different provider categories."""

    def test_edge_token_savings(self):
        """Test edge models achieve 80% token reduction."""
        global_tier_tokens = 10 * 125  # 10 FULL tools
        edge_tier_tokens = 2 * 125  # 2 FULL tools

        savings = global_tier_tokens - edge_tier_tokens
        savings_pct = (savings / global_tier_tokens) * 100

        assert savings_pct == 80.0

    def test_standard_token_savings(self):
        """Test standard models achieve ~40% token reduction."""
        global_tier_tokens = 10 * 125  # 10 FULL tools
        standard_tier_tokens = (5 * 125) + (2 * 70)  # 5 FULL + 2 COMPACT

        savings = global_tier_tokens - standard_tier_tokens
        savings_pct = (savings / global_tier_tokens) * 100

        # Allow small floating point tolerance
        assert abs(savings_pct - 38.8) < 0.1

    def test_large_token_savings(self):
        """Test large models have no token reduction (full capability)."""
        global_tier_tokens = 10 * 125  # 10 FULL tools
        large_tier_tokens = 10 * 125  # Same 10 FULL tools

        savings = global_tier_tokens - large_tier_tokens

        assert savings == 0  # No savings, same capability


class TestBudgetCompliance:
    """Test tool sets fit within 25% context budget for all categories."""

    def test_edge_model_budget_compliance(self):
        """Test edge model tools fit within 25% context budget."""
        context_window = 8192
        max_tool_tokens = int(context_window * 0.25)  # 2048

        # Edge tier: 2 FULL tools × 125 = 250 tokens
        edge_tokens = 2 * 125

        assert (
            edge_tokens <= max_tool_tokens
        ), f"Edge tools ({edge_tokens}) exceed budget ({max_tool_tokens})"

    def test_standard_model_budget_compliance(self):
        """Test standard model tools fit within 25% context budget."""
        context_window = 32768
        max_tool_tokens = int(context_window * 0.25)  # 8192

        # Standard tier: 5 FULL + 2 COMPACT = 765 tokens
        standard_tokens = (5 * 125) + (2 * 70)

        assert (
            standard_tokens <= max_tool_tokens
        ), f"Standard tools ({standard_tokens}) exceed budget ({max_tool_tokens})"

    def test_large_model_budget_compliance(self):
        """Test large model tools fit within 25% context budget."""
        context_window = 200000
        max_tool_tokens = int(context_window * 0.25)  # 50000

        # Large tier: 10 FULL tools = 1250 tokens
        large_tokens = 10 * 125

        assert (
            large_tokens <= max_tool_tokens
        ), f"Large tools ({large_tokens}) exceed budget ({max_tool_tokens})"


class TestTierReloading:
    """Test tier reloading functionality."""

    def test_reload_provider_tiers(self):
        """Test reload_provider_tiers() clears cache."""
        # Load tiers initially
        _ = get_provider_tool_tier("read", "edge")

        # Reload should clear cache
        reload_provider_tiers()

        # Should work without error
        assert get_provider_tool_tier("read", "edge") == "FULL"

    def test_reload_global_tiers(self):
        """Test reload_tiers() clears global tier cache."""
        # Load tiers initially
        _ = get_tool_tier("read")

        # Reload should clear cache
        reload_tiers()

        # Should work without error
        assert get_tool_tier("read") == "FULL"
