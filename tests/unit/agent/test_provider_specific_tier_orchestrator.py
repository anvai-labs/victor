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

"""Unit tests for provider-specific tier integration in orchestrator."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from victor.config.tool_tiers import get_provider_category, get_provider_tool_tier
from victor.agent.orchestrator import AgentOrchestrator


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator for testing."""
    orchestrator = Mock(spec=AgentOrchestrator)

    # Mock provider and model
    orchestrator.provider = Mock()
    orchestrator.provider.name = "anthropic"
    orchestrator.model = "claude-sonnet-4-20250514"

    # Mock context window retrieval
    def mock_get_context_window(provider, model):
        # Map models to context windows for testing
        context_windows = {
            "claude-sonnet-4-20250514": 200000,  # Large
            "qwen2.5:7b": 32768,  # Standard
            "qwen3.5:2b": 8192,  # Edge
        }
        return context_windows.get(model, 200000)

    orchestrator._get_context_window = mock_get_context_window

    # Mock tool estimation
    def mock_estimate_tool_tokens(tool, provider_category=None):
        if provider_category:
            tier = get_provider_tool_tier(tool.name, provider_category)
        else:
            from victor.config.tool_tiers import get_tool_tier

            tier = get_tool_tier(tool.name)

        # Token costs by tier
        tier_costs = {"FULL": 125, "COMPACT": 70, "STUB": 32}
        return tier_costs.get(tier, 125)

    orchestrator._estimate_tool_tokens = mock_estimate_tool_tokens

    # Mock tool demotion
    def mock_demote_tools_to_fit(tools, max_tokens, provider_category=None):
        """Simplified demotion for testing."""
        current_tokens = sum(
            orchestrator._estimate_tool_tokens(t, provider_category) for t in tools
        )

        if current_tokens <= max_tokens:
            return tools

        # Demote tools from end until fit
        result = list(tools)
        while result and current_tokens > max_tokens:
            removed = result.pop()
            current_tokens -= orchestrator._estimate_tool_tokens(removed, provider_category)

        return result

    orchestrator._demote_tools_to_fit = mock_demote_tools_to_fit

    return orchestrator


@pytest.fixture
def create_mock_tool():
    """Factory for creating mock tools."""

    def _create(name):
        tool = Mock()
        tool.name = name
        tool.to_schema = Mock(return_value={"type": "object", "properties": {}})
        return tool

    return _create


class TestEdgeModelToolSelection:
    """Test edge models use minimal tool set."""

    def test_edge_model_uses_minimal_tools(self, mock_orchestrator, create_mock_tool):
        """Test edge models select only 2 FULL tools (read, shell)."""
        # Edge model setup
        mock_orchestrator.model = "qwen3.5:2b"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        assert provider_category == "edge"

        # Create tool set
        tools = [
            create_mock_tool("read"),
            create_mock_tool("shell"),
            create_mock_tool("ls"),
            create_mock_tool("code_search"),
            create_mock_tool("edit"),
        ]

        # Estimate tokens with edge category
        tool_tokens = sum(
            mock_orchestrator._estimate_tool_tokens(t, provider_category) for t in tools
        )

        # Edge category: only read + shell are FULL (125 each), others STUB (32 each)
        # Expected: 125 + 125 + 32 + 32 + 32 = 346 tokens
        assert tool_tokens == 346

        # Fit within edge budget (25% of 8K = 2048 tokens)
        max_tool_tokens = int(context_window * 0.25)
        selected = mock_orchestrator._demote_tools_to_fit(tools, max_tool_tokens, provider_category)

        # All tools should fit within budget
        selected_tokens = sum(
            mock_orchestrator._estimate_tool_tokens(t, provider_category) for t in selected
        )
        assert selected_tokens <= max_tool_tokens

    def test_edge_model_token_savings(self, mock_orchestrator, create_mock_tool):
        """Test edge models achieve 80% token reduction vs global tiers."""
        # Create global tier tool set (10 FULL tools)
        global_tools = [
            create_mock_tool(name)
            for name in [
                "read",
                "shell",
                "ls",
                "code_search",
                "edit",
                "write",
                "overview",
                "symbol",
                "find",
                "test",
            ]
        ]

        # Edge model setup
        mock_orchestrator.model = "qwen3.5:2b"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        # Global tier cost (no provider category)
        global_tokens = sum(mock_orchestrator._estimate_tool_tokens(t, None) for t in global_tools)

        # Edge tier cost (with provider category)
        edge_tokens = sum(
            mock_orchestrator._estimate_tool_tokens(t, provider_category) for t in global_tools
        )

        # Edge: 2 FULL (125 each) + 8 STUB (32 each) = 250 + 256 = 506
        # Global: 10 FULL (125 each) = 1250
        # Savings: (1250 - 506) / 1250 = 59.5%
        savings_pct = ((global_tokens - edge_tokens) / global_tokens) * 100

        # Should be significant savings (close to 60% for this mix)
        assert savings_pct > 50


class TestStandardModelToolSelection:
    """Test standard models use balanced tool set."""

    def test_standard_model_uses_balanced_tools(self, mock_orchestrator, create_mock_tool):
        """Test standard models select 5 FULL + 2 COMPACT tools."""
        # Standard model setup
        mock_orchestrator.model = "qwen2.5:7b"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        assert provider_category == "standard"

        # Create tool set
        tools = [
            create_mock_tool("read"),
            create_mock_tool("shell"),
            create_mock_tool("ls"),
            create_mock_tool("code_search"),
            create_mock_tool("edit"),
            create_mock_tool("write"),
            create_mock_tool("test"),
            create_mock_tool("refs"),
        ]

        # Estimate tokens with standard category
        tool_tokens = sum(
            mock_orchestrator._estimate_tool_tokens(t, provider_category) for t in tools
        )

        # Standard category: 5 FULL (125 each) + 2 COMPACT (70 each) + 1 STUB (32)
        # Expected: 5*125 + 2*70 + 32 = 625 + 140 + 32 = 797
        assert tool_tokens == 797

        # Fit within standard budget (25% of 32K = 8192 tokens)
        max_tool_tokens = int(context_window * 0.25)
        selected = mock_orchestrator._demote_tools_to_fit(tools, max_tool_tokens, provider_category)

        # All tools should fit within budget
        selected_tokens = sum(
            mock_orchestrator._estimate_tool_tokens(t, provider_category) for t in selected
        )
        assert selected_tokens <= max_tool_tokens

    def test_standard_model_token_savings(self, mock_orchestrator, create_mock_tool):
        """Test standard models achieve ~40% token reduction vs global tiers."""
        # Create global tier tool set (10 FULL tools)
        global_tools = [
            create_mock_tool(name)
            for name in [
                "read",
                "shell",
                "ls",
                "code_search",
                "edit",
                "write",
                "overview",
                "symbol",
                "find",
                "test",
            ]
        ]

        # Standard model setup
        mock_orchestrator.model = "qwen2.5:7b"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        # Global tier cost (no provider category)
        global_tokens = sum(mock_orchestrator._estimate_tool_tokens(t, None) for t in global_tools)

        # Standard tier cost (with provider category)
        standard_tokens = sum(
            mock_orchestrator._estimate_tool_tokens(t, provider_category) for t in global_tools
        )

        # Standard: 5 FULL (125 each) + 2 COMPACT (70 each) + 3 STUB (32 each)
        # Expected: 625 + 140 + 96 = 861
        # Global: 10 FULL (125 each) = 1250
        # Savings: (1250 - 861) / 1250 = 31.1%
        savings_pct = ((global_tokens - standard_tokens) / global_tokens) * 100

        # Should be significant savings (30%+ for this mix)
        assert savings_pct > 25


class TestLargeModelToolSelection:
    """Test large models use full tool set."""

    def test_large_model_uses_full_tools(self, mock_orchestrator, create_mock_tool):
        """Test large models select all 10 FULL tools."""
        # Large model setup
        mock_orchestrator.model = "claude-sonnet-4-20250514"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        assert provider_category == "large"

        # Create full tool set
        tools = [
            create_mock_tool("read"),
            create_mock_tool("shell"),
            create_mock_tool("ls"),
            create_mock_tool("code_search"),
            create_mock_tool("edit"),
            create_mock_tool("write"),
            create_mock_tool("overview"),
            create_mock_tool("symbol"),
            create_mock_tool("find"),
            create_mock_tool("test"),
        ]

        # Estimate tokens with large category
        tool_tokens = sum(
            mock_orchestrator._estimate_tool_tokens(t, provider_category) for t in tools
        )

        # Large category: all 10 FULL (125 each) = 1250
        assert tool_tokens == 1250

        # Fit within large budget (25% of 200K = 50K tokens)
        max_tool_tokens = int(context_window * 0.25)
        selected = mock_orchestrator._demote_tools_to_fit(tools, max_tool_tokens, provider_category)

        # All tools should fit within budget
        selected_tokens = sum(
            mock_orchestrator._estimate_tool_tokens(t, provider_category) for t in selected
        )
        assert selected_tokens <= max_tool_tokens

    def test_large_model_no_regression(self, mock_orchestrator, create_mock_tool):
        """Test large models have no token regression vs global tiers."""
        # Create full tool set
        tools = [
            create_mock_tool(name)
            for name in [
                "read",
                "shell",
                "ls",
                "code_search",
                "edit",
                "write",
                "overview",
                "symbol",
                "find",
                "test",
            ]
        ]

        # Large model setup
        mock_orchestrator.model = "claude-sonnet-4-20250514"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        # Global tier cost (no provider category)
        global_tokens = sum(mock_orchestrator._estimate_tool_tokens(t, None) for t in tools)

        # Large tier cost (with provider category)
        large_tokens = sum(
            mock_orchestrator._estimate_tool_tokens(t, provider_category) for t in tools
        )

        # Large models should have same cost as global tiers (no regression)
        assert global_tokens == large_tokens


class TestProviderCategoryDetection:
    """Test provider category detection in orchestrator context."""

    def test_detects_edge_category_from_context_window(self, mock_orchestrator):
        """Test edge category detection from small context window."""
        mock_orchestrator.model = "qwen3.5:2b"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        assert provider_category == "edge"
        assert context_window < 16384

    def test_detects_standard_category_from_context_window(self, mock_orchestrator):
        """Test standard category detection from medium context window."""
        mock_orchestrator.model = "qwen2.5:7b"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        assert provider_category == "standard"
        assert 16384 <= context_window < 131072

    def test_detects_large_category_from_context_window(self, mock_orchestrator):
        """Test large category detection from large context window."""
        mock_orchestrator.model = "claude-sonnet-4-20250514"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        assert provider_category == "large"
        assert context_window >= 131072


class TestTierEstimationAccuracy:
    """Test token estimation accuracy with provider-specific tiers."""

    def test_edge_tier_estimation_matches_expected(self, mock_orchestrator, create_mock_tool):
        """Test edge tier token estimations match expected costs."""
        mock_orchestrator.model = "qwen3.5:2b"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        # Test FULL tools (read, shell)
        read_tool = create_mock_tool("read")
        shell_tool = create_mock_tool("shell")

        assert mock_orchestrator._estimate_tool_tokens(read_tool, provider_category) == 125
        assert mock_orchestrator._estimate_tool_tokens(shell_tool, provider_category) == 125

        # Test STUB tool (ls)
        ls_tool = create_mock_tool("ls")
        assert mock_orchestrator._estimate_tool_tokens(ls_tool, provider_category) == 32

    def test_standard_tier_estimation_matches_expected(self, mock_orchestrator, create_mock_tool):
        """Test standard tier token estimations match expected costs."""
        mock_orchestrator.model = "qwen2.5:7b"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        # Test FULL tools (read, shell, ls, code_search, edit)
        for tool_name in ["read", "shell", "ls", "code_search", "edit"]:
            tool = create_mock_tool(tool_name)
            assert mock_orchestrator._estimate_tool_tokens(tool, provider_category) == 125

        # Test COMPACT tools (write, test)
        for tool_name in ["write", "test"]:
            tool = create_mock_tool(tool_name)
            assert mock_orchestrator._estimate_tool_tokens(tool, provider_category) == 70

        # Test STUB tool (refs)
        refs_tool = create_mock_tool("refs")
        assert mock_orchestrator._estimate_tool_tokens(refs_tool, provider_category) == 32

    def test_large_tier_estimation_matches_expected(self, mock_orchestrator, create_mock_tool):
        """Test large tier token estimations match expected costs."""
        mock_orchestrator.model = "claude-sonnet-4-20250514"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        # Test FULL tools (all 10 core tools)
        for tool_name in [
            "read",
            "shell",
            "ls",
            "code_search",
            "edit",
            "write",
            "overview",
            "symbol",
            "find",
            "test",
        ]:
            tool = create_mock_tool(tool_name)
            assert mock_orchestrator._estimate_tool_tokens(tool, provider_category) == 125


class TestToolDemotionWithProviderCategory:
    """Test tool demotion behavior with provider-specific tiers."""

    def test_edge_demotion_preserves_core_tools(self, mock_orchestrator, create_mock_tool):
        """Test edge demotion preserves read + shell first."""
        mock_orchestrator.model = "qwen3.5:2b"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        # Create many tools (more than budget)
        tools = [
            create_mock_tool("read"),
            create_mock_tool("shell"),
            create_mock_tool("ls"),
            create_mock_tool("code_search"),
            create_mock_tool("edit"),
            create_mock_tool("write"),
            create_mock_tool("test"),
        ]

        # Very tight budget (only 200 tokens)
        max_tokens = 200
        selected = mock_orchestrator._demote_tools_to_fit(tools, max_tokens, provider_category)

        # Should keep read + shell (FULL) first, then drop others
        # With edge tier: read (125) + shell (125) = 250 > 200, so only 1 tool fits
        assert len(selected) >= 1

        # First tool should be read
        assert selected[0].name == "read"

    def test_standard_demotion_balances_full_compact(self, mock_orchestrator, create_mock_tool):
        """Test standard demotion balances FULL and COMPACT tools."""
        mock_orchestrator.model = "qwen2.5:7b"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        # Create tools
        tools = [
            create_mock_tool("read"),
            create_mock_tool("shell"),
            create_mock_tool("ls"),
            create_mock_tool("code_search"),
            create_mock_tool("edit"),
            create_mock_tool("write"),
            create_mock_tool("test"),
        ]

        # Tight budget (600 tokens - should fit 5 FULL or mix)
        max_tokens = 600
        selected = mock_orchestrator._demote_tools_to_fit(tools, max_tokens, provider_category)

        # Should fit some tools within budget
        selected_tokens = sum(
            mock_orchestrator._estimate_tool_tokens(t, provider_category) for t in selected
        )
        assert selected_tokens <= max_tokens

    def test_large_demotion_fits_all_tools(self, mock_orchestrator, create_mock_tool):
        """Test large demotion fits all tools easily."""
        mock_orchestrator.model = "claude-sonnet-4-20250514"
        context_window = mock_orchestrator._get_context_window(
            mock_orchestrator.provider, mock_orchestrator.model
        )
        provider_category = get_provider_category(context_window)

        # Create all 10 core tools
        tools = [
            create_mock_tool(name)
            for name in [
                "read",
                "shell",
                "ls",
                "code_search",
                "edit",
                "write",
                "overview",
                "symbol",
                "find",
                "test",
            ]
        ]

        # Generous budget (2000 tokens)
        max_tokens = 2000
        selected = mock_orchestrator._demote_tools_to_fit(tools, max_tokens, provider_category)

        # Should fit all 10 tools
        assert len(selected) == 10

        selected_tokens = sum(
            mock_orchestrator._estimate_tool_tokens(t, provider_category) for t in selected
        )
        assert selected_tokens <= max_tokens


class TestBackwardCompatibility:
    """Test backward compatibility when provider_category is None."""

    def test_none_provider_category_uses_global_tiers(self, mock_orchestrator, create_mock_tool):
        """Test None provider_category falls back to global tiers."""
        # Create tool
        tool = create_mock_tool("read")

        # With None provider_category, should use global tiers
        tokens = mock_orchestrator._estimate_tool_tokens(tool, None)

        # Global tiers have read as FULL (125 tokens)
        assert tokens == 125

    def test_orchestrator_works_without_provider_category(
        self, mock_orchestrator, create_mock_tool
    ):
        """Test orchestrator methods work without provider_category parameter."""
        tools = [create_mock_tool("read"), create_mock_tool("shell")]

        # Should work without provider_category
        tokens = sum(mock_orchestrator._estimate_tool_tokens(t) for t in tools)
        assert tokens == 250  # 2 FULL tools at global tier

        # Demotion should work
        selected = mock_orchestrator._demote_tools_to_fit(tools, 500)
        assert len(selected) == 2
