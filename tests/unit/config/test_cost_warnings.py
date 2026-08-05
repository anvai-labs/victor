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

"""Tests for cost-aware tool selection with user warnings."""

import pytest
from unittest.mock import MagicMock

from victor.tools.base import CostTier
from victor.tools.registry import ToolRegistry
from victor.tools.semantic_selector import (
    SemanticToolSelector,
    COST_TIER_WARNINGS,
)


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str, description: str = "Test tool"):
        self.name = name
        self.description = description
        self.parameters = {"type": "object", "properties": {}}


class TestCostTierWarnings:
    """Tests for COST_TIER_WARNINGS constants."""

    def test_high_cost_warning_exists(self):
        """Test that HIGH cost tier has a warning message."""
        assert CostTier.HIGH in COST_TIER_WARNINGS
        assert "HIGH COST" in COST_TIER_WARNINGS[CostTier.HIGH]

    def test_medium_cost_warning_exists(self):
        """Test that MEDIUM cost tier has a warning message."""
        assert CostTier.MEDIUM in COST_TIER_WARNINGS
        assert "MEDIUM COST" in COST_TIER_WARNINGS[CostTier.MEDIUM]

    def test_free_and_low_no_warning(self):
        """Test that FREE and LOW tiers don't have warnings."""
        assert CostTier.FREE not in COST_TIER_WARNINGS
        assert CostTier.LOW not in COST_TIER_WARNINGS


class TestSemanticToolSelectorCostWarnings:
    """Tests for SemanticToolSelector cost warning methods."""

    def test_init_creates_empty_warnings(self):
        """Test that initialization creates empty warnings list."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
        )
        assert selector._last_cost_warnings == []

    def test_get_last_cost_warnings_returns_copy(self):
        """Test that get_last_cost_warnings returns a copy."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
        )
        selector._last_cost_warnings = ["warning1", "warning2"]

        warnings = selector.get_last_cost_warnings()
        assert warnings == ["warning1", "warning2"]

        # Modifying returned list shouldn't affect internal state
        warnings.append("warning3")
        assert len(selector._last_cost_warnings) == 2

    def test_clear_cost_warnings(self):
        """Test that clear_cost_warnings empties the list."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
        )
        selector._last_cost_warnings = ["warning1", "warning2"]

        selector.clear_cost_warnings()
        assert selector._last_cost_warnings == []

    def test_generate_cost_warnings_with_high_cost_tool(self):
        """Test warning generation for high-cost tools."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
            cost_aware_selection=True,
        )

        # Mock registry with high-cost tool
        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_tool_cost.return_value = CostTier.HIGH

        high_cost_tool = MockTool("web_search")
        selected_tools = [(high_cost_tool, 0.85)]

        warnings = selector._generate_cost_warnings(selected_tools, mock_registry)

        assert len(warnings) == 1
        assert "web_search" in warnings[0]
        assert "HIGH COST" in warnings[0]

    def test_generate_cost_warnings_with_medium_cost_tool(self):
        """Test warning generation for medium-cost tools."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
            cost_aware_selection=True,
        )

        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_tool_cost.return_value = CostTier.MEDIUM

        medium_cost_tool = MockTool("code_review")
        selected_tools = [(medium_cost_tool, 0.75)]

        warnings = selector._generate_cost_warnings(selected_tools, mock_registry)

        assert len(warnings) == 1
        assert "code_review" in warnings[0]
        assert "MEDIUM COST" in warnings[0]

    def test_generate_cost_warnings_no_warning_for_low_cost(self):
        """Test that no warning is generated for LOW cost tools."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
            cost_aware_selection=True,
        )

        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_tool_cost.return_value = CostTier.LOW

        low_cost_tool = MockTool("read_file")
        selected_tools = [(low_cost_tool, 0.90)]

        warnings = selector._generate_cost_warnings(selected_tools, mock_registry)

        assert len(warnings) == 0

    def test_generate_cost_warnings_no_warning_for_free(self):
        """Test that no warning is generated for FREE cost tools."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
            cost_aware_selection=True,
        )

        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_tool_cost.return_value = CostTier.FREE

        free_tool = MockTool("list_directory")
        selected_tools = [(free_tool, 0.95)]

        warnings = selector._generate_cost_warnings(selected_tools, mock_registry)

        assert len(warnings) == 0

    def test_generate_cost_warnings_disabled(self):
        """Test that no warnings when cost_aware_selection is False."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
            cost_aware_selection=False,  # Disabled
        )

        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_tool_cost.return_value = CostTier.HIGH

        high_cost_tool = MockTool("web_search")
        selected_tools = [(high_cost_tool, 0.85)]

        warnings = selector._generate_cost_warnings(selected_tools, mock_registry)

        assert len(warnings) == 0

    def test_generate_cost_warnings_multiple_tools(self):
        """Test warning generation for multiple tools of different costs."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
            cost_aware_selection=True,
        )

        mock_registry = MagicMock(spec=ToolRegistry)

        # Define cost tiers for different tools
        def get_cost(tool_name):
            costs = {
                "web_search": CostTier.HIGH,
                "code_review": CostTier.MEDIUM,
                "read_file": CostTier.LOW,
                "list_directory": CostTier.FREE,
            }
            return costs.get(tool_name, CostTier.FREE)

        mock_registry.get_tool_cost.side_effect = get_cost

        selected_tools = [
            (MockTool("web_search"), 0.85),
            (MockTool("code_review"), 0.75),
            (MockTool("read_file"), 0.90),
            (MockTool("list_directory"), 0.95),
        ]

        warnings = selector._generate_cost_warnings(selected_tools, mock_registry)

        # Should have 2 warnings (HIGH and MEDIUM)
        assert len(warnings) == 2
        assert any("web_search" in w for w in warnings)
        assert any("code_review" in w for w in warnings)

    def test_generate_cost_warnings_none_cost_tier(self):
        """Test handling of tools with no cost tier."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
            cost_aware_selection=True,
        )

        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_tool_cost.return_value = None  # No cost tier

        unknown_tool = MockTool("unknown_tool")
        selected_tools = [(unknown_tool, 0.80)]

        warnings = selector._generate_cost_warnings(selected_tools, mock_registry)

        assert len(warnings) == 0


class TestCostPenalty:
    """Tests for cost penalty calculation."""

    def test_cost_penalty_high_tier(self):
        """Test cost penalty for HIGH tier."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
            cost_aware_selection=True,
            cost_penalty_factor=0.05,
        )

        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_tool_cost.return_value = CostTier.HIGH

        tool = MockTool("web_search")
        penalty = selector._get_cost_penalty(tool, mock_registry)

        # HIGH weight is 3, factor is 0.05, so penalty = 3 * 0.05 = 0.15
        assert penalty == pytest.approx(0.15)

    def test_cost_penalty_medium_tier(self):
        """Test cost penalty for MEDIUM tier."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
            cost_aware_selection=True,
            cost_penalty_factor=0.05,
        )

        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_tool_cost.return_value = CostTier.MEDIUM

        tool = MockTool("code_review")
        penalty = selector._get_cost_penalty(tool, mock_registry)

        # MEDIUM weight is 2, factor is 0.05, so penalty = 2 * 0.05 = 0.10
        assert penalty == 0.10

    def test_cost_penalty_low_tier(self):
        """Test cost penalty for LOW tier."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
            cost_aware_selection=True,
            cost_penalty_factor=0.05,
        )

        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_tool_cost.return_value = CostTier.LOW

        tool = MockTool("read_file")
        penalty = selector._get_cost_penalty(tool, mock_registry)

        # LOW weight is 1, factor is 0.05, so penalty = 1 * 0.05 = 0.05
        assert penalty == 0.05

    def test_cost_penalty_free_tier(self):
        """Test cost penalty for FREE tier is zero."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
            cost_aware_selection=True,
            cost_penalty_factor=0.05,
        )

        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_tool_cost.return_value = CostTier.FREE

        tool = MockTool("list_directory")
        penalty = selector._get_cost_penalty(tool, mock_registry)

        # FREE weight is 0, so penalty = 0 * 0.05 = 0.0
        assert penalty == 0.0

    def test_cost_penalty_disabled(self):
        """Test cost penalty returns 0 when disabled."""
        selector = SemanticToolSelector(
            embedding_provider="sentence-transformers",
            cache_embeddings=False,
            cost_aware_selection=False,  # Disabled
            cost_penalty_factor=0.05,
        )

        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry.get_tool_cost.return_value = CostTier.HIGH

        tool = MockTool("web_search")
        penalty = selector._get_cost_penalty(tool, mock_registry)

        assert penalty == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
