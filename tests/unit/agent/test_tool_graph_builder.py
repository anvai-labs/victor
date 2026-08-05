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

"""Unit tests for ToolGraphBuilder component.

Tests the SRP-compliant tool dependency graph building functionality.
"""

import pytest
from unittest.mock import MagicMock

from victor.agent.tool_graph_builder import (
    ToolGraphBuilder,
    ToolGraphConfig,
    GraphBuildResult,
)
from victor.tools.base import CostTier


class TestToolGraphConfig:
    """Tests for ToolGraphConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ToolGraphConfig()

        assert config.enabled is True
        assert config.include_cost_tiers is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ToolGraphConfig(
            enabled=False,
            include_cost_tiers=False,
        )

        assert config.enabled is False
        assert config.include_cost_tiers is False


class TestGraphBuildResult:
    """Tests for GraphBuildResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = GraphBuildResult()

        assert result.tools_registered == 0
        assert result.errors == []

    def test_custom_values(self):
        """Test custom result values."""
        result = GraphBuildResult(
            tools_registered=8,
            errors=["build error"],
        )

        assert result.tools_registered == 8
        assert result.errors == ["build error"]


class TestToolGraphBuilderInit:
    """Tests for ToolGraphBuilder initialization."""

    def test_initialization_with_defaults(self):
        """Test initialization with default config."""
        registry = MagicMock()

        builder = ToolGraphBuilder(registry=registry)

        assert builder._registry is registry
        assert builder._tool_graph is None
        assert builder._config.enabled is True
        assert builder.is_built is False

    def test_initialization_with_tool_graph(self):
        """Test initialization with tool graph."""
        registry = MagicMock()
        tool_graph = MagicMock()

        builder = ToolGraphBuilder(registry=registry, tool_graph=tool_graph)

        assert builder._tool_graph is tool_graph
        assert builder.tool_graph is tool_graph

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config."""
        registry = MagicMock()
        config = ToolGraphConfig(enabled=False)

        builder = ToolGraphBuilder(registry=registry, config=config)

        assert builder._config.enabled is False


class TestToolGraphBuilderBuild:
    """Tests for ToolGraphBuilder.build() method."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock tool registry."""
        return MagicMock()

    @pytest.fixture
    def mock_tool_graph(self):
        """Create mock tool graph."""
        graph = MagicMock()
        graph.add_tool.return_value = None
        return graph

    def test_build_disabled_returns_empty_result(self, mock_registry):
        """Test that build() returns empty result when disabled."""
        config = ToolGraphConfig(enabled=False)
        builder = ToolGraphBuilder(registry=mock_registry, config=config)

        result = builder.build()

        assert result.tools_registered == 0
        assert builder.is_built is True

    def test_build_no_graph_returns_empty_result(self, mock_registry):
        """Test that build() returns empty result when no graph provided."""
        builder = ToolGraphBuilder(registry=mock_registry)

        result = builder.build()

        assert result.tools_registered == 0
        assert builder.is_built is True

    def test_build_registers_default_tools(self, mock_registry, mock_tool_graph):
        """Test that build() registers default tool dependencies."""
        builder = ToolGraphBuilder(registry=mock_registry, tool_graph=mock_tool_graph)

        result = builder.build()

        # Should register 8 tools (code_search, semantic_code_search, read_file,
        # analyze_docs, code_review, generate_docs, security_scan, analyze_metrics)
        assert result.tools_registered == 8
        assert mock_tool_graph.add_tool.call_count == 8
        assert builder.is_built is True

    def test_build_includes_cost_tiers(self, mock_registry, mock_tool_graph):
        """Test that build() includes cost tiers in tool registration."""
        builder = ToolGraphBuilder(registry=mock_registry, tool_graph=mock_tool_graph)

        builder.build()

        # Check that add_tool was called with cost_tier parameter
        calls = mock_tool_graph.add_tool.call_args_list

        # First call should be code_search with FREE tier
        first_call = calls[0]
        assert first_call.kwargs.get("cost_tier") == CostTier.FREE

    def test_build_without_cost_tiers(self, mock_registry, mock_tool_graph):
        """Test that build() can exclude cost tiers."""
        config = ToolGraphConfig(include_cost_tiers=False)
        builder = ToolGraphBuilder(
            registry=mock_registry, tool_graph=mock_tool_graph, config=config
        )

        builder.build()

        # Check that add_tool was called without cost_tier
        calls = mock_tool_graph.add_tool.call_args_list
        first_call = calls[0]
        assert "cost_tier" not in first_call.kwargs

    def test_build_handles_errors_gracefully(self, mock_registry, mock_tool_graph):
        """Test that build() handles errors gracefully."""
        mock_tool_graph.add_tool.side_effect = Exception("Graph error")

        builder = ToolGraphBuilder(registry=mock_registry, tool_graph=mock_tool_graph)

        result = builder.build()

        # Should have 0 tools registered due to errors
        assert result.tools_registered == 0
        assert builder.is_built is True


class TestToolGraphBuilderPlanForGoals:
    """Tests for ToolGraphBuilder.plan_for_goals() method."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock tool registry with tools."""
        registry = MagicMock()

        # Create mock tools
        mock_tool1 = MagicMock()
        mock_tool1.name = "code_search"
        mock_tool1.description = "Search code"
        mock_tool1.parameters = {}

        mock_tool2 = MagicMock()
        mock_tool2.name = "read_file"
        mock_tool2.description = "Read file"
        mock_tool2.parameters = {}

        def get_tool(name):
            tools = {"code_search": mock_tool1, "read_file": mock_tool2}
            return tools.get(name)

        registry.get.side_effect = get_tool
        registry.is_tool_enabled.return_value = True

        return registry

    @pytest.fixture
    def mock_tool_graph(self):
        """Create mock tool graph."""
        graph = MagicMock()
        graph.plan.return_value = ["code_search", "read_file"]
        return graph

    def test_plan_for_goals_returns_empty_for_no_goals(self, mock_registry):
        """Test that plan_for_goals returns empty for no goals."""
        builder = ToolGraphBuilder(registry=mock_registry)

        result = builder.plan_for_goals([])

        assert result == []

    def test_plan_for_goals_returns_empty_for_no_graph(self, mock_registry):
        """Test that plan_for_goals returns empty when no graph."""
        builder = ToolGraphBuilder(registry=mock_registry)

        result = builder.plan_for_goals(["summary"])

        assert result == []

    def test_plan_for_goals_returns_tool_definitions(self, mock_registry, mock_tool_graph):
        """Test that plan_for_goals returns ToolDefinition objects."""
        builder = ToolGraphBuilder(registry=mock_registry, tool_graph=mock_tool_graph)

        result = builder.plan_for_goals(["summary"])

        assert len(result) == 2
        assert result[0].name == "code_search"
        assert result[1].name == "read_file"

    def test_plan_for_goals_passes_available_inputs(self, mock_registry, mock_tool_graph):
        """Test that plan_for_goals passes available inputs to graph."""
        builder = ToolGraphBuilder(registry=mock_registry, tool_graph=mock_tool_graph)

        builder.plan_for_goals(["summary"], available_inputs=["file_contents"])

        mock_tool_graph.plan.assert_called_once_with(["summary"], ["file_contents"])

    def test_plan_for_goals_skips_disabled_tools(self, mock_registry, mock_tool_graph):
        """Test that plan_for_goals skips disabled tools."""
        mock_registry.is_tool_enabled.side_effect = lambda name: name != "read_file"

        builder = ToolGraphBuilder(registry=mock_registry, tool_graph=mock_tool_graph)

        result = builder.plan_for_goals(["summary"])

        # Only code_search should be included (read_file is disabled)
        assert len(result) == 1
        assert result[0].name == "code_search"

    def test_plan_for_goals_skips_missing_tools(self, mock_registry, mock_tool_graph):
        """Test that plan_for_goals skips tools not in registry."""
        mock_tool_graph.plan.return_value = ["code_search", "nonexistent_tool"]

        builder = ToolGraphBuilder(registry=mock_registry, tool_graph=mock_tool_graph)

        result = builder.plan_for_goals(["summary"])

        # Only code_search should be included
        assert len(result) == 1
        assert result[0].name == "code_search"


class TestToolGraphBuilderInferGoalsFromMessage:
    """Tests for ToolGraphBuilder.infer_goals_from_message() method."""

    @pytest.fixture
    def builder(self):
        """Create ToolGraphBuilder instance."""
        registry = MagicMock()
        return ToolGraphBuilder(registry=registry)

    def test_infer_summary_goals(self, builder):
        """Test inferring summary goals from message."""
        messages = [
            "Can you summarize this code?",
            "Give me a summary of the project",
            "Analyze the codebase",
            "Provide an overview of the system",
        ]

        for msg in messages:
            goals = builder.infer_goals_from_message(msg)
            assert "summary" in goals, f"Expected 'summary' in goals for: {msg}"

    def test_infer_review_goals(self, builder):
        """Test inferring review goals from message."""
        messages = [
            "Please review this code",
            "Do a code review",
            "Audit the changes",
        ]

        for msg in messages:
            goals = builder.infer_goals_from_message(msg)
            assert "summary" in goals, f"Expected 'summary' in goals for: {msg}"

    def test_infer_documentation_goals(self, builder):
        """Test inferring documentation goals from message."""
        messages = [
            "Generate documentation",
            "Create a doc for this module",
            "Update the readme",
        ]

        for msg in messages:
            goals = builder.infer_goals_from_message(msg)
            assert "documentation" in goals, f"Expected 'documentation' in goals for: {msg}"

    def test_infer_security_goals(self, builder):
        """Test inferring security goals from message."""
        messages = [
            "Check for security vulnerabilities",
            "Scan for secrets",
            "Find security issues",
        ]

        for msg in messages:
            goals = builder.infer_goals_from_message(msg)
            assert "security_report" in goals, f"Expected 'security_report' in goals for: {msg}"

    def test_infer_metrics_goals(self, builder):
        """Test inferring metrics goals from message."""
        messages = [
            "Analyze code complexity",
            "Check the metrics",
            "Evaluate maintainability",
            "Identify technical debt",
        ]

        for msg in messages:
            goals = builder.infer_goals_from_message(msg)
            assert "metrics_report" in goals, f"Expected 'metrics_report' in goals for: {msg}"

    def test_infer_no_goals(self, builder):
        """Test that no goals are inferred for unrelated messages."""
        messages = [
            "Hello world",
            "What is Python?",
            "How do I write a function?",
        ]

        for msg in messages:
            goals = builder.infer_goals_from_message(msg)
            assert goals == [], f"Expected no goals for: {msg}"

    def test_infer_multiple_goals(self, builder):
        """Test inferring multiple goals from a single message."""
        msg = "Analyze and review the code, check for security issues"

        goals = builder.infer_goals_from_message(msg)

        assert "summary" in goals
        assert "security_report" in goals

    def test_case_insensitive(self, builder):
        """Test that goal inference is case-insensitive."""
        goals = builder.infer_goals_from_message("SUMMARIZE THIS CODE")

        assert "summary" in goals
