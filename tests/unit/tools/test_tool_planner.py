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

"""Unit tests for ToolPlanner.

Tests tool planning, goal inference, and intent-based filtering.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Any

from victor.agent.services.tool_planning_runtime import ToolPlanner
from victor.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = Mock(spec=Settings)
    return settings


@pytest.fixture
def mock_tool_registrar():
    """Create mock ToolRegistrar."""
    registrar = MagicMock()

    # Configure return values
    registrar.plan_tools.return_value = [
        Mock(name="read_file", outputs=["file_contents"]),
        Mock(name="code_review", outputs=["review"]),
    ]
    registrar.infer_goals_from_message.return_value = ["summary", "documentation"]

    return registrar


@pytest.fixture
def tool_planner(mock_tool_registrar, mock_settings):
    """Create ToolPlanner with mocked dependencies."""
    return ToolPlanner(
        tool_registrar=mock_tool_registrar,
        settings=mock_settings,
    )


class TestToolPlanning:
    """Tests for tool planning operations."""

    def test_plan_tools(self, tool_planner, mock_tool_registrar):
        """Test planning tool sequence to achieve goals."""
        goals = ["summary", "documentation"]
        available_inputs = ["file_contents"]

        # Configure mocks with name attribute
        tool1 = Mock()
        tool1.name = "read_file"
        tool2 = Mock()
        tool2.name = "code_review"
        mock_tool_registrar.plan_tools.return_value = [tool1, tool2]

        result = tool_planner.plan_tools(goals, available_inputs)

        # Verify delegation to tool registrar
        mock_tool_registrar.plan_tools.assert_called_once_with(goals, available_inputs)

        # Verify result
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].name == "read_file"
        assert result[1].name == "code_review"

    def test_plan_tools_no_available_inputs(self, tool_planner, mock_tool_registrar):
        """Test planning tools without available inputs."""
        goals = ["test_results"]

        result = tool_planner.plan_tools(goals)

        # Verify delegation with None for available_inputs
        mock_tool_registrar.plan_tools.assert_called_once_with(goals, None)
        assert isinstance(result, list)

    def test_plan_tools_empty_goals(self, tool_planner, mock_tool_registrar):
        """Test planning with empty goals list."""
        result = tool_planner.plan_tools([])

        mock_tool_registrar.plan_tools.assert_called_once_with([], None)

    def test_infer_goals_from_message(self, tool_planner, mock_tool_registrar):
        """Test inferring goals from user message."""
        message = "Please analyze this code and generate documentation"

        result = tool_planner.infer_goals_from_message(message)

        # Verify delegation
        mock_tool_registrar.infer_goals_from_message.assert_called_once_with(message)

        # Verify result
        assert isinstance(result, list)
        assert "summary" in result
        assert "documentation" in result

    def test_infer_goals_from_empty_message(self, tool_planner, mock_tool_registrar):
        """Test inferring goals from empty message."""
        mock_tool_registrar.infer_goals_from_message.return_value = []

        result = tool_planner.infer_goals_from_message("")

        mock_tool_registrar.infer_goals_from_message.assert_called_once_with("")
        assert result == []


class TestIntentFiltering:
    """Tests for intent-based tool filtering."""

    def test_filter_tools_no_intent(self, tool_planner):
        """Test filtering with no intent (no filtering)."""
        tools = [
            Mock(name="read_file"),
            Mock(name="write_file"),
            Mock(name="execute_bash"),
        ]

        result = tool_planner.filter_tools_by_intent(tools, None)

        # Should return all tools when no intent
        assert len(result) == 3
        assert result == tools

    def test_filter_tools_display_only_intent(self, tool_planner):
        """Test filtering with DISPLAY_ONLY intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Create mocks with name attribute properly set
        tool1 = Mock()
        tool1.name = "read_file"
        tool2 = Mock()
        tool2.name = "write_file"
        tool3 = Mock()
        tool3.name = "list_directory"
        tools = [tool1, tool2, tool3]

        result = tool_planner.filter_tools_by_intent(tools, ActionIntent.DISPLAY_ONLY)

        # write_file should be filtered out
        assert len(result) < len(tools)
        tool_names = [t.name for t in result]
        assert "read_file" in tool_names
        assert "list_directory" in tool_names
        # write_file should be filtered
        assert "write_file" not in tool_names

    def test_filter_tools_read_only_intent(self, tool_planner):
        """Test filtering with READ_ONLY intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Use canonical tool names from WRITE_TOOLS (not aliases)
        tool1 = Mock()
        tool1.name = "read"
        tool2 = Mock()
        tool2.name = "write"
        tool3 = Mock()
        tool3.name = "edit"
        tools = [tool1, tool2, tool3]

        result = tool_planner.filter_tools_by_intent(tools, ActionIntent.READ_ONLY)

        # write and edit tools should be filtered out (they're in WRITE_TOOLS)
        assert len(result) < len(tools)
        tool_names = [t.name for t in result]
        assert "read" in tool_names
        assert "write" not in tool_names
        assert "edit" not in tool_names

    def test_filter_tools_read_only_intent_keeps_shell_for_explicit_sqlite_request(
        self, tool_planner
    ):
        """Explicit readonly shell/SQLite requests should preserve shell availability."""
        from victor.agent.action_authorizer import ActionIntent

        read_tool = Mock()
        read_tool.name = "read"
        shell_tool = Mock()
        shell_tool.name = "shell"
        write_tool = Mock()
        write_tool.name = "write"

        result = tool_planner.filter_tools_by_intent(
            [read_tool, shell_tool, write_tool],
            ActionIntent.READ_ONLY,
            user_message="use shell tool with sqlite commands to review generations",
        )

        tool_names = [t.name for t in result]
        assert "read" in tool_names
        assert "shell" in tool_names
        assert "write" not in tool_names

    def test_filter_tools_display_only_intent_keeps_shell_for_sqllite_db_review(self, tool_planner):
        """Observed typo variants like 'sqllite db' should still preserve shell."""
        from victor.agent.action_authorizer import ActionIntent

        shell_tool = Mock()
        shell_tool.name = "shell"
        read_tool = Mock()
        read_tool.name = "read"

        result = tool_planner.filter_tools_by_intent(
            [read_tool, shell_tool],
            ActionIntent.DISPLAY_ONLY,
            user_message="also review the sqllite db for evolved prompts",
        )

        tool_names = [t.name for t in result]
        assert "read" in tool_names
        assert "shell" in tool_names

    def test_filter_tools_read_only_intent_does_not_rewrite_unknown_shell_variant(
        self, tool_planner
    ):
        """Unknown shell variants should pass through unchanged."""
        from victor.agent.action_authorizer import ActionIntent

        result = tool_planner.filter_tools_by_intent(
            [{"name": "shell_readonly"}],
            ActionIntent.READ_ONLY,
            user_message="use shell tool with sqlite commands to inspect the database",
        )

        assert len(result) == 1
        assert result[0]["name"] == "shell_readonly"

    def test_filter_tools_write_allowed_intent(self, tool_planner):
        """Test filtering with WRITE_ALLOWED intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Create mocks with name attribute properly set
        tool1 = Mock()
        tool1.name = "read_file"
        tool2 = Mock()
        tool2.name = "write_file"
        tool3 = Mock()
        tool3.name = "execute_bash"
        tools = [tool1, tool2, tool3]

        result = tool_planner.filter_tools_by_intent(tools, ActionIntent.WRITE_ALLOWED)

        # No filtering for WRITE_ALLOWED
        assert len(result) == len(tools)
        assert result == tools

    def test_filter_tools_with_dict_tools(self, tool_planner):
        """Test filtering with dict-based tool definitions."""
        from victor.agent.action_authorizer import ActionIntent

        tools = [
            {"name": "read"},
            {"name": "write"},
            {"name": "ls"},
        ]

        result = tool_planner.filter_tools_by_intent(tools, ActionIntent.DISPLAY_ONLY)

        # write should be filtered out (in WRITE_TOOLS)
        assert len(result) < len(tools)
        tool_names = [t["name"] for t in result]
        assert "read" in tool_names
        assert "write" not in tool_names

    def test_filter_tools_empty_list(self, tool_planner):
        """Test filtering with empty tools list."""
        from victor.agent.action_authorizer import ActionIntent

        result = tool_planner.filter_tools_by_intent([], ActionIntent.DISPLAY_ONLY)

        assert result == []

    def test_filter_tools_ambiguous_intent(self, tool_planner):
        """Test filtering with AMBIGUOUS intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Create mocks with name attribute properly set
        tool1 = Mock()
        tool1.name = "read_file"
        tool2 = Mock()
        tool2.name = "write_file"
        tools = [tool1, tool2]

        result = tool_planner.filter_tools_by_intent(tools, ActionIntent.AMBIGUOUS)

        # AMBIGUOUS should not filter
        assert len(result) == len(tools)


class TestToolPlannerInitialization:
    """Tests for ToolPlanner initialization."""

    def test_initialization_with_valid_dependencies(self, mock_tool_registrar, mock_settings):
        """Test successful initialization with valid dependencies."""
        tool_planner = ToolPlanner(
            tool_registrar=mock_tool_registrar,
            settings=mock_settings,
        )

        assert tool_planner.tool_registrar is mock_tool_registrar
        assert tool_planner.settings is mock_settings

    def test_initialization_stores_dependencies(self, mock_tool_registrar, mock_settings):
        """Test that initialization stores all dependencies."""
        tool_planner = ToolPlanner(
            tool_registrar=mock_tool_registrar,
            settings=mock_settings,
        )

        # Verify dependencies are accessible
        assert hasattr(tool_planner, "tool_registrar")
        assert hasattr(tool_planner, "settings")


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_tool_planner_with_none_registrar(self, mock_settings):
        """Test ToolPlanner handles None tool registrar gracefully."""
        tool_planner = ToolPlanner(
            tool_registrar=None,
            settings=mock_settings,
        )

        # Attempting to use it should fail
        with pytest.raises(AttributeError):
            tool_planner.plan_tools(["summary"])

    def test_filter_tools_with_invalid_tool_format(self, tool_planner):
        """Test filtering with tools that have invalid format."""
        from victor.agent.action_authorizer import ActionIntent

        # Tools without name attribute or key
        tools = [
            "invalid_tool",  # String instead of object/dict
            123,  # Number
            None,  # None
        ]

        result = tool_planner.filter_tools_by_intent(tools, ActionIntent.DISPLAY_ONLY)

        # Should handle gracefully (filter out invalid tools)
        assert isinstance(result, list)
