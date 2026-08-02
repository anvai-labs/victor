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

"""Tests for KeywordToolSelector.

Covers HIGH-002: Unified Tool Selection Architecture - Release 2, Phase 3.
"""

import pytest
from unittest.mock import MagicMock, patch

from victor.agent.protocols.tool_protocols import ToolSelectionContext
from victor.providers.base import ToolDefinition
from victor.tools.base import BaseTool, ToolResult
from victor.tools.registry import ToolRegistry
from victor.tools.keyword_tool_selector import KeywordToolSelector


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, name: str, description: str = "A test tool"):
        super().__init__()
        self._name = name
        self._description = description
        self._parameters = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return self._parameters

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="mock result")


class TestKeywordToolSelector:
    """Tests for KeywordToolSelector functionality."""

    @pytest.fixture
    def tool_registry(self):
        """Create a tool registry with mock tools."""
        registry = ToolRegistry()
        registry.register_tool(MockTool("read_file", "Read a file from the filesystem"))
        registry.register_tool(MockTool("write_file", "Write content to a file"))
        registry.register_tool(MockTool("code_search", "Search code in the codebase"))
        registry.register_tool(MockTool("git_status", "Check git status"))
        registry.register_tool(MockTool("shell", "Run a shell command"))
        return registry

    @pytest.fixture
    def selector(self, tool_registry):
        """Create a KeywordToolSelector with mock tools."""
        return KeywordToolSelector(
            tools=tool_registry,
            conversation_state=None,
            model="gpt-4",
            provider_name="openai",
        )

    def test_init(self, tool_registry):
        """Test initialization with various parameters."""
        selector = KeywordToolSelector(
            tools=tool_registry,
            conversation_state=None,
            model="gpt-4",
            provider_name="openai",
            enabled_tools={"read_file", "write_file"},
        )

        assert selector.tools is tool_registry
        assert selector.model == "gpt-4"
        assert selector.provider_name == "openai"
        assert selector._enabled_tools == {"read_file", "write_file"}

    def test_init_without_enabled_tools(self, tool_registry):
        """Test initialization without enabled_tools filter."""
        selector = KeywordToolSelector(
            tools=tool_registry,
            conversation_state=None,
        )

        assert selector._enabled_tools is None

    @pytest.mark.asyncio
    async def test_select_tools_with_enabled_tools_filter(self, tool_registry):
        """Test that enabled_tools filter restricts tool selection."""
        selector = KeywordToolSelector(
            tools=tool_registry,
            enabled_tools={"read_file", "code_search"},
        )

        context = ToolSelectionContext(task_type="analysis")
        result = await selector.select_tools("find a function", context)

        result_names = {t.name for t in result}
        # Should only return tools from enabled_tools
        assert result_names.issubset({"read_file", "code_search"})

    @pytest.mark.asyncio
    async def test_select_tools_with_planned_tools(self, tool_registry):
        """Test that planned_tools are included in selection."""
        selector = KeywordToolSelector(tools=tool_registry)

        context = ToolSelectionContext(
            task_type="action",
            planned_tools=["read_file", "write_file"],
        )
        result = await selector.select_tools("do something", context)

        result_names = {t.name for t in result}
        # Planned tools should be included
        assert "read_file" in result_names
        assert "write_file" in result_names

    @pytest.mark.asyncio
    @patch("victor.tools.keyword_tool_selector.get_tools_from_message")
    @patch("victor.tools.keyword_tool_selector.is_small_model")
    async def test_select_tools_uses_keyword_matching(
        self, mock_is_small, mock_get_tools, tool_registry
    ):
        """Test that keyword matching is used when no enabled_tools."""
        mock_is_small.return_value = False
        mock_get_tools.return_value = {"code_search", "git_status"}

        selector = KeywordToolSelector(tools=tool_registry)

        context = ToolSelectionContext(task_type="analysis")
        result = await selector.select_tools("search for code in git", context)

        result_names = {t.name for t in result}
        # Should include keyword-matched tools
        assert "code_search" in result_names
        assert "git_status" in result_names
        mock_get_tools.assert_called_once_with("search for code in git")

    @pytest.mark.asyncio
    @patch("victor.tools.keyword_tool_selector.is_small_model")
    @patch("victor.tools.keyword_tool_selector.get_tools_from_message")
    async def test_select_tools_limits_for_small_models(
        self, mock_get_tools, mock_is_small, tool_registry
    ):
        """Test that small models get limited tool count."""
        mock_is_small.return_value = True

        # Add more tools to exceed the 10 limit
        for i in range(15):
            tool_registry.register_tool(MockTool(f"extra_tool_{i}", f"Extra tool {i}"))

        # Return all extra tools as keyword matches
        mock_get_tools.return_value = {f"extra_tool_{i}" for i in range(15)}

        # Don't use enabled_tools - small model limit only applies without it
        selector = KeywordToolSelector(
            tools=tool_registry,
            model="small-model",
        )

        context = ToolSelectionContext(task_type="action")
        result = await selector.select_tools("do all the things", context)

        # Should be capped at 10 for small models
        assert len(result) <= 10

    def test_get_supported_features(self, selector):
        """Test get_supported_features returns keyword-only features."""
        features = selector.get_supported_features()

        assert features.supports_semantic_matching is False
        assert features.supports_context_awareness is False
        assert features.supports_cost_optimization is False
        assert features.supports_usage_learning is False
        assert features.supports_workflow_patterns is False
        assert features.requires_embeddings is False

    def test_record_tool_execution_is_noop(self, selector):
        """Test that record_tool_execution is a no-op."""
        # Should not raise any errors
        selector.record_tool_execution("read_file", True, {"task_type": "action"})
        selector.record_tool_execution("write_file", False, None)

    @pytest.mark.asyncio
    async def test_close_is_noop(self, selector):
        """Test that close is a no-op."""
        # Should not raise any errors
        await selector.close()


class TestKeywordToolSelectorStageFiltering:
    """Tests for stage-based filtering in KeywordToolSelector."""

    @pytest.fixture
    def tool_registry(self):
        """Create a tool registry with mock tools."""
        registry = ToolRegistry()
        registry.register_tool(MockTool("read_file", "Read a file"))
        registry.register_tool(MockTool("write_file", "Write a file"))
        registry.register_tool(MockTool("code_search", "Search code"))
        return registry

    @pytest.fixture
    def selector(self, tool_registry):
        """Create selector with all tools enabled."""
        return KeywordToolSelector(
            tools=tool_registry,
            enabled_tools={"read_file", "write_file", "code_search"},
        )

    def test_has_write_intent_detects_keywords(self, selector):
        """Test _has_write_intent detects write keywords."""
        assert selector._has_write_intent("create a new file") is True
        assert selector._has_write_intent("write some code") is True
        assert selector._has_write_intent("edit the function") is True
        assert selector._has_write_intent("update the config") is True
        assert selector._has_write_intent("fix the bug") is True
        assert selector._has_write_intent("delete the file") is True
        assert selector._has_write_intent("refactor the module") is True

    def test_has_write_intent_no_match(self, selector):
        """Test _has_write_intent returns False for non-write prompts."""
        assert selector._has_write_intent("find a function") is False
        assert selector._has_write_intent("explain this code") is False
        assert selector._has_write_intent("what does this do") is False
        assert selector._has_write_intent("list all files") is False

    def test_filter_tools_for_stage_no_stage(self, selector):
        """Test _filter_tools_for_stage returns all tools when no stage."""
        tools = [
            ToolDefinition(name="read_file", description="Read", parameters={}),
            ToolDefinition(name="write_file", description="Write", parameters={}),
        ]

        result = selector._filter_tools_for_stage(tools, stage=None, prompt="")

        assert len(result) == 2

    def test_filter_tools_for_stage_with_write_intent(self, selector):
        """Test _filter_tools_for_stage skips filtering when write intent."""
        from victor.agent.conversation.state_machine import ConversationStage

        tools = [
            ToolDefinition(name="read_file", description="Read", parameters={}),
            ToolDefinition(name="write_file", description="Write", parameters={}),
        ]

        result = selector._filter_tools_for_stage(
            tools, stage=ConversationStage.ANALYSIS, prompt="create a new file"
        )

        # Should return all tools due to write intent
        assert len(result) == 2

    @patch("victor.tools.keyword_tool_selector.KeywordToolSelector._is_readonly_tool")
    def test_filter_tools_for_stage_analysis(self, mock_is_readonly, selector):
        """Test _filter_tools_for_stage annotates write tools with HITL prompt."""
        from victor.agent.conversation.state_machine import ConversationStage

        # Only read_file is readonly
        mock_is_readonly.side_effect = lambda name: name == "read_file"

        tools = [
            ToolDefinition(name="read_file", description="Read", parameters={}),
            ToolDefinition(name="write_file", description="Write", parameters={}),
        ]

        result = selector._filter_tools_for_stage(
            tools, stage=ConversationStage.ANALYSIS, prompt="find code"
        )

        result_names = [t.name for t in result]
        # Both tools should be kept (not removed)
        assert "read_file" in result_names
        assert "write_file" in result_names

        # Write tool should be annotated with HITL prompt
        write_tool = next(t for t in result if t.name == "write_file")
        assert "[HITL]" in write_tool.description
        assert "confirmation" in write_tool.description.lower()

        # Read tool should NOT be annotated
        read_tool = next(t for t in result if t.name == "read_file")
        assert "[HITL]" not in read_tool.description

    def test_get_stage_core_tools_no_stage(self, selector):
        """Test _get_stage_core_tools returns all core tools when no stage."""
        with patch.object(
            selector, "_get_core_tools_cached", return_value={"read_file", "write_file"}
        ):
            result = selector._get_stage_core_tools(None)
            assert result == {"read_file", "write_file"}

    def test_get_stage_core_tools_analysis_stage(self, selector):
        """Test _get_stage_core_tools returns readonly tools for analysis."""
        from victor.agent.conversation.state_machine import ConversationStage

        with patch.object(selector, "_get_core_readonly_cached", return_value={"read_file"}):
            result = selector._get_stage_core_tools(ConversationStage.ANALYSIS)
            assert result == {"read_file"}

    def test_get_stage_core_tools_execution_stage(self, selector):
        """Test _get_stage_core_tools returns all core tools for execution."""
        from victor.agent.conversation.state_machine import ConversationStage

        with patch.object(
            selector, "_get_core_tools_cached", return_value={"read_file", "write_file"}
        ):
            result = selector._get_stage_core_tools(ConversationStage.EXECUTION)
            assert result == {"read_file", "write_file"}


class TestKeywordToolSelectorCaching:
    """Tests for caching behavior in KeywordToolSelector."""

    @pytest.fixture
    def tool_registry(self):
        """Create a tool registry with mock tools."""
        registry = ToolRegistry()
        registry.register_tool(MockTool("read_file", "Read a file"))
        return registry

    def test_core_tools_cached(self, tool_registry):
        """Test that core tools are cached after first access."""
        selector = KeywordToolSelector(tools=tool_registry)

        with patch(
            "victor.tools.selection_common.get_critical_tools",
            return_value={"read_file"},
        ) as mock_get:
            # First access
            result1 = selector._get_core_tools_cached()
            # Second access
            result2 = selector._get_core_tools_cached()

            # Should only call get_critical_tools once
            assert mock_get.call_count == 1
            assert result1 == {"read_file"}
            assert result2 == {"read_file"}

    def test_core_readonly_cached(self, tool_registry):
        """Test that core readonly tools are cached after first access."""
        selector = KeywordToolSelector(tools=tool_registry)

        with patch(
            "victor.tools.metadata_registry.get_core_readonly_tools",
            return_value=["read_file"],
        ) as mock_get:
            # First access
            result1 = selector._get_core_readonly_cached()
            # Second access
            result2 = selector._get_core_readonly_cached()

            # Should only call get_core_readonly_tools once
            assert mock_get.call_count == 1
            assert result1 == {"read_file"}
            assert result2 == {"read_file"}


class TestKeywordToolSelectorReadonlyCheck:
    """Tests for readonly tool checking."""

    @pytest.fixture
    def tool_registry(self):
        """Create a tool registry with mock tools."""
        registry = ToolRegistry()
        registry.register_tool(MockTool("read_file", "Read a file"))
        return registry

    @pytest.fixture
    def selector(self, tool_registry):
        """Create a KeywordToolSelector."""
        return KeywordToolSelector(tools=tool_registry)

    @patch("victor.tools.metadata_registry.get_global_registry")
    def test_is_readonly_tool_true(self, mock_get_registry, selector):
        """Test _is_readonly_tool returns True for readonly tools."""
        from victor.tools.base import AccessMode, ExecutionCategory

        mock_entry = MagicMock()
        mock_entry.access_mode = AccessMode.READONLY
        mock_entry.execution_category = ExecutionCategory.READ_ONLY

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_entry
        mock_get_registry.return_value = mock_registry

        assert selector._is_readonly_tool("read_file") is True

    @patch("victor.tools.metadata_registry.get_global_registry")
    def test_is_readonly_tool_false(self, mock_get_registry, selector):
        """Test _is_readonly_tool returns False for write tools."""
        from victor.tools.base import AccessMode, ExecutionCategory

        mock_entry = MagicMock()
        mock_entry.access_mode = AccessMode.WRITE
        mock_entry.execution_category = ExecutionCategory.WRITE

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_entry
        mock_get_registry.return_value = mock_registry

        assert selector._is_readonly_tool("write_file") is False

    @patch("victor.tools.metadata_registry.get_global_registry")
    def test_is_readonly_tool_not_found(self, mock_get_registry, selector):
        """Test _is_readonly_tool returns False for unknown tools."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_get_registry.return_value = mock_registry

        assert selector._is_readonly_tool("unknown_tool") is False

    def test_is_readonly_tool_handles_exception(self, selector):
        """Test _is_readonly_tool handles exceptions gracefully."""
        with patch(
            "victor.tools.metadata_registry.get_global_registry",
            side_effect=Exception("Registry error"),
        ):
            # Should return False, not raise
            assert selector._is_readonly_tool("read_file") is False
