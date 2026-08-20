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

"""Tests for tool selection module.

Tool selection now uses keywords defined in @tool decorators via ToolMetadataRegistry
as the single source of truth. See test_metadata_registry.py for keyword-based
selection tests.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.agent.tool_selection import (
    get_critical_tools,
    get_tools_from_message,
    ToolSelectionStats,
    is_small_model,
    needs_web_tools,
    calculate_adaptive_threshold,
    select_tools_by_keywords,
)


class TestCriticalTools:
    """Tests for critical tool selection."""

    def test_critical_tools_contains_essentials(self):
        """Test that get_critical_tools() returns essential tools (canonical names).

        Note: Without a registry, get_critical_tools uses _FALLBACK_CRITICAL_TOOLS.
        """
        critical_tools = get_critical_tools()  # Fallback returns canonical names
        assert "read" in critical_tools
        assert "write" in critical_tools
        assert "ls" in critical_tools
        assert "shell" in critical_tools
        assert "edit" in critical_tools
        assert "search" in critical_tools


class TestGetToolsFromMessage:
    """Tests for get_tools_from_message function.

    This function uses ToolMetadataRegistry to find tools whose @tool(keywords=[...])
    match the user's message. Without registered tools, returns empty set.
    See test_metadata_registry.py for comprehensive keyword matching tests.
    """

    def test_returns_set(self):
        """Test that function returns a set."""
        tools = get_tools_from_message("test message")
        assert isinstance(tools, set)

    def test_handles_empty_message(self):
        """Test that empty message returns empty set."""
        tools = get_tools_from_message("")
        assert tools == set()


class TestIsSmallModel:
    """Tests for is_small_model function."""

    def test_detects_small_ollama_models(self):
        """Test detection of small Ollama models."""
        assert is_small_model("llama:0.5b", "ollama")
        assert is_small_model("phi:1.5b", "ollama")
        assert is_small_model("gemma:3b", "ollama")

    def test_large_models_not_small(self):
        """Test that larger models are not marked as small."""
        assert not is_small_model("llama:7b", "ollama")

    def test_non_ollama_not_small(self):
        """Test that non-Ollama providers are not marked as small."""
        assert not is_small_model("gpt-4", "openai")


class TestNeedsWebTools:
    """Tests for needs_web_tools function."""

    def test_detects_web_keywords(self):
        """Test detection of web-related keywords."""
        assert needs_web_tools("search the web")
        assert needs_web_tools("look this up online")

    def test_no_web_for_generic(self):
        """Test that generic messages don't need web tools."""
        assert not needs_web_tools("read the file")


class TestCalculateAdaptiveThreshold:
    """Tests for calculate_adaptive_threshold function."""

    def test_returns_tuple(self):
        """Test that function returns threshold and max_tools."""
        threshold, max_tools = calculate_adaptive_threshold(
            model_name="llama:7b",
            query_word_count=10,
            conversation_depth=5,
        )
        assert isinstance(threshold, float)
        assert isinstance(max_tools, int)

    def test_threshold_within_bounds(self):
        """Test that threshold is within reasonable bounds."""
        threshold, max_tools = calculate_adaptive_threshold(
            model_name="llama:7b",
            query_word_count=10,
            conversation_depth=5,
        )
        assert 0.10 <= threshold <= 0.40
        assert 5 <= max_tools <= 15


class TestToolSelectionStats:
    """Tests for ToolSelectionStats class."""

    def test_initial_stats_zero(self):
        """Test that initial stats are zero."""
        stats = ToolSelectionStats()
        assert stats.semantic_selections == 0

    def test_record_semantic_selection(self):
        """Test recording semantic selection."""
        stats = ToolSelectionStats()
        stats.record_selection("semantic", 5)
        assert stats.semantic_selections == 1
        assert stats.total_tools_selected == 5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = ToolSelectionStats()
        stats.record_selection("semantic", 5)
        result = stats.to_dict()
        assert isinstance(result, dict)
        assert result["semantic_selections"] == 1


class TestToolSelectorRuntimeIntelligence:
    """Tests for canonical runtime-intelligence integration in ToolSelector."""

    def test_edge_filter_prefers_runtime_intelligence(self):
        """Tool filtering should use runtime intelligence before container lookups."""
        from victor.agent.tool_selection import ToolSelector

        registry = MagicMock()
        registry.list_tools.return_value = []
        runtime_intelligence = MagicMock()
        selector = ToolSelector(tools=registry, runtime_intelligence=runtime_intelligence)
        selector._get_core_tools_cached = MagicMock(return_value={"read", "ls"})
        tools = [
            SimpleNamespace(name="read"),
            SimpleNamespace(name="ls"),
            SimpleNamespace(name="edit"),
            SimpleNamespace(name="grep"),
        ]

        with (
            patch(
                "victor.agent.tool_selection.selector.get_container",
                side_effect=AssertionError("container path should not be used"),
            ),
            patch(
                "victor.agent.edge_model.select_tools_with_edge_model",
                return_value=["edit"],
            ) as mock_select,
        ):
            filtered = selector._apply_edge_model_filter(
                tools,
                user_message="fix the auth bug",
                stage=None,
            )

        assert [tool.name for tool in filtered] == ["read", "ls", "edit"]
        mock_select.assert_called_once_with(
            service=runtime_intelligence,
            user_message="fix the auth bug",
            available_tools=["read", "ls", "edit", "grep"],
            stage="initial",
            recent_tools=None,
            max_tools=6,
        )


class TestSelectToolsByKeywords:
    """Tests for select_tools_by_keywords function.

    This function now uses ToolMetadataRegistry for keyword-based selection.
    It always includes critical tools, then adds tools whose keywords match
    the user message.
    """

    def test_includes_core_tools(self):
        """Test that core tools are always included (using canonical names)."""
        tools = select_tools_by_keywords(
            message="hello world",
            all_tool_names={"read", "write", "git"},
        )
        assert "read" in tools
        assert "write" in tools

    def test_includes_critical_tools(self):
        """Test that critical tools are always included regardless of message.

        Note: Category-based tool selection requires ToolMetadataRegistry.
        Without it, only critical tools are included.
        """
        tools = select_tools_by_keywords(
            message="help me commit",
            all_tool_names={"read", "write", "git", "commit_msg"},
        )
        # Critical tools should be included
        assert "read" in tools

    def test_limits_for_small_models(self):
        """Test that small models get limited tools."""
        all_tools = {f"tool_{i}" for i in range(20)}
        all_tools.update(get_critical_tools())  # Use dynamic discovery
        tools = select_tools_by_keywords(
            message="do everything",
            all_tool_names=all_tools,
            is_small=True,
            max_tools_for_small=10,
        )
        assert len(tools) <= 10
