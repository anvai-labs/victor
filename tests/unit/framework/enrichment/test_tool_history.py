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

"""Unit tests for tool history context extraction utilities."""

import pytest
from victor.framework.enrichment.tool_history import (
    ToolHistoryExtractor,
    extract_tool_context,
    get_relevant_tool_results,
)


class TestGetRelevantToolResults:
    """Tests for get_relevant_tool_results function."""

    def test_empty_history_returns_empty_list(self):
        """Test empty history returns empty list."""
        result = get_relevant_tool_results([])
        assert result == []

    def test_none_like_empty_list(self):
        """Test handling of empty list."""
        result = get_relevant_tool_results([])
        assert result == []

    def test_filters_successful_results_only(self):
        """Test only successful results are included."""
        history = [
            {
                "tool": "web_search",
                "result": {"success": True, "content": "x" * 60},  # Long enough content
            },
            {"tool": "web_fetch", "result": {"success": False, "content": "y" * 60}},
        ]
        result = get_relevant_tool_results(history)
        assert len(result) == 1
        assert result[0]["tool"] == "web_search"

    def test_filters_by_tool_names(self):
        """Test filtering by specific tool names."""
        history = [
            {"tool": "web_search", "result": {"success": True, "content": "x" * 60}},
            {"tool": "file_read", "result": {"success": True, "content": "y" * 60}},
        ]
        result = get_relevant_tool_results(history, tool_names={"web_search"})
        assert len(result) == 1
        assert result[0]["tool"] == "web_search"

    def test_filters_by_content_length(self):
        """Test filtering by minimum content length."""
        history = [
            {
                "tool": "tool1",
                "result": {"success": True, "content": "short"},
            },  # Too short
            {
                "tool": "tool2",
                "result": {"success": True, "content": "x" * 100},
            },  # Long enough
        ]
        result = get_relevant_tool_results(history, min_content_length=50)
        assert len(result) == 1
        assert result[0]["tool"] == "tool2"

    def test_respects_max_results(self):
        """Test max_results limits processing."""
        history = [
            {"tool": f"tool{i}", "result": {"success": True, "content": "x" * 60}}
            for i in range(20)
        ]
        result = get_relevant_tool_results(history, max_results=5)
        # Should only process last 5 entries
        assert len(result) <= 5

    def test_extracts_metadata(self):
        """Test metadata is extracted from results."""
        history = [
            {
                "tool": "web_search",
                "result": {
                    "success": True,
                    "content": "x" * 60,
                    "metadata": {"query": "test query"},
                },
            }
        ]
        result = get_relevant_tool_results(history)
        assert result[0]["metadata"] == {"query": "test query"}

    def test_handles_missing_metadata(self):
        """Test handles missing metadata gracefully."""
        history = [
            {
                "tool": "web_search",
                "result": {
                    "success": True,
                    "content": "x" * 60,
                    # No metadata
                },
            }
        ]
        result = get_relevant_tool_results(history)
        assert result[0]["metadata"] == {}

    def test_handles_non_dict_result(self):
        """Test handles non-dict result value."""
        history = [{"tool": "web_search", "result": "string result"}]  # Not a dict
        result = get_relevant_tool_results(history)
        assert result == []

    def test_handles_missing_tool_name(self):
        """Test handles missing tool name."""
        history = [
            {
                # No "tool" key
                "result": {"success": True, "content": "x" * 60}
            }
        ]
        result = get_relevant_tool_results(history)
        assert len(result) == 1
        assert result[0]["tool"] == ""

    def test_handles_empty_content(self):
        """Test filters out empty content."""
        history = [
            {
                "tool": "web_search",
                "result": {"success": True, "content": ""},
            }  # Empty content
        ]
        result = get_relevant_tool_results(history)
        assert result == []

    def test_handles_none_content(self):
        """Test filters out None content."""
        history = [{"tool": "web_search", "result": {"success": True, "content": None}}]
        result = get_relevant_tool_results(history)
        assert result == []

    def test_multiple_tool_names_filter(self):
        """Test filtering with multiple tool names."""
        history = [
            {"tool": "web_search", "result": {"success": True, "content": "x" * 60}},
            {"tool": "web_fetch", "result": {"success": True, "content": "y" * 60}},
            {"tool": "file_read", "result": {"success": True, "content": "z" * 60}},
        ]
        result = get_relevant_tool_results(history, tool_names={"web_search", "web_fetch"})
        assert len(result) == 2
        tools = [r["tool"] for r in result]
        assert "web_search" in tools
        assert "web_fetch" in tools

    def test_tool_names_none_includes_all(self):
        """Test tool_names=None includes all tools."""
        history = [
            {"tool": "tool1", "result": {"success": True, "content": "x" * 60}},
            {"tool": "tool2", "result": {"success": True, "content": "y" * 60}},
        ]
        result = get_relevant_tool_results(history, tool_names=None)
        assert len(result) == 2


class TestExtractToolContext:
    """Tests for extract_tool_context function."""

    def test_empty_history_returns_empty_string(self):
        """Test empty history returns empty string."""
        result = extract_tool_context([])
        assert result == ""

    def test_no_relevant_results_returns_empty_string(self):
        """Test no relevant results returns empty string."""
        history = [{"tool": "tool1", "result": {"success": False, "content": "x" * 60}}]
        result = extract_tool_context(history)
        assert result == ""

    def test_formats_with_default_header(self):
        """Test formats with default header."""
        history = [{"tool": "web_search", "result": {"success": True, "content": "x" * 60}}]
        result = extract_tool_context(history)
        assert "Prior results in this session:" in result

    def test_formats_with_custom_header(self):
        """Test formats with custom header."""
        history = [{"tool": "web_search", "result": {"success": True, "content": "x" * 60}}]
        result = extract_tool_context(history, header="Custom Header:")
        assert "Custom Header:" in result

    def test_includes_tool_name(self):
        """Test formatted result includes tool name."""
        history = [{"tool": "web_search", "result": {"success": True, "content": "x" * 60}}]
        result = extract_tool_context(history)
        assert "web_search" in result

    def test_includes_content(self):
        """Test formatted result includes content."""
        content = "This is the result content. " * 3  # At least 50 chars
        history = [{"tool": "web_search", "result": {"success": True, "content": content}}]
        result = extract_tool_context(history)
        assert "This is the result content" in result

    def test_truncates_long_content(self):
        """Test long content is truncated."""
        long_content = "x" * 500
        history = [{"tool": "web_search", "result": {"success": True, "content": long_content}}]
        result = extract_tool_context(history, max_content_length=100)
        assert "..." in result
        assert len(result) < 500

    def test_respects_max_results(self):
        """Test max_results limits output."""
        history = [
            {
                "tool": f"tool{i}",
                "result": {"success": True, "content": f"content{i} " * 20},
            }
            for i in range(10)
        ]
        result = extract_tool_context(history, max_results=2)
        # Should only have 2 tool entries
        assert result.count("- From") == 2

    def test_filters_by_tool_names(self):
        """Test filtering by tool names."""
        history = [
            {"tool": "web_search", "result": {"success": True, "content": "x" * 60}},
            {"tool": "file_read", "result": {"success": True, "content": "y" * 60}},
        ]
        result = extract_tool_context(history, tool_names={"web_search"})
        assert "web_search" in result
        assert "file_read" not in result

    def test_format_structure(self):
        """Test format structure matches expected pattern."""
        history = [
            {
                "tool": "web_search",
                "result": {"success": True, "content": "test content " * 10},
            }
        ]
        result = extract_tool_context(history)
        # Check structure
        lines = result.split("\n")
        assert lines[0] == "Prior results in this session:"
        assert "- From web_search:" in result


class TestToolHistoryExtractor:
    """Tests for ToolHistoryExtractor class."""

    def test_init_default_values(self):
        """Test default initialization."""
        extractor = ToolHistoryExtractor()
        assert extractor.tool_names is None
        assert extractor.max_results == 3

    def test_init_custom_tool_names(self):
        """Test custom tool_names."""
        extractor = ToolHistoryExtractor(tool_names={"web_search"})
        assert extractor.tool_names == {"web_search"}

    def test_init_custom_max_results(self):
        """Test custom max_results."""
        extractor = ToolHistoryExtractor(max_results=10)
        assert extractor.max_results == 10

    def test_init_custom_header(self):
        """Test custom header."""
        extractor = ToolHistoryExtractor(header="Custom:")
        assert extractor._header == "Custom:"

    def test_extract_empty_history(self):
        """Test extract with empty history."""
        extractor = ToolHistoryExtractor()
        result = extractor.extract([])
        assert result == ""

    def test_extract_with_results(self):
        """Test extract with results."""
        extractor = ToolHistoryExtractor()
        history = [{"tool": "web_search", "result": {"success": True, "content": "x" * 60}}]
        result = extractor.extract(history)
        assert "web_search" in result

    def test_extract_respects_tool_names(self):
        """Test extract respects tool_names filter."""
        extractor = ToolHistoryExtractor(tool_names={"web_search"})
        history = [
            {"tool": "web_search", "result": {"success": True, "content": "x" * 60}},
            {"tool": "file_read", "result": {"success": True, "content": "y" * 60}},
        ]
        result = extractor.extract(history)
        assert "web_search" in result
        assert "file_read" not in result

    def test_get_relevant_results_empty_history(self):
        """Test get_relevant_results with empty history."""
        extractor = ToolHistoryExtractor()
        result = extractor.get_relevant_results([])
        assert result == []

    def test_get_relevant_results_with_data(self):
        """Test get_relevant_results with data."""
        extractor = ToolHistoryExtractor()
        history = [{"tool": "web_search", "result": {"success": True, "content": "x" * 60}}]
        result = extractor.get_relevant_results(history)
        assert len(result) == 1
        assert result[0]["tool"] == "web_search"

    def test_get_relevant_results_respects_filters(self):
        """Test get_relevant_results respects tool_names filter."""
        extractor = ToolHistoryExtractor(tool_names={"web_search"})
        history = [
            {"tool": "web_search", "result": {"success": True, "content": "x" * 60}},
            {"tool": "file_read", "result": {"success": True, "content": "y" * 60}},
        ]
        result = extractor.get_relevant_results(history)
        assert len(result) == 1
        assert result[0]["tool"] == "web_search"

    def test_add_tool_name_to_none(self):
        """Test add_tool_name when tool_names is None."""
        extractor = ToolHistoryExtractor()
        assert extractor.tool_names is None
        extractor.add_tool_name("web_search")
        assert extractor.tool_names == {"web_search"}

    def test_add_tool_name_to_existing(self):
        """Test add_tool_name to existing set."""
        extractor = ToolHistoryExtractor(tool_names={"web_search"})
        extractor.add_tool_name("web_fetch")
        assert "web_search" in extractor.tool_names
        assert "web_fetch" in extractor.tool_names

    def test_add_duplicate_tool_name(self):
        """Test adding duplicate tool name."""
        extractor = ToolHistoryExtractor(tool_names={"web_search"})
        extractor.add_tool_name("web_search")
        assert len(extractor.tool_names) == 1

    def test_remove_tool_name_existing(self):
        """Test remove_tool_name for existing tool."""
        extractor = ToolHistoryExtractor(tool_names={"web_search", "web_fetch"})
        extractor.remove_tool_name("web_search")
        assert "web_search" not in extractor.tool_names
        assert "web_fetch" in extractor.tool_names

    def test_remove_tool_name_nonexistent(self):
        """Test remove_tool_name for nonexistent tool."""
        extractor = ToolHistoryExtractor(tool_names={"web_search"})
        extractor.remove_tool_name("nonexistent")
        # Should not raise, just silently ignore
        assert extractor.tool_names == {"web_search"}

    def test_remove_tool_name_from_none(self):
        """Test remove_tool_name when tool_names is None."""
        extractor = ToolHistoryExtractor()
        extractor.remove_tool_name("web_search")
        # Should not raise
        assert extractor.tool_names is None

    def test_tool_names_property_getter(self):
        """Test tool_names property getter."""
        extractor = ToolHistoryExtractor(tool_names={"web_search"})
        assert extractor.tool_names == {"web_search"}

    def test_max_results_property_getter(self):
        """Test max_results property getter."""
        extractor = ToolHistoryExtractor(max_results=7)
        assert extractor.max_results == 7

    def test_max_results_property_setter(self):
        """Test max_results property setter."""
        extractor = ToolHistoryExtractor(max_results=5)
        extractor.max_results = 10
        assert extractor.max_results == 10

    def test_max_results_setter_enforces_minimum(self):
        """Test max_results setter enforces minimum of 1."""
        extractor = ToolHistoryExtractor()
        extractor.max_results = 0
        assert extractor.max_results >= 1
        extractor.max_results = -5
        assert extractor.max_results >= 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_history_with_mixed_result_types(self):
        """Test history with mixed result types."""
        history = [
            {"tool": "tool1", "result": {"success": True, "content": "x" * 60}},
            {"tool": "tool2", "result": "string result"},
            {"tool": "tool3", "result": None},
            {"tool": "tool4", "result": []},
            {"tool": "tool5", "result": {"success": True, "content": "y" * 60}},
        ]
        result = get_relevant_tool_results(history)
        # Should only include valid dict results with success=True
        assert len(result) == 2

    def test_very_long_history(self):
        """Test handling of very long history."""
        history = [
            {
                "tool": f"tool{i}",
                "result": {"success": True, "content": f"content{i} " * 20},
            }
            for i in range(1000)
        ]
        result = get_relevant_tool_results(history, max_results=10)
        assert len(result) <= 10

    def test_unicode_content(self):
        """Test handling of unicode content."""
        history = [
            {
                "tool": "web_search",
                "result": {"success": True, "content": "Unicode content: " * 10},
            }
        ]
        result = extract_tool_context(history)
        assert "Unicode content" in result

    def test_special_characters_in_content(self):
        """Test handling of special characters in content."""
        history = [
            {
                "tool": "web_search",
                "result": {
                    "success": True,
                    "content": "Special chars: @#$%^&*() " * 10,
                },
            }
        ]
        result = extract_tool_context(history)
        assert "Special chars" in result

    def test_newlines_in_content(self):
        """Test handling of newlines in content."""
        history = [
            {
                "tool": "web_search",
                "result": {"success": True, "content": "Line 1\nLine 2\nLine 3 " * 10},
            }
        ]
        result = extract_tool_context(history)
        assert "Line 1" in result

    def test_content_at_exact_min_length(self):
        """Test content at exactly min_content_length."""
        content = "x" * 50  # Exactly 50 chars (default min)
        history = [{"tool": "web_search", "result": {"success": True, "content": content}}]
        result = get_relevant_tool_results(history, min_content_length=50)
        assert len(result) == 1

    def test_content_below_min_length(self):
        """Test content below min_content_length."""
        content = "x" * 49  # Just below 50 chars
        history = [{"tool": "web_search", "result": {"success": True, "content": content}}]
        result = get_relevant_tool_results(history, min_content_length=50)
        assert len(result) == 0

    def test_content_at_exact_max_length(self):
        """Test content at exactly max_content_length (no truncation needed)."""
        content = "x" * 300  # Exactly default max
        history = [{"tool": "web_search", "result": {"success": True, "content": content}}]
        result = extract_tool_context(history, max_content_length=300)
        # Should not have truncation indicator
        assert "..." not in result

    def test_content_just_above_max_length(self):
        """Test content just above max_content_length."""
        content = "x" * 301  # Just above default max
        history = [{"tool": "web_search", "result": {"success": True, "content": content}}]
        result = extract_tool_context(history, max_content_length=300)
        # Should have truncation indicator
        assert "..." in result
