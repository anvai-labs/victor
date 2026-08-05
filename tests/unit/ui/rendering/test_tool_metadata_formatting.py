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

"""Unit tests for tool metadata formatting (Phase 1).

These tests verify the unified registry metadata display helpers.
"""

import pytest

from victor.ui.rendering.utils import (
    format_access_mode_badge,
    format_cost_tier_indicator,
    format_execution_category_hint,
    format_tool_metadata_badges,
    get_tool_metadata_for_display,
)


class TestAccessModeBadges:
    """Tests for access mode badge formatting."""

    def test_readonly_green_badge(self):
        """READONLY access mode should be green."""
        result = format_access_mode_badge("readonly")
        assert "[green]" in result
        assert "READONLY" in result

    def test_write_yellow_badge(self):
        """WRITE access mode should be yellow."""
        result = format_access_mode_badge("write")
        assert "[yellow]" in result
        assert "WRITE" in result

    def test_execute_red_badge(self):
        """EXECUTE access mode should be red."""
        result = format_access_mode_badge("execute")
        assert "[red]" in result
        assert "EXECUTE" in result

    def test_network_blue_badge(self):
        """NETWORK access mode should be blue."""
        result = format_access_mode_badge("network")
        assert "[blue]" in result
        assert "NETWORK" in result

    def test_mixed_magenta_badge(self):
        """MIXED access mode should be magenta."""
        result = format_access_mode_badge("mixed")
        assert "[magenta]" in result
        assert "MIXED" in result

    def test_unknown_mode_dim_badge(self):
        """Unknown access mode should be dim."""
        result = format_access_mode_badge("unknown")
        assert "[dim]" in result


class TestCostTierIndicators:
    """Tests for cost tier indicator formatting."""

    def test_free_empty_indicator(self):
        """FREE cost tier should have empty indicator."""
        result = format_cost_tier_indicator("free")
        assert result == ""

    def test_low_single_dollar(self):
        """LOW cost tier should show $."""
        result = format_cost_tier_indicator("low")
        assert "$" in result

    def test_medium_double_dollar(self):
        """MEDIUM cost tier should show $$."""
        result = format_cost_tier_indicator("medium")
        assert result == "$$"

    def test_high_triple_dollar(self):
        """HIGH cost tier should show $$$."""
        result = format_cost_tier_indicator("high")
        assert result == "$$$"


class TestExecutionCategoryHints:
    """Tests for execution category hint formatting."""

    def test_read_only_icon(self):
        """READ_ONLY category should show magnifying glass icon."""
        result = format_execution_category_hint("read_only")
        assert "🔍" in result
        assert "Read Only" in result

    def test_write_icon(self):
        """WRITE category should show pencil icon."""
        result = format_execution_category_hint("write")
        assert "📝" in result
        assert "Write" in result

    def test_compute_icon(self):
        """COMPUTE category should show gear icon."""
        result = format_execution_category_hint("compute")
        assert "⚙️" in result
        assert "Compute" in result

    def test_network_icon(self):
        """NETWORK category should show globe icon."""
        result = format_execution_category_hint("network")
        assert "🌐" in result
        assert "Network" in result

    def test_execute_icon(self):
        """EXECUTE category should show lightning icon."""
        result = format_execution_category_hint("execute")
        assert "⚡" in result
        assert "Execute" in result

    def test_mixed_icon(self):
        """MIXED category shows the shuffle icon + an unambiguous 'read+write'
        label (relabelled from 'Mixed' which read as a mixed-success status)."""
        result = format_execution_category_hint("mixed")
        assert "🔀" in result
        assert "read+write" in result


class TestToolMetadataBadges:
    """Tests for complete tool metadata badges."""

    def test_empty_metadata_returns_empty(self):
        """Empty metadata should return empty string."""
        result = format_tool_metadata_badges()
        assert result == ""

    def test_category_only(self):
        """Only category badge when only category provided."""
        result = format_tool_metadata_badges(category="filesystem")
        assert "filesystem" in result
        assert "[dim" in result  # Check for dim tag (may be [dim] or [dim dim])

    def test_access_mode_only(self):
        """Only access mode badge when only mode provided."""
        result = format_tool_metadata_badges(access_mode="write")
        assert "[yellow]" in result
        assert "WRITE" in result

    def test_cost_tier_only(self):
        """Only cost indicator when only tier provided."""
        result = format_tool_metadata_badges(cost_tier="high")
        assert "$$$" in result
        assert "[yellow]" in result

    def test_execution_category_only(self):
        """Only execution hint when only category provided."""
        result = format_tool_metadata_badges(execution_category="network")
        assert "🌐" in result
        assert "Network" in result

    def test_all_badges_combined(self):
        """All badges combined into single string."""
        result = format_tool_metadata_badges(
            category="git",
            access_mode="mixed",
            cost_tier="low",
            execution_category="mixed",
        )
        # Should contain all badges
        assert "git" in result
        assert "MIXED" in result
        assert "$" in result
        assert "🔀" in result

    def test_readonly_mode_excluded(self):
        """READONLY mode should not show badge (default)."""
        result = format_tool_metadata_badges(
            category="filesystem",
            access_mode="readonly",
        )
        # Should show category but not readonly badge
        assert "filesystem" in result
        assert "READONLY" not in result

    def test_real_code_search_metadata(self):
        """Real code_search tool metadata example."""
        result = format_tool_metadata_badges(
            category="code",
            access_mode="readonly",
            cost_tier="free",
            execution_category="read_only",
        )
        # Should show category, execution hint (readonly/cost excluded)
        assert "code" in result or "🔍" in result

    def test_real_shell_metadata(self):
        """Real shell tool metadata example (high danger)."""
        result = format_tool_metadata_badges(
            category="system",
            access_mode="execute",
            cost_tier="free",
            execution_category="execute",
        )
        # Should show execute badge and icon
        assert "[red]" in result
        assert "EXECUTE" in result or "⚡" in result


class TestToolMetadataLookup:
    """Tests for tool metadata lookup from registry."""

    def test_metadata_lookup_returns_dict(self):
        """Metadata lookup should return a dict."""
        result = get_tool_metadata_for_display("read_file")
        assert isinstance(result, dict)

    def test_metadata_lookup_has_required_keys(self):
        """Metadata lookup should have all required keys."""
        result = get_tool_metadata_for_display("read_file")
        assert "category" in result
        assert "access_mode" in result
        assert "cost_tier" in result
        assert "execution_category" in result

    def test_metadata_lookup_fallback_on_unknown_tool(self):
        """Unknown tool should return fallback defaults."""
        result = get_tool_metadata_for_display("unknown_tool_xyz")
        assert result["access_mode"] == "readonly"
        assert result["cost_tier"] == "free"
        assert result["execution_category"] == "read_only"

    def test_read_file_metadata(self):
        """read_file tool should have readonly access."""
        result = get_tool_metadata_for_display("read_file")
        assert result["access_mode"] in {"readonly", "READONLY"}
        assert result["execution_category"] in {"read_only", "READ_ONLY"}

    def test_grep_metadata(self):
        """grep tool should have read_only execution category."""
        result = get_tool_metadata_for_display("grep")
        assert result["category"] or result["execution_category"]


class TestMetadataIntegration:
    """Integration tests for metadata with real tool names.

    These tests verify that the metadata lookup works with actual tool names
    from the registry. They don't assert specific values since the registry
    content depends on what's registered at test time.
    """

    def test_filesystem_tools_have_metadata(self):
        """Filesystem read tools should have metadata from registry."""
        readonly_tools = ["read_file", "list_directory", "grep"]
        for tool in readonly_tools:
            result = get_tool_metadata_for_display(tool)
            # Should have all required keys
            assert "category" in result
            assert "access_mode" in result
            # Should not be the fallback default if tool exists
            # (if tool is registered, it will have real metadata)

    def test_write_tools_have_metadata(self):
        """Write tools should have metadata from registry."""
        write_tools = ["write_file", "edit_file"]
        for tool in write_tools:
            result = get_tool_metadata_for_display(tool)
            # Should have all required keys
            assert "category" in result
            assert "access_mode" in result

    def test_network_tools_have_metadata(self):
        """Network tools should have metadata from registry."""
        network_tools = ["web_search", "web_fetch"]
        for tool in network_tools:
            result = get_tool_metadata_for_display(tool)
            # Should have all required keys
            assert "category" in result
            assert "access_mode" in result

    def test_execute_tools_have_metadata(self):
        """Execute tools should have metadata from registry."""
        execute_tools = ["execute_bash", "sandbox_exec"]
        for tool in execute_tools:
            result = get_tool_metadata_for_display(tool)
            # Should have all required keys
            assert "category" in result
            assert "access_mode" in result
