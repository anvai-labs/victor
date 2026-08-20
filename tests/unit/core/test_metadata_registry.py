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

"""Tests for ToolMetadataRegistry keyword-based tool selection.

Tests cover:
- Keyword extraction from tool decorators
- Keyword indexing in registry
- Keyword-based tool lookup
- Text-based tool matching
"""

import pytest
from unittest.mock import MagicMock

from victor.tools.metadata_registry import (
    ToolMetadataRegistry,
    ToolMetadataEntry,
    get_global_registry,
)
from victor.tools.base import AccessMode, CostTier, DangerLevel, Priority


from victor.tools.base import ExecutionCategory


class MockTool:
    """Mock tool for testing metadata extraction."""

    def __init__(
        self,
        name: str,
        description: str = "Test tool",
        category: str = None,
        keywords: list = None,
        stages: list = None,
        priority: Priority = Priority.MEDIUM,
        access_mode: AccessMode = AccessMode.READONLY,
        danger_level: DangerLevel = DangerLevel.SAFE,
        cost_tier: CostTier = CostTier.FREE,
        # NEW: Semantic selection attributes
        mandatory_keywords: list = None,
        task_types: list = None,
        signature_params: list = None,
        execution_category: ExecutionCategory = None,
    ):
        self.name = name
        self.description = description
        self.category = category
        self.keywords = keywords or []
        self.stages = stages or []
        self.priority = priority
        self.access_mode = access_mode
        self.danger_level = danger_level
        self.cost_tier = cost_tier
        self.aliases = set()
        # NEW: Semantic selection attributes
        self.mandatory_keywords = mandatory_keywords or []
        self.task_types = task_types or []
        self.signature_params = signature_params or []
        self.execution_category = execution_category or ExecutionCategory.READ_ONLY


class TestToolMetadataEntry:
    """Tests for ToolMetadataEntry dataclass."""

    def test_from_tool_extracts_keywords(self):
        """Test that keywords are extracted from tool."""
        tool = MockTool(
            name="scan",
            keywords=["security", "vulnerability", "audit"],
        )
        entry = ToolMetadataEntry.from_tool(tool)

        assert entry.keywords == {"security", "vulnerability", "audit"}

    def test_from_tool_lowercases_keywords(self):
        """Test that keywords are lowercased."""
        tool = MockTool(
            name="scan",
            keywords=["Security", "VULNERABILITY", "Audit"],
        )
        entry = ToolMetadataEntry.from_tool(tool)

        assert entry.keywords == {"security", "vulnerability", "audit"}

    def test_from_tool_handles_empty_keywords(self):
        """Test that empty keywords list is handled."""
        tool = MockTool(name="test", keywords=[])
        entry = ToolMetadataEntry.from_tool(tool)

        assert entry.keywords == set()

    def test_from_tool_handles_none_keywords(self):
        """Test that None keywords is handled."""
        tool = MockTool(name="test", keywords=None)
        entry = ToolMetadataEntry.from_tool(tool)

        assert entry.keywords == set()

    def test_from_tool_extracts_category(self):
        """Test that category is extracted from tool."""
        tool = MockTool(name="scan", category="security")
        entry = ToolMetadataEntry.from_tool(tool)

        assert entry.category == "security"


class TestToolMetadataRegistryKeywords:
    """Tests for keyword indexing and lookup."""

    def test_register_indexes_keywords(self):
        """Test that keywords are indexed on registration."""
        registry = ToolMetadataRegistry()
        tool = MockTool(
            name="scan",
            keywords=["security", "vulnerability"],
        )

        registry.register(tool)

        assert "security" in registry._by_keyword
        assert "vulnerability" in registry._by_keyword
        assert "scan" in registry._by_keyword["security"]
        assert "scan" in registry._by_keyword["vulnerability"]

    def test_get_by_keyword_single(self):
        """Test getting tools by single keyword."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security"]))
        registry.register(MockTool(name="audit", keywords=["security", "compliance"]))

        entries = registry.get_by_keyword("security")

        assert len(entries) == 2
        names = {e.name for e in entries}
        assert names == {"scan", "audit"}

    def test_get_by_keyword_case_insensitive(self):
        """Test that keyword lookup is case-insensitive."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security"]))

        entries_lower = registry.get_by_keyword("security")
        entries_upper = registry.get_by_keyword("SECURITY")
        entries_mixed = registry.get_by_keyword("Security")

        assert len(entries_lower) == 1
        assert len(entries_upper) == 1
        assert len(entries_mixed) == 1

    def test_get_by_keyword_not_found(self):
        """Test that unknown keyword returns empty list."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security"]))

        entries = registry.get_by_keyword("unknown")

        assert entries == []

    def test_get_tools_by_keywords_multiple(self):
        """Test getting tools matching any of multiple keywords."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security"]))
        registry.register(MockTool(name="audit", keywords=["compliance"]))
        registry.register(MockTool(name="review", keywords=["quality"]))

        tools = registry.get_tools_by_keywords(["security", "compliance"])

        assert tools == {"scan", "audit"}

    def test_get_tools_by_keywords_empty_list(self):
        """Test that empty keyword list returns empty set."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security"]))

        tools = registry.get_tools_by_keywords([])

        assert tools == set()

    def test_get_tools_matching_text(self):
        """Test finding tools whose keywords appear in text."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security", "vulnerability"]))
        registry.register(MockTool(name="test", keywords=["pytest", "unittest"]))
        registry.register(MockTool(name="doc", keywords=["document", "docstring"]))

        tools = registry.get_tools_matching_text("run a security scan of the repo")

        assert "scan" in tools

    def test_get_tools_matching_text_case_insensitive(self):
        """Test that text matching is case-insensitive."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security"]))

        tools = registry.get_tools_matching_text("Run SECURITY Scan")

        assert "scan" in tools

    def test_get_tools_matching_text_no_match(self):
        """Test that non-matching text returns empty set."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security"]))

        tools = registry.get_tools_matching_text("hello world")

        assert tools == set()


class TestToolMetadataRegistryCompatibility:
    """Tests for singleton and compatibility helpers."""

    def teardown_method(self):
        ToolMetadataRegistry.reset_instance()

    def test_get_instance_returns_global_registry(self):
        """get_instance should delegate to the module-global singleton."""
        instance = ToolMetadataRegistry.get_instance()

        assert instance is get_global_registry()

    def test_refresh_from_tools_populates_singleton_registry(self):
        """refresh_from_tools should rebuild the singleton registry state."""
        registry = ToolMetadataRegistry.get_instance()
        tool = MockTool(name="scan", category="security", keywords=["security", "audit"])

        refreshed = registry.refresh_from_tools([tool], force=True)

        assert refreshed is True
        assert registry.get_metadata("scan") is not None
        stats = registry.get_statistics()
        assert stats["total_tools"] == 1
        assert stats["total_categories"] == 1
        assert stats["total_keywords"] == 2

    def test_refresh_from_tools_skips_unchanged_toolset(self):
        """refresh_from_tools should avoid rebuilding unchanged metadata."""
        registry = ToolMetadataRegistry.get_instance()
        tools = [MockTool(name="scan", category="security", keywords=["security"])]

        first = registry.refresh_from_tools(tools, force=True)
        second = registry.refresh_from_tools(tools, force=False)

        assert first is True
        assert second is False
        assert registry.needs_reindex(tools) is False

    def test_get_all_keywords(self):
        """Test getting all registered keywords."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security", "vulnerability"]))
        registry.register(MockTool(name="test", keywords=["pytest"]))

        keywords = registry.get_all_keywords()

        assert keywords == {"security", "vulnerability", "pytest"}


class TestToolMetadataRegistryCategories:
    """Tests for category-based lookup."""

    def test_get_by_category(self):
        """Test getting tools by category."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", category="security"))
        registry.register(MockTool(name="audit", category="security"))
        registry.register(MockTool(name="test", category="testing"))

        entries = registry.get_by_category("security")

        assert len(entries) == 2
        names = {e.name for e in entries}
        assert names == {"scan", "audit"}

    def test_get_by_category_not_found(self):
        """Test that unknown category returns empty list."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", category="security"))

        entries = registry.get_by_category("unknown")

        assert entries == []


class TestToolMetadataRegistrySummary:
    """Tests for registry summary and statistics."""

    def test_summary_includes_keywords(self):
        """Test that summary includes keyword counts."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security", "vulnerability"]))
        registry.register(MockTool(name="test", keywords=["pytest"]))

        summary = registry.summary()

        assert summary["total"] == 2

    def test_len_returns_tool_count(self):
        """Test that len() returns number of registered tools."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan"))
        registry.register(MockTool(name="test"))

        assert len(registry) == 2

    def test_contains_checks_name(self):
        """Test that 'in' operator checks for tool name."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan"))

        assert "scan" in registry
        assert "unknown" not in registry


class TestGlobalRegistry:
    """Tests for global registry singleton."""

    def test_get_global_registry_returns_same_instance(self):
        """Test that get_global_registry returns singleton."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()

        assert registry1 is registry2


class TestKeywordMatchResult:
    """Tests for KeywordMatchResult dataclass."""

    def test_total_score_combines_score_and_boost(self):
        """Test that total_score includes priority boost."""
        from victor.tools.metadata_registry import KeywordMatchResult

        result = KeywordMatchResult(
            tool_name="test",
            score=0.5,
            matched_keywords={"keyword"},
            priority_boost=0.2,
        )

        assert result.total_score == 0.7

    def test_total_score_with_zero_boost(self):
        """Test total_score with no priority boost."""
        from victor.tools.metadata_registry import KeywordMatchResult

        result = KeywordMatchResult(
            tool_name="test",
            score=0.8,
            matched_keywords={"a", "b"},
            priority_boost=0.0,
        )

        assert result.total_score == 0.8


class TestMatchingMetrics:
    """Tests for MatchingMetrics dataclass."""

    def test_initial_metrics_zero(self):
        """Test that initial metrics are zero."""
        from victor.tools.metadata_registry import MatchingMetrics

        metrics = MatchingMetrics()

        assert metrics.total_queries == 0
        assert metrics.total_matches == 0
        assert metrics.empty_results == 0
        assert metrics.fallback_used == 0

    def test_record_match_updates_counts(self):
        """Test that record_match updates all counts."""
        from victor.tools.metadata_registry import MatchingMetrics

        metrics = MatchingMetrics()
        metrics.record_match(
            duration_ms=10.0,
            tools_matched=3,
            keywords_matched={"security", "scan"},
        )

        assert metrics.total_queries == 1
        assert metrics.total_matches == 3
        assert metrics.avg_match_time_ms == 10.0
        assert metrics.keyword_hit_counts == {"security": 1, "scan": 1}

    def test_record_match_tracks_empty_results(self):
        """Test that empty results are counted."""
        from victor.tools.metadata_registry import MatchingMetrics

        metrics = MatchingMetrics()
        metrics.record_match(
            duration_ms=5.0,
            tools_matched=0,
            keywords_matched=set(),
        )

        assert metrics.empty_results == 1

    def test_record_match_tracks_fallback(self):
        """Test that fallback usage is tracked."""
        from victor.tools.metadata_registry import MatchingMetrics

        metrics = MatchingMetrics()
        metrics.record_match(
            duration_ms=5.0,
            tools_matched=2,
            keywords_matched=set(),
            used_fallback=True,
        )

        assert metrics.fallback_used == 1

    def test_record_category_hit(self):
        """Test category hit tracking."""
        from victor.tools.metadata_registry import MatchingMetrics

        metrics = MatchingMetrics()
        metrics.record_category_hit("security")
        metrics.record_category_hit("security")
        metrics.record_category_hit("testing")

        assert metrics.category_hit_counts == {"security": 2, "testing": 1}

    def test_to_dict_returns_summary(self):
        """Test to_dict returns proper summary."""
        from victor.tools.metadata_registry import MatchingMetrics

        metrics = MatchingMetrics()
        metrics.record_match(duration_ms=10.0, tools_matched=5, keywords_matched={"test"})

        result = metrics.to_dict()

        assert "total_queries" in result
        assert "avg_match_time_ms" in result
        assert "top_keywords" in result
        assert result["total_queries"] == 1


class TestScoredKeywordMatching:
    """Tests for scored keyword matching."""

    def test_get_tools_matching_text_scored_returns_list(self):
        """Test that scored matching returns list of results."""
        from victor.tools.metadata_registry import KeywordMatchResult

        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security", "vulnerability"]))

        results = registry.get_tools_matching_text_scored("run security scan")

        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], KeywordMatchResult)

    def test_scored_matching_includes_matched_keywords(self):
        """Test that results include matched keywords."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security", "vulnerability"]))

        results = registry.get_tools_matching_text_scored("security vulnerability check")

        assert len(results) == 1
        assert "security" in results[0].matched_keywords
        assert "vulnerability" in results[0].matched_keywords

    def test_scored_matching_sorted_by_score(self):
        """Test that results are sorted by total score."""
        registry = ToolMetadataRegistry()
        # low_score_tool has many keywords but only 1 matches -> lower base score
        registry.register(
            MockTool(
                name="low_score_tool",
                keywords=["security", "vulnerability", "audit", "compliance", "scan"],
                priority=Priority.LOW,
            )
        )
        # high_score_tool has matching keyword + CRITICAL priority
        registry.register(
            MockTool(
                name="high_score_tool",
                keywords=["security"],
                priority=Priority.CRITICAL,
            )
        )

        results = registry.get_tools_matching_text_scored("security check")

        # high_score_tool should be first due to 100% keyword match + CRITICAL boost
        assert results[0].tool_name == "high_score_tool"

    def test_scored_matching_min_score_filter(self):
        """Test that min_score filters low-scoring results."""
        registry = ToolMetadataRegistry()
        # Use distinct keywords that won't accidentally match substrings
        registry.register(
            MockTool(
                name="scan",
                keywords=[
                    "vulnerability",
                    "exploit",
                    "penetration",
                    "assessment",
                    "audit",
                ],
            )
        )

        # Only matching 0 out of 5 keywords = 0.0 base score
        # With use_fallback=False, should return empty
        results = registry.get_tools_matching_text_scored(
            "random text without matching keywords",
            min_score=0.1,
            use_fallback=False,
        )

        # Should be filtered out due to no matches
        assert len(results) == 0

    def test_scored_matching_max_results_limit(self):
        """Test that max_results limits output."""
        registry = ToolMetadataRegistry()
        for i in range(10):
            registry.register(MockTool(name=f"tool_{i}", keywords=["common"]))

        results = registry.get_tools_matching_text_scored("common keyword", max_results=3)

        assert len(results) == 3

    def test_scored_matching_fallback_when_no_match(self):
        """Test that fallback tools are returned when no keyword match."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="read", keywords=["file"]))
        registry.register(MockTool(name="write", keywords=["file"]))

        # Query with no matching keywords
        results = registry.get_tools_matching_text_scored("xyz abc 123", use_fallback=True)

        # Should return fallback tools
        names = {r.tool_name for r in results}
        assert "read" in names  # read is in fallback set

    def test_scored_matching_no_fallback_option(self):
        """Test that fallback can be disabled."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security"]))

        results = registry.get_tools_matching_text_scored("xyz", use_fallback=False)

        assert len(results) == 0


class TestCategoryDiscovery:
    """Tests for category discovery from tool decorators."""

    def test_get_all_categories(self):
        """Test getting all discovered categories."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", category="security"))
        registry.register(MockTool(name="test", category="testing"))
        registry.register(MockTool(name="doc", category="docs"))

        categories = registry.get_all_categories()

        assert categories == {"security", "testing", "docs"}

    def test_get_all_categories_empty_registry(self):
        """Test that empty registry returns empty set."""
        registry = ToolMetadataRegistry()

        categories = registry.get_all_categories()

        assert categories == set()

    def test_get_tools_by_category_with_fallback(self):
        """Test category lookup with fallback chain."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="test1", category="testing"))
        registry.register(MockTool(name="test2", category="testing"))

        tools = registry.get_tools_by_category_with_fallback(
            "unknown",
            fallback_categories=["testing"],
        )

        names = {t.name for t in tools}
        assert names == {"test1", "test2"}

    def test_category_fallback_uses_first_match(self):
        """Test that fallback uses first matching category."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="sec", category="security"))
        registry.register(MockTool(name="test", category="testing"))

        tools = registry.get_tools_by_category_with_fallback(
            "unknown",
            fallback_categories=["security", "testing"],
        )

        # Should return security tools, not testing
        assert len(tools) == 1
        assert tools[0].name == "sec"


class TestMetricsTracking:
    """Tests for metrics tracking and observability."""

    def test_metrics_property_returns_metrics(self):
        """Test that metrics property returns MatchingMetrics."""
        from victor.tools.metadata_registry import MatchingMetrics

        registry = ToolMetadataRegistry()

        assert isinstance(registry.metrics, MatchingMetrics)

    def test_reset_metrics_clears_all(self):
        """Test that reset_metrics clears all metrics."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security"]))

        # Generate some metrics
        registry.get_tools_matching_text_scored("security")
        assert registry.metrics.total_queries == 1

        # Reset
        registry.reset_metrics()

        assert registry.metrics.total_queries == 0

    def test_metrics_updated_on_scored_match(self):
        """Test that scored matching updates metrics."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="scan", keywords=["security"]))

        registry.get_tools_matching_text_scored("security scan")

        assert registry.metrics.total_queries == 1
        assert registry.metrics.total_matches >= 1
        assert "security" in registry.metrics.keyword_hit_counts


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_fallback_critical_tools(self):
        """Test getting fallback critical tools."""
        from victor.tools.metadata_registry import get_fallback_critical_tools

        fallback = get_fallback_critical_tools()

        assert "read" in fallback
        assert "write" in fallback
        assert "shell" in fallback

    def test_get_fallback_critical_tools_returns_copy(self):
        """Test that fallback returns a copy, not original."""
        from victor.tools.metadata_registry import get_fallback_critical_tools

        fallback1 = get_fallback_critical_tools()
        fallback1.add("custom_tool")

        fallback2 = get_fallback_critical_tools()

        assert "custom_tool" not in fallback2


class TestStageBasedToolSelection:
    """Tests for stage-based tool selection using @tool(stages=[...]) decorator."""

    def test_stage_extraction_from_tool(self):
        """Test that stages are extracted from tool and stored in entry."""
        tool = MockTool(
            name="read_file",
            stages=["reading", "analysis"],
        )
        entry = ToolMetadataEntry.from_tool(tool)

        assert "reading" in entry.stages
        assert "analysis" in entry.stages

    def test_stage_case_insensitive(self):
        """Test that stages are lowercased for consistent lookup."""
        tool = MockTool(
            name="test_tool",
            stages=["READING", "Execution", "analysis"],
        )
        entry = ToolMetadataEntry.from_tool(tool)

        assert "reading" in entry.stages
        assert "execution" in entry.stages
        assert "analysis" in entry.stages

    def test_registry_indexes_by_stage(self):
        """Test that registry indexes tools by stage."""
        registry = ToolMetadataRegistry()

        registry.register(MockTool(name="read", stages=["reading", "analysis"]))
        registry.register(MockTool(name="write", stages=["execution"]))
        registry.register(MockTool(name="grep", stages=["reading", "initial"]))

        reading_tools = registry.get_tools_by_stage("reading")
        assert "read" in reading_tools
        assert "grep" in reading_tools
        assert "write" not in reading_tools

        execution_tools = registry.get_tools_by_stage("execution")
        assert "write" in execution_tools
        assert "read" not in execution_tools

    def test_get_by_stage_returns_entries(self):
        """Test that get_by_stage returns ToolMetadataEntry objects."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="read", stages=["reading"]))

        entries = registry.get_by_stage("reading")

        assert len(entries) == 1
        assert entries[0].name == "read"
        assert isinstance(entries[0], ToolMetadataEntry)

    def test_get_all_stages(self):
        """Test that get_all_stages returns all discovered stages."""
        registry = ToolMetadataRegistry()

        registry.register(MockTool(name="read", stages=["reading", "analysis"]))
        registry.register(MockTool(name="write", stages=["execution"]))
        registry.register(MockTool(name="test", stages=["verification"]))

        all_stages = registry.get_all_stages()

        assert "reading" in all_stages
        assert "analysis" in all_stages
        assert "execution" in all_stages
        assert "verification" in all_stages

    def test_get_stage_tool_mapping(self):
        """Test that get_stage_tool_mapping returns complete mapping."""
        registry = ToolMetadataRegistry()

        registry.register(MockTool(name="read", stages=["reading", "analysis"]))
        registry.register(MockTool(name="write", stages=["execution"]))

        mapping = registry.get_stage_tool_mapping()

        assert "reading" in mapping
        assert "read" in mapping["reading"]
        assert "execution" in mapping
        assert "write" in mapping["execution"]

    def test_stage_lookup_case_insensitive(self):
        """Test that stage lookup is case-insensitive."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="read", stages=["reading"]))

        # All these should work
        assert "read" in registry.get_tools_by_stage("reading")
        assert "read" in registry.get_tools_by_stage("READING")
        assert "read" in registry.get_tools_by_stage("Reading")

    def test_empty_stages_not_indexed(self):
        """Test that tools without stages are not indexed by stage."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="tool_without_stages", stages=[]))

        # Should have no stages
        assert len(registry.get_all_stages()) == 0

    def test_summary_includes_stages(self):
        """Test that summary() includes stage counts."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="read", stages=["reading"]))
        registry.register(MockTool(name="write", stages=["execution"]))

        summary = registry.summary()

        assert "by_stage" in summary
        assert summary["by_stage"]["reading"] == 1
        assert summary["by_stage"]["execution"] == 1

    def test_module_level_get_tools_by_stage(self):
        """Test module-level get_tools_by_stage helper function."""
        from victor.tools.metadata_registry import (
            get_tools_by_stage,
            get_global_registry,
        )

        # Clear and setup global registry
        _ = get_global_registry()  # noqa: F841 - Test function exists
        # Note: We can't easily clear the global registry, so we just test the function exists
        # and returns a set
        result = get_tools_by_stage("reading")
        assert isinstance(result, set)

    def test_module_level_get_all_stages(self):
        """Test module-level get_all_stages helper function."""
        from victor.tools.metadata_registry import get_all_stages

        result = get_all_stages()
        assert isinstance(result, set)

    def test_module_level_get_stage_tool_mapping(self):
        """Test module-level get_stage_tool_mapping helper function."""
        from victor.tools.metadata_registry import get_stage_tool_mapping

        result = get_stage_tool_mapping()
        assert isinstance(result, dict)


class TestSemanticSelectionRegistry:
    """Tests for decorator-driven semantic selection registry features."""

    def test_mandatory_keywords_indexing(self):
        """Test that mandatory keywords are indexed correctly."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="diff",
                mandatory_keywords=["show diff", "compare files"],
            )
        )

        # Lookup by mandatory keyword
        assert "diff" in registry.get_tools_by_mandatory_keyword("show diff")
        assert "diff" in registry.get_tools_by_mandatory_keyword("compare files")
        # Case insensitive
        assert "diff" in registry.get_tools_by_mandatory_keyword("SHOW DIFF")

    def test_mandatory_keywords_text_matching(self):
        """Test matching mandatory keywords in text."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="diff",
                mandatory_keywords=["show diff", "compare files"],
            )
        )
        registry.register(
            MockTool(
                name="test_runner",
                mandatory_keywords=["run tests", "execute tests"],
            )
        )

        # Match in text
        matches = registry.get_tools_matching_mandatory_keywords("Please show diff between files")
        assert "diff" in matches
        assert "test_runner" not in matches

        # Multiple matches
        matches = registry.get_tools_matching_mandatory_keywords("show diff and run tests")
        assert "diff" in matches
        assert "test_runner" in matches

    def test_task_types_indexing(self):
        """Test that task types are indexed correctly."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="analyzer",
                task_types=["analysis", "search"],
            )
        )
        registry.register(
            MockTool(
                name="writer",
                task_types=["action", "edit"],
            )
        )

        # Lookup by task type
        assert "analyzer" in registry.get_tools_by_task_type("analysis")
        assert "analyzer" in registry.get_tools_by_task_type("search")
        assert "writer" in registry.get_tools_by_task_type("action")
        assert "writer" not in registry.get_tools_by_task_type("analysis")

    def test_task_type_tool_mapping(self):
        """Test getting complete task type to tools mapping."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="analyzer", task_types=["analysis"]))
        registry.register(MockTool(name="writer", task_types=["action"]))

        mapping = registry.get_task_type_tool_mapping()

        assert "analysis" in mapping
        assert "action" in mapping
        assert "analyzer" in mapping["analysis"]
        assert "writer" in mapping["action"]

    def test_execution_category_indexing(self):
        """Test that execution categories are indexed correctly."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="reader",
                execution_category=ExecutionCategory.READ_ONLY,
            )
        )
        registry.register(
            MockTool(
                name="writer",
                execution_category=ExecutionCategory.WRITE,
            )
        )
        registry.register(
            MockTool(
                name="fetcher",
                execution_category=ExecutionCategory.NETWORK,
            )
        )

        # Lookup by execution category
        assert "reader" in registry.get_tools_by_execution_category(ExecutionCategory.READ_ONLY)
        assert "writer" in registry.get_tools_by_execution_category(ExecutionCategory.WRITE)
        assert "fetcher" in registry.get_tools_by_execution_category(ExecutionCategory.NETWORK)

    def test_parallelizable_tools(self):
        """Test getting parallelizable tools."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="reader",
                execution_category=ExecutionCategory.READ_ONLY,
            )
        )
        registry.register(
            MockTool(
                name="writer",
                execution_category=ExecutionCategory.WRITE,
            )
        )
        registry.register(
            MockTool(
                name="fetcher",
                execution_category=ExecutionCategory.NETWORK,
            )
        )
        registry.register(
            MockTool(
                name="analyzer",
                execution_category=ExecutionCategory.COMPUTE,
            )
        )

        parallelizable = registry.get_parallelizable_tools()

        # READ_ONLY, COMPUTE, NETWORK are parallelizable
        assert "reader" in parallelizable
        assert "fetcher" in parallelizable
        assert "analyzer" in parallelizable
        # WRITE is NOT parallelizable
        assert "writer" not in parallelizable

    def test_conflicting_tools(self):
        """Test getting tools that conflict with a given tool."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="write1",
                execution_category=ExecutionCategory.WRITE,
            )
        )
        registry.register(
            MockTool(
                name="write2",
                execution_category=ExecutionCategory.WRITE,
            )
        )
        registry.register(
            MockTool(
                name="reader",
                execution_category=ExecutionCategory.READ_ONLY,
            )
        )

        # Write tools conflict with other write tools
        conflicts = registry.get_conflicting_tools("write1")
        assert "write2" in conflicts
        # But not with readers
        assert "reader" not in conflicts
        # And not with self
        assert "write1" not in conflicts

    def test_signature_params_lookup(self):
        """Test getting progress params for a tool."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="reader",
                signature_params=["path", "offset", "limit"],
            )
        )
        registry.register(
            MockTool(
                name="simple",
                signature_params=[],
            )
        )

        # Get progress params
        params = registry.get_signature_params("reader")
        assert "path" in params
        assert "offset" in params
        assert "limit" in params

        # Empty for tools without progress params
        assert len(registry.get_signature_params("simple")) == 0

        # Empty for non-existent tools
        assert len(registry.get_signature_params("nonexistent")) == 0

    def test_tools_with_signature_params(self):
        """Test getting all tools with progress params."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="reader",
                signature_params=["path", "offset"],
            )
        )
        registry.register(
            MockTool(
                name="searcher",
                signature_params=["query", "page"],
            )
        )
        registry.register(
            MockTool(
                name="simple",
                signature_params=[],
            )
        )

        tools_with_params = registry.get_tools_with_signature_params()

        assert "reader" in tools_with_params
        assert "searcher" in tools_with_params
        assert "simple" not in tools_with_params
        assert "path" in tools_with_params["reader"]
        assert "query" in tools_with_params["searcher"]

    def test_execution_category_mapping(self):
        """Test getting complete execution category to tools mapping."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="reader",
                execution_category=ExecutionCategory.READ_ONLY,
            )
        )
        registry.register(
            MockTool(
                name="writer",
                execution_category=ExecutionCategory.WRITE,
            )
        )

        mapping = registry.get_execution_category_mapping()

        assert "read_only" in mapping
        assert "write" in mapping
        assert "reader" in mapping["read_only"]
        assert "writer" in mapping["write"]

    def test_summary_includes_semantic_selection(self):
        """Test that summary() includes semantic selection statistics."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="tool1",
                task_types=["analysis"],
                execution_category=ExecutionCategory.READ_ONLY,
                mandatory_keywords=["analyze"],
                signature_params=["path"],
            )
        )
        registry.register(
            MockTool(
                name="tool2",
                task_types=["action"],
                execution_category=ExecutionCategory.WRITE,
            )
        )

        summary = registry.summary()

        assert "by_task_type" in summary
        assert summary["by_task_type"]["analysis"] == 1
        assert summary["by_task_type"]["action"] == 1

        assert "by_execution_category" in summary
        assert summary["by_execution_category"]["read_only"] == 1
        assert summary["by_execution_category"]["write"] == 1

        assert "mandatory_keywords_count" in summary
        assert summary["mandatory_keywords_count"] == 1

        assert "tools_with_signature_params" in summary
        assert summary["tools_with_signature_params"] == 1

        assert "parallelizable_tools" in summary

    def test_module_level_semantic_helpers(self):
        """Test module-level convenience functions for semantic selection."""
        from victor.tools.metadata_registry import (
            get_tools_matching_mandatory_keywords,
            get_tools_by_task_type,
            get_task_type_tool_mapping,
            get_tools_by_execution_category,
            get_parallelizable_tools,
            get_signature_params,
            get_execution_category_mapping,
        )

        # Just verify they exist and return correct types
        assert isinstance(get_tools_matching_mandatory_keywords("test"), set)
        assert isinstance(get_tools_by_task_type("analysis"), set)
        assert isinstance(get_task_type_tool_mapping(), dict)
        assert isinstance(get_tools_by_execution_category(ExecutionCategory.READ_ONLY), set)
        assert isinstance(get_parallelizable_tools(), set)
        assert isinstance(get_signature_params("any_tool"), set)
        assert isinstance(get_execution_category_mapping(), dict)


class TestCategoryKeywordsDiscovery:
    """Tests for category keywords discovery from tool decorators."""

    def test_get_tools_by_category_name(self):
        """Test getting tool names by category."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="git_commit", category="git"))
        registry.register(MockTool(name="git_push", category="git"))
        registry.register(MockTool(name="pytest", category="testing"))

        git_tools = registry.get_tools_by_category("git")

        assert git_tools == {"git_commit", "git_push"}

    def test_get_tools_by_category_returns_copy(self):
        """Test that get_tools_by_category returns a copy."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="git", category="git"))

        tools = registry.get_tools_by_category("git")
        tools.add("fake_tool")

        # Original should not be modified
        assert "fake_tool" not in registry.get_tools_by_category("git")

    def test_get_tools_by_category_nonexistent(self):
        """Test that nonexistent category returns empty set."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="git", category="git"))

        tools = registry.get_tools_by_category("nonexistent")

        assert tools == set()

    def test_get_category_keywords(self):
        """Test getting aggregated keywords from tools in a category."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="git_commit",
                category="git",
                keywords=["commit", "save", "changes"],
            )
        )
        registry.register(
            MockTool(
                name="git_push",
                category="git",
                keywords=["push", "remote", "upload"],
            )
        )
        registry.register(
            MockTool(
                name="pytest",
                category="testing",
                keywords=["test", "unit"],
            )
        )

        git_keywords = registry.get_category_keywords("git")

        assert "commit" in git_keywords
        assert "save" in git_keywords
        assert "push" in git_keywords
        assert "remote" in git_keywords
        # Not from testing category
        assert "test" not in git_keywords

    def test_get_category_keywords_nonexistent(self):
        """Test that nonexistent category returns empty set."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="git", category="git", keywords=["commit"]))

        keywords = registry.get_category_keywords("nonexistent")

        assert keywords == set()

    def test_get_all_category_keywords(self):
        """Test getting complete category to keywords mapping."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="git_commit",
                category="git",
                keywords=["commit", "changes"],
            )
        )
        registry.register(
            MockTool(
                name="pytest",
                category="testing",
                keywords=["test", "unit"],
            )
        )

        all_keywords = registry.get_all_category_keywords()

        assert "git" in all_keywords
        assert "testing" in all_keywords
        assert "commit" in all_keywords["git"]
        assert "test" in all_keywords["testing"]

    def test_detect_categories_from_text(self):
        """Test detecting categories from keyword matches in text."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="git",
                category="git",
                keywords=["commit", "push", "branch"],
            )
        )
        registry.register(
            MockTool(
                name="test",
                category="testing",
                keywords=["test", "pytest", "unittest"],
            )
        )

        # Detect git category
        detected = registry.detect_categories_from_text("I want to commit and push my changes")
        assert "git" in detected

        # Detect testing category
        detected = registry.detect_categories_from_text("run the pytest suite")
        assert "testing" in detected

        # Detect multiple categories
        detected = registry.detect_categories_from_text("commit and then run test")
        assert "git" in detected
        assert "testing" in detected

    def test_detect_categories_from_text_case_insensitive(self):
        """Test that category detection is case insensitive."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="git",
                category="git",
                keywords=["commit"],
            )
        )

        detected = registry.detect_categories_from_text("COMMIT the changes")
        assert "git" in detected

    def test_detect_categories_from_text_no_match(self):
        """Test that no match returns empty set."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="git",
                category="git",
                keywords=["commit"],
            )
        )

        detected = registry.detect_categories_from_text("hello world")
        assert detected == set()


class TestModuleLevelCategoryHelpers:
    """Tests for module-level category convenience functions."""

    def test_get_tools_by_category_module(self):
        """Test module-level get_tools_by_category."""
        from victor.tools.metadata_registry import get_tools_by_category

        result = get_tools_by_category("git")
        assert isinstance(result, set)

    def test_get_all_categories_module(self):
        """Test module-level get_all_categories."""
        from victor.tools.metadata_registry import get_all_categories

        result = get_all_categories()
        assert isinstance(result, set)

    def test_get_category_keywords_module(self):
        """Test module-level get_category_keywords."""
        from victor.tools.metadata_registry import get_category_keywords

        result = get_category_keywords("git")
        assert isinstance(result, set)

    def test_get_all_category_keywords_module(self):
        """Test module-level get_all_category_keywords."""
        from victor.tools.metadata_registry import get_all_category_keywords

        result = get_all_category_keywords()
        assert isinstance(result, dict)

    def test_detect_categories_from_text_module(self):
        """Test module-level detect_categories_from_text."""
        from victor.tools.metadata_registry import detect_categories_from_text

        result = detect_categories_from_text("test text")
        assert isinstance(result, set)


class TestAccessModeBasedDiscovery:
    """Tests for access mode based tool discovery."""

    def test_get_write_tools(self):
        """Test getting tools that modify state."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="read", access_mode=AccessMode.READONLY))
        registry.register(MockTool(name="write", access_mode=AccessMode.WRITE))
        registry.register(MockTool(name="shell", access_mode=AccessMode.EXECUTE))
        registry.register(MockTool(name="mixed", access_mode=AccessMode.MIXED))

        write_tools = registry.get_write_tools()

        assert "write" in write_tools
        assert "shell" in write_tools
        assert "mixed" in write_tools
        assert "read" not in write_tools

    def test_get_idempotent_tools(self):
        """Test getting idempotent (readonly) tools."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="read", access_mode=AccessMode.READONLY))
        registry.register(MockTool(name="search", access_mode=AccessMode.READONLY))
        registry.register(MockTool(name="write", access_mode=AccessMode.WRITE))

        idempotent = registry.get_idempotent_tools()

        assert "read" in idempotent
        assert "search" in idempotent
        assert "write" not in idempotent

    def test_get_cache_invalidating_tools(self):
        """Test getting cache invalidating tools."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="read", access_mode=AccessMode.READONLY))
        registry.register(MockTool(name="write", access_mode=AccessMode.WRITE))

        invalidating = registry.get_cache_invalidating_tools()

        assert "write" in invalidating
        assert "read" not in invalidating


class TestModuleLevelAccessModeHelpers:
    """Tests for module-level access mode convenience functions."""

    def test_get_write_tools_module(self):
        """Test module-level get_write_tools."""
        from victor.tools.metadata_registry import get_write_tools

        result = get_write_tools()
        assert isinstance(result, set)

    def test_get_idempotent_tools_module(self):
        """Test module-level get_idempotent_tools."""
        from victor.tools.metadata_registry import get_idempotent_tools

        result = get_idempotent_tools()
        assert isinstance(result, set)

    def test_get_cache_invalidating_tools_module(self):
        """Test module-level get_cache_invalidating_tools."""
        from victor.tools.metadata_registry import get_cache_invalidating_tools

        result = get_cache_invalidating_tools()
        assert isinstance(result, set)


class TestAdvancedToolQueries:
    """Tests for advanced tool query methods."""

    def test_get_tools_for_task_classification(self):
        """Test getting tool entries for task classification."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="analyzer",
                task_types=["analysis"],
            )
        )
        registry.register(
            MockTool(
                name="searcher",
                task_types=["analysis", "search"],
            )
        )

        entries = registry.get_tools_for_task_classification("analysis")

        assert len(entries) == 2
        names = {e.name for e in entries}
        assert "analyzer" in names
        assert "searcher" in names

    def test_get_all_task_types(self):
        """Test getting all registered task types."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="t1", task_types=["analysis"]))
        registry.register(MockTool(name="t2", task_types=["action", "edit"]))

        types = registry.get_all_task_types()

        assert "analysis" in types
        assert "action" in types
        assert "edit" in types

    def test_get_all_mandatory_keywords(self):
        """Test getting all registered mandatory keywords."""
        registry = ToolMetadataRegistry()
        registry.register(
            MockTool(
                name="git",
                mandatory_keywords=["show diff", "git status"],
            )
        )
        registry.register(
            MockTool(
                name="test",
                mandatory_keywords=["run tests"],
            )
        )

        keywords = registry.get_all_mandatory_keywords()

        assert "show diff" in keywords
        assert "git status" in keywords
        assert "run tests" in keywords

    def test_filter_with_danger_level(self):
        """Test filtering by danger level."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="safe", danger_level=DangerLevel.SAFE))
        registry.register(MockTool(name="medium", danger_level=DangerLevel.MEDIUM))
        registry.register(MockTool(name="high", danger_level=DangerLevel.HIGH))

        filtered = registry.filter(danger_level=DangerLevel.HIGH)

        assert len(filtered) == 1
        assert filtered[0].name == "high"

    def test_get_tools_up_to_priority(self):
        """Test getting tools up to a priority level."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="critical", priority=Priority.CRITICAL))
        registry.register(MockTool(name="high", priority=Priority.HIGH))
        registry.register(MockTool(name="medium", priority=Priority.MEDIUM))
        registry.register(MockTool(name="low", priority=Priority.LOW))

        # Get tools with priority HIGH or better (lower value = higher priority)
        tools = registry.get_tools_up_to_priority(Priority.HIGH)
        names = {t.name for t in tools}

        assert "critical" in names
        assert "high" in names
        assert "medium" not in names
        assert "low" not in names

    def test_iterator(self):
        """Test iterating over registry."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="t1"))
        registry.register(MockTool(name="t2"))
        registry.register(MockTool(name="t3"))

        names = [entry.name for entry in registry]

        assert len(names) == 3
        assert "t1" in names
        assert "t2" in names
        assert "t3" in names

    def test_get_by_stage_case_insensitive(self):
        """Test that get_by_stage is case insensitive."""
        registry = ToolMetadataRegistry()
        registry.register(MockTool(name="read", stages=["reading"]))

        entries = registry.get_by_stage("READING")
        assert len(entries) == 1

        entries = registry.get_by_stage("Reading")
        assert len(entries) == 1


class TestMetadataEntryProperties:
    """Tests for ToolMetadataEntry property methods."""

    def test_is_idempotent_property(self):
        """Test is_idempotent derived property."""
        entry = ToolMetadataEntry(
            name="test",
            category=None,
            priority=Priority.MEDIUM,
            access_mode=AccessMode.READONLY,
            danger_level=DangerLevel.SAFE,
            cost_tier=CostTier.FREE,
        )
        assert entry.is_idempotent is True

        entry2 = ToolMetadataEntry(
            name="test2",
            category=None,
            priority=Priority.MEDIUM,
            access_mode=AccessMode.WRITE,
            danger_level=DangerLevel.SAFE,
            cost_tier=CostTier.FREE,
        )
        assert entry2.is_idempotent is False

    def test_cache_invalidating_property(self):
        """Test cache_invalidating derived property."""
        readonly = ToolMetadataEntry(
            name="r",
            category=None,
            priority=Priority.MEDIUM,
            access_mode=AccessMode.READONLY,
            danger_level=DangerLevel.SAFE,
            cost_tier=CostTier.FREE,
        )
        assert readonly.cache_invalidating is False

        write = ToolMetadataEntry(
            name="w",
            category=None,
            priority=Priority.MEDIUM,
            access_mode=AccessMode.WRITE,
            danger_level=DangerLevel.SAFE,
            cost_tier=CostTier.FREE,
        )
        assert write.cache_invalidating is True

        execute = ToolMetadataEntry(
            name="e",
            category=None,
            priority=Priority.MEDIUM,
            access_mode=AccessMode.EXECUTE,
            danger_level=DangerLevel.SAFE,
            cost_tier=CostTier.FREE,
        )
        assert execute.cache_invalidating is True

        mixed = ToolMetadataEntry(
            name="m",
            category=None,
            priority=Priority.MEDIUM,
            access_mode=AccessMode.MIXED,
            danger_level=DangerLevel.SAFE,
            cost_tier=CostTier.FREE,
        )
        assert mixed.cache_invalidating is True


class TestToolMetadataRegistryMatchingText:
    """Tests for module-level get_tools_matching_text_scored."""

    def test_module_level_scored_matching(self):
        """Test module-level get_tools_matching_text_scored."""
        from victor.tools.metadata_registry import get_tools_matching_text_scored

        results = get_tools_matching_text_scored("test query", max_results=5)
        assert isinstance(results, list)

    def test_get_matching_metrics_module(self):
        """Test module-level get_matching_metrics."""
        from victor.tools.metadata_registry import get_matching_metrics

        metrics = get_matching_metrics()
        assert hasattr(metrics, "total_queries")
