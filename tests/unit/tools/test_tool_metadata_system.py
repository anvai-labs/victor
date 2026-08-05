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

"""Tests for the Tool Metadata System.

This test module covers:
1. ToolMetadata auto-generation from tool properties
2. ToolMetadataProvider contract (get_metadata())
3. @tool decorator with metadata parameters
4. ToolMetadataRegistry functionality
5. Smart reindexing with hash-based change detection
6. SemanticToolSelector integration with metadata
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from victor.tools.base import (
    BaseTool,
    CostTier,
    ToolMetadata,
    ToolMetadataProvider,
    ToolResult,
)
from victor.tools.decorators import tool
from victor.tools.metadata_registry import ToolMetadataRegistry

# =============================================================================
# Test Fixtures
# =============================================================================


class MockTool(BaseTool):
    """Simple mock tool for testing."""

    name = "mock_tool"
    description = "A mock tool for testing purposes."
    parameters = {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "First parameter"},
            "param2": {"type": "integer", "description": "Second parameter"},
        },
        "required": ["param1"],
    }

    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="mock result")


class MockToolWithExplicitMetadata(BaseTool):
    """Mock tool with explicit metadata."""

    name = "explicit_metadata_tool"
    description = "A tool with explicit metadata."
    parameters = {"type": "object", "properties": {}}

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            category="testing",
            keywords=["explicit", "test", "metadata"],
            use_cases=["testing explicit metadata", "validating contract"],
            examples=["use explicit metadata tool", "test with explicit metadata"],
            priority_hints=["Use for testing metadata system"],
        )

    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="explicit metadata result")


class GitMockTool(BaseTool):
    """Mock git tool to test category detection."""

    name = "git_commit"
    description = "Commit changes to git repository."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="committed")


class HighCostTool(BaseTool):
    """Mock high-cost tool."""

    name = "high_cost_operation"
    description = "A resource-intensive operation."
    parameters = {"type": "object", "properties": {}}

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.HIGH

    async def execute(self, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="expensive result")


# =============================================================================
# Test: ToolMetadata Auto-Generation
# =============================================================================


class TestToolMetadataAutoGeneration:
    """Tests for ToolMetadata.generate_from_tool() factory method."""

    def test_generate_from_tool_extracts_keywords_from_name(self):
        """Should extract keywords from tool name."""
        metadata = ToolMetadata.generate_from_tool(
            name="git_commit_changes",
            description="Commit changes.",
            parameters={},
        )
        assert "git" in metadata.keywords or "commit" in metadata.keywords

    def test_generate_from_tool_extracts_keywords_from_description(self):
        """Should extract significant words from description."""
        metadata = ToolMetadata.generate_from_tool(
            name="search_tool",
            description="Search and find files in the codebase.",
            parameters={},
        )
        # Should contain significant words from description
        assert any(kw in ["search", "files", "codebase"] for kw in metadata.keywords)

    def test_generate_from_tool_detects_git_category(self):
        """Should detect 'git' category from tool name."""
        metadata = ToolMetadata.generate_from_tool(
            name="git_push",
            description="Push changes.",
            parameters={},
        )
        assert metadata.category == "git"

    def test_generate_from_tool_detects_filesystem_category(self):
        """Should detect 'filesystem' category from tool name."""
        metadata = ToolMetadata.generate_from_tool(
            name="file_reader",
            description="Read files.",
            parameters={},
        )
        assert metadata.category == "filesystem"

    def test_generate_from_tool_detects_search_category(self):
        """Should detect 'search' category from tool name."""
        metadata = ToolMetadata.generate_from_tool(
            name="semantic_search",
            description="Semantic code search.",
            parameters={},
        )
        assert metadata.category == "search"

    def test_generate_from_tool_creates_use_cases(self):
        """Should generate use cases from name."""
        metadata = ToolMetadata.generate_from_tool(
            name="code_review",
            description="Review code changes.",
            parameters={},
        )
        assert len(metadata.use_cases) > 0
        assert any("code review" in uc.lower() for uc in metadata.use_cases)

    def test_generate_from_tool_creates_examples(self):
        """Should generate example requests."""
        metadata = ToolMetadata.generate_from_tool(
            name="web_search",
            description="Search the web.",
            parameters={},
        )
        assert len(metadata.examples) > 0

    def test_generate_from_tool_adds_cost_tier_hints(self):
        """Should add priority hints based on cost tier."""
        metadata = ToolMetadata.generate_from_tool(
            name="expensive_tool",
            description="Expensive operation.",
            parameters={},
            cost_tier=CostTier.HIGH,
        )
        assert len(metadata.priority_hints) > 0
        # Should contain warning about resource usage
        assert any(
            "resource" in hint.lower() or "sparingly" in hint.lower()
            for hint in metadata.priority_hints
        )

    def test_generate_from_tool_limits_keywords(self):
        """Should limit keywords to reasonable number."""
        long_description = " ".join(["significant word"] * 50)
        metadata = ToolMetadata.generate_from_tool(
            name="test_tool",
            description=long_description,
            parameters={},
        )
        assert len(metadata.keywords) <= 10


# =============================================================================
# Test: ToolMetadataProvider Contract
# =============================================================================


class TestToolMetadataProviderContract:
    """Tests for ToolMetadataProvider protocol implementation."""

    def test_basetool_implements_get_metadata(self):
        """BaseTool should implement get_metadata()."""
        tool = MockTool()
        assert hasattr(tool, "get_metadata")
        assert callable(tool.get_metadata)

    def test_get_metadata_returns_toolmetadata(self):
        """get_metadata() should return ToolMetadata instance."""
        tool = MockTool()
        metadata = tool.get_metadata()
        assert isinstance(metadata, ToolMetadata)

    def test_get_metadata_returns_explicit_when_defined(self):
        """get_metadata() should return explicit metadata when defined."""
        tool = MockToolWithExplicitMetadata()
        metadata = tool.get_metadata()
        assert metadata.category == "testing"
        assert "explicit" in metadata.keywords

    def test_get_metadata_auto_generates_when_no_explicit(self):
        """get_metadata() should auto-generate when metadata property is None."""
        tool = MockTool()
        metadata = tool.get_metadata()
        # Should have auto-generated content
        assert metadata.keywords or metadata.use_cases
        # Name should influence keywords
        assert "mock" in metadata.keywords or "mock tool" in metadata.keywords

    def test_protocol_runtime_checkable(self):
        """ToolMetadataProvider should be runtime checkable."""
        tool = MockTool()
        assert isinstance(tool, ToolMetadataProvider)


# =============================================================================
# Test: @tool Decorator with Metadata
# =============================================================================


class TestToolDecoratorMetadata:
    """Tests for @tool decorator metadata support."""

    def test_tool_decorator_without_metadata(self):
        """Tool decorator without metadata should auto-generate."""

        @tool
        def simple_tool(query: str):
            """A simple search tool."""
            return query

        tool_instance = simple_tool.Tool
        metadata = tool_instance.get_metadata()
        assert isinstance(metadata, ToolMetadata)
        # Should auto-generate from function name/docstring
        assert len(metadata.keywords) > 0

    def test_tool_decorator_with_explicit_metadata(self):
        """Tool decorator with metadata params should use explicit values."""

        @tool(
            category="custom",
            keywords=["custom", "explicit"],
            use_cases=["testing decorator metadata"],
            examples=["use custom tool"],
        )
        def custom_tool(param: str):
            """A custom tool with explicit metadata."""
            return param

        tool_instance = custom_tool.Tool
        metadata = tool_instance.get_metadata()

        assert metadata.category == "custom"
        assert "custom" in metadata.keywords
        assert "explicit" in metadata.keywords
        assert "testing decorator metadata" in metadata.use_cases

    def test_tool_decorator_partial_metadata(self):
        """Tool decorator with partial metadata should fill in gaps."""

        @tool(category="partial", keywords=["partial"])
        def partial_tool(data: str):
            """Tool with partial metadata."""
            return data

        tool_instance = partial_tool.Tool
        metadata = tool_instance.get_metadata()

        assert metadata.category == "partial"
        assert "partial" in metadata.keywords
        # Should still have the explicit metadata, not auto-generated
        assert metadata.use_cases == []  # Empty since not provided

    def test_tool_decorator_cost_tier_preserved(self):
        """Tool decorator should preserve cost tier in metadata."""

        @tool(cost_tier=CostTier.MEDIUM)
        def medium_cost_tool(query: str):
            """A medium-cost tool."""
            return query

        tool_instance = medium_cost_tool.Tool
        assert tool_instance.cost_tier == CostTier.MEDIUM


# =============================================================================
# Test: ToolMetadataRegistry
# =============================================================================


class TestToolMetadataRegistry:
    """Tests for ToolMetadataRegistry singleton."""

    def setup_method(self):
        """Reset singleton before each test."""
        ToolMetadataRegistry.reset_instance()

    def test_singleton_pattern(self):
        """Should return same instance on multiple calls."""
        registry1 = ToolMetadataRegistry.get_instance()
        registry2 = ToolMetadataRegistry.get_instance()
        assert registry1 is registry2

    def test_register_tool_adds_to_cache(self):
        """register_tool should add metadata to cache."""
        from victor.tools.metadata_registry import ToolMetadataEntry

        registry = ToolMetadataRegistry.get_instance()
        tool = MockTool()

        registry.register_tool(tool)

        metadata = registry.get_metadata("mock_tool")
        assert metadata is not None
        assert isinstance(metadata, ToolMetadataEntry)

    def test_register_tool_indexes_by_category(self):
        """register_tool should index by category."""
        registry = ToolMetadataRegistry.get_instance()
        tool = MockToolWithExplicitMetadata()

        registry.register_tool(tool)

        tools_in_category = registry.get_tools_by_category("testing")
        assert "explicit_metadata_tool" in tools_in_category

    def test_register_tool_indexes_by_keywords(self):
        """register_tool should index by keywords."""
        registry = ToolMetadataRegistry.get_instance()
        tool = MockToolWithExplicitMetadata()

        registry.register_tool(tool)

        tools_with_keyword = registry.get_by_keyword("explicit")
        assert any(entry.name == "explicit_metadata_tool" for entry in tools_with_keyword)

    def test_unregister_tool_removes_from_cache(self):
        """unregister_tool should remove metadata from cache."""
        registry = ToolMetadataRegistry.get_instance()
        tool = MockTool()

        registry.register_tool(tool)
        registry.unregister_tool("mock_tool")

        metadata = registry.get_metadata("mock_tool")
        assert metadata is None

    def test_refresh_from_tools_populates_registry(self):
        """refresh_from_tools should populate registry with all tools."""
        registry = ToolMetadataRegistry.get_instance()
        tools = [MockTool(), MockToolWithExplicitMetadata(), GitMockTool()]

        registry.refresh_from_tools(tools, force=True)

        assert registry.get_metadata("mock_tool") is not None
        assert registry.get_metadata("explicit_metadata_tool") is not None
        assert registry.get_metadata("git_commit") is not None

    def test_search_tools_finds_by_name(self):
        """get_tools_matching_text should find tools by partial name match."""
        registry = ToolMetadataRegistry.get_instance()
        registry.refresh_from_tools([MockTool(), GitMockTool()], force=True)

        results = registry.get_tools_matching_text("git")
        assert "git_commit" in results

    def test_search_tools_finds_by_keyword(self):
        """get_tools_matching_text should find tools by keyword match."""
        registry = ToolMetadataRegistry.get_instance()
        registry.register_tool(MockToolWithExplicitMetadata())

        results = registry.get_tools_matching_text("explicit")
        assert "explicit_metadata_tool" in results

    def test_get_all_categories(self):
        """get_all_categories should return unique categories."""
        registry = ToolMetadataRegistry.get_instance()
        tools = [MockToolWithExplicitMetadata(), GitMockTool()]
        registry.refresh_from_tools(tools, force=True)

        categories = registry.get_all_categories()
        assert "testing" in categories
        assert "git" in categories

    def test_get_all_returns_metadata_entries(self):
        """get_all should return metadata entries keyed by tool name."""
        registry = ToolMetadataRegistry.get_instance()
        registry.register_tool(MockToolWithExplicitMetadata())

        exported = registry.get_all()

        assert "explicit_metadata_tool" in exported
        assert exported["explicit_metadata_tool"].category == "testing"
        assert "explicit" in exported["explicit_metadata_tool"].keywords

    def test_get_statistics(self):
        """get_statistics should return useful stats."""
        registry = ToolMetadataRegistry.get_instance()
        tools = [MockTool(), MockToolWithExplicitMetadata()]
        registry.refresh_from_tools(tools, force=True)

        stats = registry.get_statistics()

        assert stats["total_tools"] == 2
        assert stats["total_categories"] >= 1
        assert stats["total_keywords"] >= 1

    def test_base_reexport_warns_and_uses_canonical_registry(self):
        """victor.tools.base should lazily expose the canonical registry class."""
        from victor.tools.metadata_registry import (
            ToolMetadataRegistry as CanonicalToolMetadataRegistry,
        )

        with pytest.deprecated_call(
            match="Direct import of ToolMetadataRegistry from victor.tools.base is deprecated"
        ):
            from victor.tools.base import (
                ToolMetadataRegistry as BaseToolMetadataRegistry,
            )

        assert BaseToolMetadataRegistry is CanonicalToolMetadataRegistry

    def test_package_reexport_uses_canonical_registry(self):
        """victor.tools should re-export the canonical registry class."""
        from victor.tools import ToolMetadataRegistry as PackageToolMetadataRegistry
        from victor.tools.metadata_registry import (
            ToolMetadataRegistry as CanonicalToolMetadataRegistry,
        )

        assert PackageToolMetadataRegistry is CanonicalToolMetadataRegistry

    def test_metadata_module_wrapper_warns_and_adapts_legacy_shape(self, monkeypatch):
        """victor.tools.metadata should remain a deprecated compatibility wrapper."""
        import victor.tools.metadata as tools_metadata
        from victor.tools.metadata import (
            ToolMetadataRegistry as LegacyToolMetadataRegistry,
        )
        from victor.tools.metadata_registry import (
            ToolMetadataRegistry as CanonicalToolMetadataRegistry,
        )
        import victor.tools.metadata as _legacy_metadata_mod

        # The wrapper warns once per process; any earlier test that touched the
        # legacy registry sets the flag and this test then sees no warning
        # (shard-order dependent — broke Test Shard 2 on the develop->main gate).
        monkeypatch.setattr(tools_metadata, "_LEGACY_METADATA_REGISTRY_WARNING_EMITTED", False)

        CanonicalToolMetadataRegistry.reset_instance()

        # The legacy-usage deprecation is emitted ONCE per process (guarded by a
        # module global). Under sharded test runs another test may trip it first,
        # so reset the flag here to guarantee this contract test observes the
        # warning regardless of execution order (isolation, not a product change).
        _legacy_metadata_mod._LEGACY_METADATA_REGISTRY_WARNING_EMITTED = False

        with pytest.deprecated_call(
            match="Importing ToolMetadataRegistry from victor.tools.metadata is deprecated"
        ):
            legacy_registry = LegacyToolMetadataRegistry.get_instance()

        legacy_registry.register_tool(MockToolWithExplicitMetadata())

        metadata = legacy_registry.get_metadata("explicit_metadata_tool")
        assert isinstance(metadata, ToolMetadata)
        assert metadata is not None
        assert metadata.category == "testing"
        assert (
            CanonicalToolMetadataRegistry.get_instance().get_metadata("explicit_metadata_tool")
            is not None
        )


# =============================================================================
# Test: Smart Reindexing with Hash-Based Change Detection
# =============================================================================


class TestSmartReindexing:
    """Tests for hash-based smart reindexing."""

    def setup_method(self):
        """Reset singleton before each test."""
        ToolMetadataRegistry.reset_instance()

    def test_needs_reindex_true_on_first_run(self):
        """needs_reindex should return True on first run."""
        registry = ToolMetadataRegistry.get_instance()
        tools = [MockTool()]

        assert registry.needs_reindex(tools) is True

    def test_needs_reindex_false_after_refresh(self):
        """needs_reindex should return False after refresh with same tools."""
        registry = ToolMetadataRegistry.get_instance()
        tools = [MockTool()]

        registry.refresh_from_tools(tools, force=True)

        assert registry.needs_reindex(tools) is False

    def test_needs_reindex_true_when_tools_added(self):
        """needs_reindex should return True when new tools added."""
        registry = ToolMetadataRegistry.get_instance()
        tools = [MockTool()]

        registry.refresh_from_tools(tools, force=True)

        # Add a new tool
        new_tools = [MockTool(), GitMockTool()]
        assert registry.needs_reindex(new_tools) is True

    def test_needs_reindex_true_when_tools_removed(self):
        """needs_reindex should return True when tools removed."""
        registry = ToolMetadataRegistry.get_instance()
        tools = [MockTool(), GitMockTool()]

        registry.refresh_from_tools(tools, force=True)

        # Remove a tool
        fewer_tools = [MockTool()]
        assert registry.needs_reindex(fewer_tools) is True

    def test_refresh_skips_when_cache_valid(self):
        """refresh_from_tools should skip when cache is valid."""
        registry = ToolMetadataRegistry.get_instance()
        tools = [MockTool()]

        # First refresh
        result1 = registry.refresh_from_tools(tools, force=True)
        assert result1 is True  # Did reindex

        # Second refresh with same tools
        result2 = registry.refresh_from_tools(tools, force=False)
        assert result2 is False  # Skipped

    def test_refresh_force_overrides_cache(self):
        """refresh_from_tools with force=True should always reindex."""
        registry = ToolMetadataRegistry.get_instance()
        tools = [MockTool()]

        registry.refresh_from_tools(tools, force=True)
        result = registry.refresh_from_tools(tools, force=True)

        assert result is True  # Forced reindex

    def test_hash_changes_when_tool_description_changes(self):
        """Hash should change when tool description changes."""
        tools1 = [MockTool()]
        hash1 = ToolMetadataRegistry._calculate_tools_hash(tools1)

        # Create tool with modified description
        class ModifiedMockTool(MockTool):
            description = "Modified description."

        tools2 = [ModifiedMockTool()]
        hash2 = ToolMetadataRegistry._calculate_tools_hash(tools2)

        assert hash1 != hash2


# =============================================================================
# Test: Integration with SemanticToolSelector
# =============================================================================


class TestSemanticToolSelectorIntegration:
    """Tests for SemanticToolSelector integration with metadata system."""

    def setup_method(self):
        """Reset singleton before each test."""
        ToolMetadataRegistry.reset_instance()

    @pytest.mark.asyncio
    async def test_create_tool_text_uses_get_metadata(self):
        """_create_tool_text should use get_metadata() for tools."""
        # Import here to avoid import issues
        from victor.tools.semantic_selector import SemanticToolSelector

        tool = MockToolWithExplicitMetadata()
        text = SemanticToolSelector._create_tool_text(tool)

        # Should include explicit metadata
        assert "testing explicit metadata" in text.lower() or "validating contract" in text.lower()
        assert "explicit" in text.lower()

    @pytest.mark.asyncio
    async def test_create_tool_text_auto_generates_for_simple_tools(self):
        """_create_tool_text should auto-generate for tools without explicit metadata."""
        from victor.tools.semantic_selector import SemanticToolSelector

        tool = MockTool()
        text = SemanticToolSelector._create_tool_text(tool)

        # Should include tool name and description
        assert "mock" in text.lower()
        # Should have auto-generated use cases or keywords
        assert "use for:" in text.lower() or "common requests:" in text.lower()

    @pytest.mark.asyncio
    async def test_initialize_populates_registry(self):
        """initialize_tool_embeddings should populate ToolMetadataRegistry."""
        from victor.tools.registry import ToolRegistry
        from victor.tools.semantic_selector import SemanticToolSelector

        # Create tool registry with test tools
        tool_registry = ToolRegistry()
        tool_registry.register(MockTool())
        tool_registry.register(MockToolWithExplicitMetadata())

        # Mock embedding service
        selector = SemanticToolSelector(cache_embeddings=False)

        with patch.object(selector, "_get_embedding", new_callable=AsyncMock) as mock_embed:
            import numpy as np

            mock_embed.return_value = np.zeros(384, dtype=np.float32)

            await selector.initialize_tool_embeddings(tool_registry)

        # Check canonical registry was populated
        from victor.tools.metadata_registry import (
            ToolMetadataRegistry as CanonicalToolMetadataRegistry,
        )

        metadata_registry = CanonicalToolMetadataRegistry.get_instance()
        assert metadata_registry.get_metadata("mock_tool") is not None
        assert metadata_registry.get_metadata("explicit_metadata_tool") is not None


# =============================================================================
# Test: Plugin Tool Support
# =============================================================================


class TestPluginToolSupport:
    """Tests for plugin tool registration support."""

    def setup_method(self):
        """Reset singleton before each test."""
        ToolMetadataRegistry.reset_instance()

    def test_incremental_registration(self):
        """Should support incremental tool registration for plugins."""
        registry = ToolMetadataRegistry.get_instance()

        # Initial tools
        registry.register_tool(MockTool())
        assert registry.get_metadata("mock_tool") is not None

        # Add plugin tool incrementally
        registry.register_tool(GitMockTool())
        assert registry.get_metadata("git_commit") is not None

        # Original tool still registered
        assert registry.get_metadata("mock_tool") is not None

    def test_plugin_tools_indexed_correctly(self):
        """Plugin tools should be indexed by category and keywords."""
        registry = ToolMetadataRegistry.get_instance()

        # Create a plugin tool with custom metadata
        class PluginTool(BaseTool):
            name = "my_plugin"
            description = "A custom plugin tool."
            parameters = {"type": "object", "properties": {}}

            @property
            def metadata(self) -> ToolMetadata:
                return ToolMetadata(
                    category="plugins",
                    keywords=["plugin", "custom", "extension"],
                    use_cases=["extending victor", "custom functionality"],
                )

            async def execute(self, context: Dict[str, Any], **kwargs: Any) -> ToolResult:
                return ToolResult(success=True, output="plugin result")

        registry.register_tool(PluginTool())

        # Should be searchable
        assert "my_plugin" in registry.get_tools_by_category("plugins")
        assert any(entry.name == "my_plugin" for entry in registry.get_by_keyword("plugin"))
        assert "my_plugin" in registry.get_tools_matching_text("extension")


# =============================================================================
# Test: ToolMetadata Serialization
# =============================================================================


class TestToolMetadataSerialization:
    """Tests for ToolMetadata serialization."""

    def test_to_dict_complete(self):
        """to_dict should include all fields."""
        metadata = ToolMetadata(
            category="test",
            keywords=["key1", "key2"],
            use_cases=["use1"],
            examples=["example1"],
            priority_hints=["hint1"],
        )

        result = metadata.to_dict()

        assert result["category"] == "test"
        assert result["keywords"] == ["key1", "key2"]
        assert result["use_cases"] == ["use1"]
        assert result["examples"] == ["example1"]
        assert result["priority_hints"] == ["hint1"]

    def test_to_dict_empty_fields(self):
        """to_dict should handle empty fields."""
        metadata = ToolMetadata()

        result = metadata.to_dict()

        assert result["category"] == ""
        assert result["keywords"] == []
        assert result["use_cases"] == []
        assert result["examples"] == []
        assert result["priority_hints"] == []


# =============================================================================
# Test: Tool Usage Stats Pre-initialization
# =============================================================================


class TestToolUsageStatsPreInit:
    """Tests for tool usage stats pre-initialization.

    Ensures ALL tools are tracked from startup, not just used ones.
    This is critical for accurate analytics and reporting.
    """

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools for testing."""

        class MockTool1(BaseTool):
            name = "tool_one"
            description = "First tool"
            parameters = {"type": "object", "properties": {}}

            async def execute(self, context, **kwargs):
                return ToolResult(success=True, output="ok")

        class MockTool2(BaseTool):
            name = "tool_two"
            description = "Second tool"
            parameters = {"type": "object", "properties": {}}

            async def execute(self, context, **kwargs):
                return ToolResult(success=True, output="ok")

        class MockTool3(BaseTool):
            name = "tool_three"
            description = "Third tool"
            parameters = {"type": "object", "properties": {}}

            async def execute(self, context, **kwargs):
                return ToolResult(success=True, output="ok")

        return [MockTool1(), MockTool2(), MockTool3()]

    def test_all_tools_pre_initialized(self, mock_tools, tmp_path):
        """All tools should be in usage cache even if not used."""
        from victor.tools.semantic_selector import SemanticToolSelector

        # Create selector with temp cache directory
        selector = SemanticToolSelector(
            cache_embeddings=False,  # Don't need embeddings for this test
        )
        selector.cache_dir = tmp_path
        selector._usage_cache_file = tmp_path / "tool_usage_stats.pkl"
        selector._tool_usage_cache = {}  # Start fresh

        # Pre-initialize all tools
        selector._initialize_all_tool_stats(mock_tools)

        # ALL 3 tools should be in cache
        assert len(selector._tool_usage_cache) == 3
        assert "tool_one" in selector._tool_usage_cache
        assert "tool_two" in selector._tool_usage_cache
        assert "tool_three" in selector._tool_usage_cache

        # Each should have 0 usage count
        for tool_name in ["tool_one", "tool_two", "tool_three"]:
            stats = selector._tool_usage_cache[tool_name]
            assert stats["usage_count"] == 0
            assert stats["success_count"] == 0
            assert stats["last_used"] == 0
            assert stats["recent_contexts"] == []

    def test_pre_init_preserves_existing_stats(self, mock_tools, tmp_path):
        """Pre-initialization should not overwrite existing stats."""
        from victor.tools.semantic_selector import SemanticToolSelector

        selector = SemanticToolSelector(cache_embeddings=False)
        selector.cache_dir = tmp_path
        selector._usage_cache_file = tmp_path / "tool_usage_stats.pkl"

        # Pre-populate with existing stats for tool_one
        selector._tool_usage_cache = {
            "tool_one": {
                "usage_count": 100,
                "success_count": 95,
                "last_used": 1234567890,
                "recent_contexts": ["query1", "query2"],
            }
        }

        # Pre-initialize all tools
        selector._initialize_all_tool_stats(mock_tools)

        # tool_one should preserve existing stats
        assert selector._tool_usage_cache["tool_one"]["usage_count"] == 100
        assert selector._tool_usage_cache["tool_one"]["success_count"] == 95

        # Other tools should be initialized to 0
        assert selector._tool_usage_cache["tool_two"]["usage_count"] == 0
        assert selector._tool_usage_cache["tool_three"]["usage_count"] == 0

    def test_pre_init_count_matches_total_tools(self, mock_tools, tmp_path):
        """Number of tracked tools should match number of registered tools."""
        from victor.tools.semantic_selector import SemanticToolSelector

        selector = SemanticToolSelector(cache_embeddings=False)
        selector.cache_dir = tmp_path
        selector._usage_cache_file = tmp_path / "tool_usage_stats.pkl"
        selector._tool_usage_cache = {}

        selector._initialize_all_tool_stats(mock_tools)

        # Should have exactly as many entries as tools
        assert len(selector._tool_usage_cache) == len(mock_tools)
