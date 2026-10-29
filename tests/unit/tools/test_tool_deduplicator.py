"""Unit tests for tool deduplication system."""

from victor.tools.deduplication import (
    DeduplicationConfig,
    ToolDeduplicator,
    ToolSource,
)
from victor.tools.deduplication.conflict_detector import (
    ConflictDetector,
    ConflictType,
)
from victor.tools.deduplication.naming_enforcer import (
    NamingEnforcer,
)


# Mock tool class for testing
class MockTool:
    """Mock tool for testing."""

    def __init__(
        self,
        name: str,
        description: str = "",
        parameters: dict | None = None,
        source: ToolSource = ToolSource.NATIVE,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self._tool_source = source


class TestToolSource:
    """Test ToolSource enum and priority weights."""

    def test_priority_weights(self):
        """Test priority weights are correct."""
        assert ToolSource.NATIVE.priority_weight == 100
        assert ToolSource.LANGCHAIN.priority_weight == 75
        assert ToolSource.MCP.priority_weight == 50
        assert ToolSource.PLUGIN.priority_weight == 25

    def test_comparison(self):
        """Test ToolSource comparison by priority."""
        assert ToolSource.NATIVE < ToolSource.LANGCHAIN
        assert ToolSource.LANGCHAIN < ToolSource.MCP
        assert ToolSource.MCP < ToolSource.PLUGIN


class TestDeduplicationConfig:
    """Test DeduplicationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeduplicationConfig()
        assert config.enabled is True
        assert config.priority_order == ["native", "langchain", "mcp", "plugin"]
        assert config.whitelist == []
        assert config.blacklist == []
        assert config.naming_enforcement is True
        assert config.semantic_similarity_threshold == 0.85

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DeduplicationConfig(
            enabled=False,
            priority_order=["plugin", "mcp", "langchain", "native"],
            whitelist=["tool1", "tool2"],
            blacklist=["tool3"],
            naming_enforcement=False,
            semantic_similarity_threshold=0.9,
        )
        assert config.enabled is False
        assert config.priority_order == ["plugin", "mcp", "langchain", "native"]
        assert config.whitelist == ["tool1", "tool2"]
        assert config.blacklist == ["tool3"]
        assert config.naming_enforcement is False
        assert config.semantic_similarity_threshold == 0.9

    def test_get_priority_map(self):
        """Test priority map generation."""
        config = DeduplicationConfig()
        priority_map = config.get_priority_map()
        assert priority_map["native"] == 4
        assert priority_map["langchain"] == 3
        assert priority_map["mcp"] == 2
        assert priority_map["plugin"] == 1


class TestToolDeduplicator:
    """Test ToolDeduplicator."""

    def test_no_deduplication_when_disabled(self):
        """Test that all tools are kept when deduplication is disabled."""
        config = DeduplicationConfig(enabled=False)
        deduplicator = ToolDeduplicator(config)

        tools = [
            MockTool(name="search", source=ToolSource.NATIVE),
            MockTool(name="search", source=ToolSource.LANGCHAIN),
        ]

        result = deduplicator.deduplicate(tools)

        assert len(result.kept_tools) == 2
        assert len(result.skipped_tools) == 0
        assert result.conflicts_resolved == 0

    def test_exact_name_conflict_resolution(self):
        """Test conflict resolution for tools with exact same name."""
        config = DeduplicationConfig()
        deduplicator = ToolDeduplicator(config)

        tools = [
            MockTool(name="search", source=ToolSource.NATIVE),
            MockTool(name="search", source=ToolSource.LANGCHAIN),
            MockTool(name="search", source=ToolSource.MCP),
        ]

        result = deduplicator.deduplicate(tools)

        # Should keep native (highest priority), skip others
        assert len(result.kept_tools) == 1
        assert len(result.skipped_tools) == 2
        assert result.conflicts_resolved == 1
        assert result.kept_tools[0].name == "search"
        assert result.kept_tools[0]._tool_source == ToolSource.NATIVE

    def test_blacklist(self):
        """Test that blacklisted tools are always skipped."""
        config = DeduplicationConfig(blacklist=["bad_tool"])
        deduplicator = ToolDeduplicator(config)

        tools = [
            MockTool(name="good_tool", source=ToolSource.NATIVE),
            MockTool(name="bad_tool", source=ToolSource.NATIVE),
        ]

        result = deduplicator.deduplicate(tools)

        assert len(result.kept_tools) == 1
        assert len(result.skipped_tools) == 1
        assert result.kept_tools[0].name == "good_tool"
        assert result.skipped_tools[0].name == "bad_tool"

    def test_whitelist(self):
        """Test that whitelisted tools bypass deduplication."""
        config = DeduplicationConfig(whitelist=["special_tool"])
        deduplicator = ToolDeduplicator(config)

        tools = [
            MockTool(name="special_tool", source=ToolSource.LANGCHAIN),
            MockTool(name="special_tool", source=ToolSource.NATIVE),
        ]

        result = deduplicator.deduplicate(tools)

        # Whitelisted tool should win even though it's lower priority
        assert len(result.kept_tools) == 1
        assert result.kept_tools[0].name == "special_tool"
        assert result.kept_tools[0]._tool_source == ToolSource.LANGCHAIN

    def test_no_conflict_different_names(self):
        """Test that tools with different names don't conflict."""
        config = DeduplicationConfig()
        deduplicator = ToolDeduplicator(config)

        tools = [
            MockTool(name="search", source=ToolSource.NATIVE),
            MockTool(name="read", source=ToolSource.LANGCHAIN),
            MockTool(name="write", source=ToolSource.MCP),
        ]

        result = deduplicator.deduplicate(tools)

        # All tools should be kept
        assert len(result.kept_tools) == 3
        assert len(result.skipped_tools) == 0
        assert result.conflicts_resolved == 0

    def test_normalized_name_matching(self):
        """Test conflict detection with normalized names."""
        config = DeduplicationConfig()
        deduplicator = ToolDeduplicator(config)

        tools = [
            MockTool(name="web_search", source=ToolSource.NATIVE),
            MockTool(name="web-search", source=ToolSource.LANGCHAIN),
            MockTool(name="web search", source=ToolSource.MCP),
        ]

        result = deduplicator.deduplicate(tools)

        # Should detect conflict and keep native
        assert len(result.kept_tools) == 1
        assert len(result.skipped_tools) == 2
        assert result.conflicts_resolved == 1

    def test_naming_enforcement(self):
        """Test naming convention enforcement."""
        config = DeduplicationConfig(naming_enforcement=True)
        deduplicator = ToolDeduplicator(config)

        # LangChain tool without prefix
        tool = MockTool(name="wikipedia", source=ToolSource.LANGCHAIN)
        new_name = deduplicator.enforce_naming(tool)

        assert new_name == "lgc_wikipedia"

    def test_naming_enforcement_already_has_prefix(self):
        """Test that tools with correct prefix are not renamed."""
        config = DeduplicationConfig(naming_enforcement=True)
        deduplicator = ToolDeduplicator(config)

        # LangChain tool with correct prefix
        tool = MockTool(name="lgc_wikipedia", source=ToolSource.LANGCHAIN)
        new_name = deduplicator.enforce_naming(tool)

        assert new_name == "lgc_wikipedia"

    def test_naming_enforcement_disabled(self):
        """Test that naming enforcement can be disabled."""
        config = DeduplicationConfig(naming_enforcement=False)
        deduplicator = ToolDeduplicator(config)

        tool = MockTool(name="wikipedia", source=ToolSource.LANGCHAIN)
        new_name = deduplicator.enforce_naming(tool)

        assert new_name == "wikipedia"

    def test_empty_tool_list(self):
        """Test deduplication with empty tool list."""
        config = DeduplicationConfig()
        deduplicator = ToolDeduplicator(config)

        result = deduplicator.deduplicate([])

        assert len(result.kept_tools) == 0
        assert len(result.skipped_tools) == 0
        assert result.conflicts_resolved == 0

    def test_source_detection_from_prefix(self):
        """Test tool source detection from naming prefix."""
        config = DeduplicationConfig()
        deduplicator = ToolDeduplicator(config)

        # Detect from lgc_ prefix
        lgc_tool = MockTool(name="lgc_search")
        assert deduplicator._get_tool_source(lgc_tool) == ToolSource.LANGCHAIN

        # Detect from mcp_ prefix
        mcp_tool = MockTool(name="mcp_search")
        assert deduplicator._get_tool_source(mcp_tool) == ToolSource.MCP

        # Detect from plg_ prefix
        plg_tool = MockTool(name="plg_search")
        assert deduplicator._get_tool_source(plg_tool) == ToolSource.PLUGIN

        # Default to native for no prefix
        native_tool = MockTool(name="search")
        assert deduplicator._get_tool_source(native_tool) == ToolSource.NATIVE


class TestConflictDetector:
    """Test ConflictDetector."""

    def test_exact_name_match(self):
        """Test exact name matching."""
        detector = ConflictDetector()

        tool1 = MockTool(name="search", description="Search files")
        tool2 = MockTool(name="search", description="Search the web")

        result = detector.are_tools_conflicting(tool1, tool2)

        assert result.is_conflict is True
        assert result.conflict_type == ConflictType.EXACT_NAME
        assert result.confidence == 1.0

    def test_no_conflict_different_names(self):
        """Test no conflict with different names."""
        detector = ConflictDetector()

        tool1 = MockTool(name="search", description="Search files")
        tool2 = MockTool(name="read", description="Read files")

        result = detector.are_tools_conflicting(tool1, tool2)

        assert result.is_conflict is False

    def test_capability_overlap_search(self):
        """Test capability overlap detection for search tools."""
        detector = ConflictDetector()

        tool1 = MockTool(
            name="file_search",
            description="Search files in directory",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        tool2 = MockTool(
            name="code_search",
            description="Search code in repository",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )

        result = detector.are_tools_conflicting(tool1, tool2)

        # Should detect conflict based on search keywords + parameter similarity
        assert result.is_conflict is True
        assert result.conflict_type == ConflictType.CAPABILITY_OVERLAP
        assert result.confidence > 0.0

    def test_capability_overlap_no_parameters(self):
        """Test capability overlap with no parameters."""
        detector = ConflictDetector()

        tool1 = MockTool(name="search", description="Search files", parameters={})
        tool2 = MockTool(name="find", description="Find files", parameters={})

        result = detector.are_tools_conflicting(tool1, tool2)

        # Should not detect conflict without parameter similarity
        assert result.is_conflict is False


class TestNamingEnforcer:
    """Test NamingEnforcer."""

    def test_enforce_langchain_prefix(self):
        """Test enforcing LangChain prefix."""
        enforcer = NamingEnforcer()

        tool = MockTool(name="wikipedia")
        new_name = enforcer.enforce_name(tool, ToolSource.LANGCHAIN)

        assert new_name == "lgc_wikipedia"

    def test_enforce_mcp_prefix(self):
        """Test enforcing MCP prefix."""
        enforcer = NamingEnforcer()

        tool = MockTool(name="github_search")
        new_name = enforcer.enforce_name(tool, ToolSource.MCP)

        assert new_name == "mcp_github_search"

    def test_enforce_plugin_prefix(self):
        """Test enforcing plugin prefix."""
        enforcer = NamingEnforcer()

        tool = MockTool(name="custom_tool")
        new_name = enforcer.enforce_name(tool, ToolSource.PLUGIN)

        assert new_name == "plg_custom_tool"

    def test_no_prefix_for_native(self):
        """Test that native tools don't get prefix."""
        enforcer = NamingEnforcer()

        tool = MockTool(name="search")
        new_name = enforcer.enforce_name(tool, ToolSource.NATIVE)

        assert new_name == "search"

    def test_already_has_prefix(self):
        """Test that tools with correct prefix are not changed."""
        enforcer = NamingEnforcer()

        tool = MockTool(name="lgc_wikipedia")
        new_name = enforcer.enforce_name(tool, ToolSource.LANGCHAIN)

        assert new_name == "lgc_wikipedia"

    def test_strip_prefix(self):
        """Test stripping prefix from tool name."""
        enforcer = NamingEnforcer()

        assert enforcer.strip_prefix("lgc_wikipedia") == "wikipedia"
        assert enforcer.strip_prefix("mcp_search") == "search"
        assert enforcer.strip_prefix("plg_tool") == "tool"
        assert enforcer.strip_prefix("search") == "search"

    def test_detect_source_from_name(self):
        """Test detecting source from tool name."""
        enforcer = NamingEnforcer()

        assert enforcer.detect_source_from_name("lgc_wikipedia") == ToolSource.LANGCHAIN
        assert enforcer.detect_source_from_name("mcp_search") == ToolSource.MCP
        assert enforcer.detect_source_from_name("plg_tool") == ToolSource.PLUGIN
        assert enforcer.detect_source_from_name("search") == ToolSource.NATIVE

    def test_enforcement_disabled(self):
        """Test that enforcement can be disabled."""
        enforcer = NamingEnforcer(enforce=False)

        tool = MockTool(name="wikipedia")
        new_name = enforcer.enforce_name(tool, ToolSource.LANGCHAIN)

        assert new_name == "wikipedia"
