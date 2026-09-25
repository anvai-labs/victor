"""Integration tests for tool deduplication across native, LangChain, and MCP tools."""

from typing import Any, Dict, Optional

import pytest

from victor.tools.deduplication import (
    DeduplicationConfig,
    ToolDeduplicator,
    ToolSource,
)
from victor.tools.registry import ToolRegistry
from victor.tools.base import BaseTool, CostTier, ToolResult, Priority


class MockNativeTool(BaseTool):
    """Mock native tool for testing."""

    def __init__(self, name: str, description: str = "Native tool"):
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.LOW

    @property
    def is_idempotent(self) -> bool:
        return True

    async def execute(self, _exec_ctx: dict, **kwargs):
        return ToolResult(success=True, output="mock result")


class MockLangChainTool:
    """Mock LangChain tool for testing."""

    def __init__(self, name: str, description: str = "LangChain tool"):
        self.name = name
        self.description = description
        self.args_schema = None
        self._tool_source = ToolSource.LANGCHAIN


class MockMCPTool:
    """Mock MCP tool for testing.

    Mirrors the real MCPTool contract: ``input_schema`` is optional and may be
    None, in which case MCPAdapterTool derives a JSON schema from the legacy
    ``parameters`` list — the path these tests exercise.
    """

    def __init__(
        self,
        name: str,
        description: str = "MCP tool",
        input_schema: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = []
        self.input_schema = input_schema
        self._tool_source = ToolSource.MCP


class TestToolRegistryDeduplication:
    """Test ToolRegistry with deduplication enabled."""

    def test_native_tool_preferred_over_langchain(self):
        """Test that native tools are preferred over LangChain tools."""
        registry = ToolRegistry()

        # Register native tool first (higher priority)
        native_tool = MockNativeTool(name="search", description="Native search")
        registry.register(native_tool)

        # Try to register LangChain tool with same name
        lc_tool = MockLangChainTool(name="search", description="Search via LangChain")
        from victor.tools.langchain_adapter_tool import LangChainAdapterTool

        lc_adapter = LangChainAdapterTool(lc_tool)
        registry.register(lc_adapter)

        # Native tool should be in registry, LangChain tool skipped
        # include_folded=True so folded canonical names (e.g. "read"->"fs",
        # "search"->"code") are observable — these tests assert dedup kept the
        # native tool, which is folded but still the dedup winner.
        tools = registry.list_tools(only_enabled=True, include_folded=True)
        tool_names = [t.name for t in tools]

        # Only native tool should be registered (LangChain skipped due to conflict)
        assert "search" in tool_names
        # LangChain adapter would be named lgc_search if registered
        assert "lgc_search" not in tool_names

    def test_native_tool_preferred_over_mcp(self):
        """Test that native tools are preferred over MCP tools."""
        registry = ToolRegistry()

        # Register native tool first (higher priority)
        native_tool = MockNativeTool(name="read", description="Native read")
        registry.register(native_tool)

        # Try to register MCP tool with same name
        mcp_tool = MockMCPTool(name="read", description="Read via MCP")
        from victor.tools.mcp_adapter_tool import MCPAdapterTool

        class MockMCPRegistry:
            async def call_tool(self, name, **kwargs):
                pass

        mcp_adapter = MCPAdapterTool(
            mcp_tool=mcp_tool, mcp_registry=MockMCPRegistry(), server_name="test_server"
        )
        registry.register(mcp_adapter)

        # Native tool should be in registry, MCP tool skipped
        # include_folded=True so folded canonical names (e.g. "read"->"fs",
        # "search"->"code") are observable — these tests assert dedup kept the
        # native tool, which is folded but still the dedup winner.
        tools = registry.list_tools(only_enabled=True, include_folded=True)
        tool_names = [t.name for t in tools]

        # Only native tool should be registered (MCP skipped due to conflict)
        assert "read" in tool_names
        # MCP adapter would be named mcp_read if registered
        assert "mcp_read" not in tool_names

    def test_langchain_preferred_over_mcp(self):
        """Test that LangChain tools are preferred over MCP tools."""
        registry = ToolRegistry()

        # Register LangChain tool first (higher priority than MCP)
        lc_tool = MockLangChainTool(name="wikipedia", description="Wikipedia via LangChain")
        from victor.tools.langchain_adapter_tool import LangChainAdapterTool

        lc_adapter = LangChainAdapterTool(lc_tool)
        registry.register(lc_adapter)

        # Try to register MCP tool with same name
        mcp_tool = MockMCPTool(name="wikipedia", description="Wikipedia via MCP")
        from victor.tools.mcp_adapter_tool import MCPAdapterTool

        class MockMCPRegistry:
            async def call_tool(self, name, **kwargs):
                pass

        mcp_adapter = MCPAdapterTool(
            mcp_tool=mcp_tool, mcp_registry=MockMCPRegistry(), server_name="test_server"
        )
        registry.register(mcp_adapter)

        # LangChain tool should be in registry, MCP tool skipped
        # include_folded=True so folded canonical names (e.g. "read"->"fs",
        # "search"->"code") are observable — these tests assert dedup kept the
        # native tool, which is folded but still the dedup winner.
        tools = registry.list_tools(only_enabled=True, include_folded=True)
        tool_names = [t.name for t in tools]

        # Only LangChain tool should be registered (MCP skipped due to conflict)
        assert "lgc_wikipedia" in tool_names
        # MCP adapter would be named mcp_wikipedia if registered
        assert "mcp_wikipedia" not in tool_names

    def test_higher_priority_tool_replaces_lower_priority_when_registered_later(self):
        """Higher-priority tools should win regardless of registration order."""
        registry = ToolRegistry()

        mcp_tool = MockMCPTool(name="read", description="Read via MCP")
        native_tool = MockNativeTool(name="read", description="Native read")

        from victor.tools.mcp_adapter_tool import MCPAdapterTool

        class MockMCPRegistry:
            async def call_tool(self, name, **kwargs):
                pass

        registry.register(MCPAdapterTool(mcp_tool, MockMCPRegistry(), "test_server"))
        registry.register(native_tool)

        # include_folded=True so folded canonical names (e.g. "read"->"fs",
        # "search"->"code") are observable — these tests assert dedup kept the
        # native tool, which is folded but still the dedup winner.
        tools = registry.list_tools(only_enabled=True, include_folded=True)
        tool_names = [t.name for t in tools]

        assert "read" in tool_names
        assert "mcp_read" not in tool_names

    def test_no_conflict_different_names(self):
        """Test that tools with different names don't conflict."""
        registry = ToolRegistry()

        # Register tools with different names
        native_tool = MockNativeTool(name="read", description="Read file")
        lc_tool = MockLangChainTool(name="wikipedia", description="Wikipedia search")
        mcp_tool = MockMCPTool(name="github_search", description="Search GitHub")

        from victor.tools.langchain_adapter_tool import LangChainAdapterTool
        from victor.tools.mcp_adapter_tool import MCPAdapterTool

        class MockMCPRegistry:
            async def call_tool(self, name, **kwargs):
                pass

        registry.register(native_tool)
        registry.register(LangChainAdapterTool(lc_tool))
        registry.register(MCPAdapterTool(mcp_tool, MockMCPRegistry(), "test_server"))

        # All tools should be registered
        # include_folded=True so folded canonical names (e.g. "read"->"fs",
        # "search"->"code") are observable — these tests assert dedup kept the
        # native tool, which is folded but still the dedup winner.
        tools = registry.list_tools(only_enabled=True, include_folded=True)
        tool_names = [t.name for t in tools]

        assert "read" in tool_names
        assert "lgc_wikipedia" in tool_names
        assert "mcp_github_search" in tool_names
        assert len(tool_names) == 3

    def test_deduplication_can_be_disabled(self):
        """Test that deduplication can be disabled via settings."""
        # This test verifies the setting exists and can be toggled
        # Actual behavior depends on ToolRegistry initialization
        from victor.config.tool_settings import ToolSettings

        settings = ToolSettings(enable_tool_deduplication=False)
        assert settings.enable_tool_deduplication is False

        settings_enabled = ToolSettings(enable_tool_deduplication=True)
        assert settings_enabled.enable_tool_deduplication is True


class TestToolDeduplicatorIntegration:
    """Test ToolDeduplicator with mixed tool sources."""

    def test_priority_order_native_langchain_mcp(self):
        """Test priority resolution with native, LangChain, and MCP tools."""
        config = DeduplicationConfig()
        deduplicator = ToolDeduplicator(config)

        # Create tools with same normalized name
        native_tool = MockNativeTool(name="search", description="Native search")
        lc_tool = MockLangChainTool(name="search", description="LangChain search")
        mcp_tool = MockMCPTool(name="search", description="MCP search")

        # Wrap LangChain and MCP tools in adapters
        from victor.tools.langchain_adapter_tool import LangChainAdapterTool
        from victor.tools.mcp_adapter_tool import MCPAdapterTool

        class MockMCPRegistry:
            async def call_tool(self, name, **kwargs):
                pass

        lc_adapter = LangChainAdapterTool(lc_tool)
        mcp_adapter = MCPAdapterTool(mcp_tool, MockMCPRegistry(), "test_server")

        tools = [native_tool, lc_adapter, mcp_adapter]
        result = deduplicator.deduplicate(tools)

        # Native tool should be kept, others skipped
        assert len(result.kept_tools) == 1
        assert result.kept_tools[0].name == "search"
        assert len(result.skipped_tools) == 2
        assert result.conflicts_resolved == 1

    def test_normalized_name_matching(self):
        """Test conflict detection with normalized names."""
        config = DeduplicationConfig()
        deduplicator = ToolDeduplicator(config)

        # Create tools with different spellings but same normalized name
        native_tool = MockNativeTool(name="web_search", description="Web search")
        lc_tool = MockLangChainTool(name="web-search", description="Web search")
        mcp_tool = MockMCPTool(name="web search", description="Web search")

        from victor.tools.langchain_adapter_tool import LangChainAdapterTool
        from victor.tools.mcp_adapter_tool import MCPAdapterTool

        class MockMCPRegistry:
            async def call_tool(self, name, **kwargs):
                pass

        lc_adapter = LangChainAdapterTool(lc_tool)
        mcp_adapter = MCPAdapterTool(mcp_tool, MockMCPRegistry(), "test_server")

        tools = [native_tool, lc_adapter, mcp_adapter]
        result = deduplicator.deduplicate(tools)

        # Should detect conflict and keep native tool
        assert len(result.kept_tools) == 1
        assert result.conflicts_resolved == 1

    def test_whitelist_bypass(self):
        """Test that whitelisted tools bypass deduplication."""
        config = DeduplicationConfig(whitelist=["special_tool"])
        deduplicator = ToolDeduplicator(config)

        native_tool = MockNativeTool(name="special_tool", description="Special tool")
        lc_tool = MockLangChainTool(name="special_tool", description="Special tool via LangChain")

        from victor.tools.langchain_adapter_tool import LangChainAdapterTool

        lc_adapter = LangChainAdapterTool(lc_tool)

        tools = [native_tool, lc_adapter]
        result = deduplicator.deduplicate(tools)

        # Native tool should be kept (whitelisted wins even if lower priority in some cases)
        assert len(result.kept_tools) == 1

    def test_blacklist_force_skip(self):
        """Test that blacklisted tools are always skipped."""
        config = DeduplicationConfig(blacklist=["bad_tool"])
        deduplicator = ToolDeduplicator(config)

        native_tool = MockNativeTool(name="bad_tool", description="Bad tool")
        lc_tool = MockLangChainTool(name="good_tool", description="Good tool")

        from victor.tools.langchain_adapter_tool import LangChainAdapterTool

        lc_adapter = LangChainAdapterTool(lc_tool)

        tools = [native_tool, lc_adapter]
        result = deduplicator.deduplicate(tools)

        # Blacklisted tool should be skipped
        assert len(result.kept_tools) == 1
        assert result.kept_tools[0].name == "lgc_good_tool"
        assert any(t.name == "bad_tool" for t in result.skipped_tools)


class TestAdapterNamingConventions:
    """Test that adapters follow unified naming conventions."""

    def test_langchain_adapter_lgc_prefix(self):
        """Test that LangChainAdapterTool uses lgc_ prefix."""
        from victor.tools.langchain_adapter_tool import (
            LangChainAdapterTool,
            DEFAULT_LANGCHAIN_PREFIX,
        )

        lc_tool = MockLangChainTool(name="wikipedia", description="Wikipedia")
        adapter = LangChainAdapterTool(lc_tool)

        assert adapter.name.startswith("lgc_")
        assert adapter.name == "lgc_wikipedia"
        assert DEFAULT_LANGCHAIN_PREFIX == "lgc"

    def test_mcp_adapter_mcp_prefix(self):
        """Test that MCPAdapterTool uses mcp_ prefix."""
        from victor.tools.mcp_adapter_tool import MCPAdapterTool, DEFAULT_MCP_PREFIX

        mcp_tool = MockMCPTool(name="github_search", description="Search GitHub")

        class MockMCPRegistry:
            async def call_tool(self, name, **kwargs):
                pass

        adapter = MCPAdapterTool(mcp_tool, MockMCPRegistry(), "test_server")

        assert adapter.name.startswith("mcp_")
        assert adapter.name == "mcp_github_search"
        assert DEFAULT_MCP_PREFIX == "mcp"

    def test_adapter_source_metadata(self):
        """Test that adapters set ToolSource metadata correctly."""
        from victor.tools.deduplication import ToolSource
        from victor.tools.langchain_adapter_tool import LangChainAdapterTool
        from victor.tools.mcp_adapter_tool import MCPAdapterTool

        lc_tool = MockLangChainTool(name="test", description="Test")
        lc_adapter = LangChainAdapterTool(lc_tool)

        mcp_tool = MockMCPTool(name="test", description="Test")

        class MockMCPRegistry:
            async def call_tool(self, name, **kwargs):
                pass

        mcp_adapter = MCPAdapterTool(mcp_tool, MockMCPRegistry(), "test_server")

        # Check source metadata
        assert hasattr(lc_adapter, "_tool_source")
        assert hasattr(mcp_adapter, "_tool_source")

        # LangChain adapter should have LANGCHAIN source
        assert lc_adapter._tool_source == ToolSource.LANGCHAIN

        # MCP adapter should have MCP source
        assert mcp_adapter._tool_source == ToolSource.MCP

    def test_mcp_adapter_schema_paths_match_contract(self):
        """The mock mirrors MCPTool: explicit input_schema wins; None falls back
        to the legacy parameters-derived schema. Both paths must construct an
        adapter without error (the missing-attribute case used to crash)."""
        from victor.tools.mcp_adapter_tool import MCPAdapterTool

        class MockMCPRegistry:
            async def call_tool(self, name, **kwargs):
                pass

        explicit = {"type": "object", "properties": {"q": {"type": "string"}}}
        with_schema = MCPAdapterTool(
            MockMCPTool(name="t1", input_schema=explicit), MockMCPRegistry(), "srv"
        )
        assert with_schema.parameters == explicit

        without_schema = MCPAdapterTool(MockMCPTool(name="t2"), MockMCPRegistry(), "srv")
        assert without_schema.parameters == {"type": "object", "properties": {}}


class TestDeduplicationConfiguration:
    """Test deduplication configuration and settings."""

    def test_default_settings(self):
        """Test that default settings are correct."""
        from victor.config.tool_settings import get_tool_settings

        settings = get_tool_settings()

        # Check default values
        assert hasattr(settings, "enable_tool_deduplication")
        assert hasattr(settings, "deduplication_priority_order")
        assert hasattr(settings, "deduplication_whitelist")
        assert hasattr(settings, "deduplication_blacklist")
        assert hasattr(settings, "deduplication_naming_enforcement")
        assert hasattr(settings, "deduplication_semantic_threshold")

    def test_priority_order_default(self):
        """Test default priority order."""
        from victor.config.tool_settings import ToolSettings

        settings = ToolSettings()
        assert settings.deduplication_priority_order == [
            "native",
            "langchain",
            "mcp",
            "plugin",
        ]

    def test_semantic_threshold_range(self):
        """Test that semantic threshold is within valid range."""
        from victor.config.tool_settings import ToolSettings

        settings = ToolSettings()
        assert 0.0 <= settings.deduplication_semantic_threshold <= 1.0
        assert settings.deduplication_semantic_threshold == 0.85
