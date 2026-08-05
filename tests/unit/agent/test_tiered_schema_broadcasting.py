# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for tiered tool schema broadcasting — cache-optimized token reduction."""

from unittest.mock import MagicMock

import pytest

from victor.providers.base import ToolDefinition
from victor.tools.enums import SchemaLevel


class FakeBaseTool:
    """Minimal BaseTool stub for testing tool_to_definition."""

    def __init__(self, name, description, parameters=None):
        self.name = name
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}}

    def to_schema(self, level):
        """Simulate schema truncation per level."""
        desc = self.description
        params = dict(self.parameters)

        if level == SchemaLevel.COMPACT:
            desc = desc[:150] + "..." if len(desc) > 150 else desc
        elif level == SchemaLevel.STUB:
            desc = desc[:80] + "..." if len(desc) > 80 else desc
            # Remove optional params for STUB
            if "properties" in params:
                required = params.get("required", [])
                params["properties"] = {
                    k: v for k, v in params["properties"].items() if k in required
                }

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": params,
            },
        }


class TestToolToDefinition:
    def _call(self, tool, level=SchemaLevel.FULL):
        from victor.agent.tool_selection import tool_to_definition

        return tool_to_definition(tool, level)

    def test_full_level_preserves_description(self):
        tool = FakeBaseTool("read", "Read a file from disk with full options " * 5)
        td = self._call(tool, SchemaLevel.FULL)
        assert td.name == "read"
        assert td.schema_level == "full"
        assert len(td.description) == len(tool.description)

    def test_compact_level_truncates(self):
        long_desc = "A" * 200
        tool = FakeBaseTool("write", long_desc)
        td = self._call(tool, SchemaLevel.COMPACT)
        assert td.schema_level == "compact"
        assert len(td.description) < len(long_desc)

    def test_stub_level_truncates_more(self):
        long_desc = "B" * 200
        tool = FakeBaseTool("search", long_desc)
        td = self._call(tool, SchemaLevel.STUB)
        assert td.schema_level == "stub"
        assert len(td.description) < 100

    def test_stub_drops_optional_params(self):
        tool = FakeBaseTool(
            "read",
            "Read file",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        )
        td = self._call(tool, SchemaLevel.STUB)
        assert "path" in td.parameters["properties"]
        assert "limit" not in td.parameters["properties"]

    def test_schema_level_excluded_from_serialization(self):
        tool = FakeBaseTool("test", "Test")
        td = self._call(tool, SchemaLevel.STUB)
        dump = td.model_dump()
        assert "schema_level" not in dump
        assert td.schema_level == "stub"


class TestToolDefinitionSchemaLevel:
    def test_default_none(self):
        td = ToolDefinition(name="x", description="y", parameters={})
        assert td.schema_level is None

    def test_explicit_value(self):
        td = ToolDefinition(name="x", description="y", parameters={}, schema_level="compact")
        assert td.schema_level == "compact"

    def test_excluded_from_dump(self):
        td = ToolDefinition(name="x", description="y", parameters={}, schema_level="full")
        dump = td.model_dump()
        assert "schema_level" not in dump


class TestTierAwareOrdering:
    def test_full_before_compact_before_stub(self):
        tools = [
            ToolDefinition(name="z_stub", description="", parameters={}, schema_level="stub"),
            ToolDefinition(name="a_full", description="", parameters={}, schema_level="full"),
            ToolDefinition(name="m_compact", description="", parameters={}, schema_level="compact"),
            ToolDefinition(name="b_full", description="", parameters={}, schema_level="full"),
        ]

        level_order = {"full": 0, "compact": 1, "stub": 2, None: 2}
        sorted_tools = sorted(
            tools,
            key=lambda t: (level_order.get(t.schema_level, 2), t.name),
        )

        assert sorted_tools[0].name == "a_full"
        assert sorted_tools[1].name == "b_full"
        assert sorted_tools[2].name == "m_compact"
        assert sorted_tools[3].name == "z_stub"

    def test_none_schema_level_treated_as_stub(self):
        tools = [
            ToolDefinition(name="unknown", description="", parameters={}, schema_level=None),
            ToolDefinition(name="known", description="", parameters={}, schema_level="full"),
        ]

        level_order = {"full": 0, "compact": 1, "stub": 2, None: 2}
        sorted_tools = sorted(
            tools,
            key=lambda t: (level_order.get(t.schema_level, 2), t.name),
        )

        assert sorted_tools[0].name == "known"
        assert sorted_tools[1].name == "unknown"


class TestCacheBoundary:
    def test_find_boundary_at_full_compact_end(self):
        from victor.providers.anthropic_provider import AnthropicProvider

        tools = [
            ToolDefinition(name="a", description="", parameters={}, schema_level="full"),
            ToolDefinition(name="b", description="", parameters={}, schema_level="compact"),
            ToolDefinition(name="c", description="", parameters={}, schema_level="stub"),
            ToolDefinition(name="d", description="", parameters={}, schema_level="stub"),
        ]
        converted = [{"name": t.name} for t in tools]

        idx = AnthropicProvider._find_cache_boundary(tools, converted)
        assert idx == 1  # Last COMPACT tool, before first STUB

    def test_all_full_boundary_at_last(self):
        from victor.providers.anthropic_provider import AnthropicProvider

        tools = [
            ToolDefinition(name="a", description="", parameters={}, schema_level="full"),
            ToolDefinition(name="b", description="", parameters={}, schema_level="full"),
        ]
        converted = [{"name": t.name} for t in tools]

        idx = AnthropicProvider._find_cache_boundary(tools, converted)
        assert idx == 1  # Last tool (all are stable)

    def test_all_stub_boundary_at_last(self):
        from victor.providers.anthropic_provider import AnthropicProvider

        tools = [
            ToolDefinition(name="a", description="", parameters={}, schema_level="stub"),
        ]
        converted = [{"name": t.name} for t in tools]

        idx = AnthropicProvider._find_cache_boundary(tools, converted)
        assert idx == 0  # Only tool, must be cached


class TestMCPDefaultSchemaLevel:
    def test_mcp_adapter_defaults_to_stub(self):
        from victor.tools.mcp_adapter_tool import MCPAdapterTool

        fake_tool = MagicMock(name="search", description="Search", parameters=[])
        fake_registry = MagicMock()
        adapter = MCPAdapterTool(fake_tool, fake_registry, "server")
        assert adapter.default_schema_level == "stub"

    def test_langchain_adapter_defaults_to_stub(self):
        from victor.tools.langchain_adapter_tool import LangChainAdapterTool

        fake_lc = MagicMock()
        fake_lc.name = "search"
        fake_lc.description = "Search"
        fake_lc.args_schema = None
        adapter = LangChainAdapterTool(fake_lc)
        assert adapter.default_schema_level == "stub"


class TestConfigGate:
    def test_tiered_schema_setting_exists(self):
        from victor.config.context_settings import ContextSettings

        settings = ContextSettings()
        assert hasattr(settings, "tiered_schema_enabled")
        assert settings.tiered_schema_enabled is True
