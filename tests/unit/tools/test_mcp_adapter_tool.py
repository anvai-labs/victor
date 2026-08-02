# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for MCPAdapterTool and MCPToolProjector."""

from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass, field

import pytest

from victor.tools.mcp_adapter_tool import (
    MCPAdapterTool,
    MCPToolProjector,
    _mcp_param_to_json_schema,
    _mcp_params_to_json_schema,
)


# Lightweight stubs matching MCPTool/MCPParameter shape (avoid MCP dependency)
@dataclass
class FakeMCPParameter:
    name: str
    type: MagicMock  # .value returns type string
    description: str
    required: bool = False
    default: object = None


@dataclass
class FakeMCPTool:
    name: str
    description: str
    parameters: list = field(default_factory=list)
    version: str = "1.0.0"


def _param(name, type_str, desc, required=False, default=None):
    t = MagicMock()
    t.value = type_str
    return FakeMCPParameter(name=name, type=t, description=desc, required=required, default=default)


def _tool(name, desc, params=None):
    return FakeMCPTool(name=name, description=desc, parameters=params or [])


@dataclass
class FakeServerEntry:
    tools_cache: list = field(default_factory=list)


class TestMCPParamConversion:
    def test_string_param(self):
        p = _param("query", "string", "Search query", required=True)
        schema = _mcp_param_to_json_schema(p)
        assert schema == {"type": "string", "description": "Search query"}

    def test_number_param_with_default(self):
        p = _param("limit", "number", "Max results", default=10)
        schema = _mcp_param_to_json_schema(p)
        assert schema == {"type": "number", "description": "Max results", "default": 10}

    def test_boolean_param(self):
        p = _param("verbose", "boolean", "Verbose output")
        schema = _mcp_param_to_json_schema(p)
        assert schema["type"] == "boolean"

    def test_params_to_json_schema(self):
        params = [
            _param("query", "string", "Search query", required=True),
            _param("limit", "number", "Max results", default=10),
        ]
        schema = _mcp_params_to_json_schema(params)
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_empty_params(self):
        schema = _mcp_params_to_json_schema([])
        assert schema == {"type": "object", "properties": {}}


class TestMCPAdapterTool:
    def test_name_without_prefix(self):
        tool = _tool("github_search", "Search GitHub repos")
        registry = MagicMock()
        adapter = MCPAdapterTool(tool, registry, "github-server")
        assert adapter.name == "mcp_github_search"

    def test_name_with_prefix(self):
        tool = _tool("search", "Search")
        registry = MagicMock()
        adapter = MCPAdapterTool(tool, registry, "github-server", name_prefix="github")
        assert adapter.name == "github_search"

    def test_description_includes_server(self):
        tool = _tool("search", "Search repos")
        registry = MagicMock()
        adapter = MCPAdapterTool(tool, registry, "github-server")
        assert "github-server" in adapter.description
        assert "Search repos" in adapter.description

    def test_parameters_json_schema(self):
        params = [_param("q", "string", "Query", required=True)]
        tool = _tool("search", "Search", params)
        registry = MagicMock()
        adapter = MCPAdapterTool(tool, registry, "server")
        schema = adapter.parameters
        assert schema["type"] == "object"
        assert "q" in schema["properties"]
        assert schema["required"] == ["q"]

    def test_mcp_server_name_property(self):
        tool = _tool("search", "Search")
        registry = MagicMock()
        adapter = MCPAdapterTool(tool, registry, "my-server")
        assert adapter.mcp_server_name == "my-server"

    def test_mcp_tool_name_property(self):
        tool = _tool("search", "Search")
        registry = MagicMock()
        adapter = MCPAdapterTool(tool, registry, "server", name_prefix="mcp")
        assert adapter.mcp_tool_name == "search"  # Original, not prefixed

    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = _tool("search", "Search")
        registry = MagicMock()
        result_mock = MagicMock(success=True, result="found 5 repos", error=None)
        registry.call_tool = AsyncMock(return_value=result_mock)

        adapter = MCPAdapterTool(tool, registry, "server")
        result = await adapter.execute({}, query="python")

        registry.call_tool.assert_called_once_with("search", query="python")
        assert result.success is True
        assert result.output == "found 5 repos"
        assert result.metadata["mcp_server"] == "server"

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        tool = _tool("search", "Search")
        registry = MagicMock()
        result_mock = MagicMock(success=False, result=None, error="timeout")
        registry.call_tool = AsyncMock(return_value=result_mock)

        adapter = MCPAdapterTool(tool, registry, "server")
        result = await adapter.execute({})

        assert result.success is False
        assert result.error == "timeout"

    @pytest.mark.asyncio
    async def test_execute_exception(self):
        tool = _tool("search", "Search")
        registry = MagicMock()
        registry.call_tool = AsyncMock(side_effect=ConnectionError("server down"))

        adapter = MCPAdapterTool(tool, registry, "server")
        result = await adapter.execute({})

        assert result.success is False
        assert "server down" in result.error


class TestMCPToolProjector:
    def _make_registry(self, servers_dict):
        """Create a mock registry with _servers dict."""
        registry = MagicMock()
        registry._servers = servers_dict
        return registry

    def test_project_single_server(self):
        tools = [_tool("read", "Read file"), _tool("write", "Write file")]
        servers = {"fs-server": FakeServerEntry(tools_cache=tools)}
        registry = self._make_registry(servers)

        result = MCPToolProjector.project(registry)
        assert len(result) == 2
        names = {t.name for t in result}
        assert names == {"mcp_read", "mcp_write"}

    def test_project_multiple_servers(self):
        servers = {
            "github": FakeServerEntry(tools_cache=[_tool("search", "Search")]),
            "slack": FakeServerEntry(tools_cache=[_tool("send", "Send message")]),
        }
        registry = self._make_registry(servers)

        result = MCPToolProjector.project(registry)
        assert len(result) == 2
        names = {t.name for t in result}
        assert names == {"mcp_search", "mcp_send"}

    def test_project_with_prefix(self):
        servers = {"fs": FakeServerEntry(tools_cache=[_tool("read", "Read")])}
        registry = self._make_registry(servers)

        result = MCPToolProjector.project(registry, prefix="mcp")
        assert len(result) == 1
        assert result[0].name == "mcp_read"

    def test_project_name_collision_prefix_server(self):
        servers = {
            "github": FakeServerEntry(tools_cache=[_tool("search", "GH search")]),
            "jira": FakeServerEntry(tools_cache=[_tool("search", "Jira search")]),
        }
        registry = self._make_registry(servers)

        result = MCPToolProjector.project(registry, conflict_strategy="prefix_server")
        assert len(result) == 2
        names = {t.name for t in result}
        assert "mcp_search" in names
        assert "mcp_jira_search" in names

    def test_project_name_collision_skip(self):
        servers = {
            "github": FakeServerEntry(tools_cache=[_tool("search", "GH search")]),
            "jira": FakeServerEntry(tools_cache=[_tool("search", "Jira search")]),
        }
        registry = self._make_registry(servers)

        result = MCPToolProjector.project(registry, conflict_strategy="skip")
        assert len(result) == 1
        assert result[0].name == "mcp_search"

    def test_project_empty_registry(self):
        registry = self._make_registry({})
        result = MCPToolProjector.project(registry)
        assert result == []

    def test_project_server_with_no_tools(self):
        servers = {"empty": FakeServerEntry(tools_cache=[])}
        registry = self._make_registry(servers)
        result = MCPToolProjector.project(registry)
        assert result == []
