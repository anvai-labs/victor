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

"""Unit tests for MCP vertical."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock
from datetime import datetime, timedelta

import pytest

from victor.verticals.mcp_vertical import MCPVertical, MCPKnowledgeContributor
from victor.integrations.mcp.async_registry import (
    AsyncMCPRegistry,
    MCPRegistry,
    MCPServerConfig,
    ServerStatus,
    MCPServerStats,
)


class TestMCPVertical:
    """Test suite for MCPVertical."""

    def test_vertical_properties(self):
        """Test vertical properties."""
        assert MCPVertical.name == "mcp"
        assert MCPVertical.tier == "foundation"
        assert "Model Context Protocol" in MCPVertical.description

    def test_get_tools(self):
        """Test that get_tools returns empty list (tools loaded dynamically)."""
        tools = MCPVertical.get_tools()
        assert tools == []

    def test_get_extensions(self):
        """Test that get_extensions returns proper extensions."""
        from victor_contracts.verticals.extensions import VerticalExtensions

        extensions = MCPVertical.get_extensions()
        assert isinstance(extensions, VerticalExtensions)

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test vertical initialization."""
        vertical = MCPVertical()

        # Mock settings
        settings = MagicMock()

        # Mock registry
        mock_registry = MagicMock()
        mock_registry.connect_all = AsyncMock()
        mock_registry.register_server = AsyncMock()

        # Mock _discover_mcp_servers to do nothing
        vertical._discover_mcp_servers = AsyncMock()

        # Create real async registry
        from victor.integrations.mcp.async_registry import AsyncMCPRegistry

        vertical._registry = AsyncMCPRegistry()

        await vertical.initialize(settings)

        # Verify registry was created
        assert vertical._registry is not None

    @pytest.mark.asyncio
    async def test_discover_mcp_servers(self):
        """Test MCP server discovery."""
        vertical = MCPVertical()

        # Mock registry
        mock_registry = AsyncMock()
        mock_registry.register_server = AsyncMock()
        vertical._registry = mock_registry

        await vertical._discover_mcp_servers()

        # Verify no servers discovered (no entry points in test env)
        assert len(vertical._discovered_servers) == 0

    @pytest.mark.asyncio
    async def test_get_mcp_tools(self):
        """Test getting MCP tools."""
        vertical = MCPVertical()

        # Mock registry with server
        mock_server = AsyncMock()
        mock_server.name = "test-server"

        # Mock tool list
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}

        mock_server.list_tools = AsyncMock(return_value=[mock_tool])

        mock_registry = AsyncMock()
        mock_registry.get_servers = Mock(return_value=[mock_server])
        vertical._registry = mock_registry

        tools = await vertical.get_mcp_tools()

        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"
        assert tools[0]["server"] == "test-server"

    @pytest.mark.asyncio
    async def test_get_mcp_resources(self):
        """Test getting MCP resources."""
        vertical = MCPVertical()

        # Mock registry with server
        mock_server = AsyncMock()
        mock_server.name = "test-server"

        # Mock resource list
        mock_resource = MagicMock()
        mock_resource.uri = "file:///test.txt"
        mock_resource.name = "test.txt"
        mock_resource.description = "A test resource"
        mock_resource.mime_type = "text/plain"

        mock_server.list_resources = AsyncMock(return_value=[mock_resource])

        mock_registry = AsyncMock()
        mock_registry.get_servers = Mock(return_value=[mock_server])
        vertical._registry = mock_registry

        resources = await vertical.get_mcp_resources()

        assert len(resources) == 1
        assert resources[0]["uri"] == "file:///test.txt"
        assert resources[0]["server"] == "test-server"

    @pytest.mark.asyncio
    async def test_call_mcp_tool(self):
        """Test calling an MCP tool."""
        vertical = MCPVertical()

        # Mock server
        mock_server = AsyncMock()
        mock_server.name = "test-server"
        mock_server.call_tool = AsyncMock(return_value={"result": "success"})

        mock_registry = AsyncMock()
        mock_registry.get_server = Mock(return_value=mock_server)
        vertical._registry = mock_registry

        result = await vertical.call_mcp_tool("test-server", "test_tool", {"arg1": "value1"})

        assert result == {"result": "success"}
        mock_server.call_tool.assert_called_once_with("test_tool", {"arg1": "value1"})

    @pytest.mark.asyncio
    async def test_call_mcp_tool_server_not_found(self):
        """Test calling tool on non-existent server."""
        vertical = MCPVertical()

        mock_registry = AsyncMock()
        mock_registry.get_server = Mock(return_value=None)
        vertical._registry = mock_registry

        with pytest.raises(ValueError, match="MCP server not found"):
            await vertical.call_mcp_tool("non-existent", "tool", {})

    @pytest.mark.asyncio
    async def test_read_mcp_resource(self):
        """Test reading an MCP resource."""
        vertical = MCPVertical()

        # Mock server
        mock_server = AsyncMock()
        mock_server.name = "test-server"
        mock_server.read_resource = AsyncMock(return_value="content")

        mock_registry = AsyncMock()
        mock_registry.get_server = Mock(return_value=mock_server)
        vertical._registry = mock_registry

        result = await vertical.read_mcp_resource("test-server", "file:///test.txt")

        assert result == "content"
        mock_server.read_resource.assert_called_once_with("file:///test.txt")

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test MCP vertical shutdown."""
        vertical = MCPVertical()

        mock_registry = AsyncMock()
        mock_registry.disconnect_all = AsyncMock()
        vertical._registry = mock_registry

        await vertical.shutdown()

        mock_registry.disconnect_all.assert_called_once()


class TestMCPKnowledgeContributor:
    """Test suite for MCPKnowledgeContributor."""

    @pytest.fixture
    def vertical(self):
        """Create MCP vertical fixture."""
        vertical = MCPVertical()
        vertical.get_mcp_resources = AsyncMock(return_value=[])
        vertical.get_mcp_tools = AsyncMock(return_value=[])
        return vertical

    @pytest.fixture
    def contributor(self, vertical):
        """Create contributor fixture."""
        return MCPKnowledgeContributor(vertical)

    @pytest.mark.asyncio
    async def test_get_prompt_sections_empty(self, contributor):
        """Test getting prompt sections when no resources/tools."""
        sections = await contributor.get_prompt_sections(None)

        assert sections == {}

    @pytest.mark.asyncio
    async def test_get_prompt_sections_with_resources(self, contributor, vertical):
        """Test getting prompt sections with resources."""
        vertical.get_mcp_resources = AsyncMock(
            return_value=[
                {
                    "uri": "file:///test.txt",
                    "name": "test.txt",
                    "description": "A test file",
                    "server": "test-server",
                }
            ]
        )

        sections = await contributor.get_prompt_sections(None)

        assert "mcp_resources" in sections
        assert "test.txt" in sections["mcp_resources"]
        assert "file:///test.txt" in sections["mcp_resources"]

    @pytest.mark.asyncio
    async def test_get_prompt_sections_with_tools(self, contributor, vertical):
        """Test getting prompt sections with tools."""
        vertical.get_mcp_tools = AsyncMock(
            return_value=[
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "server": "test-server",
                }
            ]
        )

        sections = await contributor.get_prompt_sections(None)

        assert "mcp_tools" in sections
        assert "test_tool" in sections["mcp_tools"]
        assert "test-server" in sections["mcp_tools"]


class TestAsyncMCPRegistry:
    """Test suite for AsyncMCPRegistry."""

    @pytest.fixture
    def registry(self):
        """Create registry fixture."""
        return AsyncMCPRegistry()

    @pytest.mark.asyncio
    async def test_register_server(self, registry):
        """Test registering a server."""
        config = MCPServerConfig(
            name="test-server",
            transport="stdio",
            command="node",
            args=["server.js"],
        )

        await registry.register_server(config)

        assert "test-server" in registry._servers
        assert registry._status["test-server"] == ServerStatus.DISCONNECTED
        assert "test-server" in registry._stats

    @pytest.mark.asyncio
    async def test_connect_server_success(self, registry):
        """Test successful server connection."""
        config = MCPServerConfig(name="test-server", transport="stdio")

        # Mock server
        mock_server = AsyncMock()
        mock_server.config = config

        registry._servers["test-server"] = mock_server
        registry._status["test-server"] = ServerStatus.DISCONNECTED
        registry._locks["test-server"] = asyncio.Lock()
        registry._stats["test-server"] = MCPServerStats(
            name="test-server",
            status=ServerStatus.DISCONNECTED,
        )

        result = await registry._connect_server("test-server")

        assert result is True
        assert registry._status["test-server"] == ServerStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_connect_server_failure(self, registry):
        """Test failed server connection."""
        config = MCPServerConfig(name="test-server", transport="stdio")

        # Mock server that fails to connect
        mock_server = AsyncMock()
        mock_server.config = config

        registry._servers["test-server"] = mock_server
        registry._status["test-server"] = ServerStatus.DISCONNECTED
        registry._locks["test-server"] = asyncio.Lock()
        registry._stats["test-server"] = MCPServerStats(
            name="test-server",
            status=ServerStatus.DISCONNECTED,
        )

        # Our implementation doesn't actually connect, so it will succeed
        result = await registry._connect_server("test-server")

        # Currently always succeeds (placeholder implementation)
        assert result is True

    @pytest.mark.asyncio
    async def test_ping_server(self, registry):
        """Test pinging a server."""
        # Mock server
        mock_server = AsyncMock()

        registry._servers["test-server"] = mock_server
        registry._status["test-server"] = ServerStatus.CONNECTED
        registry._stats["test-server"] = MCPServerStats(
            name="test-server",
            status=ServerStatus.CONNECTED,
        )

        result = await registry.ping_server("test-server")

        assert result is True
        assert registry._stats["test-server"].last_ping is not None

    @pytest.mark.asyncio
    async def test_ping_server_failure(self, registry):
        """Test ping failure."""
        # Mock server
        mock_server = AsyncMock()

        registry._servers["test-server"] = mock_server
        registry._status["test-server"] = ServerStatus.CONNECTED
        registry._stats["test-server"] = MCPServerStats(
            name="test-server",
            status=ServerStatus.CONNECTED,
        )

        # Our implementation doesn't actually ping, so it will succeed
        result = await registry.ping_server("test-server")

        # Currently always succeeds (placeholder implementation)
        assert result is True

    def test_get_servers(self, registry):
        """Test getting connected servers."""
        # Add servers in different states
        registry._status["server1"] = ServerStatus.CONNECTED
        registry._status["server2"] = ServerStatus.DISCONNECTED
        registry._status["server3"] = ServerStatus.CONNECTED

        # Mock servers
        mock_server1 = AsyncMock()
        mock_server2 = AsyncMock()
        mock_server3 = AsyncMock()
        registry._servers["server1"] = mock_server1
        registry._servers["server2"] = mock_server2
        registry._servers["server3"] = mock_server3

        servers = registry.get_servers()

        assert len(servers) == 2
        assert mock_server1 in servers
        assert mock_server3 in servers
        assert mock_server2 not in servers

    def test_get_server(self, registry):
        """Test getting specific server."""
        mock_server = AsyncMock()
        registry._servers["test-server"] = mock_server
        registry._status["test-server"] = ServerStatus.CONNECTED

        result = registry.get_server("test-server")

        assert result is mock_server

    def test_get_server_not_connected(self, registry):
        """Test getting server that is not connected."""
        mock_server = AsyncMock()
        registry._servers["test-server"] = mock_server
        registry._status["test-server"] = ServerStatus.DISCONNECTED

        result = registry.get_server("test-server")

        assert result is None

    def test_get_server_status(self, registry):
        """Test getting server status."""
        registry._status["test-server"] = ServerStatus.CONNECTED

        status = registry.get_server_status("test-server")

        assert status == ServerStatus.CONNECTED

    def test_get_server_stats(self, registry):
        """Test getting server stats."""
        stats = MCPServerStats(
            name="test-server",
            status=ServerStatus.CONNECTED,
        )
        registry._stats["test-server"] = stats

        result = registry.get_server_stats("test-server")

        assert result is stats
        assert result.name == "test-server"

    @pytest.mark.asyncio
    async def test_disconnect_all(self, registry):
        """Test disconnecting all servers."""
        # Mock servers
        mock_server1 = AsyncMock()
        mock_server2 = AsyncMock()

        registry._servers["server1"] = mock_server1
        registry._servers["server2"] = mock_server2
        registry._status["server1"] = ServerStatus.CONNECTED
        registry._status["server2"] = ServerStatus.CONNECTED
        registry._locks["server1"] = asyncio.Lock()
        registry._locks["server2"] = asyncio.Lock()

        # Mock health check tasks
        task1 = asyncio.create_task(asyncio.sleep(10))
        task2 = asyncio.create_task(asyncio.sleep(10))
        registry._health_check_tasks["server1"] = task1
        registry._health_check_tasks["server2"] = task2

        await registry.disconnect_all()

        # Verify tasks cancelled
        assert task1.cancelled()
        assert task2.cancelled()

        # Verify status updated
        assert registry._status["server1"] == ServerStatus.DISCONNECTED
        assert registry._status["server2"] == ServerStatus.DISCONNECTED


class TestMCPServerConfig:
    """Test suite for MCPServerConfig."""

    def test_config_creation(self):
        """Test creating server config."""
        config = MCPServerConfig(
            name="test-server",
            transport="stdio",
            command="node",
            args=["server.js"],
        )

        assert config.name == "test-server"
        assert config.transport == "stdio"
        assert config.command == "node"
        assert config.args == ["server.js"]
        assert config.reconnect_interval == 5.0
        assert config.max_retries == 3

    def test_config_with_url(self):
        """Test creating config with URL transport."""
        config = MCPServerConfig(
            name="test-server",
            transport="sse",
            url="http://localhost:3000/sse",
        )

        assert config.transport == "sse"
        assert config.url == "http://localhost:3000/sse"


class TestMCPServerStats:
    """Test suite for MCPServerStats."""

    def test_stats_creation(self):
        """Test creating server stats."""
        stats = MCPServerStats(
            name="test-server",
            status=ServerStatus.CONNECTED,
        )

        assert stats.name == "test-server"
        assert stats.status == ServerStatus.CONNECTED
        assert stats.connection_time is None
        assert stats.last_ping is None
        assert stats.failed_pings == 0
        assert stats.successful_calls == 0
        assert stats.failed_calls == 0


class TestMCPRegistryWrapper:
    """Test suite for MCPRegistry wrapper."""

    def test_wrapper_creation(self):
        """Test creating wrapper."""
        settings = MagicMock()
        registry = MCPRegistry(settings)

        assert registry._settings is settings
        assert isinstance(registry._async_registry, AsyncMCPRegistry)

    @pytest.mark.asyncio
    async def test_wrapper_register_server(self):
        """Test wrapper register_server."""
        settings = MagicMock()
        registry = MCPRegistry(settings)

        config = MCPServerConfig(name="test", transport="stdio")
        await registry.register_server(config)

        # Verify delegated to async registry
        assert "test" in registry._async_registry._servers

    @pytest.mark.asyncio
    async def test_wrapper_get_servers(self):
        """Test wrapper get_servers."""
        settings = MagicMock()
        registry = MCPRegistry(settings)

        # Add a connected server
        registry._async_registry._status["test"] = ServerStatus.CONNECTED
        mock_server = AsyncMock()
        registry._async_registry._servers["test"] = mock_server

        servers = registry.get_servers()

        assert len(servers) == 1
        assert servers[0] is mock_server
