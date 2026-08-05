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

"""Integration tests for MCP (Model Context Protocol) functionality.

Tests cover:
- MCP registry initialization and server management
- Server registration and unregistration
- Tool and resource caching
- Multi-server coordination

These tests use the actual MCP registry implementation with mocked clients
to avoid requiring actual MCP servers to be running.
"""

import json
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _check_mcp_available():
    """Check if MCP module is available."""
    try:
        from victor.integrations.mcp.registry import MCPRegistry
        from victor.integrations.mcp.client import MCPClient

        return True
    except ImportError:
        return False


# Skip entire module if MCP is not available
pytestmark = pytest.mark.skipif(
    not _check_mcp_available(),
    reason="MCP module not available",
)


# Import after skipif check
from victor.integrations.mcp.registry import (  # noqa: E402
    MCPRegistry,
    MCPServerConfig,
    ServerStatus,
    ServerEntry,
)
from victor.integrations.mcp.client import MCPClient  # noqa: E402
from victor.integrations.mcp.protocol import MCPTool, MCPResource  # noqa: E402


class TestMCPRegistryBasics:
    """Tests for basic MCP registry operations."""

    def test_registry_initialization(self):
        """Test MCP registry initializes correctly."""
        registry = MCPRegistry()
        assert registry is not None
        assert len(registry.list_servers()) == 0

    def test_register_server(self):
        """Test registering a server configuration."""
        registry = MCPRegistry()
        config = MCPServerConfig(
            name="test_server",
            command=["python", "-m", "test.server"],
            description="Test server",
        )

        registry.register_server(config)

        servers = registry.list_servers()
        assert len(servers) == 1
        assert "test_server" in servers

    def test_register_multiple_servers(self):
        """Test registering multiple servers."""
        registry = MCPRegistry()

        for i in range(3):
            config = MCPServerConfig(
                name=f"server_{i}",
                command=["test-cmd"],
            )
            registry.register_server(config)

        servers = registry.list_servers()
        assert len(servers) == 3

    def test_unregister_server(self):
        """Test unregistering a server."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["cmd"])

        registry.register_server(config)
        assert len(registry.list_servers()) == 1

        result = registry.unregister_server("test")
        assert result is True
        assert len(registry.list_servers()) == 0

    def test_unregister_nonexistent_server(self):
        """Test unregistering a server that doesn't exist."""
        registry = MCPRegistry()
        result = registry.unregister_server("nonexistent")
        assert result is False


class TestMCPServerConfig:
    """Tests for MCP server configuration."""

    def test_config_defaults(self):
        """Test MCPServerConfig default values."""
        config = MCPServerConfig(
            name="test",
            command=["test-command"],
        )

        assert config.name == "test"
        assert config.command == ["test-command"]
        assert config.auto_connect is True
        assert config.health_check_interval == 30
        assert config.max_retries == 3
        assert config.enabled is True
        assert config.tags == []
        assert config.env == {}

    def test_config_custom_values(self):
        """Test MCPServerConfig with custom values."""
        config = MCPServerConfig(
            name="custom",
            command=["custom-cmd", "--arg"],
            description="Custom server",
            auto_connect=False,
            health_check_interval=60,
            max_retries=5,
            retry_delay=10,
            enabled=False,
            tags=["tag1", "tag2"],
            env={"KEY": "value"},
        )

        assert config.name == "custom"
        assert config.command == ["custom-cmd", "--arg"]
        assert config.description == "Custom server"
        assert config.auto_connect is False
        assert config.health_check_interval == 60
        assert config.max_retries == 5
        assert config.retry_delay == 10
        assert config.enabled is False
        assert config.tags == ["tag1", "tag2"]
        assert config.env == {"KEY": "value"}


class TestMCPToolOperations:
    """Tests for MCP tool operations."""

    @pytest.fixture
    def registry_with_tools(self):
        """Create registry with mocked server and tools."""
        registry = MCPRegistry()

        # Create mock server entry with tools
        config = MCPServerConfig(name="tool_server", command=["cmd"])
        registry.register_server(config)

        # Add tools to cache
        entry = registry._servers["tool_server"]
        entry.status = ServerStatus.CONNECTED
        entry.tools_cache = [
            MCPTool(
                name="read_file",
                description="Read a file",
                inputSchema={"type": "object"},
            ),
            MCPTool(
                name="write_file",
                description="Write a file",
                inputSchema={"type": "object"},
            ),
        ]

        # Map tools to server
        registry._tool_to_server["read_file"] = "tool_server"
        registry._tool_to_server["write_file"] = "tool_server"

        return registry

    def test_get_all_tools(self, registry_with_tools):
        """Test getting all tools from all servers."""
        tools = registry_with_tools.get_all_tools()

        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names

    def test_get_tool_by_name(self, registry_with_tools):
        """Test getting a specific tool by name."""
        tool = registry_with_tools.get_tool("read_file")

        assert tool is not None
        assert tool.name == "read_file"

    def test_get_nonexistent_tool(self, registry_with_tools):
        """Test getting a tool that doesn't exist."""
        tool = registry_with_tools.get_tool("nonexistent")
        assert tool is None

    def test_get_tools_by_server(self, registry_with_tools):
        """Test getting tools from a specific server."""
        tools = registry_with_tools.get_tools_by_server("tool_server")

        assert len(tools) == 2

    def test_get_tools_from_nonexistent_server(self, registry_with_tools):
        """Test getting tools from a server that doesn't exist."""
        tools = registry_with_tools.get_tools_by_server("nonexistent")
        assert tools == []


class TestMCPResourceOperations:
    """Tests for MCP resource operations."""

    @pytest.fixture
    def registry_with_resources(self):
        """Create registry with mocked server and resources."""
        registry = MCPRegistry()

        config = MCPServerConfig(name="resource_server", command=["cmd"])
        registry.register_server(config)

        entry = registry._servers["resource_server"]
        entry.status = ServerStatus.CONNECTED
        entry.resources_cache = [
            MCPResource(
                uri="file:///tmp/test.txt",
                name="test.txt",
                description="A test file",
                mime_type="text/plain",
            ),
            MCPResource(
                uri="file:///tmp/data.json",
                name="data.json",
                description="JSON data file",
                mime_type="application/json",
            ),
        ]

        return registry

    def test_get_all_resources(self, registry_with_resources):
        """Test getting all resources from all servers."""
        resources = registry_with_resources.get_all_resources()

        assert len(resources) == 2
        uris = [r.uri for r in resources]
        assert "file:///tmp/test.txt" in uris
        assert "file:///tmp/data.json" in uris


class TestMCPRegistryStatus:
    """Tests for registry status operations."""

    def test_get_registry_status(self):
        """Test getting overall registry status."""
        registry = MCPRegistry()

        # Add some servers
        for i in range(3):
            config = MCPServerConfig(name=f"server_{i}", command=["cmd"])
            registry.register_server(config)

        status = registry.get_registry_status()

        assert "servers" in status
        assert status["total_servers"] == 3
        assert status["connected_servers"] == 0

    def test_get_server_status(self):
        """Test getting individual server status."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["cmd"])
        registry.register_server(config)

        status = registry.get_server_status("test")

        assert status is not None
        assert "status" in status

    def test_get_nonexistent_server_status(self):
        """Test getting status of non-existent server."""
        registry = MCPRegistry()
        status = registry.get_server_status("nonexistent")
        assert status is None


class TestMCPServerConnection:
    """Tests for server connection operations."""

    @pytest.mark.asyncio
    async def test_connect_nonexistent_server(self):
        """Test connecting to a server that doesn't exist."""
        registry = MCPRegistry()
        result = await registry.connect("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_server(self):
        """Test disconnecting a server that doesn't exist."""
        registry = MCPRegistry()
        result = await registry.disconnect("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_all_empty_registry(self):
        """Test connect_all on empty registry."""
        registry = MCPRegistry()
        results = await registry.connect_all()
        assert results == {}


class TestMCPToolExecution:
    """Tests for MCP tool execution."""

    @pytest.fixture
    def registry_with_client(self):
        """Create registry with mocked client for tool execution."""
        registry = MCPRegistry()

        config = MCPServerConfig(name="exec_server", command=["cmd"])
        registry.register_server(config)

        # Create mock client
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(
            return_value=MagicMock(
                success=True,
                result={"content": "File content"},
                error=None,
            )
        )

        entry = registry._servers["exec_server"]
        entry.client = mock_client
        entry.status = ServerStatus.CONNECTED
        entry.tools_cache = [
            MCPTool(name="read_file", description="Read", inputSchema={}),
        ]
        registry._tool_to_server["read_file"] = "exec_server"

        return registry

    @pytest.mark.asyncio
    async def test_call_tool_success(self, registry_with_client):
        """Test successful tool execution."""
        result = await registry_with_client.call_tool("read_file", path="/test.txt")

        assert result is not None
        assert result.success is True

    @pytest.mark.asyncio
    async def test_call_nonexistent_tool(self):
        """Test calling a tool that doesn't exist."""
        registry = MCPRegistry()
        result = await registry.call_tool("nonexistent")

        assert result is not None
        assert result.success is False
        assert "not found" in result.error.lower()


class TestMCPServerDiscovery:
    """Tests for MCP server discovery functionality."""

    def test_discover_servers_creates_registry(self):
        """Test that discover_servers returns a registry."""
        registry = MCPRegistry.discover_servers(include_claude_desktop=False)
        assert isinstance(registry, MCPRegistry)

    def test_discover_claude_desktop_no_config(self, tmp_path):
        """Test discovery when Claude Desktop config doesn't exist."""
        # _discover_claude_desktop_servers is a classmethod
        servers = MCPRegistry._discover_claude_desktop_servers()
        # May return empty list if no Claude Desktop config exists
        assert isinstance(servers, list)


class TestMCPServerReset:
    """Tests for server reset functionality."""

    def test_reset_server(self):
        """Test resetting a failed server."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["cmd"])
        registry.register_server(config)

        # Simulate failure
        entry = registry._servers["test"]
        entry.status = ServerStatus.FAILED
        entry.consecutive_failures = 5
        entry.error_message = "Connection failed"

        result = registry.reset_server("test")

        assert result is True
        assert entry.status == ServerStatus.DISCONNECTED
        assert entry.consecutive_failures == 0
        assert entry.error_message is None

    def test_reset_nonexistent_server(self):
        """Test resetting a server that doesn't exist."""
        registry = MCPRegistry()
        result = registry.reset_server("nonexistent")
        assert result is False


class TestMCPEventHandling:
    """Tests for event handling in the registry."""

    def test_register_event_callback(self):
        """Test registering an event callback."""
        registry = MCPRegistry()

        events = []

        def callback(event_type, server_name, data):
            events.append((event_type, server_name, data))

        registry.on_event(callback)

        # Emit a test event
        registry._emit_event("test_event", "test_server", {"key": "value"})

        assert len(events) == 1
        assert events[0] == ("test_event", "test_server", {"key": "value"})


class TestMCPContextManager:
    """Tests for registry context manager usage."""

    @pytest.mark.asyncio
    async def test_registry_as_context_manager(self):
        """Test using registry as async context manager."""
        async with MCPRegistry() as registry:
            assert registry is not None
            # Add a server
            config = MCPServerConfig(name="test", command=["cmd"])
            registry.register_server(config)
            assert len(registry.list_servers()) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
