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

"""Tests for MCP registry module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.integrations.mcp.registry import (
    ServerStatus,
    MCPServerConfig,
    ServerEntry,
    MCPRegistry,
)


class TestServerStatus:
    """Tests for ServerStatus enum."""

    def test_all_statuses_defined(self):
        """Test all server statuses are defined."""
        assert ServerStatus.DISCONNECTED is not None
        assert ServerStatus.CONNECTING is not None
        assert ServerStatus.CONNECTED is not None
        assert ServerStatus.UNHEALTHY is not None
        assert ServerStatus.FAILED is not None


class TestMCPServerConfig:
    """Tests for MCPServerConfig model."""

    def test_minimal_config(self):
        """Test creating config with minimal fields."""
        config = MCPServerConfig(
            name="test_server",
            command=["python", "-m", "mcp_test"],
        )
        assert config.name == "test_server"
        assert config.command == ["python", "-m", "mcp_test"]
        assert config.auto_connect is True
        assert config.enabled is True

    def test_full_config(self):
        """Test creating config with all fields."""
        config = MCPServerConfig(
            name="full_server",
            command=["node", "server.js"],
            description="Test server",
            auto_connect=False,
            health_check_interval=60,
            max_retries=5,
            retry_delay=10,
            enabled=True,
            tags=["test", "local"],
            env={"API_KEY": "test"},
        )
        assert config.name == "full_server"
        assert config.description == "Test server"
        assert config.auto_connect is False
        assert config.health_check_interval == 60
        assert config.max_retries == 5
        assert config.tags == ["test", "local"]
        assert config.env["API_KEY"] == "test"


class TestServerEntry:
    """Tests for ServerEntry dataclass."""

    def test_default_entry(self):
        """Test default server entry values."""
        config = MCPServerConfig(name="test", command=["echo"])
        entry = ServerEntry(config=config)

        assert entry.config == config
        assert entry.client is None
        assert entry.status == ServerStatus.DISCONNECTED
        assert entry.last_health_check == 0.0
        assert entry.consecutive_failures == 0
        assert entry.error_message is None
        assert entry.tools_cache == []
        assert entry.resources_cache == []


class TestMCPRegistryInit:
    """Tests for MCPRegistry initialization."""

    def test_default_init(self):
        """Test default initialization."""
        registry = MCPRegistry()

        assert registry._servers == {}
        assert registry._tool_to_server == {}
        assert registry._health_check_enabled is True
        assert registry._running is False

    def test_init_with_params(self):
        """Test initialization with parameters."""
        registry = MCPRegistry(health_check_enabled=False, default_health_interval=60)

        assert registry._health_check_enabled is False
        assert registry._default_health_interval == 60


class TestMCPRegistryConnect:
    """Tests for connect/disconnect methods."""

    @pytest.mark.asyncio
    async def test_connect_unregistered_server(self):
        """Test connecting to unregistered server returns False."""
        registry = MCPRegistry()
        result = await registry.connect("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_disabled_server(self):
        """Test connecting to disabled server returns False."""
        registry = MCPRegistry()
        config = MCPServerConfig(
            name="disabled",
            command=["echo"],
            enabled=False,
        )
        registry.register_server(config)
        result = await registry.connect("disabled")
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""

        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.tools = []
        mock_client.resources = []

        with patch("victor.integrations.mcp.registry.MCPClient", return_value=mock_client):
            result = await registry.connect("test")
            assert result is True
            assert registry._servers["test"].status == ServerStatus.CONNECTED

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure."""

        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(return_value=False)

        with patch("victor.integrations.mcp.registry.MCPClient", return_value=mock_client):
            result = await registry.connect("test")
            assert result is False
            assert registry._servers["test"].status == ServerStatus.FAILED

    @pytest.mark.asyncio
    async def test_connect_exception(self):
        """Test connection exception handling."""

        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(side_effect=Exception("Connection error"))

        with patch("victor.integrations.mcp.registry.MCPClient", return_value=mock_client):
            result = await registry.connect("test")
            assert result is False
            assert registry._servers["test"].status == ServerStatus.FAILED
            assert "Connection error" in registry._servers["test"].error_message

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnecting from a server."""

        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        # Set up connected state
        mock_client = MagicMock()
        registry._servers["test"].client = mock_client
        registry._servers["test"].status = ServerStatus.CONNECTED
        registry._servers["test"].tools_cache = [MagicMock(name="tool1")]
        registry._tool_to_server["tool1"] = "test"

        result = await registry.disconnect("test")

        assert result is True
        assert registry._servers["test"].status == ServerStatus.DISCONNECTED
        assert registry._servers["test"].client is None
        assert "tool1" not in registry._tool_to_server

    @pytest.mark.asyncio
    async def test_disconnect_unregistered(self):
        """Test disconnecting from unregistered server."""
        registry = MCPRegistry()
        result = await registry.disconnect("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_all(self):
        """Test connecting to all servers."""

        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="s1", command=["echo"]))
        registry.register_server(MCPServerConfig(name="s2", command=["echo"], auto_connect=False))
        registry.register_server(MCPServerConfig(name="s3", command=["echo"]))

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.tools = []
        mock_client.resources = []

        with patch("victor.integrations.mcp.registry.MCPClient", return_value=mock_client):
            results = await registry.connect_all()
            # Only s1 and s3 have auto_connect=True
            assert "s1" in results
            assert "s3" in results
            assert "s2" not in results

    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        """Test disconnecting from all servers."""

        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="s1", command=["echo"]))
        registry.register_server(MCPServerConfig(name="s2", command=["echo"]))

        # Set up connected state
        for name in ["s1", "s2"]:
            registry._servers[name].client = MagicMock()
            registry._servers[name].status = ServerStatus.CONNECTED

        await registry.disconnect_all()

        assert registry._servers["s1"].status == ServerStatus.DISCONNECTED
        assert registry._servers["s2"].status == ServerStatus.DISCONNECTED


class TestMCPRegistryHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_unregistered(self):
        """Test health check on unregistered server."""
        registry = MCPRegistry()
        result = await registry.health_check("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self):
        """Test health check on disconnected server."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="test", command=["echo"]))
        result = await registry.health_check("test")
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""

        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        mock_client = MagicMock()
        mock_client.ping = AsyncMock(return_value=True)
        registry._servers["test"].client = mock_client
        registry._servers["test"].status = ServerStatus.CONNECTED

        result = await registry.health_check("test")

        assert result is True
        assert registry._servers["test"].consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""

        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        mock_client = MagicMock()
        mock_client.ping = AsyncMock(return_value=False)
        registry._servers["test"].client = mock_client
        registry._servers["test"].status = ServerStatus.CONNECTED

        result = await registry.health_check("test")

        assert result is False
        assert registry._servers["test"].status == ServerStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_check_exception(self):
        """Test health check with exception."""

        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        mock_client = MagicMock()
        mock_client.ping = AsyncMock(side_effect=Exception("Network error"))
        registry._servers["test"].client = mock_client
        registry._servers["test"].status = ServerStatus.CONNECTED

        result = await registry.health_check("test")

        assert result is False
        assert registry._servers["test"].status == ServerStatus.UNHEALTHY


class TestMCPRegistryReconnect:
    """Tests for reconnection functionality."""

    @pytest.mark.asyncio
    async def test_try_reconnect_unregistered(self):
        """Test reconnecting unregistered server."""
        registry = MCPRegistry()
        result = await registry._try_reconnect("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_try_reconnect_max_retries(self):
        """Test reconnect gives up after max retries."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"], max_retries=3)
        registry.register_server(config)
        registry._servers["test"].consecutive_failures = 3

        result = await registry._try_reconnect("test")

        assert result is False
        assert registry._servers["test"].status == ServerStatus.FAILED


class TestMCPRegistryStartStop:
    """Tests for start/stop functionality."""

    @pytest.mark.asyncio
    async def test_start(self):
        """Test starting the registry."""

        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="test", command=["echo"]))

        mock_client = MagicMock()
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.tools = []
        mock_client.resources = []

        with patch("victor.integrations.mcp.registry.MCPClient", return_value=mock_client):
            await registry.start()
            assert registry._running is True
            # Clean up
            await registry.stop()

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stopping the registry."""
        registry = MCPRegistry()
        registry._running = True

        await registry.stop()

        assert registry._running is False


class TestMCPRegistryCallTool:
    """Tests for call_tool functionality."""

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self):
        """Test calling a tool that doesn't exist."""
        registry = MCPRegistry()
        result = await registry.call_tool("nonexistent_tool")

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test successful tool call."""
        from victor.integrations.mcp.protocol import MCPToolCallResult

        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(
            return_value=MCPToolCallResult(
                tool_name="my_tool",
                success=True,
                result="Done",
            )
        )
        registry._servers["test"].client = mock_client
        registry._servers["test"].status = ServerStatus.CONNECTED
        registry._tool_to_server["my_tool"] = "test"

        result = await registry.call_tool("my_tool", arg1="value")

        assert result.success is True
        assert result.result == "Done"

    def test_custom_init(self):
        """Test custom initialization."""
        registry = MCPRegistry(
            health_check_enabled=False,
            default_health_interval=60,
        )

        assert registry._health_check_enabled is False
        assert registry._default_health_interval == 60


class TestMCPRegistryRegister:
    """Tests for MCPRegistry.register_server method."""

    def test_register_server(self):
        """Test registering a server."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])

        registry.register_server(config)

        assert "test" in registry._servers
        assert registry._servers["test"].config == config
        assert registry._servers["test"].status == ServerStatus.DISCONNECTED

    def test_register_duplicate_server(self):
        """Test registering duplicate server updates config."""
        registry = MCPRegistry()
        config1 = MCPServerConfig(name="test", command=["echo"])
        config2 = MCPServerConfig(name="test", command=["cat"])

        registry.register_server(config1)
        registry.register_server(config2)

        # Should have updated config
        assert registry._servers["test"].config.command == ["cat"]


class TestMCPRegistryUnregister:
    """Tests for MCPRegistry.unregister_server method."""

    def test_unregister_server(self):
        """Test unregistering a server."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        result = registry.unregister_server("test")

        assert result is True
        assert "test" not in registry._servers

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent server."""
        registry = MCPRegistry()

        result = registry.unregister_server("nonexistent")

        assert result is False


class TestMCPRegistryServerAccess:
    """Tests for MCPRegistry server access methods."""

    def test_get_server_entry_directly(self):
        """Test getting a server entry from internal dict."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        entry = registry._servers.get("test")

        assert entry is not None
        assert entry.config.name == "test"

    def test_get_nonexistent_server_entry(self):
        """Test getting non-existent server returns None."""
        registry = MCPRegistry()

        entry = registry._servers.get("nonexistent")

        assert entry is None

    def test_list_servers(self):
        """Test listing all server names."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["echo"]))
        registry.register_server(MCPServerConfig(name="server2", command=["cat"]))

        servers = registry.list_servers()

        assert len(servers) == 2
        assert "server1" in servers
        assert "server2" in servers

    def test_list_servers_empty(self):
        """Test listing servers when empty."""
        registry = MCPRegistry()

        servers = registry.list_servers()

        assert servers == []


class TestMCPRegistryServerStatus:
    """Tests for MCPRegistry server status methods."""

    def test_get_server_status(self):
        """Test getting server status dict."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        status = registry.get_server_status("test")

        assert status is not None
        assert status["name"] == "test"
        assert status["status"] == "DISCONNECTED"

    def test_get_status_nonexistent(self):
        """Test getting status of non-existent server."""
        registry = MCPRegistry()

        status = registry.get_server_status("nonexistent")

        assert status is None

    def test_get_registry_status(self):
        """Test getting overall registry status."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["echo"]))
        registry.register_server(MCPServerConfig(name="server2", command=["cat"]))

        # Manually set one as connected
        registry._servers["server1"].status = ServerStatus.CONNECTED

        status = registry.get_registry_status()

        assert status["total_servers"] == 2
        assert status["connected_servers"] == 1
        assert "server1" in status["servers"]
        assert "server2" in status["servers"]


class TestMCPRegistryFiltering:
    """Tests for MCPRegistry filtering methods."""

    def test_get_tools_by_tag(self):
        """Test getting tools by server tag."""
        registry = MCPRegistry()
        registry.register_server(
            MCPServerConfig(name="server1", command=["echo"], tags=["file", "local"])
        )
        registry.register_server(MCPServerConfig(name="server2", command=["cat"], tags=["network"]))

        # Without connected servers, tools will be empty
        tools = registry.get_tools_by_tag("file")
        assert tools == []

    def test_get_tools_by_tag_no_match(self):
        """Test getting tools by non-existent tag."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["echo"], tags=["local"]))

        tools = registry.get_tools_by_tag("remote")

        assert tools == []

    def test_get_tools_by_server(self):
        """Test getting tools from a specific server."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["echo"]))

        # Without connected servers, tools_cache is empty
        tools = registry.get_tools_by_server("server1")
        assert tools == []

    def test_get_tools_by_nonexistent_server(self):
        """Test getting tools from non-existent server."""
        registry = MCPRegistry()

        tools = registry.get_tools_by_server("nonexistent")
        assert tools == []


class TestMCPRegistryEvents:
    """Tests for MCPRegistry event callbacks."""

    def test_on_event_callback(self):
        """Test registering event callback."""
        registry = MCPRegistry()
        events = []

        def callback(event, server, data):
            events.append((event, server, data))

        registry.on_event(callback)

        assert callback in registry._event_callbacks

    def test_emit_event(self):
        """Test emitting events."""
        registry = MCPRegistry()
        events = []

        def callback(event, server, data):
            events.append((event, server, data))

        registry.on_event(callback)
        registry._emit_event("test_event", "server1", {"key": "value"})

        assert len(events) == 1
        assert events[0] == ("test_event", "server1", {"key": "value"})

    def test_emit_event_callback_error_handled(self):
        """Test that callback errors are handled gracefully."""
        registry = MCPRegistry()

        def bad_callback(event, server, data):
            raise ValueError("Callback error")

        registry.on_event(bad_callback)

        # Should not raise despite callback error
        registry._emit_event("test_event", "server1", {})


class TestMCPRegistryToolMapping:
    """Tests for MCPRegistry tool-to-server mapping."""

    def test_register_tool_mapping(self):
        """Test registering tool-to-server mapping."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="server1", command=["echo"])
        registry.register_server(config)

        registry._tool_to_server["read_file"] = "server1"

        assert registry._tool_to_server["read_file"] == "server1"

    def test_get_tool_from_mapping(self):
        """Test getting tool info through get_tool method."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="server1", command=["echo"])
        registry.register_server(config)

        # get_tool returns None when tool not in tool_to_server
        tool = registry.get_tool("read_file")
        assert tool is None

    def test_tool_to_server_mapping(self):
        """Test tool_to_server mapping directly."""
        registry = MCPRegistry()
        registry._tool_to_server["test_tool"] = "server1"

        assert registry._tool_to_server.get("test_tool") == "server1"
        assert registry._tool_to_server.get("unknown") is None

    def test_get_all_tools_empty(self):
        """Test getting all tools when none connected."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["echo"]))

        tools = registry.get_all_tools()
        assert tools == []

    def test_get_all_resources_empty(self):
        """Test getting all resources when none connected."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["echo"]))

        resources = registry.get_all_resources()
        assert resources == []


class TestMCPRegistrySummary:
    """Tests for MCPRegistry summary/status methods."""

    def test_get_registry_status_full(self):
        """Test getting full registry status."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["echo"], enabled=True))
        registry.register_server(MCPServerConfig(name="server2", command=["cat"], enabled=False))
        registry._servers["server1"].status = ServerStatus.CONNECTED

        status = registry.get_registry_status()

        assert status["total_servers"] == 2
        assert status["connected_servers"] == 1
        assert status["total_tools"] == 0  # No tools cached
        assert status["total_resources"] == 0
        assert status["health_monitoring"] is True
        assert status["running"] is False

    def test_get_registry_status_empty(self):
        """Test getting status when empty."""
        registry = MCPRegistry()

        status = registry.get_registry_status()

        assert status["total_servers"] == 0
        assert status["connected_servers"] == 0

    def test_reset_server(self):
        """Test resetting a failed server."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        # Simulate failure
        registry._servers["test"].status = ServerStatus.FAILED
        registry._servers["test"].consecutive_failures = 5
        registry._servers["test"].error_message = "Connection refused"

        result = registry.reset_server("test")

        assert result is True
        assert registry._servers["test"].status == ServerStatus.DISCONNECTED
        assert registry._servers["test"].consecutive_failures == 0
        assert registry._servers["test"].error_message is None

    def test_reset_nonexistent_server(self):
        """Test resetting non-existent server."""
        registry = MCPRegistry()

        result = registry.reset_server("nonexistent")

        assert result is False
