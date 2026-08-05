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

"""Unit tests for MCPConnector component.

Tests the SRP-compliant MCP server connection functionality.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from victor.agent.mcp_connector import (
    MCPConnector,
    MCPConnectorConfig,
    MCPConnectResult,
    MCPServerInfo,
)


class TestMCPConnectorConfig:
    """Tests for MCPConnectorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MCPConnectorConfig()

        assert config.enabled is False
        assert config.auto_discover is True
        assert config.connection_timeout == 30.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MCPConnectorConfig(
            enabled=True,
            auto_discover=False,
            connection_timeout=60.0,
        )

        assert config.enabled is True
        assert config.auto_discover is False
        assert config.connection_timeout == 60.0


class TestMCPConnectResult:
    """Tests for MCPConnectResult dataclass."""

    def test_default_values(self):
        """Test default result values."""
        result = MCPConnectResult()

        assert result.servers_discovered == 0
        assert result.servers_connected == 0
        assert result.tools_registered == 0
        assert result.errors == []

    def test_custom_values(self):
        """Test custom result values."""
        result = MCPConnectResult(
            servers_discovered=3,
            servers_connected=2,
            tools_registered=10,
            errors=["connection failed"],
        )

        assert result.servers_discovered == 3
        assert result.servers_connected == 2
        assert result.tools_registered == 10
        assert result.errors == ["connection failed"]


class TestMCPServerInfo:
    """Tests for MCPServerInfo dataclass."""

    def test_default_values(self):
        """Test default MCPServerInfo values."""
        info = MCPServerInfo(name="test_server")

        assert info.name == "test_server"
        assert info.description == ""
        assert info.connected is False

    def test_custom_values(self):
        """Test custom MCPServerInfo values."""
        info = MCPServerInfo(
            name="my_server",
            description="My MCP server",
            connected=True,
        )

        assert info.name == "my_server"
        assert info.description == "My MCP server"
        assert info.connected is True


class TestMCPConnectorInit:
    """Tests for MCPConnector initialization."""

    def test_initialization_with_defaults(self):
        """Test initialization with default config."""
        registry = MagicMock()
        settings = MagicMock()

        connector = MCPConnector(registry=registry, settings=settings)

        assert connector._registry is registry
        assert connector._settings is settings
        assert connector._config.enabled is False
        assert connector.is_connected is False
        assert connector.mcp_registry is None

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config."""
        registry = MagicMock()
        settings = MagicMock()
        config = MCPConnectorConfig(enabled=True)

        connector = MCPConnector(registry=registry, settings=settings, config=config)

        assert connector._config.enabled is True

    def test_initialization_with_task_callback(self):
        """Test initialization with task callback."""
        registry = MagicMock()
        settings = MagicMock()
        callback = MagicMock()

        connector = MCPConnector(registry=registry, settings=settings, task_callback=callback)

        assert connector._task_callback is callback

    def test_set_task_callback(self):
        """Test setting task callback after initialization."""
        registry = MagicMock()
        settings = MagicMock()
        callback = MagicMock()

        connector = MCPConnector(registry=registry, settings=settings)
        connector.set_task_callback(callback)

        assert connector._task_callback is callback


class TestMCPConnectorConnect:
    """Tests for MCPConnector.connect() method."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock tool registry."""
        return MagicMock()

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.use_mcp_tools = False
        settings.mcp_command = None
        return settings

    @pytest.mark.asyncio
    async def test_connect_disabled_returns_empty_result(self, mock_registry, mock_settings):
        """Test that connect() returns empty result when disabled."""
        config = MCPConnectorConfig(enabled=False)
        connector = MCPConnector(registry=mock_registry, settings=mock_settings, config=config)

        result = await connector.connect()

        assert result.servers_discovered == 0
        assert result.tools_registered == 0

    @pytest.mark.asyncio
    async def test_connect_with_use_mcp_tools_setting(self, mock_registry, mock_settings):
        """Test that connect() works when use_mcp_tools setting is True."""
        mock_settings.use_mcp_tools = True
        connector = MCPConnector(registry=mock_registry, settings=mock_settings)

        # Mock MCPRegistry
        mock_mcp_registry = MagicMock()
        mock_mcp_registry.list_servers.return_value = []

        with patch(
            "victor.integrations.mcp.registry.MCPRegistry.discover_servers",
            return_value=mock_mcp_registry,
        ):
            with patch(
                "victor.tools.mcp_bridge_tool.get_mcp_tool_definitions",
                return_value=[{"name": "mcp_tool"}],
            ):
                result = await connector.connect()

        assert result.tools_registered == 1
        assert connector.is_connected is True

    @pytest.mark.asyncio
    async def test_connect_registers_mcp_command_from_settings(self, mock_registry, mock_settings):
        """Test that connect() registers MCP command from settings."""
        mock_settings.use_mcp_tools = True
        mock_settings.mcp_command = "npx mcp-server"

        connector = MCPConnector(registry=mock_registry, settings=mock_settings)

        mock_mcp_registry = MagicMock()
        mock_mcp_registry.list_servers.return_value = []

        with patch(
            "victor.integrations.mcp.registry.MCPRegistry.discover_servers",
            return_value=mock_mcp_registry,
        ):
            with patch(
                "victor.tools.mcp_bridge_tool.get_mcp_tool_definitions",
                return_value=[],
            ):
                await connector.connect()

        mock_mcp_registry.register_server.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_falls_back_to_legacy_client(self, mock_registry, mock_settings):
        """Test that connect() falls back to legacy client when MCPRegistry unavailable."""
        mock_settings.use_mcp_tools = True
        mock_settings.mcp_command = "npx mcp-server"

        connector = MCPConnector(registry=mock_registry, settings=mock_settings)

        with patch(
            "victor.integrations.mcp.registry.MCPRegistry.discover_servers",
            side_effect=ImportError("MCPRegistry not available"),
        ):
            with patch("victor.integrations.mcp.client.MCPClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client.connect = AsyncMock()
                mock_client_class.return_value = mock_client

                with patch(
                    "victor.tools.mcp_bridge_tool.get_mcp_tool_definitions",
                    return_value=[],
                ):
                    await connector.connect()

        # Should have tried to create legacy client
        mock_client_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_registers_bridge_tools(self, mock_registry, mock_settings):
        """Test that connect() registers MCP bridge tools."""
        mock_settings.use_mcp_tools = True
        connector = MCPConnector(registry=mock_registry, settings=mock_settings)

        mock_mcp_registry = MagicMock()
        mock_mcp_registry.list_servers.return_value = []

        mock_tools = [
            {"name": "mcp_tool1"},
            {"name": "mcp_tool2"},
        ]

        with patch(
            "victor.integrations.mcp.registry.MCPRegistry.discover_servers",
            return_value=mock_mcp_registry,
        ):
            with patch(
                "victor.tools.mcp_bridge_tool.get_mcp_tool_definitions",
                return_value=mock_tools,
            ):
                result = await connector.connect()

        assert result.tools_registered == 2
        assert mock_registry.register_dict.call_count == 2


class TestMCPConnectorGetServerInfo:
    """Tests for MCPConnector.get_server_info() method."""

    def test_get_server_info_empty_when_no_registry(self):
        """Test get_server_info returns empty when no registry."""
        registry = MagicMock()
        settings = MagicMock()

        connector = MCPConnector(registry=registry, settings=settings)

        result = connector.get_server_info()

        assert result == []

    def test_get_server_info_returns_server_details(self):
        """Test get_server_info returns correct server details."""
        registry = MagicMock()
        settings = MagicMock()

        connector = MCPConnector(registry=registry, settings=settings)

        # Mock MCP registry with servers
        mock_server1 = MagicMock()
        mock_server1.name = "server1"
        mock_server1.description = "Server 1"

        mock_server2 = MagicMock()
        mock_server2.name = "server2"
        mock_server2.description = "Server 2"

        connector._mcp_registry = MagicMock()
        connector._mcp_registry.list_servers.return_value = [mock_server1, mock_server2]
        connector._mcp_registry._connected_servers = {"server1": MagicMock()}

        result = connector.get_server_info()

        assert len(result) == 2
        server1_info = next(s for s in result if s.name == "server1")
        server2_info = next(s for s in result if s.name == "server2")

        assert server1_info.connected is True
        assert server2_info.connected is False


class TestMCPConnectorGetSummary:
    """Tests for MCPConnector.get_summary() method."""

    def test_get_summary_empty_when_no_servers(self):
        """Test get_summary returns empty when no servers."""
        registry = MagicMock()
        settings = MagicMock()

        connector = MCPConnector(registry=registry, settings=settings)

        result = connector.get_summary()

        assert result == {"servers": [], "connected": 0, "total": 0}

    def test_get_summary_returns_correct_structure(self):
        """Test get_summary returns correct structure."""
        registry = MagicMock()
        settings = MagicMock()

        connector = MCPConnector(registry=registry, settings=settings)

        # Mock MCP registry
        mock_server = MagicMock()
        mock_server.name = "test_server"
        mock_server.description = "Test server"

        connector._mcp_registry = MagicMock()
        connector._mcp_registry.list_servers.return_value = [mock_server]
        connector._mcp_registry._connected_servers = {"test_server": MagicMock()}

        result = connector.get_summary()

        assert result["total"] == 1
        assert result["connected"] == 1
        assert len(result["servers"]) == 1
        assert result["servers"][0]["name"] == "test_server"
        assert result["servers"][0]["connected"] is True


class TestMCPConnectorShutdown:
    """Tests for MCPConnector.shutdown() method."""

    @pytest.mark.asyncio
    async def test_shutdown_cancels_pending_tasks(self):
        """Test that shutdown() cancels pending tasks."""
        registry = MagicMock()
        settings = MagicMock()

        connector = MCPConnector(registry=registry, settings=settings)

        # Add some mock tasks
        mock_task1 = MagicMock()
        mock_task1.done.return_value = False
        mock_task2 = MagicMock()
        mock_task2.done.return_value = True  # Already done

        connector._pending_tasks = [mock_task1, mock_task2]

        await connector.shutdown()

        mock_task1.cancel.assert_called_once()
        mock_task2.cancel.assert_not_called()  # Already done
        assert connector.is_connected is False

    @pytest.mark.asyncio
    async def test_shutdown_shuts_down_mcp_registry(self):
        """Test that shutdown() shuts down MCP registry."""
        registry = MagicMock()
        settings = MagicMock()

        connector = MCPConnector(registry=registry, settings=settings)

        mock_mcp_registry = MagicMock()
        mock_mcp_registry.shutdown = AsyncMock()
        connector._mcp_registry = mock_mcp_registry

        await connector.shutdown()

        mock_mcp_registry.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_errors_gracefully(self):
        """Test that shutdown() handles errors gracefully."""
        registry = MagicMock()
        settings = MagicMock()

        connector = MCPConnector(registry=registry, settings=settings)

        mock_mcp_registry = MagicMock()
        mock_mcp_registry.shutdown = AsyncMock(side_effect=Exception("Shutdown error"))
        connector._mcp_registry = mock_mcp_registry

        # Should not raise
        await connector.shutdown()

        assert connector.is_connected is False
