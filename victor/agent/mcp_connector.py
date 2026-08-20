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

"""MCP connector for Model Context Protocol integration.

This module handles MCP server discovery, connection, and tool registration,
extracted from ToolRegistrar as part of SRP compliance refactoring.

Single Responsibility: Manage MCP server connections and tool discovery.

Design Pattern: Connector Pattern
- Discovers MCP servers from standard locations
- Manages server connections lifecycle
- Registers MCP tools with ToolRegistry

Usage:
    from victor.agent.mcp_connector import MCPConnector

    connector = MCPConnector(
        registry=tool_registry,
        settings=settings,
        config=MCPConnectorConfig(enabled=True),
    )
    await connector.connect()
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from victor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class MCPConnectorConfig:
    """Configuration for MCP connector.

    Attributes:
        enabled: Whether MCP integration is enabled
        auto_discover: Auto-discover MCP servers from standard locations
        connection_timeout: Timeout for server connections in seconds
    """

    enabled: bool = False
    auto_discover: bool = True
    connection_timeout: float = 30.0

    @classmethod
    def from_settings(cls, settings: Any) -> "MCPConnectorConfig":
        """Create config from Victor Settings object.

        Reads from settings.mcp (McpSettings) if available,
        falls back to defaults otherwise.

        Args:
            settings: Victor Settings instance.

        Returns:
            Populated MCPConnectorConfig.
        """
        mcp = getattr(settings, "mcp", None)
        if mcp is None:
            return cls()
        return cls(
            enabled=getattr(mcp, "mcp_enabled", False),
            auto_discover=getattr(mcp, "mcp_auto_discover", True),
            connection_timeout=getattr(mcp, "mcp_tool_timeout_ms", 30000) / 1000.0,
        )


@dataclass
class MCPConnectResult:
    """Result of MCP connection.

    Attributes:
        servers_discovered: Number of MCP servers discovered
        servers_connected: Number of servers successfully connected
        tools_registered: Number of MCP tools registered
        errors: List of errors encountered during connection
    """

    servers_discovered: int = 0
    servers_connected: int = 0
    tools_registered: int = 0
    tools_flattened: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class MCPServerInfo:
    """Information about an MCP server.

    Attributes:
        name: Server name
        description: Server description
        connected: Whether server is currently connected
    """

    name: str
    description: str = ""
    connected: bool = False


class MCPConnector:
    """Manages MCP server connections and tool discovery.

    Single Responsibility: MCP integration lifecycle management.

    This class handles:
    - Discovering MCP servers from standard locations
    - Managing server connection lifecycle
    - Registering MCP tools with ToolRegistry
    - Graceful shutdown of connections

    Extracted from ToolRegistrar for SRP compliance.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        settings: Any,
        config: Optional[MCPConnectorConfig] = None,
        task_callback: Optional[Callable[[Any, str], asyncio.Task]] = None,
    ):
        """Initialize the MCP connector.

        Args:
            registry: Tool registry to register MCP tools with
            settings: Application settings for MCP configuration
            config: Optional MCP connector configuration
            task_callback: Optional callback for creating background tasks
        """
        self._registry = registry
        self._settings = settings
        self._config = config or MCPConnectorConfig()
        self._task_callback = task_callback

        self._mcp_registry: Optional[Any] = None
        self._pending_tasks: List[asyncio.Task] = []
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if MCP connections have been established."""
        return self._connected

    @property
    def mcp_registry(self) -> Optional[Any]:
        """Get the underlying MCP registry."""
        return self._mcp_registry

    def set_task_callback(self, callback: Callable[[Any, str], asyncio.Task]) -> None:
        """Set callback for creating background tasks.

        Args:
            callback: Function that takes (coroutine, name) and returns Task
        """
        self._task_callback = callback

    def _create_task(self, coro: Any, name: str) -> Optional[asyncio.Task]:
        """Create a background task using callback or directly."""
        if self._task_callback:
            return self._task_callback(coro, name)
        else:
            task = asyncio.create_task(coro)
            self._pending_tasks.append(task)
            return task

    async def connect(
        self,
        vertical_mcp_servers: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> MCPConnectResult:
        """Connect to MCP servers and register tools.

        This method:
        1. Discovers MCP servers from standard locations
        2. Registers servers declared by the active vertical (McpProvider)
        3. Connects to all discovered servers
        4. Flattens MCP tools as first-class BaseTool instances

        Args:
            vertical_mcp_servers: Optional MCP servers declared by the vertical
                via McpProvider.get_mcp_servers(). Merged with auto-discovered
                servers.

        Returns:
            MCPConnectResult with connection statistics
        """
        result = MCPConnectResult()

        if not self._config.enabled and not getattr(self._settings, "use_mcp_tools", False):
            # Still connect if vertical declares MCP servers
            if not vertical_mcp_servers:
                logger.debug("MCP integration disabled")
                return result

        mcp_command = getattr(self._settings, "mcp_command", None)

        # Try MCPRegistry with auto-discovery first
        try:
            from victor.integrations.mcp.registry import MCPRegistry, MCPServerConfig

            # Auto-discover MCP servers from standard locations
            if self._config.auto_discover:
                self._mcp_registry = MCPRegistry.discover_servers()
            else:
                self._mcp_registry = MCPRegistry()

            # Register servers declared by the active vertical
            if vertical_mcp_servers:
                for name, config in vertical_mcp_servers.items():
                    try:
                        command = config.get("command", "")
                        args = config.get("args", [])
                        cmd = [command] + args if isinstance(command, str) else list(command)
                        self._mcp_registry.register_server(
                            MCPServerConfig(
                                name=name,
                                command=cmd,
                                description=config.get("description", f"Vertical MCP: {name}"),
                                auto_connect=True,
                            )
                        )
                    except Exception as e:
                        logger.warning("Failed to register vertical MCP server %s: %s", name, e)

            # Also register command from settings if specified
            if mcp_command:
                cmd_parts = mcp_command.split()
                self._mcp_registry.register_server(
                    MCPServerConfig(
                        name="settings_mcp",
                        command=cmd_parts,
                        description="MCP server from settings",
                        auto_connect=True,
                    )
                )

            result.servers_discovered = len(self._mcp_registry.list_servers())

            # Start registry and connect to servers in background
            if result.servers_discovered > 0:
                logger.info(f"MCPConnector: discovered {result.servers_discovered} server(s)")
                self._create_task(self._start_registry(), "mcp_registry_start")

        except ImportError:
            logger.debug("MCPRegistry not available, using legacy client")
            self._setup_legacy_client(mcp_command)

        # Register MCP tool definitions
        result.tools_registered = self._register_bridge_tools()

        self._connected = True
        return result

    async def _start_registry(self) -> int:
        """Start MCP registry, connect to servers, and flatten tools.

        After connecting, projects MCP tools as first-class BaseTool
        instances into the ToolRegistry via MCPToolProjector (Adapter Pattern).

        Returns:
            Number of servers connected
        """
        if not self._mcp_registry:
            return 0

        try:
            await self._mcp_registry.start()
            results = await self._mcp_registry.connect_all()
            connected = sum(1 for v in results.values() if v)

            if connected > 0:
                logger.info(f"MCPConnector: connected to {connected} server(s)")
                mcp_tools = self._mcp_registry.get_all_tools()
                if mcp_tools:
                    logger.info(f"MCPConnector: discovered {len(mcp_tools)} MCP tools")

                # Flatten MCP tools as first-class BaseTool instances
                self._flatten_mcp_tools()

            return connected

        except Exception as e:
            logger.warning(f"Failed to start MCP registry: {e}")
            return 0

    def _flatten_mcp_tools(self) -> int:
        """Project MCP tools as first-class tools in the ToolRegistry.

        Uses MCPToolProjector (Adapter Pattern) to create MCPAdapterTool
        instances that wrap each MCP tool as a native BaseTool. The LLM
        sees tool names like 'github_search' instead of 'mcp(name=...)'.

        Returns:
            Number of tools flattened and registered
        """
        if not self._mcp_registry:
            return 0

        try:
            from victor.tools.mcp_adapter_tool import MCPToolProjector
            from victor.tools.batch_registration import BatchRegistrar

            adapter_tools = MCPToolProjector.project(self._mcp_registry)

            # Use batch registration for performance (10-30 tools → single cache invalidation)
            registrar = BatchRegistrar(self._registry)
            result = registrar.register_batch(adapter_tools, fail_fast=False)

            if result.failure_count > 0:
                for tool_name, error_msg in result.failed:
                    logger.debug("Failed to register MCP tool %s: %s", tool_name, error_msg)

            registered = result.success_count
            if registered > 0:
                logger.info(
                    f"MCPConnector: batch registered {registered}/{len(adapter_tools)} MCP tools"
                )

            return registered

        except ImportError:
            logger.debug("MCPToolProjector not available, using bridge tool only")
            return 0
        except Exception as e:
            logger.warning(f"Failed to flatten MCP tools: {e}")
            return 0

    def _setup_legacy_client(self, mcp_command: Optional[str]) -> None:
        """Set up legacy single MCP client (backwards compatibility)."""
        if mcp_command:
            try:
                from victor.integrations.mcp.client import MCPClient

                mcp_client = MCPClient()
                cmd_parts = mcp_command.split()
                self._create_task(mcp_client.connect(cmd_parts), "mcp_legacy_connect")
            except Exception as exc:
                logger.warning(f"Failed to start MCP client: {exc}")

    def _register_bridge_tools(self) -> int:
        """Register MCP bridge tool definitions.

        Returns:
            Number of MCP tools registered
        """
        tools_registered = 0

        try:
            from victor.tools.mcp_bridge_tool import get_mcp_tool_definitions

            for mcp_tool in get_mcp_tool_definitions():
                self._registry.register_dict(mcp_tool)
                tools_registered += 1

        except ImportError:
            logger.debug("MCP bridge tool not available")

        return tools_registered

    def get_server_info(self) -> List[MCPServerInfo]:
        """Get information about MCP servers.

        Returns:
            List of MCPServerInfo for each server
        """
        if not self._mcp_registry:
            return []

        servers = self._mcp_registry.list_servers()
        connected_servers = getattr(self._mcp_registry, "_connected_servers", {})

        return [
            MCPServerInfo(
                name=s.name,
                description=s.description,
                connected=s.name in connected_servers,
            )
            for s in servers
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary information about MCP servers.

        Returns:
            Dictionary with MCP server summary information
        """
        servers = self.get_server_info()
        connected = sum(1 for s in servers if s.connected)

        return {
            "servers": [
                {"name": s.name, "description": s.description, "connected": s.connected}
                for s in servers
            ],
            "connected": connected,
            "total": len(servers),
        }

    async def shutdown(self) -> None:
        """Shutdown MCP connections and cleanup."""
        # Cancel pending tasks
        for task in self._pending_tasks:
            if not task.done():
                task.cancel()

        # Shutdown MCP registry
        if self._mcp_registry:
            try:
                await self._mcp_registry.shutdown()
            except Exception as e:
                logger.debug(f"Error shutting down MCP registry: {e}")

        self._connected = False
        logger.debug("MCPConnector shutdown complete")


__all__ = ["MCPConnector", "MCPConnectorConfig", "MCPConnectResult", "MCPServerInfo"]
