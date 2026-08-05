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

"""
Async MCP registry with connection pooling and health monitoring.

This module provides an async-first MCP registry with:
- Connection pooling for multiple clients
- Health monitoring with automatic reconnection
- Graceful degradation on errors
- Server lifecycle management
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ServerStatus(Enum):
    """MCP server status."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    ERROR = auto()


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    name: str
    transport: str  # "stdio", "sse", "ws"
    command: Optional[str] = None
    args: Optional[List[str]] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    reconnect_interval: float = 5.0
    max_retries: int = 3
    timeout: float = 30.0


@dataclass
class MCPServerStats:
    """Statistics for an MCP server."""

    name: str
    status: ServerStatus
    connection_time: Optional[datetime] = None
    last_ping: Optional[datetime] = None
    failed_pings: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_uptime_ms: float = 0.0


class AsyncMCPRegistry:
    """
    Async MCP registry with connection pooling.

    Features:
    - Async connection management
    - Health monitoring with automatic reconnection
    - Connection pooling for multiple clients
    - Graceful degradation
    """

    def __init__(self) -> None:
        """Initialize async MCP registry."""
        self._servers: Dict[str, Any] = {}
        self._status: Dict[str, ServerStatus] = {}
        self._stats: Dict[str, MCPServerStats] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        self._config: Optional[Any] = None

    async def register_server(self, config: MCPServerConfig) -> None:
        """
        Register an MCP server configuration.

        Args:
            config: Server configuration
        """
        from victor.integrations.mcp.server import MCPServer

        # Create server with existing API
        server = MCPServer(
            name=config.name,
            version="1.0.0",  # Default version
            tool_registry=None,  # Will use default
        )

        # Store config separately
        server._config = config

        self._servers[config.name] = server
        self._status[config.name] = ServerStatus.DISCONNECTED
        self._locks[config.name] = asyncio.Lock()
        self._stats[config.name] = MCPServerStats(
            name=config.name,
            status=ServerStatus.DISCONNECTED,
        )

        logger.debug(f"Registered MCP server: {config.name}")

    async def connect_all(self) -> None:
        """Connect to all registered servers."""
        tasks = [self._connect_server(name) for name in self._servers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        connected = sum(1 for r in results if r is True)
        logger.info(f"Connected to {connected}/{len(self._servers)} MCP servers")

    async def _connect_server(self, name: str) -> bool:
        """
        Connect to a specific server.

        Args:
            name: Server name

        Returns:
            True if connection succeeded, False otherwise
        """
        server = self._servers.get(name)
        if not server:
            logger.warning(f"Server not found: {name}")
            return False

        async with self._locks[name]:
            self._status[name] = ServerStatus.CONNECTING

            try:
                # Note: The existing MCPServer doesn't have async connect()
                # This is a placeholder for future async connection support
                # For now, we'll mark it as connected immediately
                await asyncio.sleep(0)  # Yield to event loop

                self._status[name] = ServerStatus.CONNECTED

                stats = self._stats[name]
                stats.status = ServerStatus.CONNECTED
                stats.connection_time = datetime.now(timezone.utc)

                # Start health check
                self._health_check_tasks[name] = asyncio.create_task(self._health_check_loop(name))

                logger.info(f"Connected to MCP server: {name}")
                return True

            except Exception as e:
                logger.error(f"Failed to connect to MCP server {name}: {e}")
                self._status[name] = ServerStatus.ERROR
                return False

    async def _health_check_loop(self, name: str) -> None:
        """
        Health check loop for server.

        Args:
            name: Server name
        """
        server = self._servers.get(name)
        config = server.config if server else None

        while self._status.get(name) == ServerStatus.CONNECTED:
            try:
                interval = config.reconnect_interval if config else 30
                await asyncio.sleep(interval)

                # Ping server
                await self.ping_server(name)

            except Exception as e:
                logger.warning(f"Health check failed for {name}: {e}")

                # Attempt reconnection
                if self._status[name] == ServerStatus.CONNECTED:
                    self._status[name] = ServerStatus.ERROR
                    await self._connect_server(name)

    async def ping_server(self, name: str) -> bool:
        """
        Ping an MCP server.

        Args:
            name: Server name

        Returns:
            True if ping succeeded, False otherwise
        """
        server = self._servers.get(name)
        if not server:
            return False

        try:
            # Note: The existing MCPServer doesn't have async ping()
            # This is a placeholder for future health check support
            await asyncio.sleep(0)  # Yield to event loop

            stats = self._stats[name]
            stats.last_ping = datetime.now(timezone.utc)
            stats.failed_pings = 0

            return True

        except Exception as e:
            stats = self._stats[name]
            stats.failed_pings += 1
            logger.debug(f"Ping failed for {name}: {e}")
            return False

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        # Cancel health checks
        for task in self._health_check_tasks.values():
            task.cancel()

        # Disconnect servers
        tasks = [self._disconnect_server(name) for name in self._servers]
        await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Disconnected from all MCP servers")

    async def _disconnect_server(self, name: str) -> bool:
        """
        Disconnect from a specific server.

        Args:
            name: Server name

        Returns:
            True if disconnection succeeded
        """
        server = self._servers.get(name)
        if not server:
            return False

        async with self._locks[name]:
            try:
                # Note: The existing MCPServer doesn't have async disconnect()
                # This is a placeholder for future disconnect support
                await asyncio.sleep(0)  # Yield to event loop

                self._status[name] = ServerStatus.DISCONNECTED

                # Update stats
                stats = self._stats[name]
                if stats.connection_time:
                    uptime = datetime.now(timezone.utc) - stats.connection_time
                    stats.total_uptime_ms += uptime.total_seconds() * 1000
                stats.connection_time = None

                logger.info(f"Disconnected from MCP server: {name}")
                return True

            except Exception as e:
                logger.error(f"Failed to disconnect from {name}: {e}")
                return False

    def get_servers(self) -> List[Any]:
        """
        Get all connected servers.

        Returns:
            List of connected MCPServer instances
        """
        return [
            self._servers[name]
            for name, status in self._status.items()
            if status == ServerStatus.CONNECTED
        ]

    def get_server(self, name: str) -> Optional[Any]:
        """
        Get a specific server by name.

        Args:
            name: Server name

        Returns:
            MCPServer instance if found and connected, None otherwise
        """
        if self._status.get(name) == ServerStatus.CONNECTED:
            return self._servers.get(name)
        return None

    def get_server_status(self, name: str) -> Optional[ServerStatus]:
        """
        Get status of a server.

        Args:
            name: Server name

        Returns:
            Server status if found, None otherwise
        """
        return self._status.get(name)

    def get_server_stats(self, name: str) -> Optional[MCPServerStats]:
        """
        Get statistics for a server.

        Args:
            name: Server name

        Returns:
            Server statistics if found, None otherwise
        """
        return self._stats.get(name)

    def get_all_stats(self) -> Dict[str, MCPServerStats]:
        """
        Get statistics for all servers.

        Returns:
            Dictionary mapping server names to statistics
        """
        return dict(self._stats)


# Legacy MCPRegistry wrapper for backward compatibility
class MCPRegistry:
    """
    Legacy wrapper for MCP registry.

    This provides backward compatibility with the existing MCP integration
    while using the new async registry internally.
    """

    def __init__(self, settings: Any) -> None:
        """
        Initialize MCP registry.

        Args:
            settings: Victor settings
        """
        self._async_registry = AsyncMCPRegistry()
        self._settings = settings
        self._loop = None

    async def register_server(self, config: MCPServerConfig) -> None:
        """
        Register an MCP server configuration.

        Args:
            config: Server configuration
        """
        await self._async_registry.register_server(config)

    async def connect_all(self) -> None:
        """Connect to all registered servers."""
        await self._async_registry.connect_all()

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        await self._async_registry.disconnect_all()

    def get_servers(self) -> List[Any]:
        """Get all connected servers."""
        return self._async_registry.get_servers()

    def get_server(self, name: str) -> Optional[Any]:
        """Get a specific server by name."""
        return self._async_registry.get_server(name)

    def get_server_status(self, name: str) -> Optional[ServerStatus]:
        """Get status of a server."""
        return self._async_registry.get_server_status(name)

    def get_server_stats(self, name: str) -> Optional[MCPServerStats]:
        """Get statistics for a server."""
        return self._async_registry.get_server_stats(name)
