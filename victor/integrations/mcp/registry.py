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

"""MCP Registry for auto-discovery and management of MCP servers.

Provides centralized management of multiple MCP servers including:
- Auto-discovery from configuration
- Health monitoring and reconnection
- Unified tool/resource access across servers
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from victor.integrations.mcp.client import MCPClient
from victor.integrations.mcp.protocol import MCPResource, MCPTool, MCPToolCallResult

logger = logging.getLogger(__name__)


class ServerStatus(Enum):
    """MCP server connection status."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    UNHEALTHY = auto()
    FAILED = auto()


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    name: str = Field(description="Unique server identifier")
    command: List[str] = Field(description="Command to start the server")
    description: str = Field(default="", description="Server description")
    auto_connect: bool = Field(default=True, description="Connect on registry start")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    max_retries: int = Field(default=3, description="Max reconnection attempts")
    retry_delay: int = Field(default=5, description="Delay between retries in seconds")
    enabled: bool = Field(default=True, description="Whether server is enabled")
    tags: List[str] = Field(default_factory=list, description="Server tags for filtering")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")


@dataclass
class ServerEntry:
    """Registry entry for an MCP server."""

    config: MCPServerConfig
    client: Optional[MCPClient] = None
    status: ServerStatus = ServerStatus.DISCONNECTED
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    error_message: Optional[str] = None
    tools_cache: List[MCPTool] = field(default_factory=list)
    resources_cache: List[MCPResource] = field(default_factory=list)


class MCPRegistry:
    """Central registry for managing multiple MCP servers.

    Features:
    - Auto-discovery from configuration files
    - Health monitoring with automatic reconnection
    - Unified access to tools/resources across all servers
    - Server filtering by tags and capabilities

    Example:
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(
            name="filesystem",
            command=["python", "-m", "mcp_filesystem"],
        ))
        await registry.connect_all()

        # Call tools across any server
        result = await registry.call_tool("read_file", path="/etc/hosts")
    """

    def __init__(
        self,
        health_check_enabled: bool = True,
        default_health_interval: int = 30,
    ):
        """Initialize MCP registry.

        Args:
            health_check_enabled: Enable automatic health monitoring
            default_health_interval: Default health check interval in seconds
        """
        self._servers: Dict[str, ServerEntry] = {}
        self._tool_to_server: Dict[str, str] = {}  # tool_name -> server_name
        self._health_check_enabled = health_check_enabled
        self._default_health_interval = default_health_interval
        self._health_task: Optional[asyncio.Task] = None
        self._running = False
        self._event_callbacks: List[Callable[[str, str, Any], None]] = []

    def register_server(self, config: MCPServerConfig) -> None:
        """Register an MCP server configuration.

        Args:
            config: Server configuration
        """
        if config.name in self._servers:
            logger.warning(f"Server '{config.name}' already registered, updating config")

        self._servers[config.name] = ServerEntry(config=config)
        logger.info(f"Registered MCP server: {config.name}")

    def unregister_server(self, name: str) -> bool:
        """Unregister an MCP server.

        Args:
            name: Server name

        Returns:
            True if server was removed
        """
        if name not in self._servers:
            return False

        entry = self._servers[name]
        if entry.client:
            entry.client.disconnect()

        # Remove tool mappings
        tools_to_remove = [t for t, s in self._tool_to_server.items() if s == name]
        for tool in tools_to_remove:
            del self._tool_to_server[tool]

        del self._servers[name]
        logger.info(f"Unregistered MCP server: {name}")
        return True

    async def connect(self, name: str) -> bool:
        """Connect to a specific MCP server.

        Args:
            name: Server name

        Returns:
            True if connection successful
        """
        if name not in self._servers:
            logger.error(f"Server '{name}' not registered")
            return False

        entry = self._servers[name]
        if not entry.config.enabled:
            logger.info(f"Server '{name}' is disabled, skipping connection")
            return False

        entry.status = ServerStatus.CONNECTING
        self._emit_event("connecting", name, None)

        try:
            client = MCPClient(name=f"Victor-{name}")
            success = await client.connect(entry.config.command)

            if success:
                entry.client = client
                entry.status = ServerStatus.CONNECTED
                entry.consecutive_failures = 0
                entry.error_message = None
                entry.last_health_check = time.time()

                # Cache tools and resources
                entry.tools_cache = client.tools
                entry.resources_cache = client.resources

                # Update tool-to-server mapping
                for tool in client.tools:
                    self._tool_to_server[tool.name] = name
                    logger.debug(f"Registered tool '{tool.name}' from server '{name}'")

                logger.info(
                    f"Connected to MCP server '{name}': "
                    f"{len(client.tools)} tools, {len(client.resources)} resources"
                )
                self._emit_event("connected", name, {"tools": len(client.tools)})
                return True
            else:
                entry.status = ServerStatus.FAILED
                entry.error_message = "Connection failed"
                entry.consecutive_failures += 1
                self._emit_event("failed", name, {"error": "Connection failed"})
                return False

        except Exception as e:
            entry.status = ServerStatus.FAILED
            entry.error_message = str(e)
            entry.consecutive_failures += 1
            logger.error(f"Failed to connect to MCP server '{name}': {e}")
            self._emit_event("failed", name, {"error": str(e)})
            return False

    async def disconnect(self, name: str) -> bool:
        """Disconnect from a specific MCP server.

        Args:
            name: Server name

        Returns:
            True if disconnection successful
        """
        if name not in self._servers:
            return False

        entry = self._servers[name]
        if entry.client:
            entry.client.disconnect()
            entry.client = None

        entry.status = ServerStatus.DISCONNECTED
        entry.tools_cache = []
        entry.resources_cache = []

        # Remove tool mappings
        tools_to_remove = [t for t, s in self._tool_to_server.items() if s == name]
        for tool in tools_to_remove:
            del self._tool_to_server[tool]

        logger.info(f"Disconnected from MCP server: {name}")
        self._emit_event("disconnected", name, None)
        return True

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all registered servers with auto_connect=True.

        Returns:
            Dict mapping server name to connection success
        """
        results = {}
        for name, entry in self._servers.items():
            if entry.config.auto_connect and entry.config.enabled:
                results[name] = await self.connect(name)
        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for name in list(self._servers.keys()):
            await self.disconnect(name)

    async def health_check(self, name: str) -> bool:
        """Perform health check on a server.

        Args:
            name: Server name

        Returns:
            True if server is healthy
        """
        if name not in self._servers:
            return False

        entry = self._servers[name]
        if not entry.client or entry.status != ServerStatus.CONNECTED:
            return False

        try:
            healthy = await entry.client.ping()
            entry.last_health_check = time.time()

            if healthy:
                entry.consecutive_failures = 0
                return True
            else:
                entry.status = ServerStatus.UNHEALTHY
                entry.consecutive_failures += 1
                return False

        except Exception as e:
            entry.status = ServerStatus.UNHEALTHY
            entry.consecutive_failures += 1
            entry.error_message = str(e)
            logger.warning(f"Health check failed for '{name}': {e}")
            return False

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while self._running:
            for name, entry in list(self._servers.items()):
                if not entry.config.enabled:
                    continue

                # Check if health check is due
                interval = entry.config.health_check_interval or self._default_health_interval
                if time.time() - entry.last_health_check < interval:
                    continue

                if entry.status == ServerStatus.CONNECTED:
                    healthy = await self.health_check(name)
                    if not healthy:
                        logger.warning(f"Server '{name}' became unhealthy")
                        await self._try_reconnect(name)

                elif entry.status in (ServerStatus.UNHEALTHY, ServerStatus.FAILED):
                    await self._try_reconnect(name)

            await asyncio.sleep(5)  # Check every 5 seconds

    async def _try_reconnect(self, name: str) -> bool:
        """Attempt to reconnect to a server.

        Args:
            name: Server name

        Returns:
            True if reconnection successful
        """
        entry = self._servers.get(name)
        if not entry:
            return False

        if entry.consecutive_failures >= entry.config.max_retries:
            logger.error(
                f"Server '{name}' exceeded max retries ({entry.config.max_retries}), "
                "giving up until manual reset"
            )
            entry.status = ServerStatus.FAILED
            return False

        logger.info(
            f"Attempting reconnection to '{name}' "
            f"(attempt {entry.consecutive_failures + 1}/{entry.config.max_retries})"
        )

        # Disconnect first
        if entry.client:
            entry.client.disconnect()
            entry.client = None

        await asyncio.sleep(entry.config.retry_delay)
        return await self.connect(name)

    async def start(self) -> None:
        """Start the registry and connect to all servers."""
        self._running = True
        await self.connect_all()

        if self._health_check_enabled:
            self._health_task = asyncio.create_task(self._health_monitor_loop())
            logger.info("Started MCP health monitoring")

    async def stop(self) -> None:
        """Stop the registry and disconnect all servers."""
        self._running = False

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

        await self.disconnect_all()
        logger.info("Stopped MCP registry")

    # Tool access methods

    async def call_tool(self, tool_name: str, **arguments: Any) -> MCPToolCallResult:
        """Call a tool across any registered server.

        Args:
            tool_name: Name of tool to call
            **arguments: Tool arguments

        Returns:
            Tool call result
        """
        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            return MCPToolCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found in any connected server",
            )

        entry = self._servers.get(server_name)
        if not entry or not entry.client or entry.status != ServerStatus.CONNECTED:
            return MCPToolCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Server '{server_name}' not connected",
            )

        return await entry.client.call_tool(tool_name, **arguments)

    def get_all_tools(self) -> List[MCPTool]:
        """Get all tools from all connected servers.

        Returns:
            List of all available tools
        """
        tools = []
        for entry in self._servers.values():
            if entry.status == ServerStatus.CONNECTED:
                tools.extend(entry.tools_cache)
        return tools

    def get_all_resources(self) -> List[MCPResource]:
        """Get all resources from all connected servers.

        Returns:
            List of all available resources
        """
        resources = []
        for entry in self._servers.values():
            if entry.status == ServerStatus.CONNECTED:
                resources.extend(entry.resources_cache)
        return resources

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a specific tool by name.

        Args:
            name: Tool name

        Returns:
            Tool definition or None
        """
        server_name = self._tool_to_server.get(name)
        if not server_name:
            return None

        entry = self._servers.get(server_name)
        if not entry:
            return None

        return next((t for t in entry.tools_cache if t.name == name), None)

    def get_tools_by_server(self, server_name: str) -> List[MCPTool]:
        """Get tools from a specific server.

        Args:
            server_name: Server name

        Returns:
            List of tools from that server
        """
        entry = self._servers.get(server_name)
        if not entry:
            return []
        return entry.tools_cache

    def get_tools_by_tag(self, tag: str) -> List[MCPTool]:
        """Get tools from servers with a specific tag.

        Args:
            tag: Server tag to filter by

        Returns:
            List of tools from matching servers
        """
        tools = []
        for entry in self._servers.values():
            if tag in entry.config.tags and entry.status == ServerStatus.CONNECTED:
                tools.extend(entry.tools_cache)
        return tools

    # Status and discovery methods

    def get_server_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific server.

        Args:
            name: Server name

        Returns:
            Status dictionary or None
        """
        entry = self._servers.get(name)
        if not entry:
            return None

        return {
            "name": name,
            "status": entry.status.name,
            "enabled": entry.config.enabled,
            "tools_count": len(entry.tools_cache),
            "resources_count": len(entry.resources_cache),
            "last_health_check": entry.last_health_check,
            "consecutive_failures": entry.consecutive_failures,
            "error": entry.error_message,
            "tags": entry.config.tags,
        }

    def get_registry_status(self) -> Dict[str, Any]:
        """Get overall registry status.

        Returns:
            Status dictionary
        """
        servers = {}
        total_tools = 0
        total_resources = 0
        connected = 0

        for name, entry in self._servers.items():
            servers[name] = {
                "status": entry.status.name,
                "tools": len(entry.tools_cache),
                "resources": len(entry.resources_cache),
            }
            if entry.status == ServerStatus.CONNECTED:
                connected += 1
                total_tools += len(entry.tools_cache)
                total_resources += len(entry.resources_cache)

        return {
            "servers": servers,
            "total_servers": len(self._servers),
            "connected_servers": connected,
            "total_tools": total_tools,
            "total_resources": total_resources,
            "health_monitoring": self._health_check_enabled,
            "running": self._running,
        }

    def list_servers(self) -> List[str]:
        """List all registered server names.

        Returns:
            List of server names
        """
        return list(self._servers.keys())

    def reset_server(self, name: str) -> bool:
        """Reset a failed server to allow reconnection.

        Args:
            name: Server name

        Returns:
            True if reset successful
        """
        entry = self._servers.get(name)
        if not entry:
            return False

        entry.consecutive_failures = 0
        entry.status = ServerStatus.DISCONNECTED
        entry.error_message = None
        logger.info(f"Reset server '{name}' for reconnection")
        return True

    # Event handling

    def on_event(self, callback: Callable[[str, str, Any], None]) -> None:
        """Register callback for registry events.

        Args:
            callback: Callback function(event_type, server_name, data)
        """
        self._event_callbacks.append(callback)

    def _emit_event(self, event_type: str, server_name: str, data: Any) -> None:
        """Emit event to all registered callbacks.

        Args:
            event_type: Type of event (connecting, connected, failed, disconnected)
            server_name: Server name
            data: Event data
        """
        for callback in self._event_callbacks:
            try:
                callback(event_type, server_name, data)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    # Configuration loading

    @classmethod
    def discover_servers(cls, include_claude_desktop: bool = True) -> "MCPRegistry":
        """Auto-discover MCP servers from standard locations.

        Comprehensive discovery that searches:
        1. Environment variable VICTOR_MCP_CONFIG
        2. {project}/.victor/mcp.yaml (project-local)
        3. ~/.victor/mcp.yaml (global)
        4. ~/.config/mcp/servers.yaml (XDG standard)
        5. Claude Desktop MCP configuration (if include_claude_desktop=True)
        6. Well-known MCP server executables in PATH

        Args:
            include_claude_desktop: Whether to discover Claude Desktop's MCP servers

        Returns:
            Configured MCPRegistry with discovered servers
        """
        import os

        from victor.config.settings import get_project_paths

        registry = cls()
        search_paths: List[Path] = []
        paths = get_project_paths()

        # Priority order of config locations
        env_config = os.environ.get("VICTOR_MCP_CONFIG")
        if env_config:
            search_paths.append(Path(env_config))

        # Project-local config (highest priority)
        search_paths.append(paths.project_victor_dir / "mcp.yaml")
        search_paths.append(paths.project_victor_dir / "mcp.yml")

        # Global user config
        search_paths.append(paths.global_victor_dir / "mcp.yaml")
        search_paths.append(paths.global_victor_dir / "mcp.yml")
        # Also search XDG config directory (with secure home resolution)
        try:
            from victor.config.secure_paths import get_secure_home

            secure_home = get_secure_home()
        except ImportError:
            secure_home = Path.home()
        search_paths.append(secure_home / ".config" / "mcp" / "servers.yaml")
        search_paths.append(secure_home / ".config" / "mcp" / "servers.yml")

        # Load from first found config
        for config_path in search_paths:
            if config_path.exists():
                logger.info(f"Loading MCP config from: {config_path}")
                registry = cls.from_config(config_path)
                break

        # Discover Claude Desktop servers
        if include_claude_desktop:
            claude_servers = cls._discover_claude_desktop_servers()
            for server in claude_servers:
                if server.name not in {s.config.name for s in registry._servers.values()}:
                    registry.register_server(server)
                    logger.info(f"Discovered Claude Desktop MCP server: {server.name}")

        # Discover well-known MCP server executables
        discovered_executables = cls._discover_mcp_executables()
        for server in discovered_executables:
            if server.name not in {s.config.name for s in registry._servers.values()}:
                registry.register_server(server)
                logger.info(f"Discovered MCP server executable: {server.name}")

        if not registry._servers:
            logger.debug("No MCP servers discovered")
        else:
            logger.info(f"Discovered {len(registry._servers)} MCP servers")

        return registry

    @classmethod
    def _discover_claude_desktop_servers(cls) -> List[MCPServerConfig]:
        """Discover MCP servers configured in Claude Desktop.

        Claude Desktop stores MCP server config in:
        - macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
        - Windows: %APPDATA%/Claude/claude_desktop_config.json
        - Linux: ~/.config/Claude/claude_desktop_config.json

        Returns:
            List of discovered MCPServerConfig instances
        """
        import json
        import platform

        servers: List[MCPServerConfig] = []

        try:
            from victor.config.secure_paths import get_secure_home

            home = get_secure_home()
        except ImportError:
            home = Path.home()

        # Platform-specific Claude Desktop config locations
        system = platform.system()
        config_paths: List[Path] = []

        if system == "Darwin":  # macOS
            config_paths.append(
                home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
            )
        elif system == "Windows":
            import os

            appdata = os.environ.get("APPDATA", "")
            if appdata:
                config_paths.append(Path(appdata) / "Claude" / "claude_desktop_config.json")
        else:  # Linux and others
            config_paths.append(home / ".config" / "Claude" / "claude_desktop_config.json")

        for config_path in config_paths:
            if not config_path.exists():
                continue

            try:
                config = json.loads(config_path.read_text())
                mcp_servers = config.get("mcpServers", {})

                for name, server_config in mcp_servers.items():
                    command = server_config.get("command", "")
                    args = server_config.get("args", [])
                    env = server_config.get("env", {})

                    if command:
                        full_command = [command] + args if args else [command]
                        servers.append(
                            MCPServerConfig(
                                name=f"claude_{name}",
                                command=full_command,
                                description=f"Claude Desktop MCP server: {name}",
                                env=env,
                                tags=["claude-desktop", "auto-discovered"],
                                auto_connect=False,  # Don't auto-connect by default
                            )
                        )
                        logger.debug(f"Found Claude Desktop MCP server: {name}")

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in Claude Desktop config {config_path}: {e}")
            except Exception as e:
                logger.debug(f"Error reading Claude Desktop config {config_path}: {e}")

        return servers

    @classmethod
    def _discover_mcp_executables(cls) -> List[MCPServerConfig]:
        """Discover well-known MCP server executables in PATH.

        Searches for common MCP server packages that may be installed:
        - mcp-server-* (npm packages)
        - mcp_server_* (Python packages)

        Returns:
            List of discovered MCPServerConfig instances
        """
        import shutil

        servers: List[MCPServerConfig] = []

        # Well-known MCP server executables and their configurations
        known_servers = {
            # NPM-based servers (typically installed globally)
            "mcp-server-filesystem": {
                "description": "MCP Filesystem Server - file operations",
                "tags": ["filesystem", "npm"],
            },
            "mcp-server-github": {
                "description": "MCP GitHub Server - repository operations",
                "tags": ["github", "npm"],
            },
            "mcp-server-sqlite": {
                "description": "MCP SQLite Server - database operations",
                "tags": ["database", "sqlite", "npm"],
            },
            "mcp-server-postgres": {
                "description": "MCP PostgreSQL Server - database operations",
                "tags": ["database", "postgres", "npm"],
            },
            "mcp-server-memory": {
                "description": "MCP Memory Server - key-value storage",
                "tags": ["memory", "npm"],
            },
            "mcp-server-brave-search": {
                "description": "MCP Brave Search Server - web search",
                "tags": ["search", "web", "npm"],
            },
            "mcp-server-puppeteer": {
                "description": "MCP Puppeteer Server - browser automation",
                "tags": ["browser", "puppeteer", "npm"],
            },
            "mcp-server-slack": {
                "description": "MCP Slack Server - Slack integration",
                "tags": ["slack", "chat", "npm"],
            },
            "mcp-server-google-maps": {
                "description": "MCP Google Maps Server - location services",
                "tags": ["maps", "google", "npm"],
            },
            # NPX-runnable servers
            "@anthropics/mcp-server-fetch": {
                "command_template": ["npx", "-y", "@anthropics/mcp-server-fetch"],
                "description": "Anthropic MCP Fetch Server - HTTP requests",
                "tags": ["fetch", "http", "anthropic"],
            },
        }

        for server_name, config in known_servers.items():
            # Check if executable exists in PATH
            command_template = config.get("command_template")
            if command_template:
                # NPX-based server - check if npx is available
                if shutil.which("npx"):
                    servers.append(
                        MCPServerConfig(
                            name=server_name.replace("@", "").replace("/", "_"),
                            command=command_template,
                            description=config.get("description", f"MCP Server: {server_name}"),
                            tags=["auto-discovered"] + config.get("tags", []),
                            auto_connect=False,
                        )
                    )
            else:
                # Direct executable
                executable_path = shutil.which(server_name)
                if executable_path:
                    servers.append(
                        MCPServerConfig(
                            name=server_name.replace("-", "_"),
                            command=[executable_path],
                            description=config.get("description", f"MCP Server: {server_name}"),
                            tags=["auto-discovered"] + config.get("tags", []),
                            auto_connect=False,
                        )
                    )

        # Also check for Python-based MCP servers
        python_servers = [
            "mcp_server_git",
            "mcp_server_time",
            "mcp_server_fetch",
        ]

        for server_name in python_servers:
            # Try to import and check if module exists
            try:
                import importlib.util

                spec = importlib.util.find_spec(server_name)
                if spec:
                    servers.append(
                        MCPServerConfig(
                            name=server_name,
                            command=["python", "-m", server_name],
                            description=f"Python MCP Server: {server_name}",
                            tags=["auto-discovered", "python"],
                            auto_connect=False,
                        )
                    )
            except Exception:
                pass

        return servers

    @classmethod
    def list_discovered_servers(cls) -> Dict[str, Any]:
        """List all discoverable MCP servers without connecting.

        This is useful for showing users what servers are available
        before deciding which to connect to.

        Returns:
            Dictionary with discovered servers organized by source
        """
        result: Dict[str, Any] = {
            "config_files": [],
            "claude_desktop": [],
            "executables": [],
        }

        # Check config file locations
        import os

        from victor.config.settings import get_project_paths

        paths = get_project_paths()
        config_locations = [
            ("env", os.environ.get("VICTOR_MCP_CONFIG")),
            ("project", str(paths.project_victor_dir / "mcp.yaml")),
            ("global", str(paths.global_victor_dir / "mcp.yaml")),
        ]

        for source, path in config_locations:
            if path and Path(path).exists():
                result["config_files"].append({"source": source, "path": path})

        # Discover Claude Desktop servers
        claude_servers = cls._discover_claude_desktop_servers()
        for server in claude_servers:
            result["claude_desktop"].append(
                {
                    "name": server.name,
                    "command": server.command,
                    "description": server.description,
                }
            )

        # Discover executables
        exec_servers = cls._discover_mcp_executables()
        for server in exec_servers:
            result["executables"].append(
                {
                    "name": server.name,
                    "command": server.command,
                    "description": server.description,
                    "tags": server.tags,
                }
            )

        return result

    @classmethod
    def from_config(cls, config_path: Path) -> "MCPRegistry":
        """Create registry from configuration file.

        Supports YAML and JSON formats.

        Args:
            config_path: Path to configuration file

        Returns:
            Configured MCPRegistry instance
        """
        import json

        if not config_path.exists():
            logger.warning(f"MCP config file not found: {config_path}")
            return cls()

        content = config_path.read_text()

        if config_path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                config = yaml.safe_load(content)
            except ImportError:
                logger.warning("PyYAML not installed, cannot load YAML config")
                return cls()
        else:
            config = json.loads(content)

        registry = cls(
            health_check_enabled=config.get("health_check_enabled", True),
            default_health_interval=config.get("default_health_interval", 30),
        )

        for server_config in config.get("servers", []):
            registry.register_server(MCPServerConfig(**server_config))

        return registry

    async def __aenter__(self) -> "MCPRegistry":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
