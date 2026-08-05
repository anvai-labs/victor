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

"""Tool registration facade coordinating specialized loaders.

This module provides a unified facade for tool registration, delegating to
specialized components for each responsibility (SRP compliance).

Design Pattern: Facade Pattern
==============================
ToolRegistrar is a thin facade coordinating four specialized components:

1. ToolCatalogLoader: Dynamic tool discovery from victor/tools
   - Registers tools from SharedToolRegistry
   - Applies enabled/disabled configuration

2. PluginLoader: Plugin system management
   - Discovers and loads tool plugins
   - Registers plugin tools

3. MCPConnector: MCP server discovery and connection
   - Auto-discovers MCP servers
   - Manages connection lifecycle

4. ToolGraphBuilder: Tool dependency graph setup
   - Registers tool input/output specs
   - Provides goal-based tool planning

SOLID Compliance:
- SRP: Each component has single responsibility
- OCP: New loaders can be added without modifying facade
- LSP: Components are substitutable via their protocols
- ISP: Clients only depend on facade interface
- DIP: Facade depends on abstractions (component interfaces)

Benefits of this architecture:
- Each component can be tested in isolation
- Plugin/MCP code not loaded unless enabled
- Clear dependency injection boundaries
- Easy to add new registration sources

Usage:
    from victor.agent.tool_registrar import ToolRegistrar

    registrar = ToolRegistrar(
        tools=tool_registry,
        settings=settings,
        provider=provider,
        model=model,
    )

    # Register all tools
    await registrar.initialize()

    # Get registered tool count
    stats = registrar.get_stats()
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from victor.providers.base import BaseProvider, ToolDefinition
from victor.tools.enums import CostTier
from victor.tools.registry import ToolRegistry

# Import specialized components (lazy to avoid circular imports)
# These are imported at runtime in methods that use them

logger = logging.getLogger(__name__)


@dataclass
class ToolRegistrarConfig:
    """Configuration for ToolRegistrar.

    Attributes:
        enable_plugins: Enable plugin system
        enable_mcp: Enable MCP integration
        enable_tool_graph: Enable tool dependency graph
        airgapped_mode: Disable web tools
        plugin_dirs: Additional plugin directories
        disabled_plugins: Plugin names to disable
        plugin_packages: Package names to load plugins from
        max_workers: Max workers for batch processing
        max_complexity: Max complexity for code review
    """

    enable_plugins: bool = True
    enable_mcp: bool = False
    enable_tool_graph: bool = True
    airgapped_mode: bool = False
    plugin_dirs: List[str] = field(default_factory=list)
    disabled_plugins: Set[str] = field(default_factory=set)
    plugin_packages: List[str] = field(default_factory=list)
    max_workers: int = 4
    max_complexity: int = 10
    lazy_startup: bool = True


@dataclass
class RegistrationStats:
    """Statistics about tool registration.

    Attributes:
        dynamic_tools: Tools registered from victor/tools
        plugin_tools: Tools registered from plugins
        mcp_tools: Tools registered from MCP servers
        dependency_graph_tools: Tools registered in dependency graph
        total_tools: Total registered tools
        plugins_loaded: Number of plugins loaded
        mcp_servers_connected: Number of MCP servers connected
    """

    dynamic_tools: int = 0
    plugin_tools: int = 0
    mcp_tools: int = 0
    dependency_graph_tools: int = 0
    total_tools: int = 0
    plugins_loaded: int = 0
    mcp_servers_connected: int = 0


class ToolRegistrar:
    """Facade for tool registration coordinating specialized components.

    This class provides a unified interface for tool registration while
    delegating to specialized components for each responsibility:

    - ToolCatalogLoader: Dynamic tool discovery
    - PluginLoader: Plugin management
    - MCPConnector: MCP integration
    - ToolGraphBuilder: Dependency graph setup

    SOLID Compliance:
    - SRP: Facade only coordinates, delegates actual work
    - OCP: New components can be added without modifying facade
    - DIP: Depends on component interfaces, not implementations

    Usage:
        registrar = ToolRegistrar(
            tools=tool_registry,
            settings=settings,
            provider=provider,
            model=model,
        )
        await registrar.initialize()
    """

    def __init__(
        self,
        tools: ToolRegistry,
        settings: Any,
        provider: Optional[BaseProvider] = None,
        model: Optional[str] = None,
        tool_graph: Optional[Any] = None,
        config: Optional[ToolRegistrarConfig] = None,
    ):
        """Initialize the tool registrar facade.

        Args:
            tools: Tool registry to register tools with
            settings: Application settings
            provider: LLM provider for tool configuration
            model: Model name for tool configuration
            tool_graph: Optional tool dependency graph
            config: Optional configuration
        """
        self.tools = tools
        self.settings = settings
        self.provider = provider
        self.model = model
        self.tool_graph = tool_graph
        self.config = config or ToolRegistrarConfig()

        # Specialized components (lazy-initialized)
        self._catalog_loader: Optional[Any] = None
        self._plugin_loader: Optional[Any] = None
        self._mcp_connector: Optional[Any] = None
        self._graph_builder: Optional[Any] = None

        # Legacy compatibility attributes
        self.plugin_manager: Optional[Any] = None
        self.mcp_registry: Optional[Any] = None
        self._mcp_tasks: List[asyncio.Task] = []

        # Statistics
        self._stats = RegistrationStats()

        # Background task callback
        self._create_background_task: Optional[Callable] = None

        # Tool configuration for context injection (populated in _setup_providers)
        self._tool_config: Dict[str, Any] = {}

        # Lazy loading flag - tools are not loaded until first access
        self._tools_loaded: bool = False

        logger.debug("ToolRegistrar facade initialized")

    def set_background_task_callback(self, callback: Callable[[Any, str], asyncio.Task]) -> None:
        """Set callback for creating background tasks.

        Args:
            callback: Function that takes (coroutine, name) and returns Task
        """
        self._create_background_task = callback
        # Propagate to MCP connector if initialized
        if self._mcp_connector:
            self._mcp_connector.set_task_callback(callback)

    def _create_task(self, coro: Any, name: str) -> Optional[asyncio.Task]:
        """Create a background task using callback or directly."""
        if self._create_background_task:
            return self._create_background_task(coro, name)
        else:
            task = asyncio.create_task(coro)
            self._mcp_tasks.append(task)
            return task

    # =========================================================================
    # Component Accessors (Lazy Initialization)
    # =========================================================================

    def _get_catalog_loader(self) -> Any:
        """Get or create the catalog loader component."""
        if self._catalog_loader is None:
            from victor.agent.tool_catalog_loader import (
                ToolCatalogLoader,
                ToolCatalogConfig,
            )

            self._catalog_loader = ToolCatalogLoader(
                registry=self.tools,
                settings=self.settings,
                config=ToolCatalogConfig(
                    airgapped_mode=self.config.airgapped_mode,
                    lazy_startup=self.config.lazy_startup,
                ),
            )
        return self._catalog_loader

    def _get_plugin_loader(self) -> Any:
        """Get or create the plugin loader component."""
        if self._plugin_loader is None:
            from victor.agent.plugin_loader import PluginLoader, PluginLoaderConfig

            self._plugin_loader = PluginLoader(
                registry=self.tools,
                settings=self.settings,
                config=PluginLoaderConfig(
                    enabled=self.config.enable_plugins,
                    plugin_dirs=self.config.plugin_dirs,
                    disabled_plugins=self.config.disabled_plugins,
                    plugin_packages=self.config.plugin_packages,
                ),
            )
        return self._plugin_loader

    def _get_mcp_connector(self) -> Any:
        """Get or create the MCP connector component."""
        if self._mcp_connector is None:
            from victor.agent.mcp_connector import MCPConnector, MCPConnectorConfig

            self._mcp_connector = MCPConnector(
                registry=self.tools,
                settings=self.settings,
                config=MCPConnectorConfig(
                    enabled=self.config.enable_mcp,
                ),
                task_callback=self._create_background_task,
            )
        return self._mcp_connector

    def _get_graph_builder(self) -> Any:
        """Get or create the graph builder component."""
        if self._graph_builder is None:
            from victor.agent.tool_graph_builder import (
                ToolGraphBuilder,
                ToolGraphConfig,
            )

            self._graph_builder = ToolGraphBuilder(
                registry=self.tools,
                tool_graph=self.tool_graph,
                config=ToolGraphConfig(
                    enabled=self.config.enable_tool_graph,
                ),
            )
        return self._graph_builder

    def _ensure_tools_loaded(self) -> None:
        """Ensure tools are loaded (lazy loading pattern).

        This method loads tools on first access rather than during __init__,
        improving startup time by deferring tool discovery until needed.

        Delegates to ToolCatalogLoader component (SRP compliance).
        """
        if self._tools_loaded:
            return

        # Delegate to catalog loader component
        catalog_loader = self._get_catalog_loader()
        result = catalog_loader.load()

        self._stats.dynamic_tools = result.tools_loaded

        # Mark as loaded
        self._tools_loaded = True
        logger.debug(f"Lazy-loaded {self._stats.dynamic_tools} tools via CatalogLoader")

    def get_tool(self, name: str) -> Optional[Any]:
        """Get a tool by name, hydrating only that tool when possible.

        Args:
            name: The name of the tool to retrieve

        Returns:
            The tool if found, None otherwise
        """
        tool = self.tools.get(name)
        if tool is not None:
            return tool
        self.ensure_tool_registered(name)
        return self.tools.get(name)

    def get_all_tools(self) -> List[Any]:
        """Get all registered tools, triggering lazy loading if needed.

        Returns:
            List of all registered tools
        """
        self._ensure_tools_loaded()
        return self.tools.list_tools()

    async def initialize(self) -> RegistrationStats:
        """Initialize all tool registration via specialized components.

        Performs (delegating to SRP-compliant components):
        1. Pre-registration setup (providers, configs)
        2. Dynamic tool discovery via ToolCatalogLoader
        3. Plugin loading via PluginLoader
        4. MCP integration via MCPConnector
        5. Tool dependency graph via ToolGraphBuilder

        Returns:
            RegistrationStats with registration counts
        """
        # Pre-registration setup
        self._setup_providers()

        # 1. Dynamic tool registration via CatalogLoader
        if not self._tools_loaded:
            catalog_loader = self._get_catalog_loader()
            result = (
                catalog_loader.load_bootstrap()
                if self.config.lazy_startup
                else catalog_loader.load()
            )
            self._stats.dynamic_tools = result.tools_loaded
            self._tools_loaded = result.full_catalog_loaded

        # 2. Plugin loading via PluginLoader
        if self.config.enable_plugins:
            plugin_loader = self._get_plugin_loader()
            result = plugin_loader.load()
            self._stats.plugin_tools = result.tools_registered
            self._stats.plugins_loaded = result.plugins_loaded
            # Maintain legacy compatibility
            self.plugin_manager = plugin_loader.plugin_manager

        # 3. MCP integration via MCPConnector
        if self.config.enable_mcp or getattr(self.settings, "use_mcp_tools", False):
            mcp_connector = self._get_mcp_connector()
            result = await mcp_connector.connect()
            self._stats.mcp_tools = result.tools_registered
            self._stats.mcp_servers_connected = result.servers_connected
            # Maintain legacy compatibility
            self.mcp_registry = mcp_connector.mcp_registry

        # 4. Tool dependency graph via GraphBuilder
        if self.config.enable_tool_graph and self.tool_graph:
            graph_builder = self._get_graph_builder()
            result = graph_builder.build()
            self._stats.dependency_graph_tools = result.tools_registered

        # Calculate totals
        self._stats.total_tools = len(self.tools.list_tools())

        logger.debug(
            f"Tool registration complete: {self._stats.total_tools} tools "
            f"(dynamic: {self._stats.dynamic_tools}, plugins: {self._stats.plugin_tools}, "
            f"MCP: {self._stats.mcp_tools})"
        )

        return self._stats

    def _setup_providers(self) -> None:
        """Set up provider configuration for tool context injection.

        Note: As of Phase 7, tools receive configuration via context parameter
        at execution time rather than via global setters. This method now stores
        configuration that will be injected into the tool execution context.
        """
        # Store configuration for context injection during tool execution
        # This config will be passed to ToolConfig.from_context() at runtime
        self._tool_config = {
            "provider": self.provider,
            "model": self.model,
            "max_workers": self.config.max_workers,
            "max_complexity": self.config.max_complexity,
        }

        # Load web tool config if not air-gapped
        if not self.config.airgapped_mode:
            try:
                tool_config = self.settings.load_tool_config()
                web_cfg = tool_config.get("web_tools", {}) or tool_config.get("web", {}) or {}
                self._tool_config.update(
                    {
                        "web_fetch_top": web_cfg.get("summarize_fetch_top"),
                        "web_fetch_pool": web_cfg.get("summarize_fetch_pool"),
                        "max_content_length": web_cfg.get("summarize_max_content_length"),
                    }
                )
            except Exception as exc:
                logger.debug(f"Failed to load web tool config: {exc}")

    def _register_dynamic_tools(self) -> int:
        """Register tools from the shared tool registry.

        Uses SharedToolRegistry to get pre-discovered tool definitions,
        avoiding redundant discovery across multiple orchestrator instances.
        This significantly reduces memory footprint for concurrent sessions.

        Returns:
            Number of tools registered
        """
        from victor.agent.shared_tool_registry import SharedToolRegistry

        # Get shared tool registry instance
        shared_registry = SharedToolRegistry.get_instance()

        # Get all tools ready for registration (respects airgapped mode)
        tools_to_register = shared_registry.get_all_tools_for_registration(
            airgapped_mode=self.config.airgapped_mode
        )

        registered_count = 0
        for tool in tools_to_register:
            try:
                self.tools.register(tool)
                registered_count += 1
            except Exception as e:
                tool_name = getattr(tool, "name", getattr(tool, "__name__", str(tool)))
                logger.debug(f"Skipped registering {tool_name}: {e}")

        logger.debug(f"Registered {registered_count} tools from shared registry")
        return registered_count

    def register_default_tools(self) -> int:
        """Register default tool surfaces via the canonical registrar API.

        Performs the split-phase bootstrap used by the orchestrator/runtime:
        provider setup, dynamic tool registration, and optional MCP
        integration. This preserves the current initialization order while
        moving bootstrap callers off private registrar methods.

        Returns:
            Total number of tools registered by this phase
        """
        self._setup_providers()

        registered_count = (
            self._register_bootstrap_tools()
            if self.config.lazy_startup
            else self._register_dynamic_tools()
        )
        total_registered = registered_count

        if self.config.enable_mcp or getattr(self.settings, "use_mcp_tools", False):
            total_registered += self._setup_mcp_integration()

        return total_registered

    def _register_bootstrap_tools(self) -> int:
        """Register the compact startup tool set without full catalog discovery."""
        catalog_loader = self._get_catalog_loader()
        result = catalog_loader.load_bootstrap()
        self._stats.dynamic_tools += result.tools_loaded
        logger.debug("Tool bootstrap registration complete: %s tools", result.tools_loaded)
        return result.tools_loaded

    def ensure_tool_registered(self, tool_name: str) -> bool:
        """Hydrate one tool into the registry when startup used lazy loading."""
        if not tool_name:
            return False
        if self.tools.get(tool_name) is not None:
            return True

        catalog_loader = self._get_catalog_loader()
        result = catalog_loader.ensure_tools([tool_name])
        if result.tools_loaded:
            logger.debug("Lazy-registered tool on demand: %s", tool_name)
            return self.tools.get(tool_name) is not None

        if not self._tools_loaded and not self.config.lazy_startup:
            self._ensure_tools_loaded()
            return self.tools.get(tool_name) is not None

        return False

    def ensure_tools_for_query(self, text: str) -> int:
        """Hydrate specialty tools implied by a user message."""
        if not text:
            return 0
        from victor.agent.shared_tool_registry import SharedToolRegistry

        shared_registry = SharedToolRegistry.get_instance()
        needed = shared_registry.infer_demand_tools(text)
        if not needed:
            return 0
        catalog_loader = self._get_catalog_loader()
        result = catalog_loader.ensure_tools(needed)
        if result.tools_loaded:
            logger.debug("Lazy-registered demand tools: %s", needed)
        return result.tools_loaded

    def _load_tool_configurations(self) -> None:
        """Load tool configurations from profiles.yaml.

        Loads tool enable/disable states from the 'tools' section in profiles.yaml.
        Expected format:

        tools:
          enabled:
            - read_file
            - write_file
            - execute_bash
          disabled:
            - code_review
            - security_scan

        Or:

        tools:
          code_review:
            enabled: false
          security_scan:
            enabled: false
        """
        try:
            tool_config = self.settings.load_tool_config()
            if not tool_config:
                return

            # Get all registered tool names for validation
            registered_tools = {tool.name for tool in self.tools.list_tools(only_enabled=False)}

            # Get critical tools dynamically from registry (priority=Priority.CRITICAL)
            from victor.agent.tool_selection import get_critical_tools

            core_tools = get_critical_tools(self.tools)

            # Format 1: Lists of enabled/disabled tools
            if "enabled" in tool_config:
                enabled_tools = tool_config.get("enabled", [])

                # Validate tool names
                invalid_tools = [t for t in enabled_tools if t not in registered_tools]
                if invalid_tools:
                    logger.warning(
                        f"Configuration contains invalid tool names in 'enabled' list: {', '.join(invalid_tools)}. "
                        f"Available tools: {', '.join(sorted(registered_tools))}"
                    )

                # Check if core tools are included (use dynamically discovered core tools)
                missing_core = core_tools - set(enabled_tools)
                if missing_core:
                    logger.warning(
                        f"'enabled' list is missing recommended core tools: {', '.join(missing_core)}. "
                        f"This may limit agent functionality."
                    )

                # First disable all tools
                for tool in self.tools.list_tools(only_enabled=False):
                    self.tools.disable_tool(tool.name)
                # Then enable only the specified ones
                for tool_name in enabled_tools:
                    if tool_name in registered_tools:
                        self.tools.enable_tool(tool_name)

            if "disabled" in tool_config:
                disabled_tools = tool_config.get("disabled", [])

                # Validate tool names
                invalid_tools = [t for t in disabled_tools if t not in registered_tools]
                if invalid_tools:
                    logger.warning(
                        f"Configuration contains invalid tool names in 'disabled' list: {', '.join(invalid_tools)}. "
                        f"Available tools: {', '.join(sorted(registered_tools))}"
                    )

                # Warn if disabling core tools
                disabled_core = core_tools & set(disabled_tools)
                if disabled_core:
                    logger.warning(
                        f"Disabling core tools: {', '.join(disabled_core)}. "
                        f"This may limit agent functionality."
                    )

                for tool_name in disabled_tools:
                    if tool_name in registered_tools:
                        self.tools.disable_tool(tool_name)

            # Format 2: Individual tool settings
            for tool_name, config in tool_config.items():
                if isinstance(config, dict) and "enabled" in config:
                    if tool_name not in registered_tools:
                        logger.warning(
                            f"Configuration contains invalid tool name: '{tool_name}'. "
                            f"Available tools: {', '.join(sorted(registered_tools))}"
                        )
                        continue

                    if config["enabled"]:
                        self.tools.enable_tool(tool_name)
                    else:
                        self.tools.disable_tool(tool_name)
                        # Warn if disabling core tools
                        if tool_name in core_tools:
                            logger.warning(
                                f"Disabling core tool '{tool_name}'. This may limit agent functionality."
                            )

            # Log tool states
            disabled_tools = [
                name for name, enabled in self.tools.get_tool_states().items() if not enabled
            ]
            if disabled_tools:
                logger.debug(f"Disabled tools: {', '.join(sorted(disabled_tools))}")

            # Log enabled tool count
            enabled_count = sum(1 for enabled in self.tools.get_tool_states().values() if enabled)
            logger.debug(f"Enabled tools: {enabled_count}/{len(registered_tools)}")

        except Exception as e:
            logger.warning(f"Failed to load tool configurations: {e}")

    def _load_plugin_tools(self) -> int:
        """Discover, load, and register plugin tools using PluginLoader component.

        Delegates to PluginLoader for plugin management (SRP compliance).

        Returns:
            Number of plugin tools registered
        """
        try:
            plugin_loader = self._get_plugin_loader()
            result = plugin_loader.load()

            # Store plugin_manager for backward compatibility
            # TODO: Remove self.plugin_manager attribute once callers migrate to PluginLoader
            self.plugin_manager = plugin_loader._plugin_manager

            return result.tools_registered

        except Exception as e:
            logger.warning(f"Failed to initialize plugin system: {e}")
            self.plugin_manager = None
            return 0

    def initialize_plugins(self) -> int:
        """Initialize plugin tools via the canonical registrar API."""
        return self._load_plugin_tools()

    def _setup_mcp_integration(self) -> int:
        """Set up MCP integration using registry or legacy client.

        Returns:
            Number of MCP tools registered
        """
        mcp_command = getattr(self.settings, "mcp_command", None)
        mcp_tools_registered = 0

        # Try MCPRegistry with auto-discovery first
        try:
            from victor.integrations.mcp.registry import MCPRegistry, MCPServerConfig

            # Auto-discover MCP servers from standard locations
            self.mcp_registry = MCPRegistry.discover_servers()

            # Also register command from settings if specified
            if mcp_command:
                cmd_parts = mcp_command.split()
                self.mcp_registry.register_server(
                    MCPServerConfig(
                        name="settings_mcp",
                        command=cmd_parts,
                        description="MCP server from settings",
                        auto_connect=True,
                    )
                )

            # Start registry and connect to servers in background
            if self.mcp_registry.list_servers():
                logger.debug(
                    f"MCP Registry initialized with {len(self.mcp_registry.list_servers())} server(s)"
                )
                self._create_task(self._start_mcp_registry(), "mcp_registry_start")

        except ImportError:
            logger.debug("MCPRegistry not available, using legacy client")
            self._setup_legacy_mcp(mcp_command)

        # Register MCP tool definitions
        try:
            from victor.tools.mcp_bridge_tool import get_mcp_tool_definitions

            for mcp_tool in get_mcp_tool_definitions():
                self.tools.register_dict(mcp_tool)
                mcp_tools_registered += 1
        except ImportError:
            logger.debug("MCP bridge tool not available")

        return mcp_tools_registered

    def _setup_legacy_mcp(self, mcp_command: Optional[str]) -> None:
        """Set up legacy single MCP client (backwards compatibility)."""
        if mcp_command:
            try:
                from victor.integrations.mcp.client import MCPClient

                mcp_client = MCPClient()
                cmd_parts = mcp_command.split()
                self._create_task(mcp_client.connect(cmd_parts), "mcp_legacy_connect")
                # MCP client setup is now handled via context injection
            except Exception as exc:
                logger.warning(f"Failed to start MCP client: {exc}")

    async def _start_mcp_registry(self) -> None:
        """Start MCP registry and connect to discovered servers."""
        try:
            await self.mcp_registry.start()
            results = await self.mcp_registry.connect_all()
            connected = sum(1 for v in results.values() if v)
            self._stats.mcp_servers_connected = connected
            if connected > 0:
                logger.info(f"Connected to {connected} MCP server(s)")
                # Update available tools from MCP
                mcp_tools = self.mcp_registry.get_all_tools()
                if mcp_tools:
                    logger.debug(f"Discovered {len(mcp_tools)} MCP tools")
        except Exception as e:
            logger.warning(f"Failed to start MCP registry: {e}")

    def _register_tool_dependencies(self) -> int:
        """Register tool input/output specs for planning with cost tiers.

        Returns:
            Number of tools registered in dependency graph
        """
        if not self.tool_graph:
            return 0

        registered = 0
        try:
            # Search tools - FREE tier (local operations)
            self.tool_graph.add_tool(
                "code_search",
                inputs=["query"],
                outputs=["file_candidates"],
                cost_tier=CostTier.FREE,
            )
            registered += 1

            self.tool_graph.add_tool(
                "semantic_code_search",
                inputs=["query"],
                outputs=["file_candidates"],
                cost_tier=CostTier.FREE,
            )
            registered += 1

            # File operations - FREE tier
            self.tool_graph.add_tool(
                "read_file",
                inputs=["file_candidates"],
                outputs=["file_contents"],
                cost_tier=CostTier.FREE,
            )
            registered += 1

            # Analysis tools - LOW tier
            self.tool_graph.add_tool(
                "analyze_docs",
                inputs=["file_contents"],
                outputs=["summary"],
                cost_tier=CostTier.LOW,
            )
            registered += 1

            self.tool_graph.add_tool(
                "code_review",
                inputs=["file_contents"],
                outputs=["summary"],
                cost_tier=CostTier.LOW,
            )
            registered += 1

            self.tool_graph.add_tool(
                "generate_docs",
                inputs=["file_contents"],
                outputs=["documentation"],
                cost_tier=CostTier.LOW,
            )
            registered += 1

            self.tool_graph.add_tool(
                "security_scan",
                inputs=["file_contents"],
                outputs=["security_report"],
                cost_tier=CostTier.LOW,
            )
            registered += 1

            self.tool_graph.add_tool(
                "analyze_metrics",
                inputs=["file_contents"],
                outputs=["metrics_report"],
                cost_tier=CostTier.LOW,
            )
            registered += 1

        except Exception as exc:
            logger.debug(f"Failed to register tool dependencies: {exc}")

        return registered

    def plan_tools(
        self, goals: List[str], available_inputs: Optional[List[str]] = None
    ) -> List[ToolDefinition]:
        """Plan a sequence of tools to satisfy goals using the dependency graph.

        Delegates to ToolGraphBuilder component (SRP compliance).

        Args:
            goals: List of goals to achieve
            available_inputs: Already available inputs

        Returns:
            List of ToolDefinition objects for the planned tools
        """
        if not goals or not self.tool_graph:
            return []

        graph_builder = self._get_graph_builder()
        return graph_builder.plan_for_goals(goals, available_inputs)

    def infer_goals_from_message(self, user_message: str) -> List[str]:
        """Infer planning goals from user request.

        Delegates to ToolGraphBuilder component (SRP compliance).

        Args:
            user_message: User's message

        Returns:
            List of inferred goal names
        """
        graph_builder = self._get_graph_builder()
        return graph_builder.infer_goals_from_message(user_message)

    def get_stats(self) -> RegistrationStats:
        """Get registration statistics.

        Returns:
            RegistrationStats with all counts
        """
        return self._stats

    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration for context injection.

        Returns:
            Dictionary with provider, model, and tool-specific settings
            that should be included in tool execution context.
        """
        return self._tool_config.copy()

    def get_plugin_info(self) -> Dict[str, Any]:
        """Get information about loaded plugins.

        Delegates to PluginLoader component (SRP compliance).

        Returns:
            Dictionary with plugin information
        """
        if self._plugin_loader:
            return self._plugin_loader.get_summary()

        # Legacy fallback
        if not self.plugin_manager:
            return {"plugins": [], "total": 0}

        return {
            "plugins": [
                {
                    "name": p.name,
                    "version": getattr(p, "version", "unknown"),
                    "tools": len(p.get_tools()) if hasattr(p, "get_tools") else 0,
                }
                for p in self.plugin_manager.loaded_plugins.values()
            ],
            "total": len(self.plugin_manager.loaded_plugins),
        }

    def get_mcp_info(self) -> Dict[str, Any]:
        """Get information about MCP servers.

        Delegates to MCPConnector component (SRP compliance).

        Returns:
            Dictionary with MCP server information
        """
        if self._mcp_connector:
            return self._mcp_connector.get_summary()

        # Legacy fallback
        if not self.mcp_registry:
            return {"servers": [], "connected": 0, "total": 0}

        servers = self.mcp_registry.list_servers()
        return {
            "servers": [
                {
                    "name": s.name,
                    "description": s.description,
                    "connected": s.name in getattr(self.mcp_registry, "_connected_servers", {}),
                }
                for s in servers
            ],
            "connected": self._stats.mcp_servers_connected,
            "total": len(servers),
        }

    async def shutdown(self) -> None:
        """Shutdown all components and cleanup.

        Delegates to specialized components (SRP compliance).
        """
        # Shutdown MCP connector if initialized
        if self._mcp_connector:
            try:
                await self._mcp_connector.shutdown()
            except Exception as e:
                logger.debug(f"Error shutting down MCP connector: {e}")

        # Legacy fallback: Cancel pending MCP tasks
        for task in self._mcp_tasks:
            if not task.done():
                task.cancel()

        # Legacy fallback: Shutdown MCP registry
        if self.mcp_registry and not self._mcp_connector:
            try:
                await self.mcp_registry.shutdown()
            except Exception as e:
                logger.debug(f"Error shutting down MCP registry: {e}")

        logger.debug("ToolRegistrar facade shutdown complete")

    # =========================================================================
    # Prewarm API (Phase 3: Scalability)
    # =========================================================================

    async def prewarm(
        self,
        include_plugins: bool = True,
        include_mcp: bool = True,
        timeout: float = 10.0,
    ) -> "PrewarmResult":
        """Prewarm tool catalogs to reduce cold-start latency.

        Delegates to specialized components (SRP compliance).

        This method pre-loads tools, configurations, and optionally starts
        MCP connections in the background. Call this at application startup
        or before the first user request to ensure fast tool access.

        Args:
            include_plugins: Whether to load plugins during prewarm
            include_mcp: Whether to start MCP connections (async, non-blocking)
            timeout: Maximum time to wait for prewarm (MCP may continue in background)

        Returns:
            PrewarmResult with timing and status information

        Example:
            # At application startup
            registrar = ToolRegistrar(tools, settings, provider, model)
            result = await registrar.prewarm()
            logger.debug(f"Prewarmed {result.tools_loaded} tools in {result.duration_ms}ms")
        """
        import time

        start_time = time.monotonic()
        result = PrewarmResult()

        try:
            # Phase 1: Pre-load dynamic tools via CatalogLoader
            if not self._tools_loaded:
                self._setup_providers()
                catalog_loader = self._get_catalog_loader()
                load_result = catalog_loader.load()
                self._stats.dynamic_tools = load_result.tools_loaded
                self._tools_loaded = True
                result.tools_loaded = self._stats.dynamic_tools
                logger.debug(f"Prewarmed {result.tools_loaded} dynamic tools via CatalogLoader")

            # Phase 2: Load plugins via PluginLoader
            if include_plugins and self.config.enable_plugins:
                try:
                    plugin_loader = self._get_plugin_loader()
                    load_result = plugin_loader.load()
                    result.plugins_loaded = load_result.tools_registered
                    self._stats.plugin_tools = load_result.tools_registered
                    self.plugin_manager = plugin_loader.plugin_manager
                    logger.debug(f"Prewarmed {result.plugins_loaded} plugin tools via PluginLoader")
                except Exception as e:
                    result.warnings.append(f"Plugin prewarm failed: {e}")
                    logger.debug(f"Plugin prewarm failed: {e}")

            # Phase 3: Start MCP connections via MCPConnector (async, non-blocking)
            if include_mcp and (
                self.config.enable_mcp or getattr(self.settings, "use_mcp_tools", False)
            ):
                try:
                    mcp_connector = self._get_mcp_connector()
                    # Start MCP in background task (don't block prewarm)
                    self._create_task(mcp_connector.connect(), "mcp_prewarm")
                    result.mcp_started = True
                    logger.debug("MCP prewarm started in background via MCPConnector")
                except Exception as e:
                    result.warnings.append(f"MCP prewarm failed to start: {e}")
                    logger.debug(f"MCP prewarm failed: {e}")

            # Phase 4: Pre-populate tool metadata registry cache
            try:
                from victor.tools.metadata_registry import ToolMetadataRegistry

                metadata_registry = ToolMetadataRegistry.get_instance()
                # Trigger cache population by querying common categories
                for category in ["core", "filesystem", "search", "git"]:
                    _ = metadata_registry.get_tools_by_category(category)
                result.metadata_cached = True
            except Exception as e:
                result.warnings.append(f"Metadata cache prewarm failed: {e}")

            result.success = True

        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.warning(f"Prewarm failed: {e}")

        result.duration_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            f"Prewarm complete: {result.tools_loaded} tools, "
            f"{result.plugins_loaded} plugins in {result.duration_ms:.1f}ms"
        )

        return result


@dataclass
class PrewarmResult:
    """Result of tool catalog prewarm operation.

    Attributes:
        success: Whether prewarm completed successfully
        tools_loaded: Number of dynamic tools loaded
        plugins_loaded: Number of plugin tools loaded
        mcp_started: Whether MCP prewarm was started (may still be running)
        metadata_cached: Whether tool metadata was cached
        duration_ms: Total prewarm duration in milliseconds
        error: Error message if prewarm failed
        warnings: Non-fatal warnings during prewarm
    """

    success: bool = False
    tools_loaded: int = 0
    plugins_loaded: int = 0
    mcp_started: bool = False
    metadata_cached: bool = False
    duration_ms: float = 0.0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "success": self.success,
            "tools_loaded": self.tools_loaded,
            "plugins_loaded": self.plugins_loaded,
            "mcp_started": self.mcp_started,
            "metadata_cached": self.metadata_cached,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "warnings": self.warnings,
        }
