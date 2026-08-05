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

"""Plugin loader for tool plugin management.

This module handles plugin discovery, loading, and tool registration,
extracted from ToolRegistrar as part of SRP compliance refactoring.

Single Responsibility: Manage tool plugins lifecycle.

Design Pattern: Registry Pattern
- Discovers plugins from configured directories
- Loads plugins from Python packages
- Registers plugin tools with ToolRegistry

Usage:
    from victor.agent.plugin_loader import PluginLoader

    loader = PluginLoader(
        registry=tool_registry,
        settings=settings,
        config=PluginLoaderConfig(
            plugin_dirs=["/path/to/plugins"],
            disabled_plugins={"legacy_plugin"},
        ),
    )
    result = loader.load()
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from victor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class PluginLoaderConfig:
    """Configuration for plugin loading.

    Attributes:
        enabled: Whether plugin system is enabled
        plugin_dirs: Additional plugin directories to search
        disabled_plugins: Plugin names to skip loading
        plugin_packages: Python package names to load plugins from
    """

    enabled: bool = True
    plugin_dirs: List[str] = field(default_factory=list)
    disabled_plugins: Set[str] = field(default_factory=set)
    plugin_packages: List[str] = field(default_factory=list)


@dataclass
class PluginLoadResult:
    """Result of plugin loading.

    Attributes:
        plugins_loaded: Number of plugins successfully loaded
        tools_registered: Number of plugin tools registered
        errors: List of errors encountered during loading
    """

    plugins_loaded: int = 0
    tools_registered: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class PluginInfo:
    """Information about a loaded plugin.

    Attributes:
        name: Plugin name
        version: Plugin version string
        tool_count: Number of tools provided by plugin
    """

    name: str
    version: str = "unknown"
    tool_count: int = 0


class PluginLoader:
    """Loads and manages tool plugins.

    Single Responsibility: Plugin discovery, loading, and tool registration.

    This class handles:
    - Discovering plugins from directories
    - Loading plugins from Python packages
    - Registering plugin tools with ToolRegistry
    - Managing plugin lifecycle

    Extracted from ToolRegistrar for SRP compliance.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        settings: Any,
        config: Optional[PluginLoaderConfig] = None,
    ):
        """Initialize the plugin loader.

        Args:
            registry: Tool registry to register plugin tools with
            settings: Application settings for plugin configuration
            config: Optional plugin loader configuration
        """
        self._registry = registry
        self._settings = settings
        self._config = config or PluginLoaderConfig()
        self._plugin_manager: Optional[Any] = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if plugins have been loaded."""
        return self._loaded

    @property
    def plugin_manager(self) -> Optional[Any]:
        """Get the underlying plugin manager."""
        return self._plugin_manager

    def load(self) -> PluginLoadResult:
        """Load plugins and register their tools.

        This method:
        1. Initializes the plugin registry
        2. Discovers plugins from directories
        3. Loads plugins from packages
        4. Registers all plugin tools

        Returns:
            PluginLoadResult with loading statistics
        """
        result = PluginLoadResult()

        if not self._config.enabled:
            logger.debug("Plugin system disabled by configuration")
            self._loaded = True
            return result

        try:
            from victor.tools.plugin_registry import ToolPluginRegistry
            from victor.config.settings import get_project_paths

            # Use centralized path for plugins directory
            plugin_dirs = [get_project_paths().global_plugins_dir]
            plugin_dirs.extend(self._config.plugin_dirs)

            plugin_config = getattr(self._settings, "plugin_config", {})

            self._plugin_manager = ToolPluginRegistry(
                plugin_dirs=plugin_dirs,
                config=plugin_config,
            )

            # Disable specified plugins
            for plugin_name in self._config.disabled_plugins:
                self._plugin_manager.disable_plugin(plugin_name)

            # Discover and load plugins from directories
            loaded_count = self._plugin_manager.discover_and_load()

            # Load plugins from packages
            for package_name in self._config.plugin_packages:
                plugin = self._plugin_manager.load_plugin_from_package(package_name)
                if plugin:
                    self._plugin_manager.register_plugin(plugin)

            # Register plugin tools
            tool_count = 0
            if loaded_count > 0 or self._plugin_manager.loaded_plugins:
                tool_count = self._plugin_manager.register_tools(self._registry)

            result.plugins_loaded = len(self._plugin_manager.loaded_plugins)
            result.tools_registered = tool_count

            logger.info(
                f"PluginLoader: {result.plugins_loaded} plugins, "
                f"{result.tools_registered} tools"
            )

        except Exception as e:
            result.errors.append(str(e))
            logger.warning(f"Failed to initialize plugin system: {e}")
            self._plugin_manager = None

        self._loaded = True
        return result

    def get_plugin_info(self) -> List[PluginInfo]:
        """Get information about loaded plugins.

        Returns:
            List of PluginInfo for each loaded plugin
        """
        if not self._plugin_manager:
            return []

        return [
            PluginInfo(
                name=p.name,
                version=getattr(p, "version", "unknown"),
                tool_count=len(p.get_tools()) if hasattr(p, "get_tools") else 0,
            )
            for p in self._plugin_manager.loaded_plugins.values()
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary information about plugins.

        Returns:
            Dictionary with plugin summary information
        """
        plugins = self.get_plugin_info()
        return {
            "plugins": [
                {"name": p.name, "version": p.version, "tools": p.tool_count} for p in plugins
            ],
            "total": len(plugins),
        }


__all__ = ["PluginLoader", "PluginLoaderConfig", "PluginLoadResult", "PluginInfo"]
