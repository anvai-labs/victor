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

"""Tool plugin system for extensibility.

This module provides the base classes and interfaces for creating
tool plugins that can be dynamically loaded at runtime.

Example plugin structure:
    my_plugin/
    ├── __init__.py
    ├── plugin.py      # Contains Plugin class extending ToolPlugin
    └── tools/
        └── my_tool.py

Example plugin.py:
    from victor.tools.plugin import ToolPlugin
    from victor.tools.base import BaseTool

    class MyTool(BaseTool):
        # ... tool implementation

    class Plugin(ToolPlugin):
        name = "my_plugin"
        version = "1.0.0"
        description = "My custom tool plugin"

        def get_tools(self):
            return [MyTool()]
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from victor.tools.base import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata about a loaded plugin."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    homepage: str = ""
    dependencies: List[str] = field(default_factory=list)
    path: Optional[Path] = None
    enabled: bool = True


class ToolPlugin(ABC):
    """Base class for tool plugins.

    Plugins provide a way to extend Victor with custom tools that can be
    dynamically loaded at runtime. Each plugin can provide multiple tools
    and manage its own lifecycle.

    Attributes:
        name: Unique identifier for the plugin
        version: Semantic version string (e.g., "1.0.0")
        description: Human-readable description
        author: Plugin author
        homepage: URL for plugin documentation/source

    Example:
        class MyPlugin(ToolPlugin):
            name = "my_plugin"
            version = "1.0.0"
            description = "Adds custom tools for my workflow"

            def get_tools(self):
                return [MyCustomTool(), AnotherTool()]

            def initialize(self):
                # Setup database connections, load configs, etc.
                self.db = connect_to_database(self.config)

            def cleanup(self):
                # Close connections, cleanup resources
                self.db.close()
    """

    # Plugin metadata (override in subclass)
    name: str = "unnamed_plugin"
    version: str = "0.0.0"
    description: str = ""
    author: str = ""
    homepage: str = ""
    dependencies: List[str] = []

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin with optional configuration.

        Args:
            config: Plugin-specific configuration dictionary
        """
        self.config = config or {}
        self._initialized = False
        self._tools: List[BaseTool] = []

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Return list of tools provided by this plugin.

        This method must be implemented by each plugin to return
        the tools it provides.

        Returns:
            List of BaseTool instances
        """
        pass

    def initialize(self) -> None:  # noqa: B027
        """Initialize plugin resources.

        Called once when the plugin is loaded. Use this to:
        - Establish connections (databases, APIs, etc.)
        - Load configuration files
        - Initialize shared state

        Override this method in your plugin if needed.
        This is an optional lifecycle hook with empty default implementation.
        """
        pass

    def cleanup(self) -> None:  # noqa: B027
        """Cleanup plugin resources.

        Called when the plugin is unloaded. Use this to:
        - Close connections
        - Save state
        - Release resources

        Override this method in your plugin if needed.
        This is an optional lifecycle hook with empty default implementation.
        """
        pass

    def on_tool_registered(self, tool: BaseTool) -> None:  # noqa: B027
        """Called when a tool from this plugin is registered.

        Args:
            tool: The tool that was registered

        This is an optional lifecycle hook with empty default implementation.
        """
        pass

    def on_tool_executed(self, tool_name: str, success: bool, result: Any) -> None:  # noqa: B027
        """Called after a tool from this plugin is executed.

        Useful for logging, metrics, or plugin-specific handling.

        Args:
            tool_name: Name of the executed tool
            success: Whether execution succeeded
            result: Tool execution result

        This is an optional lifecycle hook with empty default implementation.
        """
        pass

    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata.

        Returns:
            PluginMetadata instance with plugin information
        """
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            author=self.author,
            homepage=self.homepage,
            dependencies=self.dependencies,
            enabled=True,
        )

    def validate_config(self) -> List[str]:
        """Validate plugin configuration.

        Override this to check required config values.

        Returns:
            List of validation error messages (empty if valid)
        """
        return []

    def _do_initialize(self) -> None:
        """Internal initialization wrapper."""
        if self._initialized:
            return

        errors = self.validate_config()
        if errors:
            raise ValueError(f"Plugin {self.name} config validation failed: {errors}")

        self.initialize()
        self._tools = self.get_tools()
        self._initialized = True
        logger.info(
            f"Plugin '{self.name}' v{self.version} initialized with {len(self._tools)} tools"
        )

    def _do_cleanup(self) -> None:
        """Internal cleanup wrapper."""
        if not self._initialized:
            return

        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"Plugin '{self.name}' cleanup failed: {e}")
        finally:
            self._initialized = False
            self._tools = []


class FunctionToolPlugin(ToolPlugin):
    """A simple plugin that wraps standalone tool functions.

    Use this when you have @tool decorated functions that you want
    to package as a plugin without creating a full plugin class.

    Example:
        from victor.tools.decorators import tool

        @tool
        def my_custom_tool(arg1: str) -> str:
            '''My custom tool.'''
            return f"Result: {arg1}"

        # Create plugin from functions
        plugin = FunctionToolPlugin(
            name="my_functions",
            version="1.0.0",
            tool_functions=[my_custom_tool]
        )
    """

    def __init__(
        self,
        name: str,
        version: str,
        tool_functions: List[Any],
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize function-based plugin.

        Args:
            name: Plugin name
            version: Plugin version
            tool_functions: List of @tool decorated functions
            description: Plugin description
            config: Plugin configuration
        """
        super().__init__(config)
        self.name = name
        self.version = version
        self.description = description
        self._tool_functions = tool_functions

    def get_tools(self) -> List[BaseTool]:
        """Convert @tool decorated functions to BaseTool instances."""
        tools = []
        for func in self._tool_functions:
            if hasattr(func, "Tool"):
                # @tool decorator creates a .Tool attribute
                tools.append(func.Tool)
            elif hasattr(func, "_is_tool"):
                # Legacy check
                tools.append(func.Tool)
            else:
                logger.warning(f"Function {func.__name__} is not a @tool decorated function")
        return tools
