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

"""Lazy tool composition utilities for efficient tool loading.

This module provides lazy-loading wrappers for tool composition, enabling
deferred initialization of expensive tools until their first use.

Key classes:
- LazyToolRunnable: Wraps a tool factory to defer instantiation
- ToolCompositionBuilder: Builder pattern for composing multiple lazy tools

Example:
    >>> from victor.tools.composition.lazy import LazyToolRunnable
    >>>
    >>> lazy = LazyToolRunnable(lambda: ExpensiveTool())
    >>> # Tool not created yet
    >>> result = lazy.run({"input": "test"})  # Now created and cached
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class LazyToolRunnable:
    """Lazy-loading wrapper for tool composition.

    Defers tool initialization until first use, reducing startup time
    for applications with many tools that may not all be needed.

    The underlying tool is created on first access to the `tool` property
    or when `run()`/`arun()` is called. Once created, the instance is
    cached (unless caching is disabled) for subsequent calls.

    Attributes:
        name: The name of the lazy tool (from factory or explicit)

    Example:
        >>> lazy = LazyToolRunnable(lambda: ExpensiveTool())
        >>> # Tool not created yet - no memory/resources used
        >>> result = lazy.run({"input": "test"})  # Now created
        >>> result2 = lazy.run({"input": "test2"})  # Uses cached instance
        >>>
        >>> # With explicit name
        >>> lazy = LazyToolRunnable(lambda: SearchTool(), name="semantic_search")
        >>> print(lazy.name)  # "semantic_search"
        >>>
        >>> # Disable caching for tools that need fresh instances
        >>> lazy = LazyToolRunnable(lambda: StatefulTool(), cache=False)
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        name: Optional[str] = None,
        cache: bool = True,
    ) -> None:
        """Initialize a lazy tool wrapper.

        Args:
            factory: A callable that creates the tool instance when invoked.
                     Should take no arguments and return a tool-like object.
            name: Optional name for the lazy tool. If not provided, the
                  factory's __name__ attribute is used (or "anonymous").
            cache: Whether to cache the created instance. Defaults to True.
                   Set to False if each call should create a new instance.
        """
        self._factory = factory
        self._name = name
        self._cache = cache
        self._instance: Optional[Any] = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Get the name of this lazy tool.

        Returns:
            The explicit name if provided, otherwise the factory's __name__,
            or "anonymous" if the factory has no name.
        """
        if self._name:
            return self._name
        return getattr(self._factory, "__name__", "anonymous")

    @cached_property
    def tool(self) -> Any:
        """Get the underlying tool, creating it if necessary.

        This property uses functools.cached_property for efficient caching.
        The tool is created on first access and cached for subsequent calls.

        Note: If cache=False was passed to __init__, this still caches due
        to cached_property semantics. For non-cached behavior, use
        get_tool_instance() directly.

        Returns:
            The initialized tool instance.
        """
        if not self._initialized:
            logger.debug(f"Lazy-initializing tool: {self.name}")
            self._instance = self._factory()
            self._initialized = True
        return self._instance

    def get_tool_instance(self) -> Any:
        """Get a tool instance, respecting the cache setting.

        Unlike the `tool` property which always caches via cached_property,
        this method respects the cache parameter passed to __init__.

        Returns:
            The tool instance (cached or fresh based on cache setting).
        """
        if not self._cache:
            logger.debug(f"Creating fresh instance of tool: {self.name}")
            return self._factory()

        return self.tool

    def run(self, inputs: Dict[str, Any]) -> Any:
        """Execute the tool synchronously with the given inputs.

        This method creates the tool if not yet initialized (or if
        caching is disabled), then calls its `run` method.

        Args:
            inputs: Dictionary of inputs to pass to the tool's run method.

        Returns:
            The result from the tool's run method.

        Raises:
            AttributeError: If the underlying tool doesn't have a run method.
        """
        tool_instance = self.get_tool_instance()
        return tool_instance.run(inputs)

    async def arun(self, inputs: Dict[str, Any]) -> Any:
        """Execute the tool asynchronously with the given inputs.

        This method creates the tool if not yet initialized (or if
        caching is disabled), then calls its `arun` method.

        Args:
            inputs: Dictionary of inputs to pass to the tool's arun method.

        Returns:
            The result from the tool's arun method.

        Raises:
            AttributeError: If the underlying tool doesn't have an arun method.
        """
        tool_instance = self.get_tool_instance()
        return await tool_instance.arun(inputs)

    def reset(self) -> None:
        """Reset the lazy tool, clearing the cached instance.

        After calling reset(), the next access to the tool will create
        a new instance from the factory. This is useful for:
        - Releasing resources held by the tool
        - Testing tool initialization behavior
        - Forcing re-initialization after configuration changes

        Example:
            >>> lazy = LazyToolRunnable(lambda: ExpensiveTool())
            >>> _ = lazy.tool  # Creates instance
            >>> lazy.reset()   # Clears cached instance
            >>> _ = lazy.tool  # Creates new instance
        """
        self._instance = None
        self._initialized = False
        # Clear the cached_property cache by deleting from __dict__
        if "tool" in self.__dict__:
            del self.__dict__["tool"]

    @property
    def is_initialized(self) -> bool:
        """Check if the tool has been initialized.

        Returns:
            True if the tool instance has been created, False otherwise.
        """
        return self._initialized

    def __repr__(self) -> str:
        """Return a string representation of the lazy tool."""
        status = "initialized" if self._initialized else "pending"
        return f"LazyToolRunnable(name={self.name!r}, status={status}, cache={self._cache})"


class ToolCompositionBuilder:
    """Builder for composing multiple tools with optional lazy loading.

    This builder provides a fluent interface for assembling a collection
    of tools, with the option to make each tool lazy-loaded or eagerly
    initialized.

    Example:
        >>> builder = ToolCompositionBuilder()
        >>> tools = (
        ...     builder
        ...     .add("search", lambda: SearchTool(), lazy=True)
        ...     .add("analyze", lambda: AnalyzeTool(), lazy=True)
        ...     .add("format", FormatTool(), lazy=False)  # Eager
        ...     .build()
        ... )
        >>> # tools["search"] is a LazyToolRunnable
        >>> # tools["format"] is the FormatTool instance directly
    """

    def __init__(self) -> None:
        """Initialize an empty tool composition builder."""
        self._tools: Dict[str, Any] = {}

    def add(
        self,
        name: str,
        factory: Callable[[], Any],
        lazy: bool = True,
    ) -> "ToolCompositionBuilder":
        """Add a tool to the composition.

        Args:
            name: The name to use for this tool in the composition.
            factory: A callable that creates the tool instance, or
                     the tool instance itself if lazy=False.
            lazy: Whether to wrap the tool in LazyToolRunnable.
                  If True (default), the factory is wrapped and the tool
                  is not created until first use. If False, the factory
                  is called immediately (or if factory is already an
                  instance, it's used directly).

        Returns:
            self, for method chaining.

        Example:
            >>> builder = ToolCompositionBuilder()
            >>> builder.add("search", lambda: SearchTool())  # Lazy by default
            >>> builder.add("format", FormatTool(), lazy=False)  # Eager
        """
        if lazy:
            self._tools[name] = LazyToolRunnable(factory, name=name)
        else:
            # If not lazy, call the factory if it's callable, otherwise use directly
            if callable(factory):
                self._tools[name] = factory()
            else:
                self._tools[name] = factory
        return self

    def add_lazy(
        self,
        name: str,
        factory: Callable[[], Any],
        cache: bool = True,
    ) -> "ToolCompositionBuilder":
        """Add a lazy-loaded tool with explicit cache control.

        This is a convenience method for adding lazy tools with
        custom caching behavior.

        Args:
            name: The name to use for this tool.
            factory: A callable that creates the tool instance.
            cache: Whether to cache the tool instance after creation.

        Returns:
            self, for method chaining.
        """
        self._tools[name] = LazyToolRunnable(factory, name=name, cache=cache)
        return self

    def add_eager(self, name: str, tool: Any) -> "ToolCompositionBuilder":
        """Add an eagerly-initialized tool instance.

        This is a convenience method for adding pre-created tool instances.

        Args:
            name: The name to use for this tool.
            tool: The tool instance to add.

        Returns:
            self, for method chaining.
        """
        self._tools[name] = tool
        return self

    def remove(self, name: str) -> "ToolCompositionBuilder":
        """Remove a tool from the composition.

        Args:
            name: The name of the tool to remove.

        Returns:
            self, for method chaining.

        Raises:
            KeyError: If the tool name is not in the composition.
        """
        del self._tools[name]
        return self

    def has(self, name: str) -> bool:
        """Check if a tool with the given name exists in the composition.

        Args:
            name: The tool name to check.

        Returns:
            True if the tool exists, False otherwise.
        """
        return name in self._tools

    def build(self) -> Dict[str, Any]:
        """Build and return the tool composition.

        Returns:
            A dictionary mapping tool names to tool instances
            (or LazyToolRunnable wrappers for lazy tools).
        """
        return dict(self._tools)

    def clear(self) -> "ToolCompositionBuilder":
        """Clear all tools from the builder.

        Returns:
            self, for method chaining.
        """
        self._tools.clear()
        return self

    def __len__(self) -> int:
        """Return the number of tools in the composition."""
        return len(self._tools)

    def __repr__(self) -> str:
        """Return a string representation of the builder."""
        tool_names = list(self._tools.keys())
        return f"ToolCompositionBuilder(tools={tool_names})"
