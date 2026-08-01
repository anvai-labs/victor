"""Tool registration strategies.

Defines the strategy pattern for extensible tool registration.
Each strategy handles a specific type of tool registration.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolRegistrationStrategy(Protocol):
    """Protocol for tool registration strategies.

    Enables Open/Closed Principle: new tool types can be added
    without modifying existing code.

    Each strategy:
    1. Checks if it can handle a given tool type
    2. Registers the tool using the appropriate method
    3. Has a priority to determine selection order

    Example:
        class MyCustomStrategy:
            def can_handle(self, tool: Any) -> bool:
                return isinstance(tool, MyCustomType)

            def register(self, registry, tool, enabled=True):
                wrapper = self._create_wrapper(tool)
                registry._register_direct(wrapper.name, wrapper, enabled)

            @property
            def priority(self) -> int:
                return 50  # Medium priority
    """

    def can_handle(self, tool: Any) -> bool:
        """Check if this strategy can handle the tool.

        Args:
            tool: Tool object to check

        Returns:
            True if this strategy can register the tool
        """
        ...

    def register(
        self,
        registry: Any,  # ToolRegistry
        tool: Any,
        enabled: bool = True,
    ) -> None:
        """Register the tool using this strategy.

        Args:
            registry: Tool registry to register with
            tool: Tool object to register
            enabled: Whether the tool is enabled by default
        """
        ...

    @property
    def priority(self) -> int:
        """Priority for strategy selection (higher = checked first).

        Returns:
            Priority value (higher values are checked first)
        """
        ...


class FunctionDecoratorStrategy:
    """Strategy for function-decorated tools (@tool decorator).

    Handles tools created using the @tool decorator:

        @tool
        def my_tool(param: str) -> str:
            return param

    Priority: 100 (highest - checked first)
    """

    def can_handle(self, tool: Any) -> bool:
        """Check if tool is a function-decorated tool.

        Args:
            tool: Tool object to check

        Returns:
            True if tool has _is_victor_tool or Tool attribute (from @tool decorator)
        """
        return callable(tool) and (hasattr(tool, "_is_victor_tool") or hasattr(tool, "Tool"))

    def register(
        self,
        registry: Any,
        tool: Any,
        enabled: bool = True,
    ) -> None:
        """Register a function-decorated tool.

        Args:
            registry: Tool registry to register with
            tool: Decorated function
            enabled: Whether tool is enabled
        """
        tool_instance = tool.Tool
        registry._register_direct(tool_instance.name, tool_instance, enabled)
        logger.debug(f"Registered function-decorated tool: {tool_instance.name}")

    @property
    def priority(self) -> int:
        """High priority - checked first."""
        return 100


class BaseToolSubclassStrategy:
    """Strategy for BaseTool subclass instances.

    Handles tools that are instances of BaseTool subclasses:

        class MyTool(BaseTool):
            @property
            def name(self):
                return "my_tool"

        tool = MyTool()
        registry.register(tool)

    Priority: 50 (medium)
    """

    def can_handle(self, tool: Any) -> bool:
        """Check if tool is a BaseTool subclass instance.

        Args:
            tool: Tool object to check

        Returns:
            True if tool is a BaseTool instance
        """
        try:
            from victor.tools.base import BaseTool

            return isinstance(tool, BaseTool)
        except ImportError:
            return False

    def register(
        self,
        registry: Any,
        tool: Any,
        enabled: bool = True,
    ) -> None:
        """Register a BaseTool subclass instance.

        Args:
            registry: Tool registry to register with
            tool: BaseTool instance
            enabled: Whether tool is enabled
        """
        registry._register_direct(tool.name, tool, enabled)
        logger.debug(f"Registered BaseTool subclass: {tool.name}")

    @property
    def priority(self) -> int:
        """Medium priority."""
        return 50


class MCPDictStrategy:
    """Strategy for MCP dictionary-based tools.

    Handles tools defined as dictionaries (from MCP servers):

        {
            "name": "tool_name",
            "description": "Tool description",
            "parameters": {...}
        }

    Priority: 10 (lowest - checked last)
    """

    def can_handle(self, tool: Any) -> bool:
        """Check if tool is an MCP dictionary.

        Args:
            tool: Tool object to check

        Returns:
            True if tool is a dict with 'name' key
        """
        return isinstance(tool, dict) and "name" in tool

    def register(
        self,
        registry: Any,
        tool: Any,
        enabled: bool = True,
    ) -> None:
        """Register an MCP dictionary tool.

        Args:
            registry: Tool registry to register with
            tool: Dictionary tool definition
            enabled: Whether tool is enabled
        """
        registry.register_dict(tool, enabled)
        logger.debug(f"Registered MCP dict tool: {tool.get('name')}")

    @property
    def priority(self) -> int:
        """Low priority - checked last."""
        return 10


class DuckTypedToolStrategy:
    """Strategy for duck-typed tool instances from plugins.

    Contract-only packages (authored against ``victor_contracts``) cannot
    subclass the runtime ``BaseTool`` or use the ``@tool`` decorator, so they
    ship tool instances that expose ``name``, ``description``, ``parameters``,
    and an async ``execute(**kwargs)``:

        class MyPluginTool:
            @property
            def name(self) -> str: ...
            async def execute(self, **kwargs): ...

        context.register_tool(MyPluginTool())

    The legacy (non-strategy) registration path in ``ToolRegistry.register``
    already accepts this shape ("duck-typed tool with name + execute"); this
    strategy provides parity when strategy-based registration is enabled.

    Priority: 20 (after BaseTool subclasses, before MCP dicts)
    """

    def can_handle(self, tool: Any) -> bool:
        """Check if tool is a duck-typed tool instance.

        Args:
            tool: Tool object to check

        Returns:
            True if tool exposes a string ``name`` and callable ``execute``
        """
        if isinstance(tool, dict):
            return False
        name = getattr(tool, "name", None)
        return isinstance(name, str) and bool(name) and callable(getattr(tool, "execute", None))

    def register(
        self,
        registry: Any,
        tool: Any,
        enabled: bool = True,
    ) -> None:
        """Register a duck-typed tool instance.

        Args:
            registry: Tool registry to register with
            tool: Duck-typed tool instance
            enabled: Whether tool is enabled
        """
        registry._register_direct(tool.name, tool, enabled)
        logger.debug(f"Registered duck-typed tool: {tool.name}")

    @property
    def priority(self) -> int:
        """Low-medium priority - after BaseTool subclasses."""
        return 20


__all__ = [
    "ToolRegistrationStrategy",
    "FunctionDecoratorStrategy",
    "BaseToolSubclassStrategy",
    "DuckTypedToolStrategy",
    "MCPDictStrategy",
]
