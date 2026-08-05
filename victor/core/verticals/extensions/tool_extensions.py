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

"""Tool Extensions - ISP-compliant composite for tool-related protocols.

This module provides a focused extension for tool-related vertical capabilities:
- Middleware for pre/post tool call processing
- Tool dependencies for execution ordering

This replaces the tool-related parts of the monolithic VerticalExtensions class,
following Interface Segregation Principle (ISP).

Usage:
    from victor.core.verticals.extensions import ToolExtensions
    from victor.core.verticals.protocols import MiddlewareProtocol

    class CodeValidationMiddleware(MiddlewareProtocol):
        async def before_tool_call(self, tool_name: str, arguments: dict):
            # Validate code before writing
            ...

    tool_ext = ToolExtensions(
        middleware=[CodeValidationMiddleware()],
        tool_dependencies=[
            ToolDependency(tool="edit", depends_on=["read"]),
            ToolDependency(tool="write", depends_on=["read"]),
        ],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from victor.core.tool_types import ToolDependency
from victor.core.vertical_types import MiddlewarePriority, MiddlewareResult


@dataclass
class ToolExtensions:
    """Focused extension for tool-related vertical capabilities.

    Groups middleware and tool dependencies - the tool-specific parts
    that were previously bundled in VerticalExtensions.

    Attributes:
        middleware: List of middleware implementations for tool call processing.
            Middleware is executed in priority order (HIGH, NORMAL, LOW).
        tool_dependencies: List of tool dependencies for execution ordering.
            E.g., "edit" depends on "read" to ensure file is read first.

    Example:
        tool_ext = ToolExtensions(
            middleware=[
                SyntaxValidationMiddleware(),
                SecurityCheckMiddleware(),
            ],
            tool_dependencies=[
                ToolDependency("edit", ["read"]),
                ToolDependency("git_commit", ["git_add"]),
            ],
        )

        # Get middleware sorted by priority
        sorted_mw = tool_ext.get_sorted_middleware()

        # Check dependencies for a tool
        deps = tool_ext.get_dependencies_for("edit")  # Returns ["read"]
    """

    middleware: List[Any] = field(default_factory=list)  # List[MiddlewareProtocol]
    tool_dependencies: List[ToolDependency] = field(default_factory=list)

    def get_sorted_middleware(self) -> List[Any]:
        """Get middleware sorted by priority (HIGH first, then NORMAL, then LOW).

        Returns:
            List of middleware implementations sorted by priority
        """
        # Sort by priority - lower enum value = higher priority
        return sorted(
            self.middleware,
            key=lambda m: m.get_priority().value if hasattr(m, "get_priority") else 50,
        )

    def get_dependencies_for(self, tool_name: str) -> List[str]:
        """Get dependency tools for a given tool.

        Args:
            tool_name: The tool to get dependencies for

        Returns:
            List of tool names that must run before this tool
        """
        for dep in self.tool_dependencies:
            # Handle both 'tool' and 'tool_name' attributes for compatibility
            dep_tool = getattr(dep, "tool_name", getattr(dep, "tool", None))
            if dep_tool == tool_name:
                return list(dep.depends_on) if isinstance(dep.depends_on, set) else dep.depends_on
        return []

    def has_dependency(self, tool_name: str, dependency: str) -> bool:
        """Check if a tool depends on another tool.

        Args:
            tool_name: The tool to check
            dependency: The potential dependency tool

        Returns:
            True if tool_name depends on dependency
        """
        deps = self.get_dependencies_for(tool_name)
        return dependency in deps

    def get_all_dependency_tools(self) -> Set[str]:
        """Get all tools that are dependencies of other tools.

        Returns:
            Set of tool names that are dependencies
        """
        all_deps: Set[str] = set()
        for dep in self.tool_dependencies:
            if isinstance(dep.depends_on, set):
                all_deps.update(dep.depends_on)
            else:
                all_deps.update(dep.depends_on)
        return all_deps

    def get_middleware_for_tool(self, tool_name: str) -> List[Any]:
        """Get middleware applicable to a specific tool.

        Args:
            tool_name: The tool to get middleware for

        Returns:
            List of middleware that applies to this tool
        """
        applicable = []
        for mw in self.middleware:
            if hasattr(mw, "get_applicable_tools"):
                tools = mw.get_applicable_tools()
                if tools is None or tool_name in tools:
                    applicable.append(mw)
            else:
                # No filter method means applies to all
                applicable.append(mw)
        return applicable

    def merge(self, other: "ToolExtensions") -> "ToolExtensions":
        """Merge with another ToolExtensions instance.

        Middleware is concatenated. Dependencies from other override
        same-tool dependencies in self.

        Args:
            other: Another ToolExtensions to merge from

        Returns:
            New ToolExtensions with merged content
        """
        # Merge middleware (deduplicate by instance)
        seen_mw = set(id(m) for m in self.middleware)
        merged_mw = list(self.middleware)
        for mw in other.middleware:
            if id(mw) not in seen_mw:
                merged_mw.append(mw)
                seen_mw.add(id(mw))

        # Merge dependencies (other overrides same-tool entries)
        # Handle both 'tool' and 'tool_name' attributes for compatibility
        dep_map: Dict[str, ToolDependency] = {}
        for dep in self.tool_dependencies:
            dep_tool = getattr(dep, "tool_name", getattr(dep, "tool", None))
            if dep_tool:
                dep_map[dep_tool] = dep
        for dep in other.tool_dependencies:
            dep_tool = getattr(dep, "tool_name", getattr(dep, "tool", None))
            if dep_tool:
                dep_map[dep_tool] = dep

        return ToolExtensions(
            middleware=merged_mw,
            tool_dependencies=list(dep_map.values()),
        )

    def __bool__(self) -> bool:
        """Return True if any content is present."""
        return bool(self.middleware or self.tool_dependencies)


__all__ = ["ToolExtensions"]
