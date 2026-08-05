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

"""Extension dependency graph for verticals.

This module provides dependency resolution for verticals using a directed
acyclic graph (DAG) and Kahn's algorithm for topological sorting.

Design Principles:
    - Detect circular dependencies before activation
    - Resolve load order based on dependencies and priorities
    - Graceful handling of missing optional dependencies
    - Clear error messages for dependency conflicts
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from victor_contracts.verticals.manifest import ExtensionManifest

logger = logging.getLogger(__name__)


class DependencyCycleError(Exception):
    """Raised when a circular dependency is detected.

    Attributes:
        cycle: List of vertical names forming the cycle
        message: Human-readable error message
    """

    def __init__(self, cycle: List[str], message: str = "") -> None:
        self.cycle = cycle
        self.message = message or f"Circular dependency detected: {' -> '.join(cycle + [cycle[0]])}"
        super().__init__(self.message)


@dataclass
class DependencyNode:
    """Node in the dependency graph.

    Attributes:
        vertical_name: Name of the vertical
        version: Version of the vertical
        manifest: ExtensionManifest for the vertical
        dependencies: Set of vertical names this node depends on
        dependents: Set of vertical names that depend on this node
        load_priority: Higher values load first (default: 0)
        loaded: Whether this vertical has been loaded
    """

    vertical_name: str
    version: str
    manifest: Optional[ExtensionManifest] = None
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    load_priority: int = 0
    loaded: bool = False

    def __hash__(self) -> int:
        return hash(self.vertical_name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DependencyNode):
            return False
        return self.vertical_name == other.vertical_name


@dataclass
class LoadOrder:
    """Result of dependency resolution.

    Attributes:
        order: List of vertical names in load order
        missing_dependencies: Set of required dependencies that are missing
        missing_optional: Set of optional dependencies that are missing
        cycles: List of detected cycles (if any)
    """

    order: List[str]
    missing_dependencies: Set[str] = field(default_factory=set)
    missing_optional: Set[str] = field(default_factory=set)
    cycles: List[List[str]] = field(default_factory=list)

    @property
    def can_load(self) -> bool:
        """Check if all required dependencies are satisfied."""
        return len(self.missing_dependencies) == 0 and len(self.cycles) == 0

    @property
    def has_cycles(self) -> bool:
        """Check if circular dependencies were detected."""
        return len(self.cycles) > 0


class ExtensionDependencyGraph:
    """Dependency graph for vertical extensions.

    This class manages dependencies between verticals and resolves the
    correct load order using topological sorting (Kahn's algorithm).

    Example:
        >>> graph = ExtensionDependencyGraph()
        >>> graph.add_vertical("rag", "1.0.0", manifest)
        >>> graph.add_dependency("rag", "coding", required=True)
        >>> order = graph.resolve_load_order()
        >>> print(order.order)  # ['coding', 'rag']
    """

    def __init__(self) -> None:
        """Initialize the dependency graph."""
        self._nodes: Dict[str, DependencyNode] = {}
        self._lock = threading.RLock()

    def add_vertical(
        self,
        vertical_name: str,
        version: str,
        manifest: Optional[ExtensionManifest] = None,
        load_priority: int = 0,
    ) -> None:
        """Add a vertical to the graph.

        Args:
            vertical_name: Name of the vertical
            version: Version of the vertical
            manifest: ExtensionManifest (optional)
            load_priority: Load priority (higher loads first)
        """
        with self._lock:
            if vertical_name not in self._nodes:
                self._nodes[vertical_name] = DependencyNode(
                    vertical_name=vertical_name,
                    version=version,
                    manifest=manifest,
                    load_priority=load_priority,
                )
                logger.debug(f"Added vertical '{vertical_name}' to dependency graph")
            else:
                # Update existing node
                node = self._nodes[vertical_name]
                node.version = version
                node.manifest = manifest
                node.load_priority = load_priority

    def remove_vertical(self, vertical_name: str) -> None:
        """Remove a vertical from the graph.

        Args:
            vertical_name: Name of the vertical to remove
        """
        with self._lock:
            if vertical_name in self._nodes:
                # Remove from dependencies of other nodes
                node = self._nodes[vertical_name]

                # Remove from dependents of dependencies
                for dep_name in node.dependencies:
                    if dep_name in self._nodes:
                        self._nodes[dep_name].dependents.discard(vertical_name)

                # Remove from dependencies of dependents
                for dependent_name in node.dependents:
                    if dependent_name in self._nodes:
                        self._nodes[dependent_name].dependencies.discard(vertical_name)

                del self._nodes[vertical_name]
                logger.debug(f"Removed vertical '{vertical_name}' from dependency graph")

    def add_dependency(
        self,
        vertical_name: str,
        dependency_name: str,
        required: bool = True,
    ) -> None:
        """Add a dependency relationship.

        Args:
            vertical_name: Name of the vertical that has the dependency
            dependency_name: Name of the vertical being depended on
            required: Whether the dependency is required (True) or optional (False)

        Raises:
            ValueError: If vertical_name or dependency_name not in graph
        """
        with self._lock:
            if vertical_name not in self._nodes:
                raise ValueError(f"Vertical '{vertical_name}' not in graph")
            if dependency_name not in self._nodes:
                if required:
                    raise ValueError(f"Required dependency '{dependency_name}' not in graph")
                # Optional dependency - track it but don't fail
                logger.debug(
                    f"Optional dependency '{dependency_name}' not in graph for '{vertical_name}'"
                )
                return

            # Add dependency relationship
            self._nodes[vertical_name].dependencies.add(dependency_name)
            self._nodes[dependency_name].dependents.add(vertical_name)

            logger.debug(
                f"Added dependency: '{vertical_name}' -> '{dependency_name}' (required={required})"
            )

    def get_dependencies(self, vertical_name: str) -> Set[str]:
        """Get dependencies for a vertical.

        Args:
            vertical_name: Name of the vertical

        Returns:
            Set of dependency names

        Raises:
            ValueError: If vertical_name not in graph
        """
        with self._lock:
            if vertical_name not in self._nodes:
                raise ValueError(f"Vertical '{vertical_name}' not in graph")
            return self._nodes[vertical_name].dependencies.copy()

    def get_dependents(self, vertical_name: str) -> Set[str]:
        """Get dependents of a vertical.

        Args:
            vertical_name: Name of the vertical

        Returns:
            Set of dependent names

        Raises:
            ValueError: If vertical_name not in graph
        """
        with self._lock:
            if vertical_name not in self._nodes:
                raise ValueError(f"Vertical '{vertical_name}' not in graph")
            return self._nodes[vertical_name].dependents.copy()

    def resolve_load_order(self) -> LoadOrder:
        """Resolve load order using topological sort (Kahn's algorithm).

        Returns:
            LoadOrder with ordered list of vertical names

        Raises:
            DependencyCycleError: If circular dependency detected
        """
        with self._lock:
            # Check for cycles first
            cycles = self._detect_cycles()
            if cycles:
                return LoadOrder(
                    order=[],
                    cycles=cycles,
                )

            # Kahn's algorithm for topological sort
            # Start with nodes that have no dependencies
            available: List[DependencyNode] = []
            in_degree: Dict[str, int] = {}

            # Calculate in-degree for each node
            for node in self._nodes.values():
                in_degree[node.vertical_name] = len(node.dependencies)
                if in_degree[node.vertical_name] == 0:
                    available.append(node)

            # Sort by load priority (higher priority first)
            available.sort(key=lambda n: (-n.load_priority, n.vertical_name))

            order: List[str] = []

            while available:
                # Get next node (highest priority first)
                node = available.pop(0)
                order.append(node.vertical_name)

                # Update in-degree for dependents
                for dependent_name in node.dependents:
                    in_degree[dependent_name] -= 1
                    if in_degree[dependent_name] == 0:
                        available.append(self._nodes[dependent_name])

                # Re-sort available nodes by priority
                available.sort(key=lambda n: (-n.load_priority, n.vertical_name))

            # Check if all nodes were processed (should be true if no cycles)
            if len(order) != len(self._nodes):
                # This shouldn't happen if cycle detection works
                logger.warning(f"Load order incomplete: {len(order)}/{len(self._nodes)} nodes")

            return LoadOrder(order=order)

    def _detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies using DFS.

        Returns:
            List of cycles (each cycle is a list of vertical names)
        """
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node_name: str) -> bool:
            """DFS helper to detect cycles."""
            visited.add(node_name)
            rec_stack.add(node_name)
            path.append(node_name)

            node = self._nodes.get(node_name)
            if node:
                for dep_name in node.dependencies:
                    if dep_name not in visited:
                        if dfs(dep_name):
                            return True
                    elif dep_name in rec_stack:
                        # Found a cycle
                        cycle_start = path.index(dep_name)
                        cycle = path[cycle_start:] + [dep_name]
                        cycles.append(cycle)
                        return True

            path.pop()
            rec_stack.remove(node_name)
            return False

        for node_name in self._nodes:
            if node_name not in visited:
                dfs(node_name)

        return cycles

    def get_load_sequence(self, vertical_name: str) -> List[str]:
        """Get the load sequence for a specific vertical and its dependencies.

        Args:
            vertical_name: Name of the vertical

        Returns:
            List of vertical names in load order (dependencies first)

        Raises:
            ValueError: If vertical_name not in graph
            DependencyCycleError: If circular dependency detected
        """
        with self._lock:
            if vertical_name not in self._nodes:
                raise ValueError(f"Vertical '{vertical_name}' not in graph")

            # Build subgraph with only this vertical and its dependencies
            subgraph = ExtensionDependencyGraph()

            # Add the target vertical
            target_node = self._nodes[vertical_name]
            subgraph.add_vertical(
                vertical_name,
                target_node.version,
                target_node.manifest,
                target_node.load_priority,
            )

            # Recursively add dependencies
            added: Set[str] = set()

            def add_dependencies(name: str) -> None:
                # Prevent infinite recursion on cycles
                if name in added:
                    return

                added.add(name)
                node = self._nodes.get(name)
                if not node:
                    return

                for dep_name in node.dependencies:
                    # Add dependency node if not already present
                    if dep_name not in subgraph._nodes:
                        dep_node = self._nodes[dep_name]
                        subgraph.add_vertical(
                            dep_name,
                            dep_node.version,
                            dep_node.manifest,
                            dep_node.load_priority,
                        )

                    # Add dependency relationship (even if node already exists)
                    subgraph.add_dependency(name, dep_name)

                    # Recursively add dependencies of this dependency
                    add_dependencies(dep_name)

            add_dependencies(vertical_name)

            # Resolve load order
            result = subgraph.resolve_load_order()

            # Check for cycles
            if result.cycles:
                cycle = result.cycles[0]
                raise DependencyCycleError(cycle)

            return result.order

    def build_from_manifests(self, manifests: Dict[str, ExtensionManifest]) -> None:
        """Build graph from a collection of manifests.

        Args:
            manifests: Dict mapping vertical names to ExtensionManifests
        """
        with self._lock:
            # Add all verticals
            for vertical_name, manifest in manifests.items():
                self.add_vertical(
                    vertical_name,
                    manifest.version,
                    manifest,
                    manifest.load_priority,
                )

            # Add dependencies
            for vertical_name, manifest in manifests.items():
                for dep in manifest.extension_dependencies:
                    try:
                        self.add_dependency(
                            vertical_name,
                            dep.extension_name,
                            required=not dep.optional,
                        )
                    except ValueError as e:
                        if not dep.optional:
                            logger.warning(
                                f"Failed to add dependency '{dep.extension_name}' "
                                f"for '{vertical_name}': {e}"
                            )

    def clear(self) -> None:
        """Clear all nodes from the graph."""
        with self._lock:
            self._nodes.clear()
            logger.debug("Cleared dependency graph")

    def list_verticals(self) -> List[str]:
        """List all verticals in the graph.

        Returns:
            List of vertical names
        """
        with self._lock:
            return list(self._nodes.keys())

    def has_vertical(self, vertical_name: str) -> bool:
        """Check if a vertical is in the graph.

        Args:
            vertical_name: Name of the vertical

        Returns:
            True if vertical is in graph, False otherwise
        """
        with self._lock:
            return vertical_name in self._nodes

    def get_graph_depth(self) -> int:
        """Get the maximum depth of the dependency graph.

        Returns:
            Maximum depth (number of dependency levels)
        """
        with self._lock:
            if not self._nodes:
                return 0

            max_depth = 0
            for node_name in self._nodes:
                depth = self._get_node_depth(node_name, set())
                max_depth = max(max_depth, depth)

            return max_depth

    def _get_node_depth(self, node_name: str, visited: Set[str]) -> int:
        """Get depth of a node recursively."""
        if node_name in visited:
            return 0  # Cycle detection - prevent infinite recursion

        visited.add(node_name)
        node = self._nodes.get(node_name)
        if not node or not node.dependencies:
            return 0

        max_dep_depth = 0
        for dep_name in node.dependencies:
            dep_depth = self._get_node_depth(dep_name, visited.copy())
            max_dep_depth = max(max_dep_depth, dep_depth)

        return max_dep_depth + 1


__all__ = [
    "DependencyCycleError",
    "DependencyNode",
    "LoadOrder",
    "ExtensionDependencyGraph",
]
