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

"""Core tool types for dependency management.

This module defines fundamental tool types used by the core tool dependency
system. These are placed in core to avoid circular imports between
core and verticals.

Note:
    For backward compatibility, these types are also re-exported from
    `victor.core.verticals.protocols`.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor_contracts.verticals.tool_dependencies import ToolDependencyConfig


@dataclass
class ToolDependency:
    """Dependency relationship between tools.

    Attributes:
        tool_name: The tool
        depends_on: Tools that should be called before this one
        enables: Tools that are enabled after this one succeeds
        weight: Transition probability weight
    """

    tool_name: str
    depends_on: Set[str] = field(default_factory=set)
    enables: Set[str] = field(default_factory=set)
    weight: float = 1.0


@runtime_checkable
class ToolDependencyProviderProtocol(Protocol):
    """Protocol for providing tool dependency information.

    Enables verticals to define tool execution patterns and
    transition probabilities for intelligent tool selection.

    Example:
        class CodingToolDependencies(ToolDependencyProviderProtocol):
            def get_dependencies(self) -> List[ToolDependency]:
                return [
                    ToolDependency(
                        tool_name="edit_files",
                        depends_on={"read_file"},
                        enables={"run_tests"},
                        weight=0.8,
                    ),
                ]
    """

    @abstractmethod
    def get_dependencies(self) -> List[ToolDependency]:
        """Get tool dependencies for this vertical.

        Returns:
            List of tool dependency definitions
        """
        ...

    def get_tool_sequences(self) -> List[List[str]]:
        """Get common tool call sequences.

        Returns:
            List of tool name sequences representing common workflows
        """
        return []


class EmptyToolDependencyProvider:
    """Null object for missing YAML files (LSP-compliant fallback).

    Provides minimal but valid default configuration when tool
    dependency YAML is not found. All operations return safe defaults,
    allowing the system to function gracefully without YAML files.

    This class follows the Null Object pattern to satisfy LSP (Liskov
    Substitution Principle) - callers can use this instance wherever
    they expect a tool dependency provider without risk of AttributeError.

    Example:
        # When YAML is missing, factory returns this instead of __new__()
        provider = EmptyToolDependencyProvider("research")
        deps = provider.get_dependencies()  # Returns empty list
        seq = provider.get_recommended_sequence("edit")  # Returns ["read"]
    """

    def __init__(self, vertical: str):
        """Initialize empty provider with minimal configuration.

        Args:
            vertical: Name of the vertical this fallback is for.
        """
        self._vertical = vertical
        # Minimal valid configuration
        self._default_sequence: List[str] = ["read"]
        self._required_tools: Set[str] = {"read"}
        self._optional_tools: Set[str] = set()
        self._dependencies: List[ToolDependency] = []
        self._transitions: Dict[str, List[Tuple[str, float]]] = {}
        self._clusters: Dict[str, Set[str]] = {}
        self._sequences: Dict[str, List[str]] = {}
        self._dependency_map: Dict[str, ToolDependency] = {}

    @property
    def vertical(self) -> str:
        """Get the vertical name."""
        return self._vertical

    # =========================================================================
    # ToolDependencyProviderProtocol Implementation
    # =========================================================================

    def get_dependencies(self) -> List[ToolDependency]:
        """Get tool dependencies (empty for fallback).

        Returns:
            Empty list as no dependencies are defined.
        """
        return []

    def get_tool_sequences(self) -> List[List[str]]:
        """Get common tool sequences (minimal for fallback).

        Returns:
            Single minimal sequence.
        """
        return [self._default_sequence.copy()]

    # =========================================================================
    # Extended Methods (same interface as BaseToolDependencyProvider)
    # =========================================================================

    def get_tool_transitions(self) -> Dict[str, List[Tuple[str, float]]]:
        """Get tool transition probabilities (empty for fallback)."""
        return {}

    def get_tool_clusters(self) -> Dict[str, Set[str]]:
        """Get tool clusters (empty for fallback)."""
        return {}

    def get_recommended_sequence(self, task_type: str) -> List[str]:
        """Get recommended tool sequence (minimal for fallback).

        Args:
            task_type: Type of task (ignored in fallback).

        Returns:
            Minimal default sequence ["read"].
        """
        return self._default_sequence.copy()

    def get_required_tools(self) -> Set[str]:
        """Get tools essential for this vertical."""
        return self._required_tools.copy()

    def get_optional_tools(self) -> Set[str]:
        """Get tools that enhance but aren't required (empty for fallback)."""
        return set()

    def get_transition_weight(self, from_tool: str, to_tool: str) -> float:
        """Get the transition weight between two tools.

        Args:
            from_tool: Previous tool called
            to_tool: Candidate next tool

        Returns:
            Default low weight (0.3) for all transitions.
        """
        return 0.3

    def suggest_next_tool(
        self,
        current_tool: str,
        used_tools: Optional[List[str]] = None,
    ) -> str:
        """Suggest the next tool (always "read" for fallback).

        Args:
            current_tool: Currently used tool
            used_tools: History of recently used tools

        Returns:
            "read" as the default safe tool.
        """
        return "read"

    def find_cluster(self, tool: str) -> Optional[str]:
        """Find which cluster a tool belongs to (None for fallback)."""
        return None

    def get_cluster_tools(self, cluster_name: str) -> Set[str]:
        """Get all tools in a cluster (empty for fallback)."""
        return set()

    def is_valid_transition(self, from_tool: str, to_tool: str) -> bool:
        """Check if a transition is valid (all valid for fallback)."""
        return True


__all__ = [
    "ToolDependency",
    "ToolDependencyProviderProtocol",
    "EmptyToolDependencyProvider",
]
