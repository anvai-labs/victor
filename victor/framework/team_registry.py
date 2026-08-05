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

"""Team Specification Registry for reusable team blueprints.

This module provides a centralized registry for multi-agent team specifications,
enabling any vertical to register and query reusable team blueprints.

Design Philosophy:
- Singleton pattern for global team registry
- Thread-safe operations for concurrent access
- Namespace support for vertical-specific teams
- Inheritance and composition for team specs

Usage:
    from victor.framework.team_registry import (
        TeamSpecRegistry,
        get_team_registry,
        register_team_spec,
    )

    # Get the global registry
    registry = get_team_registry()

    # Register a team spec
    registry.register("coding:code_review", code_review_team_spec)

    # Get a team spec
    spec = registry.get("coding:code_review")

    # List available teams
    teams = registry.list_teams()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from victor.framework.tool_naming import canonicalize_tool_list, validate_tool_names
from victor_contracts.registries import set_default_team_registry

logger = logging.getLogger(__name__)

# Singleton instance
_registry_instance: Optional["TeamSpecRegistry"] = None
_registry_lock = threading.Lock()


@dataclass
class TeamSpecEntry:
    """Entry in the team spec registry.

    Attributes:
        name: Full qualified name (e.g., "coding:code_review")
        spec: The team specification object
        vertical: Optional vertical namespace
        tags: Tags for filtering/discovery
        description: Human-readable description
    """

    name: str
    spec: Any
    vertical: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    description: str = ""

    @property
    def namespace(self) -> Optional[str]:
        """Get the namespace portion of the name."""
        if ":" in self.name:
            return self.name.split(":")[0]
        return None

    @property
    def short_name(self) -> str:
        """Get the short name without namespace."""
        if ":" in self.name:
            return self.name.split(":", 1)[1]
        return self.name


class TeamSpecRegistry:
    """Registry for multi-agent team specifications.

    This class provides a centralized location for registering and
    querying team specifications from any vertical.

    Thread-safe for concurrent access.

    Example:
        registry = TeamSpecRegistry()

        # Register a team
        registry.register(
            "coding:code_review",
            CodeReviewTeam,
            tags={"review", "quality"},
            description="Team for reviewing code changes"
        )

        # Query teams
        review_teams = registry.find_by_tag("review")
        coding_teams = registry.find_by_vertical("coding")
    """

    def __init__(self):
        """Initialize the registry."""
        self._teams: Dict[str, TeamSpecEntry] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        spec: Any,
        *,
        vertical: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        description: str = "",
        replace: bool = False,
    ) -> None:
        """Register a team specification.

        Args:
            name: Full qualified name (e.g., "coding:code_review")
            spec: The team specification object
            vertical: Optional vertical namespace (auto-detected from name if not provided)
            tags: Tags for filtering/discovery
            description: Human-readable description
            replace: If True, replace existing registration

        Raises:
            ValueError: If name already registered and replace=False
        """
        with self._lock:
            if name in self._teams and not replace:
                raise ValueError(
                    f"Team spec '{name}' already registered. " f"Use replace=True to overwrite."
                )

            # Auto-detect vertical from namespaced name
            if vertical is None and ":" in name:
                vertical = name.split(":")[0]

            entry = TeamSpecEntry(
                name=name,
                spec=spec,
                vertical=vertical,
                tags=tags or set(),
                description=description,
            )

            self._teams[name] = entry
            logger.debug(f"Registered team spec: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a team specification.

        Args:
            name: Team spec name to unregister

        Returns:
            True if unregistered, False if not found
        """
        with self._lock:
            if name in self._teams:
                del self._teams[name]
                logger.debug(f"Unregistered team spec: {name}")
                return True
            return False

    def get(self, name: str) -> Optional[Any]:
        """Get a team specification by name.

        Args:
            name: Full qualified name

        Returns:
            Team spec or None if not found
        """
        with self._lock:
            entry = self._teams.get(name)
            return entry.spec if entry else None

    def get_entry(self, name: str) -> Optional[TeamSpecEntry]:
        """Get the full entry for a team specification.

        Args:
            name: Full qualified name

        Returns:
            TeamSpecEntry or None if not found
        """
        with self._lock:
            return self._teams.get(name)

    def list_teams(self) -> List[str]:
        """List all registered team names.

        Returns:
            List of team spec names
        """
        with self._lock:
            return list(self._teams.keys())

    def list_entries(self) -> List[TeamSpecEntry]:
        """List all team spec entries.

        Returns:
            List of TeamSpecEntry objects
        """
        with self._lock:
            return list(self._teams.values())

    def find_by_vertical(self, vertical: str) -> Dict[str, Any]:
        """Find team specs by vertical.

        Args:
            vertical: Vertical name to filter by

        Returns:
            Dict mapping names to team specs
        """
        with self._lock:
            return {
                name: entry.spec
                for name, entry in self._teams.items()
                if entry.vertical == vertical
            }

    def find_by_tag(self, tag: str) -> Dict[str, Any]:
        """Find team specs by tag.

        Args:
            tag: Tag to filter by

        Returns:
            Dict mapping names to team specs
        """
        with self._lock:
            return {name: entry.spec for name, entry in self._teams.items() if tag in entry.tags}

    def find_by_tags(self, tags: Set[str], match_all: bool = False) -> Dict[str, Any]:
        """Find team specs matching multiple tags.

        Args:
            tags: Set of tags to match
            match_all: If True, match all tags; if False, match any

        Returns:
            Dict mapping names to team specs
        """
        with self._lock:
            results = {}
            for name, entry in self._teams.items():
                if match_all:
                    if tags.issubset(entry.tags):
                        results[name] = entry.spec
                else:
                    if tags & entry.tags:
                        results[name] = entry.spec
            return results

    def find_team_for_task(
        self,
        task_type: str,
        preferred_vertical: Optional[str] = None,
    ) -> Optional[Any]:
        """Find the best team spec for a given task type across all verticals.

        This method enables cross-vertical team discovery by searching
        all registered teams for one that matches the task type.

        Args:
            task_type: Type of task (e.g., "feature", "deploy", "research", "eda")
            preferred_vertical: If set, prefer teams from this vertical

        Returns:
            Team spec that can handle the task, or None if not found
        """
        with self._lock:
            # Task type to vertical/team mapping
            # This is a simple heuristic; verticals can provide more sophisticated matching
            task_hints: Dict[str, List[str]] = {
                # Coding vertical
                "feature": ["coding"],
                "implement": ["coding"],
                "bug": ["coding"],
                "fix": ["coding"],
                "refactor": ["coding"],
                "review": ["coding", "research"],
                "test": ["coding", "data_analysis"],
                "documentation": ["coding"],
                # DevOps vertical
                "deploy": ["devops"],
                "deployment": ["devops"],
                "infrastructure": ["devops"],
                "container": ["devops"],
                "docker": ["devops"],
                "monitoring": ["devops"],
                "cicd": ["devops"],
                "pipeline": ["devops"],
                "security": ["devops", "coding"],
                # Research vertical
                "research": ["research"],
                "literature": ["research"],
                "fact_check": ["research"],
                "competitive": ["research"],
                "synthesis": ["research"],
                "technical": ["research", "coding"],
                # Data Analysis vertical
                "eda": ["data_analysis"],
                "exploration": ["data_analysis"],
                "clean": ["data_analysis"],
                "statistics": ["data_analysis"],
                "ml": ["data_analysis"],
                "visualization": ["data_analysis"],
                "report": ["data_analysis", "research"],
            }

            task_lower = task_type.lower()

            # Get list of verticals to check
            verticals_to_check = task_hints.get(task_lower, [])

            # If preferred vertical is specified and in the hints, prioritize it
            if preferred_vertical:
                if preferred_vertical in verticals_to_check:
                    verticals_to_check = [preferred_vertical] + [
                        v for v in verticals_to_check if v != preferred_vertical
                    ]
                else:
                    # Try preferred vertical first even if not in hints
                    verticals_to_check = [preferred_vertical] + verticals_to_check

            # If no hints, search all verticals
            if not verticals_to_check:
                verticals_to_check = list(
                    set(entry.vertical for entry in self._teams.values() if entry.vertical)
                )

            # Search each vertical for a matching team
            for vertical in verticals_to_check:
                # Try to find the vertical's get_team_for_task function result
                for name, entry in self._teams.items():
                    if entry.vertical != vertical:
                        continue

                    # Check if the team spec has a task type that matches
                    # Team names often contain the task type
                    short_name = entry.short_name
                    if task_lower in short_name or short_name.replace("_team", "") == task_lower:
                        return entry.spec

            # Fallback: try exact match on team short name
            for entry in self._teams.values():
                if entry.short_name == task_lower or entry.short_name == f"{task_lower}_team":
                    return entry.spec

            return None

    def clear(self) -> None:
        """Clear all registered team specs."""
        with self._lock:
            self._teams.clear()
            logger.debug("Cleared team spec registry")

    def register_from_vertical(
        self,
        vertical_name: str,
        team_specs: Dict[str, Any],
        replace: bool = True,
    ) -> int:
        """Register multiple team specs from a vertical.

        Convenience method for bulk registration with namespace prefixing.
        Tool names in team member specs are automatically canonicalized
        to ensure consistency across verticals.

        Args:
            vertical_name: Vertical name for namespace
            team_specs: Dict mapping team names to specs
            replace: If True, replace existing registrations

        Returns:
            Number of specs registered
        """
        count = 0
        for name, spec in team_specs.items():
            full_name = f"{vertical_name}:{name}"
            try:
                # Canonicalize tool names in team member specs
                self._canonicalize_team_tools(spec, full_name)

                self.register(
                    full_name,
                    spec,
                    vertical=vertical_name,
                    replace=replace,
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to register {full_name}: {e}")

        logger.info(f"Registered {count} team specs from vertical '{vertical_name}'")
        return count

    def _canonicalize_team_tools(self, spec: Any, team_name: str) -> None:
        """Canonicalize tool names in a team specification.

        Ensures consistent tool naming across all verticals by converting
        legacy names (e.g., 'read_file') to canonical names (e.g., 'read').

        Args:
            spec: Team specification object
            team_name: Team name for logging context
        """
        # Check if spec has members attribute (TeamSpec or similar)
        members = getattr(spec, "members", None)
        if not members:
            return

        for member in members:
            # Check for tools list on member
            tools = getattr(member, "tools", None)
            if tools and isinstance(tools, list):
                # Validate and warn about legacy names
                legacy = validate_tool_names(tools, context=f"team {team_name}", warn=True)

                # Canonicalize tool names
                canonical_tools = canonicalize_tool_list(tools)

                # Update member's tools if it has a setter or is mutable
                if hasattr(member, "tools"):
                    try:
                        member.tools = canonical_tools
                    except AttributeError:
                        # Frozen dataclass or immutable - log warning
                        if legacy:
                            logger.warning(
                                f"Cannot update frozen member tools in {team_name}. "
                                f"Consider using canonical names: {legacy}"
                            )


def get_team_registry() -> TeamSpecRegistry:
    """Get the global team spec registry.

    Thread-safe singleton access.

    Returns:
        Global TeamSpecRegistry instance
    """
    global _registry_instance

    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = TeamSpecRegistry()

    set_default_team_registry(_registry_instance)
    return _registry_instance


def register_team_spec(
    name: str,
    spec: Any,
    *,
    vertical: Optional[str] = None,
    tags: Optional[Set[str]] = None,
    description: str = "",
    replace: bool = False,
) -> None:
    """Register a team spec in the global registry.

    Convenience function for quick registration.

    Args:
        name: Full qualified name
        spec: Team specification object
        vertical: Optional vertical namespace
        tags: Tags for discovery
        description: Human-readable description
        replace: Replace existing if present
    """
    get_team_registry().register(
        name,
        spec,
        vertical=vertical,
        tags=tags,
        description=description,
        replace=replace,
    )


def get_team_spec(name: str) -> Optional[Any]:
    """Get a team spec from the global registry.

    Args:
        name: Full qualified name

    Returns:
        Team spec or None
    """
    return get_team_registry().get(name)


def load_all_verticals() -> int:
    """Load team specs from all verticals using protocol-based discovery.

    This function uses the protocol-based provider system which:
    1. Starts with safe defaults (no teams)
    2. Loads enhanced providers from installed packages via entry points
    3. Does NOT directly import vertical modules

    This approach ensures:
    - Framework works without any verticals installed
    - No direct dependencies on vertical modules
    - Verticals can enhance functionality via entry points

    Returns:
        Total number of teams registered across all providers.
    """
    from victor.framework.providers.protocol import get_provider_registry

    provider_registry = get_provider_registry()
    all_team_specs = provider_registry.get_all_team_specs()

    registry = get_team_registry()

    # Register all team specs from providers
    total_registered = 0
    for vertical_name, team_specs in all_team_specs.items():
        count = registry.register_from_vertical(vertical_name, team_specs)
        total_registered += count
        logger.debug(f"Loaded {count} teams from {vertical_name} provider")

    total_count = len(registry.list_teams())
    logger.info(f"Loaded {total_count} teams from {len(all_team_specs)} providers")
    return total_count


def find_team_for_task(
    task_type: str,
    preferred_vertical: Optional[str] = None,
) -> Optional[Any]:
    """Find the best team spec for a task type across all verticals.

    Convenience function that delegates to the registry.

    Args:
        task_type: Type of task (e.g., "feature", "deploy", "research", "eda")
        preferred_vertical: If set, prefer teams from this vertical

    Returns:
        Team spec that can handle the task, or None if not found
    """
    return get_team_registry().find_team_for_task(task_type, preferred_vertical)


__all__ = [
    "TeamSpecRegistry",
    "TeamSpecEntry",
    "get_team_registry",
    "register_team_spec",
    "get_team_spec",
    "load_all_verticals",
    "find_team_for_task",
]
