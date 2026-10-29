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

"""Unified mode configuration registry for operational modes.

This module provides a central registry for operational modes across verticals,
eliminating ~180 LOC of duplicated mode configurations while providing:
- Default modes applicable to all verticals (quick, standard, comprehensive)
- Vertical-specific mode overrides and extensions
- Task-based tool budget recommendations
- Mode provider abstraction for backward compatibility

Example usage:
    from victor.core.mode_config import (
        ModeConfigRegistry,
        get_mode_config,
        get_tool_budget,
    )

    # Get registry instance
    registry = ModeConfigRegistry.get_instance()

    # Register vertical-specific modes
    registry.register_vertical(
        "coding",
        modes={"architect": ModeDefinition(tool_budget=40, ...)},
        task_budgets={"refactor": 15, "debug": 12},
    )

    # Get mode config (falls back to defaults)
    config = registry.get_mode("coding", "quick")

    # Get tool budget for task type
    budget = registry.get_tool_budget("coding", "refactor")
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

# The mode value types are defined once, in the SDK; this module adds the
# registry. (It used to hold its own copies of all four; they differed only
# cosmetically.)
from victor_contracts.verticals.mode_config import (
    ModeConfig as ModeConfig,
    ModeDefinition as ModeDefinition,
    ModeLevel as ModeLevel,
    VerticalModeConfig as VerticalModeConfig,
)

logger = logging.getLogger(__name__)


DEFAULT_MODES: Dict[str, ModeDefinition] = {
    "quick": ModeDefinition(
        name="quick",
        tool_budget=5,
        max_iterations=10,
        temperature=0.5,
        description="Quick tasks with minimal exploration",
        exploration_multiplier=0.5,
    ),
    "fast": ModeDefinition(
        name="fast",
        tool_budget=5,
        max_iterations=10,
        temperature=0.5,
        description="Alias for quick mode",
        exploration_multiplier=0.5,
    ),
    "standard": ModeDefinition(
        name="standard",
        tool_budget=15,
        max_iterations=40,
        temperature=0.7,
        description="Balanced mode for typical tasks",
        exploration_multiplier=1.0,
    ),
    "default": ModeDefinition(
        name="default",
        tool_budget=10,
        max_iterations=30,
        temperature=0.7,
        description="Default operational mode",
        exploration_multiplier=1.0,
    ),
    "comprehensive": ModeDefinition(
        name="comprehensive",
        tool_budget=30,
        max_iterations=80,
        temperature=0.8,
        description="Thorough analysis and comprehensive changes",
        exploration_multiplier=2.0,
    ),
    "thorough": ModeDefinition(
        name="thorough",
        tool_budget=30,
        max_iterations=80,
        temperature=0.8,
        description="Alias for comprehensive mode",
        exploration_multiplier=2.0,
    ),
    "explore": ModeDefinition(
        name="explore",
        tool_budget=20,
        max_iterations=60,
        temperature=0.7,
        description="Extended exploration for understanding",
        exploration_multiplier=3.0,
    ),
    "plan": ModeDefinition(
        name="plan",
        tool_budget=15,
        max_iterations=50,
        temperature=0.6,
        description="Planning and analysis mode before implementation",
        exploration_multiplier=2.5,
    ),
    "extended": ModeDefinition(
        name="extended",
        tool_budget=50,
        max_iterations=100,
        temperature=0.8,
        description="Extended operations for large tasks",
        exploration_multiplier=2.5,
    ),
}

# Default task type budgets (fallback for all verticals)
DEFAULT_TASK_BUDGETS: Dict[str, int] = {
    # General tasks
    "general": 8,
    "simple": 5,
    "moderate": 12,
    "complex": 20,
    # Analysis tasks
    "analyze": 12,
    "analysis_deep": 25,
    "explore": 15,
    "search": 8,
    # Modification tasks
    "create": 8,
    "create_simple": 3,
    "edit": 6,
    "refactor": 15,
    # Specialized tasks
    "debug": 12,
    "test": 10,
    "design": 20,
}


class ModeConfigRegistry:
    """Central registry for operational mode configurations.

    Provides unified access to mode configurations across all verticals,
    with fallback to default modes and task-based budget recommendations.

    This is a singleton - use get_instance() to access.
    """

    _instance: Optional["ModeConfigRegistry"] = None

    def __init__(self):
        """Initialize the registry with default modes."""
        self._default_modes: Dict[str, ModeDefinition] = DEFAULT_MODES.copy()
        self._default_task_budgets: Dict[str, int] = DEFAULT_TASK_BUDGETS.copy()
        self._verticals: Dict[str, VerticalModeConfig] = {}
        self._mode_aliases: Dict[str, str] = {
            "fast": "quick",
            "thorough": "comprehensive",
        }

    @classmethod
    def get_instance(cls) -> "ModeConfigRegistry":
        """Get the singleton registry instance.

        Returns:
            The global ModeConfigRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def register_vertical(
        self,
        name: str,
        modes: Optional[Dict[str, ModeDefinition]] = None,
        task_budgets: Optional[Dict[str, int]] = None,
        default_mode: str = "standard",
        default_budget: int = 10,
    ) -> None:
        """Register a vertical with its mode configurations.

        Args:
            name: Vertical name (e.g., "coding", "devops")
            modes: Vertical-specific mode definitions (overrides defaults)
            task_budgets: Task type to budget mapping
            default_mode: Default mode for this vertical
            default_budget: Default budget when no mode/task specified
        """
        self._verticals[name] = VerticalModeConfig(
            vertical_name=name,
            modes=modes or {},
            task_budgets=task_budgets or {},
            default_mode=default_mode,
            default_budget=default_budget,
        )

    def get_mode(self, vertical: Optional[str], mode_name: str) -> Optional[ModeDefinition]:
        """Get mode configuration, with vertical override support.

        Args:
            vertical: Vertical name (None for default only)
            mode_name: Mode name to look up

        Returns:
            ModeDefinition or None if not found
        """
        mode_name_lower = mode_name.lower()

        # Check vertical-specific mode first (before alias resolution)
        if vertical and vertical in self._verticals:
            vertical_config = self._verticals[vertical]
            if mode_name_lower in vertical_config.modes:
                return vertical_config.modes[mode_name_lower]

        # Check default modes (before alias resolution)
        if mode_name_lower in self._default_modes:
            return self._default_modes[mode_name_lower]

        # Resolve aliases and try again
        resolved_name = self._mode_aliases.get(mode_name_lower)
        if resolved_name:
            if vertical and vertical in self._verticals:
                vertical_config = self._verticals[vertical]
                if resolved_name in vertical_config.modes:
                    return vertical_config.modes[resolved_name]
            if resolved_name in self._default_modes:
                return self._default_modes[resolved_name]

        return None

    def get_mode_configs(self, vertical: Optional[str] = None) -> Dict[str, ModeDefinition]:
        """Get all mode configurations for a vertical.

        Args:
            vertical: Optional vertical name

        Returns:
            Dict of mode name to ModeDefinition
        """
        # Start with defaults
        modes = self._default_modes.copy()

        # Override with vertical-specific modes
        if vertical and vertical in self._verticals:
            modes.update(self._verticals[vertical].modes)

        return modes

    def get_tool_budget(
        self,
        vertical: Optional[str] = None,
        mode_name: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> int:
        """Get recommended tool budget.

        Priority order:
        1. Mode-specific budget (if mode_name provided)
        2. Vertical task-type budget (if vertical and task_type provided)
        3. Default task-type budget (if task_type provided)
        4. Vertical default budget
        5. Global default (10)

        Args:
            vertical: Optional vertical name
            mode_name: Optional mode name
            task_type: Optional task type

        Returns:
            Recommended tool budget
        """
        # 1. Try mode-specific budget
        if mode_name:
            mode = self.get_mode(vertical, mode_name)
            if mode:
                return mode.tool_budget

        # 2. Try vertical task-type budget
        if vertical and vertical in self._verticals and task_type:
            vertical_config = self._verticals[vertical]
            task_lower = task_type.lower()
            if task_lower in vertical_config.task_budgets:
                return vertical_config.task_budgets[task_lower]

        # 3. Try default task-type budget
        if task_type:
            task_lower = task_type.lower()
            if task_lower in self._default_task_budgets:
                return self._default_task_budgets[task_lower]

        # 4. Try vertical default
        if vertical and vertical in self._verticals:
            return self._verticals[vertical].default_budget

        # 5. Global default
        return 10

    def get_max_iterations(
        self,
        vertical: Optional[str] = None,
        mode_name: Optional[str] = None,
    ) -> int:
        """Get max iterations for a mode.

        Args:
            vertical: Optional vertical name
            mode_name: Optional mode name

        Returns:
            Max iterations (defaults to 30)
        """
        if mode_name:
            mode = self.get_mode(vertical, mode_name)
            if mode:
                return mode.max_iterations
        return 30

    def get_default_mode(self, vertical: Optional[str] = None) -> str:
        """Get the default mode name for a vertical.

        Args:
            vertical: Optional vertical name

        Returns:
            Default mode name
        """
        if vertical and vertical in self._verticals:
            return self._verticals[vertical].default_mode
        return "default"

    def list_modes(self, vertical: Optional[str] = None) -> List[str]:
        """List available mode names.

        Args:
            vertical: Optional vertical name

        Returns:
            List of mode names
        """
        modes = set(self._default_modes.keys())
        if vertical and vertical in self._verticals:
            modes.update(self._verticals[vertical].modes.keys())
        return sorted(modes)

    def list_verticals(self) -> List[str]:
        """List registered verticals.

        Returns:
            List of vertical names
        """
        return sorted(self._verticals.keys())

    def add_mode_alias(self, alias: str, target: str) -> None:
        """Add an alias for a mode name.

        Args:
            alias: Alias name
            target: Target mode name
        """
        self._mode_aliases[alias.lower()] = target.lower()

    def get_modes(self, vertical: Optional[str] = None) -> Dict[str, ModeConfig]:
        """Get all modes as simplified ModeConfig objects.

        Returns default modes when no vertical specified, or merged modes
        with vertical overrides when vertical is provided.

        Args:
            vertical: Optional vertical name for mode overrides

        Returns:
            Dict mapping mode name to ModeConfig
        """
        result: Dict[str, ModeConfig] = {}

        # Add defaults first
        for name, mode_def in self._default_modes.items():
            result[name] = mode_def.to_mode_config()

        # Override with vertical-specific modes
        if vertical and vertical in self._verticals:
            for name, mode_def in self._verticals[vertical].modes.items():
                result[name] = mode_def.to_mode_config()

        return result

    def register_modes(self, vertical: str, modes: Dict[str, ModeConfig]) -> None:
        """Register vertical-specific modes using simplified ModeConfig.

        This method allows registering modes using the simpler ModeConfig
        dataclass. The modes will be converted to ModeDefinition internally.

        Args:
            vertical: Vertical name
            modes: Dict mapping mode name to ModeConfig
        """
        mode_definitions: Dict[str, ModeDefinition] = {}
        for name, config in modes.items():
            mode_definitions[name] = ModeDefinition(
                name=name,
                tool_budget=config.tool_budget,
                max_iterations=config.max_iterations,
                exploration_multiplier=config.exploration_multiplier,
                allowed_tools=config.allowed_tools,
            )

        if vertical in self._verticals:
            self._verticals[vertical].modes.update(mode_definitions)
        else:
            self._verticals[vertical] = VerticalModeConfig(
                vertical_name=vertical,
                modes=mode_definitions,
            )


# =============================================================================
# Convenience Functions
# =============================================================================


def get_mode_config(mode_name: str, vertical: Optional[str] = None) -> Optional[ModeDefinition]:
    """Get mode configuration.

    Convenience function for direct mode lookup.

    Args:
        mode_name: Name of the mode
        vertical: Optional vertical name

    Returns:
        ModeDefinition or None if not found
    """
    registry = ModeConfigRegistry.get_instance()
    return registry.get_mode(vertical, mode_name)


def get_tool_budget(
    mode_name: Optional[str] = None,
    task_type: Optional[str] = None,
    vertical: Optional[str] = None,
) -> int:
    """Get tool budget based on mode or task type.

    Args:
        mode_name: Optional mode name
        task_type: Optional task type
        vertical: Optional vertical name

    Returns:
        Recommended tool budget
    """
    registry = ModeConfigRegistry.get_instance()
    return registry.get_tool_budget(vertical, mode_name, task_type)


def register_vertical_modes(
    vertical: str,
    modes: Dict[str, ModeDefinition],
    task_budgets: Optional[Dict[str, int]] = None,
) -> None:
    """Register modes for a vertical.

    Convenience function for vertical registration.

    Args:
        vertical: Vertical name
        modes: Mode definitions
        task_budgets: Optional task type budgets
    """
    registry = ModeConfigRegistry.get_instance()
    registry.register_vertical(vertical, modes, task_budgets)


# =============================================================================
# Registry-Based Provider (Protocol Adapter)
# =============================================================================


class RegistryBasedModeConfigProvider:
    """Mode config provider that delegates to the central registry.

    This class implements the ModeConfigProviderProtocol pattern but uses
    the central ModeConfigRegistry, eliminating duplicate code in verticals.

    Example:
        # In vertical __init__.py or mode_config.py
        from victor.core.mode_config import (
            ModeConfigRegistry,
            ModeDefinition,
            RegistryBasedModeConfigProvider,
        )

        # Register vertical-specific modes at import time
        registry = ModeConfigRegistry.get_instance()
        registry.register_vertical(
            "coding",
            modes={
                "architect": ModeDefinition(name="architect", tool_budget=40, ...),
                "refactor": ModeDefinition(name="refactor", tool_budget=25, ...),
            },
            task_budgets={"code_generation": 3, "refactor": 15},
        )

        # Export provider for protocol compatibility
        CodingModeConfigProvider = RegistryBasedModeConfigProvider("coding")
    """

    def __init__(
        self,
        vertical: str,
        default_mode: str = "default",
        default_budget: int = 10,
    ):
        """Initialize registry-based provider.

        Args:
            vertical: Vertical name for mode lookups
            default_mode: Default mode name
            default_budget: Default tool budget
        """
        self._vertical = vertical
        self._default_mode = default_mode
        self._default_budget = default_budget

    def get_mode_configs(self) -> Dict[str, "ModeConfig"]:
        """Get mode configurations from the registry.

        Returns:
            Dict mapping mode names to ModeConfig
        """
        registry = ModeConfigRegistry.get_instance()
        return registry.get_modes(self._vertical)

    def get_default_mode(self) -> str:
        """Get the default mode name.

        Returns:
            Name of the default mode
        """
        registry = ModeConfigRegistry.get_instance()
        return registry.get_default_mode(self._vertical)

    def get_default_tool_budget(self, task_type: str | None = None) -> int:
        """Get default tool budget, optionally for a task type.

        Args:
            task_type: Optional task type for specialized budget

        Returns:
            Recommended tool budget
        """
        registry = ModeConfigRegistry.get_instance()
        return registry.get_tool_budget(
            vertical=self._vertical,
            task_type=task_type,
        )

    def get_default_max_iterations(self, task_type: str | None = None) -> int:
        """Get default max iterations.

        Args:
            task_type: Optional task type

        Returns:
            Max iterations (default 2x tool budget)
        """
        budget = self.get_default_tool_budget(task_type)
        return budget * 2

    def get_tool_budget_for_task(self, task_type: str) -> int:
        """Get recommended tool budget for a specific task type.

        Args:
            task_type: The detected task type

        Returns:
            Recommended tool budget
        """
        return self.get_default_tool_budget(task_type)

    def get_mode_for_complexity(self, complexity: str) -> str:
        """Map complexity level to mode name.

        Args:
            complexity: Complexity level (trivial, simple, moderate, complex, highly_complex)

        Returns:
            Recommended mode name
        """
        from victor.framework.defaults.mode_configs import DEFAULT_COMPLEXITY_MAP

        return DEFAULT_COMPLEXITY_MAP.get(complexity, "standard")


__all__ = [
    # Enums
    "ModeLevel",
    # Dataclasses
    "ModeConfig",
    "ModeDefinition",
    "VerticalModeConfig",
    # Constants
    "DEFAULT_MODES",
    "DEFAULT_TASK_BUDGETS",
    # Registry
    "ModeConfigRegistry",
    # Provider
    "RegistryBasedModeConfigProvider",
    # Functions
    "get_mode_config",
    "get_tool_budget",
    "register_vertical_modes",
]
