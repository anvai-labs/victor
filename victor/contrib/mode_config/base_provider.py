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

"""Base mode configuration provider for Victor verticals.

This module provides BaseModeConfigProvider, a reusable base class that
implements mode configuration using the framework's ModeConfigRegistry.
Verticals can inherit from this base class to get common mode functionality
while adding vertical-specific configurations.

Design Pattern: Template Method
- Base class provides common mode configuration infrastructure
- Verticals override get_vertical_name() and get_vertical_modes()
- Verticals can optionally override task budgets and default mode
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Dict, Optional

from victor.core.mode_config import (
    ModeConfigRegistry,
    ModeDefinition,
    RegistryBasedModeConfigProvider,
)

logger = logging.getLogger(__name__)


class BaseModeConfigProvider:
    """Base mode configuration provider for Victor verticals.

    Provides common mode configuration functionality using ModeConfigRegistry:
    - Automatic registration of vertical-specific modes
    - Mode lookup with fallback to defaults
    - Tool budget calculation with task-type support
    - Integration with framework's ModeConfigProviderProtocol

    Verticals should:
    1. Inherit from BaseModeConfigProvider
    2. Implement get_vertical_name() to return vertical identifier
    3. Implement get_vertical_modes() to return vertical-specific ModeDefinition dict
    4. Optionally override get_task_budgets() and get_default_mode()

    Example:
        class CodingModeConfig(BaseModeConfigProvider):
            def get_vertical_name(self) -> str:
                return \"coding\"

            def get_vertical_modes(self) -> Dict[str, ModeDefinition]:
                return {
                    \"architect\": ModeDefinition(
                        name=\"architect\",
                        tool_budget=40,
                        max_iterations=100,
                        description=\"Deep architectural planning\",
                    ),
                }

            def get_task_budgets(self) -> Dict[str, int]:
                return {
                    \"refactor\": 15,
                    \"debug\": 12,
                    \"test\": 10,
                }
    """

    def __init__(self, auto_register: bool = True):
        """Initialize the mode configuration provider.

        Args:
            auto_register: If True, automatically register with ModeConfigRegistry
        """
        self._registry = ModeConfigRegistry.get_instance()
        self._provider: Optional[RegistryBasedModeConfigProvider] = None

        # Auto-register vertical modes
        if auto_register:
            self._register_vertical()

        logger.info(
            f"{self.__class__.__name__} initialized for '{self.get_vertical_name()}' "
            f"with {len(self.get_vertical_modes())} custom modes"
        )

    # ==========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # ==========================================================================

    @abstractmethod
    def get_vertical_name(self) -> str:
        """Get the vertical name for mode registration.

        Returns:
            Vertical name (e.g., "devops", "rag", "research")
        """
        ...

    @abstractmethod
    def get_vertical_modes(self) -> Dict[str, ModeDefinition]:
        """Get vertical-specific mode definitions.

        Returns:
            Dict mapping mode names to ModeDefinition instances
        """
        ...

    # ==========================================================================
    # Template Methods - Can be overridden by subclasses
    # ==========================================================================

    def get_task_budgets(self) -> Dict[str, int]:
        """Get task type to tool budget mapping for this vertical.

        Returns:
            Dict mapping task types to tool budgets
        """
        return {}

    def get_default_mode(self) -> str:
        """Get the default mode for this vertical.

        Returns:
            Default mode name
        """
        return "standard"

    def get_default_budget(self) -> int:
        """Get the default tool budget for this vertical.

        Returns:
            Default tool budget
        """
        return 10

    # ==========================================================================
    # Mode Configuration - Common for all verticals
    # ==========================================================================

    def get_mode(self, mode_name: str) -> Optional[ModeDefinition]:
        """Get a mode configuration.

        Args:
            mode_name: Name of the mode

        Returns:
            ModeDefinition or None if not found
        """
        return self._registry.get_mode(self.get_vertical_name(), mode_name)

    def get_modes(self) -> Dict[str, ModeDefinition]:
        """Get all available modes for this vertical.

        Returns:
            Dict mapping mode names to ModeDefinition
        """
        return self._registry.get_mode_configs(self.get_vertical_name())

    def list_modes(self) -> list[str]:
        """List all available mode names.

        Returns:
            List of mode names
        """
        return self._registry.list_modes(self.get_vertical_name())

    # ==========================================================================
    # Tool Budget - Common for all verticals
    # ==========================================================================

    def get_tool_budget(
        self,
        mode_name: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> int:
        """Get recommended tool budget.

        Priority order:
        1. Mode-specific budget (if mode_name provided)
        2. Vertical task-type budget (if task_type provided)
        3. Default task-type budget (if task_type provided)
        4. Vertical default budget
        5. Global default (10)

        Args:
            mode_name: Optional mode name
            task_type: Optional task type

        Returns:
            Recommended tool budget
        """
        return self._registry.get_tool_budget(
            vertical=self.get_vertical_name(),
            mode_name=mode_name,
            task_type=task_type,
        )

    def get_max_iterations(self, mode_name: Optional[str] = None) -> int:
        """Get max iterations for a mode.

        Args:
            mode_name: Optional mode name

        Returns:
            Max iterations
        """
        return self._registry.get_max_iterations(
            vertical=self.get_vertical_name(),
            mode_name=mode_name,
        )

    # ==========================================================================
    # Protocol Integration
    # ==========================================================================

    def get_protocol_provider(self) -> RegistryBasedModeConfigProvider:
        """Get a protocol-compliant provider instance.

        Returns:
            RegistryBasedModeConfigProvider implementing ModeConfigProviderProtocol
        """
        if self._provider is None:
            self._provider = RegistryBasedModeConfigProvider(
                vertical=self.get_vertical_name(),
                default_mode=self.get_default_mode(),
                default_budget=self.get_default_budget(),
            )
        return self._provider

    # ==========================================================================
    # Private Helper Methods
    # ==========================================================================

    def _register_vertical(self) -> None:
        """Register this vertical with the ModeConfigRegistry."""
        self._registry.register_vertical(
            name=self.get_vertical_name(),
            modes=self.get_vertical_modes(),
            task_budgets=self.get_task_budgets(),
            default_mode=self.get_default_mode(),
            default_budget=self.get_default_budget(),
        )
        logger.debug(
            f"Registered vertical '{self.get_vertical_name()}' with "
            f"{len(self.get_vertical_modes())} modes and "
            f"{len(self.get_task_budgets())} task budgets"
        )


__all__ = [
    "BaseModeConfigProvider",
]
