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

"""Mode helper mixin with common mode creation utilities.

This module provides ModeHelperMixin, a mixin class with utility methods
for creating common mode configurations that verticals can use.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from victor.core.mode_config import ModeDefinition


class ModeHelperMixin:
    """Mixin class providing common mode configuration helpers.

    Provides utility methods for creating common mode patterns that
    verticals can use when implementing get_vertical_modes().

    This class is designed to be used as a mixin alongside BaseModeConfigProvider:

        class MyVerticalModeConfig(BaseModeConfigProvider, ModeHelperMixin):
            def get_vertical_name(self) -> str:
                return \"myvertical\"

            def get_vertical_modes(self) -> Dict[str, ModeDefinition]:
                return {
                    **self.create_quick_modes(),
                    **self.create_standard_modes(),
                    **self.create_thorough_modes(),
                    **self.create_domain_specific_modes(),
                }
    """

    @staticmethod
    def create_quick_mode(
        name: str = "quick",
        tool_budget: int = 5,
        max_iterations: int = 10,
        description: str = "Quick tasks with minimal exploration",
    ) -> ModeDefinition:
        """Create a quick mode configuration.

        Args:
            name: Mode name
            tool_budget: Maximum tool calls
            max_iterations: Maximum iterations
            description: Human-readable description

        Returns:
            ModeDefinition instance
        """
        return ModeDefinition(
            name=name,
            tool_budget=tool_budget,
            max_iterations=max_iterations,
            temperature=0.5,
            description=description,
            exploration_multiplier=0.5,
        )

    @staticmethod
    def create_standard_mode(
        name: str = "standard",
        tool_budget: int = 15,
        max_iterations: int = 40,
        description: str = "Balanced mode for typical tasks",
    ) -> ModeDefinition:
        """Create a standard mode configuration.

        Args:
            name: Mode name
            tool_budget: Maximum tool calls
            max_iterations: Maximum iterations
            description: Human-readable description

        Returns:
            ModeDefinition instance
        """
        return ModeDefinition(
            name=name,
            tool_budget=tool_budget,
            max_iterations=max_iterations,
            temperature=0.7,
            description=description,
            exploration_multiplier=1.0,
        )

    @staticmethod
    def create_thorough_mode(
        name: str = "thorough",
        tool_budget: int = 30,
        max_iterations: int = 80,
        description: str = "Thorough analysis and comprehensive changes",
    ) -> ModeDefinition:
        """Create a thorough mode configuration.

        Args:
            name: Mode name
            tool_budget: Maximum tool calls
            max_iterations: Maximum iterations
            description: Human-readable description

        Returns:
            ModeDefinition instance
        """
        return ModeDefinition(
            name=name,
            tool_budget=tool_budget,
            max_iterations=max_iterations,
            temperature=0.8,
            description=description,
            exploration_multiplier=2.0,
        )

    @staticmethod
    def create_exploration_mode(
        name: str = "explore",
        tool_budget: int = 20,
        max_iterations: int = 60,
        description: str = "Extended exploration for understanding",
    ) -> ModeDefinition:
        """Create an exploration mode configuration.

        Args:
            name: Mode name
            tool_budget: Maximum tool calls
            max_iterations: Maximum iterations
            description: Human-readable description

        Returns:
            ModeDefinition instance
        """
        return ModeDefinition(
            name=name,
            tool_budget=tool_budget,
            max_iterations=max_iterations,
            temperature=0.7,
            description=description,
            exploration_multiplier=3.0,
        )

    @staticmethod
    def create_custom_mode(
        name: str,
        tool_budget: int,
        max_iterations: int,
        temperature: float = 0.7,
        description: str = "",
        exploration_multiplier: float = 1.0,
        allowed_tools: Optional[Set[str]] = None,
        priority_tools: Optional[List[str]] = None,
    ) -> ModeDefinition:
        """Create a custom mode configuration.

        Args:
            name: Mode name
            tool_budget: Maximum tool calls
            max_iterations: Maximum iterations
            temperature: LLM temperature
            description: Human-readable description
            exploration_multiplier: Exploration multiplier
            allowed_tools: Set of allowed tool names
            priority_tools: List of priority tools

        Returns:
            ModeDefinition instance
        """
        return ModeDefinition(
            name=name,
            tool_budget=tool_budget,
            max_iterations=max_iterations,
            temperature=temperature,
            description=description,
            exploration_multiplier=exploration_multiplier,
            allowed_tools=allowed_tools,
            priority_tools=priority_tools or [],
        )

    def create_quick_modes(self) -> Dict[str, ModeDefinition]:
        """Create common quick mode variants.

        Returns:
            Dict of quick mode definitions
        """
        return {
            "quick": self.create_quick_mode(),
            "fast": self.create_quick_mode(name="fast"),
        }

    def create_standard_modes(self) -> Dict[str, ModeDefinition]:
        """Create common standard mode variants.

        Returns:
            Dict of standard mode definitions
        """
        return {
            "standard": self.create_standard_mode(),
            "default": self.create_standard_mode(
                name="default",
                tool_budget=10,
                max_iterations=30,
                description="Default operational mode",
            ),
        }

    def create_thorough_modes(self) -> Dict[str, ModeDefinition]:
        """Create common thorough mode variants.

        Returns:
            Dict of thorough mode definitions
        """
        return {
            "thorough": self.create_thorough_mode(),
            "comprehensive": self.create_thorough_mode(name="comprehensive"),
        }

    def create_domain_specific_modes(self) -> Dict[str, ModeDefinition]:
        """Create domain-specific modes (override in subclasses).

        Returns:
            Dict of domain-specific mode definitions
        """
        return {}

    @staticmethod
    def restrict_tools_for_mode(
        mode: ModeDefinition,
        allowed_tools: Set[str],
    ) -> ModeDefinition:
        """Create a mode with restricted tool access.

        Args:
            mode: Base mode definition
            allowed_tools: Set of allowed tool names

        Returns:
            New ModeDefinition with restricted tools
        """
        return ModeDefinition(
            name=f"{mode.name}_restricted",
            tool_budget=mode.tool_budget,
            max_iterations=mode.max_iterations,
            temperature=mode.temperature,
            description=f"{mode.description} (restricted tools)",
            exploration_multiplier=mode.exploration_multiplier,
            allowed_tools=allowed_tools,
            priority_tools=mode.priority_tools,
        )

    @staticmethod
    def prioritize_tools_for_mode(
        mode: ModeDefinition,
        priority_tools: List[str],
    ) -> ModeDefinition:
        """Create a mode with prioritized tools.

        Args:
            mode: Base mode definition
            priority_tools: List of priority tool names

        Returns:
            New ModeDefinition with prioritized tools
        """
        return ModeDefinition(
            name=f"{mode.name}_prioritized",
            tool_budget=mode.tool_budget,
            max_iterations=mode.max_iterations,
            temperature=mode.temperature,
            description=f"{mode.description} (prioritized tools)",
            exploration_multiplier=mode.exploration_multiplier,
            priority_tools=priority_tools,
            allowed_tools=mode.allowed_tools,
        )


__all__ = [
    "ModeHelperMixin",
]
