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

"""Mode Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for mode configuration.
Following ISP, these protocols are focused on a single responsibility:
providing operational mode configurations.

Usage:
    from victor.core.verticals.protocols.mode_provider import (
        ModeConfigProviderProtocol,
        ModeConfig,
    )

    class CodingModeProvider(ModeConfigProviderProtocol):
        def get_mode_configs(self) -> Dict[str, ModeConfig]:
            return {
                "fast": ModeConfig(name="fast", tool_budget=5, max_iterations=10),
                "thorough": ModeConfig(name="thorough", tool_budget=30, max_iterations=60),
            }
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Protocol, runtime_checkable

# =============================================================================
# Data Types
# =============================================================================


@dataclass
class ModeConfig:
    """Configuration for an operational mode.

    Attributes:
        name: Mode name (e.g., "fast", "thorough")
        tool_budget: Tool call budget
        max_iterations: Maximum iterations
        temperature: Temperature setting
        description: Human-readable description
    """

    name: str
    tool_budget: int
    max_iterations: int
    temperature: float = 0.7
    description: str = ""


# =============================================================================
# Mode Config Provider Protocol
# =============================================================================


@runtime_checkable
class ModeConfigProviderProtocol(Protocol):
    """Protocol for providing mode configurations.

    Verticals can define domain-specific operational modes with
    appropriate tool budgets and iteration limits.

    Example:
        class CodingModeProvider(ModeConfigProviderProtocol):
            def get_mode_configs(self) -> Dict[str, ModeConfig]:
                return {
                    "fast": ModeConfig(
                        name="fast",
                        tool_budget=5,
                        max_iterations=10,
                        description="Quick code changes",
                    ),
                    "thorough": ModeConfig(
                        name="thorough",
                        tool_budget=30,
                        max_iterations=60,
                        description="Deep code analysis",
                    ),
                }
    """

    @abstractmethod
    def get_mode_configs(self) -> Dict[str, ModeConfig]:
        """Get mode configurations for this vertical.

        Returns:
            Dict mapping mode names to configurations
        """
        ...

    def get_default_mode(self) -> str:
        """Get the default mode name.

        Returns:
            Name of the default mode
        """
        return "default"

    def get_default_tool_budget(self) -> int:
        """Get default tool budget when no mode is specified.

        Returns:
            Default tool call budget
        """
        return 10


__all__ = [
    "ModeConfig",
    "ModeConfigProviderProtocol",
]
