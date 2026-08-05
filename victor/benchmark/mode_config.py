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

"""Benchmark mode configuration using victor.contrib.

This module provides mode configurations for the benchmark vertical using
the BaseModeConfigProvider from victor.contrib.mode_config.

Benchmark modes are optimized for:
- fast: Quick operations with minimal tool calls
- default: Balanced settings for standard tasks
- thorough: Comprehensive analysis with higher budgets

Migrated from: mode_config.py (original implementation)
Migration date: 2026-02-28 22:57:40
"""

from typing import Dict

from victor.contrib.mode_config import BaseModeConfigProvider, ModeHelperMixin
from victor.core.mode_config import ModeDefinition


class BenchmarkModeConfig(BaseModeConfigProvider, ModeHelperMixin):
    """Mode configuration provider for benchmark vertical.

    This provider uses BaseModeConfigProvider to provide:
    - Automatic registration with ModeConfigRegistry
    - Inherited framework default modes (quick, standard, thorough)
    - Tool budget calculation with task-type support
    - 6 helper methods for creating common modes
    - Protocol compliance (ModeConfigProviderProtocol)
    """

    def get_vertical_name(self) -> str:
        """Return the vertical name for mode registration.

        Returns:
            Vertical identifier
        """
        return "benchmark"

    def get_vertical_modes(self) -> Dict[str, ModeDefinition]:
        """Define benchmark-specific mode definitions.

        Returns:
            Dict mapping mode names to ModeDefinition instances

        Note:
            Use helper methods from ModeHelperMixin to create
            common modes:
            - create_quick_mode(): Fast operations (5 tool budget)
            - create_standard_mode(): Balanced (15 tool budget)
            - create_thorough_mode(): Deep analysis (30 tool budget)
            - create_exploration_mode(): Extended exploration (20 tool budget)
            - create_custom_mode(): Custom configuration

        Helper methods for mode groups:
            - create_quick_modes(): Returns {'quick': ..., 'fast': ...}
            - create_standard_modes(): Returns {'standard': ..., 'default': ...}
            - create_thorough_modes(): Returns {'thorough': ..., 'comprehensive': ...}
        """
        # Start with common modes from helper methods
        modes = {
            **self.create_quick_modes(),
            **self.create_standard_modes(),
            **self.create_thorough_modes(),
        }

        # Add benchmark-specific modes
        modes.update(
            {
                # Example: Custom mode with specific requirements
                "default": ModeDefinition(
                    name="default",
                    tool_budget=30,
                    max_iterations=15,
                    temperature=0.2,
                    description="Balanced settings for standard benchmark tasks",
                    exploration_multiplier=1.0,
                    priority_tools=[
                        "read",
                        "grep",
                        "edit",
                        "shell",
                    ],
                ),
            }
        )

        return modes

    def get_task_budgets(self) -> Dict[str, int]:
        """Define task-specific tool budgets.

        Returns:
            Dict mapping task types to recommended tool budgets

        Example:
            return {
                "quick_fix": 5,
                "refactor": 15,
                "investigation": 20,
            }
        """
        # Benchmark task budgets — extend as needed per vertical requirements
        return {
            # Quick tasks
            "quick": 10,
            "fast": 15,
            # Standard tasks
            "default": 30,
            "standard": 30,
            # Complex tasks
            "thorough": 50,
            "comprehensive": 60,
        }

    def get_default_mode(self) -> str:
        """Specify the default mode for this vertical.

        Returns:
            Default mode name
        """
        return "default"

    def get_default_budget(self) -> int:
        """Specify the default tool budget.

        Returns:
            Default tool budget when no mode or task specified
        """
        return 30


__all__ = [
    "BenchmarkModeConfig",
]
