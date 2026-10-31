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

"""Data Analysis mode configurations using central registry.

This module registers data analysis-specific operational modes with the central
ModeConfigRegistry and exports a registry-based provider for protocol
compatibility.
"""

from __future__ import annotations

from typing import Dict

from victor.framework.extensions import (
    ModeConfigRegistry,
    RegistryBasedModeConfigProvider,
)
from victor_contracts.verticals.mode_config import ModeDefinition

# =============================================================================
# Data Analysis-Specific Modes (Registered with Central Registry)
# =============================================================================

_DATA_ANALYSIS_MODES: Dict[str, ModeDefinition] = {
    "research": ModeDefinition(
        name="research",
        tool_budget=400,
        max_iterations=750,
        temperature=0.8,
        description="Deep research analysis with multiple iterations",
        exploration_multiplier=3.0,
        allowed_stages=[
            "INITIAL",
            "DATA_LOADING",
            "EXPLORATION",
            "CLEANING",
            "ANALYSIS",
            "VISUALIZATION",
            "REPORTING",
            "COMPLETION",
        ],
    ),
}

# Data Analysis-specific task type budgets
# 5x'd 2026-10-31: agents run longer autonomous sessions
_DATA_ANALYSIS_TASK_BUDGETS: Dict[str, int] = {
    "data_profiling": 40,
    "basic_stats": 50,
    "visualization": 60,
    "correlation": 75,
    "regression": 100,
    "clustering": 100,
    "time_series": 125,
    "ml_pipeline": 200,
    "full_report": 250,
}


# =============================================================================
# Register with Central Registry
# =============================================================================


def _register_data_analysis_modes() -> None:
    """Register data analysis modes with the central registry."""
    registry = ModeConfigRegistry.get_instance()
    registry.register_vertical(
        name="data_analysis",
        modes=_DATA_ANALYSIS_MODES,
        task_budgets=_DATA_ANALYSIS_TASK_BUDGETS,
        default_mode="standard",
        default_budget=125,  # 5x'd 2026-10-31
    )


# NOTE: Import-time auto-registration removed (SOLID compliance)
# Registration happens when DataAnalysisModeConfigProvider is instantiated during
# vertical integration. The provider's __init__ calls _register_data_analysis_modes()
# for idempotent registration.


# =============================================================================
# Provider (Protocol Compatibility)
# =============================================================================


class DataAnalysisModeConfigProvider(RegistryBasedModeConfigProvider):
    """Mode configuration provider for data analysis vertical.

    Uses the central ModeConfigRegistry but provides analysis-specific
    complexity mapping.
    """

    def __init__(self) -> None:
        """Initialize data analysis mode provider."""
        # Ensure registration (idempotent - handles singleton reset)
        _register_data_analysis_modes()
        super().__init__(
            vertical="data_analysis",
            default_mode="standard",
            default_budget=125,  # 5x'd 2026-10-31
        )

    def get_mode_for_complexity(self, complexity: str) -> str:
        """Map complexity level to analysis mode.

        Args:
            complexity: Complexity level

        Returns:
            Recommended mode name
        """
        mapping = {
            "trivial": "quick",
            "simple": "quick",
            "moderate": "standard",
            "complex": "comprehensive",
            "highly_complex": "research",
        }
        return mapping.get(complexity, "standard")


__all__ = [
    "DataAnalysisModeConfigProvider",
]
