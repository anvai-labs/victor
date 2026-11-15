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

"""Static, contract-only mode configurations for the data-analysis vertical."""

from __future__ import annotations

from typing import Dict

from victor_contracts.verticals.mode_config import (
    ModeDefinition,
    StaticModeConfigProvider,
    VerticalModeConfig,
)

_DATA_ANALYSIS_MODES: Dict[str, ModeDefinition] = {
    "quick": ModeDefinition(
        name="quick",
        tool_budget=25,
        max_iterations=50,
        temperature=0.5,
        description="Quick tasks with minimal exploration",
        exploration_multiplier=0.5,
    ),
    "standard": ModeDefinition(
        name="standard",
        tool_budget=75,
        max_iterations=200,
        temperature=0.7,
        description="Balanced mode for typical tasks",
        exploration_multiplier=1.0,
    ),
    "comprehensive": ModeDefinition(
        name="comprehensive",
        tool_budget=150,
        max_iterations=400,
        temperature=0.8,
        description="Thorough analysis and comprehensive changes",
        exploration_multiplier=2.0,
    ),
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

# Data Analysis-specific task type budgets; these longer autonomous sessions
# were intentionally increased on 2026-10-31.
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


class DataAnalysisModeConfigProvider(StaticModeConfigProvider):
    """Static mode provider with analysis-specific complexity mapping."""

    def __init__(self) -> None:
        super().__init__(
            VerticalModeConfig(
                vertical_name="data_analysis",
                modes=_DATA_ANALYSIS_MODES,
                task_budgets=_DATA_ANALYSIS_TASK_BUDGETS,
                default_mode="standard",
                default_budget=125,
            )
        )

    def get_mode_for_complexity(self, complexity: str) -> str:
        mapping = {
            "trivial": "quick",
            "simple": "quick",
            "moderate": "standard",
            "complex": "comprehensive",
            "highly_complex": "research",
        }
        return mapping.get(complexity, "standard")


__all__ = ["DataAnalysisModeConfigProvider"]
