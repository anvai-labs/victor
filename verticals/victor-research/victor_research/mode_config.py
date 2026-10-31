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

"""Research mode configurations using SDK-owned static descriptors."""

from __future__ import annotations

from typing import Dict

from victor_contracts.verticals.mode_config import (
    ModeDefinition,
    StaticModeConfigProvider,
    VerticalModeConfig,
)

_RESEARCH_MODES: Dict[str, ModeDefinition] = {
    "deep": ModeDefinition(
        name="deep",
        tool_budget=150,
        max_iterations=300,
        temperature=0.8,
        description="Comprehensive research with full verification cycle",
        exploration_multiplier=2.0,
        allowed_stages=[
            "INITIAL",
            "SEARCHING",
            "READING",
            "SYNTHESIZING",
            "WRITING",
            "VERIFICATION",
            "COMPLETION",
        ],
    ),
    "academic": ModeDefinition(
        name="academic",
        tool_budget=250,
        max_iterations=500,
        temperature=0.8,
        description="Thorough literature review with extensive citation",
        exploration_multiplier=3.0,
        allowed_stages=[
            "INITIAL",
            "SEARCHING",
            "READING",
            "SYNTHESIZING",
            "WRITING",
            "VERIFICATION",
            "COMPLETION",
        ],
    ),
}

# 5x'd 2026-10-31: agents run longer autonomous sessions
_RESEARCH_TASK_BUDGETS: Dict[str, int] = {
    "simple_lookup": 15,
    "fact_check": 40,
    "comparison": 60,
    "trend_analysis": 100,
    "literature_review": 200,
    "comprehensive_report": 250,
}


def _build_mode_config(default_mode: str = "standard") -> VerticalModeConfig:
    return VerticalModeConfig(
        vertical_name="research",
        modes=dict(_RESEARCH_MODES),
        task_budgets=dict(_RESEARCH_TASK_BUDGETS),
        default_mode=default_mode,
        default_budget=75,  # 5x'd 2026-10-31
    )


class ResearchModeConfigProvider(StaticModeConfigProvider):
    """Mode configuration provider for the research vertical."""

    def __init__(self) -> None:
        super().__init__(_build_mode_config())

    def get_mode_for_complexity(self, complexity: str) -> str:
        mapping = {
            "trivial": "quick",
            "simple": "quick",
            "moderate": "standard",
            "complex": "deep",
            "highly_complex": "academic",
        }
        return mapping.get(complexity, "standard")


__all__ = ["ResearchModeConfigProvider"]
