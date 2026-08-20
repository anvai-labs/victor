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

"""Coding-specific mode configurations using SDK-owned static descriptors."""

from __future__ import annotations

from typing import Dict

from victor_contracts.verticals.mode_config import (
    ModeConfig,
    ModeDefinition,
    StaticModeConfigProvider,
    VerticalModeConfig,
)

_CODING_MODES: Dict[str, ModeDefinition] = {
    "architect": ModeDefinition(
        name="architect",
        tool_budget=40,
        max_iterations=100,
        temperature=0.8,
        description="Architecture analysis and design tasks",
        exploration_multiplier=2.5,
    ),
    "refactor": ModeDefinition(
        name="refactor",
        tool_budget=25,
        max_iterations=60,
        temperature=0.6,
        description="Code refactoring with safety checks",
        exploration_multiplier=1.5,
    ),
    "debug": ModeDefinition(
        name="debug",
        tool_budget=15,
        max_iterations=40,
        temperature=0.5,
        description="Debugging and issue investigation",
        exploration_multiplier=1.2,
    ),
    "test": ModeDefinition(
        name="test",
        tool_budget=15,
        max_iterations=40,
        temperature=0.5,
        description="Test creation and execution",
        exploration_multiplier=1.0,
    ),
}

_CODING_TASK_BUDGETS: Dict[str, int] = {
    "code_generation": 3,
    "create_simple": 2,
    "create": 5,
    "edit": 5,
    "search": 6,
    "action": 15,
    "analysis_deep": 25,
    "analyze": 12,
    "design": 25,
    "refactor": 15,
    "debug": 12,
    "test": 10,
    "general": 8,
}


def _build_mode_config(default_mode: str = "default") -> VerticalModeConfig:
    return VerticalModeConfig(
        vertical_name="coding",
        modes=dict(_CODING_MODES),
        task_budgets=dict(_CODING_TASK_BUDGETS),
        default_mode=default_mode,
        default_budget=10,
    )


class CodingModeConfigProvider(StaticModeConfigProvider):
    """Mode configuration provider for the coding vertical."""

    def __init__(self, default_mode: str = "default") -> None:
        super().__init__(_build_mode_config(default_mode))

    def get_mode_for_complexity(self, complexity: str) -> str:
        mapping = {
            "trivial": "fast",
            "simple": "fast",
            "moderate": "default",
            "complex": "thorough",
            "highly_complex": "architect",
        }
        return mapping.get(complexity, "default")


def get_mode_config(mode_name: str) -> ModeConfig | None:
    mode_definition = _CODING_MODES.get(mode_name.lower())
    if mode_definition is None:
        return None
    return mode_definition.to_mode_config()


def get_tool_budget(mode_name: str | None = None, task_type: str | None = None) -> int:
    if task_type is not None:
        return _CODING_TASK_BUDGETS.get(task_type, 10)
    if mode_name is not None:
        mode_definition = _CODING_MODES.get(mode_name.lower())
        if mode_definition is not None:
            return mode_definition.tool_budget
    return 10


__all__ = [
    "CodingModeConfigProvider",
    "get_mode_config",
    "get_tool_budget",
]
