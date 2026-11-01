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

"""DevOps mode configurations using SDK-owned static descriptors."""

from __future__ import annotations

from typing import Dict

from victor_contracts.verticals.mode_config import (
    ModeDefinition,
    StaticModeConfigProvider,
    VerticalModeConfig,
)

_DEVOPS_MODES: Dict[str, ModeDefinition] = {
    "migration": ModeDefinition(
        name="migration",
        tool_budget=300,
        max_iterations=600,
        temperature=0.7,
        description="Large-scale infrastructure migrations",
        exploration_multiplier=2.5,
        allowed_stages=[
            "INITIAL",
            "ASSESSMENT",
            "PLANNING",
            "IMPLEMENTATION",
            "VALIDATION",
            "DEPLOYMENT",
            "MONITORING",
            "COMPLETION",
        ],
    ),
}

# 5x'd 2026-10-31: agents run longer autonomous sessions
_DEVOPS_TASK_BUDGETS: Dict[str, int] = {
    "dockerfile_simple": 25,
    "dockerfile_complex": 50,
    "docker_compose": 60,
    "ci_cd_basic": 75,
    "ci_cd_advanced": 125,
    "kubernetes_manifest": 75,
    "kubernetes_helm": 125,
    "terraform_module": 100,
    "terraform_full": 200,
    "monitoring_setup": 100,
}


def _build_mode_config(default_mode: str = "standard") -> VerticalModeConfig:
    return VerticalModeConfig(
        vertical_name="devops",
        modes=dict(_DEVOPS_MODES),
        task_budgets=dict(_DEVOPS_TASK_BUDGETS),
        default_mode=default_mode,
        default_budget=100,  # 5x'd 2026-10-31
    )


class DevOpsModeConfigProvider(StaticModeConfigProvider):
    """Mode configuration provider for the DevOps vertical."""

    def __init__(self) -> None:
        super().__init__(_build_mode_config())

    def get_mode_for_complexity(self, complexity: str) -> str:
        mapping = {
            "trivial": "quick",
            "simple": "quick",
            "moderate": "standard",
            "complex": "comprehensive",
            "highly_complex": "migration",
        }
        return mapping.get(complexity, "standard")


__all__ = ["DevOpsModeConfigProvider"]
