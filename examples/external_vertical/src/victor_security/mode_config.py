# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Security mode configurations using SDK-owned static descriptors.

Authored purely against ``victor_contracts`` - the Victor runtime discovers
this provider through the ``victor.mode_configs`` entry point declared in
``pyproject.toml``.
"""

from __future__ import annotations

from victor_contracts.verticals.mode_config import (
    ModeDefinition,
    StaticModeConfigProvider,
    VerticalModeConfig,
)

# Stage names must stay consistent with SecurityAssistant.get_stages().
_SECURITY_STAGES = ["reconnaissance", "analysis", "reporting"]

_SECURITY_MODES: dict[str, ModeDefinition] = {
    "quick": ModeDefinition(
        name="quick",
        tool_budget=10,
        max_iterations=20,
        temperature=0.5,
        description="Fast triage scan over the highest-risk surfaces",
        exploration_multiplier=1.0,
        allowed_stages=list(_SECURITY_STAGES),
        priority_tools=["read", "code_search", "secret_pattern_scan"],
    ),
    "deep": ModeDefinition(
        name="deep",
        tool_budget=30,
        max_iterations=60,
        temperature=0.7,
        description="Comprehensive audit with scanner-assisted verification",
        exploration_multiplier=2.0,
        allowed_stages=list(_SECURITY_STAGES),
        priority_tools=["read", "code_search", "shell", "secret_pattern_scan"],
    ),
}

# Task budgets mirror SecurityAssistant.get_task_type_hints() tool budgets.
_SECURITY_TASK_BUDGETS: dict[str, int] = {
    "vulnerability_scan": 18,
    "dependency_audit": 14,
    "incident_review": 20,
}


def _build_mode_config(default_mode: str = "quick") -> VerticalModeConfig:
    return VerticalModeConfig(
        vertical_name="security",
        modes=dict(_SECURITY_MODES),
        task_budgets=dict(_SECURITY_TASK_BUDGETS),
        default_mode=default_mode,
        default_budget=12,
    )


class SecurityModeConfigProvider(StaticModeConfigProvider):
    """Mode configuration provider for the security vertical example."""

    def __init__(self) -> None:
        super().__init__(_build_mode_config())

    def get_mode_for_complexity(self, complexity: str) -> str:
        mapping = {
            "trivial": "quick",
            "simple": "quick",
            "moderate": "quick",
            "complex": "deep",
            "highly_complex": "deep",
        }
        return mapping.get(complexity, "quick")


__all__ = ["SecurityModeConfigProvider"]
