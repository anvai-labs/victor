# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Agent configuration settings.

Extracted from victor/config/settings.py to improve maintainability.
Contains configuration for autonomous planning and agent-level behavior.

Note: Tool-level configuration (budget, retry, cache, selection) is already
extracted in victor/config/tool_settings.py as ToolSettings.
"""

from typing import Dict

from pydantic import BaseModel, Field, field_validator


class PlanningConfig(BaseModel):
    """Configuration for autonomous planning mode."""

    enabled: bool = False
    min_complexity: str = Field(
        default="moderate", description="Minimum complexity: simple, moderate, complex"
    )
    show_plan: bool = True

    @field_validator("min_complexity")
    @classmethod
    def validate_complexity(cls, v: str) -> str:
        """Validate complexity level.

        Args:
            v: Complexity level

        Returns:
            Validated complexity level

        Raises:
            ValueError: If complexity is unknown
        """
        valid_levels = {"simple", "moderate", "complex"}
        if v not in valid_levels:
            raise ValueError(
                f"Unknown complexity level '{v}'. "
                f"Valid levels: {', '.join(sorted(valid_levels))}"
            )
        return v


class AgentSettings(BaseModel):
    """Agent-level settings extracted from main Settings class.

    Groups agent behavior configuration including planning and validation.
    Note: Tool-level configuration is in ToolSettings (tool_settings.py).
    """

    # Planning
    enable_planning: bool = False
    planning_min_complexity: str = "moderate"
    planning_show_plan: bool = True

    # Completion strategy (ADR-009): "enhanced" (default) | "rubric" | "hybrid" | "legacy".
    # Threaded into AgenticLoop construction; default leaves behavior unchanged.
    completion_strategy: str = "enhanced"

    # Effect-grounded completion gate (ADR-010 / EVR-4): COMPLETE requires a verifiable effect
    # or is downgraded to RETRY ("completion-without-effect"). Opt-in, default off per the
    # flag-graduation policy; env override VICTOR_EFFECT_GATED_COMPLETION.
    effect_gated_completion: bool = False

    # Recovery layer-targeted failure attribution (ADR-012 / EVR-5, prong 2): the RecoveryService
    # attributes each failure to an ETCLOVG harness layer (via HTIR) and records it, instead of
    # only classifying by exception type. Opt-in, default off per the flag-graduation policy;
    # env override VICTOR_RECOVERY_LAYER_ATTRIBUTION.
    recovery_layer_attribution: bool = False

    @field_validator("planning_min_complexity")
    @classmethod
    def validate_complexity(cls, v: str) -> str:
        """Validate complexity level."""
        valid_levels = {"simple", "moderate", "complex"}
        if v not in valid_levels:
            raise ValueError(
                f"Unknown complexity level '{v}'. "
                f"Valid levels: {', '.join(sorted(valid_levels))}"
            )
        return v
