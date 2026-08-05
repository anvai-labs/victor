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

"""Config Extensions - ISP-compliant composite for configuration protocols.

This module provides a focused extension for configuration-related vertical capabilities:
- Mode configuration (tool budgets, iterations, temperatures)
- RL configuration (learners, quality thresholds)
- Team specifications (multi-agent setups)

This replaces the config-related parts of the monolithic VerticalExtensions class,
following Interface Segregation Principle (ISP).

Usage:
    from victor.core.verticals.extensions import ConfigExtensions
    from victor.core.verticals.protocols import ModeConfig

    config_ext = ConfigExtensions(
        mode_config=CodingModeConfigProvider(),
        rl_config=CodingRLConfigProvider(),
        team_specs={
            "code_review": TeamSpec(name="code_review", formation="pipeline"),
        },
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.core.verticals.protocols import ModeConfig


@dataclass
class ConfigExtensions:
    """Focused extension for configuration-related vertical capabilities.

    Groups mode config, RL config, and team specs - the configuration parts
    that were previously bundled in VerticalExtensions.

    Attributes:
        mode_config: Optional mode configuration provider for operational modes.
            Defines tool budgets, iteration limits, and temperatures.
        rl_config: Optional RL configuration provider for reinforcement learning.
            Defines active learners, quality thresholds, and task mappings.
        team_specs: Dictionary of team specifications for multi-agent execution.
            Maps team names to TeamSpec instances.

    Example:
        config_ext = ConfigExtensions(
            mode_config=CodingModeConfigProvider(),
            rl_config=CodingRLConfigProvider(),
            team_specs={
                "code_review": TeamSpec(
                    name="code_review",
                    formation=TeamFormation.PIPELINE,
                    agents=[reviewer, fixer, tester],
                ),
                "feature_team": TeamSpec(
                    name="feature_team",
                    formation=TeamFormation.HIERARCHICAL,
                    agents=[planner, implementer, reviewer],
                ),
            },
        )

        # Get mode configs
        modes = config_ext.get_mode_configs()

        # Get RL learners
        learners = config_ext.get_active_learners()
    """

    mode_config: Optional[Any] = None  # ModeConfigProviderProtocol
    rl_config: Optional[Any] = None  # RLConfigProviderProtocol
    team_specs: Dict[str, Any] = field(default_factory=dict)  # Dict[str, TeamSpec]

    def get_mode_configs(self) -> Dict[str, ModeConfig]:
        """Get mode configurations from the provider.

        Returns:
            Dict mapping mode names to ModeConfig instances
        """
        if self.mode_config and hasattr(self.mode_config, "get_mode_configs"):
            return self.mode_config.get_mode_configs()
        return {}

    def get_mode(self, mode_name: str) -> Optional[ModeConfig]:
        """Get a specific mode configuration.

        Args:
            mode_name: Name of the mode to get

        Returns:
            ModeConfig if found, None otherwise
        """
        return self.get_mode_configs().get(mode_name)

    def get_default_mode(self) -> str:
        """Get the default mode name.

        Returns:
            Default mode name or "default" if not configured
        """
        if self.mode_config and hasattr(self.mode_config, "get_default_mode"):
            return self.mode_config.get_default_mode()
        return "default"

    def get_default_tool_budget(self) -> int:
        """Get the default tool budget.

        Returns:
            Default tool budget or 10 if not configured
        """
        if self.mode_config and hasattr(self.mode_config, "get_default_tool_budget"):
            return self.mode_config.get_default_tool_budget()
        return 10

    def get_rl_settings(self) -> Dict[str, Any]:
        """Get RL configuration settings.

        Returns:
            RL config dict with learners, thresholds, etc.
        """
        if self.rl_config and hasattr(self.rl_config, "get_rl_config"):
            return self.rl_config.get_rl_config()
        return {}

    def get_active_learners(self) -> List[str]:
        """Get list of active RL learners.

        Returns:
            List of learner type names
        """
        config = self.get_rl_settings()
        return config.get("active_learners", [])

    def get_quality_thresholds(self) -> Dict[str, float]:
        """Get task-specific quality thresholds.

        Returns:
            Dict mapping task types to quality threshold values
        """
        config = self.get_rl_settings()
        return config.get("quality_thresholds", {})

    def get_rl_hooks(self) -> Optional[Any]:
        """Get RL hooks for outcome recording.

        Returns:
            RLHooks instance or None
        """
        if self.rl_config and hasattr(self.rl_config, "get_rl_hooks"):
            return self.rl_config.get_rl_hooks()
        return None

    def get_team_spec(self, team_name: str) -> Optional[Any]:
        """Get a specific team specification.

        Args:
            team_name: Name of the team to get

        Returns:
            TeamSpec if found, None otherwise
        """
        return self.team_specs.get(team_name)

    def get_all_team_names(self) -> List[str]:
        """Get all available team names.

        Returns:
            List of team names
        """
        return list(self.team_specs.keys())

    def get_default_team(self) -> Optional[str]:
        """Get the default team name.

        Returns:
            Default team name or None
        """
        # Return first team if any exist
        if self.team_specs:
            return next(iter(self.team_specs.keys()))
        return None

    def has_mode_config(self) -> bool:
        """Check if mode configuration is available.

        Returns:
            True if mode_config is set
        """
        return self.mode_config is not None

    def has_rl_config(self) -> bool:
        """Check if RL configuration is available.

        Returns:
            True if rl_config is set
        """
        return self.rl_config is not None

    def has_teams(self) -> bool:
        """Check if team specifications are available.

        Returns:
            True if team_specs is not empty
        """
        return bool(self.team_specs)

    def merge(self, other: "ConfigExtensions") -> "ConfigExtensions":
        """Merge with another ConfigExtensions instance.

        Other's non-None values override self.

        Args:
            other: Another ConfigExtensions to merge from

        Returns:
            New ConfigExtensions with merged content
        """
        # Take other's config if present, else keep self's
        merged_mode = other.mode_config or self.mode_config
        merged_rl = other.rl_config or self.rl_config

        # Merge team specs (other overrides same-name entries)
        merged_teams = dict(self.team_specs)
        merged_teams.update(other.team_specs)

        return ConfigExtensions(
            mode_config=merged_mode,
            rl_config=merged_rl,
            team_specs=merged_teams,
        )

    def __bool__(self) -> bool:
        """Return True if any content is present."""
        return bool(self.mode_config or self.rl_config or self.team_specs)


__all__ = ["ConfigExtensions"]
