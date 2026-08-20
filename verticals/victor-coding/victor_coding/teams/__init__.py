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

"""Teams integration for coding vertical.

This package provides team specifications for common software development
tasks, including pre-configured teams for feature implementation,
bug fixing, refactoring, code review, and testing.

Example:
    from victor_coding.teams import (
        get_team_for_task,
        CODING_TEAM_SPECS,
    )

    # Get team for a task type
    team_spec = get_team_for_task("feature")
    print(f"Team: {team_spec.name}")
    print(f"Formation: {team_spec.formation}")
    print(f"Members: {len(team_spec.members)}")

    # Or access directly
    feature_team = CODING_TEAM_SPECS["feature_team"]

Teams are auto-registered with the global TeamSpecRegistry on import,
enabling cross-vertical team discovery via:
    from victor.framework.team_registry import get_team_registry
    registry = get_team_registry()
    coding_teams = registry.find_by_vertical("coding")
"""

import logging
from typing import Dict, List, Optional

from victor_contracts.team_schema import TeamSpec
from victor_contracts import TeamRegistryProtocol, get_default_team_registry

from victor_coding.teams.specs import (
    # Types
    CodingRoleConfig,
    # Role configurations
    CODING_ROLES,
    # Team specifications
    CODING_TEAM_SPECS,
    # Helper functions
    get_team_for_task,
    get_role_config,
    list_team_types,
    list_roles,
)

from victor_coding.teams.personas import (
    # Framework types (re-exported for convenience)
    FrameworkPersonaTraits,
    FrameworkCommunicationStyle,
    ExpertiseLevel,
    PersonaTemplate,
    # Coding-specific types
    ExpertiseCategory,
    CommunicationStyle,
    DecisionStyle,
    CodingPersonaTraits,
    PersonaTraits,  # Backward compatibility alias
    CodingPersona,
    # Pre-defined personas
    CODING_PERSONAS,
    # Helper functions
    get_persona,
    get_personas_for_role,
    get_persona_by_expertise,
    apply_persona_to_spec,
    list_personas,
)

__all__ = [
    # Types from specs
    "CodingRoleConfig",
    "TeamSpec",  # Canonical from victor_contracts.team_schema (use this)
    # Provider
    "CodingTeamSpecProvider",
    # Role configurations
    "CODING_ROLES",
    # Team specifications
    "CODING_TEAM_SPECS",
    # Helper functions from specs
    "get_team_for_task",
    "get_role_config",
    "list_team_types",
    "list_roles",
    # Framework types (re-exported for convenience)
    "FrameworkPersonaTraits",
    "FrameworkCommunicationStyle",
    "ExpertiseLevel",
    "PersonaTemplate",
    # Coding-specific types from personas
    "ExpertiseCategory",
    "CommunicationStyle",
    "DecisionStyle",
    "CodingPersonaTraits",
    "PersonaTraits",  # Backward compatibility alias
    "CodingPersona",
    # Pre-defined personas
    "CODING_PERSONAS",
    # Helper functions from personas
    "get_persona",
    "get_personas_for_role",
    "get_persona_by_expertise",
    "apply_persona_to_spec",
    "list_personas",
]

logger = logging.getLogger(__name__)


class CodingTeamSpecProvider:
    """Team specification provider for Coding vertical.

    Implements TeamSpecProviderProtocol interface for consistent
    ISP compliance across all verticals.
    """

    def get_team_specs(self) -> Dict[str, TeamSpec]:
        """Get all Coding team specifications.

        Returns:
            Dictionary mapping team names to TeamSpec instances
        """
        return CODING_TEAM_SPECS

    def get_team_for_task(self, task_type: str) -> Optional[TeamSpec]:
        """Get appropriate team for a task type.

        Args:
            task_type: Type of task

        Returns:
            TeamSpec or None if no matching team
        """
        return get_team_for_task(task_type)

    def list_team_types(self) -> List[str]:
        """List all available team types.

        Returns:
            List of team type names
        """
        return list_team_types()


def register_coding_teams(registry: TeamRegistryProtocol | None = None) -> int:
    """Register coding teams with global registry.

    This function is called during vertical integration by the framework's
    step handlers. Import-time auto-registration has been removed to avoid
    load-order coupling and duplicate registration.

    Returns:
        Number of teams registered.
    """
    try:
        registry = registry or get_default_team_registry()
        if registry is None:
            logger.debug("No default team registry available; skipping coding team registration")
            return 0
        count = registry.register_from_vertical("coding", CODING_TEAM_SPECS)
        logger.debug(f"Registered {count} coding teams via framework integration")
        return count
    except Exception as e:
        logger.warning(f"Failed to register coding teams: {e}")
        return 0


# NOTE: Import-time auto-registration removed (SOLID compliance)
# Registration now happens during vertical integration via step_handlers.py
# This avoids load-order coupling and duplicate registration issues.
