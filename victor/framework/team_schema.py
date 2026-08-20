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

"""Unified team specification schema.

This module provides a single TeamSpec schema that all verticals use,
eliminating the duplication of CodingTeamSpec, ResearchTeamSpec, etc.

Design Goals:
- Single source of truth for team specifications
- Verticals provide data only, not schema definitions
- Support for rich persona attributes (backstory, personality, expertise)
- Tool name canonicalization at schema level

Usage:
    from victor.framework.team_schema import TeamSpec, RoleConfig

    # Define a team spec
    code_review_team = TeamSpec(
        name="Code Review Team",
        description="Reviews code changes for quality",
        vertical="coding",
        formation=TeamFormation.PIPELINE,
        members=[...],
    )

    # Register with the registry
    from victor.framework.team_registry import get_team_registry
    registry = get_team_registry()
    registry.register("coding:code_review", code_review_team)
"""

from __future__ import annotations

# Re-export from SDK as canonical source (identity interop).
# Framework consumers import from here; SDK owns the class definitions.
from victor_contracts.team_schema import (  # noqa: F401
    RoleConfig,
    TeamAgentCategory,
    TeamFormation,
    TeamMemberSpec,
    TeamSpec,
    create_team_spec,
)

__all__ = [
    "TeamSpec",
    "TeamMemberSpec",
    "TeamFormation",
    "TeamAgentCategory",
    "RoleConfig",
    "create_team_spec",
]
