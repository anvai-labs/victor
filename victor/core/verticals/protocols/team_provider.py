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

"""Team Provider Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for multi-agent team management.
Following ISP, these protocols are focused on a single responsibility:
defining and providing team specifications.

Usage:
    from victor.core.verticals.protocols.team_provider import (
        TeamSpecProviderProtocol,
        VerticalTeamProviderProtocol,
    )

    class CodingTeamSpecProvider(TeamSpecProviderProtocol):
        def get_team_specs(self) -> Dict[str, Any]:
            return {
                "code_review_team": TeamSpec(
                    name="code_review_team",
                    formation=TeamFormation.PIPELINE,
                    agents=[...],
                ),
            }
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, Protocol, runtime_checkable

# =============================================================================
# Team Spec Provider Protocol
# =============================================================================


@runtime_checkable
class TeamSpecProviderProtocol(Protocol):
    """Protocol for providing team specifications.

    Enables verticals to define multi-agent team configurations
    for complex task execution.

    Example:
        class CodingTeamSpecProvider(TeamSpecProviderProtocol):
            def get_team_specs(self) -> Dict[str, Any]:
                return {
                    "code_review_team": TeamSpec(
                        name="code_review_team",
                        formation=TeamFormation.PIPELINE,
                        agents=[...],
                    ),
                }
    """

    @abstractmethod
    def get_team_specs(self) -> Dict[str, Any]:
        """Get team specifications for this vertical.

        Returns:
            Dict mapping team names to TeamSpec instances
        """
        ...

    def get_default_team(self) -> Optional[str]:
        """Get the default team name.

        Returns:
            Default team name or None
        """
        return None


# =============================================================================
# Vertical Team Provider Protocol
# =============================================================================


@runtime_checkable
class VerticalTeamProviderProtocol(Protocol):
    """Protocol for verticals providing team specifications.

    This protocol enables type-safe isinstance() checks instead of hasattr()
    when integrating vertical team specs with the framework.

    Example:
        class CodingVertical(VerticalBase, VerticalTeamProviderProtocol):
            @classmethod
            def get_team_spec_provider(cls) -> Optional[TeamSpecProviderProtocol]:
                return CodingTeamSpecProvider()
    """

    @classmethod
    def get_team_spec_provider(cls) -> Optional[TeamSpecProviderProtocol]:
        """Get the team specification provider for this vertical.

        Returns:
            TeamSpecProviderProtocol implementation or None
        """
        ...


__all__ = [
    "TeamSpecProviderProtocol",
    "VerticalTeamProviderProtocol",
]
