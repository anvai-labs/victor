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

# The definitions below live once in the SDK (promoted from this module) and
# are re-exported here as the SAME objects, so ``isinstance`` checks and
# dataclass identity agree with ``victor.core.verticals.protocols`` and
# ``victor_contracts.verticals.protocols``.
from victor_contracts.verticals.protocols.promoted import (
    TeamSpecProviderProtocol as TeamSpecProviderProtocol,
    VerticalTeamProviderProtocol as VerticalTeamProviderProtocol,
)

__all__ = [
    "TeamSpecProviderProtocol",
    "VerticalTeamProviderProtocol",
]
