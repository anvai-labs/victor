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

"""Multi-agent system components for persona-based team collaboration.

This package provides generic building blocks for constructing multi-agent
systems with configurable personas and team structures.

Example:
    from victor.framework.multi_agent import (
        PersonaTraits,
        CommunicationStyle,
        ExpertiseLevel,
        TeamTemplate,
        TeamTopology,
        TeamSpec,
        TeamMember,
    )

    # Create a persona with traits
    researcher = PersonaTraits(
        name="Research Analyst",
        role="researcher",
        description="Analyzes codebases and finds patterns",
        communication_style=CommunicationStyle.TECHNICAL,
        expertise_level=ExpertiseLevel.EXPERT,
        strengths=["pattern recognition", "thorough analysis"],
        preferred_tools=["code_search", "read"],
    )

    # Create a team template
    template = TeamTemplate(
        name="Code Review Team",
        description="Reviews code for quality and security",
        topology=TeamTopology.PIPELINE,
        member_slots={"researcher": 1, "reviewer": 2},
    )

    # Build a concrete team
    team = TeamSpec(
        template=template,
        members=[
            TeamMember(persona=researcher, role_in_team="lead_researcher", is_leader=True),
            # New code can use agent_category=TeamAgentCategory.SUPERVISOR instead.
        ],
    )
"""

import warnings
from typing import TYPE_CHECKING, Any, List

from victor.framework.multi_agent.personas import (
    CommunicationStyle,
    ExpertiseLevel,
    PersonaTemplate,
    PersonaTraits,
)

if TYPE_CHECKING:
    from victor.framework.multi_agent.persona_provider import FrameworkPersonaProvider
from victor.framework.multi_agent.teams import (
    TaskAssignmentStrategy,
    TeamAgentCategory,
    TeamFormation,
    TeamMember,
    TeamMemberSpec,
    TeamSpec,
    TeamTemplate,
    TeamTopology,
)

__all__ = [
    # Personas
    "CommunicationStyle",
    "ExpertiseLevel",
    "PersonaTemplate",
    "PersonaTraits",
    "FrameworkPersonaProvider",
    # Teams
    "TaskAssignmentStrategy",
    "TeamAgentCategory",
    "TeamFormation",
    "TeamMember",
    "TeamMemberSpec",
    "TeamSpec",
    "TeamTemplate",
    "TeamTopology",
]


PERSONA_PROVIDER_DEPRECATION = (
    "victor.framework.multi_agent.FrameworkPersonaProvider (persona_provider) is "
    "deprecated and will be removed in a future victor-ai release (no earlier than the "
    "one shipping victor-contracts 0.14.0): nothing reads it at runtime. Use "
    "victor.framework.persona_registry, which the host fills from each vertical's "
    "get_personas() hook."
)


def __getattr__(name: str) -> Any:
    # FrameworkPersonaProvider is deprecated (backlog F-019): resolve it lazily
    # so importing this package stays quiet, and attribute the warning to the
    # caller instead of the submodule import.
    if name == "FrameworkPersonaProvider":
        warnings.warn(PERSONA_PROVIDER_DEPRECATION, DeprecationWarning, stacklevel=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from victor.framework.multi_agent.persona_provider import FrameworkPersonaProvider

        return FrameworkPersonaProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted([*globals(), "FrameworkPersonaProvider"])
