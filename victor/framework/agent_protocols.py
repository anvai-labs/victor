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

"""Agent protocols for multi-agent team coordination.

This module defines the core protocols and types for CrewAI-style multi-agent
orchestration in Victor. It provides:

- AgentCapability: Enum defining what agents can do (READ, WRITE, EXECUTE, etc.)
- MessageType: Enum for inter-agent message types
- AgentMessage: Dataclass for agent-to-agent communication
- IAgentRole: Protocol defining agent role contracts
- IAgentPersona: Protocol defining agent personality and style
- ITeamMember: Protocol for individual team members
- ITeamCoordinator: Protocol for team coordination
- TeamFormation: Enum for team organization patterns

NOTE: Types MessageType, AgentMessage, and TeamFormation are now imported from
victor.teams.types. This module re-exports them for backward compatibility.

Example:
    from victor.framework.agent_protocols import (
        AgentCapability,
        IAgentRole,
        TeamFormation,
    )

    # Check if a role has delegation capability
    if AgentCapability.DELEGATE in supervisor_role.capabilities:
        # Supervisor can delegate tasks to other agents
        pass

    # Configure team formation
    coordinator.set_formation(TeamFormation.HIERARCHICAL)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

# Import canonical types from victor.teams and re-export for backward compatibility
from victor.teams.types import (
    AgentMessage,
    MessageType,
    TeamFormation,
)

# Import protocols from canonical location to avoid circular dependencies
from victor.protocols.team import (
    IDelegateFollowUpCoordinator,
    ITeamMember,
    ITeamCoordinator,
)

# =============================================================================
# Agent Capability Enum (defined locally, not part of teams.types)
# =============================================================================


class AgentCapability(str, Enum):
    """Capabilities that an agent can have.

    These capabilities determine what actions an agent is allowed to perform
    and inform the team coordinator about agent abilities.

    Attributes:
        READ: Agent can read files, search codebase, observe state
        WRITE: Agent can write files, make changes, create content
        EXECUTE: Agent can execute commands, run scripts, deploy
        SEARCH: Agent can search codebases, web, documentation
        COMMUNICATE: Agent can communicate with other agents
        DELEGATE: Agent can delegate tasks to other team members
        APPROVE: Agent can approve/reject changes made by others
    """

    READ = "read"
    """Agent can read files, search codebase, observe state."""

    WRITE = "write"
    """Agent can write files, make changes, create content."""

    EXECUTE = "execute"
    """Agent can execute commands, run scripts, deploy."""

    SEARCH = "search"
    """Agent can search codebases, web, documentation."""

    COMMUNICATE = "communicate"
    """Agent can communicate with other agents."""

    DELEGATE = "delegate"
    """Agent can delegate tasks to other team members."""

    APPROVE = "approve"
    """Agent can approve/reject changes made by others."""


# =============================================================================
# MessageType, AgentMessage, TeamFormation - imported from victor.teams.types
# =============================================================================
# NOTE: These types are now imported from victor.teams.types at the top of this
# file and re-exported for backward compatibility. The canonical definitions
# live in victor.teams.types.


# =============================================================================
# Agent Role Protocol
# =============================================================================


@runtime_checkable
class IAgentRole(Protocol):
    """Protocol defining an agent's role in a team.

    An agent role specifies what the agent is responsible for, what it can do,
    and what tools it has access to. Roles inform the system prompt and
    constrain the agent's behavior.

    Attributes:
        name: Unique identifier for this role
        capabilities: Set of AgentCapability values this role has
        allowed_tools: Set of tool names this role can use
        tool_budget: Maximum number of tool calls for this role

    Example:
        @dataclass
        class ResearcherRole:
            name: str = "researcher"
            capabilities: Set[AgentCapability] = field(
                default_factory=lambda: {AgentCapability.READ, AgentCapability.SEARCH}
            )
            allowed_tools: Set[str] = field(
                default_factory=lambda: {"read", "grep", "semantic_search"}
            )
            tool_budget: int = 20

            def get_system_prompt_section(self) -> str:
                return "You are a research specialist..."
    """

    @property
    def name(self) -> str:
        """Get the role name."""
        ...

    @property
    def capabilities(self) -> Set[AgentCapability]:
        """Get the capabilities this role has."""
        ...

    @property
    def allowed_tools(self) -> Set[str]:
        """Get the tools this role can use."""
        ...

    @property
    def tool_budget(self) -> int:
        """Get the tool budget for this role."""
        ...

    def get_system_prompt_section(self) -> str:
        """Get the system prompt section for this role.

        Returns:
            A string to be included in the agent's system prompt
            that describes the role and its responsibilities.
        """
        ...


# =============================================================================
# Agent Persona Protocol
# =============================================================================


@runtime_checkable
class IAgentPersona(Protocol):
    """Protocol defining an agent's personality and communication style.

    A persona adds character to an agent, making interactions more natural
    and consistent. It includes background, expertise, and communication style.

    Attributes:
        name: Display name for this persona
        background: Background story or expertise description
        communication_style: How this agent communicates

    Example:
        @dataclass
        class SecurityExpertPersona:
            name: str = "Security Analyst"
            background: str = "10 years of experience in cybersecurity"
            communication_style: str = "Professional, thorough, risk-focused"

            def format_message(self, content: str) -> str:
                return f"[Security Analysis] {content}"
    """

    @property
    def name(self) -> str:
        """Get the persona's display name."""
        ...

    @property
    def background(self) -> str:
        """Get the persona's background description."""
        ...

    @property
    def communication_style(self) -> str:
        """Get the persona's communication style."""
        ...

    def format_message(self, content: str) -> str:
        """Format a message according to this persona's style.

        Args:
            content: The raw message content

        Returns:
            Formatted message reflecting this persona's style
        """
        ...


# =============================================================================
# Team Member Protocol
# =============================================================================


# =============================================================================
# Team Coordinator Protocol
# =============================================================================


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Enums
    "AgentCapability",
    "MessageType",
    "TeamFormation",
    # Dataclasses
    "AgentMessage",
    # Protocols
    "IAgentRole",
    "IAgentPersona",
    "ITeamMember",
    "ITeamCoordinator",
    "IDelegateFollowUpCoordinator",
]
