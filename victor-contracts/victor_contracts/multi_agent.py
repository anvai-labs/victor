"""SDK-owned multi-agent persona and team model contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from victor_contracts.constants import get_canonical_name
from victor_contracts.team_schema import (
    TeamAgentCategory,
    TeamFormation,
    TeamMemberSpec,
)


class CommunicationStyle(Enum):
    """Communication styles for agent personas."""

    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    CONCISE = "concise"


class ExpertiseLevel(Enum):
    """Expertise levels for agent personas."""

    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"
    SPECIALIST = "specialist"


@dataclass
class PersonaTraits:
    """Generic traits that define an agent persona."""

    name: str
    role: str
    description: str
    communication_style: CommunicationStyle = CommunicationStyle.TECHNICAL
    expertise_level: ExpertiseLevel = ExpertiseLevel.EXPERT
    verbosity: float = 0.5
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    preferred_tools: List[str] = field(default_factory=list)
    risk_tolerance: float = 0.5
    creativity: float = 0.5
    custom_traits: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate trait values after initialization."""
        if not 0.0 <= self.verbosity <= 1.0:
            raise ValueError(f"verbosity must be between 0.0 and 1.0, got {self.verbosity}")
        if not 0.0 <= self.risk_tolerance <= 1.0:
            raise ValueError(
                f"risk_tolerance must be between 0.0 and 1.0, got {self.risk_tolerance}"
            )
        if not 0.0 <= self.creativity <= 1.0:
            raise ValueError(f"creativity must be between 0.0 and 1.0, got {self.creativity}")
        self.preferred_tools = [get_canonical_name(tool) for tool in self.preferred_tools]

    def to_system_prompt_fragment(self) -> str:
        """Generate a system prompt fragment for this persona."""
        lines = [
            f"You are {self.name}, a {self.role}.",
            f"Description: {self.description}",
            f"Communication style: {self.communication_style.value}",
        ]
        if self.strengths:
            lines.append(f"Strengths: {', '.join(self.strengths)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert persona traits to a dictionary."""
        data = asdict(self)
        data["communication_style"] = self.communication_style.value
        data["expertise_level"] = self.expertise_level.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonaTraits":
        """Create PersonaTraits from a dictionary."""
        data = data.copy()
        if isinstance(data.get("communication_style"), str):
            data["communication_style"] = CommunicationStyle(data["communication_style"])
        if isinstance(data.get("expertise_level"), str):
            data["expertise_level"] = ExpertiseLevel(data["expertise_level"])
        return cls(**data)


@dataclass
class PersonaTemplate:
    """Template for creating personas with defaults."""

    base_traits: PersonaTraits
    overrides: Dict[str, Any] = field(default_factory=dict)

    def create(self, **kwargs: Any) -> PersonaTraits:
        """Create a PersonaTraits instance from this template."""
        merged = asdict(self.base_traits)
        merged["communication_style"] = self.base_traits.communication_style
        merged["expertise_level"] = self.base_traits.expertise_level
        merged.update(self.overrides)
        merged.update(kwargs)
        return PersonaTraits(**merged)


class TeamTopology(Enum):
    """Team organization topologies."""

    HIERARCHY = "hierarchy"
    MESH = "mesh"
    PIPELINE = "pipeline"
    HUB_SPOKE = "hub_spoke"

    def to_formation(self) -> TeamFormation:
        """Map persona-template topology to the canonical execution formation."""
        if self is TeamTopology.PIPELINE:
            return TeamFormation.PIPELINE
        if self is TeamTopology.MESH:
            return TeamFormation.PARALLEL
        return TeamFormation.HIERARCHICAL


class TaskAssignmentStrategy(Enum):
    """Strategies for assigning tasks to team members."""

    ROUND_ROBIN = "round_robin"
    SKILL_MATCH = "skill_match"
    LOAD_BALANCED = "load_balanced"


@dataclass
class TeamMember:
    """A member of a persona-driven multi-agent team.

    `agent_category` is the canonical coordination category. `is_leader`
    remains as the legacy/persona-facing alias for supervisor membership.
    """

    persona: PersonaTraits
    role_in_team: str
    is_leader: bool = False
    agent_category: TeamAgentCategory = TeamAgentCategory.SPECIALIST
    max_concurrent_tasks: int = 1
    tool_access: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize coordination category and tool access."""
        if isinstance(self.agent_category, str):
            self.agent_category = TeamAgentCategory(self.agent_category)
        if self.is_leader:
            self.agent_category = TeamAgentCategory.SUPERVISOR
        if self.agent_category is TeamAgentCategory.SUPERVISOR:
            self.is_leader = True
        self.tool_access = [get_canonical_name(tool) for tool in self.tool_access]

    @property
    def name(self) -> str:
        """Get the member's display name from persona."""
        return self.persona.name

    @property
    def expertise_level(self) -> Any:
        """Get the member's expertise level from persona."""
        return self.persona.expertise_level

    @property
    def is_supervisor(self) -> bool:
        """Compatibility alias for the canonical supervisor category."""
        return self.agent_category is TeamAgentCategory.SUPERVISOR


@dataclass
class TeamTemplate:
    """Generic template for team configuration."""

    name: str
    description: str
    topology: TeamTopology = TeamTopology.HIERARCHY
    assignment_strategy: TaskAssignmentStrategy = TaskAssignmentStrategy.SKILL_MATCH
    member_slots: Dict[str, int] = field(default_factory=dict)
    shared_context_keys: List[str] = field(default_factory=list)
    escalation_threshold: float = 0.8
    max_iterations: int = 10
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate template values after initialization."""
        if not 0.0 <= self.escalation_threshold <= 1.0:
            raise ValueError(
                f"escalation_threshold must be between 0.0 and 1.0, "
                f"got {self.escalation_threshold}"
            )
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "topology": self.topology.value,
            "assignment_strategy": self.assignment_strategy.value,
            "member_slots": self.member_slots,
            "shared_context_keys": self.shared_context_keys,
            "escalation_threshold": self.escalation_threshold,
            "max_iterations": self.max_iterations,
            "config": self.config,
        }


@dataclass
class TeamSpec:
    """Concrete team specification with assigned members."""

    template: TeamTemplate
    members: List[TeamMember] = field(default_factory=list)

    @property
    def supervisor(self) -> Optional[TeamMember]:
        """Get the coordinating supervisor if one exists."""
        for member in self.members:
            if member.is_supervisor:
                return member
        return None

    @property
    def leader(self) -> Optional[TeamMember]:
        """Compatibility alias for supervisor."""
        return self.supervisor

    @property
    def name(self) -> str:
        """Get the team name from template."""
        return self.template.name

    @property
    def topology(self) -> TeamTopology:
        """Get the team topology from template."""
        return self.template.topology

    @property
    def formation(self) -> TeamFormation:
        """Get the canonical runtime formation for this team."""
        return self.template.topology.to_formation()

    def get_members_by_role(self, role: str) -> List[TeamMember]:
        """Get all members with a specific role."""
        return [member for member in self.members if member.role_in_team == role]

    def validate_slots(self) -> List[str]:
        """Validate that member assignments match template slots."""
        errors: List[str] = []
        role_counts: Dict[str, int] = {}
        for member in self.members:
            role_counts[member.role_in_team] = role_counts.get(member.role_in_team, 0) + 1

        for role, required_count in self.template.member_slots.items():
            actual_count = role_counts.get(role, 0)
            if actual_count < required_count:
                errors.append(
                    f"Role '{role}' requires {required_count} members, got {actual_count}"
                )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to a dictionary."""
        return {
            "template": self.template.to_dict(),
            "members": [
                {
                    "persona": member.persona.to_dict(),
                    "role_in_team": member.role_in_team,
                    "is_leader": member.is_leader,
                    "agent_category": member.agent_category.value,
                    "max_concurrent_tasks": member.max_concurrent_tasks,
                    "tool_access": member.tool_access,
                }
                for member in self.members
            ],
        }


__all__ = [
    "CommunicationStyle",
    "ExpertiseLevel",
    "PersonaTemplate",
    "PersonaTraits",
    "TaskAssignmentStrategy",
    "TeamAgentCategory",
    "TeamFormation",
    "TeamMember",
    "TeamMemberSpec",
    "TeamSpec",
    "TeamTemplate",
    "TeamTopology",
]
