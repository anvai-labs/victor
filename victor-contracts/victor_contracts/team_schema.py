"""SDK-owned declarative team schema contracts.

This module provides the vertical-authoring team schema without requiring
extracted verticals to import ``victor.framework`` runtime modules.
The runtime bridge remains available through lazy adapter methods so host
applications can execute these definitions without forcing a hard dependency
during import time.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from victor_contracts.constants import get_canonical_name

if TYPE_CHECKING:
    from victor.teams.types import TeamMember


class TeamFormation(str, Enum):
    """Declarative team organization patterns."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    PIPELINE = "pipeline"
    CONSENSUS = "consensus"


class TeamAgentCategory(str, Enum):
    """Declarative coordination category for a team member.

    ``SUPERVISOR`` is the explicit contract for the coordinating member in a
    hierarchical team. ``SPECIALIST`` members perform delegated domain work.
    ``is_manager`` remains supported as a compatibility alias, but new team
    specs should use this field to make coordination intent visible at the
    contract boundary.
    """

    SPECIALIST = "specialist"
    SUPERVISOR = "supervisor"


def get_runtime_team_registry() -> Any:
    """Return the Victor host team registry when the runtime package is installed."""

    module = importlib.import_module("victor.framework.team_registry")
    return module.get_team_registry()


@dataclass
class MemoryConfig:
    """Configuration for per-agent memory behavior.

    Controls how an agent stores and retrieves memories across task executions.
    This allows fine-grained control over memory persistence, search behavior,
    and memory type preferences.

    Attributes:
        enabled: Whether memory is enabled for this agent
        persist_across_sessions: Store memories beyond current session
        search_own_memories_only: Only search this agent's memories (vs team)
        memory_types: Types of memories to store (entity, episodic, semantic)
        max_memories_per_query: Maximum memories returned per search
        relevance_threshold: Minimum relevance score for memory retrieval (0.0-1.0)
        auto_summarize: Automatically summarize long memories
        ttl_seconds: Time-to-live for memories (None = permanent)

    Example:
        memory_config = MemoryConfig(
            enabled=True,
            persist_across_sessions=True,
            memory_types={"entity", "episodic"},
            max_memories_per_query=20,
            relevance_threshold=0.7,
        )
    """

    enabled: bool = True
    persist_across_sessions: bool = False
    search_own_memories_only: bool = False
    memory_types: Set[str] = field(default_factory=lambda: {"entity", "episodic", "semantic"})
    max_memories_per_query: int = 10
    relevance_threshold: float = 0.5
    auto_summarize: bool = True
    ttl_seconds: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        return {
            "enabled": self.enabled,
            "persist_across_sessions": self.persist_across_sessions,
            "search_own_memories_only": self.search_own_memories_only,
            "memory_types": list(self.memory_types),
            "max_memories_per_query": self.max_memories_per_query,
            "relevance_threshold": self.relevance_threshold,
            "auto_summarize": self.auto_summarize,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryConfig":
        """Construct from a dictionary representation."""
        return cls(
            enabled=data.get("enabled", True),
            persist_across_sessions=data.get("persist_across_sessions", False),
            search_own_memories_only=data.get("search_own_memories_only", False),
            memory_types=set(data.get("memory_types", ["entity", "episodic", "semantic"])),
            max_memories_per_query=data.get("max_memories_per_query", 10),
            relevance_threshold=data.get("relevance_threshold", 0.5),
            auto_summarize=data.get("auto_summarize", True),
            ttl_seconds=data.get("ttl_seconds"),
        )


ROLE_MAPPING: Dict[str, str] = {
    "supervisor": "PLANNER",
    "manager": "PLANNER",
    "lead": "PLANNER",
    "coordinator": "PLANNER",
    "researcher": "RESEARCHER",
    "research": "RESEARCHER",
    "planner": "PLANNER",
    "plan": "PLANNER",
    "executor": "EXECUTOR",
    "execute": "EXECUTOR",
    "impl": "EXECUTOR",
    "implementer": "EXECUTOR",
    "reviewer": "REVIEWER",
    "review": "REVIEWER",
    "critic": "REVIEWER",
    "writer": "EXECUTOR",
    "analyzer": "RESEARCHER",
    "verifier": "REVIEWER",
}


@dataclass
class TeamMemberSpec:
    """Simplified specification for a team member.

    This provides a user-friendly way to specify team members without
    needing to know about the host runtime's SubAgentRole internals. Supports rich persona
    attributes inspired by CrewAI for natural agent characterization.

    Attributes:
        role: Role name (researcher, planner, executor, reviewer)
        goal: What this member should accomplish
        name: Optional display name (auto-generated if not provided)
        tool_budget: Maximum tool calls (default based on role)
        agent_category: Coordination category; use ``SUPERVISOR`` for the
            coordinating member in hierarchical teams
        is_manager: Legacy alias for supervisor (for hierarchical)
        priority: Execution priority (lower = earlier, for sequential/pipeline)
        backstory: Rich persona description for agent's history and character
        expertise: List of domain expertise areas for context-aware behavior
        personality: Communication style and behavioral traits
        max_delegation_depth: Maximum levels of delegation allowed (0 = none)
        memory: Whether to persist discoveries across tasks (simple flag)
        memory_config: Detailed memory configuration (overrides simple memory)
        cache: Whether to cache tool results
        verbose: Whether to show detailed execution logs
        max_iterations: Per-member iteration limit
        can_delegate: Whether the member may delegate (implied for the supervisor)
        provider: Per-member provider override (None inherits the team's)
        model: Per-member model override (None inherits the team's)
        temperature: Per-member sampling temperature (None inherits the team's)
        reasoning_effort: "low"/"medium"/"high" for reasoning-capable models
        formation_role: Context-driven formation role (e.g. "generator"/"critic")

    Example:
        # Simple usage
        TeamMemberSpec(
            role="researcher",
            goal="Find all authentication code and patterns",
            backstory="You are a security expert with 10 years experience.",
            tool_budget=20,
            memory=True,
        )

        # Rich persona with CrewAI-style attributes
        TeamMemberSpec(
            role="researcher",
            goal="Find authentication vulnerabilities",
            name="Security Analyst",
            backstory="10 years of security experience at major tech companies. "
                      "Previously led red team exercises at a Fortune 500 company.",
            expertise=["security", "authentication", "oauth", "jwt"],
            personality="methodical and thorough; communicates findings with severity ratings",
            max_delegation_depth=2,
            memory=True,
            memory_config=MemoryConfig(
                enabled=True,
                persist_across_sessions=True,
                memory_types={"entity", "semantic"},
                relevance_threshold=0.7,
            ),
            tool_budget=25,
        )
    """

    role: str
    goal: str
    name: Optional[str] = None
    tool_budget: Optional[int] = None
    allowed_tools: Optional[List[str]] = None
    agent_category: TeamAgentCategory = TeamAgentCategory.SPECIALIST
    is_manager: bool = False
    can_delegate: bool = False
    priority: int = 0
    backstory: str = ""
    expertise: List[str] = field(default_factory=list)
    personality: str = ""
    max_delegation_depth: int = 0
    memory: bool = False
    memory_config: Optional[MemoryConfig] = None
    cache: bool = True
    verbose: bool = False
    max_iterations: Optional[int] = None
    # Heterogeneous execution: per-member provider/model/temperature override.
    # When unset, the member inherits the team orchestrator's settings. Enables
    # cross-vendor teams (e.g. writer on one vendor, reviewer on another) and
    # per-role sampling (e.g. writer hot, reviewer at temperature 0).
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    # Per-member reasoning effort ("low"/"medium"/"high") for reasoning-capable
    # models. None inherits/omits; forwarded only to providers that support it.
    reasoning_effort: Optional[str] = None
    # Formation role for context-driven formations (e.g. "generator"/"critic"
    # for REFLECTION). None = positional binding.
    formation_role: Optional[str] = None

    def __post_init__(self) -> None:
        """Normalize canonical tool names at definition time."""
        if isinstance(self.agent_category, str):
            self.agent_category = TeamAgentCategory(self.agent_category)
        if self.is_manager:
            self.agent_category = TeamAgentCategory.SUPERVISOR
        if self.agent_category == TeamAgentCategory.SUPERVISOR:
            self.is_manager = True
            self.can_delegate = True
            if self.max_delegation_depth < 1:
                self.max_delegation_depth = 1
        if self.allowed_tools:
            self.allowed_tools = [get_canonical_name(tool) for tool in self.allowed_tools]

    @property
    def is_supervisor(self) -> bool:
        """Return whether this member is the team's coordinating supervisor."""
        return self.agent_category == TeamAgentCategory.SUPERVISOR or self.is_manager

    def to_team_member(self, index: int = 0) -> "TeamMember":
        """Convert this SDK definition into the host runtime TeamMember model."""
        try:
            from victor.core.shared_types import SubAgentRole
            from victor.teams.types import FormationRole, TeamMember
        except ImportError as exc:  # pragma: no cover - only exercised outside host runtime
            raise RuntimeError(
                "TeamMemberSpec.to_team_member() requires the Victor runtime host package."
            ) from exc

        role_lower = self.role.lower()
        role_name = ROLE_MAPPING.get(role_lower, "EXECUTOR")
        sub_role = getattr(SubAgentRole, role_name, SubAgentRole.EXECUTOR)

        name = self.name or f"{self.role.title()} Agent"
        member_id = f"{role_lower}_{uuid.uuid4().hex[:8]}"

        default_budgets = {
            SubAgentRole.RESEARCHER: 20,
            SubAgentRole.PLANNER: 15,
            SubAgentRole.EXECUTOR: 25,
            SubAgentRole.REVIEWER: 15,
        }
        tool_budget = self.tool_budget or default_budgets.get(sub_role, 15)

        member = TeamMember(
            id=member_id,
            role=sub_role,
            name=name,
            goal=self.goal,
            tool_budget=tool_budget,
            allowed_tools=self.allowed_tools,
            agent_category=self.agent_category,
            is_manager=self.is_manager,
            can_delegate=self.can_delegate,
            priority=self.priority if self.priority else index,
            backstory=self.backstory,
            expertise=list(self.expertise),
            personality=self.personality,
            max_delegation_depth=self.max_delegation_depth,
            memory=self.memory,
            memory_config=self.memory_config,
            cache=self.cache,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            formation_role=(
                FormationRole(self.formation_role).value
                if self.formation_role is not None
                else None
            ),
        )

        if member.memory_enabled:
            try:
                from victor.storage.memory import get_memory_coordinator

                member.attach_memory_coordinator(get_memory_coordinator())
            except ImportError:
                pass

        return member


@dataclass
class RoleConfig:
    """Configuration for a reusable role definition."""

    base_role: str
    tools: List[str]
    tool_budget: int
    description: str = ""

    def __post_init__(self) -> None:
        """Canonicalize tool names on creation."""
        if self.tools:
            self.tools = [get_canonical_name(tool) for tool in self.tools]


@dataclass
class TeamSpec:
    """Declarative team specification used by extracted verticals."""

    name: str
    description: str
    vertical: str
    formation: TeamFormation
    members: List[TeamMemberSpec]
    total_tool_budget: int = 100
    max_iterations: int = 50
    tags: List[str] = field(default_factory=list)
    task_types: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize member tool names if present."""
        for member in self.members:
            if member.allowed_tools:
                member.allowed_tools = [get_canonical_name(tool) for tool in member.allowed_tools]

    @property
    def member_count(self) -> int:
        """Get the number of team members."""
        return len(self.members)

    @property
    def roles(self) -> Set[str]:
        """Get the set of roles in this team."""
        return {member.role for member in self.members}

    def get_member_by_role(self, role: str) -> Optional[TeamMemberSpec]:
        """Get the first member with the given role."""
        for member in self.members:
            if member.role == role:
                return member
        return None

    def get_members_by_role(self, role: str) -> List[TeamMemberSpec]:
        """Get all members matching the given role."""
        return [member for member in self.members if member.role == role]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "vertical": self.vertical,
            "formation": (
                self.formation.value if hasattr(self.formation, "value") else str(self.formation)
            ),
            "members": [
                {
                    "role": member.role,
                    "goal": member.goal,
                    "name": member.name,
                    "tool_budget": member.tool_budget,
                    "allowed_tools": list(member.allowed_tools or []),
                    "agent_category": member.agent_category.value,
                    "backstory": member.backstory,
                    "expertise": list(member.expertise),
                    "personality": member.personality,
                    "memory": member.memory,
                    "memory_config": (
                        member.memory_config.to_dict() if member.memory_config else None
                    ),
                    "is_manager": member.is_manager,
                    "can_delegate": member.can_delegate,
                    "priority": member.priority,
                    "cache": member.cache,
                    "verbose": member.verbose,
                    "max_iterations": member.max_iterations,
                    "max_delegation_depth": member.max_delegation_depth,
                    "provider": member.provider,
                    "model": member.model,
                    "temperature": member.temperature,
                    "reasoning_effort": member.reasoning_effort,
                    "formation_role": member.formation_role,
                }
                for member in self.members
            ],
            "total_tool_budget": self.total_tool_budget,
            "max_iterations": self.max_iterations,
            "tags": list(self.tags),
            "task_types": list(self.task_types),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamSpec":
        """Construct from a dictionary representation."""
        members: List[TeamMemberSpec] = []
        for member_data in data.get("members", []):
            memory_config_data = member_data.get("memory_config")
            member = TeamMemberSpec(
                role=member_data["role"],
                goal=member_data["goal"],
                name=member_data.get("name"),
                tool_budget=member_data.get("tool_budget"),
                allowed_tools=member_data.get("allowed_tools"),
                agent_category=TeamAgentCategory(
                    member_data.get(
                        "agent_category",
                        member_data.get(
                            "agent_class",
                            (
                                TeamAgentCategory.SUPERVISOR.value
                                if member_data.get("is_manager", False)
                                else TeamAgentCategory.SPECIALIST.value
                            ),
                        ),
                    )
                ),
                backstory=member_data.get("backstory", ""),
                expertise=list(member_data.get("expertise", [])),
                personality=member_data.get("personality", ""),
                memory=member_data.get("memory", False),
                memory_config=(
                    MemoryConfig.from_dict(memory_config_data) if memory_config_data else None
                ),
                is_manager=member_data.get("is_manager", False),
                can_delegate=member_data.get("can_delegate", False),
                priority=member_data.get("priority", 0),
                cache=member_data.get("cache", True),
                verbose=member_data.get("verbose", False),
                max_iterations=member_data.get("max_iterations"),
                max_delegation_depth=member_data.get("max_delegation_depth", 0),
                provider=member_data.get("provider"),
                model=member_data.get("model"),
                temperature=member_data.get("temperature"),
                reasoning_effort=member_data.get("reasoning_effort"),
                formation_role=member_data.get("formation_role"),
            )
            members.append(member)

        formation_value = data.get("formation", TeamFormation.SEQUENTIAL.value)
        formation = TeamFormation(getattr(formation_value, "value", formation_value))

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            vertical=data["vertical"],
            formation=formation,
            members=members,
            total_tool_budget=data.get("total_tool_budget", 100),
            max_iterations=data.get("max_iterations", 50),
            tags=list(data.get("tags", [])),
            task_types=list(data.get("task_types", [])),
            metadata=dict(data.get("metadata", {})),
        )


def create_team_spec(
    name: str,
    description: str,
    vertical: str,
    formation: TeamFormation,
    members: List[TeamMemberSpec],
    **kwargs: Any,
) -> TeamSpec:
    """Factory function for building a team spec."""
    return TeamSpec(
        name=name,
        description=description,
        vertical=vertical,
        formation=formation,
        members=members,
        **kwargs,
    )


__all__ = [
    "MemoryConfig",
    "ROLE_MAPPING",
    "RoleConfig",
    "TeamAgentCategory",
    "TeamFormation",
    "TeamMemberSpec",
    "TeamSpec",
    "create_team_spec",
    "get_runtime_team_registry",
]
