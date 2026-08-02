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

"""Teams - High-level API for multi-agent coordination.

This module provides a simplified API for creating and running agent teams,
exposing the existing teams infrastructure via the framework API.

Example:
    from victor.framework import Agent
    from victor.framework.teams import AgentTeam, TeamMemberSpec, TeamFormation

    # Create a team via Agent class method
    team = await Agent.create_team(
        name="Feature Implementation",
        goal="Implement user authentication",
        members=[
            TeamMemberSpec(role="researcher", goal="Find auth patterns"),
            TeamMemberSpec(role="planner", goal="Design implementation"),
            TeamMemberSpec(role="executor", goal="Write the code"),
            TeamMemberSpec(role="reviewer", goal="Review and fix"),
        ],
        formation=TeamFormation.PIPELINE,
    )

    # Run the team
    result = await team.run()
    print(result.final_output)

    # Or stream events
    async for event in team.stream():
        if event.type == TeamEventType.MEMBER_START:
            print(f"Starting: {event.member_name}")
        elif event.type == TeamEventType.MEMBER_COMPLETE:
            print(f"Complete: {event.member_name}")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from victor.teams.types import (
    MemoryConfig,
    TeamAgentCategory,
    TeamConfig,
    TeamMember,
)
from victor.teams import (
    MemberResult,
    create_coordinator,
    TeamFormation,
    TeamResult,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.core.shared_types import SubAgentRole
    from victor.core.protocols import OrchestratorProtocol as AgentOrchestrator
    from victor.protocols.team import ITeamCoordinator as TeamCoordinator
    from victor.framework.agent import Agent


class TeamEventType(str, Enum):
    """Types of events emitted during team execution.

    Events track team lifecycle and member progress.
    """

    # Lifecycle events
    TEAM_START = "team_start"
    """Team execution has started."""

    TEAM_COMPLETE = "team_complete"
    """Team execution has completed."""

    TEAM_ERROR = "team_error"
    """Team execution failed with an error."""

    # Member events
    MEMBER_START = "member_start"
    """A team member has started working."""

    MEMBER_PROGRESS = "member_progress"
    """A team member has made progress."""

    MEMBER_COMPLETE = "member_complete"
    """A team member has completed."""

    MEMBER_ERROR = "member_error"
    """A team member failed with an error."""

    # Communication events
    MESSAGE_SENT = "message_sent"
    """Inter-agent message was sent."""

    HANDOFF = "handoff"
    """Work was handed off between members."""


# Register domain event mapping with canonical EventType taxonomy
def _register_team_event_taxonomy() -> None:
    try:
        from victor.core.events.taxonomy import EventTaxonomyRegistry
        from victor.framework.events import EventType

        EventTaxonomyRegistry.register_domain(
            "team",
            TeamEventType,
            {
                TeamEventType.TEAM_START: EventType.STREAM_START,
                TeamEventType.TEAM_COMPLETE: EventType.STREAM_END,
                TeamEventType.TEAM_ERROR: EventType.ERROR,
                TeamEventType.MEMBER_START: EventType.PROGRESS,
                TeamEventType.MEMBER_PROGRESS: EventType.PROGRESS,
                TeamEventType.MEMBER_COMPLETE: EventType.MILESTONE,
                TeamEventType.MEMBER_ERROR: EventType.ERROR,
                TeamEventType.MESSAGE_SENT: EventType.CONTENT,
                TeamEventType.HANDOFF: EventType.STAGE_CHANGE,
            },
        )
    except ImportError:
        pass


_register_team_event_taxonomy()


@dataclass
class TeamEvent:
    """An event from team execution.

    Attributes:
        type: The type of event
        team_name: Name of the team
        member_id: ID of the member (if applicable)
        member_name: Name of the member (if applicable)
        progress: Progress percentage (0-100)
        message: Status message
        result: Member result (if complete)
        error: Error message (if failed)
        metadata: Additional event data
        timestamp: Unix timestamp when event was created
    """

    type: TeamEventType
    team_name: str
    member_id: Optional[str] = None
    member_name: Optional[str] = None
    progress: float = 0.0
    message: str = ""
    result: Optional[MemberResult] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def is_member_event(self) -> bool:
        """Check if this is a member-related event."""
        return self.type in (
            TeamEventType.MEMBER_START,
            TeamEventType.MEMBER_PROGRESS,
            TeamEventType.MEMBER_COMPLETE,
            TeamEventType.MEMBER_ERROR,
        )

    @property
    def is_lifecycle_event(self) -> bool:
        """Check if this is a lifecycle event."""
        return self.type in (
            TeamEventType.TEAM_START,
            TeamEventType.TEAM_COMPLETE,
            TeamEventType.TEAM_ERROR,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "team_name": self.team_name,
            "member_id": self.member_id,
            "member_name": self.member_name,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# Role name to SubAgentRole name mapping (string-based to avoid eager import)
ROLE_MAPPING: Dict[str, str] = {
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
    needing to know about SubAgentRole internals. Supports rich persona
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
    priority: int = 0
    # Rich persona attributes (CrewAI-compatible)
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
    # models (e.g. reviewer thinks hard at "high", writer at "low"). None
    # inherits/omits; forwarded only to providers/models that support it.
    reasoning_effort: Optional[str] = None
    # Formation role for context-driven formations (e.g. "generator"/"critic"
    # for REFLECTION). None = positional binding.
    formation_role: Optional[str] = None

    def __post_init__(self) -> None:
        """Normalize compatibility aliases for coordination category."""
        if isinstance(self.agent_category, str):
            self.agent_category = TeamAgentCategory(self.agent_category)
        if self.is_manager:
            self.agent_category = TeamAgentCategory.SUPERVISOR
        if self.agent_category == TeamAgentCategory.SUPERVISOR:
            self.is_manager = True
            if self.max_delegation_depth < 1:
                self.max_delegation_depth = 1

    @property
    def is_supervisor(self) -> bool:
        """Return whether this spec designates the team supervisor."""
        return self.agent_category == TeamAgentCategory.SUPERVISOR or self.is_manager

    def to_team_member(self, index: int = 0) -> TeamMember:
        """Convert to internal TeamMember.

        Args:
            index: Member index for ID generation

        Returns:
            TeamMember instance with memory coordinator attached if memory is enabled
        """
        from victor.core.shared_types import SubAgentRole

        # Resolve role via string mapping → enum
        role_lower = self.role.lower()
        role_name = ROLE_MAPPING.get(role_lower, "EXECUTOR")
        sub_role = getattr(SubAgentRole, role_name, SubAgentRole.EXECUTOR)

        # Generate name if not provided
        name = self.name or f"{self.role.title()} Agent"

        # Generate ID
        member_id = f"{role_lower}_{uuid.uuid4().hex[:8]}"

        # Default budget based on role
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
            allowed_tools=self.allowed_tools,  # Pass through allowed_tools
            agent_category=self.agent_category,
            is_manager=self.is_manager,
            can_delegate=self.is_supervisor,
            priority=self.priority if self.priority else index,
            # Pass through all persona attributes
            backstory=self.backstory,
            expertise=self.expertise,
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
            formation_role=self.formation_role,
        )

        # Auto-attach memory coordinator if memory is enabled
        if member.memory_enabled:
            try:
                from victor.storage.memory import get_memory_coordinator

                member.attach_memory_coordinator(get_memory_coordinator())
            except ImportError:
                pass  # Memory module not available, skip

        return member


class AgentTeam:
    """High-level wrapper for multi-agent team execution.

    AgentTeam provides a simplified API for creating and running teams of
    agents that coordinate to achieve a shared goal. It wraps the existing
    TeamCoordinator infrastructure with a user-friendly interface.

    Attributes:
        name: Team name
        goal: Team's overall objective
        formation: How agents are organized
        config: Underlying TeamConfig

    Example:
        # Create team
        team = await AgentTeam.create(
            orchestrator=orchestrator,
            name="Code Review Team",
            goal="Review and improve the authentication module",
            members=[
                TeamMemberSpec(role="researcher", goal="Find code patterns"),
                TeamMemberSpec(role="reviewer", goal="Identify issues"),
                TeamMemberSpec(role="executor", goal="Fix issues"),
            ],
            formation=TeamFormation.PIPELINE,
        )

        # Run and get result
        result = await team.run()
        print(result.final_output)

        # Or stream events
        async for event in team.stream():
            print(f"{event.type}: {event.message}")
    """

    def __init__(
        self,
        coordinator: TeamCoordinator,
        config: TeamConfig,
    ):
        """Initialize AgentTeam. Use AgentTeam.create() instead.

        Args:
            coordinator: TeamCoordinator for execution
            config: Team configuration
        """
        self._coordinator = coordinator
        self._config = config
        self._result: Optional[TeamResult] = None
        self._events: List[TeamEvent] = []
        self._event_queue: asyncio.Queue[TeamEvent] = asyncio.Queue()

    @classmethod
    async def create(
        cls,
        orchestrator: "AgentOrchestrator",
        name: str,
        goal: str,
        members: List[TeamMemberSpec],
        *,
        formation: TeamFormation = TeamFormation.SEQUENTIAL,
        total_tool_budget: int = 100,
        max_iterations: int = 50,
        timeout_seconds: int = 600,
        shared_context: Optional[Dict[str, Any]] = None,
    ) -> "AgentTeam":
        """Create a new AgentTeam instance.

        Args:
            orchestrator: AgentOrchestrator for spawning sub-agents
            name: Human-readable team name
            goal: Overall team objective
            members: List of team member specifications
            formation: How agents are organized (default: SEQUENTIAL)
            total_tool_budget: Total tool calls across all members
            max_iterations: Maximum total iterations
            timeout_seconds: Maximum execution time
            shared_context: Initial context shared with all members

        Returns:
            Configured AgentTeam instance

        Example:
            team = await AgentTeam.create(
                orchestrator=agent.get_orchestrator(),
                name="Feature Team",
                goal="Implement user authentication",
                members=[
                    TeamMemberSpec(role="researcher", goal="Find patterns"),
                    TeamMemberSpec(role="executor", goal="Write code"),
                ],
            )
        """
        normalized_formation = TeamFormation(getattr(formation, "value", formation))

        # Convert specs to TeamMembers
        team_members = [spec.to_team_member(index=i) for i, spec in enumerate(members)]

        # For hierarchical, ensure we have a supervisor.
        if normalized_formation == TeamFormation.HIERARCHICAL:
            has_supervisor = any(m.is_supervisor for m in team_members)
            if not has_supervisor and team_members:
                # Make the first member the supervisor without discarding its
                # persona, memory, model, or provider settings.
                team_members[0] = replace(
                    team_members[0],
                    agent_category=TeamAgentCategory.SUPERVISOR,
                    is_manager=True,
                    can_delegate=True,
                    max_delegation_depth=max(1, team_members[0].max_delegation_depth),
                )

        # Create config
        config = TeamConfig(
            name=name,
            goal=goal,
            members=team_members,
            formation=normalized_formation,
            total_tool_budget=total_tool_budget,
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
            shared_context=shared_context or {},
        )

        # Create coordinator through unified factory
        coordinator = create_coordinator(
            orchestrator,
            enable_observability=False,
            enable_rl=False,
        )

        return cls(coordinator, config)

    @classmethod
    async def from_agent(
        cls,
        agent: "Agent",
        name: str,
        goal: str,
        members: List[TeamMemberSpec],
        **kwargs: Any,
    ) -> "AgentTeam":
        """Create a team from an existing Agent instance.

        Convenience method that extracts the orchestrator from the agent.

        Args:
            agent: Agent instance
            name: Team name
            goal: Team goal
            members: Team member specifications
            **kwargs: Additional arguments passed to create()

        Returns:
            Configured AgentTeam instance

        Example:
            agent = await Agent.create(provider="anthropic")
            team = await AgentTeam.from_agent(
                agent,
                name="Review Team",
                goal="Review the code",
                members=[...],
            )
        """
        orchestrator = agent.get_orchestrator()
        return await cls.create(
            orchestrator=orchestrator,
            name=name,
            goal=goal,
            members=members,
            **kwargs,
        )

    @classmethod
    async def create_review_team(
        cls,
        orchestrator: "AgentOrchestrator",
        name: str,
        goal: str,
        *,
        writer: TeamMemberSpec,
        reviewer: TeamMemberSpec,
        reviser: Optional[TeamMemberSpec] = None,
        **kwargs: Any,
    ) -> "AgentTeam":
        """Create a cross-vendor review team (writer -> reviewer [-> reviser]).

        A thin preset over :meth:`create` using ``TeamFormation.PIPELINE`` so
        each member's output feeds the next. The point is heterogeneity: give
        ``writer`` and ``reviewer`` different providers (e.g. writer on OpenAI,
        reviewer on Anthropic) so one vendor reviews another's work — the
        Omnigent "Polly" pattern. Emits a warning (does not fail) when the
        writer and reviewer share a provider, since that defeats cross-vendor
        review.

        Args:
            orchestrator: AgentOrchestrator for spawning members.
            name: Team name.
            goal: Overall objective.
            writer: Member that produces the work (set ``provider``/``model``).
            reviewer: Member that reviews it (set a *different* ``provider``).
            reviser: Optional member that applies the review feedback.
            **kwargs: Forwarded to :meth:`create` (formation is fixed PIPELINE).

        Returns:
            Configured AgentTeam in PIPELINE formation.
        """
        if (
            writer.provider
            and reviewer.provider
            and writer.provider.lower() == reviewer.provider.lower()
        ):
            logger.warning(
                "Review team '%s': writer and reviewer share provider '%s'; "
                "cross-vendor review is more effective with different vendors.",
                name,
                writer.provider,
            )

        members = [writer, reviewer] + ([reviser] if reviser is not None else [])
        kwargs.pop("formation", None)
        return await cls.create(
            orchestrator=orchestrator,
            name=name,
            goal=goal,
            members=members,
            formation=TeamFormation.PIPELINE,
            **kwargs,
        )

    @classmethod
    async def create_reflection_team(
        cls,
        orchestrator: "AgentOrchestrator",
        name: str,
        goal: str,
        *,
        generator: TeamMemberSpec,
        critic: TeamMemberSpec,
        rounds: int = 3,
        **kwargs: Any,
    ) -> "AgentTeam":
        """Create an iterative reflection team (generate → critique → refine).

        Uses ``TeamFormation.REFLECTION``: the ``generator`` produces a solution,
        the ``critic`` critiques it, and the generator refines — looping up to
        ``rounds`` times or until the critic is satisfied. Give the two members
        different providers for cross-vendor critique (e.g. generator on OpenAI,
        critic on Anthropic) — the iterative complement to
        :meth:`create_review_team`'s single pass.

        Args:
            orchestrator: AgentOrchestrator for spawning members.
            name: Team name.
            goal: Overall objective.
            generator: Member that produces/refines the solution.
            critic: Member that critiques it (use a *different* provider).
            rounds: Maximum reflection iterations (early-exits on satisfaction).
            **kwargs: Forwarded to :meth:`create` (formation is fixed REFLECTION).

        Returns:
            Configured AgentTeam in REFLECTION formation.
        """
        if (
            generator.provider
            and critic.provider
            and generator.provider.lower() == critic.provider.lower()
        ):
            logger.warning(
                "Reflection team '%s': generator and critic share provider '%s'; "
                "cross-vendor critique is more effective with different vendors.",
                name,
                generator.provider,
            )

        # Bind roles explicitly so context-agent binding never depends on order.
        generator.formation_role = "generator"
        critic.formation_role = "critic"

        kwargs.pop("formation", None)
        shared_context = dict(kwargs.pop("shared_context", None) or {})
        shared_context.setdefault("reflection_max_iterations", rounds)
        return await cls.create(
            orchestrator=orchestrator,
            name=name,
            goal=goal,
            members=[generator, critic],
            formation=TeamFormation.REFLECTION,
            shared_context=shared_context,
            **kwargs,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def name(self) -> str:
        """Get team name."""
        return self._config.name

    @property
    def goal(self) -> str:
        """Get team goal."""
        return self._config.goal

    @property
    def formation(self) -> TeamFormation:
        """Get team formation."""
        return self._config.formation

    @property
    def config(self) -> TeamConfig:
        """Get underlying team configuration."""
        return self._config

    @property
    def result(self) -> Optional[TeamResult]:
        """Get team result (if completed)."""
        return self._result

    @property
    def members(self) -> List[TeamMember]:
        """Get team members."""
        return self._config.members

    # =========================================================================
    # Execution
    # =========================================================================

    async def run(self) -> TeamResult:
        """Execute the team and return the result.

        Blocks until team execution completes.

        Returns:
            TeamResult with execution outcome

        Example:
            result = await team.run()
            if result.success:
                print(result.final_output)
            else:
                print(f"Team failed: {result.member_results}")
        """
        # Set up progress callback
        self._coordinator.set_progress_callback(self._handle_progress)

        # Execute team
        self._result = await self._coordinator.execute_team(
            self._config,
            on_member_complete=self._handle_member_complete,
        )

        return self._result

    async def stream(self) -> AsyncIterator[TeamEvent]:
        """Stream events as the team executes.

        Yields events as team members work, providing real-time
        visibility into team progress.

        Yields:
            TeamEvent objects representing team actions

        Example:
            async for event in team.stream():
                if event.type == TeamEventType.MEMBER_START:
                    print(f"Starting: {event.member_name}")
                elif event.type == TeamEventType.MEMBER_COMPLETE:
                    print(f"Complete: {event.member_name}")
                elif event.type == TeamEventType.TEAM_COMPLETE:
                    print(f"Team finished: {event.message}")
        """
        # Emit start event
        yield TeamEvent(
            type=TeamEventType.TEAM_START,
            team_name=self.name,
            message=f"Starting team '{self.name}' with {len(self.members)} members",
            metadata={"formation": self.formation.value},
        )

        # Set up progress callback to queue events
        self._coordinator.set_progress_callback(self._queue_progress_event)

        # Start execution in background
        execution_task = asyncio.create_task(
            self._coordinator.execute_team(
                self._config,
                on_member_complete=self._queue_member_complete_event,
            )
        )

        # Yield queued events while execution runs
        try:
            while not execution_task.done():
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=0.1,
                    )
                    yield event
                except asyncio.TimeoutError:
                    continue

            # Get final result
            self._result = execution_task.result()

            # Drain remaining events
            while not self._event_queue.empty():
                yield self._event_queue.get_nowait()

            # Emit completion event
            yield TeamEvent(
                type=(
                    TeamEventType.TEAM_COMPLETE
                    if self._result.success
                    else TeamEventType.TEAM_ERROR
                ),
                team_name=self.name,
                message=f"Team '{self.name}' completed: {'success' if self._result.success else 'failed'}",
                metadata={
                    "success": self._result.success,
                    "total_tool_calls": self._result.total_tool_calls,
                    "total_duration": self._result.total_duration,
                },
            )

        except Exception as e:
            yield TeamEvent(
                type=TeamEventType.TEAM_ERROR,
                team_name=self.name,
                error=str(e),
                message=f"Team execution failed: {str(e)}",
            )
            raise

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _handle_progress(
        self,
        team_id: str,
        message: str,
        percent: float,
    ) -> None:
        """Handle progress callback from coordinator.

        Args:
            team_id: Team identifier
            message: Status message
            percent: Progress percentage
        """
        event = TeamEvent(
            type=TeamEventType.MEMBER_PROGRESS,
            team_name=self.name,
            progress=percent,
            message=message,
        )
        self._events.append(event)

    def _handle_member_complete(
        self,
        member_id: str,
        result: MemberResult,
    ) -> None:
        """Handle member completion callback.

        Args:
            member_id: ID of completed member
            result: Member result
        """
        member = self._config.get_member(member_id)
        event = TeamEvent(
            type=(TeamEventType.MEMBER_COMPLETE if result.success else TeamEventType.MEMBER_ERROR),
            team_name=self.name,
            member_id=member_id,
            member_name=member.name if member else member_id,
            result=result,
            error=result.error if not result.success else None,
            message=f"{member.name if member else member_id} completed",
        )
        self._events.append(event)

    def _queue_progress_event(
        self,
        team_id: str,
        message: str,
        percent: float,
    ) -> None:
        """Queue progress event for streaming.

        Args:
            team_id: Team identifier
            message: Status message
            percent: Progress percentage
        """
        # Determine if this is a member start based on message content
        event_type = TeamEventType.MEMBER_PROGRESS
        member_name = None

        if "Running" in message and "..." in message:
            event_type = TeamEventType.MEMBER_START
            member_name = message.replace("Running ", "").replace("...", "").strip()
        elif "Pipeline stage" in message:
            event_type = TeamEventType.MEMBER_START
            # Extract member name from "Pipeline stage N/M: MemberName"
            if ":" in message:
                member_name = message.split(":")[-1].strip()

        event = TeamEvent(
            type=event_type,
            team_name=self.name,
            member_name=member_name,
            progress=percent,
            message=message,
        )
        self._event_queue.put_nowait(event)

    def _queue_member_complete_event(
        self,
        member_id: str,
        result: MemberResult,
    ) -> None:
        """Queue member completion event for streaming.

        Args:
            member_id: ID of completed member
            result: Member result
        """
        member = self._config.get_member(member_id)
        event = TeamEvent(
            type=(TeamEventType.MEMBER_COMPLETE if result.success else TeamEventType.MEMBER_ERROR),
            team_name=self.name,
            member_id=member_id,
            member_name=member.name if member else member_id,
            result=result,
            error=result.error if not result.success else None,
            message=f"{member.name if member else member_id} completed",
        )
        self._event_queue.put_nowait(event)

    # =========================================================================
    # Inspection
    # =========================================================================

    def get_events(self) -> List[TeamEvent]:
        """Get all captured events.

        Returns:
            List of team events
        """
        return self._events.copy()

    def get_member_result(self, member_id: str) -> Optional[MemberResult]:
        """Get result for a specific member.

        Args:
            member_id: Member ID to look up

        Returns:
            MemberResult or None if not found/not complete
        """
        if self._result:
            return self._result.member_results.get(member_id)
        return None

    def __repr__(self) -> str:
        return (
            f"AgentTeam(name={self.name!r}, "
            f"goal={self.goal[:50]!r}..., "
            f"formation={self.formation.value}, "
            f"members={len(self.members)})"
        )


# =============================================================================
# Convenience Constructors
# =============================================================================


def team_start_event(team_name: str, **kwargs: Any) -> TeamEvent:
    """Create a team start event."""
    return TeamEvent(
        type=TeamEventType.TEAM_START,
        team_name=team_name,
        **kwargs,
    )


def team_complete_event(
    team_name: str,
    success: bool = True,
    **kwargs: Any,
) -> TeamEvent:
    """Create a team complete event."""
    return TeamEvent(
        type=TeamEventType.TEAM_COMPLETE if success else TeamEventType.TEAM_ERROR,
        team_name=team_name,
        **kwargs,
    )


def member_start_event(
    team_name: str,
    member_id: str,
    member_name: str,
    **kwargs: Any,
) -> TeamEvent:
    """Create a member start event."""
    return TeamEvent(
        type=TeamEventType.MEMBER_START,
        team_name=team_name,
        member_id=member_id,
        member_name=member_name,
        **kwargs,
    )


def member_complete_event(
    team_name: str,
    member_id: str,
    member_name: str,
    result: MemberResult,
    **kwargs: Any,
) -> TeamEvent:
    """Create a member complete event."""
    return TeamEvent(
        type=(TeamEventType.MEMBER_COMPLETE if result.success else TeamEventType.MEMBER_ERROR),
        team_name=team_name,
        member_id=member_id,
        member_name=member_name,
        result=result,
        error=result.error if not result.success else None,
        **kwargs,
    )


__all__ = [
    # Core classes
    "AgentTeam",
    "TeamMemberSpec",
    "TeamEvent",
    "TeamEventType",
    # Re-exports from teams infrastructure
    "MemoryConfig",
    "TeamFormation",
    "TeamConfig",
    "TeamMember",
    "TeamResult",
    "MemberResult",
    # Event constructors
    "team_start_event",
    "team_complete_event",
    "member_start_event",
    "member_complete_event",
]
