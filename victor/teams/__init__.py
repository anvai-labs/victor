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

"""Victor Teams - Unified multi-agent team coordination.

This is the recommended import location for all team functionality.
Import from here instead of victor.framework or victor.agent.teams.

**Basic Usage**:
    from victor.teams import (
        TeamFormation,
        MessageType,
        AgentMessage,
        UnifiedTeamCoordinator,
        create_coordinator,
    )

    # Create a coordinator
    coordinator = create_coordinator(orchestrator)
    coordinator.add_member(researcher).add_member(executor)
    coordinator.set_formation(TeamFormation.PIPELINE)
    result = await coordinator.execute_task("Implement feature", {})

**Formation Patterns**:
    # Sequential: Chain agents one after another
    coordinator.set_formation(TeamFormation.SEQUENTIAL)

    # Parallel: Execute all agents simultaneously
    coordinator.set_formation(TeamFormation.PARALLEL)

    # Hierarchical: Supervisor delegates to specialists
    coordinator.set_formation(TeamFormation.HIERARCHICAL)

    # Pipeline: Processing pipeline with output passing
    coordinator.set_formation(TeamFormation.PIPELINE)

    # Consensus: Multi-round agreement building
    coordinator.set_formation(TeamFormation.CONSENSUS)

**Testing**:
    # Lightweight coordinator for testing (no dependencies)
    coordinator = create_coordinator(lightweight=True)

**Production**:
    # Full-featured coordinator with observability and RL
    coordinator = create_coordinator(
        orchestrator=my_orchestrator,
        enable_observability=True,
        enable_rl=True,
    )

Factory Functions:
    create_coordinator: Create appropriate coordinator based on requirements

Types (canonical):
    TeamFormation: Team organization patterns
    TeamAgentCategory: Member coordination categories
    MessageType: Message types for inter-agent communication
    AgentMessage: Agent message structure
    MemberResult: Result from member execution
    TeamResult: Result from team execution

Protocols:
    ITeamCoordinator: Base coordinator protocol
    ITeamMember: Team member protocol
    IObservableCoordinator: Observability protocol
    IRLCoordinator: RL integration protocol
    IEnhancedTeamCoordinator: Full-featured protocol

Coordinators:
    UnifiedTeamCoordinator: Production coordinator with all features

Mixins:
    ObservabilityMixin: Add observability to custom coordinators
    RLMixin: Add RL integration to custom coordinators

Documentation:
    - MIGRATION_GUIDE.md: Step-by-step migration instructions
    - RELEASE_NOTES.md: Release notes and changes
    - CONSOLIDATION.md: Architecture consolidation details
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

# Canonical types - single source of truth
from victor.teams.types import (
    AgentMessage,
    MemberResult,
    MessagePriority,
    MessageType,
    TeamAgentCategory,
    TeamFormation,
    TeamParticipant,
    TeamResult,
)

# Protocols (import from canonical location to avoid circular import)
# Note: victor.teams.protocols is a re-export shim - import from victor.protocols.team directly
# Deferred import to avoid circular dependency with victor.protocols.team
if TYPE_CHECKING:
    from victor.protocols.team import (
        IAgent,
        IDelegateFollowUpCoordinator,
        IEnhancedTeamCoordinator,
        IMessageBusProvider,
        IObservableCoordinator,
        IRLCoordinator,
        ISharedMemoryProvider,
        ISupervisorAgent,
        ITeamCoordinator,
        ITeamMember,
        TeamCoordinatorProtocol,
        TeamMemberProtocol,
    )
else:
    # Lazy load protocols at runtime to break circular import
    # victor.protocols.team imports from victor.teams.types, which creates a cycle
    # if victor.teams.__init__ imports from victor.protocols.team at module level
    IAgent = None  # type: ignore
    IDelegateFollowUpCoordinator = None  # type: ignore
    IEnhancedTeamCoordinator = None  # type: ignore
    IMessageBusProvider = None  # type: ignore
    IObservableCoordinator = None  # type: ignore
    IRLCoordinator = None  # type: ignore
    ISharedMemoryProvider = None  # type: ignore
    ISupervisorAgent = None  # type: ignore
    ITeamCoordinator = None  # type: ignore
    ITeamMember = None  # type: ignore
    TeamCoordinatorProtocol = None  # type: ignore
    TeamMemberProtocol = None  # type: ignore

# Coordinator: lazy import to break circular dependency with
# victor.coordination.formations.base → victor.teams.types
# UnifiedTeamCoordinator is accessed via create_coordinator() or
# direct import from victor.teams.unified_coordinator.

# Framework coordinator (for testing/lightweight usage)
# Note: Imported in create_coordinator() to avoid circular dependency

# Communication infrastructure (lazy to avoid agent.teams → coordinator chain)
# Import at call site instead of module level.
# from victor.agent.teams.communication import TeamMessageBus, TeamSharedMemory

# Team configuration types (canonical)
from victor.teams.types import TeamConfig, TeamMember

# Mixins
from victor.teams.mixins import ObservabilityMixin, RLMixin

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator


def create_coordinator(
    orchestrator: Optional["AgentOrchestrator"] = None,
    lightweight: bool = False,
    enable_observability: bool = False,
    enable_rl: bool = False,
) -> ITeamCoordinator:
    """Create a team coordinator instance.

    This is a factory function that creates the appropriate coordinator
    based on the provided parameters. Use this instead of directly
    instantiating coordinator classes.

    Args:
        orchestrator: The agent orchestrator for coordination context.
            Required for production coordinators.
        lightweight: If True, create a lightweight coordinator for testing.
            Lightweight coordinators have minimal dependencies and are
            suitable for unit tests.
        enable_observability: If True, enable observability features
            (metrics, traces, events).
        enable_rl: If True, enable reinforcement learning features.

    Returns:
        A coordinator instance implementing ITeamCoordinator.

    Example:
        from victor.teams import create_coordinator

        # Production coordinator with full features
        coordinator = create_coordinator(
            orchestrator=my_orchestrator,
            enable_observability=True,
            enable_rl=True,
        )

        # Lightweight coordinator for testing
        coordinator = create_coordinator(lightweight=True)
    """
    if lightweight:
        # For testing, use UnifiedTeamCoordinator in lightweight mode
        from victor.teams.unified_coordinator import UnifiedTeamCoordinator

        return UnifiedTeamCoordinator(
            orchestrator=None,
            enable_observability=False,
            enable_rl=False,
        )
    else:
        # Production coordinator
        if orchestrator is None:
            raise ValueError(
                "orchestrator is required for production coordinators. "
                "Use lightweight=True for testing without an orchestrator."
            )

        # Create UnifiedTeamCoordinator with requested features
        from victor.teams.unified_coordinator import UnifiedTeamCoordinator

        return UnifiedTeamCoordinator(
            orchestrator=orchestrator,
            enable_observability=enable_observability,
            enable_rl=enable_rl,
        )


def __getattr__(name: str) -> Any:
    """Lazy load protocols to avoid circular import.

    This function is called when an attribute is not found in the module.
    It lazily imports the protocols from victor.protocols.team only when
    they are actually accessed, breaking the circular import cycle.
    """
    if name == "UnifiedTeamCoordinator":
        from victor.teams.unified_coordinator import UnifiedTeamCoordinator

        globals()["UnifiedTeamCoordinator"] = UnifiedTeamCoordinator
        return UnifiedTeamCoordinator
    if name == "TeamCoordinator":
        # Deprecated compatibility shim (legacy integrations/tests).
        # New code should use create_coordinator() + execute_team_config/execute_task.
        from victor.agent.teams.coordinator import TeamCoordinator

        globals()["TeamCoordinator"] = TeamCoordinator
        return TeamCoordinator
    if name == "StateGraphNodeConfig":
        from victor.teams.unified_coordinator import StateGraphNodeConfig

        globals()["StateGraphNodeConfig"] = StateGraphNodeConfig
        return StateGraphNodeConfig
    if name in {
        "IAgent",
        "IDelegateFollowUpCoordinator",
        "IEnhancedTeamCoordinator",
        "IMessageBusProvider",
        "IObservableCoordinator",
        "IRLCoordinator",
        "ISharedMemoryProvider",
        "ISupervisorAgent",
        "ITeamCoordinator",
        "ITeamMember",
        "TeamCoordinatorProtocol",
        "TeamMemberProtocol",
    }:
        from victor.protocols.team import (
            IAgent,
            IDelegateFollowUpCoordinator,
            IEnhancedTeamCoordinator,
            IMessageBusProvider,
            IObservableCoordinator,
            IRLCoordinator,
            ISharedMemoryProvider,
            ISupervisorAgent,
            ITeamCoordinator,
            ITeamMember,
            TeamCoordinatorProtocol,
            TeamMemberProtocol,
        )

        # Store in module globals for future access
        globals()[name] = locals()[name]
        return locals()[name]
    # Communication infrastructure (lazy to avoid agent.teams → coordinator chain)
    if name in {"TeamMessageBus", "TeamSharedMemory"}:
        from victor.agent.teams.communication import TeamMessageBus, TeamSharedMemory

        globals()["TeamMessageBus"] = TeamMessageBus
        globals()["TeamSharedMemory"] = TeamSharedMemory
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
