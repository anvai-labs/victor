"""Protocol definitions for agent factory."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    runtime_checkable,
)

__all__ = [
    "IAgentFactory",
    "IAgent",
]


@runtime_checkable
class IAgentFactory(Protocol):
    """Protocol for unified agent creation.

    Defines interface for creating ANY agent type (foreground, background, team_member)
    using shared infrastructure. This protocol enables:

    - **Single Responsibility**: Factory only handles agent creation
    - **Open/Closed**: Extensible for new agent types via mode parameter
    - **Liskov Substitution**: Any IAgentFactory implementation is interchangeable
    - **Interface Segregation**: Focused protocol with single method
    - **Dependency Inversion**: Consumers depend on abstraction, not concrete factory

    All agent entrypoints (Agent.create, BackgroundAgentManager.start_agent, etc.)
    delegate to implementations of this protocol, ensuring consistent code maintenance
    and eliminating code proliferation.

    Usage:
        from victor.agent.protocols import IAgentFactory

        async def create_researcher(factory: IAgentFactory) -> IAgent:
            return await factory.create_agent(
                mode="foreground",
                task="research",
                config=my_config
            )
    """

    async def create_agent(
        self,
        mode: str,
        config: Optional[Any] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Create any agent type using shared infrastructure.

        This is the ONLY method that should create agents. All other entrypoints
        (Agent.create, BackgroundAgentManager.start_agent, Vertical.create_agent)
        must delegate here to ensure consistent code maintenance.

        Args:
            mode: Agent creation mode - "foreground", "background", or "team_member"
            config: Optional unified agent configuration (UnifiedAgentConfig)
            task: Optional task description (for background agents)
            **kwargs: Additional agent-specific parameters

        Returns:
            Agent instance (Agent, BackgroundAgent, or TeamMember/SubAgent)

        Examples:
            # Foreground agent
            agent = await factory.create_agent(mode="foreground")

            # Background agent with task
            agent = await factory.create_agent(
                mode="background",
                task="Implement feature X"
            )

            # Team member
            agent = await factory.create_agent(
                mode="team_member",
                role="researcher"
            )
        """
        ...


@runtime_checkable
class IAgent(Protocol):
    """Canonical agent protocol.

    ALL agent types (Agent, BackgroundAgent, TeamMember, SubAgent) must implement
    this protocol to ensure Liskov Substitution Principle compliance.

    This enables:
    - Polymorphic agent handling
    - Type-safe agent composition
    - Consistent agent interfaces across all contexts
    """

    @property
    def id(self) -> str:
        """Unique agent identifier."""
        ...

    @property
    def orchestrator(self) -> Any:
        """Agent orchestrator instance."""
        ...

    async def execute(self, task: str, context: Dict[str, Any]) -> str:
        """Execute a task.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Execution result
        """
        ...
