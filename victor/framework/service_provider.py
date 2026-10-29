"""Framework Service Provider for Dependency Injection.

This module integrates the Framework API layer with Victor's ServiceContainer,
enabling proper lifecycle management, testability, and dependency injection.

Design Patterns:
- Service Provider Pattern: Bundles related service registrations
- Factory Pattern: Lazy creation of expensive services
- Composition Root Pattern: Centralizes dependency graph construction

Usage:
    from victor.core.container import ServiceContainer
    from victor.framework.service_provider import (
        FrameworkServiceProvider,
        configure_framework_services,
    )

    # Option 1: Use convenience function
    container = ServiceContainer()
    configure_framework_services(container)

    # Option 2: Use provider directly
    provider = FrameworkServiceProvider()
    provider.register_services(container)

    # Resolve services
    configurator = container.get(ToolConfiguratorService)
    registry = container.get(EventRegistryService)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    Type,
    runtime_checkable,
)

from victor.core.container import (
    ServiceContainer,
    ServiceLifetime,
    get_container,
)

if TYPE_CHECKING:
    from victor.framework.agent_components import (
        AgentBuilder,
        AgentBridge,
    )
    from victor.framework.event_registry import EventRegistry
    from victor.framework.tool_config import ToolConfigurator


logger = logging.getLogger(__name__)


# =============================================================================
# Service Protocols for Framework Components
# =============================================================================


@runtime_checkable
class ToolConfiguratorService(Protocol):
    """Protocol for tool configuration services."""

    def configure_from_toolset(self, orchestrator: Any, toolset: Any) -> Any:
        """Configure tools from a ToolSet."""
        ...

    def configure(self, orchestrator: Any, tools: Any, mode: Any) -> Any:
        """Configure tools with specified mode."""
        ...

    def add_filter(self, filter: Any) -> None:
        """Add a tool filter."""
        ...

    def remove_filter(self, filter: Any) -> bool:
        """Remove a tool filter."""
        ...


@runtime_checkable
class EventRegistryService(Protocol):
    """Protocol for event registry services."""

    def register_converter(self, event_type: Any, converter: Any) -> None:
        """Register an event converter."""
        ...

    def get_converter(self, event_type: Any) -> Any:
        """Get converter for an event type."""
        ...

    def convert(self, event: Any, target: Any) -> Any:
        """Convert an event to target format."""
        ...


@runtime_checkable
class AgentBuilderService(Protocol):
    """Protocol for agent builder services."""

    def preset(self, preset: Any) -> "AgentBuilderService":
        """Apply a preset configuration."""
        ...

    def provider(self, name: str) -> "AgentBuilderService":
        """Set provider."""
        ...

    def model(self, name: str) -> "AgentBuilderService":
        """Set model."""
        ...

    async def build(self) -> Any:
        """Build the agent."""
        ...


@runtime_checkable
class AgentSessionService(Protocol):
    """Protocol for agent session services."""

    async def send(self, message: str) -> Any:
        """Send a message."""
        ...

    def pause(self) -> None:
        """Pause the session."""
        ...

    def resume(self) -> None:
        """Resume the session."""
        ...

    async def close(self) -> None:
        """Close the session."""
        ...


# =============================================================================
# Service Factory Functions
# =============================================================================


def _create_tool_configurator(container: ServiceContainer) -> "ToolConfigurator":
    """Factory for creating ToolConfigurator."""
    from victor.framework.tool_config import ToolConfigurator

    return ToolConfigurator()


def _create_event_registry(container: ServiceContainer) -> "EventRegistry":
    """Factory for creating EventRegistry.

    Returns the singleton instance to maintain registered converters.
    """
    from victor.framework.event_registry import get_event_registry

    return get_event_registry()


def _create_agent_builder(container: ServiceContainer) -> "AgentBuilder":
    """Factory for creating AgentBuilder.

    Creates a new builder instance (transient) that can use
    container services for configuration.
    """
    from victor.framework.agent_components import AgentBuilder

    # Create builder with optional container reference
    builder = AgentBuilder()
    # Store container reference for dependency injection during build
    builder._container = container
    return builder


def _create_agent_bridge(container: ServiceContainer) -> "AgentBridge":
    """Factory for creating AgentBridge.

    Creates a bridge configured with container services.
    """
    from victor.framework.agent_components import AgentBridge

    return AgentBridge()


# =============================================================================
# Service Provider Implementation
# =============================================================================


@dataclass
class ServiceRegistration:
    """Describes a service to register."""

    service_type: Type
    factory: Callable[[ServiceContainer], Any]
    lifetime: ServiceLifetime
    description: str


class FrameworkServiceProvider:
    """Service provider that registers all framework services.

    Following the Service Provider pattern, this class bundles all
    framework-related service registrations into a single composable unit.

    Example:
        container = ServiceContainer()
        provider = FrameworkServiceProvider()
        provider.register_services(container)

        # Or with customization
        provider = FrameworkServiceProvider(
            include_tool_config=True,
            include_event_registry=True,
            include_builder=True,
        )
        provider.register_services(container)
    """

    def __init__(
        self,
        include_tool_config: bool = True,
        include_event_registry: bool = True,
        include_builder: bool = True,
        include_bridge: bool = True,
    ) -> None:
        """Initialize provider with options.

        Args:
            include_tool_config: Register ToolConfigurator service
            include_event_registry: Register EventRegistry service
            include_builder: Register AgentBuilder service
            include_bridge: Register AgentBridge service
        """
        self._include_tool_config = include_tool_config
        self._include_event_registry = include_event_registry
        self._include_builder = include_builder
        self._include_bridge = include_bridge
        self._registered = False

    def get_registrations(self) -> list[ServiceRegistration]:
        """Get list of service registrations.

        Returns:
            List of ServiceRegistration objects
        """
        registrations = []

        if self._include_tool_config:
            registrations.append(
                ServiceRegistration(
                    service_type=ToolConfiguratorService,
                    factory=_create_tool_configurator,
                    lifetime=ServiceLifetime.SINGLETON,
                    description="Tool configuration service",
                )
            )

        if self._include_event_registry:
            registrations.append(
                ServiceRegistration(
                    service_type=EventRegistryService,
                    factory=_create_event_registry,
                    lifetime=ServiceLifetime.SINGLETON,
                    description="Event registry service",
                )
            )

        if self._include_builder:
            registrations.append(
                ServiceRegistration(
                    service_type=AgentBuilderService,
                    factory=_create_agent_builder,
                    lifetime=ServiceLifetime.TRANSIENT,
                    description="Agent builder service (transient)",
                )
            )

        if self._include_bridge:
            registrations.append(
                ServiceRegistration(
                    service_type=AgentSessionService,
                    factory=_create_agent_bridge,
                    lifetime=ServiceLifetime.SCOPED,
                    description="Agent bridge service (scoped)",
                )
            )

        return registrations

    def register_services(
        self,
        container: ServiceContainer,
        replace_existing: bool = False,
    ) -> None:
        """Register all framework services with the container.

        Args:
            container: Target service container
            replace_existing: If True, replace existing registrations
        """
        registrations = self.get_registrations()

        for reg in registrations:
            try:
                if replace_existing:
                    container.register_or_replace(
                        reg.service_type,
                        reg.factory,
                        reg.lifetime,
                    )
                else:
                    container.register(
                        reg.service_type,
                        reg.factory,
                        reg.lifetime,
                    )
                logger.debug(f"Registered {reg.description}")
            except Exception as e:
                logger.warning(f"Failed to register {reg.service_type.__name__}: {e}")

        self._registered = True

    @property
    def is_registered(self) -> bool:
        """Check if services have been registered."""
        return self._registered


# =============================================================================
# Scoped Session Management
# =============================================================================


class FrameworkScope:
    """Scoped container for framework services.

    Provides request-level isolation for framework services,
    particularly useful for managing agent sessions.

    Example:
        container = ServiceContainer()
        configure_framework_services(container)

        async with FrameworkScope(container) as scope:
            builder = scope.get_builder()
            agent = await builder.provider("anthropic").build()
            result = await agent.run("Hello")
    """

    def __init__(self, container: ServiceContainer) -> None:
        """Initialize framework scope.

        Args:
            container: Parent service container
        """
        self._container = container
        self._scope = container.create_scope()
        self._active = True

    def get_configurator(self) -> "ToolConfigurator":
        """Get tool configurator service."""
        return self._scope.get(ToolConfiguratorService)

    def get_registry(self) -> "EventRegistry":
        """Get event registry service."""
        return self._scope.get(EventRegistryService)

    def get_builder(self) -> "AgentBuilder":
        """Get agent builder service.

        Returns a new builder instance (transient).
        """
        return self._scope.get(AgentBuilderService)

    async def __aenter__(self) -> "FrameworkScope":
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context and cleanup."""
        self._scope.dispose()
        self._active = False

    def __enter__(self) -> "FrameworkScope":
        """Enter sync context."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit sync context and cleanup."""
        self._scope.dispose()
        self._active = False


# =============================================================================
# Convenience Functions
# =============================================================================


def configure_framework_services(
    container: Optional[ServiceContainer] = None,
    replace_existing: bool = False,
) -> ServiceContainer:
    """Configure all framework services in a container.

    Convenience function for quick setup.

    Args:
        container: Target container (uses global if None)
        replace_existing: If True, replace existing registrations

    Returns:
        Configured service container

    Example:
        # Use global container
        container = configure_framework_services()

        # Use custom container
        my_container = ServiceContainer()
        configure_framework_services(my_container)
    """
    if container is None:
        container = get_container()

    provider = FrameworkServiceProvider()
    provider.register_services(container, replace_existing=replace_existing)

    return container


def get_tool_configurator(
    container: Optional[ServiceContainer] = None,
) -> "ToolConfigurator":
    """Get tool configurator from container.

    Args:
        container: Container to use (global if None)

    Returns:
        ToolConfigurator instance
    """
    if container is None:
        container = get_container()
    return container.get(ToolConfiguratorService)


def get_event_registry(container: Optional[ServiceContainer] = None) -> "EventRegistry":
    """Get event registry from container.

    Args:
        container: Container to use (global if None)

    Returns:
        EventRegistry instance
    """
    if container is None:
        container = get_container()
    return container.get(EventRegistryService)


def create_builder(container: Optional[ServiceContainer] = None) -> "AgentBuilder":
    """Create new agent builder from container.

    Args:
        container: Container to use (global if None)

    Returns:
        New AgentBuilder instance
    """
    if container is None:
        container = get_container()
    return container.get(AgentBuilderService)


def create_framework_scope(
    container: Optional[ServiceContainer] = None,
) -> FrameworkScope:
    """Create a scoped container for framework services.

    Args:
        container: Parent container (global if None)

    Returns:
        FrameworkScope instance

    Example:
        with create_framework_scope() as scope:
            builder = scope.get_builder()
            # ... use builder
    """
    if container is None:
        container = get_container()
    return FrameworkScope(container)
