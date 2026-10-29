"""Integration tests for Framework DI Container.

Tests the full integration of framework services working together
through the ServiceContainer, including lifecycle management,
service resolution, and scoped containers.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from victor.core.container import (
    ServiceContainer,
    reset_container,
)
from victor.framework.service_provider import (
    AgentBuilderService,
    AgentSessionService,
    EventRegistryService,
    FrameworkScope,
    FrameworkServiceProvider,
    ToolConfiguratorService,
    configure_framework_services,
    create_builder,
)
from victor.framework.agent_components import (
    AgentSession,
    BuilderPreset,
    SessionLifecycleHooks,
)
from victor.framework.tool_config import (
    AirgappedFilter,
    CostTierFilter,
    ToolConfigurator,
)
from victor.framework.event_registry import EventRegistry
from victor.framework.events import EventType
from victor.framework.task import TaskResult


class TestFrameworkDIIntegration:
    """Integration tests for Framework DI container."""

    @pytest.fixture(autouse=True)
    def reset_global_container(self):
        """Reset global container before each test."""
        reset_container()
        yield
        reset_container()

    def test_configure_services_creates_all_services(self):
        """Test that configuring services creates all expected services."""
        container = ServiceContainer()
        configure_framework_services(container)

        # Verify all services registered
        assert container.is_registered(ToolConfiguratorService)
        assert container.is_registered(EventRegistryService)
        assert container.is_registered(AgentBuilderService)
        assert container.is_registered(AgentSessionService)

    def test_tool_configurator_singleton(self):
        """Test that ToolConfigurator is a singleton."""
        container = ServiceContainer()
        configure_framework_services(container)

        config1 = container.get(ToolConfiguratorService)
        config2 = container.get(ToolConfiguratorService)

        assert config1 is config2

    def test_event_registry_singleton(self):
        """Test that EventRegistry is a singleton."""
        container = ServiceContainer()
        configure_framework_services(container)

        registry1 = container.get(EventRegistryService)
        registry2 = container.get(EventRegistryService)

        assert registry1 is registry2

    def test_agent_builder_transient(self):
        """Test that AgentBuilder is transient (new instance each time)."""
        container = ServiceContainer()
        configure_framework_services(container)

        builder1 = container.get(AgentBuilderService)
        builder2 = container.get(AgentBuilderService)

        assert builder1 is not builder2

    def test_builder_has_container_reference(self):
        """Test that builder receives container reference for DI."""
        container = ServiceContainer()
        configure_framework_services(container)

        builder = container.get(AgentBuilderService)

        assert hasattr(builder, "_container")
        assert builder._container is container


class TestFrameworkScopeIntegration:
    """Integration tests for FrameworkScope."""

    @pytest.fixture(autouse=True)
    def reset_global_container(self):
        """Reset global container before each test."""
        reset_container()
        yield
        reset_container()

    def test_scope_provides_same_singleton_as_container(self):
        """Test that scoped container returns same singletons."""
        container = ServiceContainer()
        configure_framework_services(container)

        parent_config = container.get(ToolConfiguratorService)

        with FrameworkScope(container) as scope:
            scoped_config = scope.get_configurator()
            assert scoped_config is parent_config

    def test_scope_provides_fresh_transients(self):
        """Test that scoped container returns fresh transient instances."""
        container = ServiceContainer()
        configure_framework_services(container)

        parent_builder = container.get(AgentBuilderService)

        with FrameworkScope(container) as scope:
            scoped_builder = scope.get_builder()
            assert scoped_builder is not parent_builder

    def test_multiple_scopes_isolated(self):
        """Test that multiple scopes are isolated."""
        container = ServiceContainer()
        configure_framework_services(container)

        with FrameworkScope(container) as scope1:
            builder1 = scope1.get_builder()

        with FrameworkScope(container) as scope2:
            builder2 = scope2.get_builder()

        # Different scope = different builder instance
        assert builder1 is not builder2

    @pytest.mark.asyncio
    async def test_async_scope_context_manager(self):
        """Test async context manager for FrameworkScope."""
        container = ServiceContainer()
        configure_framework_services(container)

        async with FrameworkScope(container) as scope:
            configurator = scope.get_configurator()
            registry = scope.get_registry()
            builder = scope.get_builder()

            assert configurator is not None
            assert registry is not None
            assert builder is not None


class TestBuilderWithContainerIntegration:
    """Integration tests for AgentBuilder with ServiceContainer."""

    @pytest.fixture(autouse=True)
    def reset_global_container(self):
        """Reset global container before each test."""
        reset_container()
        yield
        reset_container()

    def test_builder_uses_tool_configurator_from_container(self):
        """Test that builder uses ToolConfigurator from container."""
        container = ServiceContainer()
        configure_framework_services(container)

        # Get configurator and add filter
        configurator = container.get(ToolConfiguratorService)
        configurator.add_filter(AirgappedFilter())

        # Builder should be able to access the same configurator
        builder = container.get(AgentBuilderService)

        # Verify builder has container reference
        assert builder._container is container

    def test_builder_chain_with_container(self):
        """Test builder method chaining with container integration."""
        container = ServiceContainer()
        configure_framework_services(container)

        builder = create_builder(container)

        # Chain methods should work
        result = (
            builder.provider("anthropic")
            .model("claude-3")
            .tools(["read", "write"])
            .add_tool_filter(MagicMock())
            .thinking(True)
        )

        assert result is builder
        assert builder._options.provider == "anthropic"
        assert builder._options.model == "claude-3"

    def test_preset_configuration_with_container(self):
        """Test preset configuration with container."""
        container = ServiceContainer()
        configure_framework_services(container)

        builder = create_builder(container)
        builder.preset(BuilderPreset.AIRGAPPED)

        assert builder._options.airgapped is True


class TestSessionWithContainerIntegration:
    """Integration tests for AgentSession with ServiceContainer."""

    @pytest.fixture(autouse=True)
    def reset_global_container(self):
        """Reset global container before each test."""
        reset_container()
        yield
        reset_container()

    @pytest.fixture
    def mock_agent(self):
        """Create mock agent for session tests."""
        agent = MagicMock()
        agent._orchestrator = MagicMock()
        agent._orchestrator.messages = []
        agent.run = AsyncMock(
            return_value=TaskResult(
                content="Test response",
                success=True,
                tool_calls=[],
            )
        )
        agent.reset = AsyncMock()
        return agent

    def test_session_with_scoped_container(self, mock_agent):
        """Test session creates scoped container."""
        container = ServiceContainer()
        configure_framework_services(container)

        session = AgentSession(
            mock_agent,
            "Initial prompt",
            container=container,
        )

        assert session.scope is not None

    @pytest.mark.asyncio
    async def test_session_scope_disposed_on_close(self, mock_agent):
        """Test session disposes scoped container on close."""
        container = ServiceContainer()
        configure_framework_services(container)

        session = AgentSession(
            mock_agent,
            "Initial prompt",
            container=container,
        )

        await session.close()

        assert session._scope is None

    @pytest.mark.asyncio
    async def test_session_lifecycle_with_container(self, mock_agent):
        """Test full session lifecycle with container integration."""
        container = ServiceContainer()
        configure_framework_services(container)

        events = []
        hooks = SessionLifecycleHooks(
            on_start=lambda s: events.append("start"),
            on_close=lambda s, m: events.append(f"close:{m.total_turns}"),
        )

        session = AgentSession(
            mock_agent,
            "Initial prompt",
            hooks=hooks,
            container=container,
        )

        # Execute a turn
        await session.send("Test message")

        # Close session
        await session.close()

        assert events == ["start", "close:1"]

    @pytest.mark.asyncio
    async def test_session_metrics_tracking(self, mock_agent):
        """Test session metrics are tracked correctly."""
        container = ServiceContainer()
        configure_framework_services(container)

        session = AgentSession(
            mock_agent,
            "Initial prompt",
            container=container,
        )

        # Execute multiple turns
        await session.send("Turn 1")
        await session.send("Turn 2")

        assert session.metrics.total_turns == 2
        assert session.metrics.successful_turns == 2


class TestToolConfiguratorIntegration:
    """Integration tests for ToolConfigurator with DI."""

    @pytest.fixture(autouse=True)
    def reset_global_container(self):
        """Reset global container before each test."""
        reset_container()
        yield
        reset_container()

    def test_configurator_with_filters(self):
        """Test configurator with multiple filters."""
        container = ServiceContainer()
        configure_framework_services(container)

        configurator = container.get(ToolConfiguratorService)

        # Add filters
        airgapped_filter = AirgappedFilter()
        cost_filter = CostTierFilter()

        configurator.add_filter(airgapped_filter)
        configurator.add_filter(cost_filter)

        # Filters should be added
        assert airgapped_filter in configurator._filters
        assert cost_filter in configurator._filters

    def test_configurator_filter_removal(self):
        """Test removing filters from configurator."""
        container = ServiceContainer()
        configure_framework_services(container)

        configurator = container.get(ToolConfiguratorService)

        filter_instance = AirgappedFilter()
        configurator.add_filter(filter_instance)
        removed = configurator.remove_filter(filter_instance)

        assert removed is True
        assert filter_instance not in configurator._filters


class TestEventRegistryIntegration:
    """Integration tests for EventRegistry with DI."""

    @pytest.fixture(autouse=True)
    def reset_global_container(self):
        """Reset global container before each test."""
        reset_container()
        EventRegistry._instance = None  # Reset singleton
        yield
        reset_container()
        EventRegistry._instance = None

    def test_registry_converter_registration(self):
        """Test registering converters with registry."""
        container = ServiceContainer()
        configure_framework_services(container)

        registry = container.get(EventRegistryService)

        # Default converters should be registered
        content_converter = registry.get_converter(EventType.CONTENT)
        assert content_converter is not None

    def test_registry_event_conversion(self):
        """Test converting events through registry."""
        container = ServiceContainer()
        configure_framework_services(container)

        registry = container.get(EventRegistryService)

        # Get a converter
        converter = registry.get_converter(EventType.CONTENT)
        assert converter is not None


class TestFullStackIntegration:
    """End-to-end integration tests for the full DI stack."""

    @pytest.fixture(autouse=True)
    def reset_global_container(self):
        """Reset global container before each test."""
        reset_container()
        EventRegistry._instance = None
        yield
        reset_container()
        EventRegistry._instance = None

    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        agent = MagicMock()
        agent._orchestrator = MagicMock()
        agent._orchestrator.messages = []
        agent.run = AsyncMock(
            return_value=TaskResult(
                content="Response",
                success=True,
                tool_calls=[{"name": "read"}],
            )
        )
        agent.reset = AsyncMock()
        return agent

    @pytest.mark.asyncio
    async def test_full_workflow_with_di(self, mock_agent):
        """Test complete workflow using DI container."""
        # Setup container
        container = ServiceContainer()
        configure_framework_services(container)

        # Get services
        configurator = container.get(ToolConfiguratorService)
        container.get(EventRegistryService)
        container.get(AgentBuilderService)

        # Configure tool filters
        configurator.add_filter(AirgappedFilter())

        # Create session with hooks
        lifecycle_events = []
        hooks = SessionLifecycleHooks(
            on_start=lambda s: lifecycle_events.append("session_started"),
            on_turn_end=lambda s, r: lifecycle_events.append(f"turn_complete:{r.tool_count}"),
            on_close=lambda s, m: lifecycle_events.append(f"session_closed:{m.total_turns}"),
        )

        # Create session
        session = AgentSession(
            mock_agent,
            "Analyze code",
            hooks=hooks,
            container=container,
        )

        # Execute workflow
        await session.send("What do you see?")
        await session.close()

        # Verify lifecycle events
        assert "session_started" in lifecycle_events
        assert "turn_complete:1" in lifecycle_events
        assert "session_closed:1" in lifecycle_events

        # Verify metrics
        assert session.metrics.total_turns == 1
        assert session.metrics.total_tool_calls == 1

    @pytest.mark.asyncio
    async def test_scoped_workflow_isolation(self, mock_agent):
        """Test that scoped workflows are properly isolated."""
        container = ServiceContainer()
        configure_framework_services(container)

        # Create two scopes with different sessions
        events1 = []
        events2 = []

        async with FrameworkScope(container):
            session1 = AgentSession(
                mock_agent,
                "Session 1",
                hooks=SessionLifecycleHooks(
                    on_start=lambda s: events1.append("start"),
                ),
            )
            await session1.send("msg1")

        async with FrameworkScope(container):
            session2 = AgentSession(
                mock_agent,
                "Session 2",
                hooks=SessionLifecycleHooks(
                    on_start=lambda s: events2.append("start"),
                ),
            )
            await session2.send("msg2")

        # Each scope should have its own events
        assert events1 == ["start"]
        assert events2 == ["start"]

    def test_service_provider_selective_registration(self):
        """Test selective service registration."""
        container = ServiceContainer()

        # Only register some services
        provider = FrameworkServiceProvider(
            include_tool_config=True,
            include_event_registry=False,
            include_builder=True,
            include_bridge=False,
        )
        provider.register_services(container)

        assert container.is_registered(ToolConfiguratorService)
        assert not container.is_registered(EventRegistryService)
        assert container.is_registered(AgentBuilderService)
        assert not container.is_registered(AgentSessionService)

    def test_replace_existing_registration(self):
        """Test replacing existing service registration."""
        container = ServiceContainer()

        # Register a mock first
        mock_configurator = MagicMock()
        container.register(ToolConfiguratorService, lambda c: mock_configurator)

        # Replace with real implementation
        provider = FrameworkServiceProvider()
        provider.register_services(container, replace_existing=True)

        # Should have real implementation now
        resolved = container.get(ToolConfiguratorService)
        assert resolved is not mock_configurator
        assert isinstance(resolved, ToolConfigurator)
