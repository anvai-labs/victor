"""Tests for victor/framework/service_provider.py.

Tests the Dependency Injection integration for framework components.
"""

import pytest
from unittest.mock import MagicMock

from victor.core.container import (
    ServiceContainer,
    ServiceLifetime,
    reset_container,
)
from victor.framework.service_provider import (
    AgentBuilderService,
    AgentSessionService,
    EventRegistryService,
    FrameworkScope,
    FrameworkServiceProvider,
    ServiceRegistration,
    ToolConfiguratorService,
    configure_framework_services,
    create_builder,
    create_framework_scope,
    get_event_registry,
    get_tool_configurator,
)


class TestServiceRegistration:
    """Test ServiceRegistration dataclass."""

    def test_registration_creation(self):
        """Test creating a service registration."""
        reg = ServiceRegistration(
            service_type=ToolConfiguratorService,
            factory=lambda c: MagicMock(),
            lifetime=ServiceLifetime.SINGLETON,
            description="Test service",
        )

        assert reg.service_type == ToolConfiguratorService
        assert reg.lifetime == ServiceLifetime.SINGLETON
        assert reg.description == "Test service"
        assert callable(reg.factory)


class TestFrameworkServiceProvider:
    """Test FrameworkServiceProvider class."""

    def test_provider_creation_defaults(self):
        """Test provider with default options."""
        provider = FrameworkServiceProvider()

        registrations = provider.get_registrations()

        # Should have all services by default
        service_types = [r.service_type for r in registrations]
        assert ToolConfiguratorService in service_types
        assert EventRegistryService in service_types
        assert AgentBuilderService in service_types
        assert AgentSessionService in service_types

    def test_provider_creation_selective(self):
        """Test provider with selective options."""
        provider = FrameworkServiceProvider(
            include_tool_config=True,
            include_event_registry=False,
            include_builder=True,
            include_bridge=False,
        )

        registrations = provider.get_registrations()

        service_types = [r.service_type for r in registrations]
        assert ToolConfiguratorService in service_types
        assert EventRegistryService not in service_types
        assert AgentBuilderService in service_types
        assert AgentSessionService not in service_types

    def test_provider_get_registrations_lifetimes(self):
        """Test that registrations have correct lifetimes."""
        provider = FrameworkServiceProvider()
        registrations = provider.get_registrations()

        reg_map = {r.service_type: r for r in registrations}

        # Singleton services
        assert reg_map[ToolConfiguratorService].lifetime == ServiceLifetime.SINGLETON
        assert reg_map[EventRegistryService].lifetime == ServiceLifetime.SINGLETON

        # Transient services
        assert reg_map[AgentBuilderService].lifetime == ServiceLifetime.TRANSIENT

        # Scoped services
        assert reg_map[AgentSessionService].lifetime == ServiceLifetime.SCOPED

    def test_provider_register_services(self):
        """Test registering services to container."""
        container = ServiceContainer()
        provider = FrameworkServiceProvider()

        provider.register_services(container)

        assert provider.is_registered
        assert container.is_registered(ToolConfiguratorService)
        assert container.is_registered(EventRegistryService)
        assert container.is_registered(AgentBuilderService)
        assert container.is_registered(AgentSessionService)

    def test_provider_register_services_replace(self):
        """Test replacing existing registrations."""
        container = ServiceContainer()

        # Register a mock first
        mock_service = MagicMock()
        container.register(ToolConfiguratorService, lambda c: mock_service)

        provider = FrameworkServiceProvider()
        provider.register_services(container, replace_existing=True)

        # Should have replaced with real implementation
        resolved = container.get(ToolConfiguratorService)
        assert resolved is not mock_service


class TestServiceResolution:
    """Test resolving services from container."""

    @pytest.fixture
    def container(self):
        """Create configured container."""
        container = ServiceContainer()
        provider = FrameworkServiceProvider()
        provider.register_services(container)
        return container

    def test_resolve_tool_configurator(self, container):
        """Test resolving ToolConfigurator."""
        service = container.get(ToolConfiguratorService)

        # Should implement the protocol
        assert hasattr(service, "configure_from_toolset")
        assert hasattr(service, "configure")
        assert hasattr(service, "add_filter")

    def test_resolve_event_registry(self, container):
        """Test resolving EventRegistry."""
        service = container.get(EventRegistryService)

        # Should implement the protocol
        assert hasattr(service, "register_converter")
        assert hasattr(service, "get_converter")

    def test_resolve_agent_builder(self, container):
        """Test resolving AgentBuilder."""
        service = container.get(AgentBuilderService)

        # Should implement the protocol
        assert hasattr(service, "preset")
        assert hasattr(service, "provider")
        assert hasattr(service, "model")
        assert hasattr(service, "build")

    def test_singleton_same_instance(self, container):
        """Test that singleton services return same instance."""
        config1 = container.get(ToolConfiguratorService)
        config2 = container.get(ToolConfiguratorService)

        assert config1 is config2

    def test_transient_different_instances(self, container):
        """Test that transient services return different instances."""
        builder1 = container.get(AgentBuilderService)
        builder2 = container.get(AgentBuilderService)

        assert builder1 is not builder2

    def test_builder_has_container_reference(self, container):
        """Test that builder has container reference for DI."""
        builder = container.get(AgentBuilderService)

        # Builder should have container reference
        assert hasattr(builder, "_container")
        assert builder._container is container


class TestFrameworkScope:
    """Test FrameworkScope scoped container."""

    @pytest.fixture
    def container(self):
        """Create configured container."""
        container = ServiceContainer()
        provider = FrameworkServiceProvider()
        provider.register_services(container)
        return container

    def test_scope_creation(self, container):
        """Test creating a framework scope."""
        scope = FrameworkScope(container)

        assert scope._active is True
        assert scope._container is container

    def test_scope_get_configurator(self, container):
        """Test getting configurator from scope."""
        with FrameworkScope(container) as scope:
            configurator = scope.get_configurator()
            assert configurator is not None

    def test_scope_get_registry(self, container):
        """Test getting registry from scope."""
        with FrameworkScope(container) as scope:
            registry = scope.get_registry()
            assert registry is not None

    def test_scope_get_builder(self, container):
        """Test getting builder from scope."""
        with FrameworkScope(container) as scope:
            builder = scope.get_builder()
            assert builder is not None

    def test_scope_context_manager_cleanup(self, container):
        """Test that scope cleans up on exit."""
        scope = FrameworkScope(container)

        with scope:
            assert scope._active is True

        assert scope._active is False

    @pytest.mark.asyncio
    async def test_scope_async_context_manager(self, container):
        """Test async context manager."""
        scope = FrameworkScope(container)

        async with scope:
            assert scope._active is True
            configurator = scope.get_configurator()
            assert configurator is not None

        assert scope._active is False


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_global(self):
        """Reset global container before each test."""
        reset_container()
        yield
        reset_container()

    def test_configure_framework_services_new_container(self):
        """Test configuring a new container."""
        container = ServiceContainer()
        result = configure_framework_services(container)

        assert result is container
        assert container.is_registered(ToolConfiguratorService)
        assert container.is_registered(EventRegistryService)

    def test_configure_framework_services_global(self):
        """Test configuring global container."""
        result = configure_framework_services()

        # Should return global container
        from victor.core.container import get_container

        assert result is get_container()
        assert result.is_registered(ToolConfiguratorService)

    def test_get_tool_configurator(self):
        """Test get_tool_configurator convenience function."""
        container = ServiceContainer()
        configure_framework_services(container)

        configurator = get_tool_configurator(container)

        assert configurator is not None
        assert hasattr(configurator, "configure")

    def test_get_event_registry(self):
        """Test get_event_registry convenience function."""
        container = ServiceContainer()
        configure_framework_services(container)

        registry = get_event_registry(container)

        assert registry is not None
        assert hasattr(registry, "get_converter")

    def test_create_builder(self):
        """Test create_builder convenience function."""
        container = ServiceContainer()
        configure_framework_services(container)

        builder = create_builder(container)

        assert builder is not None
        assert hasattr(builder, "build")

    def test_create_framework_scope(self):
        """Test create_framework_scope convenience function."""
        container = ServiceContainer()
        configure_framework_services(container)

        scope = create_framework_scope(container)

        assert scope is not None
        assert isinstance(scope, FrameworkScope)


class TestIntegrationWithRealServices:
    """Integration tests with real service implementations."""

    def test_tool_configurator_integration(self):
        """Test that resolved ToolConfigurator works."""
        container = ServiceContainer()
        configure_framework_services(container)

        configurator = container.get(ToolConfiguratorService)

        # Should be able to add filters
        from victor.framework.tool_config import AirgappedFilter

        configurator.add_filter(AirgappedFilter())
        # Should not raise

    def test_event_registry_integration(self):
        """Test that resolved EventRegistry works."""
        container = ServiceContainer()
        configure_framework_services(container)

        registry = container.get(EventRegistryService)

        # Should have converters registered
        from victor.framework.events import EventType

        converter = registry.get_converter(EventType.CONTENT)
        assert converter is not None

    def test_builder_chain_methods(self):
        """Test that builder chain methods work."""
        container = ServiceContainer()
        configure_framework_services(container)

        builder = container.get(AgentBuilderService)

        # Should support method chaining
        result = builder.provider("anthropic").model("claude-3-5-sonnet")

        # Should return builder for chaining
        assert result is builder


class TestServiceProtocols:
    """Test protocol compliance."""

    def test_tool_configurator_protocol_compliance(self):
        """Test ToolConfigurator implements protocol."""
        from victor.framework.tool_config import ToolConfigurator

        assert isinstance(ToolConfigurator(), ToolConfiguratorService)

    def test_event_registry_protocol_compliance(self):
        """Test EventRegistry implements protocol methods."""
        from victor.framework.event_registry import EventRegistry

        registry = EventRegistry.get_instance()

        # Check it has the required methods (even if protocol doesn't match exactly)
        assert hasattr(registry, "register_converter")
        assert hasattr(registry, "get_converter")
        assert callable(registry.register_converter)
        assert callable(registry.get_converter)

    def test_agent_builder_protocol_compliance(self):
        """Test AgentBuilder implements protocol."""
        from victor.framework.agent_components import AgentBuilder

        assert isinstance(AgentBuilder(), AgentBuilderService)
