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

"""Tests for orchestrator DI integration (Phase 10).

Tests the service protocols, service provider, and DI container integration
for AgentOrchestrator components.
"""

import pytest
from unittest.mock import MagicMock, patch

# Suppress deprecation warnings for complexity_classifier shim during migration
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

from victor.core.container import ServiceContainer, ServiceLifetime

# =============================================================================
# Protocol Tests
# =============================================================================


class TestServiceProtocols:
    """Tests for service protocol definitions."""

    def test_protocols_importable(self):
        """Test that all protocols can be imported."""
        from victor.agent.protocols import (
            ProviderManagerProtocol,
            ToolRegistryProtocol,
            ConversationControllerProtocol,
            ToolPipelineProtocol,
            StreamingControllerProtocol,
            TaskAnalyzerProtocol,
            ToolSelectorProtocol,
            ObservabilityProtocol,
            MetricsCollectorProtocol,
            ToolCacheProtocol,
            TaskTrackerProtocol,
            ToolOutputFormatterProtocol,
            ResponseSanitizerProtocol,
            ArgumentNormalizerProtocol,
            ProjectContextProtocol,
            ComplexityClassifierProtocol,
            ActionAuthorizerProtocol,
            SearchRouterProtocol,
            ConversationStateMachineProtocol,
            MessageHistoryProtocol,
            ToolExecutorProtocol,
        )

        # Verify they are types
        assert isinstance(ProviderManagerProtocol, type)
        assert isinstance(ToolRegistryProtocol, type)
        assert isinstance(ConversationControllerProtocol, type)

    def test_protocol_is_runtime_checkable(self):
        """Test that protocols support isinstance checks."""
        from victor.agent.protocols import ResponseSanitizerProtocol

        # Create a mock that satisfies the protocol
        mock_sanitizer = MagicMock()
        mock_sanitizer.sanitize = MagicMock(return_value="sanitized")

        # Should be able to use isinstance (won't match since mock doesn't have annotations)
        # This just verifies the @runtime_checkable decorator is present
        assert hasattr(ResponseSanitizerProtocol, "__protocol_attrs__") or hasattr(
            ResponseSanitizerProtocol, "_is_runtime_protocol"
        )

    def test_tool_registry_protocol_methods(self):
        """Test ToolRegistryProtocol has expected methods."""
        from victor.agent.protocols import ToolRegistryProtocol

        # Check that protocol defines expected methods
        # These will be in __abstractmethods__ or __protocol_attrs__
        expected_methods = ["register", "get", "list_tools", "get_tool_cost"]

        # Get all attributes that look like methods
        for method in expected_methods:
            assert hasattr(ToolRegistryProtocol, method), f"Missing method: {method}"


# =============================================================================
# Service Provider Tests
# =============================================================================


class TestOrchestratorServiceProvider:
    """Tests for OrchestratorServiceProvider."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.search.unified_embedding_model = "test-model"
        settings.enable_observability = True
        settings.max_conversation_history = 50
        return settings

    def test_provider_creation(self, mock_settings):
        """Test service provider can be created."""
        from victor.agent.service_provider import OrchestratorServiceProvider

        provider = OrchestratorServiceProvider(mock_settings)
        assert provider._settings is mock_settings

    def test_register_singleton_services(self, mock_settings):
        """Test singleton service registration."""
        from victor.agent.service_provider import OrchestratorServiceProvider
        from victor.agent.protocols import (
            ComplexityClassifierProtocol,
            ResponseSanitizerProtocol,
            ActionAuthorizerProtocol,
            SearchRouterProtocol,
        )

        container = ServiceContainer()
        provider = OrchestratorServiceProvider(mock_settings)

        provider.register_singleton_services(container)

        # Verify singletons are registered
        assert container.is_registered(ComplexityClassifierProtocol)
        assert container.is_registered(ResponseSanitizerProtocol)
        assert container.is_registered(ActionAuthorizerProtocol)
        assert container.is_registered(SearchRouterProtocol)

    def test_register_scoped_services(self, mock_settings):
        """Test scoped service registration."""
        from victor.agent.service_provider import OrchestratorServiceProvider
        from victor.agent.protocols import (
            ConversationStateMachineProtocol,
            TaskTrackerProtocol,
            MessageHistoryProtocol,
        )

        container = ServiceContainer()
        provider = OrchestratorServiceProvider(mock_settings)

        provider.register_scoped_services(container)

        # Verify scoped services are registered
        assert container.is_registered(ConversationStateMachineProtocol)
        assert container.is_registered(TaskTrackerProtocol)
        assert container.is_registered(MessageHistoryProtocol)

    def test_create_state_machine_uses_runtime_intelligence(self, mock_settings):
        """Scoped state machines should be created with runtime intelligence attached."""
        from victor.agent.service_provider import OrchestratorServiceProvider

        container = ServiceContainer()
        provider = OrchestratorServiceProvider(mock_settings)
        runtime_intelligence = MagicMock()

        with patch(
            "victor.agent.services.runtime_intelligence.RuntimeIntelligenceService.from_container",
            return_value=runtime_intelligence,
        ) as mock_from_container:
            state_machine = provider._create_conversation_state_machine(container)

        mock_from_container.assert_called_once_with(container)
        assert state_machine._runtime_intelligence is runtime_intelligence

    def test_create_tool_selector_uses_runtime_intelligence(self, mock_settings):
        """Singleton tool selector should be created with runtime intelligence attached."""
        from victor.agent.service_provider import OrchestratorServiceProvider

        container = ServiceContainer()
        provider = OrchestratorServiceProvider(mock_settings)
        provider.container = container
        runtime_intelligence = MagicMock()

        with patch(
            "victor.agent.services.runtime_intelligence.RuntimeIntelligenceService.from_container",
            return_value=runtime_intelligence,
        ) as mock_from_container:
            selector = provider._create_tool_selector()

        mock_from_container.assert_called_once_with(container)
        assert selector._runtime_intelligence is runtime_intelligence

    def test_singleton_same_instance(self, mock_settings):
        """Test that singletons return same instance."""
        from victor.agent.service_provider import OrchestratorServiceProvider
        from victor.agent.protocols import ResponseSanitizerProtocol

        container = ServiceContainer()
        provider = OrchestratorServiceProvider(mock_settings)
        provider.register_singleton_services(container)

        instance1 = container.get(ResponseSanitizerProtocol)
        instance2 = container.get(ResponseSanitizerProtocol)

        assert instance1 is instance2

    def test_scoped_services_different_per_scope(self, mock_settings):
        """Test that scoped services are different per scope."""
        from victor.agent.service_provider import OrchestratorServiceProvider
        from victor.agent.protocols import ConversationStateMachineProtocol

        container = ServiceContainer()
        provider = OrchestratorServiceProvider(mock_settings)
        provider.register_scoped_services(container)

        with container.create_scope() as scope1:
            instance1 = scope1.get(ConversationStateMachineProtocol)

        with container.create_scope() as scope2:
            instance2 = scope2.get(ConversationStateMachineProtocol)

        assert instance1 is not instance2

    def test_scoped_services_same_within_scope(self, mock_settings):
        """Test that scoped services are same within same scope."""
        from victor.agent.service_provider import OrchestratorServiceProvider
        from victor.agent.protocols import ConversationStateMachineProtocol

        container = ServiceContainer()
        provider = OrchestratorServiceProvider(mock_settings)
        provider.register_scoped_services(container)

        with container.create_scope() as scope:
            instance1 = scope.get(ConversationStateMachineProtocol)
            instance2 = scope.get(ConversationStateMachineProtocol)
            assert instance1 is instance2

    def test_register_all_services(self, mock_settings):
        """Test register_services registers both singleton and scoped."""
        from victor.agent.service_provider import OrchestratorServiceProvider
        from victor.agent.protocols import (
            ResponseSanitizerProtocol,
            ConversationStateMachineProtocol,
        )

        container = ServiceContainer()
        provider = OrchestratorServiceProvider(mock_settings)

        provider.register_services(container)

        # Both types should be registered
        assert container.is_registered(ResponseSanitizerProtocol)
        assert container.is_registered(ConversationStateMachineProtocol)

    def test_tool_coordinator_protocol_is_not_registered_by_default(self, mock_settings):
        """ToolCoordinatorProtocol is compatibility-only, not a primary DI service."""
        from victor.agent.service_provider import OrchestratorServiceProvider

        with pytest.warns(
            DeprecationWarning,
            match="ToolCoordinatorProtocol is deprecated compatibility surface",
        ):
            from victor.agent.protocols import ToolCoordinatorProtocol

        container = ServiceContainer()
        provider = OrchestratorServiceProvider(mock_settings)

        provider.register_singleton_services(container)

        assert container.is_registered(ToolCoordinatorProtocol) is False

    def test_service_provider_no_longer_exposes_tool_coordinator_factory(self, mock_settings):
        """Deprecated tool-coordinator shims are no longer created by the service provider."""
        from victor.agent.service_provider import OrchestratorServiceProvider

        provider = OrchestratorServiceProvider(mock_settings)

        assert hasattr(provider, "_create_tool_coordinator") is False

    def test_create_rl_coordinator_uses_service_runtime_host(self, mock_settings):
        """RL coordinator factory should resolve through the service runtime host."""
        from victor.agent.service_provider import OrchestratorServiceProvider

        provider = OrchestratorServiceProvider(mock_settings)
        coordinator = MagicMock()

        with patch(
            "victor.agent.services.rl_runtime.get_rl_coordinator",
            return_value=coordinator,
        ) as get_rl_coordinator:
            result = provider._create_rl_coordinator()

        assert result is coordinator
        get_rl_coordinator.assert_called_once_with()


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConfigureOrchestratorServices:
    """Tests for configure_orchestrator_services convenience function."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.search.unified_embedding_model = "test-model"
        settings.enable_observability = True
        settings.max_conversation_history = 50
        return settings

    def test_configure_registers_services(self, mock_settings):
        """Test that configure_orchestrator_services works."""
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.agent.protocols import ResponseSanitizerProtocol

        container = ServiceContainer()
        configure_orchestrator_services(container, mock_settings)

        assert container.is_registered(ResponseSanitizerProtocol)


# =============================================================================
# Integration Tests
# =============================================================================


class TestBootstrapIntegration:
    """Tests for bootstrap integration."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings with all required attributes."""
        settings = MagicMock()
        settings.search.unified_embedding_model = "test-model"
        settings.enable_observability = True
        settings.max_conversation_history = 50
        settings.analytics_enabled = False
        return settings

    def test_bootstrap_registers_orchestrator_services(self, mock_settings):
        """Test that bootstrap registers orchestrator services."""
        from victor.core.bootstrap import bootstrap_container
        from victor.agent.protocols import ResponseSanitizerProtocol

        # Bootstrap with mock settings
        container = bootstrap_container(settings=mock_settings)

        # Orchestrator services should be registered
        assert container.is_registered(ResponseSanitizerProtocol)


# =============================================================================
# Null Implementation Tests
# =============================================================================


class TestNullImplementations:
    """Tests for null/no-op implementations."""

    def test_null_observability(self):
        """Test _NullObservability doesn't raise."""
        from victor.agent.service_provider import _NullObservability

        obs = _NullObservability()

        # All methods should be no-ops
        obs.on_tool_start("test", {}, "id-1")
        obs.on_tool_end("test", {}, True, "id-1")
        obs.wire_state_machine(MagicMock())
        obs.on_error(Exception("test"), {})

    def test_null_task_analyzer(self):
        """Test _NullTaskAnalyzer returns defaults."""
        from victor.agent.service_provider import _NullTaskAnalyzer

        analyzer = _NullTaskAnalyzer()

        result = analyzer.analyze("test prompt")
        assert result["complexity"] == "unknown"
        assert result["intent"] == "unknown"

        assert analyzer.classify_complexity("test") is None
        assert analyzer.detect_intent("test") is None


# =============================================================================
# Protocol Conformance Tests
# =============================================================================


class TestProtocolConformance:
    """Tests that real implementations conform to protocols."""

    def test_response_sanitizer_conforms(self):
        """Test ResponseSanitizer conforms to protocol."""
        from victor.agent.response_sanitizer import ResponseSanitizer
        from victor.agent.protocols import ResponseSanitizerProtocol

        sanitizer = ResponseSanitizer()

        # Verify it has the required method
        assert hasattr(sanitizer, "sanitize")
        assert callable(sanitizer.sanitize)

    def test_complexity_classifier_conforms(self):
        """Test ComplexityClassifier conforms to protocol."""
        from victor.framework.task import TaskComplexityService as ComplexityClassifier
        from victor.agent.protocols import ComplexityClassifierProtocol

        classifier = ComplexityClassifier()

        # Verify it has the required method
        assert hasattr(classifier, "classify")
        assert callable(classifier.classify)

    def test_conversation_state_machine_conforms(self):
        """Test ConversationStateMachine conforms to protocol."""
        from victor.agent.conversation.state_machine import ConversationStateMachine

        state_machine = ConversationStateMachine()

        # Verify it has required properties and methods
        # Note: Implementation uses get_stage() method, not stage property
        assert hasattr(state_machine, "get_stage")
        assert callable(state_machine.get_stage)
        assert hasattr(state_machine, "record_tool_execution")
        assert callable(state_machine.record_tool_execution)

    def test_message_history_conforms(self):
        """Test MessageHistory conforms to protocol."""
        from victor.agent.message_history import MessageHistory

        history = MessageHistory(system_prompt="test")

        # Verify it has required methods
        # Note: Implementation uses add_message(), not add()
        assert hasattr(history, "add_message")
        assert callable(history.add_message)
        assert hasattr(history, "get_messages_for_provider")
        assert callable(history.get_messages_for_provider)
        assert hasattr(history, "clear")
        assert callable(history.clear)


# =============================================================================
# Service Resolution Tests
# =============================================================================


class TestServiceResolution:
    """Tests for resolving services from container."""

    @pytest.fixture
    def configured_container(self):
        """Create a fully configured container."""
        settings = MagicMock()
        settings.search.unified_embedding_model = "test-model"
        settings.enable_observability = True
        settings.max_conversation_history = 50

        from victor.agent.service_provider import configure_orchestrator_services

        container = ServiceContainer()
        configure_orchestrator_services(container, settings)
        return container

    def test_resolve_sanitizer(self, configured_container):
        """Test resolving ResponseSanitizer."""
        from victor.agent.protocols import ResponseSanitizerProtocol

        sanitizer = configured_container.get(ResponseSanitizerProtocol)
        assert sanitizer is not None
        assert hasattr(sanitizer, "sanitize")

    def test_resolve_complexity_classifier(self, configured_container):
        """Test resolving ComplexityClassifier."""
        from victor.agent.protocols import ComplexityClassifierProtocol

        classifier = configured_container.get(ComplexityClassifierProtocol)
        assert classifier is not None
        assert hasattr(classifier, "classify")

    def test_resolve_scoped_state_machine(self, configured_container):
        """Test resolving ConversationStateMachine in scope."""
        from victor.agent.protocols import ConversationStateMachineProtocol

        with configured_container.create_scope() as scope:
            state_machine = scope.get(ConversationStateMachineProtocol)
            assert state_machine is not None
            # Note: Implementation uses get_stage() method, not stage property
            assert hasattr(state_machine, "get_stage")
            assert callable(state_machine.get_stage)
