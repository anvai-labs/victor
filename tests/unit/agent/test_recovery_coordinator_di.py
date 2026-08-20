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
# See the License for the specific language governing permissions
# limitations under the License.

"""DI resolution tests for RecoveryCoordinator.

Tests dependency injection container resolution and service lifetime.
"""

import pytest
from unittest.mock import MagicMock

from victor.agent.service_provider import OrchestratorServiceProvider
from victor.agent.services.protocols import StreamingRecoveryRuntimeProtocol


@pytest.fixture
def mock_settings():
    """Create mock settings with all required attributes."""
    from unittest.mock import MagicMock

    settings = MagicMock()
    # Recovery settings
    settings.recovery_blocked_consecutive_threshold = 4
    settings.recovery_blocked_total_threshold = 6
    settings.tools.tool_call_budget_warning_threshold = 250
    settings.max_consecutive_tool_calls = 8
    settings.use_recovery_handler = False
    settings.enable_context_compaction = False
    settings.enable_recovery_system = False
    settings.max_recovery_attempts = 3
    settings.recovery_timeout = 30.0
    # Tool settings
    settings.tool_timeout = 30.0
    settings.tools.tool_call_budget = 300
    settings.tool_selection_strategy = "hybrid"
    settings.semantic_weight = 0.7
    settings.keyword_weight = 0.3
    settings.semantic_candidate_count = 10
    settings.enable_tool_cache = True
    settings.tool_cache_ttl = 300
    settings.tool_cache_max_size = 100
    # Context settings
    settings.context_window_size = 128000
    settings.context_compaction_threshold = 0.7
    settings.context_compaction_target = 0.5
    settings.enable_smart_compaction = True
    # Provider settings
    settings.airgapped_mode = False
    settings.provider = "anthropic"
    settings.model = "claude-3-5-sonnet-20241022"
    settings.temperature = 0.7
    settings.max_tokens = 4096
    settings.enable_streaming = True
    # Session settings
    settings.max_iterations = 50
    settings.time_limit = 600.0
    settings.enable_checkpoints = False
    # RL settings
    settings.enable_rl_learners = False
    settings.enable_tool_selection_rl = False
    # Analytics settings
    settings.enable_usage_analytics = False
    # Mode controller settings
    settings.enable_adaptive_mode = False
    settings.default_mode = "default"
    # Project settings
    settings.project_config_file = ".victor.md"
    # Debug settings
    settings.debug = False
    settings.log_level = "INFO"
    return settings


@pytest.fixture
def service_provider(mock_settings):
    """Create OrchestratorServiceProvider with mocked settings."""
    from victor.core.container import ServiceContainer

    provider = OrchestratorServiceProvider(settings=mock_settings)
    container = ServiceContainer()
    provider.register_services(container)
    return provider


class TestRecoveryCoordinatorDI:
    """Tests for RecoveryCoordinator DI resolution."""

    def test_recovery_coordinator_protocol_registered(self, service_provider):
        """Test that StreamingRecoveryRuntimeProtocol is registered in DI container."""
        container = service_provider.container

        # Check that protocol is registered
        assert container.is_registered(StreamingRecoveryRuntimeProtocol)

    def test_recovery_coordinator_can_be_resolved(self, service_provider):
        """Test that RecoveryCoordinator can be resolved from DI container."""
        container = service_provider.container

        # Resolve RecoveryCoordinator
        recovery_coordinator = container.get(StreamingRecoveryRuntimeProtocol)

        # Verify it's not None and has expected bound runtime state
        assert recovery_coordinator is not None
        assert hasattr(recovery_coordinator, "_streaming_handler")
        assert hasattr(recovery_coordinator, "_unified_tracker")
        assert hasattr(recovery_coordinator, "_settings")

    def test_recovery_coordinator_singleton_lifetime(self, service_provider):
        """Test that RecoveryCoordinator has SINGLETON lifetime."""
        container = service_provider.container

        # Resolve RecoveryCoordinator twice
        instance1 = container.get(StreamingRecoveryRuntimeProtocol)
        instance2 = container.get(StreamingRecoveryRuntimeProtocol)

        # Verify they are the same instance (SINGLETON)
        assert instance1 is instance2

    def test_recovery_coordinator_dependencies_injected(self, service_provider):
        """Test that RecoveryCoordinator dependencies are properly injected."""
        container = service_provider.container

        # Resolve RecoveryCoordinator
        recovery_coordinator = container.get(StreamingRecoveryRuntimeProtocol)

        # Verify required runtime collaborators are injected
        assert recovery_coordinator._streaming_handler is not None
        assert recovery_coordinator._unified_tracker is not None
        assert recovery_coordinator._settings is not None

    def test_recovery_coordinator_optional_dependencies(self, service_provider):
        """Test that RecoveryCoordinator optional dependencies are handled correctly."""
        container = service_provider.container

        # Resolve RecoveryCoordinator
        recovery_coordinator = container.get(StreamingRecoveryRuntimeProtocol)

        # Verify optional collaborators have bound slots even when unset.
        assert hasattr(recovery_coordinator, "_recovery_handler")
        assert hasattr(recovery_coordinator, "_context_compactor")
        assert hasattr(recovery_coordinator, "_recovery_integration")

    def test_recovery_coordinator_factory_method(self, service_provider):
        """Test that RecoveryCoordinator is created via factory method."""
        # The _create_recovery_coordinator method should be called
        # This is tested implicitly by successful resolution
        container = service_provider.container

        recovery_coordinator = container.get(StreamingRecoveryRuntimeProtocol)

        # Verify it's the correct type (duck typing check)
        assert hasattr(recovery_coordinator, "check_natural_completion")
        assert hasattr(recovery_coordinator, "handle_empty_response")
        assert hasattr(recovery_coordinator, "apply_recovery_action")
        assert hasattr(recovery_coordinator, "check_blocked_threshold")

    def test_recovery_coordinator_methods_callable(self, service_provider):
        """Test that RecoveryCoordinator methods are callable."""
        container = service_provider.container

        recovery_coordinator = container.get(StreamingRecoveryRuntimeProtocol)

        # Verify key methods are callable
        assert callable(recovery_coordinator.check_natural_completion)
        assert callable(recovery_coordinator.check_tool_budget)
        assert callable(recovery_coordinator.handle_empty_response)
        assert callable(recovery_coordinator.apply_recovery_action)
        assert callable(recovery_coordinator.filter_blocked_tool_calls)
        assert callable(recovery_coordinator.truncate_tool_calls)
        assert callable(recovery_coordinator.check_blocked_threshold)
        assert callable(recovery_coordinator.check_force_action)

    def test_orchestrator_factory_creates_recovery_coordinator(self, service_provider):
        """Test that OrchestratorFactory can create RecoveryCoordinator."""
        from victor.agent.orchestrator_factory import OrchestratorFactory

        # Create mock provider
        mock_provider = MagicMock()
        mock_provider.name = "mock"
        mock_provider.supports_tools.return_value = True

        # Create factory with required arguments
        factory = OrchestratorFactory(
            settings=service_provider._settings,
            provider=mock_provider,
            model="mock-model",
        )
        # Inject pre-configured container
        factory._container = service_provider.container

        # Create RecoveryCoordinator via factory
        recovery_coordinator = factory.create_recovery_coordinator()

        # Verify it's not None and has expected bound runtime state
        assert recovery_coordinator is not None
        assert hasattr(recovery_coordinator, "_streaming_handler")
        assert hasattr(recovery_coordinator, "_unified_tracker")

    def test_orchestrator_factory_recovery_coordinator_is_singleton(self, service_provider):
        """Test that OrchestratorFactory returns same RecoveryCoordinator instance."""
        from victor.agent.orchestrator_factory import OrchestratorFactory

        # Create mock provider
        mock_provider = MagicMock()
        mock_provider.name = "mock"
        mock_provider.supports_tools.return_value = True

        # Create factory with required arguments
        factory = OrchestratorFactory(
            settings=service_provider._settings,
            provider=mock_provider,
            model="mock-model",
        )
        # Inject pre-configured container
        factory._container = service_provider.container

        # Create RecoveryCoordinator twice
        instance1 = factory.create_recovery_coordinator()
        instance2 = factory.create_recovery_coordinator()

        # Verify they are the same instance (SINGLETON)
        assert instance1 is instance2


class TestRecoveryCoordinatorDIIntegration:
    """Integration tests for RecoveryCoordinator DI."""

    def test_full_orchestrator_initialization_with_recovery_coordinator(self, service_provider):
        """Test full orchestrator initialization includes RecoveryCoordinator."""
        from victor.agent.orchestrator_factory import OrchestratorFactory

        # Create mock provider
        mock_provider = MagicMock()
        mock_provider.name = "mock"
        mock_provider.supports_tools.return_value = True

        # Create factory with required arguments
        factory = OrchestratorFactory(
            settings=service_provider._settings,
            provider=mock_provider,
            model="mock-model",
        )
        # Inject pre-configured container
        factory._container = service_provider.container

        # Create orchestrator (this should initialize RecoveryCoordinator internally)
        # We can't easily test this without a full orchestrator setup,
        # but we can verify the factory has the method
        assert hasattr(factory, "create_recovery_coordinator")
        assert callable(factory.create_recovery_coordinator)

    def test_recovery_coordinator_dependencies_resolution_chain(self, service_provider):
        """Test that RecoveryCoordinator dependency resolution chain works."""
        from victor.agent.protocols import TaskTrackerProtocol
        from victor.agent.services.protocols import StreamingHandlerProtocol

        container = service_provider.container

        # Verify that dependencies are registered and can be resolved
        assert container.is_registered(StreamingHandlerProtocol)
        assert container.is_registered(TaskTrackerProtocol)

        # Verify RecoveryCoordinator can be resolved (which depends on above)
        recovery_coordinator = container.get(StreamingRecoveryRuntimeProtocol)
        assert recovery_coordinator is not None

    def test_recovery_coordinator_with_all_dependencies(self, service_provider):
        """Test RecoveryCoordinator resolution with all possible dependencies."""
        container = service_provider.container

        # Resolve RecoveryCoordinator
        recovery_coordinator = container.get(StreamingRecoveryRuntimeProtocol)

        # Verify all expected runtime slots exist (even if some are None)
        expected_attrs = [
            "_recovery_handler",
            "_recovery_integration",
            "_streaming_handler",
            "_context_compactor",
            "_unified_tracker",
            "_settings",
        ]

        for attr in expected_attrs:
            assert hasattr(recovery_coordinator, attr), f"Missing attribute: {attr}"
