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

"""DI resolution tests for TaskCoordinator.

Tests dependency injection container resolution and service lifetime.
"""

import pytest
from unittest.mock import Mock, MagicMock

from victor.agent.service_provider import OrchestratorServiceProvider
from victor.agent.services.protocols import TaskRuntimeProtocol
from victor.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings with all required attributes."""
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
    settings.tool_budget = 15
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
    settings.session_idle_timeout = 300.0
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
    provider.container = container  # Store container reference for tests

    # Register all services via provider (no manual mocking needed for DI tests)
    provider.register_services(container)
    return provider


class TestTaskCoordinatorDI:
    """Tests for TaskCoordinator DI resolution."""

    def test_task_coordinator_protocol_registered(self, service_provider):
        """Test that TaskRuntimeProtocol is registered in DI container."""
        container = service_provider.container

        # Check that protocol is registered
        assert container.is_registered(TaskRuntimeProtocol)

    def test_task_coordinator_can_be_resolved(self, service_provider):
        """Test that TaskCoordinator can be resolved from DI container."""
        container = service_provider.container

        # Resolve TaskCoordinator
        task_coordinator = container.get(TaskRuntimeProtocol)

        # Verify it's not None and has expected attributes
        assert task_coordinator is not None
        assert hasattr(task_coordinator, "task_analyzer")
        assert hasattr(task_coordinator, "unified_tracker")
        assert hasattr(task_coordinator, "prompt_builder")
        assert hasattr(task_coordinator, "settings")

    def test_task_coordinator_singleton_lifetime(self, service_provider):
        """Test that TaskCoordinator has SINGLETON lifetime."""
        container = service_provider.container

        # Resolve TaskCoordinator twice
        instance1 = container.get(TaskRuntimeProtocol)
        instance2 = container.get(TaskRuntimeProtocol)

        # Verify they are the same instance (SINGLETON)
        assert instance1 is instance2

    def test_task_coordinator_dependencies_injected(self, service_provider):
        """Test that TaskCoordinator dependencies are properly injected."""
        container = service_provider.container

        # Resolve TaskCoordinator
        task_coordinator = container.get(TaskRuntimeProtocol)

        # Verify required dependencies are injected
        assert task_coordinator.task_analyzer is not None
        assert task_coordinator.unified_tracker is not None
        assert task_coordinator.prompt_builder is not None
        assert task_coordinator.settings is not None

    def test_task_coordinator_methods_callable(self, service_provider):
        """Test that TaskCoordinator methods are callable."""
        container = service_provider.container

        task_coordinator = container.get(TaskRuntimeProtocol)

        # Verify key methods are callable
        assert callable(task_coordinator.prepare_task)
        assert callable(task_coordinator.apply_intent_guard)
        assert callable(task_coordinator.apply_task_guidance)

    def test_orchestrator_factory_creates_task_coordinator(self, service_provider):
        """Test that OrchestratorFactory can create TaskCoordinator."""
        from victor.agent.orchestrator_factory import OrchestratorFactory
        from unittest.mock import Mock

        # Create factory (it will create its own container internally)
        factory = OrchestratorFactory(
            settings=service_provider._settings,
            provider=Mock(),
            model="test-model",
        )

        # Override factory's container with our test container
        factory._container = service_provider.container

        # Create TaskCoordinator via factory
        task_coordinator = factory.create_task_coordinator()

        # Verify it's not None and has expected attributes
        assert task_coordinator is not None
        assert hasattr(task_coordinator, "task_analyzer")
        assert hasattr(task_coordinator, "unified_tracker")

    def test_orchestrator_factory_task_coordinator_is_singleton(self, service_provider):
        """Test that OrchestratorFactory returns same TaskCoordinator instance."""
        from victor.agent.orchestrator_factory import OrchestratorFactory
        from unittest.mock import Mock

        # Create factory (it will create its own container internally)
        factory = OrchestratorFactory(
            settings=service_provider._settings,
            provider=Mock(),
            model="test-model",
        )

        # Override factory's container with our test container
        factory._container = service_provider.container

        # Create TaskCoordinator twice
        instance1 = factory.create_task_coordinator()
        instance2 = factory.create_task_coordinator()

        # Verify they are the same instance (SINGLETON)
        assert instance1 is instance2


class TestTaskCoordinatorDIIntegration:
    """Integration tests for TaskCoordinator DI."""

    def test_task_coordinator_dependencies_resolution_chain(self, service_provider):
        """Test that TaskCoordinator dependency resolution chain works."""
        from victor.agent.protocols import (
            TaskAnalyzerProtocol,
            TaskTrackerProtocol,
            SystemPromptBuilderProtocol,
        )

        container = service_provider.container

        # Verify that dependencies are registered and can be resolved
        assert container.is_registered(TaskAnalyzerProtocol)
        assert container.is_registered(TaskTrackerProtocol)
        assert container.is_registered(SystemPromptBuilderProtocol)

        # Verify TaskCoordinator can be resolved (which depends on above)
        task_coordinator = container.get(TaskRuntimeProtocol)
        assert task_coordinator is not None

    def test_task_coordinator_with_all_dependencies(self, service_provider):
        """Test TaskCoordinator resolution with all possible dependencies."""
        container = service_provider.container

        # Resolve TaskCoordinator
        task_coordinator = container.get(TaskRuntimeProtocol)

        # Verify all expected attributes exist
        expected_attrs = [
            "task_analyzer",
            "unified_tracker",
            "prompt_builder",
            "settings",
        ]

        for attr in expected_attrs:
            assert hasattr(task_coordinator, attr), f"Missing attribute: {attr}"

    def test_full_orchestrator_initialization_with_task_coordinator(self, service_provider):
        """Test full orchestrator initialization includes TaskCoordinator."""
        from victor.agent.orchestrator_factory import OrchestratorFactory
        from unittest.mock import Mock

        # Create factory (it will create its own container internally)
        factory = OrchestratorFactory(
            settings=service_provider._settings,
            provider=Mock(),
            model="test-model",
        )

        # Verify the factory has the method
        assert hasattr(factory, "create_task_coordinator")
        assert callable(factory.create_task_coordinator)
