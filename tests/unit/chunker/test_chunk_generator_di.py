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

"""DI resolution tests for ChunkGenerator.

Tests dependency injection container resolution and service lifetime.
"""

import pytest
from unittest.mock import Mock

from victor.agent.service_provider import OrchestratorServiceProvider
from victor.agent.services.protocols import ChunkRuntimeProtocol
from victor.config.settings import Settings


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
    provider.register_services(container)
    return provider


class TestChunkGeneratorDI:
    """Tests for ChunkGenerator DI resolution."""

    def test_chunk_generator_protocol_registered(self, service_provider):
        """Test that ChunkRuntimeProtocol is registered in DI container."""
        container = service_provider.container

        # Check that protocol is registered
        assert container.is_registered(ChunkRuntimeProtocol)

    def test_chunk_generator_can_be_resolved(self, service_provider):
        """Test that ChunkGenerator can be resolved from DI container."""
        container = service_provider.container

        # Resolve ChunkGenerator
        chunk_generator = container.get(ChunkRuntimeProtocol)

        # Verify it's not None and has expected attributes
        assert chunk_generator is not None
        assert hasattr(chunk_generator, "streaming_handler")
        assert hasattr(chunk_generator, "settings")

    def test_chunk_generator_singleton_lifetime(self, service_provider):
        """Test that ChunkGenerator has SINGLETON lifetime."""
        container = service_provider.container

        # Resolve ChunkGenerator twice
        instance1 = container.get(ChunkRuntimeProtocol)
        instance2 = container.get(ChunkRuntimeProtocol)

        # Verify they are the same instance (SINGLETON)
        assert instance1 is instance2

    def test_chunk_generator_dependencies_injected(self, service_provider):
        """Test that ChunkGenerator dependencies are properly injected."""
        container = service_provider.container

        # Resolve ChunkGenerator
        chunk_generator = container.get(ChunkRuntimeProtocol)

        # Verify required dependencies are injected
        assert chunk_generator.streaming_handler is not None
        assert chunk_generator.settings is not None

    def test_chunk_generator_methods_callable(self, service_provider):
        """Test that ChunkGenerator methods are callable."""
        container = service_provider.container

        chunk_generator = container.get(ChunkRuntimeProtocol)

        # Verify key methods are callable
        assert callable(chunk_generator.generate_tool_start_chunk)
        assert callable(chunk_generator.generate_tool_result_chunks)
        assert callable(chunk_generator.generate_thinking_status_chunk)
        assert callable(chunk_generator.generate_budget_error_chunk)
        assert callable(chunk_generator.generate_force_response_error_chunk)
        assert callable(chunk_generator.generate_final_marker_chunk)
        assert callable(chunk_generator.generate_metrics_chunk)
        assert callable(chunk_generator.generate_content_chunk)
        assert callable(chunk_generator.get_budget_exhausted_chunks)

    def test_orchestrator_factory_creates_chunk_generator(self, service_provider):
        """Test that OrchestratorFactory can create ChunkGenerator."""
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

        # Create ChunkGenerator via factory
        chunk_generator = factory.create_chunk_generator()

        # Verify it's not None and has expected attributes
        assert chunk_generator is not None
        assert hasattr(chunk_generator, "streaming_handler")
        assert hasattr(chunk_generator, "settings")

    def test_orchestrator_factory_chunk_generator_is_singleton(self, service_provider):
        """Test that OrchestratorFactory returns same ChunkGenerator instance."""
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

        # Create ChunkGenerator twice
        instance1 = factory.create_chunk_generator()
        instance2 = factory.create_chunk_generator()

        # Verify they are the same instance (SINGLETON)
        assert instance1 is instance2


class TestChunkGeneratorDIIntegration:
    """Integration tests for ChunkGenerator DI."""

    def test_chunk_generator_dependencies_resolution_chain(self, service_provider):
        """Test that ChunkGenerator dependency resolution chain works."""
        from victor.agent.protocols import StreamingHandlerProtocol

        container = service_provider.container

        # Verify that dependencies are registered and can be resolved
        assert container.is_registered(StreamingHandlerProtocol)

        # Verify ChunkGenerator can be resolved (which depends on above)
        chunk_generator = container.get(ChunkRuntimeProtocol)
        assert chunk_generator is not None

    def test_chunk_generator_with_all_dependencies(self, service_provider):
        """Test ChunkGenerator resolution with all possible dependencies."""
        container = service_provider.container

        # Resolve ChunkGenerator
        chunk_generator = container.get(ChunkRuntimeProtocol)

        # Verify all expected attributes exist
        expected_attrs = [
            "streaming_handler",
            "settings",
        ]

        for attr in expected_attrs:
            assert hasattr(chunk_generator, attr), f"Missing attribute: {attr}"

    def test_full_orchestrator_initialization_with_chunk_generator(self, service_provider):
        """Test full orchestrator initialization includes ChunkGenerator."""
        from victor.agent.orchestrator_factory import OrchestratorFactory
        from unittest.mock import Mock

        # Create factory (it will create its own container internally)
        factory = OrchestratorFactory(
            settings=service_provider._settings,
            provider=Mock(),
            model="test-model",
        )

        # Verify the factory has the method
        assert hasattr(factory, "create_chunk_generator")
        assert callable(factory.create_chunk_generator)
