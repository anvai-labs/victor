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

"""Tests for OrchestratorFactory.

Part of CRITICAL-001: Monolithic Orchestrator decomposition.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Suppress deprecation warnings for complexity_classifier shim during migration
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

from victor.agent.orchestrator_factory import (
    OrchestratorFactory,
    create_orchestrator_factory,
)
from victor.agent.coordinators.exploration_state_passed import (
    ExplorationStatePassedCoordinator,
)
from victor.agent.coordinators.safety_state_passed import SafetyStatePassedCoordinator
from victor.agent.coordinators.system_prompt_state_passed import (
    SystemPromptStatePassedCoordinator,
)
from victor.agent.services import ServiceStreamingRuntime


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock()
    settings.streaming_metrics_enabled = True
    settings.streaming_metrics_history_size = 100
    settings.cache_dir = "/tmp/cache"
    settings.enable_prometheus_export = True
    settings.use_predefined_patterns = True
    settings.sequence_learning_rate = 0.3
    settings.enable_recovery_system = True
    settings.enable_observability = True
    return settings


@pytest.fixture
def mock_provider():
    """Create mock provider."""
    provider = MagicMock()
    provider.name = "test_provider"
    provider.__class__.__name__ = "TestProvider"
    return provider


@pytest.fixture
def mock_container():
    """Create mock DI container."""
    from victor.agent.response_sanitizer import ResponseSanitizer
    from victor.framework.task import TaskComplexityService as ComplexityClassifier
    from victor.agent.action_authorizer import ActionAuthorizer
    from victor.agent.search_router import SearchRouter
    from victor.context.project_context import ProjectContext
    from victor.agent.usage_analytics import UsageAnalytics
    from victor.agent.tool_sequence_tracker import ToolSequenceTracker
    from victor.agent.recovery.handler import RecoveryHandler

    container = MagicMock()

    # Mock get() to return actual instances for services
    def mock_get(protocol):
        # Map protocols to concrete instances
        from victor.agent.protocols import (
            ResponseSanitizerProtocol,
            ComplexityClassifierProtocol,
            ActionAuthorizerProtocol,
            SearchRouterProtocol,
            ProjectContextProtocol,
            UsageAnalyticsProtocol,
            ToolSequenceTrackerProtocol,
            RecoveryHandlerProtocol,
        )

        protocol_map = {
            ResponseSanitizerProtocol: ResponseSanitizer(),
            ComplexityClassifierProtocol: ComplexityClassifier(),
            ActionAuthorizerProtocol: ActionAuthorizer(),
            SearchRouterProtocol: SearchRouter(),
            ProjectContextProtocol: ProjectContext(),
            UsageAnalyticsProtocol: UsageAnalytics(),
            ToolSequenceTrackerProtocol: ToolSequenceTracker(),
            RecoveryHandlerProtocol: MagicMock(
                spec=RecoveryHandler
            ),  # Use mock for complex dependency
        }
        return protocol_map.get(protocol, MagicMock())

    container.get = MagicMock(side_effect=mock_get)
    container.get_optional = MagicMock(return_value=None)
    return container


@pytest.fixture
def factory(mock_settings, mock_provider, mock_container):
    """Create factory with mocked dependencies."""
    f = OrchestratorFactory(
        settings=mock_settings,
        provider=mock_provider,
        model="test-model",
        temperature=0.7,
        max_tokens=4096,
    )
    # Pre-set the container to avoid lazy initialization issues
    f._container = mock_container
    return f


class TestOrchestratorFactoryInit:
    """Tests for OrchestratorFactory initialization."""

    def test_factory_stores_settings(self, factory, mock_settings):
        """Factory stores settings."""
        assert factory.settings == mock_settings

    def test_factory_stores_provider(self, factory, mock_provider):
        """Factory stores provider."""
        assert factory.provider == mock_provider

    def test_factory_stores_model(self, factory):
        """Factory stores model."""
        assert factory.model == "test-model"

    def test_factory_stores_temperature(self, factory):
        """Factory stores temperature."""
        assert factory.temperature == 0.7

    def test_factory_stores_max_tokens(self, factory):
        """Factory stores max_tokens."""
        assert factory.max_tokens == 4096

    def test_factory_default_tool_selection(self, factory):
        """Factory has empty default tool_selection."""
        assert factory.tool_selection == {}

    def test_factory_default_thinking(self, factory):
        """Factory has False default thinking."""
        assert factory.thinking is False


class TestCreateSanitizer:
    """Tests for create_sanitizer method."""

    def test_create_sanitizer_returns_sanitizer(self, factory):
        """create_sanitizer returns a ResponseSanitizer instance."""
        from victor.agent.response_sanitizer import ResponseSanitizer

        sanitizer = factory.create_sanitizer()
        assert isinstance(sanitizer, ResponseSanitizer)

    def test_create_sanitizer_uses_di_if_available(self, factory, mock_container):
        """create_sanitizer checks DI container via get_optional then falls back."""
        from victor.agent.protocols import ResponseSanitizerProtocol

        sanitizer = factory.create_sanitizer()

        # create_sanitizer uses get_optional (soft lookup) so the factory
        # can fall back to a direct constructor when the protocol isn't
        # registered (e.g. test environments).
        mock_container.get_optional.assert_called_with(ResponseSanitizerProtocol)


class TestCreateProjectContext:
    """Tests for create_project_context method."""

    def test_create_project_context_returns_context(self, factory):
        """create_project_context returns a ProjectContext instance."""
        from victor.context.project_context import ProjectContext

        context = factory.create_project_context()
        assert isinstance(context, ProjectContext)

    def test_create_project_context_uses_di_if_available(self, factory, mock_container):
        """create_project_context uses DI container (always)."""
        from victor.agent.protocols import ProjectContextProtocol

        context = factory.create_project_context()

        # Verify DI container was used
        mock_container.get.assert_called_with(ProjectContextProtocol)


class TestCreateStreamingChatAdapter:
    """Tests for create_streaming_chat_adapter method."""

    def test_create_streaming_chat_adapter_returns_adapter(self, factory):
        """create_streaming_chat_adapter returns the canonical chat-stream adapter."""
        runtime_owner = MagicMock()

        adapter = factory.create_streaming_chat_adapter(runtime_owner)

        assert isinstance(adapter, ServiceStreamingRuntime)
        assert adapter._orchestrator is runtime_owner


class TestCreateComplexityClassifier:
    """Tests for create_complexity_classifier method."""

    def test_create_complexity_classifier_returns_classifier(self, factory):
        """create_complexity_classifier returns a ComplexityClassifier instance."""
        from victor.framework.task import TaskComplexityService as ComplexityClassifier

        classifier = factory.create_complexity_classifier()
        assert isinstance(classifier, ComplexityClassifier)


class TestCreateActionAuthorizer:
    """Tests for create_action_authorizer method."""

    def test_create_action_authorizer_returns_authorizer(self, factory):
        """create_action_authorizer returns an ActionAuthorizer instance."""
        from victor.agent.action_authorizer import ActionAuthorizer

        authorizer = factory.create_action_authorizer()
        assert isinstance(authorizer, ActionAuthorizer)


class TestCreateSearchRouter:
    """Tests for create_search_router method."""

    def test_create_search_router_returns_router(self, factory):
        """create_search_router returns a SearchRouter instance."""
        from victor.agent.search_router import SearchRouter

        router = factory.create_search_router()
        assert isinstance(router, SearchRouter)


class TestCreateStreamingMetricsCollector:
    """Tests for create_streaming_metrics_collector method."""

    def test_create_streaming_metrics_collector_when_enabled(self, factory, mock_settings):
        """create_streaming_metrics_collector returns collector when enabled."""
        from victor.observability.analytics.streaming_metrics import (
            StreamingMetricsCollector,
        )

        mock_settings.streaming_metrics_enabled = True

        collector = factory.create_streaming_metrics_collector()
        assert isinstance(collector, StreamingMetricsCollector)

    def test_create_streaming_metrics_collector_when_disabled(self, factory, mock_settings):
        """create_streaming_metrics_collector returns None when disabled."""
        mock_settings.streaming_metrics_enabled = False

        collector = factory.create_streaming_metrics_collector()

        assert collector is None


class TestCreateUsageAnalytics:
    """Tests for create_usage_analytics method."""

    def test_create_usage_analytics_returns_analytics(self, factory):
        """create_usage_analytics returns UsageAnalytics instance."""
        from victor.agent.usage_analytics import UsageAnalytics

        analytics = factory.create_usage_analytics()
        assert isinstance(analytics, UsageAnalytics)


class TestCreateSequenceTracker:
    """Tests for create_sequence_tracker method."""

    def test_create_sequence_tracker_returns_tracker(self, factory):
        """create_sequence_tracker returns ToolSequenceTracker instance."""
        from victor.agent.tool_sequence_tracker import ToolSequenceTracker

        tracker = factory.create_sequence_tracker()
        assert isinstance(tracker, ToolSequenceTracker)


class TestCreateRecoveryHandler:
    """Tests for create_recovery_handler method."""

    def test_create_recovery_handler_when_enabled(self, factory, mock_settings):
        """create_recovery_handler returns handler when enabled."""
        from victor.agent.recovery import RecoveryHandler

        mock_settings.enable_recovery_system = True

        handler = factory.create_recovery_handler()
        # Handler may be None if creation fails due to missing dependencies
        # but should not raise an exception
        assert handler is None or isinstance(handler, RecoveryHandler)

    def test_create_recovery_handler_when_disabled(self, factory, mock_settings):
        """create_recovery_handler returns a handler from DI container when disabled."""
        mock_settings.enable_recovery_system = False

        handler = factory.create_recovery_handler()

        # Handler is always returned from DI container (may be null implementation)
        # In production, this would be _NullRecoveryHandler when disabled
        assert handler is not None


class TestCanonicalCoordinatorBuilders:
    """Tests for canonical coordination helper surfaces on OrchestratorFactory."""

    def test_create_coordination_advisor_runtime_prefers_service_protocol(
        self, factory, mock_container
    ):
        from victor.agent.services.protocols import CoordinationAdvisorRuntimeProtocol

        runtime = MagicMock(name="coordination_advisor_runtime")
        mock_container.get = MagicMock(return_value=runtime)

        result = factory.create_coordination_advisor_runtime()

        assert result is runtime
        mock_container.get.assert_called_once_with(CoordinationAdvisorRuntimeProtocol)

    def test_create_coordination_advisor_returns_framework_advisor(self, factory):
        vertical_context = MagicMock(name="vertical_context")
        advisor = MagicMock(name="coordination_advisor")

        with patch(
            "victor.framework.coordination_runtime.create_vertical_coordination_advisor",
            return_value=advisor,
        ) as mock_create:
            result = factory.create_coordination_advisor(vertical_context)

        assert result is advisor
        mock_create.assert_called_once_with(
            vertical_context=vertical_context,
            team_learner=None,
            selection_strategy=getattr(factory.settings, "team_selection_strategy", "hybrid"),
        )

    def test_create_mode_workflow_team_coordinator_removed(self, factory):
        """The deprecated mode-workflow wrapper factory should stay removed."""
        assert hasattr(factory, "create_mode_workflow_team_coordinator") is False

    def test_create_exploration_state_passed_coordinator_uses_settings_root(
        self, factory, mock_settings
    ):
        mock_settings.working_directory = "/tmp/factory-project"

        coordinator = factory.create_exploration_state_passed_coordinator()

        assert isinstance(coordinator, ExplorationStatePassedCoordinator)
        assert coordinator._project_root == Path("/tmp/factory-project")

    def test_system_prompt_coordinator_factory_removed(self, factory):
        assert not hasattr(factory, "create_system_prompt_coordinator")

    def test_prompt_runtime_support_factory_removed(self, factory):
        assert not hasattr(factory, "create_prompt_runtime_support")

    def test_create_system_prompt_state_passed_coordinator_binds_task_analyzer(
        self, factory, mock_container
    ):
        from victor.agent.protocols import TaskAnalyzerProtocol

        analyzer = MagicMock()
        mock_container.get_optional.side_effect = lambda protocol: (
            analyzer if protocol is TaskAnalyzerProtocol else None
        )

        coordinator = factory.create_system_prompt_state_passed_coordinator()

        assert isinstance(coordinator, SystemPromptStatePassedCoordinator)
        assert coordinator._task_analyzer is analyzer

    def test_create_safety_state_passed_coordinator_returns_wrapper(self, factory):
        coordinator = factory.create_safety_state_passed_coordinator()

        assert isinstance(coordinator, SafetyStatePassedCoordinator)


class TestCreateObservability:
    """Tests for create_observability method."""

    def test_create_observability_when_enabled(self, factory, mock_settings):
        """create_observability returns integration when enabled."""
        from victor.observability.integration import ObservabilityIntegration

        mock_settings.enable_observability = True

        observability = factory.create_observability()
        assert isinstance(observability, ObservabilityIntegration)

    def test_create_observability_when_disabled(self, factory, mock_settings):
        """create_observability returns None when disabled."""
        mock_settings.enable_observability = False

        observability = factory.create_observability()

        assert observability is None


class TestCreateOrchestratorFactory:
    """Tests for create_orchestrator_factory convenience function."""

    def test_create_orchestrator_factory_returns_factory(self, mock_settings, mock_provider):
        """create_orchestrator_factory returns OrchestratorFactory."""
        factory = create_orchestrator_factory(
            settings=mock_settings,
            provider=mock_provider,
            model="test-model",
        )

        assert isinstance(factory, OrchestratorFactory)
        assert factory.settings == mock_settings
        assert factory.provider == mock_provider
        assert factory.model == "test-model"


class TestContainerProperty:
    """Tests for container property lazy initialization."""

    def test_container_property_initializes_lazily(self, mock_settings, mock_provider):
        """Container property triggers bootstrap on first access."""
        factory = OrchestratorFactory(
            settings=mock_settings,
            provider=mock_provider,
            model="test-model",
        )

        # Initially None
        assert factory._container is None

        # Access triggers initialization
        with patch("victor.core.bootstrap.ensure_bootstrapped") as mock_bootstrap:
            mock_bootstrap.return_value = MagicMock()
            container = factory.container
            mock_bootstrap.assert_called_once_with(mock_settings)
            assert container is not None


class TestCreateToolCache:
    """Tests for create_tool_cache method."""

    def test_create_tool_cache_when_enabled(self, factory, mock_settings):
        """create_tool_cache returns ToolCache when enabled."""
        mock_settings.tool_cache_enabled = True
        mock_settings.tool_cache_ttl = 600
        mock_settings.tool_cache_allowlist = []

        with patch("victor.config.settings.get_project_paths") as mock_paths:
            mock_paths.return_value.global_cache_dir = "/tmp/cache"
            with patch("victor.storage.cache.tool_cache.ToolCache") as mock_cache_cls:
                mock_cache = MagicMock()
                mock_cache_cls.return_value = mock_cache
                cache = factory.create_tool_cache()
                assert cache == mock_cache

    def test_create_tool_cache_when_disabled(self, factory, mock_settings):
        """create_tool_cache returns None when disabled."""
        mock_settings.tool_cache_enabled = False

        cache = factory.create_tool_cache()

        assert cache is None

    def test_create_tool_cache_returns_none_when_initialization_fails(self, factory, mock_settings):
        """create_tool_cache degrades gracefully when disk cache init fails."""
        mock_settings.tool_cache_enabled = True
        mock_settings.tool_cache_ttl = 600
        mock_settings.tool_cache_allowlist = []

        with patch("victor.config.settings.get_project_paths") as mock_paths:
            mock_paths.return_value.global_cache_dir = "/tmp/cache"
            with patch(
                "victor.storage.cache.tool_cache.ToolCache",
                side_effect=OSError("cache init failed"),
            ):
                cache = factory.create_tool_cache()

        assert cache is None


class TestCreateMemoryComponents:
    """Tests for create_memory_components method."""

    def test_create_memory_components_when_enabled(self, factory, mock_settings):
        """create_memory_components returns tuple when enabled."""
        mock_settings.conversation_memory_enabled = True
        mock_settings.max_context_tokens = 100000
        mock_settings.response_token_reserve = 4096

        with patch("victor.config.settings.get_project_paths") as mock_paths:
            mock_project = MagicMock()
            mock_project.project_victor_dir = MagicMock()
            mock_project.project_db = "/tmp/test.db"
            mock_project.project_root = "/tmp/project"
            mock_paths.return_value = mock_project

            with patch("victor.agent.conversation.store.ConversationStore") as mock_store_cls:
                mock_store = MagicMock()
                mock_session = MagicMock()
                mock_session.session_id = "test-session-id"
                mock_store.create_session.return_value = mock_session
                mock_store_cls.return_value = mock_store

                memory, session_id = factory.create_memory_components("test_provider")

                assert memory is not None
                assert session_id == "test-session-id"

    def test_create_memory_components_retries_transient_database_lock(self, factory, mock_settings):
        """create_memory_components retries transient SQLite lock failures."""
        mock_settings.conversation_memory_enabled = True
        mock_settings.max_context_tokens = 100000
        mock_settings.response_token_reserve = 4096

        with patch("victor.config.settings.get_project_paths") as mock_paths:
            mock_project = MagicMock()
            mock_project.project_victor_dir = MagicMock()
            mock_project.project_db = "/tmp/test.db"
            mock_project.project_root = "/tmp/project"
            mock_paths.return_value = mock_project

            with (
                patch("victor.agent.conversation.store.ConversationStore") as mock_store_cls,
                patch("victor.agent.factory.runtime_builders.time.sleep") as mock_sleep,
            ):
                mock_store = MagicMock()
                mock_session = MagicMock()
                mock_session.session_id = "test-session-id"
                mock_store.create_session.return_value = mock_session
                mock_store_cls.side_effect = [
                    RuntimeError("database is locked"),
                    mock_store,
                ]

                memory, session_id = factory.create_memory_components("test_provider")

        assert memory is mock_store
        assert session_id == "test-session-id"
        assert mock_store_cls.call_count == 2
        mock_sleep.assert_called_once()

    def test_create_memory_components_records_recovered_lock_diagnostics(
        self, factory, mock_settings
    ):
        """Recovered SQLite lock retries should be available as structured diagnostics."""
        mock_settings.conversation_memory_enabled = True
        mock_settings.max_context_tokens = 100000
        mock_settings.response_token_reserve = 4096

        with patch("victor.config.settings.get_project_paths") as mock_paths:
            mock_project = MagicMock()
            mock_project.project_victor_dir = MagicMock()
            mock_project.project_db = "/tmp/test.db"
            mock_project.project_root = "/tmp/project"
            mock_paths.return_value = mock_project

            with (
                patch("victor.agent.conversation.store.ConversationStore") as mock_store_cls,
                patch("victor.agent.factory.runtime_builders.time.sleep"),
            ):
                mock_store = MagicMock()
                mock_session = MagicMock()
                mock_session.session_id = "test-session-id"
                mock_store.create_session.return_value = mock_session
                mock_store_cls.side_effect = [
                    RuntimeError("database is locked"),
                    RuntimeError("database table is locked"),
                    mock_store,
                ]

                memory, session_id = factory.create_memory_components("test_provider")

        assert memory is mock_store
        assert session_id == "test-session-id"
        diagnostics = factory.get_memory_initialization_diagnostics()
        assert diagnostics["status"] == "initialized"
        assert diagnostics["recovered_from_lock"] is True
        assert diagnostics["lock_retries"] == 2
        assert diagnostics["db_path"] == "/tmp/test.db"
        assert diagnostics["session_id"] == "test-session-id"
        assert diagnostics["last_error"] == "database table is locked"

    def test_create_memory_components_does_not_retry_non_lock_failure(self, factory, mock_settings):
        """create_memory_components only retries lock-like failures."""
        mock_settings.conversation_memory_enabled = True
        mock_settings.max_context_tokens = 100000
        mock_settings.response_token_reserve = 4096

        with patch("victor.config.settings.get_project_paths") as mock_paths:
            mock_project = MagicMock()
            mock_project.project_victor_dir = MagicMock()
            mock_project.project_db = "/tmp/test.db"
            mock_project.project_root = "/tmp/project"
            mock_paths.return_value = mock_project

            with (
                patch("victor.agent.conversation.store.ConversationStore") as mock_store_cls,
                patch("victor.agent.factory.runtime_builders.time.sleep") as mock_sleep,
            ):
                mock_store_cls.side_effect = RuntimeError("schema mismatch")

                memory, session_id = factory.create_memory_components("test_provider")

        assert memory is None
        assert session_id is None
        assert mock_store_cls.call_count == 1
        mock_sleep.assert_not_called()

    def test_create_memory_components_when_disabled(self, factory, mock_settings):
        """create_memory_components returns (None, None) when disabled."""
        mock_settings.conversation_memory_enabled = False

        memory, session_id = factory.create_memory_components("test_provider")

        assert memory is None
        assert session_id is None


class TestCreateUsageLogger:
    """Tests for create_usage_logger method."""

    def test_create_usage_logger_uses_di_if_available(self, factory, mock_container):
        """create_usage_logger uses DI container if available."""
        di_logger = MagicMock()
        mock_container.get_optional.return_value = di_logger

        logger = factory.create_usage_logger()

        assert logger == di_logger

    def test_create_usage_logger_fallback(self, factory, mock_container, mock_settings):
        """create_usage_logger falls back to basic logger."""
        mock_container.get_optional.return_value = None
        mock_settings.analytics_enabled = True

        with patch("victor.config.settings.get_project_paths") as mock_paths:
            mock_paths.return_value.global_logs_dir = MagicMock()
            mock_paths.return_value.global_logs_dir.__truediv__ = MagicMock(
                return_value="/tmp/usage.jsonl"
            )

            with patch("victor.analytics.logger.UsageLogger") as mock_logger_cls:
                mock_logger = MagicMock()
                mock_logger_cls.return_value = mock_logger

                logger = factory.create_usage_logger()

                assert logger == mock_logger


# =============================================================================
# Workflow Optimization Components Tests
# =============================================================================


class TestRuntimeIntelligenceWiring:
    """Tests for RuntimeIntelligenceService wiring on factory-built components."""

    def test_create_context_compactor_uses_runtime_intelligence(
        self, factory, mock_container, mock_settings
    ):
        """Context compactor should be wired through RuntimeIntelligenceService."""
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )

        mock_settings.context_proactive_threshold = 0.9
        mock_settings.context_min_messages_after_compact = 8
        mock_settings.max_tool_output_chars = 8192
        mock_settings.max_tool_output_lines = 200
        mock_settings.context_proactive_compaction = True
        mock_settings.tool_result_truncation = True
        decision_service = MagicMock()
        mock_container.get_optional.side_effect = lambda protocol: (
            decision_service if protocol is LLMDecisionServiceProtocol else None
        )

        controller = MagicMock()
        compactor = factory.create_context_compactor(controller)

        assert compactor._runtime_intelligence is not None
        assert compactor._runtime_intelligence._decision_service is decision_service

    def test_create_tool_selector_uses_runtime_intelligence(
        self, factory, mock_container, mock_settings
    ):
        """Tool selector should be wired through RuntimeIntelligenceService."""
        from victor.agent.services.protocols.decision_service import (
            LLMDecisionServiceProtocol,
        )

        mock_settings.fallback_max_tools = 8
        mock_settings.tools = MagicMock(
            max_tool_schema_tokens=0,
            schema_promotion_threshold=0.8,
            max_mcp_tools_per_turn=12,
        )
        decision_service = MagicMock()
        mock_container.get_optional.side_effect = lambda protocol: (
            decision_service if protocol is LLMDecisionServiceProtocol else None
        )
        tools = MagicMock()
        tools.list_tools.return_value = []
        conversation_state = MagicMock()
        unified_tracker = MagicMock()

        selector = factory.create_tool_selector(
            tools=tools,
            semantic_selector=None,
            conversation_state=conversation_state,
            unified_tracker=unified_tracker,
            model="claude-opus-4",
            provider_name="anthropic",
            tool_selection={},
            on_selection_recorded=None,
        )

        assert selector._runtime_intelligence is not None
        assert selector._runtime_intelligence._decision_service is decision_service
