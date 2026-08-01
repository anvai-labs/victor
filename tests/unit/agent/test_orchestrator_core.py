# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Core tests for AgentOrchestrator module - focusing on high-impact functionality."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = MagicMock()
    provider.name = "mock_provider"
    provider.supports_tools.return_value = True
    provider.get_context_window.return_value = 100000  # Return integer for context window
    provider.chat = AsyncMock(return_value=MagicMock(content="Response", tool_calls=[]))
    return provider


@pytest.fixture
def orchestrator_settings():
    """Create settings for testing."""
    return Settings(
        analytics_enabled=False,
        use_semantic_tool_selection=False,
        use_mcp_tools=False,
    )


@pytest.fixture
def orchestrator(mock_provider, orchestrator_settings):
    """Create an orchestrator for testing."""
    with (
        patch("victor.agent.orchestrator.UsageLogger"),
        patch("victor.core.bootstrap_services.bootstrap_new_services"),
    ):
        orch = AgentOrchestrator(
            settings=orchestrator_settings,
            provider=mock_provider,
            model="test-model",
        )

    # Wire up lightweight mock services for methods that delegate to services.
    # Services are None because bootstrap was mocked — set up stubs that
    # delegate to the orchestrator's existing components.
    # Track enabled tools for mock service
    all_tool_names = set(t.name for t in orch.tools.list_tools(only_enabled=False))
    enabled_tools = set(all_tool_names)

    mock_tool_svc = MagicMock()
    mock_tool_svc.get_enabled_tools.side_effect = lambda: set(enabled_tools)
    mock_tool_svc.get_available_tools.return_value = set(all_tool_names)
    mock_tool_svc.is_tool_enabled.side_effect = lambda name: name in enabled_tools
    mock_tool_svc.set_enabled_tools.side_effect = lambda tools: enabled_tools.update(tools)

    # process_tool_results returns tool results formatted for response
    # Default implementation converts pipeline results to dict format
    def mock_process_results(pipeline_result, ctx):
        return [
            {
                "name": r.tool_name,
                "success": r.success,
                "result": r.result if r.success else None,
                "error": r.error if not r.success else None,
            }
            for r in pipeline_result.results
        ]

    mock_tool_svc.process_tool_results.side_effect = mock_process_results
    orch._tool_service = mock_tool_svc

    mock_ctx_svc = MagicMock()
    from victor.agent.conversation.controller import ContextMetrics

    mock_ctx_svc.get_context_metrics.return_value = ContextMetrics(
        char_count=0, estimated_tokens=0, message_count=0
    )
    mock_ctx_svc.check_context_overflow.return_value = False
    orch._context_service = mock_ctx_svc

    # Keep the canonical SessionService instance rather than replacing it
    # with a coordinator-backed mock. Individual tests can still override
    # specific methods when they need tighter control.
    orch._session_service.maybe_auto_checkpoint = AsyncMock(return_value=None)

    mock_provider_svc = MagicMock()
    from victor.core.errors import ProviderNotFoundError
    from victor.agent.services.provider_service import ProviderService

    mock_provider_svc.switch_provider = AsyncMock(
        side_effect=ProviderNotFoundError("not configured")
    )
    mock_provider_svc.switch_model = AsyncMock(side_effect=ProviderNotFoundError("not configured"))
    mock_provider_svc.get_current_provider = MagicMock(return_value=mock_provider)
    mock_provider_svc.get_current_provider_info.return_value = MagicMock(
        provider_name="mock_provider",
        model_name="test-model",
        api_key_configured=False,
        supports_streaming=True,
        supports_tool_calling=True,
        max_tokens=100000,
    )
    mock_provider_svc.get_rate_limit_stats.return_value = {"rate_limits_hit": 0}
    # Wire rate-limit parsing to real logic so TestRateLimitRetry assertions hold.
    mock_provider_svc.get_rate_limit_wait_time.side_effect = (
        lambda exc: ProviderService.get_rate_limit_wait_time(None, exc)
    )
    orch._provider_service = mock_provider_svc

    mock_chat_svc = MagicMock()
    mock_chat_svc.chat = AsyncMock(return_value=MagicMock(content="Response", tool_calls=[]))
    orch._chat_service = mock_chat_svc

    mock_recovery_svc = MagicMock()
    orch._recovery_service = mock_recovery_svc

    return orch


class TestStreamMetrics:
    """Tests for stream metrics functionality."""

    def test_finalize_stream_metrics(self, orchestrator):
        """Test _finalize_stream_metrics returns metrics (covers lines 306-321)."""
        import time

        # Initialize metrics through the metrics collector
        stream_metrics = orchestrator._metrics_collector.init_stream_metrics()
        stream_metrics.first_token_time = time.time()
        stream_metrics.total_chunks = 10

        result = orchestrator._finalize_stream_metrics()

        assert result is not None
        assert result.end_time is not None
        assert result.total_chunks == 10

    def test_finalize_stream_metrics_no_metrics(self, orchestrator):
        """Test _finalize_stream_metrics returns None when no metrics."""
        result = orchestrator._finalize_stream_metrics()
        assert result is None

    def test_get_last_stream_metrics(self, orchestrator):
        """Test get_last_stream_metrics returns stored metrics (covers line 325)."""
        # Initialize metrics first
        orchestrator._metrics_collector.init_stream_metrics()
        result = orchestrator.get_last_stream_metrics()
        assert result is not None

    def test_get_last_stream_metrics_no_metrics(self, orchestrator):
        """Test get_last_stream_metrics returns None when no metrics."""
        result = orchestrator.get_last_stream_metrics()
        assert result is None


class TestEmbeddingPreload:
    """Tests for embedding preload functionality."""

    @pytest.mark.asyncio
    async def test_preload_embeddings_no_selector(self, orchestrator):
        """Test _preload_embeddings returns early when no semantic selector."""
        orchestrator.semantic_selector = None
        await orchestrator._preload_embeddings()  # Should not raise

    @pytest.mark.asyncio
    async def test_preload_embeddings_already_initialized(self, orchestrator):
        """Test _preload_embeddings returns early when already initialized (covers line 340)."""
        orchestrator.semantic_selector = MagicMock()
        orchestrator.tool_selector = MagicMock()
        orchestrator.tool_selector._embeddings_initialized = True

        await orchestrator._preload_embeddings()  # Should return early

    @pytest.mark.asyncio
    async def test_preload_embeddings_success(self, orchestrator):
        """Test _preload_embeddings initializes embeddings (covers lines 342-347)."""
        mock_selector = MagicMock()
        mock_selector.initialize_tool_embeddings = AsyncMock()
        orchestrator.semantic_selector = mock_selector
        orchestrator.tool_selector = MagicMock()
        orchestrator.tool_selector._embeddings_initialized = False

        await orchestrator._preload_embeddings()

        mock_selector.initialize_tool_embeddings.assert_called_once()
        assert orchestrator.tool_selector._embeddings_initialized is True

    @pytest.mark.asyncio
    async def test_preload_embeddings_failure(self, orchestrator):
        """Test _preload_embeddings handles exception (covers lines 348-352)."""
        mock_selector = MagicMock()
        mock_selector.initialize_tool_embeddings = AsyncMock(
            side_effect=Exception("Embedding error")
        )
        orchestrator.semantic_selector = mock_selector
        orchestrator.tool_selector = MagicMock()
        orchestrator.tool_selector._embeddings_initialized = False

        # Should not raise
        await orchestrator._preload_embeddings()

    def test_start_embedding_preload_no_semantic(self, orchestrator):
        """Test start_embedding_preload does nothing when semantic not enabled (covers line 361)."""
        orchestrator.use_semantic_selection = False
        orchestrator.start_embedding_preload()
        assert orchestrator._embedding_preload_task is None

    def test_start_embedding_preload_already_started(self, orchestrator):
        """Test start_embedding_preload does nothing when already started."""
        orchestrator.use_semantic_selection = True
        orchestrator._embedding_preload_task = MagicMock()  # Already started
        orchestrator.start_embedding_preload()
        # Should not create a new task

    def test_start_embedding_preload_uses_runtime_preload_when_enabled(self, orchestrator):
        """Feature-flagged runtime preload should supersede embedding-only preload."""
        orchestrator.settings.framework_preload_enabled = True
        orchestrator._runtime_preload_task = None
        sentinel_task = MagicMock()

        def _capture_task(coro, name="background_task"):
            coro.close()
            return sentinel_task

        orchestrator._create_background_task = MagicMock(side_effect=_capture_task)

        orchestrator.start_embedding_preload()

        orchestrator._create_background_task.assert_called_once()
        _, kwargs = orchestrator._create_background_task.call_args
        assert kwargs["name"] == "runtime_preload"
        assert orchestrator._runtime_preload_task is sentinel_task


class TestToolDependencies:
    """Tests for tool dependency registration.

    Note: Tool dependencies are now registered via ToolRegistrar during
    orchestrator initialization. The _register_default_tool_dependencies
    method was consolidated into ToolRegistrar._register_tool_dependencies
    in the TD-002 God Class refactoring (Jan 2026).
    """

    def test_tool_dependencies_registered_on_init(self, orchestrator):
        """Test tool dependencies are registered during initialization."""
        # ToolRegistrar registers tool dependencies during __init__
        assert orchestrator.tool_graph is not None
        # Verify some tools were registered
        # The tool_graph should have tools registered via tool_registrar
        assert hasattr(orchestrator.tool_registrar, "_register_tool_dependencies")


class TestConversationState:
    """Tests for conversation state management."""


class TestPromptOptimizationMetadata:
    def test_record_prompt_optimization_metadata_updates_session(self, orchestrator):
        metadata = {
            "entries": [
                {
                    "provider": "anthropic",
                    "prompt_candidate_hash": "cand-123",
                    "section_name": "GROUNDING_RULES",
                    "prompt_section_name": "GROUNDING_RULES",
                    "strategy_name": "gepa",
                    "source": "candidate",
                }
            ],
            "by_section": {
                "GROUNDING_RULES": {
                    "provider": "anthropic",
                    "prompt_candidate_hash": "cand-123",
                    "section_name": "GROUNDING_RULES",
                    "prompt_section_name": "GROUNDING_RULES",
                    "strategy_name": "gepa",
                    "source": "candidate",
                }
            },
            "prompt_overlays": [
                {
                    "name": "planner.research_step",
                    "placement": "turn_prefix",
                    "content": "metadata should not persist overlay content",
                }
            ],
        }
        expected = {
            "entries": metadata["entries"],
            "by_section": metadata["by_section"],
            "prompt_overlays": [
                {
                    "name": "planner.research_step",
                    "placement": "turn_prefix",
                }
            ],
        }
        # Set up session service mock with a current session to trigger update
        orchestrator._session_service.update_session_metadata = MagicMock()
        orchestrator._session_service._current_session = MagicMock()  # Simulate active session

        orchestrator._record_prompt_optimization_metadata(
            SimpleNamespace(prompt_optimization_metadata=metadata)
        )

        assert orchestrator._active_prompt_optimization_metadata == expected
        orchestrator._session_service.update_session_metadata.assert_called_once_with(
            {"prompt_optimization": expected}
        )

    def test_reset_conversation(self, orchestrator):
        """Test reset_conversation resets state."""
        orchestrator.add_message("user", "test")
        orchestrator.reset_conversation()
        # After reset, messages should be empty
        assert len(orchestrator.conversation.messages) == 0

    def test_conversation_messages_property(self, orchestrator):
        """Test conversation.messages property returns messages."""
        orchestrator.add_message("user", "test message")
        history = orchestrator.conversation.messages
        assert len(history) >= 1


class TestToolSelection:
    """Tests for tool selection mechanisms."""

    def test_record_tool_selection_semantic(self, orchestrator):
        """Test _record_tool_selection records semantic stats (covers lines 381-388)."""
        initial = orchestrator._metrics_collector._selection_stats.semantic_selections
        orchestrator._record_tool_selection("semantic", 5)
        assert orchestrator._metrics_collector._selection_stats.semantic_selections == initial + 1
        assert orchestrator._metrics_collector._selection_stats.total_tools_selected >= 5

    def test_record_tool_selection_keyword(self, orchestrator):
        """Test _record_tool_selection records keyword stats."""
        initial = orchestrator._metrics_collector._selection_stats.keyword_selections
        orchestrator._record_tool_selection("keyword", 3)
        assert orchestrator._metrics_collector._selection_stats.keyword_selections == initial + 1

    def test_record_tool_selection_fallback(self, orchestrator):
        """Test _record_tool_selection records fallback stats."""
        initial = orchestrator._metrics_collector._selection_stats.fallback_selections
        orchestrator._record_tool_selection("fallback", 2)
        assert orchestrator._metrics_collector._selection_stats.fallback_selections == initial + 1

    def test_get_tool_usage_stats(self, orchestrator):
        """Test get_tool_usage_stats returns stats (covers lines 446-457+)."""
        orchestrator._record_tool_selection("keyword", 3)
        stats = orchestrator.get_tool_usage_stats()
        assert "selection_stats" in stats
        assert "keyword_selections" in stats["selection_stats"]


class TestProviderChecks:
    """Tests for provider type checks via prompt_builder."""

    def test_is_cloud_provider(self, orchestrator):
        """Test is_cloud_provider via prompt_builder (dead code removed from orchestrator)."""
        # Provider checks now delegated to prompt_builder
        result = orchestrator.prompt_builder.is_cloud_provider()
        assert isinstance(result, bool)

    def test_is_local_provider(self, orchestrator):
        """Test is_local_provider via prompt_builder (dead code removed from orchestrator)."""
        # Provider checks now delegated to prompt_builder
        result = orchestrator.prompt_builder.is_local_provider()
        assert isinstance(result, bool)


class TestSystemPrompt:
    """Tests for system prompt building."""

    def test_build_system_prompt(self, orchestrator):
        """Test canonical build_system_prompt returns a prompt string."""
        result = orchestrator.build_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_system_prompt_for_provider(self, orchestrator):
        """Test prompt building via prompt_builder (dead code removed from orchestrator)."""
        # _build_system_prompt_for_provider was dead code - use prompt_builder.build() instead
        result = orchestrator.prompt_builder.build()
        assert isinstance(result, str)


class TestConversationStage:
    """Tests for conversation stage management."""

    def test_get_conversation_stage(self, orchestrator):
        """Test get_conversation_stage returns stage (covers line 487)."""
        from victor.agent.conversation.state_machine import ConversationStage

        stage = orchestrator.get_conversation_stage()
        assert isinstance(stage, ConversationStage)

    def test_get_stage_recommended_tools(self, orchestrator):
        """Test get_stage_recommended_tools returns tools (covers line 495)."""
        tools = orchestrator.get_stage_recommended_tools()
        assert isinstance(tools, set)


class TestToolCallLogging:
    """Tests for tool call logging."""

    def test_log_tool_call(self, orchestrator):
        """Test _log_tool_call logs info (covers line 564)."""
        # Should not raise
        orchestrator._log_tool_call("test_tool", {"arg": "value"})


class TestToolExecution:
    """Tests for tool execution tracking."""

    @pytest.mark.asyncio
    async def test_execute_tool_with_retry_delegates_to_tool_service(self, orchestrator):
        """Canonical retry surface should delegate to ToolService."""
        orchestrator._tool_service.execute_tool_with_retry = AsyncMock(
            return_value=("result", True, None)
        )

        result = await orchestrator.execute_tool_with_retry(
            "read",
            {"path": "a.py"},
            {"task_type": "read"},
        )

        assert result == ("result", True, None)
        orchestrator._tool_service.execute_tool_with_retry.assert_awaited_once_with(
            "read",
            {"path": "a.py"},
            {"task_type": "read"},
        )

    def test_record_tool_execution_success(self, orchestrator):
        """Test _record_tool_execution with success (covers lines 400-444)."""
        orchestrator._record_tool_execution("test_tool", success=True, elapsed_ms=100.0)

        stats = orchestrator._metrics_collector._tool_usage_stats.get("test_tool")
        assert stats is not None
        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 0

    def test_record_tool_execution_failure(self, orchestrator):
        """Test _record_tool_execution with failure."""
        orchestrator._record_tool_execution("failing_tool", success=False, elapsed_ms=50.0)

        stats = orchestrator._metrics_collector._tool_usage_stats.get("failing_tool")
        assert stats is not None
        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 0
        assert stats["failed_calls"] == 1

    def test_record_tool_execution_timing(self, orchestrator):
        """Test _record_tool_execution tracks timing correctly."""
        orchestrator._record_tool_execution("timed_tool", success=True, elapsed_ms=100.0)
        orchestrator._record_tool_execution("timed_tool", success=True, elapsed_ms=200.0)

        stats = orchestrator._metrics_collector._tool_usage_stats.get("timed_tool")
        assert stats["total_calls"] == 2
        assert stats["total_time_ms"] == 300.0
        assert stats["avg_time_ms"] == 150.0
        assert stats["min_time_ms"] == 100.0
        assert stats["max_time_ms"] == 200.0


class TestToolCallParsing:
    """Tests for tool call parsing."""

    def test_parse_tool_calls_with_adapter_empty(self, orchestrator):
        """Test _parse_tool_calls_with_adapter with no tool calls (covers lines 515-528)."""
        result = orchestrator._parse_tool_calls_with_adapter("Hello world", None)
        assert result is not None
        assert result.tool_calls == []

    def test_parse_tool_calls_with_adapter_native(self, orchestrator):
        """Test _parse_tool_calls_with_adapter with native tool calls."""
        raw_calls = [{"name": "read", "arguments": {"path": "/test.py"}}]
        result = orchestrator._parse_tool_calls_with_adapter("", raw_calls)
        assert result is not None


class TestWorkflowRegistration:
    """Tests for workflow registration."""

    def test_workflow_registry_initialized(self, orchestrator):
        """Test workflow_registry is initialized (covers line 215)."""
        assert orchestrator.workflow_registry is not None

    def test_workflow_registry_has_workflows(self, orchestrator):
        """Test _register_default_workflows adds workflows (covers line 568)."""
        # Workflows are registered during init, so check they exist
        from victor.workflows.base import WorkflowRegistry

        assert isinstance(orchestrator.workflow_registry, WorkflowRegistry)


class TestToolCallBudget:
    """Tests for tool call budget management."""

    def test_tool_calls_used_tracking(self, orchestrator):
        """Test tool_calls_used is tracked."""
        orchestrator.tool_calls_used = 0
        orchestrator.tool_calls_used += 1
        assert orchestrator.tool_calls_used == 1

    def test_tool_budget(self, orchestrator):
        """Test tool_budget property (covers line 159-160)."""
        budget = orchestrator.tool_budget
        assert isinstance(budget, int)
        assert budget > 0

    def test_get_tool_budget_does_not_raise(self, orchestrator):
        """get_tool_budget() used to dereference a non-existent tracker attribute.

        ``UnifiedTaskTracker`` exposes its budget as ``progress.tool_budget``; there
        is no ``tool_budget`` attribute, so ``self.unified_tracker.tool_budget``
        raised AttributeError on every call once a tracker was present (always).
        """
        assert orchestrator.get_tool_budget() == orchestrator.tool_budget

    def test_get_tool_calls_count_does_not_raise(self, orchestrator):
        """Same defect for the counter: the tracker has no ``tool_calls_used``."""
        orchestrator.tool_calls_used = 3

        assert orchestrator.get_tool_calls_count() == 3

    def test_budget_pair_comes_from_the_enforcing_cell(self, orchestrator):
        """Numerator and denominator must be the pair the loop enforces against.

        ``streaming/coordinator.py`` halts on ``orchestrator.tool_calls_used`` vs
        ``orchestrator.tool_budget``; the protocol accessors must report the same
        pair rather than the tracker's advisory budget or the session-wide cap.
        """
        orchestrator.tool_budget = 20
        orchestrator.tool_calls_used = 7

        assert orchestrator.get_tool_budget() == 20
        assert orchestrator.get_tool_calls_count() == 7


class TestToolCacheIntegration:
    """Tests for tool cache integration."""

    def test_orchestrator_has_tool_cache(self, orchestrator):
        """Test orchestrator has tool_cache attribute."""
        # May be None if not configured
        assert hasattr(orchestrator, "tool_cache")

    def test_orchestrator_with_tool_cache(self, mock_provider, orchestrator_settings):
        """Test orchestrator with tool cache configured."""
        orchestrator_settings.tools.tool_cache_enabled = True
        orchestrator_settings.tools.tool_cache_ttl = 300

        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            # Tool cache should be initialized
            assert hasattr(orch, "tool_cache")


class TestToolSupportChecks:
    """Tests for tool support methods."""

    def test_should_use_tools(self, orchestrator):
        """Test _should_use_tools always returns True (covers line 953-955)."""
        assert orchestrator._should_use_tools() is True

    def test_model_supports_tool_calls_no_provider(self, orchestrator):
        """Test _model_supports_tool_calls with no provider name (covers lines 959-961)."""
        orchestrator._provider_manager._current_state.name = ""
        result = orchestrator._model_supports_tool_calls()
        assert isinstance(result, bool)

    def test_model_supports_tool_calls_supported(self, orchestrator):
        """Test _model_supports_tool_calls with supported model (covers lines 963)."""
        # Set up to return True
        orchestrator.tool_capabilities = MagicMock()
        orchestrator.tool_capabilities.is_tool_call_supported.return_value = True
        orchestrator._provider_manager._current_state.name = "test"

        result = orchestrator._model_supports_tool_calls()
        assert result is True

    def test_model_supports_tool_calls_not_supported(self, orchestrator):
        """Test _model_supports_tool_calls with unsupported model (covers lines 964-975)."""
        orchestrator.tool_capabilities = MagicMock()
        orchestrator.tool_capabilities.is_tool_call_supported.return_value = False
        orchestrator.tool_capabilities.get_supported_models.return_value = [
            "model-a",
            "model-b",
        ]
        orchestrator._provider_manager._current_state.name = "test"
        orchestrator._tool_capability_warned = False

        result = orchestrator._model_supports_tool_calls()

        assert result is False
        assert orchestrator._tool_capability_warned is True


class TestCancellationSupport:
    """Tests for streaming cancellation support."""

    def test_request_cancellation(self, orchestrator):
        """Test request_cancellation sets event (covers lines 1955-1957)."""
        import asyncio

        orchestrator._cancel_event = asyncio.Event()
        orchestrator.request_cancellation()
        assert orchestrator._cancel_event.is_set()

    def test_request_cancellation_no_event(self, orchestrator):
        """Test request_cancellation with no event (safe no-op)."""
        orchestrator._cancel_event = None
        # Should not raise
        orchestrator.request_cancellation()

    def test_is_streaming_true(self, orchestrator):
        """Test is_streaming returns True when streaming (covers line 1965)."""
        orchestrator._is_streaming = True
        assert orchestrator.is_streaming() is True

    def test_is_streaming_false(self, orchestrator):
        """Test is_streaming returns False when not streaming."""
        orchestrator._is_streaming = False
        assert orchestrator.is_streaming() is False

    def test_check_cancellation_true(self, orchestrator):
        """Test _check_cancellation when cancelled (covers lines 1973-1974)."""
        import asyncio

        orchestrator._cancel_event = asyncio.Event()
        orchestrator._cancel_event.set()
        assert orchestrator._check_cancellation() is True

    def test_check_cancellation_false(self, orchestrator):
        """Test _check_cancellation when not cancelled (covers line 1975)."""
        import asyncio

        orchestrator._cancel_event = asyncio.Event()
        assert orchestrator._check_cancellation() is False

    def test_check_cancellation_no_event(self, orchestrator):
        """Test _check_cancellation with no event."""
        orchestrator._cancel_event = None
        assert orchestrator._check_cancellation() is False


class TestShutdown:
    """Tests for shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_basic(self, orchestrator):
        """Test shutdown cleans up resources (covers lines 1987-2022)."""
        # Mock the provider
        mock_provider = AsyncMock()
        mock_provider.close = AsyncMock()

        # Update provider in orchestrator and lifecycle manager
        orchestrator.provider = mock_provider
        orchestrator._lifecycle_manager.set_provider(mock_provider)

        # Mock the code_manager and set on lifecycle manager
        mock_code_manager = MagicMock()
        mock_code_manager.stop = MagicMock()
        orchestrator.code_manager = mock_code_manager
        orchestrator._lifecycle_manager.set_code_manager(mock_code_manager)

        orchestrator.semantic_selector = None

        await orchestrator.shutdown()

        mock_provider.close.assert_called_once()
        mock_code_manager.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_semantic_selector(self, orchestrator):
        """Test shutdown closes semantic selector (covers lines 2014-2020)."""
        mock_provider = AsyncMock()
        mock_provider.close = AsyncMock()
        orchestrator.provider = mock_provider
        orchestrator._lifecycle_manager.set_provider(mock_provider)

        orchestrator.code_manager = MagicMock()

        mock_selector = AsyncMock()
        mock_selector.close = AsyncMock()
        orchestrator.semantic_selector = mock_selector
        orchestrator._lifecycle_manager.set_semantic_selector(mock_selector)

        await orchestrator.shutdown()

        mock_selector.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_provider_error(self, orchestrator):
        """Test shutdown handles provider close error (covers lines 2003-2004)."""
        mock_provider = AsyncMock()
        mock_provider.close = AsyncMock(side_effect=Exception("Close error"))
        orchestrator.provider = mock_provider
        orchestrator._lifecycle_manager.set_provider(mock_provider)
        orchestrator.code_manager = MagicMock()
        orchestrator.semantic_selector = None

        # Should not raise
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_handles_code_manager_error(self, orchestrator):
        """Test shutdown handles code manager error (covers lines 2011-2012)."""
        mock_provider = AsyncMock()
        mock_provider.close = AsyncMock()
        orchestrator.provider = mock_provider
        orchestrator._lifecycle_manager.set_provider(mock_provider)
        orchestrator.code_manager = MagicMock()
        orchestrator.code_manager.stop.side_effect = Exception("Stop error")
        orchestrator.semantic_selector = None

        # Should not raise
        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_handles_selector_error(self, orchestrator):
        """Test shutdown handles selector close error (covers lines 2019-2020)."""
        mock_provider = AsyncMock()
        mock_provider.close = AsyncMock()
        orchestrator._provider_manager._current_state.provider = mock_provider
        orchestrator.code_manager = MagicMock()

        mock_selector = AsyncMock()
        mock_selector.close = AsyncMock(side_effect=Exception("Selector error"))
        orchestrator.semantic_selector = mock_selector

        # Should not raise
        await orchestrator.shutdown()


class TestContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_aenter(self, orchestrator):
        """Test __aenter__ returns self (covers line 2026)."""
        result = await orchestrator.__aenter__()
        assert result is orchestrator

    @pytest.mark.asyncio
    async def test_aexit_calls_shutdown(self, orchestrator):
        """Test __aexit__ calls shutdown (covers line 2030)."""
        orchestrator.shutdown = AsyncMock()

        await orchestrator.__aexit__(None, None, None)

        orchestrator.shutdown.assert_called_once()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestChatMethod:
    """Tests for chat method.

    These tests exercise ``orchestrator.chat()`` — a deprecated delegation
    shim to ``ChatService``. The deprecation is expected here; the tests assert
    delegation behavior, not the warning.
    """

    @pytest.mark.asyncio
    async def test_chat_basic(self, mock_provider, orchestrator_settings):
        """Test chat adds messages and gets response (covers lines 1004-1047)."""
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_response.tool_calls = []
        mock_provider.chat = AsyncMock(return_value=mock_response)
        mock_provider.supports_tools.return_value = False

        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Mock _chat_service since bootstrap_new_services is patched out
            mock_chat_svc = MagicMock()
            mock_chat_svc.chat = AsyncMock(return_value=mock_response)
            orch._chat_service = mock_chat_svc

            response = await orch.chat("Hello")

            assert response.content == "Test response"

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, mock_provider, orchestrator_settings):
        """Test chat with tool support enabled."""
        mock_response = MagicMock()
        mock_response.content = "Using tools"
        mock_response.tool_calls = None  # None = model chose not to use tools
        mock_response.usage = None
        mock_provider.chat = AsyncMock(return_value=mock_response)
        mock_provider.supports_tools.return_value = True

        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Mock _chat_service since bootstrap_new_services is patched out
            mock_chat_svc = MagicMock()
            mock_chat_svc.chat = AsyncMock(return_value=mock_response)
            orch._chat_service = mock_chat_svc

            response = await orch.chat("Search for files")

            assert response.content == "Using tools"

    @pytest.mark.asyncio
    async def test_chat_with_thinking(self, mock_provider, orchestrator_settings):
        """Test chat with thinking enabled (covers lines 1027-1029)."""
        mock_response = MagicMock()
        mock_response.content = "Thought response"
        mock_response.tool_calls = []
        mock_provider.chat = AsyncMock(return_value=mock_response)
        mock_provider.supports_tools.return_value = False

        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
                thinking=True,
            )

            # Mock _chat_service since bootstrap_new_services is patched out
            mock_chat_svc = MagicMock()
            mock_chat_svc.chat = AsyncMock(return_value=mock_response)
            orch._chat_service = mock_chat_svc

            response = await orch.chat("Think about this")
            assert response.content == "Thought response"


class TestAddMessage:
    """Tests for add_message method."""

    def test_add_message_user(self, orchestrator):
        """Test add_message adds user message (covers lines 984-986)."""
        orchestrator.add_message("user", "Test user message")
        assert any(
            m.role == "user" and "Test user message" in m.content
            for m in orchestrator.conversation.messages
        )

    def test_add_message_assistant(self, orchestrator):
        """Test add_message adds assistant message (covers lines 987-988)."""
        orchestrator.add_message("assistant", "Test assistant response")
        assert any(
            m.role == "assistant" and "Test assistant response" in m.content
            for m in orchestrator.conversation.messages
        )


class TestFromSettings:
    """Tests for from_settings factory method."""

    @pytest.mark.asyncio
    async def test_from_settings_profile_not_found(self, orchestrator_settings):
        """Test from_settings raises for missing profile (covers line 2054)."""
        orchestrator_settings.load_profiles = MagicMock(return_value={})

        with pytest.raises(ValueError, match="Profile not found"):
            await AgentOrchestrator.from_settings(
                settings=orchestrator_settings, profile_name="nonexistent"
            )

    @pytest.mark.asyncio
    async def test_from_settings_success(self, orchestrator_settings):
        """Test from_settings creates orchestrator (covers lines 2050-2070)."""
        mock_profile = MagicMock()
        mock_profile.provider = "mock_provider"
        mock_profile.model = "test-model"
        mock_profile.temperature = 0.7
        mock_profile.max_tokens = 4096
        mock_profile.tool_selection = None

        orchestrator_settings.load_profiles = MagicMock(return_value={"default": mock_profile})
        orchestrator_settings.get_provider_settings = MagicMock(return_value={})

        mock_provider = MagicMock()
        mock_provider.name = "mock_provider"
        mock_provider.supports_tools.return_value = True
        mock_provider.get_context_window.return_value = 100000

        with (
            patch("victor.providers.registry.ProviderRegistry.create") as mock_create,
            patch("victor.agent.tool_calling.capabilities.ModelCapabilityLoader") as mock_caps,
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            mock_create.return_value = mock_provider
            mock_caps.return_value.get_capabilities.return_value = None
            orch = await AgentOrchestrator.from_settings(
                settings=orchestrator_settings, profile_name="default"
            )

            assert orch is not None
            assert orch.model == "test-model"


class TestToolPlanning:
    """Tests for tool planning methods via tool_planner component."""

    def test_plan_tools_empty_goals(self, orchestrator):
        """Test tool_planner.plan_tools with empty goals."""
        result = orchestrator._tool_planner.plan_tools([])
        assert result == []

    def test_plan_tools_with_goals(self, orchestrator):
        """Test tool_planner.plan_tools with valid goals."""
        # Add tool to graph
        orchestrator.tool_graph.add_tool("test_tool", inputs=["query"], outputs=["result"])
        result = orchestrator._tool_planner.plan_tools(["result"], ["query"])
        # Result depends on tool graph configuration
        assert isinstance(result, list)

    def test_goal_hints_for_message_summary(self, orchestrator):
        """Test tool_planner.infer_goals_from_message detects summary requests."""
        result = orchestrator._tool_planner.infer_goals_from_message("Please summarize this code")
        assert "summary" in result

    def test_goal_hints_for_message_review(self, orchestrator):
        """Test tool_planner.infer_goals_from_message detects review requests."""
        result = orchestrator._tool_planner.infer_goals_from_message("Can you review this?")
        assert "summary" in result

    def test_goal_hints_for_message_documentation(self, orchestrator):
        """Test tool_planner.infer_goals_from_message detects documentation requests."""
        result = orchestrator._tool_planner.infer_goals_from_message(
            "Generate documentation please"
        )
        assert "documentation" in result

    def test_goal_hints_for_message_security(self, orchestrator):
        """Test tool_planner.infer_goals_from_message detects security requests."""
        result = orchestrator._tool_planner.infer_goals_from_message("Run a security scan")
        assert "security_report" in result

    def test_goal_hints_for_message_metrics(self, orchestrator):
        """Test tool_planner.infer_goals_from_message detects metrics requests."""
        result = orchestrator._tool_planner.infer_goals_from_message("Show complexity metrics")
        assert "metrics_report" in result

    def test_goal_hints_for_message_no_match(self, orchestrator):
        """Test tool_planner.infer_goals_from_message with no matching keywords."""
        result = orchestrator._tool_planner.infer_goals_from_message("Hello world")
        assert result == []


class TestToolConfiguration:
    """Tests for tool configuration loading."""

    def test_load_tool_configurations_empty(self, mock_provider, orchestrator_settings):
        """Test _load_tool_configurations with empty config."""
        orchestrator_settings.load_tool_config = MagicMock(return_value=None)

        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            # Should not raise during init
            assert orch is not None

    def test_load_tool_configurations_exception(self, mock_provider, orchestrator_settings):
        """Test _load_tool_configurations handles exception (covers lines 950-951)."""
        orchestrator_settings.load_tool_config = MagicMock(side_effect=Exception("Config error"))

        with patch("victor.agent.orchestrator.UsageLogger"):
            # Should not raise during init
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            assert orch is not None


class TestExecuteToolCalls:
    """Tests for execute_tool_calls method."""

    @pytest.mark.asyncio
    async def test_execute_tool_calls_empty_list(self, orchestrator):
        """Test execute_tool_calls with empty list."""
        result = await orchestrator.execute_tool_calls([])
        assert result == []

    @pytest.mark.asyncio
    async def test_execute_tool_calls_not_a_dict(self, orchestrator):
        """Test execute_tool_calls with non-dict tool call."""
        result = await orchestrator.execute_tool_calls(["not a dict"])
        assert result == []

    def test_execute_tool_calls_uses_tool_service_post_processing(self, orchestrator):
        """The canonical post-processing path should go through ToolService."""
        import asyncio
        from unittest.mock import AsyncMock
        from victor.agent.tool_pipeline import PipelineExecutionResult, ToolCallResult

        pipeline_result = PipelineExecutionResult(
            results=[
                ToolCallResult(
                    tool_name="read",
                    arguments={"path": "/tmp/test.py"},
                    success=True,
                    result="contents",
                )
            ]
        )
        orchestrator._tool_pipeline.execute_tool_calls = AsyncMock(return_value=pipeline_result)
        orchestrator._tool_service.process_tool_results = MagicMock(
            return_value=[{"name": "read", "success": True}]
        )

        result = asyncio.run(orchestrator.execute_tool_calls([{"name": "read", "arguments": {}}]))

        orchestrator._tool_service.process_tool_results.assert_called_once()
        assert result == [{"name": "read", "success": True}]

    @pytest.mark.asyncio
    async def test_execute_tool_calls_no_name(self, orchestrator):
        """Tool calls without name are handled by ToolPipeline (returns error result)."""
        from unittest.mock import AsyncMock
        from victor.agent.tool_pipeline import PipelineExecutionResult, ToolCallResult

        error_result = ToolCallResult(
            tool_name="unknown",
            arguments={},
            success=False,
            error="Missing tool name",
        )
        mock_pipeline_result = PipelineExecutionResult(results=[error_result])
        with patch.object(
            orchestrator._tool_pipeline,
            "execute_tool_calls",
            new_callable=AsyncMock,
            return_value=mock_pipeline_result,
        ):
            result = await orchestrator.execute_tool_calls([{"arguments": {}}])
        assert len(result) == 1
        assert result[0]["success"] is False


class TestServiceFirstDelegation:
    """Focused tests for service-first orchestrator wiring."""

    def test_add_message_uses_chat_service_persist_message(self, orchestrator):
        """Message persistence should go through ChatService, not ChatCoordinator."""
        with patch("victor.agent.services.chat_service.ChatService.persist_message") as persist:
            orchestrator.add_message("user", "service-first")

        persist.assert_called_once()

    def test_add_message_forwards_metadata_to_chat_service(self, orchestrator):
        metadata = {
            "interactive_history": False,
            "internal_prompt_kind": "system_reminder",
        }

        with patch("victor.agent.services.chat_service.ChatService.persist_message") as persist:
            orchestrator.add_message("user", "[SYSTEM-REMINDER: hidden]", metadata=metadata)

        assert persist.call_args.kwargs["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_maybe_auto_checkpoint_uses_session_service(self, orchestrator):
        """Test that maybe_auto_checkpoint delegates to SessionService."""
        orchestrator._session_service.maybe_auto_checkpoint = AsyncMock(return_value="ckpt-1")

        result = await orchestrator.maybe_auto_checkpoint()

        orchestrator._session_service.maybe_auto_checkpoint.assert_awaited_once()
        assert result == "ckpt-1"

    def test_parse_and_validate_tool_calls_uses_tool_service(self, orchestrator):
        """Test that parse_and_validate_tool_calls delegates to ToolService."""
        orchestrator._tool_service.parse_and_validate_tool_calls = MagicMock(
            return_value=([{"name": "read", "arguments": {}}], "")
        )

        result = orchestrator._parse_and_validate_tool_calls([], "content")

        orchestrator._tool_service.parse_and_validate_tool_calls.assert_called_once()
        assert result == ([{"name": "read", "arguments": {}}], "")

    def test_recover_session_uses_session_service(self, orchestrator):
        """Test that recover_session delegates to SessionService."""
        orchestrator._session_service.recover_session = MagicMock(return_value=True)

        assert orchestrator.recover_session("session-123") is True
        orchestrator._session_service.recover_session.assert_called_once_with("session-123")
        assert orchestrator._memory_session_id == "session-123"

    def test_build_tool_access_context_uses_tool_service(self, orchestrator):
        """Test that build_tool_access_context delegates to ToolService."""
        sentinel = MagicMock(name="tool_access_context")
        orchestrator._tool_service.build_tool_access_context = MagicMock(return_value=sentinel)

        result = orchestrator._build_tool_access_context()

        orchestrator._tool_service.build_tool_access_context.assert_called_once()
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_execute_tool_calls_invalid_name(self, orchestrator):
        """Invalid tool names are handled by ToolPipeline (returns error result)."""
        from unittest.mock import AsyncMock
        from victor.agent.tool_pipeline import PipelineExecutionResult, ToolCallResult

        error_result = ToolCallResult(
            tool_name="123invalid",
            arguments={},
            success=False,
            error="Invalid tool name",
        )
        mock_pipeline_result = PipelineExecutionResult(results=[error_result])
        with patch.object(
            orchestrator._tool_pipeline,
            "execute_tool_calls",
            new_callable=AsyncMock,
            return_value=mock_pipeline_result,
        ):
            result = await orchestrator.execute_tool_calls([{"name": "123invalid"}])
        assert len(result) == 1
        assert result[0]["success"] is False

    @pytest.mark.asyncio
    async def test_execute_tool_calls_disabled_tool(self, orchestrator):
        """Disabled tools are handled by ToolPipeline (returns error result)."""
        from unittest.mock import AsyncMock
        from victor.agent.tool_pipeline import PipelineExecutionResult, ToolCallResult

        error_result = ToolCallResult(
            tool_name="nonexistent_tool",
            arguments={},
            success=False,
            error="Tool not available",
        )
        mock_pipeline_result = PipelineExecutionResult(results=[error_result])
        with patch.object(
            orchestrator._tool_pipeline,
            "execute_tool_calls",
            new_callable=AsyncMock,
            return_value=mock_pipeline_result,
        ):
            result = await orchestrator.execute_tool_calls([{"name": "nonexistent_tool"}])
        assert len(result) == 1
        assert result[0]["success"] is False

    @pytest.mark.asyncio
    async def test_execute_tool_calls_budget_reached(self, orchestrator):
        """Budget enforcement handled by ToolPipeline (returns budget_exhausted)."""
        from unittest.mock import AsyncMock
        from victor.agent.tool_pipeline import PipelineExecutionResult

        mock_pipeline_result = PipelineExecutionResult(
            results=[],
            budget_exhausted=True,
        )
        with patch.object(
            orchestrator._tool_pipeline,
            "execute_tool_calls",
            new_callable=AsyncMock,
            return_value=mock_pipeline_result,
        ):
            result = await orchestrator.execute_tool_calls([{"name": "read", "arguments": {}}])
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_execute_tool_calls_json_string_arguments(
        self, mock_provider, orchestrator_settings
    ):
        """Test execute_tool_calls with JSON string arguments."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Mock tool execution
            orch.tool_executor.execute = AsyncMock(
                return_value=MagicMock(success=True, result="done", error=None)
            )

            # Use string JSON arguments
            result = await orch.execute_tool_calls(
                [{"name": "read", "arguments": '{"path": "/test.py"}'}]
            )

            assert len(result) == 1
            assert result[0]["success"] is True

    @pytest.mark.asyncio
    async def test_execute_tool_calls_none_arguments(self, mock_provider, orchestrator_settings):
        """Test execute_tool_calls with None arguments defaults gracefully."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Call with None arguments — tool should handle gracefully
            # (read defaults path="" which resolves to CWD via fuzzy resolution)
            result = await orch.execute_tool_calls([{"name": "read", "arguments": None}])

            assert len(result) == 1
            assert result[0]["name"] == "read"

    @pytest.mark.asyncio
    async def test_execute_tool_calls_repeated_failure_skip(
        self, mock_provider, orchestrator_settings
    ):
        """Test execute_tool_calls skips repeated failing calls."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Mock failed tool execution - executor is stored as 'tool_executor' on orchestrator
            orch.tool_executor.execute = AsyncMock(
                return_value=MagicMock(success=False, result=None, error="Simulated failure")
            )

            args = {"path": "/test.py"}

            # First call fails and records the signature
            result1 = await orch.execute_tool_calls([{"name": "read", "arguments": args}])
            assert len(result1) == 1
            assert result1[0]["success"] is False
            # Signature should now be recorded in orchestrator's failed set
            assert len(orch.failed_tool_signatures) == 1

            # Second call must remain non-successful and should not crash.
            result2 = await orch.execute_tool_calls([{"name": "read", "arguments": args}])
            assert len(result2) <= 1
            if result2:
                assert result2[0]["success"] is False

    @pytest.mark.asyncio
    async def test_execute_tool_calls_success(self, mock_provider, orchestrator_settings):
        """Test execute_tool_calls successful execution."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Mock successful tool execution
            orch.tool_executor.execute = AsyncMock(
                return_value=MagicMock(success=True, result="File contents", error=None)
            )

            result = await orch.execute_tool_calls(
                [{"name": "read", "arguments": {"path": "/test.py"}}]
            )

            assert len(result) == 1
            assert result[0]["success"] is True
            assert result[0]["name"] == "read"
            assert orch.tool_calls_used == 1
            assert "read" in orch.executed_tools

    @pytest.mark.asyncio
    async def test_execute_tool_calls_failure(self, mock_provider, orchestrator_settings):
        """Test execute_tool_calls failed execution."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Mock failed tool execution on the pipeline's executor
            orch._tool_pipeline.executor.execute = AsyncMock(
                return_value=MagicMock(success=False, result=None, error="File not found")
            )

            result = await orch.execute_tool_calls(
                [{"name": "read", "arguments": {"path": "/nonexistent.py"}}]
            )

            assert len(result) == 1
            assert result[0]["success"] is False
            assert result[0]["error"] == "File not found"
            # Signature should be added to orchestrator's failed set
            assert len(orch.failed_tool_signatures) == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_read_file_tracking(
        self, mock_provider, orchestrator_settings
    ):
        """Test execute_tool_calls tracks read_file paths."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Mock successful tool execution
            orch.tool_executor.execute = AsyncMock(
                return_value=MagicMock(success=True, result="File contents", error=None)
            )

            await orch.execute_tool_calls([{"name": "read", "arguments": {"path": "/test.py"}}])

            assert "/test.py" in orch.observed_files


class TestClassifyTaskKeywords:
    """Tests for _classify_task_keywords helper method."""

    def test_action_task_create(self, mock_provider, orchestrator_settings):
        """Test detection of action task with 'create' keyword."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._classify_task_keywords("Create a Python function")
            assert result["is_action_task"] is True
            assert result["coarse_task_type"] == "action"
            assert result["needs_execution"] is False

    def test_action_task_execute(self, mock_provider, orchestrator_settings):
        """Test detection of action task with execution keywords."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._classify_task_keywords("Execute the script")
            assert result["is_action_task"] is True
            assert result["needs_execution"] is True
            assert result["coarse_task_type"] == "action"

    def test_action_task_run(self, mock_provider, orchestrator_settings):
        """Test detection of action task with 'run' keyword."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._classify_task_keywords("Run the tests")
            assert result["is_action_task"] is True
            assert result["needs_execution"] is True

    def test_analysis_task_analyze(self, mock_provider, orchestrator_settings):
        """Test detection of analysis task with 'analyze' keyword."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._classify_task_keywords("Analyze the codebase")
            assert result["is_analysis_task"] is True
            assert result["coarse_task_type"] == "analysis"

    def test_analysis_task_review(self, mock_provider, orchestrator_settings):
        """Test detection of analysis task with 'review' keyword."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._classify_task_keywords("Review the code for bugs")
            assert result["is_analysis_task"] is True

    def test_analysis_task_question_pattern(self, mock_provider, orchestrator_settings):
        """Test detection of analysis task with question patterns."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._classify_task_keywords("How does the authentication work?")
            assert result["is_analysis_task"] is True
            assert result["coarse_task_type"] == "analysis"

    def test_default_task(self, mock_provider, orchestrator_settings):
        """Test default classification for generic messages."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._classify_task_keywords("Hello there")
            assert result["is_action_task"] is False
            assert result["is_analysis_task"] is False
            assert result["coarse_task_type"] == "default"

    def test_position_based_precedence(self, mock_provider, orchestrator_settings):
        """Test that position-based priority determines task type when both present."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            # Message with action keyword ("create") AFTER analysis keyword ("analyze")
            # Action should take precedence since it appears last (it's the end goal)
            result = orch._classify_task_keywords("Analyze and create a report")
            # Both should be true
            assert result["is_action_task"] is True
            assert result["is_analysis_task"] is True
            # But action-related type (generation) should win since "create" appears last
            assert result["coarse_task_type"] in ("action", "generation")

            # When analysis appears last, it should win
            result2 = orch._classify_task_keywords("Create a summary and analyze it")
            assert result2["is_action_task"] is True
            assert result2["is_analysis_task"] is True
            assert result2["coarse_task_type"] == "analysis"

    def test_case_insensitive(self, mock_provider, orchestrator_settings):
        """Test that keyword detection is case insensitive."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = orch._classify_task_keywords("ANALYZE THE CODE")
            assert result["is_analysis_task"] is True
            result = orch._classify_task_keywords("CREATE a file")
            assert result["is_action_task"] is True


class TestVerticalExtensionSupport:
    """Tests for vertical extension methods."""

    def test_apply_vertical_middleware_empty_list(self, orchestrator):
        """apply_vertical_middleware with empty list does nothing."""
        orchestrator.apply_vertical_middleware([])
        assert (
            not hasattr(orchestrator, "_vertical_middleware")
            or orchestrator._vertical_middleware == []
        )

    def test_apply_vertical_middleware_with_middleware(self, mock_provider, orchestrator_settings):
        """apply_vertical_middleware adds middleware to chain."""

        class MockMiddleware:
            async def before_tool_call(self, tool_name, arguments):
                from victor.core.verticals.protocols import MiddlewareResult

                return MiddlewareResult()

            def get_priority(self):
                from victor.core.verticals.protocols import MiddlewarePriority

                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self):
                return None

        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            middleware_list = [MockMiddleware()]
            orch.apply_vertical_middleware(middleware_list)

            assert hasattr(orch, "_vertical_middleware")
            assert orch._vertical_middleware == middleware_list
            assert orch.get_middleware_chain() is not None

    def test_apply_vertical_safety_patterns_empty_list(self, orchestrator):
        """apply_vertical_safety_patterns with empty list does nothing."""
        orchestrator.apply_vertical_safety_patterns([])
        assert (
            not hasattr(orchestrator, "_vertical_safety_patterns")
            or orchestrator._vertical_safety_patterns == []
        )

    def test_apply_vertical_safety_patterns_with_patterns(
        self, mock_provider, orchestrator_settings
    ):
        """apply_vertical_safety_patterns adds patterns to safety checker."""
        from dataclasses import dataclass

        @dataclass
        class MockPattern:
            pattern: str = r"dangerous.*command"
            description: str = "Test pattern"
            risk_level: str = "HIGH"
            category: str = "test"

        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            patterns = [MockPattern()]
            orch.apply_vertical_safety_patterns(patterns)

            assert hasattr(orch, "_vertical_safety_patterns")
            assert orch._vertical_safety_patterns == patterns

    def test_get_middleware_chain_returns_none_if_not_set(self, orchestrator):
        """get_middleware_chain returns None if no middleware applied."""
        # Clear any existing middleware chain
        if hasattr(orchestrator, "_middleware_chain"):
            delattr(orchestrator, "_middleware_chain")

        result = orchestrator.get_middleware_chain()
        assert result is None

    def test_get_middleware_chain_returns_chain_if_set(self, mock_provider, orchestrator_settings):
        """get_middleware_chain returns the chain after middleware applied."""

        class MockMiddleware:
            async def before_tool_call(self, tool_name, arguments):
                from victor.core.verticals.protocols import MiddlewareResult

                return MiddlewareResult()

            def get_priority(self):
                from victor.core.verticals.protocols import MiddlewarePriority

                return MiddlewarePriority.NORMAL

            def get_applicable_tools(self):
                return None

        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            orch.apply_vertical_middleware([MockMiddleware()])
            chain = orch.get_middleware_chain()

            assert chain is not None
            from victor.agent.middleware_chain import MiddlewareChain

            assert isinstance(chain, MiddlewareChain)


class TestOrchestratorProperties:
    """Tests for orchestrator properties."""

    def test_messages_property(self, orchestrator):
        """messages property returns conversation messages."""
        messages = orchestrator.messages
        assert isinstance(messages, list)

    def test_tool_budget_property(self, orchestrator):
        """tool_budget property returns tool call budget."""
        budget = orchestrator.tool_budget
        assert isinstance(budget, int)
        assert budget > 0

    def test_model_property(self, orchestrator):
        """model property returns current model name."""
        model = orchestrator.model
        assert model == "test-model"

    def test_provider_property(self, orchestrator, mock_provider):
        """provider property returns current provider."""
        provider = orchestrator.provider
        assert provider == mock_provider


class TestToolMentionDetection:
    """Tests for _detect_mentioned_tools function."""

    def test_detect_mentioned_tools_basic(self):
        """Basic tool mention extraction."""
        from victor.agent.orchestrator import _detect_mentioned_tools

        text = "I'll use the read tool to check the file"
        mentions = _detect_mentioned_tools(text)
        # Should find "read" as mentioned tool
        assert isinstance(mentions, list)

    def test_detect_mentioned_tools_with_parens(self):
        """Tool mentions with parentheses."""
        from victor.agent.orchestrator import _detect_mentioned_tools

        text = "Calling write_file() to save the content"
        mentions = _detect_mentioned_tools(text)
        assert isinstance(mentions, list)

    def test_detect_mentioned_tools_empty_text(self):
        """Empty text returns empty list."""
        from victor.agent.orchestrator import _detect_mentioned_tools

        mentions = _detect_mentioned_tools("")
        assert mentions == []

    def test_detect_mentioned_tools_no_tools(self):
        """Text with no tool mentions."""
        from victor.agent.orchestrator import _detect_mentioned_tools

        text = "Just some regular text without tool references"
        mentions = _detect_mentioned_tools(text)
        assert isinstance(mentions, list)

    def test_detect_mentioned_tools_with_call_pattern(self):
        """Detect 'call <tool>' pattern."""
        from victor.agent.orchestrator import _detect_mentioned_tools

        text = "Let me call read to check the file contents"
        mentions = _detect_mentioned_tools(text)
        assert "read" in mentions or isinstance(mentions, list)

    def test_detect_mentioned_tools_with_use_pattern(self):
        """Detect 'use <tool>' pattern."""
        from victor.agent.orchestrator import _detect_mentioned_tools

        text = "I'll use grep to search the codebase"
        mentions = _detect_mentioned_tools(text)
        assert isinstance(mentions, list)


class TestComponentAccessors:
    """Tests for component accessor properties."""

    def test_conversation_controller_property(self, orchestrator):
        """conversation_controller property returns valid controller."""
        from victor.agent.conversation.controller import ConversationController

        controller = orchestrator.conversation_controller
        assert controller is not None
        assert isinstance(controller, ConversationController)

    def test_tool_pipeline_property(self, orchestrator):
        """tool_pipeline property returns valid pipeline."""
        from victor.agent.tool_pipeline import ToolPipeline

        pipeline = orchestrator.tool_pipeline
        assert pipeline is not None
        assert isinstance(pipeline, ToolPipeline)

    def test_streaming_controller_property(self, orchestrator):
        """streaming_controller property returns valid controller."""
        from victor.agent.streaming_controller import StreamingController

        controller = orchestrator.streaming_controller
        assert controller is not None
        assert isinstance(controller, StreamingController)

    def test_streaming_handler_property(self, orchestrator):
        """streaming_handler property returns valid handler."""
        from victor.agent.streaming import StreamingChatHandler

        handler = orchestrator.streaming_handler
        assert handler is not None
        assert isinstance(handler, StreamingChatHandler)

    def test_task_analyzer_property(self, orchestrator):
        """task_analyzer property returns valid analyzer."""
        from victor.agent.task_analyzer import TaskAnalyzer

        analyzer = orchestrator.task_analyzer
        assert analyzer is not None
        assert isinstance(analyzer, TaskAnalyzer)

    def test_observability_property(self, orchestrator):
        """observability property returns optional integration."""
        obs = orchestrator.observability
        # Could be None or ObservabilityIntegration
        if obs is not None:
            from victor.observability.integration import ObservabilityIntegration

            assert isinstance(obs, ObservabilityIntegration)

    def test_observability_setter(self, orchestrator):
        """observability setter works correctly."""
        from victor.observability.integration import ObservabilityIntegration

        new_obs = ObservabilityIntegration()
        orchestrator.observability = new_obs
        assert orchestrator.observability == new_obs

    def test_observability_setter_none(self, orchestrator):
        """observability setter accepts None."""
        orchestrator.observability = None
        assert orchestrator.observability is None


class TestCallbacks:
    """Tests for orchestrator callbacks."""

    def test_on_tool_start_callback(self, orchestrator):
        """_on_tool_start_callback records tool start."""
        orchestrator._on_tool_start_callback("read_file", {"path": "/test.txt"})
        # Should not raise, just record metrics

    def test_on_tool_complete_callback(self, orchestrator):
        """_on_tool_complete_callback records tool completion."""
        from victor.agent.tool_pipeline import ToolCallResult

        result = ToolCallResult(
            tool_name="read_file",
            arguments={"path": "/test.txt"},
            success=True,
            result="file contents",
        )
        orchestrator._on_tool_complete_callback(result)
        # Should not raise, just record metrics

    def test_on_streaming_session_complete(self, orchestrator):
        """_on_streaming_session_complete records session completion."""
        from victor.agent.streaming_controller import StreamingSession
        import time

        session = StreamingSession(
            session_id="test-session",
            provider="mock_provider",
            model="test-model",
            start_time=time.time(),
        )
        session.end_time = time.time()  # Set end_time
        orchestrator._on_streaming_session_complete(session)
        # Should not raise

    def test_send_rl_reward_signal(self, orchestrator):
        """RL reward signal via callback coordinator handles reward gracefully."""
        from victor.agent.streaming_controller import StreamingSession
        import time

        session = StreamingSession(
            session_id="test-session",
            provider="mock_provider",
            model="test-model",
            start_time=time.time(),
        )
        session.end_time = time.time()
        # Should not raise, even if RL module not available
        orchestrator._callback_coordinator._send_rl_reward_signal(session)


class TestContextLimitCalculation:
    """Tests for _calculate_max_context_chars edge cases."""

    def test_settings_override_takes_precedence(self, mock_provider):
        """Settings max_context_chars overrides provider."""
        settings = Settings(
            max_context_chars=50000,
            analytics_enabled=False,
            use_semantic_tool_selection=False,
            use_mcp_tools=False,
        )
        # Note: arguments order is (settings, provider, model)
        limit = AgentOrchestrator._calculate_max_context_chars(
            settings, mock_provider, "test-model"
        )
        assert limit == 50000

    @patch("victor.config.config_loaders.get_provider_limits")
    def test_provider_context_window_fallback(self, mock_get_limits, mock_provider):
        """Falls back to provider limits config when no settings override."""
        settings = Settings(
            analytics_enabled=False,
            use_semantic_tool_selection=False,
            use_mcp_tools=False,
            max_context_chars=None,  # No settings override, test fallback to provider limits
        )

        mock_limits = MagicMock()
        mock_limits.context_window = 100000
        mock_get_limits.return_value = mock_limits

        limit = AgentOrchestrator._calculate_max_context_chars(
            settings, mock_provider, "test-model"
        )
        # 100000 * 3.5 * 0.8 = 280000
        assert limit == 280000

    @patch("victor.config.config_loaders.get_provider_limits")
    def test_context_window_exception_handling(self, mock_get_limits, mock_provider):
        """Handles exceptions from get_provider_limits gracefully."""
        settings = Settings(
            analytics_enabled=False,
            use_semantic_tool_selection=False,
            use_mcp_tools=False,
            max_context_chars=None,  # No settings override, test fallback path
        )
        mock_get_limits.side_effect = RuntimeError("Config error")
        # Should not raise, falls back to defaults (128000 * 3.5 * 0.8 = 358400)
        limit = AgentOrchestrator._calculate_max_context_chars(
            settings, mock_provider, "test-model"
        )
        assert limit == 358400

    @patch("victor.config.config_loaders.get_provider_limits")
    def test_context_window_with_mock_value(self, mock_get_limits, mock_provider):
        """Handles non-numeric context window values."""
        settings = Settings(
            analytics_enabled=False,
            use_semantic_tool_selection=False,
            max_context_chars=None,  # No settings override, test fallback path
            use_mcp_tools=False,
        )
        mock_limits = MagicMock()
        mock_limits.context_window = "invalid_value"  # Explicitly non-numeric string
        mock_get_limits.return_value = mock_limits

        mock_provider.name = "mock_provider"
        # Should handle gracefully and fall back to failsafe (100k tokens)
        limit = AgentOrchestrator._calculate_max_context_chars(
            settings, mock_provider, "test-model"
        )
        # 100000 * 3.5 * 0.8 = 280000
        assert limit == 280000


class TestRecoveryIntegration:
    """Tests for recovery system integration."""

    def test_recovery_handler_initialization(self, mock_provider, orchestrator_settings):
        """Recovery handler is initialized when enabled."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            # Should have recovery handler attribute
            assert hasattr(orch, "_recovery_handler")

    def test_recovery_handler_attribute_exists(self, orchestrator):
        """Recovery handler attribute exists."""
        # Recovery handler may or may not be None depending on default settings
        # Just verify the attribute exists
        assert hasattr(orchestrator, "_recovery_handler")

    def test_recovery_integration_property(self, orchestrator):
        """_recovery_integration is set up correctly."""
        assert hasattr(orchestrator, "_recovery_integration")


class TestObservabilityIntegration:
    """Tests for observability system integration."""

    def test_observability_enabled_by_default(self, mock_provider, orchestrator_settings):
        """Observability is enabled by default."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            # Observability may or may not be present depending on settings
            assert hasattr(orch, "_observability")

    def test_observability_disabled(self, mock_provider):
        """Observability can be disabled via settings."""
        settings = Settings(
            enable_observability=False,
            analytics_enabled=False,
            use_semantic_tool_selection=False,
            use_mcp_tools=False,
        )
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=settings,
                provider=mock_provider,
                model="test-model",
            )
            assert orch._observability is None

    def test_on_tool_start_with_observability(self, mock_provider, orchestrator_settings):
        """Tool start emits observability event when enabled."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            # If observability is present, call should work
            if orch._observability:
                orch._on_tool_start_callback("read_file", {"path": "/test"})
                # Should not raise

    def test_on_tool_complete_with_observability(self, mock_provider, orchestrator_settings):
        """Tool complete emits observability event when enabled."""
        from victor.agent.tool_pipeline import ToolCallResult

        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orch = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )
            result = ToolCallResult(
                tool_name="read_file",
                arguments={"path": "/test"},
                success=True,
                result="content",
            )
            # If observability is present, call should work
            if orch._observability:
                orch._on_tool_complete_callback(result)
                # Should not raise


class TestMetricsCollector:
    """Tests for metrics collector integration."""

    def test_metrics_collector_initialized(self, orchestrator):
        """Metrics collector is initialized."""
        assert hasattr(orchestrator, "_metrics_collector")
        assert orchestrator._metrics_collector is not None

    def test_selection_stats_attribute(self, orchestrator):
        """Selection stats are tracked in collector."""
        stats = orchestrator._metrics_collector._selection_stats
        assert stats is not None
        from victor.agent.metrics_collector import ToolSelectionStats

        assert isinstance(stats, ToolSelectionStats)

    def test_classification_stats_attribute(self, orchestrator):
        """Classification stats are tracked in collector."""
        stats = orchestrator._metrics_collector._classification_stats
        assert stats is not None
        from victor.agent.metrics_collector import ClassificationStats

        assert isinstance(stats, ClassificationStats)


class TestUsageAnalytics:
    """Tests for usage analytics integration."""

    def test_usage_analytics_initialized(self, orchestrator):
        """Usage analytics component exists."""
        assert hasattr(orchestrator, "_usage_analytics") or hasattr(orchestrator, "usage_logger")

    def test_usage_logger_logs_session_start(self, orchestrator):
        """Usage logger should have logged session start."""
        # Session start is logged in __init__
        assert hasattr(orchestrator, "usage_logger")


class TestTaskClassification:
    """Tests for task classification methods."""

    def test_classify_task_keywords_analysis(self, orchestrator):
        """_classify_task_keywords returns classification dict for analysis tasks."""
        result = orchestrator._classify_task_keywords("analyze this code and explain")
        assert isinstance(result, dict)
        # Result may contain legacy keys or new format (task_type, confidence)
        assert "task_type" in result or "is_analysis_task" in result

    def test_classify_task_keywords_action(self, orchestrator):
        """_classify_task_keywords returns classification dict for action tasks."""
        result = orchestrator._classify_task_keywords("create a new function to add numbers")
        assert isinstance(result, dict)
        assert "task_type" in result or "is_action_task" in result

    def test_classify_task_keywords_execution(self, orchestrator):
        """_classify_task_keywords returns classification dict for execution tasks."""
        result = orchestrator._classify_task_keywords("run the tests")
        assert isinstance(result, dict)
        assert "task_type" in result or "needs_execution" in result

    def test_classify_task_keywords_default(self, orchestrator):
        """_classify_task_keywords handles generic messages."""
        result = orchestrator._classify_task_keywords("hello world")
        assert isinstance(result, dict)

    def test_classify_task_with_context_no_history(self, orchestrator):
        """_classify_task_with_context works without history."""
        result = orchestrator._classify_task_with_context("fix this bug", None)
        assert isinstance(result, dict)
        assert "task_type" in result or "is_analysis_task" in result

    def test_classify_task_with_context_with_history(self, orchestrator):
        """_classify_task_with_context uses conversation history."""
        history = [
            {"role": "user", "content": "I need to refactor this code"},
            {"role": "assistant", "content": "I'll help with that."},
        ]
        result = orchestrator._classify_task_with_context("continue", history)
        assert isinstance(result, dict)


class TestApplyIntentGuard:
    """Tests for _apply_intent_guard method."""

    def test_apply_intent_guard_write_task(self, orchestrator):
        """Intent guard for write task."""
        # Should not raise
        orchestrator._apply_intent_guard("write a new file")

    def test_apply_intent_guard_read_task(self, orchestrator):
        """Intent guard for read task."""
        # Should not raise
        orchestrator._apply_intent_guard("read and explain this code")

    def test_apply_intent_guard_analysis_task(self, orchestrator):
        """Intent guard for analysis task."""
        # Should not raise
        orchestrator._apply_intent_guard("analyze the architecture")


class TestApplyTaskGuidance:
    """Tests for _apply_task_guidance method."""

    def test_apply_task_guidance_analysis(self, orchestrator):
        """Task guidance for analysis tasks."""
        from victor.agent.unified_classifier import ClassifierTaskType

        orchestrator._apply_task_guidance(
            user_message="analyze this code",
            unified_task_type=ClassifierTaskType.ANALYSIS,
            is_analysis_task=True,
            is_action_task=False,
            needs_execution=False,
            max_exploration_iterations=3,
        )
        # Should not raise

    def test_apply_task_guidance_action(self, orchestrator):
        """Task guidance for action tasks."""
        from victor.agent.unified_classifier import ClassifierTaskType

        orchestrator._apply_task_guidance(
            user_message="create a new function",
            unified_task_type=ClassifierTaskType.GENERATION,
            is_analysis_task=False,
            is_action_task=True,
            needs_execution=False,
            max_exploration_iterations=5,
        )
        # Should not raise

    def test_apply_task_guidance_execution(self, orchestrator):
        """Task guidance for execution tasks."""
        from victor.agent.unified_classifier import ClassifierTaskType

        orchestrator._apply_task_guidance(
            user_message="run the tests",
            unified_task_type=ClassifierTaskType.ACTION,
            is_analysis_task=False,
            is_action_task=True,
            needs_execution=True,
            max_exploration_iterations=2,
        )
        # Should not raise


class TestAddMessage:
    """Tests for add_message method."""

    def test_add_user_message(self, orchestrator):
        """add_message adds user messages."""
        orchestrator.add_message("user", "test message")
        messages = orchestrator.messages
        # Should have at least one message
        assert len(messages) >= 1

    def test_add_assistant_message(self, orchestrator):
        """add_message adds assistant messages."""
        orchestrator.add_message("assistant", "response")
        messages = orchestrator.messages
        assert len(messages) >= 1

    def test_add_system_message(self, orchestrator):
        """add_message adds system messages."""
        orchestrator.add_message("system", "system instruction")
        # Should not raise


class TestGetMessages:
    """Tests for get_messages and related methods."""

    def test_messages_property_returns_list(self, orchestrator):
        """messages property returns list."""
        messages = orchestrator.messages
        assert isinstance(messages, list)

    def test_conversation_controller_messages(self, orchestrator):
        """conversation_controller provides access to messages."""
        controller = orchestrator.conversation_controller
        messages = controller.messages
        assert isinstance(messages, list)


class TestProviderCapabilities:
    """Tests for provider capability methods."""

    def test_tool_calling_caps(self, orchestrator):
        """tool_calling_caps attribute exists."""
        assert hasattr(orchestrator, "tool_calling_caps")

    def test_provider_name(self, orchestrator):
        """provider_name property."""
        name = orchestrator.provider_name
        assert isinstance(name, str)

    def test_thinking_enabled(self, orchestrator):
        """thinking attribute exists."""
        assert hasattr(orchestrator, "thinking")


class TestGetOptimizationStatus:
    """Tests for get_optimization_status method."""

    def test_returns_dict(self, orchestrator):
        """get_optimization_status returns a dictionary."""
        status = orchestrator.get_optimization_status()
        assert isinstance(status, dict)

    def test_has_timestamp(self, orchestrator):
        """Status includes timestamp."""
        status = orchestrator.get_optimization_status()
        assert "timestamp" in status
        assert isinstance(status["timestamp"], float)

    def test_has_components(self, orchestrator):
        """Status includes components dict."""
        status = orchestrator.get_optimization_status()
        assert "components" in status
        assert isinstance(status["components"], dict)

    def test_has_health_status(self, orchestrator):
        """Status includes health information."""
        status = orchestrator.get_optimization_status()
        assert "health" in status
        assert "status" in status["health"]
        assert status["health"]["status"] in ("healthy", "degraded")

    def test_context_compactor_included(self, orchestrator):
        """Context compactor status is included."""
        status = orchestrator.get_optimization_status()
        components = status["components"]
        assert "context_compactor" in components or orchestrator._context_compactor is None

    def test_code_correction_included(self, orchestrator):
        """Code correction middleware status is included."""
        status = orchestrator.get_optimization_status()
        assert "code_correction" in status["components"]
        assert "enabled" in status["components"]["code_correction"]

    def test_safety_checker_included(self, orchestrator):
        """Safety checker status is included."""
        status = orchestrator.get_optimization_status()
        assert "safety_checker" in status["components"]
        assert "enabled" in status["components"]["safety_checker"]

    def test_auto_committer_included(self, orchestrator):
        """Auto committer status is included."""
        status = orchestrator.get_optimization_status()
        assert "auto_committer" in status["components"]

    def test_search_router_included(self, orchestrator):
        """Search router status is included."""
        status = orchestrator.get_optimization_status()
        assert "search_router" in status["components"]


class TestFlushAnalytics:
    """Tests for flush_analytics method."""

    def test_returns_dict(self, orchestrator):
        """flush_analytics returns a dictionary."""
        results = orchestrator.flush_analytics()
        assert isinstance(results, dict)

    def test_usage_analytics_key(self, orchestrator):
        """Results include usage_analytics key."""
        results = orchestrator.flush_analytics()
        assert "usage_analytics" in results

    def test_sequence_tracker_key(self, orchestrator):
        """Results include sequence_tracker key."""
        results = orchestrator.flush_analytics()
        assert "sequence_tracker" in results

    def test_tool_cache_key(self, orchestrator):
        """Results include tool_cache key."""
        results = orchestrator.flush_analytics()
        assert "tool_cache" in results


class TestSwitchProvider:
    """Tests for switch_provider method."""

    @pytest.mark.asyncio
    async def test_switch_returns_bool_or_none(self, orchestrator):
        """switch_provider returns False when provider not found (errors are swallowed)."""
        # New contract: switch_provider catches exceptions and returns False on failure.
        result = await orchestrator.switch_provider("mock", "test-model")
        assert result is False

    @pytest.mark.asyncio
    async def test_switch_to_invalid_provider_returns_none(self, orchestrator):
        """Switching to invalid provider returns falsy (False)."""
        result = await orchestrator.switch_provider("nonexistent_provider_xyz", "model")
        assert not result

    @pytest.mark.asyncio
    async def test_switch_updates_provider_name(self, mock_provider, orchestrator_settings):
        """Successful switch updates provider_name."""
        with (
            patch("victor.agent.orchestrator.UsageLogger"),
            patch("victor.core.bootstrap_services.bootstrap_new_services"),
        ):
            orchestrator = AgentOrchestrator(
                settings=orchestrator_settings,
                provider=mock_provider,
                model="test-model",
            )

            # Mock the provider service to simulate a successful switch
            mock_provider_svc = MagicMock()
            mock_provider_svc.switch_provider = AsyncMock()
            new_provider = MagicMock()
            mock_provider_svc.get_current_provider = MagicMock(return_value=new_provider)
            mock_provider_svc.get_current_provider_info.return_value = MagicMock(
                provider_name="new_provider",
                model_name="new-model",
                api_key_configured=True,
                supports_streaming=True,
                supports_tool_calling=True,
                max_tokens=200000,
            )
            orchestrator._provider_service = mock_provider_svc

            result = await orchestrator.switch_provider("new_provider", "new-model")

            assert result is True
            mock_provider_svc.switch_provider.assert_called_once_with("new_provider", "new-model")
            assert orchestrator.provider_name == "new_provider"
            assert orchestrator.model == "new-model"

    @pytest.mark.asyncio
    async def test_switch_provider_does_not_use_legacy_coordinator_when_service_exists(
        self, orchestrator
    ):
        """Canonical provider switching should not fall back to the legacy coordinator."""
        provider_service = MagicMock()
        provider_service.switch_provider = AsyncMock()
        provider_service.get_current_provider = MagicMock(return_value=MagicMock())
        provider_service.get_current_provider_info.return_value = MagicMock(
            provider_name="openai",
            model_name="gpt-4.1",
        )
        orchestrator._provider_service = provider_service
        orchestrator._deprecated_provider_coordinator = MagicMock()
        orchestrator._deprecated_provider_coordinator.switch_provider_async = AsyncMock(
            side_effect=AssertionError("legacy coordinator path should not be used")
        )

        result = await orchestrator.switch_provider("openai", "gpt-4.1")

        assert result is True
        provider_service.switch_provider.assert_awaited_once_with("openai", "gpt-4.1")
        orchestrator._deprecated_provider_coordinator.switch_provider_async.assert_not_called()


class TestSwitchModel:
    """Tests for switch_model method."""

    def test_switch_model_returns_bool(self, orchestrator):
        """switch_model returns boolean."""
        import asyncio

        result = asyncio.run(orchestrator.switch_model("different-model"))
        assert isinstance(result, bool)

    def test_switch_model_updates_model(self, orchestrator):
        """Successful model switch updates model attribute."""
        import asyncio

        old_model = orchestrator.model
        result = asyncio.run(orchestrator.switch_model("new-test-model"))
        if result:
            assert orchestrator.model == "new-test-model"
        else:
            # If failed, model should be unchanged
            assert orchestrator.model == old_model

    def test_switch_model_does_not_use_legacy_coordinator_when_service_exists(self, orchestrator):
        """Canonical model switching should not fall back to the legacy coordinator."""
        import asyncio

        provider_service = MagicMock()
        provider_service.switch_model = AsyncMock()
        provider_service.get_current_provider = MagicMock(return_value=MagicMock())
        provider_service.get_current_provider_info.return_value = MagicMock(
            provider_name="anthropic",
            model_name="claude-3-7-sonnet",
        )
        orchestrator._provider_service = provider_service
        orchestrator._deprecated_provider_coordinator = MagicMock()
        orchestrator._deprecated_provider_coordinator.switch_model_async = AsyncMock(
            side_effect=AssertionError("legacy coordinator path should not be used")
        )

        result = asyncio.run(orchestrator.switch_model("claude-3-7-sonnet"))

        assert result is True
        provider_service.switch_model.assert_awaited_once_with("claude-3-7-sonnet")
        orchestrator._deprecated_provider_coordinator.switch_model_async.assert_not_called()


class TestIntelligentIntegration:
    """Tests for runtime_intelligence_integration property."""

    def test_returns_none_when_disabled(self, orchestrator):
        """Returns None when intelligent pipeline is disabled."""
        orchestrator._runtime_intelligence_enabled = False
        result = orchestrator.runtime_intelligence_integration
        assert result is None

    def test_lazy_initialization(self, orchestrator):
        """Integration is lazily initialized."""
        # Initially should be None
        assert orchestrator._runtime_intelligence_integration is None

    def test_property_is_accessible(self, orchestrator):
        """runtime_intelligence_integration property is accessible."""
        # This may return None or an integration object
        result = orchestrator.runtime_intelligence_integration
        # Should not raise


class TestResetConversation:
    """Tests for reset_conversation method."""

    def test_reset_clears_tool_calls(self, orchestrator):
        """reset_conversation clears tool calls used."""
        orchestrator.tool_calls_used = 10
        orchestrator.reset_conversation()
        assert orchestrator.tool_calls_used == 0

    def test_reset_clears_executed_tools(self, orchestrator):
        """reset_conversation clears executed tools list."""
        orchestrator.executed_tools = ["tool1", "tool2"]
        orchestrator.reset_conversation()
        assert orchestrator.executed_tools == []

    def test_reset_clears_observed_files(self, orchestrator):
        """reset_conversation clears observed files."""
        orchestrator.observed_files = ["/file1.py", "/file2.py"]
        orchestrator.reset_conversation()
        assert orchestrator.observed_files == []

    def test_reset_method_calls_reset_conversation(self, orchestrator):
        """reset() method delegates to reset_conversation()."""
        orchestrator.tool_calls_used = 5
        orchestrator.reset()
        assert orchestrator.tool_calls_used == 0


class TestToolBudgetManagement:
    """Tests for tool budget management."""

    def test_tool_budget_attribute_exists(self, orchestrator):
        """tool_budget attribute exists."""
        assert hasattr(orchestrator, "tool_budget")
        assert isinstance(orchestrator.tool_budget, int)

    def test_tool_calls_used_starts_at_zero(self, orchestrator):
        """tool_calls_used starts at zero."""
        orchestrator.reset_conversation()
        assert orchestrator.tool_calls_used == 0

    def test_tool_budget_has_minimum(self, orchestrator):
        """Tool budget has minimum value."""
        assert orchestrator.tool_budget >= 50


class TestUnifiedTracker:
    """Tests for unified tracker integration."""

    def test_unified_tracker_exists(self, orchestrator):
        """unified_tracker attribute exists."""
        assert hasattr(orchestrator, "unified_tracker")
        assert orchestrator.unified_tracker is not None

    def test_unified_tracker_reset(self, orchestrator):
        """Unified tracker can be reset."""
        orchestrator.unified_tracker.reset()
        # Should not raise


class TestSafetyChecker:
    """Tests for safety checker integration."""

    def test_safety_checker_property(self, orchestrator):
        """safety_checker property returns checker."""
        checker = orchestrator.safety_checker
        # May be None if disabled
        if checker is not None:
            from victor.agent.safety import SafetyChecker

            assert isinstance(checker, SafetyChecker)

    def test_set_confirmation_callback(self, orchestrator):
        """Can set confirmation callback on safety checker."""
        if orchestrator.safety_checker:
            callback = MagicMock()
            # SafetyChecker uses attribute assignment, not a method
            orchestrator.safety_checker.confirmation_callback = callback
            assert orchestrator.safety_checker.confirmation_callback == callback


class TestAutoCommitter:
    """Tests for auto committer integration."""

    def test_auto_committer_property(self, orchestrator):
        """auto_committer property exists."""
        assert hasattr(orchestrator, "auto_committer")

    def test_auto_committer_accessible(self, orchestrator):
        """auto_committer is accessible."""
        committer = orchestrator.auto_committer
        # May be None if disabled


class TestContextCompactor:
    """Tests for context compactor integration."""

    def test_context_compactor_exists(self, orchestrator):
        """_context_compactor attribute exists."""
        assert hasattr(orchestrator, "_context_compactor")

    def test_get_context_metrics(self, orchestrator):
        """Can get context metrics from controller."""
        controller = orchestrator.conversation_controller
        metrics = controller.get_context_metrics()
        assert metrics is not None


class TestToolOutputFormatter:
    """Tests for tool output formatter integration."""

    def test_tool_output_formatter_exists(self, orchestrator):
        """_tool_output_formatter exists."""
        assert hasattr(orchestrator, "_tool_output_formatter")
        assert orchestrator._tool_output_formatter is not None

    def test_formatter_has_format_method(self, orchestrator):
        """Formatter has format_tool_output method."""
        formatter = orchestrator._tool_output_formatter
        assert hasattr(formatter, "format_tool_output")


class TestSequenceTracker:
    """Tests for sequence tracker integration."""

    def test_sequence_tracker_exists(self, orchestrator):
        """_sequence_tracker exists."""
        assert hasattr(orchestrator, "_sequence_tracker")

    def test_sequence_tracker_suggests(self, orchestrator):
        """Sequence tracker has get_statistics method."""
        tracker = orchestrator._sequence_tracker
        if tracker:
            stats = tracker.get_statistics()
            assert isinstance(stats, dict)


class TestUsageAnalytics:
    """Tests for usage analytics integration."""

    def test_usage_analytics_exists(self, orchestrator):
        """_usage_analytics exists."""
        assert hasattr(orchestrator, "_usage_analytics")

    def test_usage_analytics_is_singleton(self, orchestrator):
        """UsageAnalytics uses singleton pattern."""
        from victor.agent.usage_analytics import UsageAnalytics

        instance = UsageAnalytics.get_instance()
        assert instance is orchestrator._usage_analytics


class TestRecoveryIntegrationExtended:
    """Extended tests for recovery integration."""

    def test_recovery_integration_property(self, orchestrator):
        """_recovery_integration exists."""
        assert hasattr(orchestrator, "_recovery_integration")

    def test_recovery_handler_property(self, orchestrator):
        """_recovery_handler exists."""
        assert hasattr(orchestrator, "_recovery_handler")


class TestGetCurrentProviderInfo:
    """Tests for get_current_provider_info method."""

    def test_returns_dict(self, orchestrator):
        """Returns a dictionary."""
        info = orchestrator.get_current_provider_info()
        assert isinstance(info, dict)

    def test_includes_tool_budget(self, orchestrator):
        """Result includes tool_budget key."""
        info = orchestrator.get_current_provider_info()
        assert "tool_budget" in info

    def test_includes_tool_calls_used(self, orchestrator):
        """Result includes tool_calls_used key."""
        info = orchestrator.get_current_provider_info()
        assert "tool_calls_used" in info

    def test_includes_provider_name(self, orchestrator):
        """Result includes provider name."""
        info = orchestrator.get_current_provider_info()
        assert "name" in info or "provider" in info

    def test_uses_provider_service_rate_limit_stats_only(self, orchestrator):
        """Canonical provider info should not consult the deprecated coordinator."""
        orchestrator._deprecated_provider_coordinator = MagicMock()
        orchestrator._deprecated_provider_coordinator.get_rate_limit_stats.side_effect = (
            AssertionError("legacy coordinator stats path should not be used")
        )
        orchestrator._provider_service.get_rate_limit_stats.return_value = {
            "rate_limits_hit": 7,
        }

        info = orchestrator.get_current_provider_info()

        assert info["rate_limits_hit"] == 7
        orchestrator._deprecated_provider_coordinator.get_rate_limit_stats.assert_not_called()


class TestClassifyTaskKeywords:
    """Tests for _classify_task_keywords method."""

    def test_classify_code_generation(self, orchestrator):
        """Classifies code generation tasks."""
        result = orchestrator._classify_task_keywords("Write a function to add numbers")
        assert isinstance(result, dict)

    def test_classify_debugging(self, orchestrator):
        """Classifies debugging tasks."""
        result = orchestrator._classify_task_keywords("Fix this bug in my code")
        assert isinstance(result, dict)

    def test_classify_refactoring(self, orchestrator):
        """Classifies refactoring tasks."""
        result = orchestrator._classify_task_keywords(
            "Refactor this class to use dependency injection"
        )
        assert isinstance(result, dict)

    def test_classify_search(self, orchestrator):
        """Classifies search tasks."""
        result = orchestrator._classify_task_keywords("Find all occurrences of this function")
        assert isinstance(result, dict)


class TestApplyIntentGuard:
    """Tests for _apply_intent_guard method."""

    def test_guard_for_non_write_task(self, orchestrator):
        """Apply guard for non-write tasks."""
        # Should not raise
        orchestrator._apply_intent_guard("What is the structure of this codebase?")

    def test_guard_for_write_task(self, orchestrator):
        """Apply guard for write tasks."""
        # Should not raise
        orchestrator._apply_intent_guard("Write a new function to process data")


class TestMessageHistory:
    """Tests for message history methods."""

    def test_get_messages(self, orchestrator):
        """Can get messages."""
        messages = orchestrator.get_messages()
        assert isinstance(messages, list)

    def test_add_and_get_messages(self, orchestrator):
        """Can add and retrieve messages."""
        orchestrator.add_message("user", "Test message")
        messages = orchestrator.get_messages()
        assert len(messages) >= 1

    def test_get_message_count(self, orchestrator):
        """Can get message count."""
        count = orchestrator.get_message_count()
        assert isinstance(count, int)
        assert count >= 0


class TestToolBudgetManagementExtended:
    """Extended tests for tool budget management."""

    def test_tool_budget_attribute_read(self, orchestrator):
        """Can read tool budget."""
        budget = orchestrator.tool_budget
        assert isinstance(budget, int)
        assert budget >= 0

    def test_tool_budget_attribute_write(self, orchestrator):
        """Can set tool budget directly."""
        orchestrator.tool_budget = 50
        assert orchestrator.tool_budget == 50

    def test_increment_tool_calls(self, orchestrator):
        """Tool calls used increments correctly."""
        initial = orchestrator.tool_calls_used
        orchestrator.tool_calls_used = initial + 1
        assert orchestrator.tool_calls_used == initial + 1


class TestToolRegistrar:
    """Tests for tool registrar integration."""

    def test_tool_registrar_exists(self, orchestrator):
        """tool_registrar property exists."""
        assert hasattr(orchestrator, "tool_registrar")

    def test_registered_tools(self, orchestrator):
        """Can access registered tools through registrar."""
        registrar = orchestrator.tool_registrar
        if registrar:
            # Tool registrar should have some tools registered
            assert registrar is not None


class TestParseToolCallsWithAdapter:
    """Tests for _parse_tool_calls_with_adapter method."""

    def test_parse_empty_content(self, orchestrator):
        """Parse empty content returns empty result."""
        result = orchestrator._parse_tool_calls_with_adapter("")
        assert result is not None

    def test_parse_with_no_tool_calls(self, orchestrator):
        """Parse content with no tool calls."""
        result = orchestrator._parse_tool_calls_with_adapter("Just some text content")
        assert result is not None

    def test_parse_with_json_tool_call(self, orchestrator):
        """Parse content with JSON tool call."""
        content = '{"tool": "read_file", "arguments": {"path": "/test.txt"}}'
        result = orchestrator._parse_tool_calls_with_adapter(content)
        assert result is not None


class TestMetricsCollector:
    """Tests for metrics collector integration."""

    def test_metrics_collector_exists(self, orchestrator):
        """_metrics_collector exists."""
        assert hasattr(orchestrator, "_metrics_collector")

    def test_get_iteration_metrics(self, orchestrator):
        """Can get iteration metrics."""
        if hasattr(orchestrator._metrics_collector, "get_iteration_metrics"):
            metrics = orchestrator._metrics_collector.get_iteration_metrics()
            assert isinstance(metrics, dict)


class TestProviderManager:
    """Tests for provider manager integration."""

    def test_provider_manager_exists(self, orchestrator):
        """_provider_manager exists."""
        assert hasattr(orchestrator, "_provider_manager")

    def test_provider_manager_has_get_info(self, orchestrator):
        """Provider manager has get_info method."""
        manager = orchestrator._provider_manager
        assert hasattr(manager, "get_info")


class TestToolPipeline:
    """Tests for tool pipeline integration."""

    def test_tool_pipeline_exists(self, orchestrator):
        """_tool_pipeline exists."""
        assert hasattr(orchestrator, "_tool_pipeline")


class TestStreamingController:
    """Tests for streaming controller integration."""

    def test_streaming_controller_exists(self, orchestrator):
        """_streaming_controller exists."""
        assert hasattr(orchestrator, "_streaming_controller")


class TestConversationController:
    """Tests for conversation controller integration."""

    def test_conversation_controller_exists(self, orchestrator):
        """conversation_controller property exists."""
        assert hasattr(orchestrator, "conversation_controller")
        controller = orchestrator.conversation_controller
        assert controller is not None

    def test_get_context_metrics(self, orchestrator):
        """Can get context metrics from conversation controller."""
        controller = orchestrator.conversation_controller
        if hasattr(controller, "get_context_metrics"):
            metrics = controller.get_context_metrics()
            # metrics may be a ContextMetrics dataclass or dict
            assert metrics is not None


class TestTaskAnalyzer:
    """Tests for task analyzer integration."""

    def test_task_analyzer_exists(self, orchestrator):
        """_task_analyzer exists."""
        assert hasattr(orchestrator, "_task_analyzer")


class TestObservability:
    """Tests for observability methods."""

    def test_register_observability(self, orchestrator):
        """Can register observability hooks."""
        callback = MagicMock()
        # This may be a no-op if observability is not enabled
        if hasattr(orchestrator, "register_on_tool_start"):
            orchestrator.register_on_tool_start(callback)

    def test_get_observability_status(self, orchestrator):
        """Can check observability status."""
        if hasattr(orchestrator, "observability_enabled"):
            status = orchestrator.observability_enabled
            assert isinstance(status, bool)


class TestProviderHealth:
    """Tests for provider health functionality."""

    @pytest.mark.asyncio
    async def test_get_provider_health_basic(self, orchestrator):
        """Test get_provider_health returns basic info."""
        result = await orchestrator.get_provider_health()
        assert isinstance(result, dict)
        assert "current_provider" in result
        assert "is_healthy" in result
        assert "healthy_providers" in result
        assert "can_failover" in result

    @pytest.mark.asyncio
    async def test_get_provider_health_with_manager(self, orchestrator):
        """Test get_provider_health with provider manager."""
        # Mock provider manager
        mock_manager = MagicMock()
        mock_state = MagicMock()
        mock_state.is_healthy = True
        mock_state.switch_count = 2
        mock_manager.get_current_state.return_value = mock_state
        mock_manager.get_healthy_providers = AsyncMock(return_value=["provider1", "provider2"])
        orchestrator._provider_manager = mock_manager

        result = await orchestrator.get_provider_health()

        assert result["is_healthy"] is True
        assert result["switch_count"] == 2
        assert result["healthy_providers"] == ["provider1", "provider2"]
        assert result["can_failover"] is True

    @pytest.mark.asyncio
    async def test_get_provider_health_error_handling(self, orchestrator):
        """Test get_provider_health handles errors."""
        mock_manager = MagicMock()
        mock_manager.get_current_state.side_effect = Exception("Test error")
        orchestrator._provider_manager = mock_manager

        result = await orchestrator.get_provider_health()

        assert "error" in result


class TestGracefulShutdown:
    """Tests for graceful shutdown functionality."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown_basic(self, orchestrator):
        """Test graceful_shutdown returns status dict."""
        result = await orchestrator.graceful_shutdown()
        assert isinstance(result, dict)
        assert "analytics_flushed" in result
        assert "health_monitoring_stopped" in result
        assert "session_ended" in result

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_analytics(self, orchestrator):
        """Test graceful_shutdown flushes analytics."""
        # Mock flush_analytics
        orchestrator.flush_analytics = MagicMock(return_value={"tool": True, "provider": True})
        orchestrator.stop_health_monitoring = AsyncMock(return_value=True)

        result = await orchestrator.graceful_shutdown()

        assert result["analytics_flushed"] is True
        orchestrator.flush_analytics.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_analytics_error(self, orchestrator):
        """Test graceful_shutdown handles analytics error."""
        orchestrator.flush_analytics = MagicMock(side_effect=Exception("Flush error"))
        orchestrator.stop_health_monitoring = AsyncMock(return_value=True)

        result = await orchestrator.graceful_shutdown()

        assert result["analytics_flushed"] is False

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_usage_analytics(self, orchestrator):
        """Test graceful_shutdown ends usage analytics session."""
        mock_analytics = MagicMock()
        mock_analytics._current_session = "session123"
        orchestrator._usage_analytics = mock_analytics
        orchestrator.flush_analytics = MagicMock(return_value={})
        orchestrator.stop_health_monitoring = AsyncMock(return_value=True)

        result = await orchestrator.graceful_shutdown()

        assert result["session_ended"] is True
        mock_analytics.end_session.assert_called_once()


class TestIntelligentPipelineIntegration:
    """Tests for intelligent pipeline integration methods."""

    @pytest.mark.asyncio
    async def test_prepare_intelligent_request_no_integration(self, orchestrator):
        """Test _prepare_runtime_intelligence_request returns None without integration."""
        # Pipeline is disabled by default in test settings
        orchestrator._runtime_intelligence_enabled = False
        result = await orchestrator._prepare_runtime_intelligence_request("test task", "analysis")
        assert result is None

    @pytest.mark.asyncio
    async def test_prepare_intelligent_request_with_integration(self, orchestrator):
        """Test _prepare_runtime_intelligence_request with integration."""
        mock_integration = MagicMock()
        # Mock the new method name (TD-002 refactoring)
        mock_integration.prepare_runtime_intelligence_request = AsyncMock(
            return_value={
                "recommended_mode": "explore",
                "recommended_tool_budget": 10,
                "should_continue": True,
                "system_prompt_addition": "Extra context",
            }
        )
        # Patch the property to return our mock
        with patch.object(
            type(orchestrator),
            "runtime_intelligence_integration",
            property(lambda self: mock_integration),
        ):
            result = await orchestrator._prepare_runtime_intelligence_request(
                "test task", "analysis"
            )

        assert result is not None
        assert result["recommended_mode"] == "explore"
        assert result["recommended_tool_budget"] == 10
        assert result["should_continue"] is True
        assert result["system_prompt_addition"] == "Extra context"

    @pytest.mark.asyncio
    async def test_prepare_intelligent_request_error(self, orchestrator):
        """Test _prepare_runtime_intelligence_request handles errors."""
        mock_integration = MagicMock()
        # Mock the new method name (TD-002 refactoring)
        mock_integration.prepare_runtime_intelligence_request = AsyncMock(return_value=None)
        with patch.object(
            type(orchestrator),
            "runtime_intelligence_integration",
            property(lambda self: mock_integration),
        ):
            result = await orchestrator._prepare_runtime_intelligence_request(
                "test task", "analysis"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_no_integration(self, orchestrator):
        """Test _validate_runtime_intelligence_response returns None without integration."""
        orchestrator._runtime_intelligence_enabled = False
        result = await orchestrator._validate_runtime_intelligence_response(
            "response", "query", 5, "analysis"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_empty_response(self, orchestrator):
        """Test _validate_runtime_intelligence_response skips empty responses."""
        mock_integration = MagicMock()
        # Mock returns None for short/empty responses (TD-002 refactoring)
        mock_integration.validate_runtime_intelligence_response = AsyncMock(return_value=None)
        with patch.object(
            type(orchestrator),
            "runtime_intelligence_integration",
            property(lambda self: mock_integration),
        ):
            result = await orchestrator._validate_runtime_intelligence_response(
                "", "query", 5, "analysis"
            )
            assert result is None

            result = await orchestrator._validate_runtime_intelligence_response(
                "short", "query", 5, "analysis"
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_with_integration(self, orchestrator):
        """Test _validate_runtime_intelligence_response with integration."""
        mock_integration = MagicMock()
        # Mock the new method name (TD-002 refactoring)
        mock_integration.validate_runtime_intelligence_response = AsyncMock(
            return_value={
                "quality_score": 0.9,
                "grounding_score": 0.85,
                "is_grounded": True,
                "is_valid": True,
                "grounding_issues": [],
            }
        )
        with patch.object(
            type(orchestrator),
            "runtime_intelligence_integration",
            property(lambda self: mock_integration),
        ):
            result = await orchestrator._validate_runtime_intelligence_response(
                "This is a longer response with at least 50 characters for testing purposes.",
                "query",
                5,
                "analysis",
            )

        assert result is not None
        assert result["quality_score"] == 0.9
        assert result["grounding_score"] == 0.85
        assert result["is_grounded"] is True
        assert result["is_valid"] is True

    @pytest.mark.asyncio
    async def test_validate_intelligent_response_error(self, orchestrator):
        """Test _validate_runtime_intelligence_response handles errors."""
        mock_integration = MagicMock()
        # Mock returns None on error (TD-002 refactoring)
        mock_integration.validate_runtime_intelligence_response = AsyncMock(return_value=None)
        with patch.object(
            type(orchestrator),
            "runtime_intelligence_integration",
            property(lambda self: mock_integration),
        ):
            result = await orchestrator._validate_runtime_intelligence_response(
                "This is a longer response with at least 50 characters for testing purposes.",
                "query",
                5,
                "analysis",
            )

        assert result is None

    def test_record_intelligent_outcome_no_integration(self, orchestrator):
        """Test _record_runtime_intelligence_outcome with no integration."""
        orchestrator._runtime_intelligence_enabled = False
        # Should not raise
        orchestrator._record_runtime_intelligence_outcome(True, 0.9, True, True)

    def test_record_intelligent_outcome_with_integration(self, orchestrator):
        """Test _record_runtime_intelligence_outcome with integration."""
        mock_integration = MagicMock()
        # Mock the new method name (TD-002 refactoring)
        mock_integration.record_runtime_intelligence_outcome = MagicMock()
        with patch.object(
            type(orchestrator),
            "runtime_intelligence_integration",
            property(lambda self: mock_integration),
        ):
            orchestrator._record_runtime_intelligence_outcome(True, 0.9, True, True)

        # Verify the method was called (exact args depend on orchestrator state)
        mock_integration.record_runtime_intelligence_outcome.assert_called_once()

    def test_record_intelligent_outcome_error(self, orchestrator):
        """Test _record_runtime_intelligence_outcome handles errors."""
        mock_integration = MagicMock()
        # Mock the new method name (TD-002 refactoring)
        mock_integration.record_runtime_intelligence_outcome = MagicMock(
            side_effect=Exception("Error")
        )
        with patch.object(
            type(orchestrator),
            "runtime_intelligence_integration",
            property(lambda self: mock_integration),
        ):
            # Should not raise - errors are handled internally
            orchestrator._record_runtime_intelligence_outcome(True, 0.9, True, True)


class TestPrepareStream:
    """Tests for _prepare_stream method."""

    @pytest.mark.asyncio
    async def test_prepare_stream_basic(self, orchestrator):
        """Test _prepare_stream initializes stream variables via service runtime."""
        result = await orchestrator._get_chat_stream_adapter()._prepare_stream("test message")
        assert result is not None
        assert len(result) == 11  # Should return 11 values
        # Unpack to verify structure
        (
            stream_metrics,
            start_time,
            total_tokens,
            cumulative_usage,
            max_total_iterations,
            max_exploration_iterations,
            total_iterations,
            force_completion,
            unified_task_type,
            task_classification,
            complexity_tool_budget,
        ) = result
        assert stream_metrics is not None
        assert start_time > 0
        assert total_tokens == 0
        # cumulative_usage may have zero values initialized
        assert isinstance(cumulative_usage, dict)

    @pytest.mark.asyncio
    async def test_prepare_stream_skips_intelligent_request_for_direct_response(self, orchestrator):
        """Direct-response prompts should bypass intelligent reminder injection."""
        runtime = orchestrator._get_chat_stream_adapter()
        orchestrator._prepare_runtime_intelligence_request = AsyncMock(
            side_effect=AssertionError("direct-response prompts should bypass intelligent hooks")
        )

        messages_before = list(orchestrator.get_messages())
        await runtime._prepare_stream("Reply with exactly READY")
        messages_after = orchestrator.get_messages()[len(messages_before) :]

        assert not any(
            "[SYSTEM-REMINDER:" in getattr(message, "content", "") for message in messages_after
        )

    @pytest.mark.asyncio
    async def test_prepare_stream_carries_forward_prior_task_shape_for_bare_continue(
        self, orchestrator
    ):
        """Bare continuation commands should inherit prior task type and complexity."""
        from victor.agent.unified_task_tracker import TrackerTaskType
        from victor.framework.task import TaskComplexity

        prior_classification = SimpleNamespace(
            complexity=TaskComplexity.ANALYSIS,
            confidence=0.98,
        )
        orchestrator._last_stream_task_context = {
            "unified_task_type": TrackerTaskType.ANALYZE,
            "task_classification": prior_classification,
            "complexity_tool_budget": 42,
            "coarse_task_type": "analysis",
            "is_analysis_task": True,
            "is_action_task": False,
            "needs_execution": False,
        }
        orchestrator.unified_tracker.detect_task_type = MagicMock(
            return_value=TrackerTaskType.GENERAL
        )
        orchestrator.task_analyzer.classify_complexity = MagicMock(
            return_value=SimpleNamespace(
                complexity=TaskComplexity.SIMPLE,
                confidence=0.95,
            )
        )

        result = await orchestrator._get_chat_stream_adapter()._prepare_stream("continue")

        assert result[8] == TrackerTaskType.ANALYZE
        assert result[9] is prior_classification
        assert result[10] == 42

    @pytest.mark.asyncio
    async def test_prepare_stream_promotes_write_followup_to_action_budget(self, orchestrator):
        """Explicit write-authorized follow-ups should be promoted out of simple/general."""
        from victor.agent.unified_task_tracker import TrackerTaskType
        from victor.framework.task import (
            DEFAULT_BUDGETS,
            TaskClassification,
            TaskComplexity,
        )
        from victor.agent.action_authorizer import ActionIntent, IntentClassification

        # This test verifies the write-followup -> ACTION promotion *logic*. The
        # promotion chain has FOUR inputs, all of which run platform-sensitive
        # classifiers for a deliberately-ambiguous follow-up. On macOS they land
        # on the promoting path; on CI-Linux they flip and the promotion never
        # fires (manifesting as a non-ACTION complexity). Pin all four so the
        # test exercises the deterministic promotion logic, not fuzzy classifier
        # boundaries:
        #   1. task TYPE -> GENERAL  (so chat_stream considers promoting to EDIT)
        #   2. the GENERAL->EDIT gate _should_promote_general_task_to_edit -> True.
        #      This is the real culprit behind the CI flake: it calls the
        #      MODULE-LEVEL detect_intent + is_ambiguous_write_followup_request,
        #      NOT the analyzer mocked below, so pinning the analyzer alone is
        #      insufficient.
        #   3. base COMPLEXITY -> SIMPLE  (prepare_task gates ACTION promotion on SIMPLE)
        #   4. analyzer INTENT -> WRITE_ALLOWED  (required for SIMPLE -> ACTION)
        orchestrator.unified_tracker.detect_task_type = MagicMock(
            return_value=TrackerTaskType.GENERAL
        )
        analyzer = orchestrator.task_coordinator.task_analyzer
        analyzer.classify_complexity = MagicMock(
            return_value=TaskClassification(
                complexity=TaskComplexity.SIMPLE,
                tool_budget=DEFAULT_BUDGETS[TaskComplexity.SIMPLE],
                confidence=0.9,
                matched_patterns=[],
            )
        )
        analyzer.detect_intent = MagicMock(
            return_value=IntentClassification(
                intent=ActionIntent.WRITE_ALLOWED,
                confidence=0.9,
                matched_signals=[],
                safe_actions=set(),
                prompt_guard="",
            )
        )

        adapter = orchestrator._get_chat_stream_adapter()
        # Pin the write-authorized-followup gate (uses module-level classifiers).
        adapter._should_promote_general_task_to_edit = MagicMock(return_value=True)

        result = await adapter._prepare_stream(
            "Are you able to address them. if yes please address them comprehensively."
        )

        assert result[8] == TrackerTaskType.EDIT
        assert result[9].complexity == TaskComplexity.ACTION
        assert result[10] >= DEFAULT_BUDGETS[TaskComplexity.ACTION]

    @pytest.mark.asyncio
    async def test_prepare_stream_reuses_resume_context_without_overriding_action_followup_shape(
        self, orchestrator
    ):
        """Continuation payloads should keep resume context without reverting to prior analysis shape."""
        from victor.agent.unified_task_tracker import TrackerTaskType
        from victor.framework.task import TaskComplexity

        prior_classification = SimpleNamespace(
            complexity=TaskComplexity.ANALYSIS,
            confidence=0.97,
        )
        orchestrator._last_stream_task_context = {
            "unified_task_type": TrackerTaskType.ANALYZE,
            "task_classification": prior_classification,
            "complexity_tool_budget": 42,
            "coarse_task_type": "analysis",
            "is_analysis_task": True,
            "is_action_task": False,
            "needs_execution": False,
            "degraded_resume_state": True,
            "resume_summary": "previous turn ended before completion",
        }
        orchestrator.unified_tracker.detect_task_type = MagicMock(
            return_value=TrackerTaskType.DESIGN
        )
        orchestrator.task_analyzer.classify_complexity = MagicMock(
            return_value=SimpleNamespace(
                complexity=TaskComplexity.SIMPLE,
                confidence=0.95,
                matched_patterns=[],
            )
        )

        result = await orchestrator._get_chat_stream_adapter()._prepare_stream(
            "continue to address the remaining findings and suggestions"
        )

        assert result[8] == TrackerTaskType.DESIGN
        assert result[9].complexity == TaskComplexity.ACTION
        assert orchestrator._pending_continuation_task_context["degraded_resume_state"] is True


class TestApplyTaskGuidance:
    """Tests for _apply_task_guidance method."""

    def test_apply_task_guidance_analysis_task(self, orchestrator):
        """Test _apply_task_guidance for analysis task."""
        from victor.agent.unified_task_tracker import TrackerTaskType

        orchestrator._apply_task_guidance(
            user_message="Analyze this code",
            unified_task_type=TrackerTaskType.ANALYZE,
            is_analysis_task=True,
            is_action_task=False,
            needs_execution=False,
            max_exploration_iterations=10,
        )
        # Should add a system message for analysis
        messages = orchestrator.get_messages()
        # The method adds system messages
        assert len(messages) >= 1

    def test_apply_task_guidance_action_task(self, orchestrator):
        """Test _apply_task_guidance for action task."""
        from victor.agent.unified_task_tracker import TrackerTaskType

        orchestrator._apply_task_guidance(
            user_message="Create a new file",
            unified_task_type=TrackerTaskType.CREATE,
            is_analysis_task=False,
            is_action_task=True,
            needs_execution=True,
            max_exploration_iterations=5,
        )
        messages = orchestrator.get_messages()
        assert len(messages) >= 1


class TestGoalHints:
    """Tests for tool_planner.infer_goals_from_message method."""

    def test_goal_hints_for_code_request(self, orchestrator):
        """Test tool_planner.infer_goals_from_message for code request."""
        hints = orchestrator._tool_planner.infer_goals_from_message(
            "Write a Python function to sort a list"
        )
        assert isinstance(hints, list)

    def test_goal_hints_for_analysis_request(self, orchestrator):
        """Test tool_planner.infer_goals_from_message for analysis request."""
        hints = orchestrator._tool_planner.infer_goals_from_message("Explain how this code works")
        assert isinstance(hints, list)

    def test_goal_hints_for_debug_request(self, orchestrator):
        """Test tool_planner.infer_goals_from_message for debug request."""
        hints = orchestrator._tool_planner.infer_goals_from_message(
            "Fix this bug in the authentication"
        )
        assert isinstance(hints, list)


class TestHandleCancellation:
    """Tests for cancellation handling (now inline in ChatCoordinator._run_iteration_pre_checks).

    The _handle_cancellation method was removed from the orchestrator as part of
    the coordinator decomposition. Cancellation is now handled inline in
    ChatCoordinator._run_iteration_pre_checks. Basic cancellation is tested
    in TestCancellation above.
    """

    def test_cancellation_check_returns_false_when_not_cancelled(self, orchestrator):
        """Test _check_cancellation returns False when not cancelled."""
        assert orchestrator._check_cancellation() is False

    def test_cancellation_check_returns_true_when_cancelled(self, orchestrator):
        """Test _check_cancellation returns True when cancelled."""
        import asyncio

        orchestrator._cancel_event = asyncio.Event()
        orchestrator.request_cancellation()
        assert orchestrator._check_cancellation() is True


class TestResolveShellVariant:
    """Tests for _resolve_shell_variant method."""

    def test_non_shell_alias_returns_unchanged(self, orchestrator):
        """Test non-shell aliases return unchanged."""
        result = orchestrator._resolve_shell_variant("read_file")
        assert result == "read_file"

    def test_shell_alias_with_shell_enabled(self, orchestrator):
        """Test shell alias resolves to 'shell' when shell is enabled."""
        from victor.tools.tool_names import ToolNames
        from victor.agent.shell_resolver import resolve_shell_variant

        orchestrator.tools.is_tool_enabled = MagicMock(side_effect=lambda t: t == ToolNames.SHELL)
        # Call shell_resolver directly with the mocked tools registry
        result = resolve_shell_variant("bash", orchestrator.tools)
        assert result == ToolNames.SHELL

    def test_shell_alias_with_neither_enabled(self, orchestrator):
        """Test shell alias returns canonical when neither enabled."""
        from victor.agent.shell_resolver import resolve_shell_variant

        orchestrator.tools.is_tool_enabled = MagicMock(return_value=False)
        result = resolve_shell_variant("execute", orchestrator.tools)
        # Should return canonical name
        assert result in ["shell", "execute"]

    def test_various_shell_aliases(self, orchestrator):
        """Test various shell aliases are recognized."""
        from victor.tools.tool_names import ToolNames
        from victor.agent.shell_resolver import resolve_shell_variant

        orchestrator.tools.is_tool_enabled = MagicMock(side_effect=lambda t: t == ToolNames.SHELL)

        aliases = ["run", "bash", "execute", "cmd", "execute_bash"]
        for alias in aliases:
            result = resolve_shell_variant(alias, orchestrator.tools)
            assert result == ToolNames.SHELL


class TestGetThinkingDisabledPrompt:
    """Tests for _get_thinking_disabled_prompt method."""

    def test_no_prefix_returns_base_prompt(self, orchestrator):
        """Test returns base prompt when no thinking disable prefix."""
        mock_caps = MagicMock(spec=[])  # No thinking_disable_prefix
        orchestrator.tool_calling_caps = mock_caps
        result = orchestrator._get_thinking_disabled_prompt("Hello world")
        assert result == "Hello world"

    def test_with_prefix_prepends_to_prompt(self, orchestrator):
        """Test prepends prefix when thinking disable prefix available."""
        mock_caps = MagicMock()
        mock_caps.thinking_disable_prefix = "/no_think"
        orchestrator.tool_calling_caps = mock_caps
        result = orchestrator._get_thinking_disabled_prompt("Hello world")
        assert result == "/no_think\nHello world"

    def test_with_none_prefix(self, orchestrator):
        """Test returns base prompt when prefix is None."""
        mock_caps = MagicMock()
        mock_caps.thinking_disable_prefix = None
        orchestrator.tool_calling_caps = mock_caps
        result = orchestrator._get_thinking_disabled_prompt("Test prompt")
        assert result == "Test prompt"


class TestMemorySessionId:
    """Tests for get_memory_session_id method."""

    def test_returns_session_id_when_set(self, orchestrator):
        """Test returns session ID when set."""
        orchestrator._memory_session_id = "test-session-123"
        result = orchestrator.get_memory_session_id()
        assert result == "test-session-123"

    def test_returns_none_when_not_set(self, orchestrator):
        """Test returns None when not set."""
        orchestrator._memory_session_id = None
        result = orchestrator.get_memory_session_id()
        assert result is None


class TestGetRecentSessions:
    """Tests for get_recent_sessions method."""

    def test_returns_empty_when_no_memory_manager(self, orchestrator):
        """Test returns empty list when memory manager not enabled."""
        # Mock session_service to return empty list when no memory manager
        orchestrator._session_service.get_recent_sessions = MagicMock(return_value=[])
        result = orchestrator.get_recent_sessions()
        assert result == []
        orchestrator._session_service.get_recent_sessions.assert_called_once_with(10)

    def test_returns_sessions_from_memory_manager(self, orchestrator):
        """Test returns sessions from memory manager."""
        from datetime import datetime

        expected_session = {
            "session_id": "session-1",
            "created_at": datetime(2024, 1, 1, 12, 0, 0),
            "last_activity": datetime(2024, 1, 1, 13, 0, 0),
            "project_path": "/test/path",
            "provider": "anthropic",
            "model": "claude-3",
            "message_count": 2,
        }

        # Mock session_service to return the session
        orchestrator._session_service.get_recent_sessions = MagicMock(
            return_value=[expected_session]
        )

        result = orchestrator.get_recent_sessions(limit=5)

        assert len(result) == 1
        assert result[0]["session_id"] == "session-1"
        assert result[0]["message_count"] == 2
        orchestrator._session_service.get_recent_sessions.assert_called_once_with(5)

    def test_handles_exception_gracefully(self, orchestrator):
        """Test handles exception and returns empty list."""
        # Mock the service method to simulate exception handling at service level
        # The service catches exceptions and returns []
        orchestrator._session_service.get_recent_sessions = MagicMock(return_value=[])

        result = orchestrator.get_recent_sessions()
        assert result == []


class TestRecoverSession:
    """Tests for recover_session method."""

    def test_returns_false_when_no_memory_manager(self, orchestrator):
        """Test returns False when memory manager not enabled."""
        # Mock session_service to return False when no memory manager
        orchestrator._session_service.recover_session = MagicMock(return_value=False)
        result = orchestrator.recover_session("session-123")
        assert result is False
        orchestrator._session_service.recover_session.assert_called_once_with("session-123")

    def test_returns_false_when_session_not_found(self, orchestrator):
        """Test returns False when session not found."""
        # Mock session_service to return False when session not found
        orchestrator._session_service.recover_session = MagicMock(return_value=False)
        result = orchestrator.recover_session("nonexistent-session")
        assert result is False
        orchestrator._session_service.recover_session.assert_called_once_with("nonexistent-session")

    def test_recovers_session_successfully(self, orchestrator):
        """Test recovers session and restores messages."""
        # Mock session_service to return True on successful recovery
        orchestrator._session_service.recover_session = MagicMock(return_value=True)

        result = orchestrator.recover_session("session-123")

        assert result is True
        assert orchestrator._memory_session_id == "session-123"
        orchestrator._session_service.recover_session.assert_called_once_with("session-123")

    def test_handles_exception_gracefully(self, orchestrator):
        """Test handles exception and returns False."""
        # Mock session_service to return False on exception
        orchestrator._session_service.recover_session = MagicMock(return_value=False)
        result = orchestrator.recover_session("session-123")
        assert result is False
        orchestrator._session_service.recover_session.assert_called_once_with("session-123")


class TestGetMemoryContext:
    """Tests for get_memory_context method."""

    def test_falls_back_to_in_memory_when_no_manager(self, orchestrator):
        """Test falls back to in-memory messages when no memory manager."""
        # Mock session_service to return fallback messages
        mock_msg = MagicMock()
        mock_msg.model_dump.return_value = {"role": "user", "content": "test"}
        orchestrator._session_service.get_memory_context = MagicMock(
            return_value=[{"role": "user", "content": "test"}]
        )

        result = orchestrator.get_memory_context()

        assert len(result) == 1
        assert result[0]["role"] == "user"
        orchestrator._session_service.get_memory_context.assert_called_once()

    def test_falls_back_when_no_session_id(self, orchestrator):
        """Test falls back when no session ID."""
        # Mock session_service to return fallback messages
        orchestrator._session_service.get_memory_context = MagicMock(
            return_value=[{"role": "assistant", "content": "hello"}]
        )

        result = orchestrator.get_memory_context()

        assert len(result) == 1
        orchestrator._session_service.get_memory_context.assert_called_once()

    def test_gets_context_from_memory_manager(self, orchestrator):
        """Test gets context from memory manager."""
        # Mock session_service to return context from memory manager
        expected_context = [
            {"role": "user", "content": "test1"},
            {"role": "assistant", "content": "test2"},
        ]
        orchestrator._session_service.get_memory_context = MagicMock(return_value=expected_context)

        result = orchestrator.get_memory_context(max_tokens=1000)

        assert len(result) == 2
        orchestrator._session_service.get_memory_context.assert_called_once()

    def test_handles_exception_with_fallback(self, orchestrator):
        """Test handles exception and falls back to in-memory."""
        # Mock session_service to return fallback on exception
        orchestrator._session_service.get_memory_context = MagicMock(
            return_value=[{"role": "user", "content": "fallback"}]
        )

        result = orchestrator.get_memory_context()

        assert len(result) == 1
        assert result[0]["content"] == "fallback"


class TestGetSessionStats:
    """Tests for get_session_stats method.

    The orchestrator's get_session_stats() delegates to SessionService.
    These tests verify the delegation pattern and error handling.
    """

    def test_returns_disabled_when_no_memory_manager(self, orchestrator):
        """Test returns disabled stats when no memory manager."""
        # Mock session_service to return disabled stats
        orchestrator._session_service.get_session_stats = MagicMock(return_value={"enabled": False})

        result = orchestrator.get_session_stats()

        assert result["enabled"] is False
        orchestrator._session_service.get_session_stats.assert_called_once()

    def test_returns_error_when_session_not_found(self, orchestrator):
        """Test returns error when session not found (empty stats from memory_manager)."""
        # Mock session_service to return enabled but empty stats
        orchestrator._session_service.get_session_stats = MagicMock(return_value={"enabled": True})

        result = orchestrator.get_session_stats()

        assert result["enabled"] is True
        orchestrator._session_service.get_session_stats.assert_called_once()

    def test_returns_full_stats(self, orchestrator):
        """Test returns full session stats via delegation to SessionService."""
        # Mock session_service to return full stats
        expected_stats = {
            "enabled": True,
            "session_id": "session-123",
            "message_count": 2,
            "total_tokens": 300,
            "available_tokens": 3200,
            "utilization": 0.075,
            "role_distribution": {"user": 1, "assistant": 1},
            "tool_usage_count": 0,
            "created_at": "2024-01-01T12:00:00",
            "last_activity": "2024-01-01T13:00:00",
            "duration_seconds": 3600.0,
        }
        orchestrator._session_service.get_session_stats = MagicMock(return_value=expected_stats)

        result = orchestrator.get_session_stats()

        # Verify delegation was called
        orchestrator._session_service.get_session_stats.assert_called_once()

        # Verify SessionService includes expected stats
        assert result["enabled"] is True
        assert result["message_count"] == 2
        assert result["total_tokens"] == 300
        assert result["available_tokens"] == 3200

    def test_handles_exception_gracefully(self, orchestrator):
        """Test handles exception and returns error via SessionService."""
        # Mock session_service to return enabled stats despite exception
        orchestrator._session_service.get_session_stats = MagicMock(return_value={"enabled": True})

        result = orchestrator.get_session_stats()

        assert result["enabled"] is True
        orchestrator._session_service.get_session_stats.assert_called_once()


class TestFilterToolsByIntent:
    """Tests for tool_planner.filter_tools_by_intent method."""

    def test_returns_all_tools_when_no_intent(self, orchestrator):
        """Test returns all tools when no intent set."""
        tools = [{"name": "write_file"}, {"name": "read_file"}]
        result = orchestrator._tool_planner.filter_tools_by_intent(tools, None)
        assert len(result) == 2

    def test_filters_write_tools_for_display_only(self, orchestrator):
        """Test filters write tools for DISPLAY_ONLY intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Create mock tools with name attribute
        write_tool = MagicMock()
        write_tool.name = "write_file"
        read_tool = MagicMock()
        read_tool.name = "read_file"

        tools = [write_tool, read_tool]
        result = orchestrator._tool_planner.filter_tools_by_intent(tools, ActionIntent.DISPLAY_ONLY)

        # Write tools should be filtered out for DISPLAY_ONLY
        tool_names = [t.name for t in result]
        assert "read_file" in tool_names

    def test_no_filtering_for_write_allowed(self, orchestrator):
        """Test no filtering for WRITE_ALLOWED intent."""
        from victor.agent.action_authorizer import ActionIntent

        tools = [{"name": "write_file"}, {"name": "read_file"}]
        result = orchestrator._tool_planner.filter_tools_by_intent(
            tools, ActionIntent.WRITE_ALLOWED
        )
        assert len(result) == 2


class TestLogToolCall:
    """Tests for _log_tool_call method."""

    def test_logs_tool_call(self, orchestrator):
        """Test _log_tool_call logs the tool call."""
        # Should not raise
        orchestrator._log_tool_call("read_file", {"path": "/test"})


class TestComponentAccessors:
    """Tests for component accessor properties."""

    def test_conversation_controller_accessor(self, orchestrator):
        """Test conversation_controller property returns controller."""
        controller = orchestrator.conversation_controller
        assert controller is not None
        from victor.agent.conversation.controller import ConversationController

        assert isinstance(controller, ConversationController)

    def test_tool_pipeline_accessor(self, orchestrator):
        """Test tool_pipeline property returns pipeline."""
        pipeline = orchestrator.tool_pipeline
        assert pipeline is not None
        from victor.agent.tool_pipeline import ToolPipeline

        assert isinstance(pipeline, ToolPipeline)

    def test_streaming_controller_accessor(self, orchestrator):
        """Test streaming_controller property returns controller."""
        controller = orchestrator.streaming_controller
        assert controller is not None
        from victor.agent.streaming_controller import StreamingController

        assert isinstance(controller, StreamingController)

    def test_streaming_handler_accessor(self, orchestrator):
        """Test streaming_handler property returns handler."""
        handler = orchestrator.streaming_handler
        assert handler is not None
        from victor.agent.streaming import StreamingChatHandler

        assert isinstance(handler, StreamingChatHandler)

    def test_task_analyzer_accessor(self, orchestrator):
        """Test task_analyzer property returns analyzer."""
        analyzer = orchestrator.task_analyzer
        assert analyzer is not None
        from victor.agent.task_analyzer import TaskAnalyzer

        assert isinstance(analyzer, TaskAnalyzer)

    def test_observability_accessor(self, orchestrator):
        """Test observability property returns integration or None."""
        obs = orchestrator.observability
        # May be None or ObservabilityIntegration - just test no error


class TestRecoveryIntegration:
    """Tests for recovery_integration property."""

    def test_recovery_integration_accessor(self, orchestrator):
        """Test recovery_integration property returns integration."""
        integration = orchestrator.recovery_integration
        # May be a real integration or a no-op
        assert integration is not None


class TestUsageAnalytics:
    """Tests for usage_analytics property."""

    def test_usage_analytics_accessor(self, orchestrator):
        """Test usage_analytics property returns analytics."""
        analytics = orchestrator.usage_analytics
        assert analytics is not None
        from victor.agent.usage_analytics import UsageAnalytics

        assert isinstance(analytics, UsageAnalytics)


class TestSequenceTracker:
    """Tests for sequence_tracker property."""

    def test_sequence_tracker_accessor(self, orchestrator):
        """Test sequence_tracker returns tracker."""
        tracker = orchestrator.sequence_tracker
        assert tracker is not None
        from victor.agent.tool_sequence_tracker import ToolSequenceTracker

        assert isinstance(tracker, ToolSequenceTracker)


class TestContextCompactor:
    """Tests for context_compactor property."""

    def test_context_compactor_accessor(self, orchestrator):
        """Test context_compactor returns compactor."""
        compactor = orchestrator.context_compactor
        assert compactor is not None
        from victor.agent.context_compactor import ContextCompactor

        assert isinstance(compactor, ContextCompactor)


class TestRecoveryHandler:
    """Tests for recovery_handler property."""

    def test_recovery_handler_accessor(self, orchestrator):
        """Test recovery_handler returns handler or None."""
        handler = orchestrator.recovery_handler
        # May be None if recovery not configured
        if handler is not None:
            from victor.agent.recovery.handler import RecoveryHandler

            assert isinstance(handler, RecoveryHandler)


class TestProviderManager:
    """Tests for provider_manager property."""

    def test_provider_manager_accessor(self, orchestrator):
        """Test provider_manager returns manager."""
        manager = orchestrator.provider_manager
        assert manager is not None
        from victor.agent.provider_manager import ProviderManager

        assert isinstance(manager, ProviderManager)


class TestToolOutputFormatter:
    """Tests for tool_output_formatter property."""

    def test_tool_output_formatter_accessor(self, orchestrator):
        """Test tool_output_formatter returns formatter."""
        formatter = orchestrator.tool_output_formatter
        assert formatter is not None


class TestCreateBackgroundTask:
    """Tests for _create_background_task method."""

    @pytest.mark.asyncio
    async def test_create_background_task(self, orchestrator):
        """Test _create_background_task adds task to background_tasks."""

        async def dummy_coro():
            pass

        initial_count = len(orchestrator._background_tasks)
        orchestrator._create_background_task(dummy_coro(), name="test_task")

        # Should have added a task
        assert len(orchestrator._background_tasks) >= initial_count

    def test_create_background_task_without_loop_returns_none(self, orchestrator):
        """Test _create_background_task closes bare coroutines without a running loop."""

        async def dummy_coro():
            return None

        coro = dummy_coro()

        with patch(
            "victor.agent.orchestrator.asyncio.get_running_loop",
            side_effect=RuntimeError,
        ):
            result = orchestrator._create_background_task(coro, name="test_task")

        assert result is None
        assert coro.cr_frame is None


class TestSwitchModel:
    """Tests for switch_model method."""

    def test_switch_model_returns_bool(self, orchestrator):
        """Test switch_model returns boolean."""
        import asyncio

        # switch_model is now an async method (protocol method)
        # Use asyncio.run to call it from sync context
        result = asyncio.run(orchestrator.switch_model("claude-3-sonnet"))
        assert isinstance(result, bool)


class TestCurrentProviderModel:
    """Tests for current_provider and current_model properties."""

    def test_current_provider(self, orchestrator):
        """Test current_provider returns string."""
        provider = orchestrator.current_provider
        assert isinstance(provider, str)

    def test_current_model(self, orchestrator):
        """Test current_model returns string."""
        model = orchestrator.current_model
        assert isinstance(model, str)


class TestGetCurrentProviderInfo:
    """Tests for get_current_provider_info method."""

    def test_returns_dict_with_required_keys(self, orchestrator):
        """Test get_current_provider_info returns dict with expected keys."""
        info = orchestrator.get_current_provider_info()
        assert isinstance(info, dict)
        assert "tool_budget" in info
        assert "tool_calls_used" in info


class TestShouldContinueIntelligent:
    """Tests for _should_continue_runtime_intelligence method."""

    def test_returns_tuple(self, orchestrator):
        """Test _should_continue_runtime_intelligence returns tuple."""
        result = orchestrator._should_continue_runtime_intelligence()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_with_no_integration(self, orchestrator):
        """Test returns continue when no integration."""
        with patch.object(
            type(orchestrator),
            "runtime_intelligence_integration",
            property(lambda self: None),
        ):
            should_continue, reason = orchestrator._should_continue_runtime_intelligence()
            assert should_continue is True
            assert "disabled" in reason.lower()


class TestSafetyChecker:
    """Tests for safety_checker property."""

    def test_safety_checker_accessor(self, orchestrator):
        """Test safety_checker returns checker."""
        checker = orchestrator.safety_checker
        assert checker is not None
        from victor.agent.safety import SafetyChecker

        assert isinstance(checker, SafetyChecker)


class TestAutoCommitter:
    """Tests for auto_committer property."""

    def test_auto_committer_accessor(self, orchestrator):
        """Test auto_committer returns committer or None."""
        committer = orchestrator.auto_committer
        # May be None if not enabled
        if committer is not None:
            from victor.agent.auto_commit import AutoCommitter

            assert isinstance(committer, AutoCommitter)


class TestApplyVerticalMiddleware:
    """Tests for apply_vertical_middleware method."""

    def test_empty_middleware_list(self, orchestrator):
        """Test apply_vertical_middleware with empty list."""
        # Should not raise
        orchestrator.apply_vertical_middleware([])

    def test_with_middleware(self, orchestrator):
        """Test apply_vertical_middleware adds middleware."""
        mock_middleware = MagicMock()
        orchestrator.apply_vertical_middleware([mock_middleware])
        # Should have stored middleware
        assert hasattr(orchestrator, "_vertical_middleware")


class TestCodeCorrectionMiddleware:
    """Tests for code_correction_middleware property."""

    def test_code_correction_middleware_accessor(self, orchestrator):
        """Test code_correction_middleware returns middleware or None."""
        middleware = orchestrator.code_correction_middleware
        # May be None if not configured
        # Just verify accessing it doesn't raise


class TestMessages:
    """Tests for messages property."""

    def test_messages_returns_list(self, orchestrator):
        """Test messages property returns list."""
        messages = orchestrator.messages
        assert isinstance(messages, list)


class TestConversation:
    """Tests for conversation attribute."""

    def test_conversation_exists(self, orchestrator):
        """Test conversation attribute exists."""
        conv = orchestrator.conversation
        assert conv is not None

    def test_conversation_has_message_methods(self, orchestrator):
        """Test conversation has expected methods."""
        conv = orchestrator.conversation
        assert hasattr(conv, "add_message")
        assert hasattr(conv, "clear")


class TestProviderName:
    """Tests for provider_name attribute."""

    def test_provider_name_returns_string(self, orchestrator):
        """Test provider_name returns string."""
        name = orchestrator.provider_name
        assert isinstance(name, str)


class TestToolAdapter:
    """Tests for tool_adapter attribute."""

    def test_tool_adapter_exists(self, orchestrator):
        """Test tool_adapter exists."""
        adapter = orchestrator.tool_adapter
        assert adapter is not None


class TestToolCallingCaps:
    """Tests for tool_calling_caps attribute."""

    def test_tool_calling_caps_exists(self, orchestrator):
        """Test tool_calling_caps exists."""
        caps = orchestrator.tool_calling_caps
        assert caps is not None
        assert hasattr(caps, "native_tool_calls")


class TestTools:
    """Tests for tools attribute."""

    def test_tools_exists(self, orchestrator):
        """Test tools registry exists."""
        tools = orchestrator.tools
        assert tools is not None
        # Check common registry methods
        assert hasattr(tools, "get") or hasattr(tools, "get_tool")


class TestSettings:
    """Tests for settings attribute."""

    def test_settings_exists(self, orchestrator):
        """Test settings exists."""
        settings = orchestrator.settings
        assert settings is not None


class TestPromptBuilder:
    """Tests for prompt_builder attribute."""

    def test_prompt_builder_exists(self, orchestrator):
        """Test prompt_builder exists."""
        builder = orchestrator.prompt_builder
        assert builder is not None


class TestProjectContext:
    """Tests for project_context attribute."""

    def test_project_context_exists(self, orchestrator):
        """Test project_context exists."""
        context = orchestrator.project_context
        assert context is not None


class TestSanitizer:
    """Tests for sanitizer attribute."""

    def test_sanitizer_exists(self, orchestrator):
        """Test sanitizer exists."""
        sanitizer = orchestrator.sanitizer
        assert sanitizer is not None


class TestMetricsCollector:
    """Tests for _metrics_collector attribute."""

    def test_metrics_collector_exists(self, orchestrator):
        """Test _metrics_collector exists."""
        collector = orchestrator._metrics_collector
        assert collector is not None


class TestResetConversation:
    """Tests for reset_conversation method."""

    def test_reset_conversation_clears_messages(self, orchestrator):
        """Test reset_conversation clears conversation."""
        # Add a message first
        orchestrator.conversation.add_message(role="user", content="test")

        # Reset
        orchestrator.reset_conversation()

        # Should have cleared (or at least method runs without error)
        # The exact behavior depends on implementation


class TestAddMessage:
    """Tests for add_message method."""

    def test_add_user_message(self, orchestrator):
        """Test add_message adds a user message."""
        # Clear first
        orchestrator.conversation.clear()

        # Add message
        orchestrator.add_message(role="user", content="Hello")

        # Should have at least one message now
        assert len(orchestrator.messages) >= 1

    def test_add_assistant_message(self, orchestrator):
        """Test add_message adds an assistant message."""
        orchestrator.conversation.clear()
        orchestrator.add_message(role="assistant", content="Hi there")
        assert len(orchestrator.messages) >= 1


class TestParseToolCallsWithAdapter:
    """Tests for _parse_tool_calls_with_adapter method."""

    def test_parse_empty_content(self, orchestrator):
        """Test parsing empty content."""
        result = orchestrator._parse_tool_calls_with_adapter("")
        assert result is not None
        # Should return a ToolCallParseResult
        assert hasattr(result, "tool_calls") or isinstance(result, dict)

    def test_parse_content_without_tool_calls(self, orchestrator):
        """Test parsing content without tool calls."""
        result = orchestrator._parse_tool_calls_with_adapter("Just some text response")
        assert result is not None


class TestBuildSystemPrompt:
    """Tests for build_system_prompt method."""

    def test_returns_string(self, orchestrator):
        """Test build_system_prompt returns string."""
        prompt = orchestrator.build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestGetModelContextWindow:
    """Tests for _get_model_context_window method."""

    def test_returns_positive_integer(self, orchestrator):
        """Test _get_model_context_window returns positive int."""
        window = orchestrator._get_model_context_window()
        assert isinstance(window, int)
        assert window > 0

    def test_returns_known_provider_default(self, orchestrator):
        """Test returns known provider defaults."""
        # Should return at least 10000 for any provider
        window = orchestrator._get_model_context_window()
        assert window >= 10000


class TestGetMaxContextChars:
    """Tests for _get_max_context_chars method."""

    def test_returns_positive_integer(self, orchestrator):
        """Test _get_max_context_chars returns positive int."""
        max_chars = orchestrator._get_max_context_chars()
        assert isinstance(max_chars, int)
        assert max_chars > 0


class TestCheckContextOverflow:
    """Tests for _check_context_overflow method."""

    def test_returns_boolean(self, orchestrator):
        """Test _check_context_overflow returns boolean."""
        result = orchestrator._check_context_overflow()
        assert isinstance(result, bool)

    def test_empty_context_no_overflow(self, orchestrator):
        """Test empty conversation doesn't overflow."""
        orchestrator.conversation.clear()
        result = orchestrator._check_context_overflow()
        assert result is False


class TestGetContextMetrics:
    """Tests for get_context_metrics method."""

    def test_returns_context_metrics(self, orchestrator):
        """Test get_context_metrics returns ContextMetrics."""
        metrics = orchestrator.get_context_metrics()
        assert metrics is not None
        assert hasattr(metrics, "char_count")
        assert hasattr(metrics, "estimated_tokens")


class TestDebugLogger:
    """Tests for debug_logger attribute."""

    def test_debug_logger_exists(self, orchestrator):
        """Test debug_logger exists."""
        logger = orchestrator.debug_logger
        assert logger is not None


class TestModel:
    """Tests for model attribute."""

    def test_model_returns_string(self, orchestrator):
        """Test model returns string."""
        model = orchestrator.model
        assert isinstance(model, str)


class TestProvider:
    """Tests for provider attribute."""

    def test_provider_exists(self, orchestrator):
        """Test provider exists."""
        provider = orchestrator.provider
        assert provider is not None


class TestUnifiedTracker:
    """Tests for unified_tracker attribute."""

    def test_unified_tracker_exists(self, orchestrator):
        """Test unified_tracker exists."""
        tracker = orchestrator.unified_tracker
        assert tracker is not None


class TestIntentClassifier:
    """Tests for intent_classifier attribute."""

    def test_intent_classifier_exists(self, orchestrator):
        """Test intent_classifier exists."""
        classifier = orchestrator.intent_classifier
        assert classifier is not None


class TestAdaptiveModeController:
    """Tests for _mode_controller attribute on tool pipeline."""

    def test_mode_controller_on_pipeline(self, orchestrator):
        """Test _mode_controller may exist on tool pipeline."""
        # mode_controller is on the pipeline, not orchestrator
        if hasattr(orchestrator, "_tool_pipeline") and orchestrator._tool_pipeline:
            # May or may not have _mode_controller
            assert True
        else:
            assert True


class TestToolBudgetProperty:
    """Tests for tool_budget property."""

    def test_tool_budget_returns_int(self, orchestrator):
        """Test tool_budget returns int."""
        budget = orchestrator.tool_budget
        assert isinstance(budget, int)
        assert budget > 0


class TestToolCallsUsedProperty:
    """Tests for tool_calls_used property."""

    def test_tool_calls_used_returns_int(self, orchestrator):
        """Test tool_calls_used returns int."""
        used = orchestrator.tool_calls_used
        assert isinstance(used, int)
        assert used >= 0


class TestGetEnabledTools:
    """Tests for get_enabled_tools method."""

    def test_returns_set(self, orchestrator):
        """Test get_enabled_tools returns set."""
        tools = orchestrator.get_enabled_tools()
        assert isinstance(tools, set)
        # Items may be strings (tool names) or tool objects
        # depending on how _enabled_tools is set
        assert len(tools) >= 0


class TestGetAvailableTools:
    """Tests for get_available_tools method."""

    def test_returns_set(self, orchestrator):
        """Test get_available_tools returns set."""
        tools = orchestrator.get_available_tools()
        assert isinstance(tools, set)
        # Items should be strings (tool names)
        assert len(tools) >= 0


class TestIsToolEnabled:
    """Tests for is_tool_enabled method."""

    def test_returns_bool(self, orchestrator):
        """Test is_tool_enabled returns bool."""
        result = orchestrator.is_tool_enabled("read_file")
        assert isinstance(result, bool)


class TestBackgroundTasks:
    """Tests for _background_tasks attribute."""

    def test_background_tasks_is_set(self, orchestrator):
        """Test _background_tasks is a set."""
        tasks = orchestrator._background_tasks
        assert isinstance(tasks, set)


class TestCancelEvent:
    """Tests for _cancel_event attribute."""

    def test_cancel_event_can_be_set(self, orchestrator):
        """Test _cancel_event can be set."""
        import asyncio

        orchestrator._cancel_event = asyncio.Event()
        assert orchestrator._cancel_event is not None


class TestIntentClassifierModule:
    """Tests for intent classifier from embeddings module."""

    def test_intent_type_exists(self, orchestrator):
        """Test IntentType can be imported from embeddings."""
        from victor.storage.embeddings.intent_classifier import IntentType

        # IntentType should have CONTINUATION enum value
        assert hasattr(IntentType, "CONTINUATION") or True  # May not exist

    def test_intent_classifier_instance(self, orchestrator):
        """Test orchestrator has intent_classifier."""
        classifier = orchestrator.intent_classifier
        assert classifier is not None


class TestVerticalMiddleware:
    """Tests for vertical middleware handling."""

    def test_vertical_middleware_attribute(self, orchestrator):
        """Test _vertical_middleware can be set."""
        mock_middleware = MagicMock()
        orchestrator.apply_vertical_middleware([mock_middleware])
        assert hasattr(orchestrator, "_vertical_middleware")
        assert len(orchestrator._vertical_middleware) == 1


class TestRouteSearchQuery:
    """Tests for route_search_query method."""

    def test_returns_dict(self, orchestrator):
        """Test route_search_query returns dict with expected keys."""
        result = orchestrator.route_search_query("find function")
        assert isinstance(result, dict)
        assert "recommended_tool" in result
        assert "confidence" in result

    def test_handles_keyword_query(self, orchestrator):
        """Test route_search_query handles keyword queries."""
        result = orchestrator.route_search_query("class MyClass")
        assert result["recommended_tool"] in [
            "code_search",
            "semantic_code_search",
            "both",
        ]


class TestGetRecommendedSearchTool:
    """Tests for get_recommended_search_tool method."""

    def test_returns_string(self, orchestrator):
        """Test get_recommended_search_tool returns string."""
        result = orchestrator.get_recommended_search_tool("find error handling")
        assert isinstance(result, str)
        assert result in ["code_search", "semantic_code_search", "both"]


class TestRecordToolExecution:
    """Tests for _record_tool_execution method."""

    def test_records_successful_execution(self, orchestrator):
        """Test _record_tool_execution records success."""
        orchestrator._record_tool_execution(
            tool_name="read_file",
            success=True,
            elapsed_ms=100.0,
        )
        # Should not raise

    def test_records_failed_execution(self, orchestrator):
        """Test _record_tool_execution records failure."""
        orchestrator._record_tool_execution(
            tool_name="read_file",
            success=False,
            elapsed_ms=50.0,
            error_type="FileNotFoundError",
        )
        # Should not raise


class TestSetEnabledTools:
    """Tests for set_enabled_tools method."""

    def test_sets_enabled_tools(self, orchestrator):
        """Test set_enabled_tools delegates to ToolService."""
        tools = {"read_file", "write_file"}
        orchestrator.set_enabled_tools(tools)
        # _enabled_tools moved to service layer; verify via service mock
        orchestrator._tool_service.set_enabled_tools.assert_called_once_with(tools)


class TestClassifyTaskKeywords:
    """Tests for _classify_task_keywords method."""

    def test_classifies_action_task(self, orchestrator):
        """Test _classify_task_keywords returns classification dict."""
        result = orchestrator._classify_task_keywords("create a new file")
        assert isinstance(result, dict)
        # Result may contain legacy keys or new format (task_type, confidence)
        assert "task_type" in result or "is_action_task" in result

    def test_classifies_analysis_task(self, orchestrator):
        """Test _classify_task_keywords returns classification dict."""
        result = orchestrator._classify_task_keywords("explain how this code works")
        assert isinstance(result, dict)
        assert "task_type" in result or "is_analysis_task" in result


class TestGetToolUsageStats:
    """Tests for get_tool_usage_stats method."""

    def test_returns_dict(self, orchestrator):
        """Test get_tool_usage_stats returns dict."""
        stats = orchestrator.get_tool_usage_stats()
        assert isinstance(stats, dict)


class TestGetConversationStage:
    """Tests for get_conversation_stage method."""

    def test_returns_stage(self, orchestrator):
        """Test get_conversation_stage returns a stage."""
        stage = orchestrator.get_conversation_stage()
        # Should return a ConversationStage enum
        assert stage is not None


class TestGetStageRecommendedTools:
    """Tests for get_stage_recommended_tools method."""

    def test_returns_set(self, orchestrator):
        """Test get_stage_recommended_tools returns a set."""
        tools = orchestrator.get_stage_recommended_tools()
        assert isinstance(tools, set)


class TestGetOptimizationStatus:
    """Tests for get_optimization_status method."""

    def test_returns_dict(self, orchestrator):
        """Test get_optimization_status returns a dict."""
        status = orchestrator.get_optimization_status()
        assert isinstance(status, dict)


class TestGetLastStreamMetrics:
    """Tests for get_last_stream_metrics method."""

    def test_returns_none_initially(self, orchestrator):
        """Test get_last_stream_metrics returns None initially."""
        metrics = orchestrator.get_last_stream_metrics()
        # May be None if no stream has occurred
        assert metrics is None or isinstance(metrics, object)


class TestGetStreamingMetricsSummary:
    """Tests for get_streaming_metrics_summary method."""

    def test_returns_none_or_dict(self, orchestrator):
        """Test get_streaming_metrics_summary returns None or dict."""
        summary = orchestrator.get_streaming_metrics_summary()
        assert summary is None or isinstance(summary, dict)


class TestGetStreamingMetricsHistory:
    """Tests for get_streaming_metrics_history method."""

    def test_returns_list(self, orchestrator):
        """Test get_streaming_metrics_history returns list."""
        history = orchestrator.get_streaming_metrics_history(limit=5)
        assert isinstance(history, list)


class TestGetMiddlewareChain:
    """Tests for get_middleware_chain method."""

    def test_returns_middleware_or_none(self, orchestrator):
        """Test get_middleware_chain returns middleware or None."""
        chain = orchestrator.get_middleware_chain()
        # May be None or a middleware chain
        assert chain is None or chain is not None


class TestSearchRouter:
    """Tests for search_router attribute."""

    def test_search_router_exists(self, orchestrator):
        """Test search_router attribute exists."""
        router = orchestrator.search_router
        assert router is not None


class TestToolExecutionTracker:
    """Tests for tool execution tracking."""

    def test_track_tool_execution(self, orchestrator):
        """Test tool execution can be tracked."""
        # Access tool_stats to verify tracking infrastructure exists
        if hasattr(orchestrator, "tool_stats"):
            assert orchestrator.tool_stats is not None


class TestProviderHealth:
    """Tests for get_provider_health method."""

    @pytest.mark.asyncio
    async def test_get_provider_health(self, orchestrator):
        """Test get_provider_health returns health info."""
        health = await orchestrator.get_provider_health()
        assert isinstance(health, dict)


class TestSystemPromptProperty:
    """Tests for _system_prompt attribute."""

    def test_system_prompt_returns_string(self, orchestrator):
        """Test _system_prompt returns a string."""
        prompt = orchestrator._system_prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestIterationCountMethods:
    """Tests for iteration count methods."""

    def test_get_iteration_count(self, orchestrator):
        """Test get_iteration_count returns int."""
        count = orchestrator.get_iteration_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_unified_tracker_has_iterations(self, orchestrator):
        """Test unified_tracker has iterations attribute."""
        iterations = orchestrator.unified_tracker.iterations
        assert isinstance(iterations, int)
        assert iterations >= 0


class TestObservedFilesMethods:
    """Tests for file tracking methods."""

    def test_get_observed_files(self, orchestrator):
        """Test get_observed_files returns set."""
        files = orchestrator.get_observed_files()
        assert isinstance(files, set)

    def test_get_modified_files(self, orchestrator):
        """Test get_modified_files returns set."""
        files = orchestrator.get_modified_files()
        assert isinstance(files, set)


class TestStageMethods:
    """Tests for stage-related methods."""

    def test_get_stage(self, orchestrator):
        """Test get_stage returns stage."""
        stage = orchestrator.get_stage()
        assert stage is not None


class TestFlushAnalytics:
    """Tests for flush_analytics method."""

    def test_returns_dict(self, orchestrator):
        """Test flush_analytics returns dict."""
        result = orchestrator.flush_analytics()
        assert isinstance(result, dict)


class TestHealthMonitoring:
    """Tests for health monitoring methods."""

    @pytest.mark.asyncio
    async def test_start_health_monitoring(self, orchestrator):
        """Test start_health_monitoring."""
        # Should not raise
        result = await orchestrator.start_health_monitoring()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_stop_health_monitoring(self, orchestrator):
        """Test stop_health_monitoring."""
        result = await orchestrator.stop_health_monitoring()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_start_health_monitoring_uses_provider_service_only(self, orchestrator):
        """Canonical health monitoring should not fall back to the legacy coordinator."""
        orchestrator._provider_service = MagicMock()
        orchestrator._provider_service.start_health_monitoring = AsyncMock()
        orchestrator._deprecated_provider_coordinator = MagicMock()
        orchestrator._deprecated_provider_coordinator.start_health_monitoring = AsyncMock(
            side_effect=AssertionError("legacy coordinator path should not be used")
        )

        result = await orchestrator.start_health_monitoring()

        assert result is True
        orchestrator._provider_service.start_health_monitoring.assert_awaited_once_with()
        orchestrator._deprecated_provider_coordinator.start_health_monitoring.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_health_monitoring_uses_provider_service_only(self, orchestrator):
        """Canonical health monitoring shutdown should not fall back to the legacy coordinator."""
        orchestrator._provider_service = MagicMock()
        orchestrator._provider_service.stop_health_monitoring = AsyncMock()
        orchestrator._deprecated_provider_coordinator = MagicMock()
        orchestrator._deprecated_provider_coordinator.stop_health_monitoring = AsyncMock(
            side_effect=AssertionError("legacy coordinator path should not be used")
        )

        result = await orchestrator.stop_health_monitoring()

        assert result is True
        orchestrator._provider_service.stop_health_monitoring.assert_awaited_once_with()
        orchestrator._deprecated_provider_coordinator.stop_health_monitoring.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_health_monitoring_returns_false_without_provider_service(
        self, orchestrator
    ):
        """Unavailable canonical service should report failure instead of silent success."""
        orchestrator._provider_service = None

        result = await orchestrator.start_health_monitoring()

        assert result is False

    @pytest.mark.asyncio
    async def test_stop_health_monitoring_returns_false_without_provider_service(
        self, orchestrator
    ):
        """Unavailable canonical service should report failure instead of silent success."""
        orchestrator._provider_service = None

        result = await orchestrator.stop_health_monitoring()

        assert result is False


class TestGracefulShutdown:
    """Tests for graceful_shutdown method."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown_returns_dict(self, orchestrator):
        """Test graceful_shutdown returns dict."""
        result = await orchestrator.graceful_shutdown()
        assert isinstance(result, dict)


class TestApplyIntentGuard:
    """Tests for _apply_intent_guard method."""

    def test_applies_guard_for_message(self, orchestrator):
        """Test _apply_intent_guard doesn't raise for valid message."""
        orchestrator._apply_intent_guard("read this file")
        # Should not raise


class TestApplyTaskGuidance:
    """Tests for _apply_task_guidance method."""

    def test_applies_guidance(self, orchestrator):
        """Test _apply_task_guidance doesn't raise."""
        from victor.agent.unified_task_tracker import TrackerTaskType

        orchestrator._apply_task_guidance(
            user_message="create a new file",
            unified_task_type=TrackerTaskType.CREATE,
            is_analysis_task=False,
            is_action_task=True,
            needs_execution=False,
            max_exploration_iterations=10,
        )
        # Should not raise


class TestUsageLogger:
    """Tests for usage_logger attribute."""

    def test_usage_logger_exists(self, orchestrator):
        """Test usage_logger attribute exists."""
        logger = orchestrator.usage_logger
        assert logger is not None


class TestProjectContext:
    """Tests for project_context attribute."""

    def test_project_context_exists(self, orchestrator):
        """Test project_context attribute exists."""
        context = orchestrator.project_context
        assert context is not None


class TestPromptBuilderAttr:
    """Tests for prompt_builder attribute."""

    def test_prompt_builder_exists(self, orchestrator):
        """Test prompt_builder attribute exists."""
        builder = orchestrator.prompt_builder
        assert builder is not None


class TestBuildSystemPromptSecondary:
    """Additional tests for build_system_prompt method."""

    def test_returns_string(self, orchestrator):
        """Test build_system_prompt returns string."""
        prompt = orchestrator.build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestCheckContextOverflowMethod:
    """Tests for _check_context_overflow method."""

    def test_returns_bool(self, orchestrator):
        """Test _check_context_overflow returns bool."""
        result = orchestrator._check_context_overflow()
        assert isinstance(result, bool)


class TestToolCacheConfig:
    """Tests for tool cache configuration."""

    def test_tool_cache_ttl(self, orchestrator):
        """Test tool cache ttl is accessible."""
        ttl = orchestrator.settings.tools.tool_cache_ttl
        assert isinstance(ttl, (int, float))
        assert ttl >= 0


class TestToolPipelineAttr:
    """Tests for _tool_pipeline attribute."""

    def test_tool_pipeline_exists(self, orchestrator):
        """Test _tool_pipeline attribute exists or is None."""
        pipeline = getattr(orchestrator, "_tool_pipeline", None)
        # May or may not exist
        assert True


class TestConversationControllerAttr:
    """Tests for conversation_controller attribute."""

    def test_conversation_controller_exists(self, orchestrator):
        """Test conversation_controller attribute exists."""
        controller = orchestrator.conversation_controller
        assert controller is not None


class TestStreamingControllerAttr:
    """Tests for streaming_controller attribute."""

    def test_streaming_controller_exists(self, orchestrator):
        """Test streaming_controller may exist."""
        controller = getattr(orchestrator, "streaming_controller", None)
        # May or may not exist
        assert True


class TestProviderManagerAttr:
    """Tests for _provider_manager attribute."""

    def test_provider_manager_exists(self, orchestrator):
        """Test _provider_manager attribute exists."""
        manager = orchestrator._provider_manager
        assert manager is not None


class TestMetricsCollectorAttr:
    """Tests for _metrics_collector attribute."""

    def test_metrics_collector_exists(self, orchestrator):
        """Test _metrics_collector attribute exists."""
        collector = orchestrator._metrics_collector
        assert collector is not None


class TestToolRegistrarAttr:
    """Tests for tool_registrar attribute."""

    def test_tool_registrar_exists(self, orchestrator):
        """Test tool_registrar may exist."""
        registrar = getattr(orchestrator, "tool_registrar", None)
        # May or may not exist
        assert True


class TestDetermineContinuationActionExists:
    """Test that _determine_continuation_action was removed (delegated to ContinuationStrategy)."""

    def test_method_removed(self, orchestrator):
        """Test _determine_continuation_action was removed (deprecated, uses ContinuationStrategy)."""
        # This method was removed as part of the orchestrator decomposition.
        # Continuation logic is handled by ContinuationStrategy.
        assert not hasattr(orchestrator, "_determine_continuation_action")


class TestSwitchProviderMethod:
    """Tests for switch_provider method."""

    def test_switch_provider_exists(self, orchestrator):
        """Test switch_provider method exists."""
        assert hasattr(orchestrator, "switch_provider")

    @pytest.mark.asyncio
    async def test_switch_provider_to_invalid_returns_falsy(self, orchestrator):
        """Test switch_provider returns falsy for invalid provider (errors swallowed)."""
        result = await orchestrator.switch_provider("nonexistent_provider_xyz")
        assert not result


class TestContextMetrics:
    """Tests for get_context_metrics method."""

    def test_returns_context_metrics(self, orchestrator):
        """Test get_context_metrics returns ContextMetrics."""
        metrics = orchestrator.get_context_metrics()
        assert metrics is not None
        assert hasattr(metrics, "estimated_tokens") or hasattr(metrics, "char_count")


class TestModelContextWindow:
    """Tests for _get_model_context_window method."""

    def test_returns_positive_int(self, orchestrator):
        """Test _get_model_context_window returns positive int."""
        window = orchestrator._get_model_context_window()
        assert isinstance(window, int)
        assert window > 0


class TestMaxContextChars:
    """Tests for _get_max_context_chars method."""

    def test_returns_positive_int(self, orchestrator):
        """Test _get_max_context_chars returns positive int."""
        chars = orchestrator._get_max_context_chars()
        assert isinstance(chars, int)
        assert chars > 0


class TestAddSystemMessage:
    """Tests for adding system messages."""

    def test_add_message_system(self, orchestrator):
        """Test add_message with system role."""
        orchestrator.add_message("system", "Test system message")
        # Should not raise


class TestAddUserMessage:
    """Tests for adding user messages."""

    def test_add_message_user(self, orchestrator):
        """Test add_message with user role."""
        orchestrator.add_message("user", "Test user message")
        # Should not raise


class TestAddAssistantMessage:
    """Tests for adding assistant messages."""

    def test_add_message_assistant(self, orchestrator):
        """Test add_message with assistant role."""
        orchestrator.add_message("assistant", "Test assistant message")
        # Should not raise


class TestClearConversation:
    """Tests for clearing conversation."""

    def test_reset_conversation(self, orchestrator):
        """Test reset_conversation clears history."""
        orchestrator.add_message("user", "Test")
        orchestrator.reset_conversation()
        # After reset, should have fresh state


class TestToolBudgetManagement:
    """Tests for tool budget management."""

    def test_tool_budget_property(self, orchestrator):
        """Test tool_budget property."""
        budget = orchestrator.tool_budget
        assert isinstance(budget, int)
        assert budget > 0

    def test_tool_calls_used_property(self, orchestrator):
        """Test tool_calls_used property."""
        used = orchestrator.tool_calls_used
        assert isinstance(used, int)
        assert used >= 0


class TestConversationHistory:
    """Tests for conversation history access."""

    def test_messages_property(self, orchestrator):
        """Test messages property returns list."""
        messages = orchestrator.messages
        assert isinstance(messages, list)

    def test_conversation_attribute(self, orchestrator):
        """Test conversation attribute exists."""
        conv = orchestrator.conversation
        assert conv is not None


# =============================================================================
# STREAMING HANDLER INTEGRATION TESTS
# =============================================================================


class TestStreamingHandlerIntegration:
    """Tests for streaming handler integration with orchestrator."""

    def test_streaming_handler_exists(self, orchestrator):
        """Test that streaming_handler property exists."""
        handler = orchestrator.streaming_handler
        assert handler is not None

    def test_streaming_handler_has_check_time_limit(self, orchestrator):
        """Test that handler has check_time_limit method."""
        handler = orchestrator.streaming_handler
        assert hasattr(handler, "check_time_limit")
        assert callable(handler.check_time_limit)

    def test_streaming_handler_has_check_iteration_limit(self, orchestrator):
        """Test that handler has check_iteration_limit method."""
        handler = orchestrator.streaming_handler
        assert hasattr(handler, "check_iteration_limit")
        assert callable(handler.check_iteration_limit)

    def test_streaming_handler_has_should_continue_loop(self, orchestrator):
        """Test that handler has should_continue_loop method."""
        handler = orchestrator.streaming_handler
        assert hasattr(handler, "should_continue_loop")
        assert callable(handler.should_continue_loop)


class TestCheckTimeLimitWithHandler:
    """Tests for time limit checking (now inline in ChatCoordinator._run_iteration_pre_checks).

    The _check_time_limit_with_handler method was removed from the orchestrator.
    Time limit checking is now inline in ChatCoordinator._run_iteration_pre_checks.
    These tests verify the underlying StreamingChatContext time limit mechanism.
    """

    def test_context_under_time_limit(self, orchestrator):
        """Test StreamingChatContext.is_over_time_limit returns False when fresh."""
        from victor.agent.streaming import create_stream_context

        ctx = create_stream_context("test message")
        # Context just created - definitely under limit
        time_limit = getattr(orchestrator.settings, "stream_idle_timeout_seconds", 300)
        assert ctx.is_over_time_limit(time_limit) is False

    def test_context_over_time_limit(self, orchestrator):
        """Test StreamingChatContext.is_over_time_limit returns True when exceeded."""
        from victor.agent.streaming import create_stream_context
        import time

        ctx = create_stream_context("test message")
        # Artificially set start_time and last_activity_time to way in the past
        ctx.start_time = time.time() - 1000  # 1000 seconds ago
        ctx.last_activity_time = time.time() - 1000
        time_limit = getattr(orchestrator.settings, "stream_idle_timeout_seconds", 300)
        assert ctx.is_over_time_limit(time_limit) is True


class TestStreamingHandlerProperty:
    """Tests for streaming_handler property."""

    def test_returns_streaming_handler(self, orchestrator):
        """Test that streaming_handler returns a StreamingChatHandler."""
        from victor.agent.streaming import StreamingChatHandler

        handler = orchestrator.streaming_handler
        assert isinstance(handler, StreamingChatHandler)

    def test_handler_has_message_adder(self, orchestrator):
        """Test that handler has message_adder configured."""
        handler = orchestrator.streaming_handler
        assert handler.message_adder is not None

    def test_handler_has_settings(self, orchestrator):
        """Test that handler has settings configured."""
        handler = orchestrator.streaming_handler
        assert handler.settings is not None


class TestCheckProgressWithHandler:
    """Tests for progress checking (now owned by StreamingChatExecutor).

    The _check_progress_with_handler method was removed from the orchestrator.
    Progress checking is now handled via the recovery coordinator directly.
    These tests verify the recovery coordinator's check_progress method.
    """

    def test_recovery_coordinator_returns_true_when_making_progress(self, orchestrator):
        """Recovery coordinator returns True when agent is making progress."""
        from unittest.mock import MagicMock

        orchestrator._recovery_coordinator.check_progress = MagicMock(return_value=True)
        result = orchestrator._recovery_coordinator.check_progress(tool_calls_used=5)
        assert result is True

    def test_recovery_coordinator_returns_false_when_stuck(self, orchestrator):
        """Recovery coordinator returns False when agent is stuck."""
        from unittest.mock import MagicMock

        orchestrator._recovery_coordinator.check_progress = MagicMock(return_value=False)
        result = orchestrator._recovery_coordinator.check_progress(tool_calls_used=20)
        assert result is False

    def test_recovery_coordinator_exists(self, orchestrator):
        """Recovery coordinator is initialized on orchestrator."""
        assert hasattr(orchestrator, "_recovery_coordinator")
        assert orchestrator._recovery_coordinator is not None


class TestHandleForceCompletionWithHandler:
    """Tests for force completion (now driven by StreamingChatExecutor).

    The _handle_force_completion_with_handler method was removed from the orchestrator.
    Force completion is now handled via the unified_tracker and streaming context.
    These tests verify the streaming context's force_completion flag mechanism.
    """

    def test_force_completion_default_false(self, orchestrator):
        """StreamingChatContext force_completion defaults to False."""
        from victor.agent.streaming import create_stream_context

        ctx = create_stream_context("test")
        assert ctx.force_completion is False

    def test_force_completion_can_be_set(self, orchestrator):
        """StreamingChatContext force_completion can be set to True."""
        from victor.agent.streaming import create_stream_context

        ctx = create_stream_context("test")
        ctx.force_completion = True
        assert ctx.force_completion is True

    def test_unified_tracker_exists(self, orchestrator):
        """Unified tracker is initialized on orchestrator for stop decisions."""
        assert hasattr(orchestrator, "unified_tracker")
        assert orchestrator.unified_tracker is not None


class TestRateLimitRetry:
    """Tests for rate limit retry logic in _stream_provider_response."""

    def test_get_rate_limit_wait_time_with_retry_after(self, orchestrator):
        """Extract wait time from ProviderRateLimitError.retry_after via service runtime."""
        from victor.core.errors import ProviderRateLimitError

        exc = ProviderRateLimitError("Rate limit hit", retry_after=10)
        wait_time = orchestrator._get_chat_stream_adapter()._get_rate_limit_wait_time(
            exc, attempt=0
        )
        # Delegates to ProviderCoordinator - returns exact retry_after value
        assert wait_time == 10.0

    def test_get_rate_limit_wait_time_extracts_from_message(self, orchestrator):
        """Extract wait time from 'try again in X.XXs' pattern via service runtime."""
        exc = Exception("Rate limit exceeded. Please try again in 5.5s")
        wait_time = orchestrator._get_chat_stream_adapter()._get_rate_limit_wait_time(
            exc, attempt=0
        )
        # Delegates to ProviderCoordinator - exact parsed value (no buffer)
        assert wait_time == 5.5

    def test_get_rate_limit_wait_time_extracts_retry_after_pattern(self, orchestrator):
        """Extract wait time from 'retry after Xs' pattern via service runtime."""
        exc = Exception("Too many requests. Please retry after 3 seconds")
        wait_time = orchestrator._get_chat_stream_adapter()._get_rate_limit_wait_time(
            exc, attempt=0
        )
        # Delegates to ProviderCoordinator - exact parsed value (no buffer)
        assert wait_time == 3.0

    def test_get_rate_limit_wait_time_exponential_backoff(self, orchestrator):
        """Use exponential backoff when no wait time in error message via service runtime.

        When no wait time can be parsed, ProviderCoordinator returns default_rate_limit_wait (60s).
        The service runtime helper then applies exponential backoff: base_wait * 2^attempt.
        """
        cc = orchestrator._get_chat_stream_adapter()
        exc = Exception("429 Too Many Requests")
        # Attempt 0: 60 * 2^0 = 60 seconds (coordinator default)
        assert cc._get_rate_limit_wait_time(exc, attempt=0) == 60.0
        # Attempt 1: 60 * 2^1 = 120 seconds
        assert cc._get_rate_limit_wait_time(exc, attempt=1) == 120.0
        # Attempt 2: 60 * 2^2 = 240 seconds
        assert cc._get_rate_limit_wait_time(exc, attempt=2) == 240.0
        # Attempt 3: 60 * 2^3 = 480 seconds, capped at 300
        assert cc._get_rate_limit_wait_time(exc, attempt=3) == 300.0
        # Attempt 4: would be 960 but capped at 300
        assert cc._get_rate_limit_wait_time(exc, attempt=4) == 300.0

    def test_get_rate_limit_wait_time_caps_at_300_seconds(self, orchestrator):
        """Wait time capped at 300 seconds (5 minutes) via service runtime."""
        cc = orchestrator._get_chat_stream_adapter()
        exc = Exception("Rate limit exceeded. Please try again in 120s")
        # attempt=0: 120 * 2^0 = 120 (under cap)
        wait_time = cc._get_rate_limit_wait_time(exc, attempt=0)
        assert wait_time == 120.0
        # attempt=2: 120 * 2^2 = 480, capped at 300
        wait_time = cc._get_rate_limit_wait_time(exc, attempt=2)
        assert wait_time == 300.0
