"""Tests verifying orchestrator delegates correctly via both paths.

Tests that the orchestrator methods produce the same results whether
using direct coordinator delegation or service adapter delegation,
through the canonical service-adapter boundary.
"""

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.providers.base import CompletionResponse, Message, StreamChunk


class TestAdapterImports:
    """Test that adapter module imports work correctly."""

    def test_import_context_adapter(self):
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

        assert ContextServiceAdapter is not None

    def test_import_all_from_package(self):
        from victor.agent.services.adapters import ContextServiceAdapter

        assert ContextServiceAdapter is not None


class TestServiceProtocolImports:
    """Test that all 6 service protocols are importable."""

    def test_import_all_service_protocols(self):
        from victor.agent.services.protocols import (
            ChatServiceProtocol,
            ContextServiceProtocol,
            ProviderServiceProtocol,
            RecoveryServiceProtocol,
            SessionServiceProtocol,
            ToolServiceProtocol,
        )

        assert all(
            [
                ChatServiceProtocol,
                ToolServiceProtocol,
                SessionServiceProtocol,
                ContextServiceProtocol,
                ProviderServiceProtocol,
                RecoveryServiceProtocol,
            ]
        )

    def test_provider_service_protocol_has_required_methods(self):
        from victor.agent.services.protocols import ProviderServiceProtocol

        required = {
            "switch_provider",
            "switch_model",
            "get_current_provider_info",
            "check_provider_health",
            "get_available_providers",
            "start_health_monitoring",
            "stop_health_monitoring",
            "get_rate_limit_wait_time",
            "get_rate_limit_stats",
            "is_healthy",
        }
        actual = {name for name in dir(ProviderServiceProtocol) if not name.startswith("_")}
        assert required.issubset(actual), f"Missing: {required - actual}"

    def test_tool_service_protocol_has_required_methods(self):
        from victor.agent.services.protocols import ToolServiceProtocol

        required = {
            "select_tools",
            "execute_tool",
            "execute_tools_parallel",
            "get_tool_budget",
            "set_tool_budget",
            "get_tool_usage_stats",
            "reset_tool_budget",
            "process_tool_results",
            "get_available_tools",
            "get_enabled_tools",
            "set_enabled_tools",
            "is_tool_enabled",
            "resolve_tool_alias",
            "parse_and_validate_tool_calls",
            "execute_tool_with_retry",
            "normalize_tool_arguments",
            "build_tool_access_context",
            "validate_tool_call",
            "normalize_arguments_full",
            "on_tool_complete",
            "is_healthy",
        }
        actual = {name for name in dir(ToolServiceProtocol) if not name.startswith("_")}
        assert required.issubset(actual), f"Missing: {required - actual}"

    def test_recovery_service_protocol_has_required_methods(self):
        from victor.agent.services.protocols import RecoveryServiceProtocol

        required = {
            "classify_error",
            "select_recovery_action",
            "execute_recovery",
            "can_retry",
            "should_attempt_recovery",
            "handle_recovery_with_integration",
            "apply_recovery_action",
            "check_natural_completion",
            "handle_empty_response",
            "get_recovery_fallback_message",
            "check_tool_budget",
            "truncate_tool_calls",
            "filter_blocked_tool_calls",
            "check_blocked_threshold",
            "check_force_action",
            "is_healthy",
        }
        actual = {name for name in dir(RecoveryServiceProtocol) if not name.startswith("_")}
        assert required.issubset(actual), f"Missing: {required - actual}"

    def test_context_service_protocol_has_required_methods(self):
        from victor.agent.services.protocols import ContextServiceProtocol

        required = {
            "get_context_metrics",
            "check_context_overflow",
            "compact_context",
            "add_message",
            "get_messages",
            "is_healthy",
        }
        actual = {name for name in dir(ContextServiceProtocol) if not name.startswith("_")}
        assert required.issubset(actual), f"Missing: {required - actual}"


class TestAdapterProtocolConformance:
    """Test that adapters expose the expected interface methods."""

    def test_context_adapter_has_required_methods(self):
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

        adapter = ContextServiceAdapter(MagicMock())
        assert callable(getattr(adapter, "get_context_metrics", None))
        assert callable(getattr(adapter, "check_context_overflow", None))
        assert callable(getattr(adapter, "compact_context", None))
        assert callable(getattr(adapter, "add_message", None))
        assert callable(getattr(adapter, "add_messages", None))
        assert callable(getattr(adapter, "get_messages", None))
        assert callable(getattr(adapter, "clear_messages", None))
        assert callable(getattr(adapter, "get_max_tokens", None))
        assert callable(getattr(adapter, "set_max_tokens", None))
        assert callable(getattr(adapter, "estimate_tokens", None))
        assert callable(getattr(adapter, "is_healthy", None))

    @pytest.mark.asyncio
    async def test_context_adapter_supports_controller_backed_chat_runtime_operations(
        self,
    ):
        from victor.agent.conversation.controller import ConversationController
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter

        controller = ConversationController()
        controller.set_system_prompt("system prompt")
        adapter = ContextServiceAdapter(controller)

        adapter.add_message("user", "hello")

        assert [message.role for message in controller.messages] == ["system", "user"]
        assert await adapter.check_context_overflow() is False

        adapter.clear_messages(retain_system=True)

        assert [message.role for message in controller.messages] == ["system"]
        assert controller.messages[0].content == "system prompt"

    def test_bootstrap_new_services_registers_context_adapter_backed_by_controller(
        self,
    ):
        from victor.agent.conversation.controller import ConversationController
        from victor.agent.services.adapters.context_adapter import ContextServiceAdapter
        from victor.agent.services.protocols import ContextServiceProtocol
        from victor.core.bootstrap_services import bootstrap_new_services
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        controller = ConversationController()
        controller.set_system_prompt("system prompt")
        streaming = MagicMock()
        flag_manager = MagicMock()
        flag_manager.is_enabled.return_value = False

        with patch(
            "victor.core.feature_flags.get_feature_flag_manager",
            return_value=flag_manager,
        ):
            bootstrap_new_services(
                container,
                conversation_controller=controller,
                streaming_coordinator=streaming,
            )

        context_service = container.get_optional(ContextServiceProtocol)

        assert isinstance(context_service, ContextServiceAdapter)
        context_service.add_message("user", "hello")
        assert [message.role for message in controller.messages] == ["system", "user"]

    def test_protocol_adapter_runtime_state_setters_proxy_to_host_orchestrator(self):
        from victor.agent.services.orchestrator_protocol_adapter import (
            OrchestratorProtocolAdapter,
        )

        orchestrator = SimpleNamespace(temperature=0.2, tool_budget=5)
        adapter = OrchestratorProtocolAdapter(orchestrator)

        adapter.temperature = 0.6
        adapter.tool_budget = 12

        assert orchestrator.temperature == 0.6
        assert orchestrator.tool_budget == 12

    @pytest.mark.asyncio
    async def test_orchestrator_protocol_adapter_prefers_execute_tool_calls(self):
        from victor.agent.services.orchestrator_protocol_adapter import (
            OrchestratorProtocolAdapter,
        )

        orchestrator = MagicMock()
        orchestrator.execute_tool_calls = AsyncMock(
            return_value=[{"name": "read", "success": True}]
        )
        orchestrator._handle_tool_calls = AsyncMock(
            side_effect=AssertionError("legacy _handle_tool_calls bridge should not be used")
        )

        adapter = OrchestratorProtocolAdapter(orchestrator)
        result = await adapter.execute_tool_calls([{"name": "read", "arguments": {}}])

        assert result == [{"name": "read", "success": True}]
        orchestrator.execute_tool_calls.assert_awaited_once_with(
            [{"name": "read", "arguments": {}}]
        )

    @pytest.mark.asyncio
    async def test_orchestrator_protocol_adapter_exposes_sync_chat_runtime_surface(
        self,
    ):
        from victor.agent.services.orchestrator_protocol_adapter import (
            OrchestratorProtocolAdapter,
        )
        from victor.agent.services.protocols.chat_runtime import SyncChatRuntimeProtocol

        response = CompletionResponse(content="adapter-chat", role="assistant")
        orchestrator = MagicMock()
        orchestrator.chat = AsyncMock(return_value=response)
        orchestrator.get_last_skill_match_info.return_value = {"skill_match": "python"}

        adapter = OrchestratorProtocolAdapter(orchestrator)
        result = await adapter.chat("hello", use_planning=True)

        assert isinstance(adapter, SyncChatRuntimeProtocol)
        assert result is response
        assert adapter.get_last_skill_match_info() == {"skill_match": "python"}
        orchestrator.chat.assert_awaited_once_with("hello", use_planning=True)
        orchestrator.get_last_skill_match_info.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_orchestrator_protocol_adapter_exposes_stream_chat_and_helper_surface(
        self,
    ):
        from victor.agent.services.orchestrator_protocol_adapter import (
            OrchestratorProtocolAdapter,
        )

        stream_chunk = StreamChunk(content="stream", is_final=True)
        context_limit_helper = MagicMock()
        context_limit_helper.handle_limits = AsyncMock(return_value=(True, stream_chunk))
        orchestrator = MagicMock()

        async def _stream_chat(user_message: str, **kwargs):
            assert user_message == "hello"
            assert kwargs == {"mode": "compat"}
            yield stream_chunk

        orchestrator.stream_chat = _stream_chat
        orchestrator._get_context_limit_runtime = MagicMock(return_value=context_limit_helper)

        adapter = OrchestratorProtocolAdapter(orchestrator)
        chunks = [chunk async for chunk in adapter.stream_chat("hello", mode="compat")]

        assert chunks == [stream_chunk]
        assert adapter._get_context_limit_runtime() is context_limit_helper
        orchestrator._get_context_limit_runtime.assert_called_once_with()
        handled, result_chunk = await context_limit_helper.handle_limits(
            "hello",
            5,
            1000,
            1,
            0.8,
        )
        assert handled is True
        assert result_chunk is stream_chunk

    @pytest.mark.asyncio
    async def test_orchestrator_protocol_adapter_exposes_service_chat_runtime_handlers(
        self,
    ):
        from victor.agent.services.orchestrator_protocol_adapter import (
            OrchestratorProtocolAdapter,
        )
        from victor.agent.services.protocols.chat_runtime import (
            ChatRuntimeHelperAccessProtocol,
        )

        response = CompletionResponse(content="planned", role="assistant")
        planning_helper = MagicMock()
        planning_helper.run = AsyncMock(return_value=response)
        context_limit_helper = MagicMock()
        context_limit_helper.handle_limits = AsyncMock(return_value=(False, None))
        orchestrator = MagicMock()
        orchestrator._get_planning_chat_runtime = MagicMock(return_value=planning_helper)
        orchestrator._get_context_limit_runtime = MagicMock(return_value=context_limit_helper)

        adapter = OrchestratorProtocolAdapter(orchestrator)
        assert isinstance(adapter, ChatRuntimeHelperAccessProtocol)
        assert adapter._get_planning_chat_runtime() is planning_helper
        assert adapter._get_context_limit_runtime() is context_limit_helper
        planning_response = await planning_helper.run("plan this")
        handled, chunk = await context_limit_helper.handle_limits("plan this", 5, 1000, 1, 0.8)

        assert planning_response is response
        assert handled is False
        assert chunk is None
        orchestrator._get_planning_chat_runtime.assert_called_once_with()
        planning_helper.run.assert_awaited_once_with("plan this")
        orchestrator._get_context_limit_runtime.assert_called_once_with()
        context_limit_helper.handle_limits.assert_awaited_once_with(
            "plan this",
            5,
            1000,
            1,
            0.8,
        )

    @pytest.mark.asyncio
    async def test_orchestrator_protocol_adapter_service_runtime_handlers_require_canonical_helpers(
        self,
    ):
        from victor.agent.services.orchestrator_protocol_adapter import (
            OrchestratorProtocolAdapter,
        )

        orchestrator = MagicMock()

        adapter = OrchestratorProtocolAdapter(orchestrator)

        with pytest.raises(AttributeError, match="_get_planning_chat_runtime"):
            adapter._get_planning_chat_runtime()

        with pytest.raises(AttributeError, match="_get_context_limit_runtime"):
            adapter._get_context_limit_runtime()

        assert "_run_planning_chat_runtime" not in type(adapter).__dict__
        assert "_handle_context_and_iteration_limits_runtime" not in type(adapter).__dict__

    def test_orchestrator_protocol_adapter_exposes_planning_runtime_surface(self):
        from victor.agent.services.orchestrator_protocol_adapter import (
            OrchestratorProtocolAdapter,
        )

        provider = MagicMock(name="provider")
        profile = MagicMock(name="profile")
        orchestrator = MagicMock()
        orchestrator.provider = provider
        orchestrator.model = "gpt-test"
        orchestrator.max_tokens = 2048
        orchestrator.profile = profile
        orchestrator._planning_model_override = "openai:o4-mini"
        orchestrator._system_prompt_override = "original system prompt"

        adapter = OrchestratorProtocolAdapter(orchestrator)
        adapter.set_system_prompt("new system prompt")

        assert adapter.provider is provider
        assert adapter.model == "gpt-test"
        assert adapter.max_tokens == 2048
        assert adapter.profile is profile
        assert adapter.planning_model_override == "openai:o4-mini"
        assert adapter._system_prompt_override == "original system prompt"
        orchestrator.set_system_prompt.assert_called_once_with("new system prompt")

    @pytest.mark.asyncio
    async def test_orchestrator_protocol_adapter_resolves_helpers_even_if_stale_wrapper_attrs_exist(
        self,
    ):
        from victor.agent.services.orchestrator_protocol_adapter import (
            OrchestratorProtocolAdapter,
        )

        response = CompletionResponse(content="planned", role="assistant")
        stream_chunk = StreamChunk(content="service", is_final=True)
        planning_helper = MagicMock()
        planning_helper.run = AsyncMock(return_value=response)
        context_limit_helper = MagicMock()
        context_limit_helper.handle_limits = AsyncMock(return_value=(True, stream_chunk))

        orchestrator = MagicMock()
        orchestrator._chat_service = None
        orchestrator._get_planning_chat_runtime = MagicMock(return_value=planning_helper)
        orchestrator._get_context_limit_runtime = MagicMock(return_value=context_limit_helper)
        orchestrator._run_planning_chat_runtime = AsyncMock(
            side_effect=AssertionError("legacy planning wrapper should not be used")
        )
        orchestrator._handle_context_and_iteration_limits_runtime = AsyncMock(
            side_effect=AssertionError("legacy context-limit wrapper should not be used")
        )

        adapter = OrchestratorProtocolAdapter(orchestrator)
        planning_response = await adapter._get_planning_chat_runtime().run("plan this")
        handled, chunk = await adapter._get_context_limit_runtime().handle_limits(
            "plan this",
            5,
            1000,
            1,
            0.8,
        )

        assert planning_response is response
        assert handled is True
        assert chunk is stream_chunk
        orchestrator._get_planning_chat_runtime.assert_called_once_with()
        planning_helper.run.assert_awaited_once_with("plan this")
        orchestrator._get_context_limit_runtime.assert_called_once_with()
        context_limit_helper.handle_limits.assert_awaited_once_with(
            "plan this",
            5,
            1000,
            1,
            0.8,
        )

    def test_orchestrator_tool_strategy_event_prefers_metrics_service(self):
        orchestrator = object.__new__(AgentOrchestrator)
        orchestrator._metrics_coordinator = MagicMock()
        orchestrator.model = "test-model"
        orchestrator._is_tool_strategy_v2_enabled = MagicMock(return_value=True)

        provider = MagicMock()
        provider.name = "openai"
        tools = [SimpleNamespace(name="read"), SimpleNamespace(name="git_diff")]

        AgentOrchestrator._emit_tool_strategy_event(
            orchestrator,
            strategy="semantic_selection",
            tool_count=2,
            tool_tokens=128,
            context_window=32768,
            provider=provider,
            reason="small_context_window",
            tools=tools,
        )

        orchestrator._metrics_coordinator.emit_tool_strategy_event.assert_called_once_with(
            strategy="semantic_selection",
            tool_count=2,
            tool_tokens=128,
            context_window=32768,
            provider=provider,
            model="test-model",
            reason="small_context_window",
            tools=tools,
            v2_enabled=True,
        )

    def test_finish_task_report_includes_continuation_ledger_metadata(self):
        orchestrator = object.__new__(AgentOrchestrator)
        orchestrator.provider = SimpleNamespace(name="anthropic")
        orchestrator.model = "claude-sonnet"
        orchestrator._metrics_coordinator = MagicMock()
        orchestrator._metrics_coordinator.get_last_tool_strategy_event.return_value = {
            "tool_tokens": 144
        }
        orchestrator._metrics_coordinator.finish_task_report.return_value = {"task_id": "t-1"}
        orchestrator._context_service = MagicMock()
        orchestrator._context_service.get_performance_metrics.return_value = {}
        orchestrator._current_stream_context = SimpleNamespace(
            unified_task_type=SimpleNamespace(value="edit"),
            task_intent="Fix the failing parser regression",
            plan_steps=["Inspect failing tests", "Patch parser callsite"],
            intent_log=[{"kind": "analysis", "summary": "Read regression trace"}],
            resume_summary="Need to verify parser patch with targeted tests",
            degraded_resume_state=True,
            build_continuation_ledger=MagicMock(
                return_value=(
                    "Intent: Fix the failing parser regression\n"
                    "Plan: Inspect failing tests; Patch parser callsite"
                )
            ),
            compaction_summary="summarized",
            compaction_occurred=True,
            last_compaction_turn=4,
            compaction_message_removed_count=2,
            last_compaction_strategy="hybrid",
            last_compaction_reason="pre_tool_output",
            last_compaction_policy_reason="tool_output_exceeds_remaining_budget",
        )

        AgentOrchestrator._finish_task_report(
            orchestrator,
            True,
            user_message="Fix the parser regression",
        )

        finish_kwargs = orchestrator._metrics_coordinator.finish_task_report.call_args.kwargs
        metadata = finish_kwargs["metadata"]
        assert metadata["task_intent"] == "Fix the failing parser regression"
        assert metadata["plan_steps"] == [
            "Inspect failing tests",
            "Patch parser callsite",
        ]
        assert metadata["resume_summary"] == "Need to verify parser patch with targeted tests"
        assert metadata["degraded_resume_state"] is True
        assert metadata["continuation_ledger"].startswith("Intent:")

    def test_start_task_report_preserves_caller_metadata(self):
        orchestrator = object.__new__(AgentOrchestrator)
        orchestrator.provider = SimpleNamespace(name="anthropic")
        orchestrator.model = "claude-sonnet"
        orchestrator._metrics_coordinator = MagicMock()
        orchestrator._metrics_coordinator.start_task_report.return_value = "t-1"
        orchestrator._current_stream_context = None
        orchestrator.unified_tracker = None
        orchestrator._current_task_type = None
        orchestrator._task_type = None

        assert (
            AgentOrchestrator._start_task_report(
                orchestrator,
                "Summarize the changes",
                stream=True,
                metadata={"source": "test"},
            )
            == "t-1"
        )

        kwargs = orchestrator._metrics_coordinator.start_task_report.call_args.kwargs
        assert kwargs["metadata"] == {
            "stream": True,
            "provider": "anthropic",
            "model": "claude-sonnet",
            "source": "test",
        }


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestChatServiceBootstrapLaziness:
    """Chat service bootstrap should not force deprecated coordinator init.

    These tests exercise ``orchestrator.chat()``/``orchestrator.stream_chat()``
    — deprecated delegation shims to ``ChatService``. The deprecation is
    expected here; the tests assert delegation/laziness behavior, not the warning.
    """

    def test_initialize_services_keeps_chat_and_tool_coordinators_lazy(self):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.runtime.provider_runtime import LazyRuntimeProxy

        obj = object.__new__(AgentOrchestrator)
        chat_service = MagicMock()
        tool_service = MagicMock()
        session_service = MagicMock()
        context_service = MagicMock()
        provider_service = MagicMock()
        recovery_service = MagicMock()

        obj._container = MagicMock()
        obj._container.get_optional.side_effect = [
            chat_service,
            tool_service,
            session_service,
            context_service,
            provider_service,
            recovery_service,
        ]
        obj._container.is_registered.return_value = True
        obj._conversation_controller = MagicMock()
        obj._streaming_controller = MagicMock()
        obj._deprecated_provider_coordinator = MagicMock()
        obj._recovery_coordinator = MagicMock()
        obj._recovery_handler = MagicMock()
        obj._recovery_integration = MagicMock()
        obj._provider_manager = MagicMock()
        obj._lifecycle_manager = MagicMock()
        obj._checkpoint_manager = MagicMock()
        obj._session_cost_tracker = MagicMock()
        obj._tool_planner = MagicMock()
        obj._streaming_handler = MagicMock()
        obj._context_compactor = MagicMock()
        obj.unified_tracker = MagicMock()
        obj.settings = MagicMock()
        obj._presentation = MagicMock()
        obj.tools = MagicMock()
        obj.mode_controller = MagicMock()
        obj.argument_normalizer = MagicMock()
        obj._tool_pipeline = MagicMock()
        obj.tool_cache = MagicMock()
        obj.memory_manager = MagicMock()
        obj._memory_session_id = "mem-1"
        obj.tool_budget = 100
        obj._factory = MagicMock()
        obj._factory.create_streaming_chat_adapter.return_value = MagicMock()
        obj._turn_executor = None
        obj._protocol_adapter = None
        obj.observability = MagicMock()

        class TrapChatCoordinator:
            touched = False

            def __getattr__(self, name):
                self.touched = True
                raise AssertionError(
                    f"chat coordinator should remain lazy during bootstrap: {name}"
                )

        trap_chat = TrapChatCoordinator()
        obj._deprecated_chat_coordinator = LazyRuntimeProxy(
            factory=lambda: trap_chat,
            name="chat_coordinator",
        )

        class TrapToolCoordinator:
            touched = False

            def __getattr__(self, name):
                self.touched = True
                raise AssertionError(
                    f"tool coordinator should remain lazy during bootstrap: {name}"
                )

        trap_tool = TrapToolCoordinator()
        obj._deprecated_tool_coordinator = LazyRuntimeProxy(
            factory=lambda: trap_tool,
            name="tool_coordinator",
        )

        with (
            patch.object(AgentOrchestrator, "_register_coordinators_for_services"),
            patch.object(AgentOrchestrator, "_bootstrap_service_layer"),
        ):
            obj._initialize_services()

        tool_service.bind_runtime_components.assert_called_once()
        tool_kwargs = tool_service.bind_runtime_components.call_args.kwargs
        assert tool_kwargs["tool_registry"] is obj.tools
        assert tool_kwargs["tool_pipeline"] is obj._tool_pipeline
        assert tool_kwargs["tool_cache"] is obj.tool_cache
        assert tool_kwargs["mode_controller"] is obj.mode_controller
        assert tool_kwargs["tool_planner"] is obj._tool_planner
        assert tool_kwargs["argument_normalizer"] is obj.argument_normalizer
        assert "retry_executor" not in tool_kwargs
        assert "tool_call_parser" not in tool_kwargs
        assert obj._deprecated_tool_coordinator.initialized is False
        assert trap_tool.touched is False

        chat_service.bind_runtime_components.assert_called_once()
        kwargs = chat_service.bind_runtime_components.call_args.kwargs
        assert kwargs["turn_executor"].initialized is False
        assert callable(kwargs["planning_handler"])
        assert kwargs["planning_handler"].__self__ is obj._planning_chat_runtime
        assert kwargs["planning_handler"].__self__._runtime is obj
        assert (
            kwargs["stream_chat_handler"]
            is obj._factory.create_streaming_chat_adapter.return_value.stream_chat
        )
        assert callable(kwargs["context_limit_handler"])
        assert kwargs["context_limit_handler"].__self__ is obj._context_limit_runtime
        assert kwargs["context_limit_handler"].__self__._runtime is obj
        assert callable(kwargs["turn_setup_handler"])
        assert kwargs["turn_setup_handler"].__self__ is obj
        assert callable(kwargs["turn_teardown_handler"])
        assert kwargs["turn_teardown_handler"].__self__ is obj
        obj._factory.create_streaming_chat_adapter.assert_called_once_with(obj._protocol_adapter)
        assert obj._deprecated_chat_coordinator.initialized is False
        assert trap_chat.touched is False

    @pytest.mark.asyncio
    async def test_chat_service_runtime_handlers_use_service_first_helpers(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        chat_service = MagicMock()
        tool_service = MagicMock()
        session_service = MagicMock()
        context_service = MagicMock()
        provider_service = MagicMock()
        recovery_service = MagicMock()

        obj._container = MagicMock()
        obj._container.get_optional.side_effect = [
            chat_service,
            tool_service,
            session_service,
            context_service,
            provider_service,
            recovery_service,
        ]
        obj._container.is_registered.return_value = True
        obj._conversation_controller = MagicMock()
        obj._streaming_controller = MagicMock()
        obj._deprecated_provider_coordinator = MagicMock()
        obj._recovery_coordinator = MagicMock()
        obj._recovery_handler = MagicMock()
        obj._recovery_integration = MagicMock()
        obj._provider_manager = MagicMock()
        obj._lifecycle_manager = MagicMock()
        obj._checkpoint_manager = MagicMock()
        obj._session_cost_tracker = MagicMock()
        obj._tool_planner = MagicMock()
        obj._streaming_handler = MagicMock()
        obj._context_compactor = MagicMock()
        obj.unified_tracker = MagicMock()
        obj.settings = MagicMock()
        obj._presentation = MagicMock()
        obj._presentation.icon.return_value = "!"
        obj.tools = MagicMock()
        obj.mode_controller = MagicMock()
        obj.argument_normalizer = MagicMock()
        obj._tool_pipeline = MagicMock()
        obj.tool_cache = MagicMock()
        obj.memory_manager = MagicMock()
        obj._memory_session_id = "mem-1"
        obj.tool_budget = 100
        obj._factory = MagicMock()
        obj._factory.create_streaming_chat_adapter.return_value = MagicMock()
        obj._turn_executor = None
        obj._protocol_adapter = None
        obj.observability = MagicMock()
        obj._system_added = False
        obj._task_analyzer = MagicMock()
        obj._task_analyzer.analyze.return_value = "task-analysis"
        obj.conversation = MagicMock()
        obj.conversation.messages = [Message(role="user", content="existing")]
        obj.add_message = MagicMock()
        obj._check_context_overflow = MagicMock(return_value=False)
        obj._record_runtime_intelligence_outcome = MagicMock()

        class TrapChatCoordinator:
            touched = False

            def __getattr__(self, name):
                self.touched = True
                raise AssertionError(f"chat coordinator should remain lazy during runtime: {name}")

        trap_chat = TrapChatCoordinator()

        from victor.agent.runtime.provider_runtime import LazyRuntimeProxy

        obj._deprecated_chat_coordinator = LazyRuntimeProxy(
            factory=lambda: trap_chat,
            name="chat_coordinator",
        )

        planning_response = CompletionResponse(content="planned", role="assistant")

        with (
            patch.object(AgentOrchestrator, "_register_coordinators_for_services"),
            patch.object(AgentOrchestrator, "_bootstrap_service_layer"),
            patch("victor.agent.services.planning_runtime.PlanningRuntimeService") as planning_cls,
        ):
            planning_instance = MagicMock()
            planning_instance.chat_with_planning = AsyncMock(return_value=planning_response)
            planning_cls.return_value = planning_instance

            obj._initialize_services()

            kwargs = chat_service.bind_runtime_components.call_args.kwargs
            planning_handler = kwargs["planning_handler"]
            context_limit_handler = kwargs["context_limit_handler"]

            response = await planning_handler("plan this")
            handled, chunk = await context_limit_handler("plan this", 5, 1000, 1, 0.8)

        assert response is planning_response
        assert handled is False
        assert chunk is None
        planning_instance.chat_with_planning.assert_awaited_once_with(
            "plan this",
            task_analysis="task-analysis",
        )
        obj.conversation.ensure_system_prompt.assert_called_once()
        obj.add_message.assert_any_call("user", "plan this", metadata=ANY)
        obj.add_message.assert_any_call("assistant", "planned", metadata=ANY)
        obj._factory.create_streaming_chat_adapter.assert_called_once_with(obj._protocol_adapter)
        assert obj._deprecated_chat_coordinator.initialized is False
        assert trap_chat.touched is False

    def test_agent_orchestrator_no_longer_exposes_legacy_chat_runtime_wrappers(self):
        from victor.agent.orchestrator import AgentOrchestrator

        assert "_run_planning_chat_runtime" not in AgentOrchestrator.__dict__
        assert "_handle_context_and_iteration_limits_runtime" not in AgentOrchestrator.__dict__

    def test_get_planning_chat_runtime_prefers_cached_helper_and_orchestrator_host(
        self,
    ):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.planning_chat_runtime import PlanningChatRuntime

        obj = object.__new__(AgentOrchestrator)
        obj._protocol_adapter = MagicMock(name="protocol_adapter")

        first = obj._get_planning_chat_runtime()
        second = obj._get_planning_chat_runtime()

        assert isinstance(first, PlanningChatRuntime)
        assert first is second
        assert first._runtime is obj

    def test_get_provider_management_runtime_prefers_cached_helper_and_protocol_adapter(
        self,
    ):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.provider_management_runtime import (
            ProviderManagementRuntime,
        )

        obj = object.__new__(AgentOrchestrator)
        adapter = MagicMock(name="protocol_adapter")
        obj._protocol_adapter = adapter

        first = obj._get_provider_management_runtime()
        second = obj._get_provider_management_runtime()

        assert isinstance(first, ProviderManagementRuntime)
        assert first is second
        assert first._runtime is adapter

    def test_get_recovery_runtime_prefers_cached_helper_and_protocol_adapter(self):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.recovery_runtime import RecoveryRuntime

        obj = object.__new__(AgentOrchestrator)
        adapter = MagicMock(name="protocol_adapter")
        obj._protocol_adapter = adapter

        first = obj._get_recovery_runtime()
        second = obj._get_recovery_runtime()

        assert isinstance(first, RecoveryRuntime)
        assert first is second
        assert first._runtime is adapter

    def test_get_tool_execution_runtime_prefers_cached_helper_and_protocol_adapter(
        self,
    ):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.tool_execution_runtime import ToolExecutionRuntime

        obj = object.__new__(AgentOrchestrator)
        adapter = MagicMock(name="protocol_adapter")
        obj._protocol_adapter = adapter

        first = obj._get_tool_execution_runtime()
        second = obj._get_tool_execution_runtime()

        assert isinstance(first, ToolExecutionRuntime)
        assert first is second
        assert first._runtime is adapter

    def test_get_session_runtime_prefers_cached_helper_and_protocol_adapter(self):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.session_runtime import SessionRuntime

        obj = object.__new__(AgentOrchestrator)
        adapter = MagicMock(name="protocol_adapter")
        obj._protocol_adapter = adapter

        first = obj._get_session_runtime()
        second = obj._get_session_runtime()

        assert isinstance(first, SessionRuntime)
        assert first is second
        assert first._runtime is adapter

    @pytest.mark.asyncio
    async def test_switch_provider_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.switch_provider = AsyncMock(return_value=True)
        obj._provider_management_runtime = helper
        callback = MagicMock()

        result = await AgentOrchestrator.switch_provider(obj, "openai", "gpt-4.1", callback)

        assert result is True
        helper.switch_provider.assert_awaited_once_with("openai", "gpt-4.1", callback)

    @pytest.mark.asyncio
    async def test_switch_model_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.switch_model = AsyncMock(return_value=True)
        obj._provider_management_runtime = helper

        result = await AgentOrchestrator.switch_model(obj, "gpt-4.1")

        assert result is True
        helper.switch_model.assert_awaited_once_with("gpt-4.1")

    @pytest.mark.asyncio
    async def test_start_health_monitoring_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.start_health_monitoring = AsyncMock(return_value=True)
        obj._provider_management_runtime = helper

        result = await AgentOrchestrator.start_health_monitoring(obj)

        assert result is True
        helper.start_health_monitoring.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_stop_health_monitoring_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.stop_health_monitoring = AsyncMock(return_value=True)
        obj._provider_management_runtime = helper

        result = await AgentOrchestrator.stop_health_monitoring(obj)

        assert result is True
        helper.stop_health_monitoring.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_get_provider_health_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.get_provider_health = AsyncMock(return_value={"healthy": True})
        obj._provider_management_runtime = helper

        result = await AgentOrchestrator.get_provider_health(obj)

        assert result == {"healthy": True}
        helper.get_provider_health.assert_awaited_once_with()

    def test_get_current_provider_info_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.get_current_provider_info.return_value = {"provider_name": "openai"}
        obj._provider_management_runtime = helper

        result = AgentOrchestrator.get_current_provider_info(obj)

        assert result == {"provider_name": "openai"}
        helper.get_current_provider_info.assert_called_once_with()

    def test_create_recovery_context_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.create_recovery_context.return_value = {"iteration": 1}
        obj._recovery_runtime = helper
        stream_ctx = MagicMock(name="stream_ctx")

        result = AgentOrchestrator.create_recovery_context(obj, stream_ctx)

        assert result == {"iteration": 1}
        helper.create_recovery_context.assert_called_once_with(stream_ctx)

    @pytest.mark.asyncio
    async def test_handle_recovery_with_integration_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.handle_recovery_with_integration = AsyncMock(
            return_value=SimpleNamespace(action="continue")
        )
        obj._recovery_runtime = helper
        stream_ctx = MagicMock(name="stream_ctx")

        result = await AgentOrchestrator._handle_recovery_with_integration(
            obj,
            stream_ctx,
            "response",
            [{"name": "read_file"}],
            ["read_file"],
        )

        assert result.action == "continue"
        helper.handle_recovery_with_integration.assert_awaited_once_with(
            stream_ctx,
            "response",
            [{"name": "read_file"}],
            ["read_file"],
        )

    def test_apply_recovery_action_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.apply_recovery_action.return_value = StreamChunk(content="Recovered", is_final=True)
        obj._recovery_runtime = helper
        stream_ctx = MagicMock(name="stream_ctx")
        recovery_action = MagicMock(name="recovery_action")

        result = AgentOrchestrator._apply_recovery_action(obj, recovery_action, stream_ctx)

        assert result.content == "Recovered"
        helper.apply_recovery_action.assert_called_once_with(recovery_action, stream_ctx)

    @pytest.mark.asyncio
    async def test_execute_tool_calls_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.execute_tool_calls = AsyncMock(return_value=[{"name": "read", "success": True}])
        obj._tool_execution_runtime = helper

        result = await AgentOrchestrator.execute_tool_calls(
            obj,
            [{"name": "read", "arguments": {}}],
        )

        assert result == [{"name": "read", "success": True}]
        helper.execute_tool_calls.assert_awaited_once_with([{"name": "read", "arguments": {}}])

    def test_sync_session_service_runtime_state_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._session_runtime = helper

        AgentOrchestrator._sync_session_service_runtime_state(obj)

        helper.sync_runtime_state.assert_called_once_with()

    def test_get_recent_sessions_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.get_recent_sessions.return_value = [{"session_id": "session-1"}]
        obj._session_runtime = helper

        result = AgentOrchestrator.get_recent_sessions(obj, limit=5)

        assert result == [{"session_id": "session-1"}]
        helper.get_recent_sessions.assert_called_once_with(5)

    def test_recover_session_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.recover_session.return_value = True
        obj._session_runtime = helper

        result = AgentOrchestrator.recover_session(obj, "session-123")

        assert result is True
        helper.recover_session.assert_called_once_with("session-123")

    def test_get_memory_context_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.get_memory_context.return_value = [{"role": "user", "content": "hello"}]
        obj._session_runtime = helper

        result = AgentOrchestrator.get_memory_context(obj, max_tokens=200)

        assert result == [{"role": "user", "content": "hello"}]
        helper.get_memory_context.assert_called_once_with(max_tokens=200)

    def test_get_session_stats_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.get_session_stats.return_value = {"enabled": True}
        obj._session_runtime = helper

        result = AgentOrchestrator.get_session_stats(obj)

        assert result == {"enabled": True}
        helper.get_session_stats.assert_called_once_with()

    def test_record_prompt_optimization_metadata_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._session_runtime = helper
        turn_context = MagicMock(name="turn_context")

        AgentOrchestrator._record_prompt_optimization_metadata(obj, turn_context)

        helper.record_prompt_optimization_metadata.assert_called_once_with(turn_context)

    def test_get_active_prompt_optimization_metadata_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.get_active_prompt_optimization_metadata.return_value = {
            "entries": [],
            "by_section": {},
        }
        obj._session_runtime = helper

        result = AgentOrchestrator.get_active_prompt_optimization_metadata(obj)

        assert result == {"entries": [], "by_section": {}}
        helper.get_active_prompt_optimization_metadata.assert_called_once_with()

    def test_get_task_guidance_runtime_prefers_cached_helper_and_protocol_adapter(self):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.task_guidance_runtime import TaskGuidanceRuntime

        obj = object.__new__(AgentOrchestrator)
        adapter = MagicMock(name="protocol_adapter")
        obj._protocol_adapter = adapter

        first = obj._get_task_guidance_runtime()
        second = obj._get_task_guidance_runtime()

        assert isinstance(first, TaskGuidanceRuntime)
        assert first is second
        assert first._runtime is adapter

    def test_get_tool_selection_runtime_prefers_cached_helper_and_protocol_adapter(
        self,
    ):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.tool_selection_runtime import ToolSelectionRuntime

        obj = object.__new__(AgentOrchestrator)
        adapter = MagicMock(name="protocol_adapter")
        obj._protocol_adapter = adapter

        first = obj._get_tool_selection_runtime()
        second = obj._get_tool_selection_runtime()

        assert isinstance(first, ToolSelectionRuntime)
        assert first is second
        assert first._runtime is adapter

    def test_prepare_task_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.prepare_task.return_value = ("classification", 7)
        obj._task_guidance_runtime = helper

        result = AgentOrchestrator._prepare_task(obj, "plan this", "analysis")

        assert result == ("classification", 7)
        helper.prepare_task.assert_called_once_with("plan this", "analysis")

    def test_apply_intent_guard_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._task_guidance_runtime = helper

        AgentOrchestrator._apply_intent_guard(obj, "read this file")

        helper.apply_intent_guard.assert_called_once_with("read this file")

    def test_apply_task_guidance_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._task_guidance_runtime = helper

        AgentOrchestrator._apply_task_guidance(
            obj,
            user_message="analyze architecture",
            unified_task_type="analysis",
            is_analysis_task=True,
            is_action_task=False,
            needs_execution=False,
            max_exploration_iterations=8,
        )

        helper.apply_task_guidance.assert_called_once_with(
            user_message="analyze architecture",
            unified_task_type="analysis",
            is_analysis_task=True,
            is_action_task=False,
            needs_execution=False,
            max_exploration_iterations=8,
        )

    @pytest.mark.asyncio
    async def test_select_tools_for_turn_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.select_tools_for_turn = AsyncMock(return_value=[{"name": "search"}])
        obj._tool_selection_runtime = helper

        result = await AgentOrchestrator._select_tools_for_turn(
            obj,
            "inspect the repo",
            ["analyze"],
        )

        assert result == [{"name": "search"}]
        helper.select_tools_for_turn.assert_awaited_once_with(
            "inspect the repo",
            ["analyze"],
            planned_tools=None,
        )

    def test_get_context_limit_runtime_prefers_cached_helper_and_orchestrator_host(
        self,
    ):
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.services.context_limit_runtime import ContextLimitRuntime

        obj = object.__new__(AgentOrchestrator)
        obj._protocol_adapter = MagicMock(name="protocol_adapter")

        first = obj._get_context_limit_runtime()
        second = obj._get_context_limit_runtime()

        assert isinstance(first, ContextLimitRuntime)
        assert first is second
        assert first._runtime is obj

    @pytest.mark.asyncio
    async def test_planning_handler_skips_duplicate_turn_recording_after_direct_chat_fallback(
        self,
    ):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        chat_service = MagicMock()
        tool_service = MagicMock()
        session_service = MagicMock()
        context_service = MagicMock()
        provider_service = MagicMock()
        recovery_service = MagicMock()

        obj._container = MagicMock()
        obj._container.get_optional.side_effect = [
            chat_service,
            tool_service,
            session_service,
            context_service,
            provider_service,
            recovery_service,
        ]
        obj._container.is_registered.return_value = True
        obj._conversation_controller = MagicMock()
        obj._streaming_controller = MagicMock()
        obj._deprecated_provider_coordinator = MagicMock()
        obj._recovery_coordinator = MagicMock()
        obj._recovery_handler = MagicMock()
        obj._recovery_integration = MagicMock()
        obj._provider_manager = MagicMock()
        obj._lifecycle_manager = MagicMock()
        obj._checkpoint_manager = MagicMock()
        obj._session_cost_tracker = MagicMock()
        obj._tool_planner = MagicMock()
        obj._streaming_handler = MagicMock()
        obj._context_compactor = MagicMock()
        obj.unified_tracker = MagicMock()
        obj.settings = MagicMock()
        obj._presentation = MagicMock()
        obj._presentation.icon.return_value = "!"
        obj.tools = MagicMock()
        obj.mode_controller = MagicMock()
        obj.argument_normalizer = MagicMock()
        obj._tool_pipeline = MagicMock()
        obj.tool_cache = MagicMock()
        obj.memory_manager = MagicMock()
        obj._memory_session_id = "mem-1"
        obj.tool_budget = 100
        obj._factory = MagicMock()
        obj._factory.create_streaming_chat_adapter.return_value = MagicMock()
        obj._turn_executor = None
        obj._protocol_adapter = None
        obj.observability = MagicMock()
        obj._system_added = False
        obj._task_analyzer = MagicMock()
        obj._task_analyzer.analyze.return_value = "task-analysis"
        obj.conversation = MagicMock()
        obj.conversation.messages = [Message(role="system", content="existing")]
        obj.add_message = MagicMock()
        obj._check_context_overflow = MagicMock(return_value=False)
        obj._record_runtime_intelligence_outcome = MagicMock()

        direct_response = CompletionResponse(content="already recorded", role="assistant")

        async def _direct_chat(user_message: str) -> CompletionResponse:
            obj.add_message("user", user_message)
            obj.add_message("assistant", direct_response.content)
            obj.conversation.messages.extend(
                [
                    Message(role="user", content=user_message),
                    Message(role="assistant", content=direct_response.content),
                ]
            )
            return direct_response

        async def _direct_chat_with_planning(
            user_message: str,
            task_analysis=None,
        ) -> CompletionResponse:
            assert task_analysis == "task-analysis"
            return await _direct_chat(user_message)

        obj.chat = AsyncMock(side_effect=_direct_chat)

        with (
            patch.object(AgentOrchestrator, "_register_coordinators_for_services"),
            patch.object(AgentOrchestrator, "_bootstrap_service_layer"),
            patch("victor.agent.services.planning_runtime.PlanningRuntimeService") as planning_cls,
        ):
            planning_instance = MagicMock()
            planning_instance.chat_with_planning = AsyncMock(side_effect=_direct_chat_with_planning)
            planning_cls.return_value = planning_instance

            obj._initialize_services()

            planning_handler = chat_service.bind_runtime_components.call_args.kwargs[
                "planning_handler"
            ]
            response = await planning_handler("plan this")

        assert response is direct_response
        planning_instance.chat_with_planning.assert_awaited_once_with(
            "plan this",
            task_analysis="task-analysis",
        )
        assert obj.add_message.call_args_list == [
            (("user", "plan this"), {}),
            (("assistant", "already recorded"), {}),
        ]
        obj.conversation.ensure_system_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_chat_service_stream_handler_uses_service_runtime(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        chat_service = MagicMock()
        tool_service = MagicMock()
        session_service = MagicMock()
        context_service = MagicMock()
        provider_service = MagicMock()
        recovery_service = MagicMock()

        obj._container = MagicMock()
        obj._container.get_optional.side_effect = [
            chat_service,
            tool_service,
            session_service,
            context_service,
            provider_service,
            recovery_service,
        ]
        obj._container.is_registered.return_value = True
        obj._conversation_controller = MagicMock()
        obj._streaming_controller = MagicMock()
        obj._deprecated_provider_coordinator = MagicMock()
        obj._recovery_coordinator = MagicMock()
        obj._recovery_handler = MagicMock()
        obj._recovery_integration = MagicMock()
        obj._provider_manager = MagicMock()
        obj._lifecycle_manager = MagicMock()
        obj._checkpoint_manager = MagicMock()
        obj._session_cost_tracker = MagicMock()
        obj._tool_planner = MagicMock()
        obj._streaming_handler = MagicMock()
        obj._context_compactor = MagicMock()
        obj.unified_tracker = MagicMock()
        obj.settings = MagicMock()
        obj._presentation = MagicMock()
        obj.tools = MagicMock()
        obj.mode_controller = MagicMock()
        obj.argument_normalizer = MagicMock()
        obj._tool_pipeline = MagicMock()
        obj.tool_cache = MagicMock()
        obj.memory_manager = MagicMock()
        obj._memory_session_id = "mem-1"
        obj.tool_budget = 100
        obj._factory = MagicMock()
        obj._turn_executor = None
        obj._protocol_adapter = None
        obj.observability = MagicMock()

        class TrapChatCoordinator:
            touched = False

            def __getattr__(self, name):
                self.touched = True
                raise AssertionError(
                    f"chat coordinator should remain lazy during streaming: {name}"
                )

        trap_chat = TrapChatCoordinator()

        from victor.agent.runtime.provider_runtime import LazyRuntimeProxy

        obj._deprecated_chat_coordinator = LazyRuntimeProxy(
            factory=lambda: trap_chat,
            name="chat_coordinator",
        )

        stream_chunk = StreamChunk(content="service-stream", is_final=True)

        async def _runtime_stream_chat(user_message: str, **kwargs):
            assert user_message == "hello"
            assert kwargs == {"_preserve_iteration": True}
            yield stream_chunk

        runtime = MagicMock()
        runtime.stream_chat = _runtime_stream_chat
        obj._factory.create_streaming_chat_adapter.return_value = runtime

        with (
            patch.object(AgentOrchestrator, "_register_coordinators_for_services"),
            patch.object(AgentOrchestrator, "_bootstrap_service_layer"),
        ):
            obj._initialize_services()

            kwargs = chat_service.bind_runtime_components.call_args.kwargs
            stream_chat_handler = kwargs["stream_chat_handler"]
            chunks = [c async for c in stream_chat_handler("hello", _preserve_iteration=True)]

        assert chunks == [stream_chunk]
        obj._factory.create_streaming_chat_adapter.assert_called_once_with(obj._protocol_adapter)
        assert obj._deprecated_chat_coordinator.initialized is False
        assert trap_chat.touched is False

    @pytest.mark.asyncio
    async def test_orchestrator_stream_chat_delegates_to_chat_service(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        # chat()/stream_chat() re-stamp the session_id contextvar from
        # active_session_id (commit 89df2774, #360); __new__ skips __init__.
        obj.active_session_id = None
        chunk = StreamChunk(content="service", is_final=True)
        constraints = object()

        async def _stream_chat(user_message: str, **kwargs):
            assert user_message == "hello"
            assert kwargs == {"mode": "compat", "constraints": constraints}
            yield chunk

        obj._chat_service = SimpleNamespace(stream_chat=_stream_chat)

        chunks = [c async for c in obj.stream_chat("hello", mode="compat", constraints=constraints)]

        assert chunks == [chunk]

    @pytest.mark.asyncio
    async def test_orchestrator_stream_chat_does_not_inject_legacy_loop_kwargs(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj.active_session_id = None
        calls = []

        async def _stream_chat(user_message: str, **kwargs):
            calls.append((user_message, kwargs))
            yield StreamChunk(content="service", is_final=True)

        obj._chat_service = SimpleNamespace(stream_chat=_stream_chat)

        chunks = [c async for c in obj.stream_chat("hello")]

        assert [chunk.content for chunk in chunks] == ["service"]
        assert calls == [("hello", {})]

    @pytest.mark.asyncio
    async def test_orchestrator_stream_chat_propagates_chat_service_errors(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj.active_session_id = None

        async def _stream_chat(_user_message: str, **_kwargs):
            raise RuntimeError("stream failed")
            yield  # pragma: no cover

        obj._chat_service = SimpleNamespace(stream_chat=_stream_chat)

        with pytest.raises(RuntimeError, match="stream failed"):
            _ = [c async for c in obj.stream_chat("hello")]

    def test_prepare_chat_service_turn_runtime_applies_stream_skills_and_constraints(
        self,
    ):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj._apply_skill_for_turn = MagicMock()
        obj._constraint_activator = MagicMock()
        obj.vertical = "coding"

        constraints = object()
        AgentOrchestrator._prepare_chat_service_turn_runtime(
            obj,
            "hello",
            stream=True,
            constraints=constraints,
            vertical="review",
        )

        obj._apply_skill_for_turn.assert_called_once_with("hello")
        obj._constraint_activator.activate_constraints.assert_called_once_with(
            constraints=constraints,
            vertical="review",
        )

    def test_prepare_chat_service_turn_runtime_skips_non_stream_skill_activation(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj._apply_skill_for_turn = MagicMock()
        obj._constraint_activator = MagicMock()
        obj.vertical = "coding"

        AgentOrchestrator._prepare_chat_service_turn_runtime(
            obj,
            "hello",
            stream=False,
            constraints=None,
            vertical=None,
        )

        obj._apply_skill_for_turn.assert_not_called()
        obj._constraint_activator.activate_constraints.assert_not_called()

    def test_teardown_chat_service_turn_runtime_deactivates_constraints_when_present(
        self,
    ):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj._constraint_activator = MagicMock()

        AgentOrchestrator._teardown_chat_service_turn_runtime(
            obj,
            "hello",
            stream=True,
            constraints=object(),
            vertical="review",
        )

        obj._constraint_activator.deactivate_constraints.assert_called_once_with()

    def test_teardown_chat_service_turn_runtime_skips_empty_constraints(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj._constraint_activator = MagicMock()

        AgentOrchestrator._teardown_chat_service_turn_runtime(
            obj,
            "hello",
            stream=True,
            constraints=None,
            vertical=None,
        )

        obj._constraint_activator.deactivate_constraints.assert_not_called()

    def test_apply_skill_for_turn_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._skill_runtime = helper

        AgentOrchestrator._apply_skill_for_turn(obj, "hello")

        helper.apply_skill_for_turn.assert_called_once_with("hello")

    def test_get_last_skill_match_info_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.get_last_skill_match_info.return_value = {"auto_skill": "debug"}
        obj._skill_runtime = helper

        result = AgentOrchestrator.get_last_skill_match_info(obj)

        assert result == {"auto_skill": "debug"}
        helper.get_last_skill_match_info.assert_called_once_with()

    def test_clear_active_skills_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._skill_runtime = helper

        AgentOrchestrator.clear_active_skills(obj)

        helper.clear_active_skills.assert_called_once_with()

    def test_get_skill_user_prefix_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.get_skill_user_prefix.return_value = "ACTIVE SKILL: debug"
        obj._skill_runtime = helper

        result = AgentOrchestrator.get_skill_user_prefix(obj)

        assert result == "ACTIVE SKILL: debug"
        helper.get_skill_user_prefix.assert_called_once_with()

    def test_inject_skill_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._skill_runtime = helper
        skill = SimpleNamespace(name="debug")

        AgentOrchestrator.inject_skill(obj, skill)

        helper.inject_skill.assert_called_once_with(skill)

    def test_inject_skills_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._skill_runtime = helper
        skills = [(SimpleNamespace(name="debug"), 0.9)]

        AgentOrchestrator.inject_skills(obj, skills)

        helper.inject_skills.assert_called_once_with(skills)

    def test_sync_prompt_builder_runtime_state_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._prompt_builder_runtime = helper

        AgentOrchestrator._sync_prompt_builder_runtime_state(obj)

        helper.sync_prompt_builder_runtime_state.assert_called_once_with()

    def test_build_system_prompt_uses_helper_before_pipeline(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.build_system_prompt_fallback.return_value = "fallback prompt"
        obj._prompt_builder_runtime = helper
        obj._prompt_pipeline = MagicMock()
        obj._prompt_pipeline.build_system_prompt.return_value = "pipeline prompt"

        result = AgentOrchestrator.build_system_prompt(obj)

        assert result == "pipeline prompt"
        helper.sync_prompt_builder_runtime_state.assert_called_once_with()
        obj._prompt_pipeline.build_system_prompt.assert_called_once_with()
        helper.build_system_prompt_fallback.assert_not_called()

    def test_build_system_prompt_delegates_fallback_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.build_system_prompt_fallback.return_value = "fallback prompt"
        obj._prompt_builder_runtime = helper
        obj._prompt_pipeline = None

        result = AgentOrchestrator.build_system_prompt(obj)

        assert result == "fallback prompt"
        helper.sync_prompt_builder_runtime_state.assert_called_once_with()
        helper.build_system_prompt_fallback.assert_called_once_with()

    def test_update_system_prompt_for_query_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._prompt_builder_runtime = helper

        AgentOrchestrator.update_system_prompt_for_query(obj, query_classification="coding")

        helper.update_system_prompt_for_query.assert_called_once_with(query_classification="coding")

    def test_refresh_system_prompt_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._prompt_builder_runtime = helper

        AgentOrchestrator.refresh_system_prompt(
            obj,
            query_classification="coding",
            preserve_existing_classification=False,
        )

        helper.refresh_system_prompt.assert_called_once_with(
            query_classification="coding",
            preserve_existing_classification=False,
        )

    def test_get_system_prompt_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        helper.get_system_prompt.return_value = "system prompt"
        obj._prompt_builder_runtime = helper

        result = AgentOrchestrator.get_system_prompt(obj)

        assert result == "system prompt"
        helper.get_system_prompt.assert_called_once_with()

    def test_set_system_prompt_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._prompt_builder_runtime = helper

        AgentOrchestrator.set_system_prompt(obj, "new system prompt")

        helper.set_system_prompt.assert_called_once_with("new system prompt")

    def test_append_to_system_prompt_delegates_to_helper(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        helper = MagicMock()
        obj._prompt_builder_runtime = helper

        AgentOrchestrator.append_to_system_prompt(obj, "extra guidance")

        helper.append_to_system_prompt.assert_called_once_with("extra guidance")

    @pytest.mark.asyncio
    async def test_orchestrator_chat_delegates_to_chat_service(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj.active_session_id = None
        response = CompletionResponse(content="done", role="assistant")
        obj._chat_service = SimpleNamespace(chat=AsyncMock(return_value=response))

        result = await obj.chat("hello")

        assert result is response
        obj._chat_service.chat.assert_awaited_once_with("hello", use_planning=False)

    @pytest.mark.asyncio
    async def test_orchestrator_chat_passes_use_planning_through_service(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj.active_session_id = None
        response = CompletionResponse(content="planned", role="assistant")
        obj._chat_service = SimpleNamespace(chat=AsyncMock(return_value=response))

        result = await obj.chat("hello", use_planning=True)

        assert result is response
        obj._chat_service.chat.assert_awaited_once_with("hello", use_planning=True)

    @pytest.mark.asyncio
    async def test_orchestrator_chat_forwards_runtime_kwargs_to_chat_service(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj.active_session_id = None
        response = CompletionResponse(content="done", role="assistant")
        obj._chat_service = SimpleNamespace(chat=AsyncMock(return_value=response))

        constraints = object()
        result = await obj.chat(
            "hello",
            use_planning=True,
            constraints=constraints,
            vertical="review",
            stream=True,
        )

        assert result is response
        obj._chat_service.chat.assert_awaited_once_with(
            "hello",
            use_planning=True,
            constraints=constraints,
            vertical="review",
            stream=True,
        )

    @pytest.mark.asyncio
    async def test_orchestrator_chat_propagates_chat_service_errors(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        obj.active_session_id = None
        obj._chat_service = SimpleNamespace(chat=AsyncMock(side_effect=RuntimeError("chat failed")))

        with pytest.raises(RuntimeError, match="chat failed"):
            await obj.chat("hello")

    def test_get_chat_stream_adapter_prefers_factory_and_caches(self):
        from victor.agent.orchestrator import AgentOrchestrator

        obj = object.__new__(AgentOrchestrator)
        adapter = MagicMock(name="protocol_adapter")
        chat_stream_adapter = MagicMock(name="chat_stream_adapter")
        obj._protocol_adapter = adapter
        obj._factory = MagicMock()
        obj._factory.create_streaming_chat_adapter.return_value = chat_stream_adapter

        first = obj._get_chat_stream_adapter()
        second = obj._get_chat_stream_adapter()

        assert first is chat_stream_adapter
        assert second is chat_stream_adapter
        obj._factory.create_streaming_chat_adapter.assert_called_once_with(adapter)

    def test_orchestrator_protocol_adapter_proxies_streaming_runtime_host_state(self):
        from types import SimpleNamespace

        from victor.agent.services.orchestrator_protocol_adapter import (
            OrchestratorProtocolAdapter,
        )

        orchestrator = SimpleNamespace(
            _current_stream_context="ctx",
            _container=MagicMock(name="container"),
        )
        adapter = OrchestratorProtocolAdapter(orchestrator)

        assert adapter._current_stream_context == "ctx"
        assert adapter._container is orchestrator._container

        adapter._runtime_intelligence = "runtime-intelligence"
        assert orchestrator._runtime_intelligence == "runtime-intelligence"

        del adapter._current_stream_context
        assert not hasattr(orchestrator, "_current_stream_context")
