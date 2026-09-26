"""Focused tests for service runtime parity helpers."""

import logging
from types import SimpleNamespace
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.services.session_service import SessionService
from victor.agent.services.provider_service import ProviderService
from victor.agent.services.recovery_service import RecoveryService
from victor.agent.services.tool_service import ToolService, ToolServiceConfig
from victor.providers.base import StreamChunk


def _make_tool_service() -> ToolService:
    return ToolService(
        config=ToolServiceConfig(),
        tool_selector=MagicMock(),
        tool_executor=MagicMock(),
        tool_registrar=MagicMock(),
    )


@pytest.mark.asyncio
async def test_tool_service_execute_tool_with_retry_uses_bound_retry_executor():
    service = _make_tool_service()
    retry_executor = MagicMock()
    retry_executor.execute_tool_with_retry = AsyncMock(return_value=("result", True, None))
    service.bind_runtime_components(retry_executor=retry_executor)

    result = await service.execute_tool_with_retry("read", {"path": "a.py"}, {"task_type": "read"})

    retry_executor.execute_tool_with_retry.assert_awaited_once()
    assert result == ("result", True, None)


@pytest.mark.asyncio
async def test_tool_service_execute_tool_with_retry_uses_service_owned_retry_runtime():
    service = _make_tool_service()
    pipeline = MagicMock()
    pipeline._execute_single_call = AsyncMock(
        return_value=SimpleNamespace(success=True, error=None)
    )
    # The service-owned retry runtime consults the pipeline's registry to read
    # the tool's declared access mode before authorizing replay; "read" is a
    # declared READONLY tool, so the retry must reach the single-call executor.
    from victor.tools.enums import AccessMode

    read_tool = MagicMock()
    read_tool.access_mode = AccessMode.READONLY
    pipeline.tools.get.return_value = read_tool

    service.bind_runtime_components(tool_pipeline=pipeline)

    result = await service.execute_tool_with_retry("read", {"path": "a.py"}, {"task_type": "read"})

    pipeline._execute_single_call.assert_awaited_once_with(
        {"name": "read", "arguments": {"path": "a.py"}},
        {"task_type": "read"},
    )
    assert result == (pipeline._execute_single_call.return_value, True, None)


def test_tool_service_parse_and_validate_tool_calls_matches_runtime_contract():
    service = _make_tool_service()
    parser = MagicMock()
    parser.normalize_args.side_effect = lambda _name, args: args
    service.bind_runtime_components(tool_call_parser=parser)
    service.set_enabled_tools({"read"})

    tool_call = SimpleNamespace(to_dict=lambda: {"name": "read", "arguments": '{"path":"a.py"}'})
    parse_result = SimpleNamespace(
        tool_calls=[tool_call],
        warnings=[],
        remaining_content="trimmed content",
    )
    tool_adapter = MagicMock()
    tool_adapter.parse_tool_calls.return_value = parse_result

    tool_calls, remaining = service.parse_and_validate_tool_calls(None, "content", tool_adapter)

    assert tool_calls == [{"name": "read", "arguments": {"path": "a.py"}}]
    assert remaining == "trimmed content"


def test_tool_service_records_fallback_parse_diagnostics(caplog):
    service = _make_tool_service()
    parser = MagicMock()
    parser.normalize_args.side_effect = lambda _name, args: args
    service.bind_runtime_components(tool_call_parser=parser)
    service.set_enabled_tools({"read"})

    tool_call = SimpleNamespace(to_dict=lambda: {"name": "read", "arguments": {"path": "a.py"}})
    parse_result = SimpleNamespace(
        tool_calls=[tool_call],
        warnings=["native parse empty"],
        remaining_content="trimmed content",
        parse_method="json_fallback",
        confidence=0.82,
    )
    tool_adapter = MagicMock()
    tool_adapter.provider_name = "openai"
    tool_adapter.model = "qwen2.5-coder:1.5b"
    tool_adapter.parse_tool_calls.return_value = parse_result

    with caplog.at_level(logging.INFO):
        service.parse_and_validate_tool_calls(None, "content", tool_adapter)

    diagnostics = service.get_last_tool_call_parse_diagnostics()
    assert diagnostics is not None
    assert diagnostics["provider_name"] == "openai"
    assert diagnostics["model_name"] == "qwen2.5-coder:1.5b"
    assert diagnostics["parse_method"] == "json_fallback"
    assert diagnostics["adapter_parse_invoked"] is True
    assert diagnostics["raw_tool_call_count"] == 0
    assert diagnostics["extracted_tool_call_count"] == 1
    assert diagnostics["final_tool_call_count"] == 1
    assert diagnostics["warning_count"] == 1
    assert diagnostics["fallback_used"] is True
    assert "tool_call_parse provider=openai model=qwen2.5-coder:1.5b" in caplog.text
    assert "parse_method=json_fallback" in caplog.text


def test_tool_service_records_native_passthrough_parse_diagnostics(caplog):
    service = _make_tool_service()
    parser = MagicMock()
    parser.normalize_args.side_effect = lambda _name, args: args
    service.bind_runtime_components(tool_call_parser=parser)
    service.set_enabled_tools({"write"})

    tool_adapter = MagicMock()
    tool_adapter.provider_name = "zai"
    tool_adapter.model = "glm-5.1"
    raw_tool_calls = [{"name": "write", "arguments": {"path": "notes.txt", "content": "hello"}}]

    with caplog.at_level(logging.INFO):
        tool_calls, remaining = service.parse_and_validate_tool_calls(
            raw_tool_calls,
            "",
            tool_adapter,
        )

    assert tool_calls == raw_tool_calls
    assert remaining == ""
    diagnostics = service.get_last_tool_call_parse_diagnostics()
    assert diagnostics is not None
    assert diagnostics["provider_name"] == "zai"
    assert diagnostics["model_name"] == "glm-5.1"
    assert diagnostics["parse_method"] == "native_passthrough"
    assert diagnostics["adapter_parse_invoked"] is False
    assert diagnostics["raw_tool_call_count"] == 1
    assert diagnostics["extracted_tool_call_count"] == 1
    assert diagnostics["final_tool_call_count"] == 1
    assert diagnostics["fallback_used"] is False
    assert diagnostics["native_passthrough"] is True
    assert "tool_call_parse provider=zai model=glm-5.1" in caplog.text
    assert "parse_method=native_passthrough" in caplog.text


def test_tool_service_validate_tool_call_matches_legacy_contract():
    service = _make_tool_service()
    service.bind_runtime_components(
        tool_registry=SimpleNamespace(get_registered_tools=lambda: {"read"})
    )
    sanitizer = SimpleNamespace(is_valid_tool_name=lambda name: name == "read")

    validation = service.validate_tool_call({"name": "read", "arguments": {}}, sanitizer)

    assert validation.valid is True
    assert validation.canonical_name == "read"


def test_tool_service_bound_pipeline_budget_is_authoritative():
    service = _make_tool_service()

    class _PipelineBudgetStub:
        def __init__(self) -> None:
            self.config = SimpleNamespace(tool_budget=9)
            self._calls_used = 4

        @property
        def tool_budget(self) -> int:
            return self.config.tool_budget

        @property
        def calls_used(self) -> int:
            return self._calls_used

        def set_tool_budget(self, budget: int) -> None:
            self.config.tool_budget = budget

        def start_new_turn(self) -> None:
            self._calls_used = 0

        def consume_budget(self, amount: int = 1) -> None:
            self._calls_used += amount

    pipeline = _PipelineBudgetStub()
    service.bind_runtime_components(tool_pipeline=pipeline)

    assert service.budget == 9
    assert service.budget_used == 4
    assert service.get_remaining_budget() == 5
    assert service.get_tool_budget() == 5

    service.set_tool_budget(12)
    assert pipeline.config.tool_budget == 12
    assert service.budget == 12

    service.start_new_turn()
    assert pipeline.calls_used == 0
    assert service.budget_used == 0
    assert service.get_remaining_budget() == 12


def test_tool_retry_executor_available_from_services():
    """ToolRetryExecutor should be available from services module."""
    from victor.agent.services.tool_retry import ToolRetryExecutor

    assert ToolRetryExecutor is not None


def test_tool_coordinator_contracts_available_from_services():
    """Tool contract DTOs should be available from services module."""
    from victor.agent.services.tool_contracts import (
        NormalizedArgs,
        ToolCallValidation,
    )

    assert NormalizedArgs is not None
    assert ToolCallValidation is not None


def test_orchestrator_protocol_adapter_available_from_services():
    """OrchestratorProtocolAdapter should be available from services module."""
    from victor.agent.services.orchestrator_protocol_adapter import (
        OrchestratorProtocolAdapter,
    )

    assert OrchestratorProtocolAdapter is not None


def test_service_rl_runtime_prompt_rollout_helper_uses_global_coordinator():
    from victor.agent.services.rl_runtime import create_prompt_rollout_experiment

    coordinator = MagicMock()
    coordinator.create_prompt_rollout_experiment.return_value = "prompt_exp_service"

    with patch("victor.agent.services.rl_runtime.get_rl_coordinator", return_value=coordinator):
        experiment_id = create_prompt_rollout_experiment(
            section_name="GROUNDING_RULES",
            provider="anthropic",
            treatment_hash="candidate_hash",
            traffic_split=0.2,
            min_samples_per_variant=25,
        )

    assert experiment_id == "prompt_exp_service"
    coordinator.create_prompt_rollout_experiment.assert_called_once_with(
        section_name="GROUNDING_RULES",
        provider="anthropic",
        treatment_hash="candidate_hash",
        control_hash=None,
        traffic_split=0.2,
        min_samples_per_variant=25,
    )


@pytest.mark.asyncio
async def test_service_rl_runtime_prompt_rollout_async_helper_uses_global_coordinator():
    from victor.agent.services.rl_runtime import create_prompt_rollout_experiment_async

    coordinator = MagicMock()
    coordinator.create_prompt_rollout_experiment_async = AsyncMock(
        return_value="prompt_exp_service_async"
    )

    with patch(
        "victor.agent.services.rl_runtime.get_rl_coordinator_async",
        new=AsyncMock(return_value=coordinator),
    ):
        experiment_id = await create_prompt_rollout_experiment_async(
            section_name="GROUNDING_RULES",
            provider="anthropic",
            treatment_hash="candidate_hash",
            traffic_split=0.2,
            min_samples_per_variant=25,
        )

    assert experiment_id == "prompt_exp_service_async"
    coordinator.create_prompt_rollout_experiment_async.assert_called_once_with(
        section_name="GROUNDING_RULES",
        provider="anthropic",
        treatment_hash="candidate_hash",
        control_hash=None,
        traffic_split=0.2,
        min_samples_per_variant=25,
    )


def test_service_rl_runtime_rollout_analysis_helper_uses_global_coordinator():
    from victor.agent.services.rl_runtime import analyze_prompt_rollout_experiment

    coordinator = MagicMock()
    coordinator.analyze_prompt_rollout_experiment.return_value = {"auto_action": "rollout"}

    with patch("victor.agent.services.rl_runtime.get_rl_coordinator", return_value=coordinator):
        report = analyze_prompt_rollout_experiment(
            section_name="GROUNDING_RULES",
            provider="anthropic",
            treatment_hash="candidate_hash",
        )

    assert report == {"auto_action": "rollout"}
    coordinator.analyze_prompt_rollout_experiment.assert_called_once_with(
        section_name="GROUNDING_RULES",
        provider="anthropic",
        treatment_hash="candidate_hash",
    )


def test_service_rl_runtime_suite_processing_helper_uses_global_coordinator():
    from victor.agent.services.rl_runtime import (
        process_prompt_candidate_evaluation_suite,
    )

    coordinator = MagicMock()
    coordinator.process_prompt_candidate_evaluation_suite.return_value = SimpleNamespace(
        to_dict=lambda: {"prompt_rollout": {"created": True}}
    )

    with patch("victor.agent.services.rl_runtime.get_rl_coordinator", return_value=coordinator):
        workflow = process_prompt_candidate_evaluation_suite(
            {"runs": []},
            create_rollout=True,
        )

    assert workflow == {"prompt_rollout": {"created": True}}
    coordinator.process_prompt_candidate_evaluation_suite.assert_called_once_with(
        {"runs": []},
        min_pass_rate=0.5,
        promote_best=False,
        create_rollout=True,
        rollout_control_hash=None,
        rollout_traffic_split=0.1,
        rollout_min_samples_per_variant=100,
        analyze_rollout=False,
        apply_rollout_decision=False,
        rollout_decision_dry_run=False,
    )


@pytest.mark.asyncio
async def test_service_rl_runtime_rollout_apply_async_helper_uses_global_coordinator():
    from victor.agent.services.rl_runtime import (
        apply_prompt_rollout_recommendation_async,
    )

    coordinator = MagicMock()
    coordinator.apply_prompt_rollout_recommendation_async = AsyncMock(
        return_value={"action": "rollout", "applied": True}
    )

    with patch(
        "victor.agent.services.rl_runtime.get_rl_coordinator_async",
        new=AsyncMock(return_value=coordinator),
    ):
        decision = await apply_prompt_rollout_recommendation_async(
            section_name="GROUNDING_RULES",
            provider="anthropic",
            treatment_hash="candidate_hash",
            dry_run=True,
        )

    assert decision == {"action": "rollout", "applied": True}
    coordinator.apply_prompt_rollout_recommendation_async.assert_called_once_with(
        section_name="GROUNDING_RULES",
        provider="anthropic",
        treatment_hash="candidate_hash",
        dry_run=True,
    )


@pytest.mark.asyncio
async def test_service_rl_runtime_suite_processing_async_helper_uses_global_coordinator():
    from victor.agent.services.rl_runtime import (
        process_prompt_candidate_evaluation_suite_async,
    )

    coordinator = MagicMock()
    coordinator.process_prompt_candidate_evaluation_suite_async = AsyncMock(
        return_value=SimpleNamespace(to_dict=lambda: {"prompt_rollout_analysis": {"ok": True}})
    )

    with patch(
        "victor.agent.services.rl_runtime.get_rl_coordinator_async",
        new=AsyncMock(return_value=coordinator),
    ):
        workflow = await process_prompt_candidate_evaluation_suite_async(
            {"runs": []},
            analyze_rollout=True,
        )

    assert workflow == {"prompt_rollout_analysis": {"ok": True}}
    coordinator.process_prompt_candidate_evaluation_suite_async.assert_called_once_with(
        {"runs": []},
        min_pass_rate=0.5,
        promote_best=False,
        create_rollout=False,
        rollout_control_hash=None,
        rollout_traffic_split=0.1,
        rollout_min_samples_per_variant=100,
        analyze_rollout=True,
        apply_rollout_decision=False,
        rollout_decision_dry_run=False,
    )


def test_agent_services_package_exports_prompt_rollout_helpers():
    from victor.agent.services import (
        analyze_prompt_rollout_experiment as package_rollout_analysis_helper,
        apply_prompt_rollout_recommendation_async as package_rollout_apply_helper_async,
        process_prompt_candidate_evaluation_suite as package_suite_helper,
        process_prompt_candidate_evaluation_suite_async as package_suite_helper_async,
        create_prompt_rollout_experiment as package_rollout_helper,
        create_prompt_rollout_experiment_async as package_rollout_helper_async,
    )
    from victor.agent.services.rl_runtime import (
        analyze_prompt_rollout_experiment as runtime_rollout_analysis_helper,
        apply_prompt_rollout_recommendation_async as runtime_rollout_apply_helper_async,
        process_prompt_candidate_evaluation_suite as runtime_suite_helper,
        process_prompt_candidate_evaluation_suite_async as runtime_suite_helper_async,
        create_prompt_rollout_experiment as runtime_rollout_helper,
        create_prompt_rollout_experiment_async as runtime_rollout_helper_async,
    )

    assert package_rollout_helper is runtime_rollout_helper
    assert package_rollout_helper_async is runtime_rollout_helper_async
    assert package_rollout_analysis_helper is runtime_rollout_analysis_helper
    assert package_rollout_apply_helper_async is runtime_rollout_apply_helper_async
    assert package_suite_helper is runtime_suite_helper
    assert package_suite_helper_async is runtime_suite_helper_async


def test_tool_service_normalize_arguments_full_matches_legacy_contract():
    service = _make_tool_service()
    argument_normalizer = SimpleNamespace(
        normalize_arguments=lambda args, _tool_name: (args, "direct")
    )
    tool_adapter = SimpleNamespace(normalize_arguments=lambda args, _tool_name: args)

    normalized = service.normalize_arguments_full(
        "read",
        "read",
        '{"path":"a.py"}',
        argument_normalizer,
        tool_adapter,
        failed_signatures={("read", '{"path": "b.py"}')},
    )

    assert normalized.args == {"path": "a.py"}
    assert normalized.strategy == "direct"
    assert normalized.signature == ("read", '{"path":"a.py"}')
    assert normalized.is_repeated_failure is False


@pytest.mark.asyncio
async def test_tool_service_on_tool_complete_uses_canonical_completion_flow():
    service = _make_tool_service()
    metrics_collector = MagicMock()
    observability = MagicMock()
    add_message = MagicMock()
    bus = MagicMock()
    bus.emit = AsyncMock()
    read_files_session = set()
    nudge_flag = [False]
    result = SimpleNamespace(
        tool_name="read",
        success=True,
        result="print('hello')\n",
        error=None,
        arguments={"path": "main.py"},
        execution_time_ms=42.0,
    )

    with patch("victor.core.events.get_observability_bus", return_value=bus):
        service.on_tool_complete(
            result=result,
            metrics_collector=metrics_collector,
            read_files_session=read_files_session,
            required_files=["main.py"],
            required_outputs=["summary"],
            nudge_sent_flag=nudge_flag,
            add_message=add_message,
            observability=observability,
            iteration_count=2,
        )
        await asyncio.sleep(0)

    metrics_collector.on_tool_complete.assert_called_once_with(result)
    observability.on_tool_end.assert_called_once()
    assert "main.py" in read_files_session
    assert nudge_flag[0] is True
    add_message.assert_called_once()
    assert bus.emit.await_count >= 1


def test_tool_service_build_tool_access_context_uses_bound_mode_controller():
    service = _make_tool_service()
    mode_controller = SimpleNamespace(config=SimpleNamespace(name="review"))
    service.bind_runtime_components(mode_controller=mode_controller)
    service.set_enabled_tools({"read", "grep"})

    context = service.build_tool_access_context()

    assert context.session_enabled_tools == {"read", "grep"}
    assert context.current_mode == "review"


@pytest.mark.asyncio
async def test_session_service_save_and_restore_checkpoint_round_trip():
    session_state = SimpleNamespace(
        tool_calls_used=7,
        observed_files={"a.py"},
        get_token_usage=lambda: {"input": 12, "output": 8},
    )
    checkpoint_manager = MagicMock()
    checkpoint_manager.save_checkpoint = AsyncMock(return_value="ckpt-123")
    checkpoint_manager.restore_checkpoint = AsyncMock(
        return_value={
            "session_id": "mem-1",
            "tool_calls_used": 3,
            "token_usage": {"input": 20, "output": 10},
            "observed_files": ["b.py"],
        }
    )

    service = SessionService(
        session_state_manager=session_state,
        memory_manager=None,
        checkpoint_manager=checkpoint_manager,
    )
    service._memory_session_id = "mem-1"

    checkpoint_id = await service.save_checkpoint("before fix", ["manual"])
    restored = await service.restore_checkpoint("ckpt-123")

    checkpoint_manager.save_checkpoint.assert_awaited_once()
    checkpoint_manager.restore_checkpoint.assert_awaited_once_with("ckpt-123")
    assert checkpoint_id == "ckpt-123"
    assert restored is True
    assert service._session_state.tool_calls_used == 3
    assert service._session_state.observed_files == {"b.py"}


@pytest.mark.asyncio
async def test_provider_service_switch_provider_uses_bound_provider_manager():
    manager = MagicMock()
    manager.switch_provider = AsyncMock(return_value=True)
    manager.provider = SimpleNamespace(name="openai", model="gpt-4.1", max_tokens=200000)
    manager.provider_name = "openai"
    manager.model = "gpt-4.1"
    manager.switch_count = 2

    service = ProviderService(registry=MagicMock())
    service.bind_runtime_components(provider_manager=manager)

    await service.switch_provider("openai", "gpt-4.1")

    manager.switch_provider.assert_awaited_once_with("openai", "gpt-4.1")
    assert service.provider_name == "openai"
    assert service.model == "gpt-4.1"
    assert service.get_current_provider() is manager.provider


@pytest.mark.asyncio
async def test_recovery_service_streaming_methods_fallback_to_bound_recovery_coordinator():
    service = RecoveryService()
    coordinator = MagicMock()
    coordinator.handle_recovery_with_integration = AsyncMock(
        return_value=SimpleNamespace(action="continue")
    )
    coordinator.check_natural_completion.return_value = "done"
    coordinator.handle_empty_response.return_value = ("chunk", True)
    coordinator.get_recovery_fallback_message.return_value = "fallback"
    coordinator.check_tool_budget.return_value = "warn"
    coordinator.filter_blocked_tool_calls.return_value = ([{"name": "read"}], [], 0)
    coordinator.check_blocked_threshold.return_value = None

    service.bind_runtime_components(recovery_coordinator=coordinator)

    ctx = SimpleNamespace()

    recovery_action = await service.handle_recovery_with_integration(
        ctx,
        "content",
        [{"name": "read"}],
    )

    assert recovery_action.action == "continue"
    coordinator.handle_recovery_with_integration.assert_awaited_once()
    assert service.check_natural_completion(ctx, False, 0) == "done"
    assert service.handle_empty_response(ctx) == ("chunk", True)
    assert service.get_recovery_fallback_message(ctx) == "fallback"
    assert service.check_tool_budget(ctx, 10) == "warn"
    assert service.filter_blocked_tool_calls(ctx, [{"name": "read"}]) == (
        [{"name": "read"}],
        [],
        0,
    )
    assert service.check_blocked_threshold(ctx, False) is None
    assert service.check_force_action(ctx) == (False, None)

    message_adder = MagicMock()
    retry_action = SimpleNamespace(
        action="retry",
        reason="retry",
        message="retry now",
        new_temperature=None,
        failure_type=None,
    )
    assert service.apply_recovery_action(retry_action, ctx, message_adder=message_adder) is None
    message_adder.assert_called_once_with("user", "retry now")
    assert service.truncate_tool_calls(ctx, [{"name": "read"}, {"name": "grep"}], 1) == (
        [{"name": "read"}],
        True,
    )


@pytest.mark.asyncio
async def test_recovery_service_streaming_methods_use_native_runtime_components():
    service = RecoveryService()
    recovery_integration = MagicMock()
    recovery_integration.enabled = True
    recovery_integration.handle_response = AsyncMock(
        return_value=SimpleNamespace(
            action="continue",
            reason="ok",
            failure_type=None,
            strategy_name=None,
            message=None,
            new_temperature=None,
        )
    )
    streaming_handler = MagicMock()
    streaming_handler.check_natural_completion.return_value = True
    streaming_handler.handle_empty_response.return_value = SimpleNamespace(
        chunks=[StreamChunk(content="recover-empty")]
    )
    streaming_handler.check_tool_budget.return_value = SimpleNamespace(
        chunks=[StreamChunk(content="warn")]
    )
    streaming_handler.filter_blocked_tool_calls.return_value = (
        [{"name": "read"}],
        [StreamChunk(content="blocked")],
        1,
    )
    streaming_handler.check_blocked_threshold.return_value = SimpleNamespace(
        chunks=[StreamChunk(content="threshold")],
        clear_tool_calls=True,
    )
    context_compactor = MagicMock()
    context_compactor.get_statistics.return_value = {"current_utilization": 0.42}
    unified_tracker = MagicMock()
    settings = SimpleNamespace(
        recovery_blocked_consecutive_threshold=4,
        recovery_blocked_total_threshold=6,
    )

    service.bind_runtime_components(
        recovery_integration=recovery_integration,
        streaming_handler=streaming_handler,
        context_compactor=context_compactor,
        unified_tracker=unified_tracker,
        settings=settings,
    )

    streaming_context = SimpleNamespace(
        total_iterations=3,
        max_total_iterations=7,
        force_completion=True,
    )
    ctx = SimpleNamespace(
        provider_name="openai",
        model="gpt-5.4",
        tool_calls_used=2,
        tool_budget=8,
        temperature=0.3,
        last_quality_score=0.8,
        unified_task_type=SimpleNamespace(value="analysis"),
        is_analysis_task=True,
        is_action_task=False,
        streaming_context=streaming_context,
        iteration=3,
    )

    recovery_action = await service.handle_recovery_with_integration(
        ctx,
        "content",
        [{"name": "read"}],
        mentioned_tools=["read"],
    )

    assert recovery_action.action == "continue"
    recovery_integration.handle_response.assert_awaited_once()
    assert service.has_native_streaming_runtime() is True
    assert service.check_natural_completion(ctx, False, 10).is_final is True
    assert service.handle_empty_response(ctx) == (
        streaming_handler.handle_empty_response.return_value.chunks[0],
        True,
    )
    assert service.check_tool_budget(ctx, 10).content == "warn"
    filtered, blocked_chunks, blocked_count = service.filter_blocked_tool_calls(
        ctx,
        [{"name": "read"}],
    )
    assert filtered == [{"name": "read"}]
    assert blocked_count == 1
    assert blocked_chunks[0].content == "blocked"
    threshold_chunk, should_clear = service.check_blocked_threshold(ctx, True)
    assert threshold_chunk.content == "threshold"
    assert should_clear is True
