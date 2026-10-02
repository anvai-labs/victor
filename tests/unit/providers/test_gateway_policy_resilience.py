"""Gateway enforcement cannot be escaped by resilience retries or failover."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from victor.providers.base import ProviderError
from victor.providers.resilience import ProviderRetryConfig, ResilientProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 503])
async def test_gateway_failure_never_retries_or_falls_back(status):
    primary = SimpleNamespace(
        name="gateway",
        extra_config={"gateway": {"url": "http://localhost:8787"}},
        chat=AsyncMock(side_effect=ProviderError("policy refusal", status_code=status)),
    )
    fallback = SimpleNamespace(name="direct", chat=AsyncMock(return_value="escaped"))
    provider = ResilientProvider(
        primary,
        fallback_providers=[fallback],
        retry_config=ProviderRetryConfig(max_retries=2, base_delay_seconds=0, jitter_factor=0),
    )
    with pytest.raises(ProviderError):
        await provider.chat([], model="test")
    assert primary.chat.await_count == 1
    fallback.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_stream_refusal_never_falls_back():
    async def stream(*args, **kwargs):
        raise ProviderError("policy_quarantined", status_code=403)
        yield

    primary = SimpleNamespace(
        name="gateway", extra_config={"gateway": {"url": "http://localhost:8787"}}, stream=stream
    )
    fallback = SimpleNamespace(name="direct", stream=Mock(side_effect=stream))
    provider = ResilientProvider(primary, fallback_providers=[fallback])
    with pytest.raises(ProviderError):
        async for _ in provider.stream([], model="test"):
            pass
    fallback.stream.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_failure_in_fallback_chain_stops_further_egress():
    primary = SimpleNamespace(
        name="primary",
        extra_config={"gateway": None},
        chat=AsyncMock(side_effect=ValueError("unavailable")),
    )
    gateway = SimpleNamespace(
        name="gateway",
        extra_config={"gateway": {"url": "http://localhost:8787"}},
        chat=AsyncMock(side_effect=ProviderError("policy refusal", status_code=403)),
    )
    last = SimpleNamespace(name="last", chat=AsyncMock(return_value="escaped"))
    provider = ResilientProvider(
        primary, fallback_providers=[gateway, last], retry_config=ProviderRetryConfig(max_retries=0)
    )
    with pytest.raises(ProviderError):
        await provider.chat([], model="test")
    gateway.chat.assert_awaited_once()
    last.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_direct_provider_fallback_remains_available():
    primary = SimpleNamespace(
        name="primary",
        extra_config={"gateway": None},
        chat=AsyncMock(side_effect=ValueError("unavailable")),
    )
    fallback = SimpleNamespace(name="fallback", chat=AsyncMock(return_value="ok"))
    provider = ResilientProvider(
        primary, fallback_providers=[fallback], retry_config=ProviderRetryConfig(max_retries=0)
    )
    assert await provider.chat([], model="test") == "ok"


@pytest.mark.asyncio
async def test_gateway_disables_client_transport_retries():
    from victor.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(
        gateway={"url": "http://localhost:8787", "virtual_key": "vk_test"}, max_retries=4
    )
    try:
        assert provider.max_retries == 0
        assert provider.client.max_retries == 0
    finally:
        await provider.client.close()


@pytest.mark.parametrize("code", ["policy_blocked", "policy_quarantined", "policy_unavailable"])
async def test_policy_error_mapping_preserves_safe_code_and_receipt(code):
    import json
    from victor.providers.sandhi_transport import map_sandhi_error

    error = RuntimeError(
        json.dumps(
            {
                "code": "upstream_error",
                "http_status": 503,
                "message": "sensitive upstream body",
                "details": {
                    "upstream_body": json.dumps(
                        {
                            "error": {
                                "code": code,
                                "request_id": "a" * 32,
                                "message": "sensitive upstream body",
                            }
                        }
                    )
                },
            }
        )
    )
    mapped = await map_sandhi_error(error, "openai", 30)
    assert mapped.details["policy_code"] == code
    assert mapped.details["policy_receipt"] == "a" * 32
    assert "sensitive" not in str(mapped)
    assert mapped.raw_error is None


async def test_malformed_upstream_policy_code_is_not_interpreted_as_a_decision():
    import json
    from victor.core.errors import ProviderPolicyError
    from victor.providers.sandhi_transport import map_sandhi_error

    error = RuntimeError(
        json.dumps(
            {
                "code": "upstream_error",
                "http_status": 500,
                "details": {"upstream_body": json.dumps({"error": {"code": []}})},
            }
        )
    )
    assert not isinstance(await map_sandhi_error(error, "openai", 30), ProviderPolicyError)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["retry_with_exponential_backoff", "retry_with_jitter"])
@pytest.mark.parametrize("code", ["policy_blocked", "policy_quarantined", "policy_unavailable"])
async def test_runtime_recovery_preserves_terminal_policy_decision(method, code):
    from victor.agent.services.recovery_service import RecoveryService, RecoveryContextImpl
    from victor.core.errors import ProviderPolicyError

    service = RecoveryService()
    service._base_retry_delay = 0.001
    error = ProviderPolicyError(code, receipt="a" * 32)
    operation = AsyncMock(side_effect=error)
    assert not service.can_retry(error, 0)
    assert await service.classify_error(error) == "policy"
    # Even stale classification and layer-based recovery cannot override a denial.
    context = RecoveryContextImpl(error, "unknown", 0, {}, {})
    service._layer_attribution_enabled = True
    assert await service.select_recovery_action(context) == "fail"
    with pytest.raises(ProviderPolicyError) as caught:
        await getattr(service, method)(operation)
    assert caught.value is error
    operation.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_recovery_still_retries_transient_errors():
    from victor.agent.services.recovery_service import RecoveryService

    service = RecoveryService()
    operation = AsyncMock(side_effect=[TimeoutError(), "ok"])
    assert await service.retry_with_exponential_backoff(operation, base_delay=0.001) == "ok"
    assert operation.await_count == 2


@pytest.mark.asyncio
async def test_response_completion_does_not_retry_denied_request():
    from victor.agent.response_completer import ResponseCompleter
    from victor.core.errors import ProviderPolicyError

    error = ProviderPolicyError("policy_blocked", receipt="b" * 32)
    provider = SimpleNamespace(chat=AsyncMock(side_effect=error))
    with pytest.raises(ProviderPolicyError) as caught:
        await ResponseCompleter(provider)._generate_response_with_retry([], "test", 0.5, 50)
    assert caught.value is error
    provider.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_subagent_does_not_replay_turn_after_policy_denial(monkeypatch):
    from victor.agent.subagents import SubAgent
    from victor.core.errors import ProviderPolicyError

    error = ProviderPolicyError("policy_blocked", receipt="c" * 32)
    member = SimpleNamespace(
        orchestrator=SimpleNamespace(chat=AsyncMock(side_effect=error)),
        config=SimpleNamespace(task="test", role=SimpleNamespace(value="researcher")),
        _presentation=SimpleNamespace(icon=lambda *args, **kwargs: ""),
    )
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    with pytest.raises(ProviderPolicyError) as caught:
        await SubAgent._execute_with_retry(member)
    assert caught.value is error
    member.orchestrator.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_agentic_loop_preserves_denial_instead_of_empty_result():
    from victor.framework.agentic_loop import AgenticLoop
    from victor.core.errors import ProviderPolicyError

    error = ProviderPolicyError("policy_blocked", receipt="d" * 32)
    loop = AgenticLoop(orchestrator=Mock(spec=[]), enable_fulfillment_check=False, max_iterations=1)
    loop.perception.perceive = AsyncMock(side_effect=error)
    with pytest.raises(ProviderPolicyError) as caught:
        await loop.run("test")
    assert caught.value is error
    loop.perception.perceive.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_failure_completion_preserves_policy_receipt():
    from victor.agent.response_completer import ResponseCompleter, ToolFailureContext
    from victor.core.errors import ProviderPolicyError

    error = ProviderPolicyError("policy_quarantined", receipt="e" * 32)
    provider = SimpleNamespace(chat=AsyncMock(side_effect=error))
    with pytest.raises(ProviderPolicyError) as caught:
        await ResponseCompleter(provider).ensure_response(
            [], "test", 0.5, 50, failure_context=ToolFailureContext(failed_tools=[{"name": "test"}])
        )
    assert caught.value is error
    provider.chat.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_real_stategraph_preserves_denial_through_act_and_runtime(streaming):
    from unittest.mock import MagicMock
    from victor.framework.agentic_graph.executor import AgenticLoopGraphExecutor
    from victor.core.errors import ProviderPolicyError

    error = ProviderPolicyError("policy_blocked", receipt="f" * 32)
    context = MagicMock()
    context.metadata = {}
    executor = AgenticLoopGraphExecutor(execution_context=context, max_iterations=1)
    executor.turn_executor = SimpleNamespace(execute_turn=AsyncMock(side_effect=error))
    with pytest.raises(ProviderPolicyError) as caught:
        if streaming:
            async for _ in executor.stream("test"):
                pass
        else:
            await executor.run("test")
    assert caught.value is error
    executor.turn_executor.execute_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_stategraph_loop_adapter_preserves_denial():
    from victor.framework.agentic_loop import AgenticLoop
    from victor.core.errors import ProviderPolicyError

    error = ProviderPolicyError("policy_blocked", receipt="f" * 32)
    loop = AgenticLoop(orchestrator=Mock(spec=[]), enable_fulfillment_check=False, max_iterations=1)
    loop._create_stategraph_executor = Mock(
        return_value=SimpleNamespace(run=AsyncMock(side_effect=error))
    )
    with pytest.raises(ProviderPolicyError) as caught:
        await loop._run_with_stategraph("test", None)
    assert caught.value is error


def test_policy_classification_cannot_enable_other_recovery_apis():
    from victor.agent.services.recovery_service import RecoveryService, RecoveryStrategy

    service = RecoveryService()
    assert not service.should_attempt_recovery("policy")
    assert service.select_recovery_strategy("policy", 0) is RecoveryStrategy.GIVE_UP
    service.get_model_fallbacks = Mock(return_value=["other"])
    assert not service.can_use_model_fallback("test", "policy")


@pytest.mark.asyncio
async def test_managed_gateway_completion_does_not_retry_transport_error():
    from victor.agent.response_completer import ResponseCompleter
    from victor.providers.factory import ManagedProvider
    from victor.core.errors import ProviderConnectionError

    error = ProviderConnectionError("interrupted")
    raw = SimpleNamespace(
        name="test", extra_config={"gateway": {}}, chat=AsyncMock(side_effect=error)
    )
    managed = ManagedProvider(raw)
    with pytest.raises(ProviderConnectionError):
        await ResponseCompleter(managed)._generate_response_with_retry([], "test", 0.5, 50)
    raw.chat.assert_awaited_once()


def test_gateway_boundary_detects_nested_wrappers_without_enabling_direct_mode():
    from victor.providers.gateway_boundary import uses_gateway
    from victor.providers.factory import ManagedProvider

    direct = SimpleNamespace(name="direct", extra_config={})
    assert not uses_gateway(ManagedProvider(direct))
    gateway = SimpleNamespace(name="gateway", extra_config={"gateway": {}})
    resilient = ResilientProvider(gateway)
    assert uses_gateway(ManagedProvider(gateway, resilient_provider=resilient))
    direct.provider = direct
    assert not uses_gateway(direct)


@pytest.mark.asyncio
async def test_managed_subagent_does_not_replay_gateway_transport_failure(monkeypatch):
    from victor.agent.subagents import SubAgent
    from victor.providers.factory import ManagedProvider
    from victor.core.errors import ProviderConnectionError

    error = ProviderConnectionError("interrupted")
    raw = SimpleNamespace(name="gateway", extra_config={"gateway": {}})
    member = SimpleNamespace(
        orchestrator=SimpleNamespace(
            provider=ManagedProvider(raw), chat=AsyncMock(side_effect=error)
        ),
        config=SimpleNamespace(task="test", role=SimpleNamespace(value="researcher")),
        _presentation=SimpleNamespace(icon=lambda *args, **kwargs: ""),
    )
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    with pytest.raises(ProviderConnectionError):
        await SubAgent._execute_with_retry(member)
    member.orchestrator.chat.assert_awaited_once()
