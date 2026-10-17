"""Runtime accumulation and metrics must share one live usage dictionary."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from victor.agent.session_state_accessor import SessionStateAccessor
from victor.agent.session_state_manager import SessionStateManager
from victor.agent.services.metrics_service import AgentMetricsService
from victor.agent.session_cost_tracker import SessionCostTracker
from victor.providers.base import CompletionResponse


async def test_runtime_usage_reaches_metrics_and_public_snapshot():
    state = SessionStateManager()
    accessor = SessionStateAccessor(state)
    metrics = AgentMetricsService(
        MagicMock(), SessionCostTracker(), accessor.cumulative_token_usage
    )
    for _ in range(2):
        response = CompletionResponse(
            content="ok",
            role="assistant",
            usage={
                "prompt_tokens": 19,
                "completion_tokens": 7,
                "total_tokens": 26,
                "cache_read_input_tokens": 4,
                "reasoning_tokens": 2,
            },
        )
        provider = SimpleNamespace(name="unknown", chat=AsyncMock(return_value=response))
        await metrics.complete(provider, messages=[], model="m")
    usage = metrics.get_token_usage()
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (38, 14, 52)
    assert (usage.cached_tokens, usage.reasoning_tokens) == (8, 4)
    snapshot = state.get_token_usage()
    snapshot["total_tokens"] = 0
    assert metrics.get_token_usage().total_tokens == 52
    accessor.cumulative_token_usage = {"prompt_tokens": 1, "total_tokens": 1}
    assert metrics.get_token_usage().total_tokens == 1
    accessor.cumulative_token_usage = accessor.cumulative_token_usage
    assert metrics.get_token_usage().total_tokens == 1


async def test_reset_and_restore_keep_runtime_and_metrics_on_one_accumulator():
    state = SessionStateManager()
    accessor = SessionStateAccessor(state)
    live_usage = accessor.cumulative_token_usage
    metrics = AgentMetricsService(MagicMock(), SessionCostTracker(), live_usage)
    response = CompletionResponse(
        content="ok",
        role="assistant",
        usage={"prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26},
    )
    provider = SimpleNamespace(name="unknown", chat=AsyncMock(return_value=response))
    await metrics.complete(provider, messages=[], model="m")
    checkpoint = state.get_checkpoint_state()

    state.reset(preserve_token_usage=True)
    assert accessor.cumulative_token_usage is live_usage
    await metrics.complete(provider, messages=[], model="m")
    assert state.get_token_usage()["total_tokens"] == metrics.get_token_usage().total_tokens == 52

    state.reset()
    assert accessor.cumulative_token_usage is live_usage
    assert state.get_token_usage()["total_tokens"] == metrics.get_token_usage().total_tokens == 0
    await metrics.complete(provider, messages=[], model="m")
    assert state.get_token_usage()["total_tokens"] == metrics.get_token_usage().total_tokens == 26

    state.apply_checkpoint_state(checkpoint)
    assert accessor.cumulative_token_usage is live_usage
    await metrics.complete(provider, messages=[], model="m")
    assert state.get_token_usage()["total_tokens"] == metrics.get_token_usage().total_tokens == 52
    # Neither runtime writes nor a caller changing its checkpoint can alter the other owner.
    assert checkpoint["execution_state"]["token_usage"]["total_tokens"] == 26
    checkpoint["execution_state"]["token_usage"]["total_tokens"] = 999
    assert metrics.get_token_usage().total_tokens == 52
    snapshot = state.get_token_usage()
    snapshot["total_tokens"] = -1
    assert metrics.get_token_usage().total_tokens == 52
