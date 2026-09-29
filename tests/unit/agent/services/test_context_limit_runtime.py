from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from victor.agent.services.context_limit_runtime import ContextLimitRuntime
from victor.providers.base import CompletionResponse, Message


def _make_runtime_host():
    provider = SimpleNamespace(chat=AsyncMock())
    host = SimpleNamespace(
        _check_context_overflow=MagicMock(return_value=False),
        _conversation_controller=MagicMock(),
        _presentation=MagicMock(),
        _get_thinking_disabled_prompt=MagicMock(side_effect=lambda prompt: prompt),
        messages=[Message(role="user", content="existing")],
        provider=provider,
        model="test-model",
        temperature=0.2,
        max_tokens=4096,
        sanitizer=MagicMock(),
        add_message=MagicMock(),
        _record_runtime_intelligence_outcome=MagicMock(),
    )
    host._presentation.icon.return_value = "!"
    host.sanitizer.sanitize.side_effect = lambda text: text
    return host


@pytest.mark.asyncio
async def test_context_limit_runtime_returns_noop_within_limits():
    runtime_host = _make_runtime_host()
    runtime = ContextLimitRuntime(runtime_host)

    handled, chunk = await runtime.handle_limits("hello", 5, 1000, 1, 0.8)

    assert handled is False
    assert chunk is None
    runtime_host.provider.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_limit_runtime_compacts_history_before_forcing_summary():
    runtime_host = _make_runtime_host()
    runtime_host._check_context_overflow.return_value = True
    runtime_host._conversation_controller.smart_compact_history.return_value = 2
    runtime = ContextLimitRuntime(runtime_host)

    handled, chunk = await runtime.handle_limits("hello", 5, 1000, 1, 0.8)

    assert handled is False
    assert "Compacted history (2 messages)" in chunk.content
    runtime_host._conversation_controller.inject_compaction_context.assert_called_once_with()
    runtime_host.provider.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_limit_runtime_forces_summary_on_iteration_limit():
    runtime_host = _make_runtime_host()
    runtime_host.provider.chat.return_value = CompletionResponse(
        content="summary", role="assistant"
    )
    runtime = ContextLimitRuntime(runtime_host)

    handled, chunk = await runtime.handle_limits("hello", 5, 1000, 6, 0.8)

    assert handled is True
    assert chunk.content == "summary"
    assert chunk.is_final is True
    runtime_host.add_message.assert_called_once_with("assistant", "summary", metadata=ANY)
    runtime_host._record_runtime_intelligence_outcome.assert_called_once_with(
        success=True,
        quality_score=0.8,
        user_satisfied=True,
        completed=True,
    )


@pytest.mark.asyncio
async def test_final_summary_denial_is_not_recorded_as_success():
    from victor.core.errors import ProviderPolicyError

    host = _make_runtime_host()
    error = ProviderPolicyError("policy_blocked", receipt="a" * 32)
    host.provider.chat.side_effect = error
    with pytest.raises(ProviderPolicyError) as caught:
        await ContextLimitRuntime(host).handle_limits("hello", 5, 1000, 6, 0.8)
    assert caught.value is error
    host._record_runtime_intelligence_outcome.assert_not_called()
