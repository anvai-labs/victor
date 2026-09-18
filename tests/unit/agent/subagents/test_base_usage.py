"""Opt-in usage capture preserves neutral counters and default payloads."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.subagents.base import SubAgent, SubAgentConfig, SubAgentRole
from victor.evaluation.protocol import TokenUsage


@pytest.mark.parametrize("capture", [False, True])
async def test_usage_capture_is_opt_in(capture):
    config = SubAgentConfig(
        role=SubAgentRole.EXECUTOR,
        task="work",
        allowed_tools=[],
        tool_budget=1,
        context_limit=1000,
        capture_usage=capture,
    )
    member = SubAgent(config, MagicMock())
    member.orchestrator = MagicMock(tool_calls_used=2)
    member.orchestrator.get_messages.return_value = []
    member.orchestrator.get_token_usage.return_value = TokenUsage(
        input_tokens=17, output_tokens=5, total_tokens=22
    )
    member._execute_with_retry = AsyncMock(
        return_value=SimpleNamespace(content="done", tool_calls=[], metadata={})
    )
    member._run_context_lifecycle = AsyncMock(return_value={})
    result = await member.execute()
    assert result.success
    if capture:
        assert result.details["usage"]["total_tokens"] == 22
        member.orchestrator.get_token_usage.assert_called_once()
    else:
        assert "usage" not in result.details
        member.orchestrator.get_token_usage.assert_not_called()


async def test_usage_capture_rejects_missing_contract():
    config = SubAgentConfig(
        role=SubAgentRole.EXECUTOR,
        task="work",
        allowed_tools=[],
        tool_budget=1,
        context_limit=1000,
        capture_usage=True,
    )
    member = SubAgent(config, MagicMock())
    member.orchestrator = MagicMock(tool_calls_used=0)
    member.orchestrator.get_token_usage.return_value = None
    member._execute_with_retry = AsyncMock(
        return_value=SimpleNamespace(content="done", metadata={})
    )
    result = await member.execute()
    assert not result.success
    assert "TokenUsage" in result.error
