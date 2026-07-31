"""ADR-023 pillar 2b: SubAgent converts a MemberApprovalPause into an awaiting result.

A durable team member whose tool hits a policy ASK raises `MemberApprovalPause` (a
BaseException that rides through the runtime-core `except Exception` handlers) deep in its
orchestrator; `SubAgent.execute` catches it and returns an awaiting result, and
`execute_task` returns the awaiting **dict** that the SEQUENTIAL formation pauses on. These
tests drive that catch/convert without a real orchestrator.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from victor.agent.member_approval_context import MemberApprovalPause
from victor.agent.subagents.base import SubAgent, SubAgentConfig, SubAgentRole
from victor.framework.hitl import ApprovalRequest


class _MinimalContext:
    @property
    def settings(self) -> Any:
        return MagicMock(tool_budget=10, max_context_chars=10000)

    @property
    def provider(self) -> Any:
        return MagicMock()

    @property
    def provider_name(self) -> str:
        return "test"

    @property
    def model(self) -> str:
        return "test-model"

    @property
    def tool_registry(self) -> Any:
        return MagicMock()

    @property
    def temperature(self) -> float:
        return 0.7

    @property
    def vertical_context(self) -> Any:
        return None


def _member(agent_id: str = "m1") -> SubAgent:
    config = SubAgentConfig(
        role=SubAgentRole.EXECUTOR,
        task="do it",
        allowed_tools=["run_command"],
        tool_budget=10,
        context_limit=1000,
        agent_id=agent_id,
    )
    member = SubAgent(config, _MinimalContext())
    # Pre-set a fake orchestrator so execute() skips real construction.
    member.orchestrator = MagicMock(tool_calls_used=1, get_messages=lambda: [])
    return member


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        id="r1",
        title="run_command",
        description="run a shell command",
        context={"tool_name": "run_command", "arguments": {"cmd": "ls"}},
    )


def _arm_pause(member: SubAgent, request: ApprovalRequest, monkeypatch) -> None:
    async def _raise() -> Any:
        raise MemberApprovalPause(request)

    monkeypatch.setattr(member, "_execute_with_retry", _raise)


async def test_execute_catches_pause_returns_awaiting_result(monkeypatch) -> None:
    member = _member("m1")
    _arm_pause(member, _request(), monkeypatch)

    result = await member.execute()

    assert result.success is False
    assert result.details["awaiting_approval"] is True
    assert result.details["approval_request"]["title"] == "run_command"
    assert result.error is None  # a pause is a signal, not an error


async def test_execute_task_returns_awaiting_dict(monkeypatch) -> None:
    member = _member("m1")
    _arm_pause(member, _request(), monkeypatch)

    out = await member.execute_task("ignored", {})

    assert isinstance(out, dict)  # dict (not str) so the normalizer reads metadata
    assert out["success"] is False
    assert out["metadata"]["awaiting_approval"] is True
    ar = out["metadata"]["approval_request"]
    assert ar["title"] == "run_command"
    assert ar["context"]["tool_name"] == "run_command"  # the tool the lane/checkpoint read
