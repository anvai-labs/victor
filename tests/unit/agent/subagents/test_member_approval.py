"""ADR-023 pillar 2 (slice 2a): team members tag policy ASK requests with member_id.

A team member (SubAgent) shares the session's DI container, so its ASK-gated tools already
resolve the session terminal approval handler. `SubAgent._build_member_approval_handler`
wraps that handler to stamp the member's identity onto each `ApprovalRequest` so the shared
modal shows which member is asking. These tests isolate the global container via monkeypatch.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from victor.agent.subagents.base import SubAgent, SubAgentConfig, SubAgentRole
from victor.core.container import ServiceContainer
from victor.framework.hitl import ApprovalRequest, ApprovalStatus
from victor.framework.policies import register_policy_approval_handler


class _MinimalContext:
    """Smallest object satisfying the runtime-checkable SubAgentContext protocol."""

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


def _make_member(agent_id: str = "m1") -> SubAgent:
    config = SubAgentConfig(
        role=SubAgentRole.EXECUTOR,
        task="do it",
        allowed_tools=["run_command"],
        tool_budget=10,
        context_limit=1000,
        agent_id=agent_id,
    )
    return SubAgent(config, _MinimalContext())


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        id="req-1",
        title="run_command",
        description="run a shell command",
        context={"tool_name": "run_command", "arguments": {"cmd": "ls"}},
    )


async def test_wrapper_tags_member_id_and_delegates(monkeypatch) -> None:
    seen: dict = {}

    async def recording_handler(request: ApprovalRequest):
        seen["context"] = dict(request.context)
        return (ApprovalStatus.APPROVED, None, "tui_user")

    container = ServiceContainer()
    register_policy_approval_handler(recording_handler, container=container)
    monkeypatch.setattr("victor.core.get_container", lambda: container)

    member = _make_member("m1")
    handler = member._build_member_approval_handler()
    assert handler is not None

    status, _, responder = await handler(_request())

    assert status is ApprovalStatus.APPROVED  # delegated to the session handler
    assert seen["context"]["member_id"] == member.id == "m1"
    assert seen["context"]["member_role"] == "executor"
    # Original tool context is preserved.
    assert seen["context"]["tool_name"] == "run_command"


async def test_wrapper_reject_propagates(monkeypatch) -> None:
    async def rejecting_handler(request: ApprovalRequest):
        return (ApprovalStatus.REJECTED, "no", "tui_user")

    container = ServiceContainer()
    register_policy_approval_handler(rejecting_handler, container=container)
    monkeypatch.setattr("victor.core.get_container", lambda: container)

    handler = _make_member("m2")._build_member_approval_handler()
    assert handler is not None
    status, message, _ = await handler(_request())
    assert status is ApprovalStatus.REJECTED
    assert message == "no"


def test_no_registered_handler_returns_none(monkeypatch) -> None:
    # Empty container → no session handler → wrapper is None → member keeps ask_fallback.
    monkeypatch.setattr("victor.core.get_container", lambda: ServiceContainer())
    assert _make_member("m3")._build_member_approval_handler() is None


# ── ADR-023 pillar 2b: durable-pause trigger ──────────────────────

import pytest  # noqa: E402

from victor.agent.member_approval_context import (  # noqa: E402
    MemberApprovalPause,
    current_member_durable_pause_enabled,
)


async def test_durable_mode_raises_pause_instead_of_delegating(monkeypatch) -> None:
    called = {"inner": False}

    async def recording_handler(request: ApprovalRequest):
        called["inner"] = True
        return (ApprovalStatus.APPROVED, None, "tui_user")

    container = ServiceContainer()
    register_policy_approval_handler(recording_handler, container=container)
    monkeypatch.setattr("victor.core.get_container", lambda: container)

    token = current_member_durable_pause_enabled.set(True)
    try:
        handler = _make_member("m1")._build_member_approval_handler()
        assert handler is not None
        req = _request()
        with pytest.raises(MemberApprovalPause) as exc:
            await handler(req)
    finally:
        current_member_durable_pause_enabled.reset(token)

    assert exc.value.request is req  # carries the ApprovalRequest for the checkpoint
    assert called["inner"] is False  # durable pause did NOT block on the modal
    # member tagging still applied before pausing
    assert req.context["member_id"] == "m1"


async def test_disarmed_delegates_as_slice_2a(monkeypatch) -> None:
    # Not armed → unchanged slice-2a behavior (delegates to the session handler).
    async def recording_handler(request: ApprovalRequest):
        return (ApprovalStatus.APPROVED, None, "tui_user")

    container = ServiceContainer()
    register_policy_approval_handler(recording_handler, container=container)
    monkeypatch.setattr("victor.core.get_container", lambda: container)

    assert current_member_durable_pause_enabled.get() is None
    handler = _make_member("m1")._build_member_approval_handler()
    status, _, _ = await handler(_request())
    assert status is ApprovalStatus.APPROVED


async def test_durable_mode_arms_wrapper_without_session_handler(monkeypatch) -> None:
    # No modal handler, but durable pause armed → still build a wrapper that raises the pause.
    monkeypatch.setattr("victor.core.get_container", lambda: ServiceContainer())
    token = current_member_durable_pause_enabled.set(True)
    try:
        handler = _make_member("m4")._build_member_approval_handler()
        assert handler is not None
        with pytest.raises(MemberApprovalPause):
            await handler(_request())
    finally:
        current_member_durable_pause_enabled.reset(token)
