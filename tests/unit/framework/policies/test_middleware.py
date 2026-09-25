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

"""Tests for PolicyEngineMiddleware (the bridge onto MiddlewareProtocol)."""

from typing import Optional, Tuple

import pytest

from victor.core.verticals.protocols import MiddlewarePriority
from victor.framework.hitl import ApprovalRequest, ApprovalStatus
from victor.framework.policies import (
    AskOnToolsPolicy,
    MaxToolCallsPolicy,
    Phase,
    Policy,
    PolicyContext,
    PolicyEngine,
    PolicyEngineMiddleware,
    PolicyEvent,
    PolicyVerdict,
)


def _static_context(cost=0.0, model=None):
    def provider() -> PolicyContext:
        return PolicyContext(cost_usd=cost, model=model)

    return provider


def _approval_handler(status: ApprovalStatus):
    async def handler(
        request: ApprovalRequest,
    ) -> Tuple[ApprovalStatus, Optional[str], Optional[str]]:
        return status, "handled", "tester"

    return handler


class _DenyPolicy(Policy):
    name = "deny_all"

    async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
        return PolicyVerdict.deny("not allowed", policy_name="deny_all")


class _ModifyArgsPolicy(Policy):
    name = "modify"

    async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
        return PolicyVerdict.allow(modified_arguments={"path": "/safe/path"})


class _RedactResultPolicy(Policy):
    name = "redact"

    def phases(self):
        return {Phase.TOOL_RESULT}

    async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
        return PolicyVerdict.allow(modified_result="[REDACTED]")


# -- Protocol surface -------------------------------------------------------


def test_priority_is_critical():
    mw = PolicyEngineMiddleware(PolicyEngine([]))
    assert mw.get_priority() is MiddlewarePriority.CRITICAL


def test_applies_to_all_tools():
    mw = PolicyEngineMiddleware(PolicyEngine([]))
    assert mw.get_applicable_tools() is None


# -- before_tool_call: ALLOW / DENY -----------------------------------------


async def test_allow_proceeds():
    mw = PolicyEngineMiddleware(PolicyEngine([]), _static_context())
    result = await mw.before_tool_call("read_file", {"path": "x"})
    assert result.proceed is True


async def test_deny_blocks_with_message():
    mw = PolicyEngineMiddleware(PolicyEngine([_DenyPolicy()]), _static_context())
    result = await mw.before_tool_call("write_file", {"path": "x"})
    assert result.proceed is False
    assert "not allowed" in (result.error_message or "")
    assert "write_file" in (result.error_message or "")


async def test_modified_arguments_passed_through_on_allow():
    mw = PolicyEngineMiddleware(PolicyEngine([_ModifyArgsPolicy()]), _static_context())
    result = await mw.before_tool_call("write_file", {"path": "/etc/passwd"})
    assert result.proceed is True
    assert result.modified_arguments == {"path": "/safe/path"}


# -- before_tool_call: ASK with HITL ----------------------------------------


async def test_ask_approved_proceeds():
    engine = PolicyEngine([AskOnToolsPolicy(["run_command"])])
    mw = PolicyEngineMiddleware(
        engine,
        _static_context(),
        approval_handler=_approval_handler(ApprovalStatus.APPROVED),
    )
    result = await mw.before_tool_call("run_command", {"cmd": "ls"})
    assert result.proceed is True


async def test_ask_rejected_blocks():
    engine = PolicyEngine([AskOnToolsPolicy(["run_command"])])
    mw = PolicyEngineMiddleware(
        engine,
        _static_context(),
        approval_handler=_approval_handler(ApprovalStatus.REJECTED),
    )
    result = await mw.before_tool_call("run_command", {"cmd": "rm -rf /"})
    assert result.proceed is False
    assert "Approval declined" in (result.error_message or "")


async def test_ask_non_target_tool_proceeds():
    # A tool not in the ask-list should not trigger an approval prompt.
    engine = PolicyEngine([AskOnToolsPolicy(["run_command"])])
    mw = PolicyEngineMiddleware(
        engine,
        _static_context(),
        approval_handler=_approval_handler(ApprovalStatus.REJECTED),
    )
    result = await mw.before_tool_call("read_file", {"path": "x"})
    assert result.proceed is True


# -- ASK fallback when no handler configured --------------------------------


async def test_ask_fallback_deny_when_no_handler():
    engine = PolicyEngine([AskOnToolsPolicy(["run_command"])])
    mw = PolicyEngineMiddleware(engine, _static_context(), ask_fallback="deny")
    result = await mw.before_tool_call("run_command", {"cmd": "ls"})
    assert result.proceed is False


async def test_ask_fallback_allow_when_no_handler():
    engine = PolicyEngine([AskOnToolsPolicy(["run_command"])])
    mw = PolicyEngineMiddleware(engine, _static_context(), ask_fallback="allow")
    result = await mw.before_tool_call("run_command", {"cmd": "ls"})
    assert result.proceed is True


# -- after_tool_call: result redaction --------------------------------------


async def test_after_tool_call_redacts_result():
    mw = PolicyEngineMiddleware(PolicyEngine([_RedactResultPolicy()]), _static_context())
    out = await mw.after_tool_call("read_file", {"path": "x"}, "secret data", True)
    assert out == "[REDACTED]"


async def test_after_tool_call_no_change_returns_none():
    # No TOOL_RESULT policy -> returns None (leave result unchanged per chain contract).
    mw = PolicyEngineMiddleware(PolicyEngine([]), _static_context())
    out = await mw.after_tool_call("read_file", {"path": "x"}, "data", True)
    assert out is None


# -- context provider resilience --------------------------------------------


@pytest.mark.parametrize("invalid", ["raises", "none", "nan", "inf", "negative"])
async def test_failing_context_provider_blocks(invalid):
    def boom() -> PolicyContext:
        if invalid == "none":
            return None
        if invalid in {"nan", "inf", "negative"}:
            return PolicyContext(cost_usd=-1.0 if invalid == "negative" else float(invalid))
        raise RuntimeError("provider down")

    mw = PolicyEngineMiddleware(PolicyEngine([MaxToolCallsPolicy(limit=1)]), boom)
    first = await mw.before_tool_call("read_file", {})
    assert first.proceed is False
    assert "provider down" not in first.error_message


async def test_configured_approval_handler_failure_does_not_use_allow_fallback():
    async def broken(request):
        raise RuntimeError("private details")

    mw = PolicyEngineMiddleware(
        PolicyEngine([AskOnToolsPolicy(["write"])]),
        approval_handler=broken,
        ask_fallback="allow",
    )
    assert not (await mw.before_tool_call("write", {})).proceed


@pytest.mark.parametrize("handler", [0, object()])
async def test_noncallable_approval_handler_cannot_autoapprove(handler):
    mw = PolicyEngineMiddleware(
        PolicyEngine([AskOnToolsPolicy(["write"])]), approval_handler=handler
    )
    assert not (await mw.before_tool_call("write", {})).proceed


async def test_falsey_callable_approval_handler_is_invoked():
    class Handler:
        called = False

        def __bool__(self):
            return False

        async def __call__(self, request):
            self.called = True
            return ApprovalStatus.REJECTED, "denied", "reviewer"

    handler = Handler()
    mw = PolicyEngineMiddleware(
        PolicyEngine([AskOnToolsPolicy(["write"])]), approval_handler=handler
    )
    assert not (await mw.before_tool_call("write", {})).proceed
    assert handler.called


async def test_no_context_provider_uses_empty_context():
    mw = PolicyEngineMiddleware(PolicyEngine([]))
    result = await mw.before_tool_call("read_file", {})
    assert result.proceed is True
