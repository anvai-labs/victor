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


async def test_failing_context_provider_degrades_gracefully():
    def boom() -> PolicyContext:
        raise RuntimeError("provider down")

    # MaxToolCalls doesn't need context; a failing provider must not crash the gate.
    mw = PolicyEngineMiddleware(PolicyEngine([MaxToolCallsPolicy(limit=1)]), boom)
    first = await mw.before_tool_call("read_file", {})
    second = await mw.before_tool_call("read_file", {})
    assert first.proceed is True
    assert second.proceed is False


async def test_no_context_provider_uses_empty_context():
    mw = PolicyEngineMiddleware(PolicyEngine([]))
    result = await mw.before_tool_call("read_file", {})
    assert result.proceed is True
