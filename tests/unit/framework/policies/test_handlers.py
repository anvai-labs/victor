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

"""Tests for policy approval handlers."""

from victor.framework.hitl import ApprovalRequest, ApprovalStatus
from victor.framework.policies import (
    AskOnToolsPolicy,
    PolicyEngine,
    PolicyEngineMiddleware,
    make_console_approval_handler,
)
from victor.framework.policies.handlers import console_approval_handler


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        id="req_1",
        title="Approve tool: run_command",
        description="Policy requests approval to run 'run_command'.",
    )


async def test_console_handler_approves_when_confirmed():
    handler = make_console_approval_handler(confirm_fn=lambda prompt: True)
    status, response, responder = await handler(_request())
    assert status is ApprovalStatus.APPROVED
    assert responder == "console_user"


async def test_console_handler_rejects_when_declined():
    handler = make_console_approval_handler(confirm_fn=lambda prompt: False)
    status, _, _ = await handler(_request())
    assert status is ApprovalStatus.REJECTED


async def test_console_handler_includes_description_in_prompt():
    seen = {}

    def confirm(prompt: str) -> bool:
        seen["prompt"] = prompt
        return True

    await make_console_approval_handler(confirm_fn=confirm)(_request())
    assert "run_command" in seen["prompt"]
    assert "approval" in seen["prompt"].lower()


async def test_console_handler_fails_safe_on_prompt_error():
    def boom(prompt: str) -> bool:
        raise RuntimeError("no tty")

    handler = make_console_approval_handler(confirm_fn=boom)
    status, _, _ = await handler(_request())
    assert status is ApprovalStatus.REJECTED


async def test_default_console_handler_is_usable():
    # The module-level default exists and is callable as an ApprovalHandler.
    assert callable(console_approval_handler)


async def test_handler_drives_middleware_ask_approval():
    # End-to-end: an ASK verdict resolved by a console handler that approves.
    engine = PolicyEngine([AskOnToolsPolicy(["run_command"])])
    mw = PolicyEngineMiddleware(
        engine,
        approval_handler=make_console_approval_handler(confirm_fn=lambda p: True),
    )
    result = await mw.before_tool_call("run_command", {"cmd": "ls"})
    assert result.proceed is True


async def test_handler_drives_middleware_ask_rejection():
    engine = PolicyEngine([AskOnToolsPolicy(["run_command"])])
    mw = PolicyEngineMiddleware(
        engine,
        approval_handler=make_console_approval_handler(confirm_fn=lambda p: False),
    )
    result = await mw.before_tool_call("run_command", {"cmd": "ls"})
    assert result.proceed is False
