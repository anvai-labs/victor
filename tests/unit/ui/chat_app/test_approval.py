# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Chat UI tool-approval handler: decision mapping + container registration.

These run without the optional ``chat-ui`` extra; ``chainlit`` is faked via ``sys.modules``
only for the one test that exercises the prompt coroutine.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from victor.framework.hitl import ApprovalRequest, ApprovalStatus
from victor.framework.policies import register_policy_approval_handler
from victor.ui.chat_app.approval import chainlit_approval_handler, decision_from_action


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        id="req-1",
        title="Approve tool: bash",
        description="Run `ls -la`",
        context={"tool_name": "bash"},
        timeout_seconds=30,
    )


def test_decision_approve_legacy_value_shape():
    # Chainlit 1.x flat {"value": ...}
    status, _, responder = decision_from_action({"value": "approve"})
    assert status is ApprovalStatus.APPROVED
    assert responder == "chat_ui_user"


def test_decision_approve_2x_payload_shape():
    # Chainlit 2.x returns the chosen action as name + payload
    status, _, _ = decision_from_action({"name": "approve", "payload": {"value": "approve"}})
    assert status is ApprovalStatus.APPROVED


def test_decision_reject_2x_payload_shape():
    status, _, _ = decision_from_action({"name": "reject", "payload": {"value": "reject"}})
    assert status is ApprovalStatus.REJECTED


def test_decision_reject_legacy_value_shape():
    status, _, _ = decision_from_action({"value": "reject"})
    assert status is ApprovalStatus.REJECTED


def test_decision_timeout_on_none():
    status, _, _ = decision_from_action(None)
    assert status is ApprovalStatus.TIMEOUT


def test_decision_unknown_value_fails_safe_to_reject():
    status, _, _ = decision_from_action({"value": "something-else"})
    assert status is ApprovalStatus.REJECTED


@pytest.mark.parametrize(
    "click,expected",
    [
        ({"name": "approve", "payload": {"value": "approve"}}, ApprovalStatus.APPROVED),
        ({"name": "reject", "payload": {"value": "reject"}}, ApprovalStatus.REJECTED),
        ({"value": "approve"}, ApprovalStatus.APPROVED),
        (None, ApprovalStatus.TIMEOUT),
    ],
)
async def test_handler_prompts_and_maps(monkeypatch, click, expected):
    sent = {}

    class _FakeAskActionMessage:
        def __init__(self, content, actions, timeout):
            sent["content"] = content
            sent["actions"] = actions
            sent["timeout"] = timeout

        async def send(self):
            return click

    fake_cl = SimpleNamespace(
        AskActionMessage=_FakeAskActionMessage,
        Action=lambda **kw: kw,
    )
    monkeypatch.setitem(sys.modules, "chainlit", fake_cl)

    status, _, _ = await chainlit_approval_handler(_request())
    assert status is expected
    # The prompt surfaced the tool and respected the request timeout.
    assert "bash" in sent["content"]
    assert sent["timeout"] == 30


async def test_handler_fails_safe_on_prompt_error(monkeypatch):
    class _BoomMessage:
        def __init__(self, *a, **k):
            pass

        async def send(self):
            raise RuntimeError("ws closed")

    monkeypatch.setitem(
        sys.modules,
        "chainlit",
        SimpleNamespace(AskActionMessage=_BoomMessage, Action=lambda **kw: kw),
    )
    status, _, _ = await chainlit_approval_handler(_request())
    assert status is ApprovalStatus.REJECTED


class _FakeContainer:
    def __init__(self, frozen=False, existing=None):
        self._frozen = frozen
        self._store = {}
        if existing is not None:
            self._store[existing.__class__] = existing

    def get_optional(self, t):
        return self._store.get(t)

    def register_instance(self, t, instance):
        # Mirror ServiceContainer.register_instance — the method
        # register_policy_approval_handler() actually calls (register() treats its 2nd arg as a
        # factory, which would mis-store the non-callable PolicyApprovalHandler instance).
        if self._frozen:
            raise RuntimeError("frozen")
        self._store[t] = instance


def test_register_handler_into_container():
    from victor.framework.policies import PolicyApprovalHandler

    container = _FakeContainer()
    assert register_policy_approval_handler(chainlit_approval_handler, container) is True
    holder = container.get_optional(PolicyApprovalHandler)
    assert holder is not None and callable(holder.handler)


def test_register_is_idempotent_when_already_present():
    from victor.framework.policies import PolicyApprovalHandler

    existing = PolicyApprovalHandler(lambda req: None)
    container = _FakeContainer(existing=existing)
    assert register_policy_approval_handler(chainlit_approval_handler, container) is True
    # Did not overwrite the pre-existing holder.
    assert container.get_optional(PolicyApprovalHandler) is existing


def test_register_returns_false_on_frozen_container():
    container = _FakeContainer(frozen=True)
    assert register_policy_approval_handler(chainlit_approval_handler, container) is False
