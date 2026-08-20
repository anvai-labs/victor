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

"""Regression tests for the policy ASK approval-handler container-swap bug.

A chat UI used to register its interactive approval handler into the *global* DI
container during ``on_chat_start``. The first turn then bootstrapped a fresh container
and disposed the previous global — orphaning the handler — so the policy middleware
resolved ``PolicyApprovalHandler`` as ``None`` and every ASK-gated tool hit
``ask_fallback`` (deny). The fix moves registration onto ``VictorClient``, which registers
into the container it actually builds (after bootstrap, before ``Agent.create``).

The existing ``test_builder_wiring.py`` uses a fake container and cannot catch this; these
tests drive the real ``bootstrap_container`` → ``set_container`` swap and the real
``VictorClient._ensure_initialized`` path.
"""

from victor.agent.factory.coordination_builders import CoordinationBuildersMixin
from victor.config.settings import Settings
from victor.framework.client import VictorClient
from victor.framework.session_config import SessionConfig


class _Host(CoordinationBuildersMixin):
    """Minimal host that resolves the approval handler from a real container."""

    def __init__(self, settings, container):
        self.settings = settings
        self.container = container
        self.model = "claude-opus-4-8"


def _governed_settings():
    s = Settings()
    s.governance.enabled = True
    s.governance.interactive_approval = False  # force container-registered resolution path
    return s


async def _handler(_request):  # a stand-in interactive elicitation callback
    return None


def test_global_registration_is_orphaned_by_container_swap():
    """Document the bug: registering into the global container before bootstrap loses it."""
    from victor.core.bootstrap import bootstrap_container
    from victor.core.container import reset_container
    from victor.framework.policies import (
        PolicyApprovalHandler,
        register_policy_approval_handler,
    )

    reset_container()
    try:
        # Old chat-UI path: register into the current global container A (no container arg).
        assert register_policy_approval_handler(_handler) is True

        # First turn bootstraps a fresh container B and disposes A.
        container_b = bootstrap_container()

        # The handler registered into A is gone from the now-global container B.
        assert container_b.get_optional(PolicyApprovalHandler) is None
    finally:
        reset_container()


async def test_set_approval_handler_survives_initialization(monkeypatch):
    """The fix: VictorClient registers the handler into the container the agent builds."""
    from victor.core.container import reset_container
    from victor.framework.policies import PolicyApprovalHandler

    reset_container()

    # Stub Agent.create so _ensure_initialized runs end-to-end without a provider/network.
    # The approval-handler registration happens BEFORE Agent.create, so the stub does not
    # affect what we are testing — it only keeps the test fast and offline.
    class _FakeAgent:
        execution_context = None  # no get_orchestrator → context resolution is skipped

    async def _fake_create(*args, **kwargs):
        return _FakeAgent()

    monkeypatch.setattr("victor.framework.agent.Agent.create", _fake_create)

    try:
        client = VictorClient(
            SessionConfig.from_cli_flags(
                tool_approval_enabled=True,
                ask_on_tools=["shell"],
                ask_fallback="deny",
            )
        )
        client.set_approval_handler(_handler)

        await client.initialize()  # drives the real _ensure_initialized path

        container = client._container
        assert container is not None
        registered = container.get_optional(PolicyApprovalHandler)
        assert registered is not None, "handler must survive container bootstrap"
        assert registered.handler is _handler

        # And the real middleware-build resolution returns it (not None → no ask_fallback).
        host = _Host(_governed_settings(), container=container)
        assert host._resolve_policy_approval_handler(host.settings.governance) is _handler
    finally:
        reset_container()
