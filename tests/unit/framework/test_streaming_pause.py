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

"""FEP-0029: the streaming turn boundary surfaces a durable pause as an AWAITING_APPROVAL event."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from victor.agent.paused_run_store import (
    InMemoryPausedRunStore,
    get_paused_run_store,
    reset_paused_run_store,
    set_paused_run_store,
)
from victor.framework import _internal as internal
from victor.framework import message_execution as me
from victor.framework.approval_pause import ApprovalPause
from victor.framework.client import _to_stream_event
from victor.framework.events import EventType, awaiting_approval_event
from victor.framework.hitl import ApprovalRequest


@pytest.fixture(autouse=True)
def _store():
    set_paused_run_store(InMemoryPausedRunStore())
    yield
    reset_paused_run_store()


def _orchestrator(*, durable: bool) -> Any:
    return SimpleNamespace(
        settings=SimpleNamespace(governance=SimpleNamespace(durable=durable)),
        model="m",
        active_session_id="sess-9",
        agent_id=None,
    )


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        id="req-1",
        title="Approve tool: run_command",
        description="",
        context={"tool_name": "run_command", "arguments": {"cmd": "rm -rf x"}},
    )


async def test_stream_surfaces_pause_as_awaiting_event(monkeypatch: Any) -> None:
    monkeypatch.setattr(me, "_resolve_chat_runtime", lambda *a, **k: object())

    async def _raising(*a: Any, **k: Any):
        if False:  # make this an async generator
            yield None
        raise ApprovalPause(_request())

    monkeypatch.setattr(internal, "stream_with_events", _raising)

    events = [
        e
        async for e in me.stream_message_events(
            orchestrator=_orchestrator(durable=True), user_message="do it"
        )
    ]

    # The stream ends with an AWAITING_APPROVAL event carrying the resume token + request.
    assert events[-1].type == EventType.AWAITING_APPROVAL
    run_id = events[-1].metadata["run_id"]
    assert run_id
    assert events[-1].metadata["approval_request"]["title"] == "Approve tool: run_command"

    # And the pause was recorded (via the shared record_pause_from_approval) with the gated tool.
    stored = get_paused_run_store().get(run_id)
    assert stored is not None and stored.session_id == "sess-9"
    assert stored.pending_tool == {"tool_name": "run_command", "arguments": {"cmd": "rm -rf x"}}


async def test_to_stream_event_maps_awaiting_approval() -> None:
    event = awaiting_approval_event("rid-2", {"id": "r", "title": "t"})
    se = _to_stream_event(event)
    assert se.event_type == EventType.AWAITING_APPROVAL.value
    assert se.metadata["run_id"] == "rid-2"
    assert se.metadata["approval_request"]["title"] == "t"
    assert se.success is False
