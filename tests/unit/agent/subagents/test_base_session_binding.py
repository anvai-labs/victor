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

"""Member-scoped session binding on the streaming path.

Concurrent streaming members must not share the parent's upstream
session-KV handle: ``stream_execute`` binds the member's resolved session
id for the duration of the stream and resets it in a ``finally``.
"""

from unittest.mock import MagicMock

import pytest
from victor.providers.base import StreamChunk

from victor.agent.subagents.base import SubAgent, SubAgentConfig, SubAgentRole


class _RecordingVar:
    """ContextVar double recording set()/reset() traffic."""

    def __init__(self) -> None:
        self.set_calls: list = []
        self.reset_calls: list = []

    def set(self, value):
        token = ("token", value)
        self.set_calls.append(token)
        return token

    def reset(self, token):
        self.reset_calls.append(token)


class _StreamOrchestrator:
    tool_calls_used = 3

    def get_messages(self):
        return ["m1", "m2"]

    async def stream_chat(self, task):
        yield StreamChunk(content="working", is_final=False)
        yield StreamChunk(content="done", is_final=True, metadata={"existing": 1})


@pytest.mark.asyncio
async def test_stream_binds_member_session_and_resets_after(monkeypatch):
    config = SubAgentConfig(
        role=SubAgentRole.EXECUTOR,
        task="do it",
        allowed_tools=["run_command"],
        tool_budget=10,
        context_limit=1000,
        member_id="m1",
        agent_id="agent_m1",
        parent_session_id="session_root",
    )
    member = SubAgent(config, MagicMock())
    member.orchestrator = _StreamOrchestrator()

    fake_var = _RecordingVar()
    monkeypatch.setattr("victor.core.context.set_session_id", fake_var.set)
    monkeypatch.setattr("victor.core.context.session_id", fake_var)

    chunks = [chunk async for chunk in member.stream_execute()]

    # The member's resolved session id was bound for the stream...
    assert fake_var.set_calls == [("token", config.resolve_member_session_id())]
    # ...and released exactly once when the stream settled.
    assert len(fake_var.reset_calls) == 1

    assert [c.content for c in chunks] == ["working", "done", ""]
    final = chunks[-1]
    assert final.is_final is True
    assert final.metadata["success"] is True
    assert final.metadata["tool_calls_used"] == 3
    assert final.metadata["role"] == config.role.value
