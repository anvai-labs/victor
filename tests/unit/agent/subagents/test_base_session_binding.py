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

"""Real ContextVar regressions for member work and stream-consumer isolation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from victor.providers.base import StreamChunk

from victor.agent.subagents.base import SubAgent, SubAgentConfig, SubAgentRole
from victor.agent.subagents.orchestrator import SubAgentOrchestrator
from victor.core.context import get_session_id, session_id, set_session_id


class _StreamOrchestrator:
    tool_calls_used = 3

    def __init__(self, on_advance=lambda: None):
        self.sessions = []
        self.closed_session = None
        self.on_advance = on_advance

    def get_messages(self):
        return ["m1", "m2"]

    async def stream_chat(self, task):
        try:
            for content, final in (("working", False), ("done", True)):
                await asyncio.sleep(0)
                self.sessions.append(get_session_id())
                self.on_advance()
                yield StreamChunk(content=content, is_final=final, metadata={"existing": 1})
        finally:
            self.closed_session = get_session_id()


def _member():
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
    return member


@pytest.fixture
def parent_session():
    token = set_session_id("session_root")
    try:
        yield "session_root"
    finally:
        session_id.reset(token)


@pytest.mark.asyncio
async def test_stream_binds_member_work_but_restores_caller_before_every_yield(parent_session):
    member = _member()
    chunks = []
    async for chunk in member.stream_execute():
        assert get_session_id() == parent_session
        chunks.append(chunk)
    assert get_session_id() == parent_session
    assert member.orchestrator.sessions == ["session_root-m1"] * 2
    assert member.orchestrator.closed_session == "session_root-m1"
    assert [c.content for c in chunks] == ["working", "done", ""]
    final = chunks[-1]
    assert final.is_final is True
    assert final.metadata["success"] is True
    assert final.metadata["tool_calls_used"] == 3
    assert final.metadata["role"] == member.config.role.value


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_on_final", [False, True])
async def test_early_break_and_cleanup_in_another_context_restore_parent(
    parent_session, stop_on_final
):
    member = _member()
    stream = member.stream_execute()
    async for chunk in stream:
        if not stop_on_final or chunk.is_final:
            break
    # Cleanup may happen later, in an async-generator finalizer task. Parent
    # attribution must already be restored, and cleanup must not reset a token
    # that was created in the consumer's different context.
    assert get_session_id() == parent_session
    await asyncio.create_task(stream.aclose())
    assert get_session_id() == parent_session
    assert member.orchestrator.closed_session == "session_root-m1"


@pytest.mark.asyncio
async def test_lazy_orchestrator_setup_uses_member_session(parent_session, monkeypatch):
    member = _member()
    recorder = member.orchestrator
    member.orchestrator = None
    setup_sessions = []

    def create():
        setup_sessions.append(get_session_id())
        return recorder

    monkeypatch.setattr(member, "_create_constrained_orchestrator", create)
    async for _chunk in member.stream_execute():
        assert get_session_id() == parent_session
    assert setup_sessions == ["session_root-m1"]


@pytest.mark.asyncio
async def test_spawn_timeout_closes_member_stream_without_leaking_session(
    parent_session, monkeypatch
):
    import victor.agent.subagents.orchestrator as module

    now = [0.0]
    member = _member()
    member.orchestrator = _StreamOrchestrator(on_advance=lambda: now.__setitem__(0, 2.0))
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(module, "SubAgent", lambda *_args: member)
    parent = SubAgentOrchestrator(MagicMock())
    chunks = []
    async for chunk in parent.stream_spawn(SubAgentRole.EXECUTOR, "x", timeout_seconds=1):
        assert get_session_id() == parent_session
        chunks.append(chunk)
    assert len(chunks) == 1
    assert chunks[0].metadata["timeout"] is True
    assert member.orchestrator.closed_session == "session_root-m1"
    assert get_session_id() == parent_session
    assert not parent.active_subagents


@pytest.mark.asyncio
async def test_spawn_synchronous_stream_factory_error_cleans_active_member(
    parent_session, monkeypatch
):
    def fail_before_iteration(self):
        assert get_session_id() == parent_session
        raise RuntimeError("stream factory failed")

    monkeypatch.setattr(SubAgent, "stream_execute", fail_before_iteration)
    parent = SubAgentOrchestrator(MagicMock())
    chunks = [chunk async for chunk in parent.stream_spawn(SubAgentRole.EXECUTOR, "x")]
    assert len(chunks) == 1
    assert chunks[0].is_final
    assert chunks[0].metadata["error_type"] == "RuntimeError"
    assert chunks[0].metadata["error_message"] == "stream factory failed"
    assert chunks[0].metadata["success"] is False
    assert not parent.active_subagents
    assert get_session_id() == parent_session
