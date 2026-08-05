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

"""Tests for the streaming ACT adapter (FEP-0007 Addendum A, step 3 connective tissue).

These stub the streaming executor and its per-run setup helpers, so they assert the adapter's
bridge behavior — driving ``execute_turn_streaming`` per turn, re-yielding chunks, mapping the
TurnResult onto the framework outcome, and preparing the per-run session — not the executor
internals (covered by the streaming-ACT-port tests).
"""

from types import SimpleNamespace

from victor.agent.services.streaming_act_adapter import (
    StreamActSession,
    StreamingActAdapter,
)
from victor.providers.base import StreamChunk


async def test_stream_turn_act_bridges_and_maps_turn_result():
    captured = {}

    async def fake_execute_turn_streaming(
        orch,
        runtime_owner,
        stream_ctx,
        *,
        user_message,
        goals,
        recovery,
        create_recovery_context,
        result,
    ):
        captured.update(
            orch=orch,
            runtime_owner=runtime_owner,
            stream_ctx=stream_ctx,
            user_message=user_message,
            goals=goals,
            recovery=recovery,
            create_recovery_context=create_recovery_context,
        )
        yield StreamChunk(content="a")
        yield StreamChunk(content="b")
        result.turn_result = SimpleNamespace(content="ab")

    executor = SimpleNamespace(execute_turn_streaming=fake_execute_turn_streaming)
    stream_ctx = SimpleNamespace(total_iterations=0)
    session = StreamActSession(
        orch="ORCH",
        runtime_owner="RO",
        stream_ctx=stream_ctx,
        goals="GOALS",
        recovery="REC",
        create_recovery_context="CRC",
    )
    adapter = StreamingActAdapter(executor, session)
    outcome = SimpleNamespace(turn_result=None)

    chunks = [
        chunk
        async for chunk in adapter.stream_turn_act(
            query="q",
            state={},
            perception=None,
            plan=None,
            turn_index=3,
            outcome=outcome,
        )
    ]

    # Chunks pass through and the produced TurnResult is mapped onto the framework outcome.
    assert [c.content for c in chunks] == ["a", "b"]
    assert outcome.turn_result.content == "ab"
    # The stream context's iteration counter tracks the loop turn.
    assert stream_ctx.total_iterations == 3
    # The session's collaborators + the loop query are forwarded verbatim.
    assert captured["orch"] == "ORCH"
    assert captured["runtime_owner"] == "RO"
    assert captured["stream_ctx"] is stream_ctx
    assert captured["user_message"] == "q"
    assert captured["goals"] == "GOALS"
    assert captured["recovery"] == "REC"
    assert captured["create_recovery_context"] == "CRC"


def _prepare_executor(events, stream_ctx, *, recovery_service=None, recovery_coordinator="COORD"):
    async def fake_create_stream_context(user_message, **kwargs):
        events.append(("create_ctx", user_message))
        return stream_ctx

    orch = SimpleNamespace(
        _recovery_service=recovery_service,
        _recovery_coordinator=recovery_coordinator,
        create_recovery_context="CRC",
        _current_stream_context=None,
    )
    runtime_owner = SimpleNamespace(
        _orchestrator=orch, _create_stream_context=fake_create_stream_context
    )

    async def fake_extract(o, msg):
        events.append(("extract", msg))

    executor = SimpleNamespace(
        _runtime_owner=runtime_owner,
        _reset_streaming_turn_state=lambda o: events.append(("reset",)),
        _extract_task_requirements=fake_extract,
        _apply_run_guidance=lambda o, ctx, msg, mei: events.append(("guidance", mei)),
        _initialize_task_intent=lambda o, ctx, msg: "GOALS",
    )
    return executor, orch


async def test_prepare_builds_session_from_executor_preamble():
    events = []
    stream_ctx = SimpleNamespace(max_exploration_iterations=4)
    executor, orch = _prepare_executor(events, stream_ctx)

    adapter = await StreamingActAdapter.prepare(executor, "hello")
    session = adapter.session

    assert session.orch is orch
    assert session.stream_ctx is stream_ctx
    assert session.goals == "GOALS"
    assert session.recovery == "COORD"  # _recovery_service None -> coordinator fallback
    assert session.create_recovery_context == "CRC"
    assert orch._current_stream_context is stream_ctx
    # Preamble ran with the right inputs (guidance used stream_ctx.max_exploration_iterations).
    assert ("create_ctx", "hello") in events
    assert ("extract", "hello") in events
    assert ("guidance", 4) in events
    assert ("reset",) in events


async def test_prepare_prefers_recovery_service_when_present():
    events = []
    stream_ctx = SimpleNamespace(max_exploration_iterations=2)
    executor, _orch = _prepare_executor(events, stream_ctx, recovery_service="SVC")

    adapter = await StreamingActAdapter.prepare(executor, "hi")

    assert adapter.session.recovery == "SVC"
