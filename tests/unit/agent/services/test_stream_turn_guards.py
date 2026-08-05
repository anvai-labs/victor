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

"""Tests for the extracted per-turn guard factory (FEP-0007 Phase 2 step)."""

from types import SimpleNamespace

from victor.agent.services import chat_stream_executor
from victor.agent.services.chat_stream_executor import StreamingChatExecutor
from victor.agent.turn_policy import (
    NudgePolicy,
    PlateauDetector,
    SpinDetector,
    TurnEvaluationController,
)
from victor.framework.search_novelty import SearchNoveltyTracker


async def test_extract_task_requirements_sets_orch_state():
    # A plain prompt yields no required files/outputs -> no event emit, no bus needed.
    orch = SimpleNamespace(
        _read_files_session={"stale.py"},
        _all_files_read_nudge_sent=True,
    )
    await StreamingChatExecutor._extract_task_requirements(orch, "hello there")
    assert isinstance(orch._required_files, list)
    assert isinstance(orch._required_outputs, list)
    assert orch._read_files_session == set()  # cleared
    assert orch._all_files_read_nudge_sent is False


def _guidance_orch(messages):
    return SimpleNamespace(
        _apply_intent_guard=lambda msg: messages.setdefault("intent", []).append(msg),
        _apply_task_guidance=lambda *a: messages.setdefault("task_guidance", []).append(a),
        add_message=lambda role, content, metadata=None: messages.setdefault("msg", []).append(
            content
        ),
    )


def test_apply_run_guidance_action_task_injects_guidance():
    messages = {}
    stream_ctx = SimpleNamespace(
        is_analysis_task=False,
        unified_task_type=SimpleNamespace(value="create"),
        coarse_task_type="action",
        is_action_task=True,
        needs_execution=True,
    )
    StreamingChatExecutor._apply_run_guidance(
        _guidance_orch(messages), stream_ctx, "build a script", 12
    )
    assert messages["intent"] == ["build a script"]
    assert len(messages["task_guidance"]) == 1
    assert any("ACTION-GUIDANCE" in c for c in messages.get("msg", []))


def test_apply_run_guidance_non_action_skips_message():
    messages = {}
    stream_ctx = SimpleNamespace(
        is_analysis_task=True,
        unified_task_type=SimpleNamespace(value="analyze"),
        coarse_task_type="analysis",
        is_action_task=False,
        needs_execution=False,
    )
    StreamingChatExecutor._apply_run_guidance(
        _guidance_orch(messages), stream_ctx, "explain the code", 50
    )
    assert messages.get("msg", []) == []  # no action-guidance for non-action tasks


def test_initialize_task_intent_returns_goals_and_seeds_ctx():
    seen = {}
    orch = SimpleNamespace(
        _tool_planner=SimpleNamespace(infer_goals_from_message=lambda m: ["goal1"]),
    )
    stream_ctx = SimpleNamespace(
        coarse_task_type="analysis",
        set_task_intent=lambda m: seen.__setitem__("intent", m),
        extend_plan_steps=lambda g: seen.__setitem__("steps", g),
        record_intent_event=lambda *a, **k: seen.__setitem__("event", True),
    )
    goals = StreamingChatExecutor._initialize_task_intent(orch, stream_ctx, "do the thing")
    assert goals == ["goal1"]  # returned for downstream tool planning
    assert seen["intent"] == "do the thing"
    assert seen["steps"] == ["goal1"]
    assert seen["event"] is True


def _provider_turn_executor():
    # Build an executor instance without running __init__ (we only exercise the
    # extracted ACT provider sub-step, which depends on no constructor state).
    return StreamingChatExecutor.__new__(StreamingChatExecutor)


async def test_stream_provider_turn_plans_tools_and_streams(monkeypatch):
    seen = {}

    class _Planner:
        def plan_tools(self, goals, available_inputs):
            seen["plan"] = (goals, available_inputs)
            return ["planned"]

    orch = SimpleNamespace(
        observed_files={"a.py"},
        _tool_planner=_Planner(),
        get_session_tools=lambda: None,
        thinking=False,
    )
    stream_ctx = SimpleNamespace(
        is_qa_task=False,
        context_msg="ctx",
        provider_kwargs={"execution_mode": "normal"},
    )

    async def _fake_stream(tools, provider_kwargs, stream_ctx):
        seen["stream"] = (tools, provider_kwargs)
        return ("hello", [{"name": "edit"}], 1.5, False)

    runtime_owner = SimpleNamespace(_stream_provider_response=_fake_stream)

    executor = _provider_turn_executor()

    async def _fake_get_tools_cached(self, o, ctx_msg, g, planned_tools=None):
        seen["tools_cached"] = (ctx_msg, g, planned_tools)
        return ["resolved_tool"]

    monkeypatch.setattr(
        StreamingChatExecutor, "_get_tools_cached", _fake_get_tools_cached, raising=True
    )

    tools, full_content, tool_calls, garbage = await executor._stream_provider_turn(
        orch, runtime_owner, stream_ctx, ["goal1"]
    )

    # planned tools seeded onto the context and used for tool resolution.
    assert stream_ctx.planned_tools == ["planned"]
    assert seen["plan"] == (["goal1"], ["query", "file_contents"])
    assert seen["tools_cached"] == ("ctx", ["goal1"], ["planned"])
    # provider response returned verbatim (the discarded float is the token estimate).
    assert tools == ["resolved_tool"]
    assert full_content == "hello"
    assert tool_calls == [{"name": "edit"}]
    assert garbage is False


async def test_stream_provider_turn_qa_task_skips_tools_and_enables_thinking():
    orch = SimpleNamespace(
        observed_files=set(),
        _tool_planner=SimpleNamespace(plan_tools=lambda *a: None),
        get_session_tools=lambda: None,
        thinking=True,
    )
    stream_ctx = SimpleNamespace(
        is_qa_task=True,
        context_msg="ctx",
        provider_kwargs={},
    )

    captured = {}

    async def _fake_stream(tools, provider_kwargs, stream_ctx):
        captured["tools"] = tools
        captured["provider_kwargs"] = provider_kwargs
        return ("", None, 0.0, True)

    runtime_owner = SimpleNamespace(_stream_provider_response=_fake_stream)
    executor = _provider_turn_executor()

    tools, full_content, tool_calls, garbage = await executor._stream_provider_turn(
        orch, runtime_owner, stream_ctx, None
    )

    # QA tasks pass tools=None; thinking=True injects the thinking provider kwarg.
    assert tools is None
    assert captured["tools"] is None
    assert captured["provider_kwargs"]["thinking"] == {
        "type": "enabled",
        "budget_tokens": 10000,
    }
    assert garbage is True


class _FakeToolExecResult:
    def __init__(self, *, tool_calls_executed=0, should_return=False, chunks=None):
        self.tool_calls_executed = tool_calls_executed
        self.should_return = should_return
        self.chunks = chunks or []
        self.tool_results = []


def _tools_turn_orch(seen):
    return SimpleNamespace(
        observed_files={"a.py"},
        tool_calls_used=3,
        tool_budget=50,
    )


async def test_execute_tools_turn_streams_and_records_outcome():
    seen = {}
    orch = _tools_turn_orch(seen)
    stream_ctx = SimpleNamespace(
        tool_calls_used=0,
        record_iteration_tool_count=lambda n: seen.__setitem__("recorded", n),
    )

    class _StreamingHandler:
        def update_observed_files(self, files):
            seen["observed"] = files

        async def execute_tools_streaming(self, **kwargs):
            seen["exec_kwargs"] = kwargs
            kwargs["result"].tool_calls_executed = 2
            kwargs["result"].should_return = False
            for c in ("chunk-a", "chunk-b"):
                yield c

    runtime_owner = SimpleNamespace(_tool_execution_handler=_StreamingHandler())
    executor = _provider_turn_executor()

    guard_calls = []
    executor._maybe_inject_write_action_guard = (
        lambda o, ctx, *, user_message, tool_calls_used: guard_calls.append(tool_calls_used)
    )

    holder = chat_stream_executor._ToolTurnOutcome()
    chunks = [
        c
        async for c in executor._execute_tools_turn(
            orch,
            runtime_owner,
            stream_ctx,
            user_message="do it",
            tool_calls=[{"name": "edit"}],
            full_content="content",
            result_holder=holder,
        )
    ]

    # Tool chunks streamed verbatim.
    assert chunks == ["chunk-a", "chunk-b"]
    # observed files snapshot passed to the handler.
    assert seen["observed"] == {"a.py"}
    # Post-exec bookkeeping: tool-call accounting + record + write-action guard.
    assert orch.tool_calls_used == 5  # 3 + 2 executed
    assert stream_ctx.tool_calls_used == 5
    assert seen["recorded"] == 2
    assert guard_calls == [5]
    # Result written to the holder for run() to read.
    assert holder.result.tool_calls_executed == 2
    assert holder.result.should_return is False


async def test_execute_tools_turn_buffered_handler_yields_result_chunks():
    seen = {}
    orch = _tools_turn_orch(seen)
    stream_ctx = SimpleNamespace(tool_calls_used=0)  # no record_iteration_tool_count

    buffered = _FakeToolExecResult(tool_calls_executed=1, should_return=True, chunks=["buf-1"])

    class _BufferedHandler:
        def update_observed_files(self, files):
            seen["observed"] = files

        async def execute_tools(self, **kwargs):
            return buffered

    # Buffered handler must NOT expose execute_tools_streaming.
    handler = _BufferedHandler()
    assert not hasattr(handler, "execute_tools_streaming")
    runtime_owner = SimpleNamespace(_tool_execution_handler=handler)

    executor = _provider_turn_executor()
    executor._maybe_inject_write_action_guard = lambda *a, **k: None

    holder = chat_stream_executor._ToolTurnOutcome()
    chunks = [
        c
        async for c in executor._execute_tools_turn(
            orch,
            runtime_owner,
            stream_ctx,
            user_message="do it",
            tool_calls=[{"name": "edit"}],
            full_content="content",
            result_holder=holder,
        )
    ]

    assert chunks == ["buf-1"]
    assert orch.tool_calls_used == 4  # 3 + 1
    # should_return propagated via the holder (loop control stays in run()).
    assert holder.result.should_return is True


def _emit_orch(messages):
    return SimpleNamespace(
        sanitizer=SimpleNamespace(sanitize=lambda c: c, strip_markup=lambda c: c),
        add_message=lambda role, content, **k: messages.append((role, content)),
        _chunk_generator=SimpleNamespace(
            generate_content_chunk=lambda text, is_final=False: SimpleNamespace(
                content=text, is_final=is_final
            )
        ),
        _record_runtime_intelligence_outcome=lambda **k: None,
    )


def _emit_executor():
    executor = _provider_turn_executor()
    executor._prepare_visible_content = lambda fc, *, user_message: fc

    async def _govern(orch, text):
        return text, False

    executor._govern_final_response = _govern
    return executor


async def _collect_emit(
    executor,
    *,
    orch,
    full_content,
    tool_calls,
    forced_task_completion=False,
    recovery=None,
    create_recovery_context=lambda ctx: ctx,
    runtime_owner=None,
    tools=None,
):
    decision = chat_stream_executor._EmitDecision()
    chunks = [
        c
        async for c in executor._emit_assistant_turn(
            orch,
            runtime_owner,
            SimpleNamespace(user_message="q"),
            recovery=recovery,
            create_recovery_context=create_recovery_context,
            full_content=full_content,
            tool_calls=tool_calls,
            forced_task_completion=forced_task_completion,
            user_message="q",
            tools=tools,
            decision=decision,
        )
    ]
    return chunks, decision


async def test_emit_content_non_final_yields_and_falls_through():
    messages = []
    orch = _emit_orch(messages)
    executor = _emit_executor()

    chunks, decision = await _collect_emit(
        executor, orch=orch, full_content="hello", tool_calls=[{"name": "edit"}]
    )

    # Tool calls present -> not a final emit; content yielded, loop continues.
    assert len(chunks) == 1
    assert chunks[0].content == "hello"
    assert chunks[0].is_final is False
    assert decision.assistant_content_yielded is True
    assert decision.prev_iteration_had_content is True
    assert decision.should_return is False
    assert decision.tool_calls == [{"name": "edit"}]
    assert ("assistant", "hello") in messages


async def test_emit_content_forced_completion_is_final_returns():
    messages = []
    orch = _emit_orch(messages)
    executor = _emit_executor()

    chunks, decision = await _collect_emit(
        executor,
        orch=orch,
        full_content="all done",
        tool_calls=None,
        forced_task_completion=True,
    )

    # forced completion + no tool calls -> final emit, ends the stream.
    assert chunks[0].is_final is True
    assert decision.assistant_content_yielded is True
    assert decision.should_return is True


async def test_emit_tool_calls_only_records_empty_assistant_message():
    messages = []
    orch = _emit_orch(messages)
    executor = _emit_executor()

    chunks, decision = await _collect_emit(
        executor, orch=orch, full_content="", tool_calls=[{"name": "edit"}]
    )

    # No content but tool calls -> empty assistant message recorded, nothing yielded.
    assert chunks == []
    assert ("assistant", "") in messages
    assert decision.should_return is False
    assert decision.should_continue is False
    assert decision.tool_calls == [{"name": "edit"}]


async def test_emit_empty_natural_completion_returns():
    messages = []
    orch = _emit_orch(messages)
    executor = _emit_executor()
    final = SimpleNamespace(content="natural done", is_final=True)
    recovery = SimpleNamespace(
        check_natural_completion=lambda ctx, has_tool_calls, content_length: final
    )

    chunks, decision = await _collect_emit(
        executor, orch=orch, full_content="", tool_calls=None, recovery=recovery
    )

    assert chunks == [final]
    assert decision.should_return is True


async def test_emit_empty_recovery_chunk_continues():
    messages = []
    orch = _emit_orch(messages)
    executor = _emit_executor()
    rchunk = SimpleNamespace(content="retry", is_final=False)
    recovery = SimpleNamespace(
        check_natural_completion=lambda ctx, has_tool_calls, content_length: None,
        handle_empty_response=lambda ctx: (rchunk, False),
    )

    chunks, decision = await _collect_emit(
        executor, orch=orch, full_content="", tool_calls=None, recovery=recovery
    )

    assert chunks == [rchunk]
    assert decision.should_continue is True
    assert decision.should_return is False


async def test_emit_empty_recovery_recovers_tool_calls():
    messages = []
    orch = _emit_orch(messages)
    executor = _emit_executor()
    recovery = SimpleNamespace(
        check_natural_completion=lambda ctx, has_tool_calls, content_length: None,
        handle_empty_response=lambda ctx: (None, False),
    )

    async def _recover(ctx, tools):
        return True, [{"name": "shell"}], None

    runtime_owner = SimpleNamespace(_handle_empty_response_recovery=_recover)

    chunks, decision = await _collect_emit(
        executor,
        orch=orch,
        full_content="",
        tool_calls=None,
        recovery=recovery,
        runtime_owner=runtime_owner,
    )

    # Recovery produced tool calls -> replace tool_calls, fall through (no return/continue).
    assert chunks == []
    assert decision.tool_calls == [{"name": "shell"}]
    assert decision.should_return is False
    assert decision.should_continue is False


async def test_emit_empty_recovery_failure_yields_fallback_and_returns():
    messages = []
    orch = _emit_orch(messages)
    executor = _emit_executor()
    recovery = SimpleNamespace(
        check_natural_completion=lambda ctx, has_tool_calls, content_length: None,
        handle_empty_response=lambda ctx: (None, False),
        get_recovery_fallback_message=lambda ctx: "fallback message",
    )

    async def _recover(ctx, tools):
        return False, None, None

    runtime_owner = SimpleNamespace(_handle_empty_response_recovery=_recover)

    chunks, decision = await _collect_emit(
        executor,
        orch=orch,
        full_content="",
        tool_calls=None,
        recovery=recovery,
        runtime_owner=runtime_owner,
    )

    assert chunks[0].content == "fallback message"
    assert chunks[0].is_final is True
    assert decision.should_return is True
