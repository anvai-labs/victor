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

"""Tests for the streaming-ACT seam (FEP-0007 Addendum A, step 1).

These cover ``execute_turn_streaming`` — the contiguous provider → emit → tools ACT primitive
that produces a ``TurnResult`` — and the ``_build_streaming_turn_result`` mapper. The underlying
sub-step helpers are stubbed so these assert the ACT *assembly* (chunk ordering, tool gating,
and the produced ``TurnResult``), not the helper internals (which have their own tests).
"""

from types import SimpleNamespace

import pytest

from victor.agent.services.chat_stream_executor import (
    StreamingActResult,
    StreamingChatExecutor,
)
from victor.agent.streaming.tool_execution import ToolExecutionResult
from victor.providers.base import StreamChunk


def _executor() -> StreamingChatExecutor:
    """A bare executor instance; sub-step helpers are stubbed per-test."""
    return StreamingChatExecutor.__new__(StreamingChatExecutor)


def _orch() -> SimpleNamespace:
    # ACT only parses tool calls via the orchestrator; pass them through unchanged.
    return SimpleNamespace(_parse_and_validate_tool_calls=lambda tc, fc: (tc, fc))


async def _drain(executor, orch, stream_ctx, result):
    return [
        chunk
        async for chunk in executor.execute_turn_streaming(
            orch,
            object(),
            stream_ctx,
            user_message="hi",
            goals=None,
            recovery=None,
            create_recovery_context=None,
            result=result,
        )
    ]


async def test_execute_turn_streaming_no_tools_builds_text_turn_result():
    ex = _executor()

    async def fake_provider(orch, runtime_owner, stream_ctx, goals):
        return (None, "hello world", None, False)

    async def fake_emit(
        orch, runtime_owner, stream_ctx, *, decision, full_content, tool_calls, **_
    ):
        decision.assistant_content_yielded = True
        decision.tool_calls = tool_calls
        yield StreamChunk(content=full_content)

    async def fake_tools(*_a, **_k):
        raise AssertionError("tools must not run when there are no tool calls")
        yield  # pragma: no cover  (marks this as an async generator)

    ex._stream_provider_turn = fake_provider
    ex._emit_assistant_turn = fake_emit
    ex._execute_tools_turn = fake_tools

    result = StreamingActResult()
    chunks = await _drain(ex, _orch(), SimpleNamespace(is_qa_task=False), result)

    assert [c.content for c in chunks] == ["hello world"]
    assert result.turn_result.content == "hello world"
    assert result.turn_result.has_tool_calls is False
    assert result.turn_result.tool_calls_count == 0
    assert result.tool_exec_result is None
    assert result.assistant_content_yielded is True


async def test_execute_turn_streaming_runs_tools_and_maps_results():
    ex = _executor()
    tool_calls = [{"name": "read_file", "arguments": {"path": "a.py"}}]

    async def fake_provider(orch, runtime_owner, stream_ctx, goals):
        return (["read_file"], "", tool_calls, False)

    async def fake_emit(orch, runtime_owner, stream_ctx, *, decision, tool_calls, **_):
        decision.tool_calls = tool_calls
        decision.assistant_content_yielded = False
        return
        yield  # unreachable; marks this as an async generator

    async def fake_tools(orch, runtime_owner, stream_ctx, *, result_holder, **_):
        result_holder.result = ToolExecutionResult(
            tool_results=[{"success": True, "name": "read_file"}],
            tool_calls_executed=1,
        )
        yield StreamChunk(content="[tool: read_file]")

    ex._stream_provider_turn = fake_provider
    ex._emit_assistant_turn = fake_emit
    ex._execute_tools_turn = fake_tools

    result = StreamingActResult()
    chunks = await _drain(ex, _orch(), SimpleNamespace(is_qa_task=False), result)

    assert [c.content for c in chunks] == ["[tool: read_file]"]
    turn = result.turn_result
    assert turn.has_tool_calls is True
    assert turn.tool_calls_count == 1
    assert turn.tool_results == [{"success": True, "name": "read_file"}]
    assert turn.successful_tool_count == 1
    assert result.tool_exec_result is not None
    assert result.tools == ["read_file"]


async def test_execute_turn_streaming_skips_tools_when_emit_returns():
    ex = _executor()
    tool_calls = [{"name": "x", "arguments": {}}]

    async def fake_provider(orch, runtime_owner, stream_ctx, goals):
        return (None, "done", tool_calls, False)

    async def fake_emit(orch, runtime_owner, stream_ctx, *, decision, tool_calls, **_):
        decision.should_return = True  # emit decided to end the stream this turn
        decision.tool_calls = tool_calls
        return
        yield  # unreachable; marks this as an async generator

    async def fake_tools(*_a, **_k):
        raise AssertionError("tools must not run when emit.should_return is set")
        yield  # pragma: no cover

    ex._stream_provider_turn = fake_provider
    ex._emit_assistant_turn = fake_emit
    ex._execute_tools_turn = fake_tools

    result = StreamingActResult()
    await _drain(ex, _orch(), SimpleNamespace(is_qa_task=False), result)

    assert result.emit_should_return is True
    assert result.tool_exec_result is None
    # The model requested tools, but emit short-circuited before execution.
    assert result.turn_result.has_tool_calls is True


async def test_execute_turn_streaming_surfaces_garbage_flag():
    ex = _executor()

    async def fake_provider(orch, runtime_owner, stream_ctx, goals):
        return (None, "junk", None, True)

    async def fake_emit(orch, runtime_owner, stream_ctx, *, decision, tool_calls, **_):
        decision.tool_calls = tool_calls
        return
        yield  # unreachable

    ex._stream_provider_turn = fake_provider
    ex._emit_assistant_turn = fake_emit

    result = StreamingActResult()
    await _drain(ex, _orch(), SimpleNamespace(is_qa_task=False), result)

    assert result.garbage_detected is True


@pytest.mark.parametrize("is_qa", [True, False])
def test_build_streaming_turn_result_maps_fields(is_qa):
    turn = StreamingChatExecutor._build_streaming_turn_result(
        SimpleNamespace(is_qa_task=is_qa),
        full_content="hi",
        tool_calls=[{"name": "a", "arguments": {}}],
        tool_exec_result=None,
    )
    assert turn.content == "hi"
    assert turn.is_qa_response is is_qa
    assert turn.has_tool_calls is True
    assert turn.tool_calls_count == 1
    assert turn.tool_results == []
    assert turn.response.tool_calls == [{"name": "a", "arguments": {}}]


def test_build_streaming_turn_result_empty_content_is_safe():
    turn = StreamingChatExecutor._build_streaming_turn_result(
        SimpleNamespace(),
        full_content="",
        tool_calls=None,
        tool_exec_result=None,
    )
    assert turn.content == ""
    assert turn.has_tool_calls is False
    assert turn.response.tool_calls is None


def test_build_streaming_turn_result_stamps_forced_completion_metadata():
    forced = StreamingChatExecutor._build_streaming_turn_result(
        SimpleNamespace(is_qa_task=False),
        full_content="done",
        tool_calls=None,
        tool_exec_result=None,
        forced_completion=True,
    )
    assert forced.response.metadata == {"forced_task_completion": True}

    not_forced = StreamingChatExecutor._build_streaming_turn_result(
        SimpleNamespace(is_qa_task=False),
        full_content="done",
        tool_calls=None,
        tool_exec_result=None,
        forced_completion=False,
    )
    assert not_forced.response.metadata is None


class _FakeCompletionDetector:
    def __init__(self, confidence):
        self._confidence = confidence
        self._state = SimpleNamespace(last_summary="")

    def analyze_response(self, content):  # noqa: D401 - test stub
        self.analyzed = content

    def get_completion_confidence(self):
        return self._confidence


def test_detect_high_confidence_completion_high_no_tools_forces_stop():
    from victor.agent.task_completion import CompletionConfidence

    ex = _executor()
    orch = SimpleNamespace(
        _task_completion_detector=_FakeCompletionDetector(CompletionConfidence.HIGH)
    )
    stream_ctx = SimpleNamespace(force_completion=False, skip_continuation=False)

    assert (
        ex._detect_high_confidence_completion(
            orch, stream_ctx, full_content="The answer is 42.", tool_calls=None
        )
        is True
    )
    assert stream_ctx.force_completion is True
    assert stream_ctx.skip_continuation is True


def test_detect_high_confidence_completion_defers_when_tools_pending():
    from victor.agent.task_completion import CompletionConfidence

    ex = _executor()
    ex._clear_deferred_active_completion_signal = lambda detector: None  # isolate from internals
    orch = SimpleNamespace(
        _task_completion_detector=_FakeCompletionDetector(CompletionConfidence.HIGH)
    )
    stream_ctx = SimpleNamespace(force_completion=False, skip_continuation=False)

    assert (
        ex._detect_high_confidence_completion(
            orch,
            stream_ctx,
            full_content="x",
            tool_calls=[{"name": "read", "arguments": {}}],
        )
        is False
    )
    assert stream_ctx.force_completion is False


def test_detect_high_confidence_completion_non_high_and_no_detector_return_false():
    from victor.agent.task_completion import CompletionConfidence

    ex = _executor()
    medium = SimpleNamespace(
        _task_completion_detector=_FakeCompletionDetector(CompletionConfidence.MEDIUM)
    )
    assert (
        ex._detect_high_confidence_completion(
            medium, SimpleNamespace(), full_content="x", tool_calls=None
        )
        is False
    )
    none = SimpleNamespace(_task_completion_detector=None)
    assert (
        ex._detect_high_confidence_completion(
            none, SimpleNamespace(), full_content="x", tool_calls=None
        )
        is False
    )
