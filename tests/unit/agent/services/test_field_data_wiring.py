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

"""Placement tests: the field-data recorder fires through the REAL session path.

Dormancy lesson (FEP-0041): the recorder's existence proves nothing — these
tests drive ``StreamingChatExecutor.run_unified`` (the sole live streaming
entry point) with the production wiring and a fake loop, asserting the JSONL
store receives the start/end records, honors cancellation, and — default OFF —
writes nothing at all.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from victor.agent.services.chat_runtime_services import ChatGovernance, ChatStreamLifecycle
from victor.providers.base import StreamChunk

SECRET_USER_PROMPT = "SECRET-PROMPT-the-user-really-typed-this"


def _settings(*, enabled: bool, store_dir):
    return SimpleNamespace(
        evaluation=SimpleNamespace(
            field_data_enabled=enabled,
            field_data_store_dir=str(store_dir) if store_dir is not None else None,
            field_data_max_file_bytes=5 * 1024 * 1024,
        ),
        agent=SimpleNamespace(per_turn_auditor=True, effect_gated_completion=False),
        search=SimpleNamespace(graph=SimpleNamespace(codebase_intent_indexing_enabled=False)),
    )


def _patch_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "victor.agent.services.judge_calibration_gate.resolve_completion_strategy",
        lambda *args: "enhanced",
    )
    monkeypatch.setattr("victor.framework.effect_gate.resolve_effect_gate_enabled", lambda _: False)
    monkeypatch.setattr(
        "victor.framework.per_turn_auditor.resolve_per_turn_auditor_enabled", lambda _: False
    )
    monkeypatch.delenv("VICTOR_FIELD_DATA_ENABLED", raising=False)
    monkeypatch.delenv("VICTOR_PER_TURN_AUDITOR", raising=False)


def _make_executor(monkeypatch: pytest.MonkeyPatch, settings, *, loop):
    from victor.agent.services.chat_stream_executor import StreamingChatExecutor
    from victor.agent.services.streaming_act_adapter import StreamingActAdapter

    events: list[str] = []
    cancelled = False

    class Lifecycle:
        def is_cancelled(self):
            return cancelled

        def finish(self):
            events.append("finish")

    owner = SimpleNamespace(
        _message_policy_gate=None,
        settings=settings,
        turn_executor=None,
        provider=SimpleNamespace(name="ollama"),
        model="qwen3-coder-tools:30b",
    )
    executor = StreamingChatExecutor(
        SimpleNamespace(
            _orchestrator=owner,
            services=SimpleNamespace(
                governance=ChatGovernance(),
                stream_lifecycle=ChatStreamLifecycle(Lifecycle()),
                intelligence=SimpleNamespace(
                    executor_runtime=lambda: None,
                    record_outcome=lambda **kwargs: None,
                ),
            ),
        )
    )
    monkeypatch.setattr(executor, "_get_conversation_history", lambda *args: [])
    monkeypatch.setattr(
        StreamingActAdapter,
        "prepare",
        AsyncMock(
            return_value=SimpleNamespace(
                session=SimpleNamespace(
                    stream_ctx=SimpleNamespace(
                        total_iterations=7,
                        tool_calls_used=12,
                        start_time=100.0,
                    )
                )
            )
        ),
    )
    monkeypatch.setattr("victor.framework.agentic_loop.AgenticLoop", lambda **kwargs: loop())
    return executor


async def _consume(executor) -> list[StreamChunk]:
    return [chunk async for chunk in executor.run_unified(SECRET_USER_PROMPT)]


def _read_store(store_dir):
    from victor.evaluation.field_data import read_store

    return read_store(store_dir)


@pytest.mark.asyncio
async def test_run_unified_records_start_and_end_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_loop(monkeypatch)

    class Loop:
        async def run_streaming(self, *args, **kwargs):
            yield StreamChunk(content="working")
            yield StreamChunk(
                content="done",
                is_final=True,
                metadata={"agentic_loop_success": True, "agentic_loop_iterations": 7},
            )

    executor = _make_executor(monkeypatch, _settings(enabled=True, store_dir=tmp_path), loop=Loop)
    chunks = await _consume(executor)

    assert chunks[-1].is_final is True
    records = _read_store(tmp_path)
    assert [r["event"] for r in records] == ["session_start", "session_end"]

    start = records[0]
    # The RESOLVED executor mode (AUTO resolves to the StateGraph executor).
    assert start["mode"] == "stategraph"
    assert start["provider"] == "ollama"
    assert start["model"] == "qwen3-coder-tools:30b"
    # Flags come from the PRODUCTION seam — the same resolve this run itself
    # used (patched here for loop isolation), so recorded exposure always
    # matches what the session actually ran with.
    assert start["flags"] == {
        "per_turn_auditor": False,
        "codebase_intent_indexing_enabled": False,
    }

    end = records[1]
    assert end["completion_status"] == "completed"
    # The loop's terminal verdict rides the recorded outcome (ADR-036
    # demotion evidence: an ON-arm skew in failed completions).
    assert end["success"] is True
    assert end["turn_count"] == 7
    assert end["tool_call_count"] == 12
    assert end["duration_seconds"] is not None
    # Privacy: the user's prompt text never reaches the store.
    dumped = "\n".join(p.read_text() for p in tmp_path.glob("*.jsonl"))
    assert SECRET_USER_PROMPT not in dumped


@pytest.mark.asyncio
async def test_run_unified_records_cancellation(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _patch_loop(monkeypatch)

    # A lifecycle that flips to cancelled after the first chunk: run_unified's
    # chunk loop checks is_cancelled() before each dispatch.
    from victor.agent.services.chat_stream_executor import StreamingChatExecutor
    from victor.agent.services.streaming_act_adapter import StreamingActAdapter

    cancelled = False

    class CancelAfterFirstLoop:
        async def run_streaming(self, *args, **kwargs):
            yield StreamChunk(content="working")

    class Lifecycle:
        def is_cancelled(self):
            return cancelled

        def finish(self):
            pass

    owner = SimpleNamespace(
        _message_policy_gate=None,
        settings=_settings(enabled=True, store_dir=tmp_path),
        turn_executor=None,
        provider=SimpleNamespace(name="ollama"),
        model="qwen3-coder-tools:30b",
    )
    executor = StreamingChatExecutor(
        SimpleNamespace(
            _orchestrator=owner,
            services=SimpleNamespace(
                governance=ChatGovernance(),
                stream_lifecycle=ChatStreamLifecycle(Lifecycle()),
                intelligence=SimpleNamespace(
                    executor_runtime=lambda: None,
                    record_outcome=lambda **kwargs: None,
                ),
            ),
        )
    )
    monkeypatch.setattr(executor, "_get_conversation_history", lambda *args: [])
    monkeypatch.setattr(
        StreamingActAdapter,
        "prepare",
        AsyncMock(
            return_value=SimpleNamespace(
                session=SimpleNamespace(
                    stream_ctx=SimpleNamespace(
                        total_iterations=1, tool_calls_used=1, start_time=0.0
                    )
                )
            )
        ),
    )
    monkeypatch.setattr(
        "victor.framework.agentic_loop.AgenticLoop",
        lambda **kwargs: CancelAfterFirstLoop(),
    )

    chunks_iter = executor.run_unified(SECRET_USER_PROMPT)
    first = await chunks_iter.__anext__()
    cancelled = True
    terminal = await chunks_iter.__anext__()

    assert first.is_final is not True
    assert terminal.is_final is True
    # The end record lands when the generator is closed (client disconnect),
    # mirroring how the real stream lifecycle tears the run down.
    await chunks_iter.aclose()
    records = _read_store(tmp_path)
    assert len(records) == 2
    assert records[-1]["completion_status"] == "cancelled"


@pytest.mark.asyncio
async def test_run_unified_records_errors_without_breaking_the_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _patch_loop(monkeypatch)

    class Loop:
        async def run_streaming(self, *args, **kwargs):
            yield StreamChunk(content="partial")
            raise RuntimeError("provider exploded")

    executor = _make_executor(monkeypatch, _settings(enabled=True, store_dir=tmp_path), loop=Loop)

    with pytest.raises(RuntimeError, match="provider exploded"):
        await _consume(executor)

    records = _read_store(tmp_path)
    assert len(records) == 2
    end = records[-1]
    assert end["completion_status"] == "error"
    assert end["error_kind"] == "RuntimeError"
    # No free-text error message: only the exception class name is stored.
    assert "provider exploded" not in json.dumps(records)


@pytest.mark.asyncio
async def test_run_unified_default_off_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Default OFF: zero records, zero store files, no behavior change."""
    _patch_loop(monkeypatch)
    monkeypatch.setattr("victor.evaluation.field_data.default_store_dir", lambda: tmp_path)

    class Loop:
        async def run_streaming(self, *args, **kwargs):
            yield StreamChunk(content="working")
            yield StreamChunk(
                content="done",
                is_final=True,
                metadata={"agentic_loop_success": True, "agentic_loop_iterations": 3},
            )

    executor = _make_executor(monkeypatch, None, loop=Loop)
    chunks = await _consume(executor)

    assert chunks[-1].is_final is True
    assert list(tmp_path.glob("*.jsonl")) == []


@pytest.mark.asyncio
async def test_run_unified_default_off_never_imports_the_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The heavy victor.evaluation import must stay off the default chat path.

    A ``None`` sys.modules entry is a tripwire: any import attempt inside
    run_unified raises immediately, so this pins the guarded-import structure
    behaviorally (disabled collection costs the session nothing).
    """
    _patch_loop(monkeypatch)
    import sys

    monkeypatch.setitem(sys.modules, "victor.evaluation.field_data", None)

    class Loop:
        async def run_streaming(self, *args, **kwargs):
            yield StreamChunk(content="done", is_final=True)

    executor = _make_executor(monkeypatch, None, loop=Loop)
    chunks = await _consume(executor)
    assert chunks[-1].is_final is True
