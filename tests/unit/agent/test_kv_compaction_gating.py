"""KV prefix stability: compaction gated to user-turn boundaries (F14)."""

from __future__ import annotations

import asyncio
from typing import Optional

from victor.agent.context_compactor import CompactionTrigger
from victor.agent.context_manager import _kv_aware_compaction_monitor
from victor.agent.message_history import MessageHistory
from victor.providers.base import Message


class FakeCompactor:
    """Compactor double: scripted trigger, counts compaction runs."""

    def __init__(self, trigger: CompactionTrigger):
        self.trigger = trigger
        self.compact_calls = 0
        self.controller = None  # history sourced via controller._history

    async def should_compact_async(self):
        return (self.trigger != CompactionTrigger.NONE), self.trigger

    def should_compact(self, task_complexity=None):
        return (self.trigger != CompactionTrigger.NONE), self.trigger

    stopped = 0

    async def stop_background_compaction(self):
        type(self).stopped = getattr(type(self), "stopped", 0) + 1

    async def check_and_compact_async(self):
        self.compact_calls += 1


class FakeController:
    def __init__(self, messages):
        self._history = _FakeHistory(messages)


class _FakeHistory:
    def __init__(self, messages):
        self._messages = messages


def _mid_turn_messages() -> list[Message]:
    return [
        Message(role="user", content="task"),
        Message(role="assistant", content="partial work"),
        Message(role="tool", content="TOOL_OUTPUT mid", tool_call_id="c1"),
    ]


def _boundary_messages() -> list[Message]:
    return [Message(role="user", content="task")]


async def _run_one_tick(compactor, messages) -> None:
    task = asyncio.create_task(_kv_aware_compaction_monitor(compactor, interval_seconds=0.01))
    await asyncio.sleep(0.08)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_monitor_defers_mid_user_turn():
    compactor = FakeCompactor(CompactionTrigger.THRESHOLD)
    compactor.controller = FakeController(_mid_turn_messages())
    await _run_one_tick(compactor, compactor.controller._history._messages)
    assert compactor.compact_calls == 0, "mid-turn compaction mutates sent bytes"


async def test_monitor_fires_at_turn_boundary():
    compactor = FakeCompactor(CompactionTrigger.THRESHOLD)
    compactor.controller = FakeController(_boundary_messages())
    await _run_one_tick(compactor, compactor.controller._history._messages)
    assert compactor.compact_calls >= 1


async def test_monitor_overflow_overrides_mid_turn():
    compactor = FakeCompactor(CompactionTrigger.OVERFLOW)
    compactor.controller = FakeController(_mid_turn_messages())
    await _run_one_tick(compactor, compactor.controller._history._messages)
    assert compactor.compact_calls >= 1, "overflow beats cache stability"


async def test_monitor_none_trigger_never_compacts():
    compactor = FakeCompactor(CompactionTrigger.NONE)
    compactor.controller = FakeController(_boundary_messages())
    await _run_one_tick(compactor, compactor.controller._history._messages)
    assert compactor.compact_calls == 0


async def test_executor_inline_check_defers_mid_turn(monkeypatch):
    from victor.agent.services.turn_execution_runtime import TurnExecutor

    runtime = object.__new__(TurnExecutor)
    history = MessageHistory()
    history._messages = list(_mid_turn_messages())
    history._context_compactor = FakeCompactor(CompactionTrigger.THRESHOLD)
    runtime._chat_context = history

    lifecycle_called = []

    async def _fake_lifecycle(_msg):
        lifecycle_called.append(_msg)
        return False

    runtime._check_lifecycle_context_compaction = _fake_lifecycle  # type: ignore[method-assign]
    runtime._resolve_context_service = lambda: None  # type: ignore[method-assign]

    class _Classification:
        class complexity:
            value = "medium"

    await TurnExecutor._check_context_compaction(runtime, "task", _Classification())
    assert lifecycle_called == [], "mid-turn inline check must defer before any compaction path"


async def test_executor_inline_check_runs_at_boundary(monkeypatch):
    from victor.agent.services.turn_execution_runtime import TurnExecutor

    runtime = object.__new__(TurnExecutor)
    history = MessageHistory()
    history._messages = list(_boundary_messages())
    history._context_compactor = FakeCompactor(CompactionTrigger.THRESHOLD)
    runtime._chat_context = history

    lifecycle_called = []

    async def _fake_lifecycle(_msg):
        lifecycle_called.append(_msg)
        return True  # lifecycle path handled it

    runtime._check_lifecycle_context_compaction = _fake_lifecycle  # type: ignore[method-assign]

    class _Classification:
        class complexity:
            value = "medium"

    await TurnExecutor._check_context_compaction(runtime, "task", _Classification())
    assert lifecycle_called == ["task"], "boundary check must reach the compaction paths"


# ── changed-line coverage: wrappers, probe paths, monitor survival ─────


async def test_context_manager_start_stop_wrappers():
    from victor.agent.context_manager import ContextManager

    manager = object.__new__(ContextManager)
    manager._context_compactor = FakeCompactor(CompactionTrigger.THRESHOLD)
    await ContextManager.start_background_compaction(manager, interval_seconds=0.01)
    try:
        task = getattr(manager, "_kv_monitor_task", None)
        assert task is not None and not task.done()
    finally:
        await ContextManager.stop_background_compaction(manager)
    assert getattr(manager, "_kv_monitor_task", None) is None


async def test_monitor_survives_compactor_errors():
    class ExplodingCompactor(FakeCompactor):
        async def should_compact_async(self):
            raise RuntimeError("probe failure")

    compactor = ExplodingCompactor(CompactionTrigger.THRESHOLD)
    compactor.controller = FakeController(_boundary_messages())
    task = asyncio.create_task(_kv_aware_compaction_monitor(compactor, interval_seconds=0.01))
    await asyncio.sleep(0.08)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert compactor.compact_calls == 0  # survived without compacting


async def test_executor_overflow_probe_exception_defaults_to_defer():
    from victor.agent.services.turn_execution_runtime import TurnExecutor

    class ExplodingProbe(FakeCompactor):
        def should_compact(self, task_complexity=None):
            raise RuntimeError("probe failure")

    runtime = object.__new__(TurnExecutor)
    history = MessageHistory()
    history._messages = list(_mid_turn_messages())
    history._context_compactor = ExplodingProbe(CompactionTrigger.THRESHOLD)
    runtime._chat_context = history
    runtime._check_lifecycle_context_compaction = lambda _msg: _always_false()
    runtime._resolve_context_service = lambda: None

    class _Classification:
        class complexity:
            value = "medium"

    await TurnExecutor._check_context_compaction(runtime, "task", _Classification())


async def _always_false():
    return False


async def test_executor_mid_turn_overflow_overrides_and_compacts():
    from victor.agent.services.turn_execution_runtime import TurnExecutor

    runtime = object.__new__(TurnExecutor)
    history = MessageHistory()
    history._messages = list(_mid_turn_messages())
    history._context_compactor = FakeCompactor(CompactionTrigger.OVERFLOW)
    runtime._chat_context = history
    lifecycle = []

    async def _fake_lifecycle(_msg):
        lifecycle.append(_msg)
        return True

    runtime._check_lifecycle_context_compaction = _fake_lifecycle
    runtime._resolve_context_service = lambda: None

    class _Classification:
        class complexity:
            value = "medium"

    await TurnExecutor._check_context_compaction(runtime, "task", _Classification())
    assert lifecycle == ["task"], "overflow override must reach compaction paths"
