"""P7: the RL tool-outcome emitter must demote success for error-marker strings.

Unified tools report failures as ``### ❌`` strings with invocation-level
success=True; recording those as RL successes hid a 50.4% real error rate
from the Q-values.
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import patch

from victor.agent.tool_executor import ToolExecutor


class _CaptureHooks:
    def __init__(self) -> None:
        self.events: List[Any] = []

    def emit(self, event: Any) -> None:
        self.events.append(event)


def _make_executor() -> ToolExecutor:
    executor = ToolExecutor.__new__(ToolExecutor)
    return executor


def _emit(executor: ToolExecutor, hooks: _CaptureHooks, *, success: bool, result: Any):
    with patch("victor.agent.tool_executor._get_rl_hooks", return_value=hooks):
        executor._emit_rl_tool_event(
            "code", success, 0.05, {"provider": "zai", "model": "glm-5.2"}, result=result
        )


def test_error_marker_string_demotes_rl_success():
    hooks = _CaptureHooks()
    _emit(_make_executor(), hooks, success=True, result="### ❌ ERROR\nunrecognized arguments: -rn")
    assert len(hooks.events) == 1
    event = hooks.events[0]
    assert event.success is False
    assert event.quality_score <= 0.3


def test_warning_marker_keeps_rl_success_true():
    hooks = _CaptureHooks()
    _emit(_make_executor(), hooks, success=True, result="### ⚠️ SHELL OPERATOR NOT SUPPORTED\nhint")
    assert hooks.events[0].success is True


def test_dict_result_unchanged():
    hooks = _CaptureHooks()
    _emit(_make_executor(), hooks, success=True, result={"success": True, "stdout": "ok"})
    assert hooks.events[0].success is True


def test_none_result_unchanged():
    hooks = _CaptureHooks()
    _emit(_make_executor(), hooks, success=False, result=None)
    assert hooks.events[0].success is False
