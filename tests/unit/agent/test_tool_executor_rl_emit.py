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


# --- RLM-2: structured outcome metadata drains into the event ----------------


def test_emits_drained_tool_outcome_metadata():
    """Fields attached by the tool land on the TOOL_EXECUTED event."""
    from victor.tools.rl_signal import (
        attach_tool_outcome_metadata,
        drain_tool_outcome_metadata,
        seed_tool_outcome_metadata,
    )

    seed_tool_outcome_metadata()
    attach_tool_outcome_metadata(error_category="not_found", return_code=127, duration_ms=11.0)
    hooks = _CaptureHooks()
    _emit(_make_executor(), hooks, success=False, result="### ❌ ERROR\nfoo: command not found")

    assert len(hooks.events) == 1
    metadata = hooks.events[0].metadata
    assert metadata["error_category"] == "not_found"
    assert metadata["return_code"] == 127
    assert metadata["duration_ms"] == 11.0
    assert metadata["execution_time"] == 0.05
    # Negative: the drain cleared the channel — nothing leaks to a later emit.
    assert drain_tool_outcome_metadata() == {}


def test_emits_without_attached_metadata_keeps_plain_event():
    from victor.tools.rl_signal import drain_tool_outcome_metadata, seed_tool_outcome_metadata

    seed_tool_outcome_metadata()  # seeded but nothing attached
    hooks = _CaptureHooks()
    _emit(_make_executor(), hooks, success=True, result={"success": True})

    metadata = hooks.events[0].metadata
    assert metadata == {"execution_time": 0.05}
    assert drain_tool_outcome_metadata() == {}
