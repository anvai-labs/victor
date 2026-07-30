"""Unit tests for the inferred phase tracker (pure, no Textual)."""

from __future__ import annotations

from victor.ui.chat_app.event_mapping import RenderAction, RenderKind
from victor.ui.tui.phase import Phase, PhaseTracker


def _token(text: str = "x") -> RenderAction:
    return RenderAction(RenderKind.TOKEN, text=text)


def _thinking() -> RenderAction:
    return RenderAction(RenderKind.THINKING, text="reasoning…")


def _tool_start(name: str = "read") -> RenderAction:
    return RenderAction(RenderKind.TOOL_START, tool_name=name)


def _tool_end(name: str = "read", success: bool = True) -> RenderAction:
    return RenderAction(RenderKind.TOOL_END, tool_name=name, success=success)


def test_idle_by_default() -> None:
    assert PhaseTracker().phase is Phase.IDLE


def test_begin_turn_is_waiting() -> None:
    tracker = PhaseTracker()
    tracker.begin_turn()
    assert tracker.phase is Phase.WAITING
    assert tracker.label() == "working…"


def test_thinking_is_planning() -> None:
    tracker = PhaseTracker()
    tracker.begin_turn()
    tracker.update(_thinking())
    assert tracker.phase is Phase.PLANNING
    assert tracker.label() == "planning"


def test_tool_start_is_acting_with_name_and_count() -> None:
    tracker = PhaseTracker()
    tracker.begin_turn()
    tracker.update(_tool_start("grep"))
    assert tracker.phase is Phase.ACTING
    assert tracker.active_tool == "grep"
    assert tracker.tool_count == 1
    assert tracker.label() == "acting · grep"


def test_tool_end_clears_active_tool_but_stays_acting() -> None:
    tracker = PhaseTracker()
    tracker.begin_turn()
    tracker.update(_tool_start("grep"))
    tracker.update(_tool_end("grep"))
    assert tracker.phase is Phase.ACTING
    assert tracker.active_tool is None
    assert tracker.label() == "acting"


def test_token_after_tools_is_responding() -> None:
    tracker = PhaseTracker()
    tracker.begin_turn()
    tracker.update(_tool_start())
    tracker.update(_tool_end())
    tracker.update(_token("answer"))
    assert tracker.phase is Phase.RESPONDING
    assert tracker.label() == "responding"


def test_empty_token_does_not_flip_to_responding() -> None:
    tracker = PhaseTracker()
    tracker.begin_turn()
    tracker.update(_thinking())
    tracker.update(_token(""))  # no text
    assert tracker.phase is Phase.PLANNING


def test_multiple_tools_increment_count() -> None:
    tracker = PhaseTracker()
    tracker.begin_turn()
    tracker.update(_tool_start("a"))
    tracker.update(_tool_end("a"))
    tracker.update(_tool_start("b"))
    assert tracker.tool_count == 2


def test_end_then_reset() -> None:
    tracker = PhaseTracker()
    tracker.begin_turn()
    tracker.update(_token("hi"))
    tracker.end_turn()
    assert tracker.phase is Phase.DONE
    tracker.reset()
    assert tracker.phase is Phase.IDLE
    assert tracker.tool_count == 0
