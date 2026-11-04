"""Unit tests for the StatusBar footer composition (queued segment).

The TUI rework consolidated model/provider/profile into the header and the
TokenUsageBar; ``StatusBar._compose_line`` now composes phase · tools · queued ·
session only. The former ``compose_line_for_prompt`` token/cost/context tests
live in ``test_token_usage_bar.py`` against the 3-line panel.
"""

from __future__ import annotations

from typing import Optional

from victor.ui.tui.status_bar import StatusBar


def _line(*, queued: int = 0, waiting_seconds: Optional[int] = None) -> str:
    return StatusBar._compose_line(
        phase_label="acting",
        tool_count=0,
        waiting_seconds=waiting_seconds,
        queued=queued,
    )


def test_queued_segment_omitted_when_zero() -> None:
    assert "queued" not in _line()
    assert "queued" not in _line(queued=0)


def test_queued_segment_renders_count() -> None:
    assert "1 queued" in _line(queued=1)
    assert "2 queued" in _line(queued=2)


def test_queued_segment_orders_after_tools_before_session() -> None:
    line = StatusBar._compose_line(
        phase_label="acting",
        tool_count=3,
        waiting_seconds=None,
        queued=2,
        session_id="sess-9",
    )
    assert line.index("3 tools") < line.index("2 queued")
    assert line.index("2 queued") < line.index("sess-9")


def test_waiting_indicator_still_wins_over_queued() -> None:
    line = _line(waiting_seconds=7, queued=4)
    assert "waiting (7s)" in line
    assert "4 queued" in line


def test_model_provider_profile_stay_consolidated_out() -> None:
    """The rework's contract: identity fields render in header/tokenbar, not here."""
    line = StatusBar._compose_line(
        phase_label="acting",
        tool_count=1,
        waiting_seconds=None,
        model="glm-5.3",
        provider="zai",
        profile="zai-glm53",
    )
    assert "glm-5.3" not in line
    assert "zai" not in line
