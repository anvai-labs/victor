"""Unit tests for the StatusBar footer composition (queued segment)."""

from __future__ import annotations

from typing import Optional

from victor.ui.tui.status_bar import StatusBar


def _line(*, queued: int = 0, waiting_seconds: Optional[int] = None) -> str:
    return StatusBar._compose_line(
        phase_label="acting",
        tool_count=0,
        total_tokens=None,
        cost_usd=None,
        waiting_seconds=waiting_seconds,
        queued=queued,
    )


def test_queued_segment_omitted_when_zero() -> None:
    assert "queued" not in _line()
    assert "queued" not in _line(queued=0)


def test_queued_segment_renders_count() -> None:
    assert "1 queued" in _line(queued=1)
    assert "2 queued" in _line(queued=2)


def test_queued_segment_orders_after_tools() -> None:
    line = StatusBar._compose_line(
        phase_label="acting",
        tool_count=3,
        total_tokens=1234,
        cost_usd=0.5,
        waiting_seconds=None,
        queued=2,
    )
    assert line.index("3 tools") < line.index("2 queued")
    assert line.index("2 queued") < line.index("1.2k tok")
    assert line.index("1.2k tok") < line.index("$0.5000")


def test_waiting_indicator_still_wins_over_queued() -> None:
    line = _line(waiting_seconds=7, queued=4)
    assert "waiting on model (7s)" in line
    assert "4 queued" in line
