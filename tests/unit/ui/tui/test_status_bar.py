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


def test_compose_line_for_prompt_no_data() -> None:
    """Test compose_line_for_prompt with no data."""
    # Create a StatusBar instance to call the instance method
    bar = StatusBar.__new__(StatusBar)
    result = bar.compose_line_for_prompt()
    assert result == "[dim italic]No usage data yet[/]"


def test_compose_line_for_prompt_tokens_only() -> None:
    """Test compose_line_for_prompt with only tokens."""
    bar = StatusBar.__new__(StatusBar)
    result = bar.compose_line_for_prompt(total_tokens=1234)
    assert "[dim]Tokens:[/] [bold]1.2k[/]" in result
    assert "Cost:" not in result
    assert "Context:" not in result


def test_compose_line_for_prompt_cost_only() -> None:
    """Test compose_line_for_prompt with only cost."""
    bar = StatusBar.__new__(StatusBar)
    result = bar.compose_line_for_prompt(cost_usd=0.0123)
    assert "[dim]Cost:[/] [bold]$0.0123[/]" in result
    assert "Tokens:" not in result
    assert "Context:" not in result


def test_compose_line_for_prompt_context_low() -> None:
    """Test compose_line_for_prompt with low context usage (green)."""
    bar = StatusBar.__new__(StatusBar)
    result = bar.compose_line_for_prompt(context_used_pct=45.0)
    assert "[dim]Context:[/] [green]45%[/]" in result
    assert "Tokens:" not in result
    assert "Cost:" not in result


def test_compose_line_for_prompt_context_medium() -> None:
    """Test compose_line_for_prompt with medium context usage (yellow)."""
    bar = StatusBar.__new__(StatusBar)
    result = bar.compose_line_for_prompt(context_used_pct=75.0)
    assert "[dim]Context:[/] [yellow]75%[/]" in result


def test_compose_line_for_prompt_context_high() -> None:
    """Test compose_line_for_prompt with high context usage (red)."""
    bar = StatusBar.__new__(StatusBar)
    result = bar.compose_line_for_prompt(context_used_pct=90.0)
    assert "[dim]Context:[/] [red]90%[/]" in result


def test_compose_line_for_prompt_all_fields() -> None:
    """Test compose_line_for_prompt with all fields."""
    bar = StatusBar.__new__(StatusBar)
    result = bar.compose_line_for_prompt(total_tokens=5000, cost_usd=0.05, context_used_pct=60.0)
    assert "[dim]Tokens:[/] [bold]5.0k[/]" in result
    assert "[dim]Cost:[/] [bold]$0.0500[/]" in result
    assert "[dim]Context:[/] [yellow]60%[/]" in result
    assert "  ·  " in result
