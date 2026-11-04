"""Unit tests for the TokenUsageBar widget."""

from __future__ import annotations


from victor.ui.tui.token_usage_bar import TokenUsageBar, _human_tokens


def test_human_tokens_small() -> None:
    """Test token formatting for small numbers."""
    assert _human_tokens(0) == "0"
    assert _human_tokens(1) == "1"
    assert _human_tokens(999) == "999"


def test_human_tokens_thousands() -> None:
    """Test token formatting for thousands."""
    assert _human_tokens(1000) == "1.0k"
    assert _human_tokens(1234) == "1.2k"
    assert _human_tokens(12345) == "12.3k"
    assert _human_tokens(100000) == "100.0k"


def test_human_tokens_edge_cases() -> None:
    """Test token formatting edge cases."""
    assert _human_tokens(999) == "999"
    assert _human_tokens(1000) == "1.0k"
    assert _human_tokens(10500) == "10.5k"


def test_token_usage_bar_no_data() -> None:
    """Test TokenUsageBar with no data."""
    bar = TokenUsageBar()
    bar.set_usage()
    # Should show "No usage data yet"
    content = str(bar.render())
    assert "No usage data yet" in content


def test_token_usage_bar_tokens_only() -> None:
    """Test TokenUsageBar with only tokens."""
    bar = TokenUsageBar()
    bar.set_usage(total_tokens=1234)
    content = str(bar.render())
    assert "turn" in content
    assert "1.2k" in content
    assert "session total" not in content
    assert "context" not in content


def test_token_usage_bar_cost_only() -> None:
    """Test TokenUsageBar with only cost."""
    bar = TokenUsageBar()
    bar.set_usage(cost_usd=0.0123)
    content = str(bar.render())
    assert "turn" in content
    assert "$0.0123" in content
    assert "context" not in content


def test_token_usage_bar_context_low() -> None:
    """Test TokenUsageBar with low context usage (green)."""
    bar = TokenUsageBar()
    bar.set_usage(context_used_pct=45.0)
    rendered = bar.render()
    content = str(rendered)
    assert "context" in content
    assert "45%" in content
    # Check that at least one span has green style
    assert any("green" in str(span.style) for span in rendered.spans)


def test_token_usage_bar_context_medium() -> None:
    """Test TokenUsageBar with medium context usage (yellow)."""
    bar = TokenUsageBar()
    bar.set_usage(context_used_pct=75.0)
    rendered = bar.render()
    content = str(rendered)
    assert "context" in content
    assert "75%" in content
    # Check that at least one span has yellow style
    assert any("yellow" in str(span.style) for span in rendered.spans)


def test_token_usage_bar_context_high() -> None:
    """Test TokenUsageBar with high context usage (red)."""
    bar = TokenUsageBar()
    bar.set_usage(context_used_pct=90.0)
    rendered = bar.render()
    content = str(rendered)
    assert "context" in content
    assert "90%" in content
    # Check that at least one span has red style
    assert any("red" in str(span.style) for span in rendered.spans)


def test_token_usage_bar_all_fields() -> None:
    """Test TokenUsageBar with all fields populated."""
    bar = TokenUsageBar()
    bar.set_usage(total_tokens=5000, cost_usd=0.05, context_used_pct=60.0)
    content = str(bar.render())
    assert "turn" in content
    assert "5.0k" in content
    assert "$0.0500" in content
    assert "context" in content
    assert "60%" in content


def test_token_usage_bar_separator() -> None:
    """Test that fields are separated by dots."""
    bar = TokenUsageBar()
    bar.set_usage(total_tokens=1000, cost_usd=0.01)
    content = str(bar.render())
    assert "  ·  " in content


def test_token_usage_bar_boundary_values() -> None:
    """Test context percentage boundary values."""
    # Just below green threshold
    bar = TokenUsageBar()
    bar.set_usage(context_used_pct=59.9)
    rendered = bar.render()
    assert any("green" in str(span.style) for span in rendered.spans)

    # At yellow threshold
    bar.set_usage(context_used_pct=60.0)
    rendered = bar.render()
    assert any("yellow" in str(span.style) for span in rendered.spans)

    # Just below red threshold
    bar.set_usage(context_used_pct=84.9)
    rendered = bar.render()
    assert any("yellow" in str(span.style) for span in rendered.spans)

    # At red threshold
    bar.set_usage(context_used_pct=85.0)
    rendered = bar.render()
    assert any("red" in str(span.style) for span in rendered.spans)


def test_token_usage_bar_cost_precision() -> None:
    """Test cost formatting precision."""
    bar = TokenUsageBar()
    bar.set_usage(cost_usd=0.0)
    assert "$0.0000" in str(bar.render())

    bar.set_usage(cost_usd=0.123456789)
    assert "$0.1235" in str(bar.render())  # Rounded to 4 decimals


def test_token_usage_bar_none_values() -> None:
    """Test that None values are handled correctly."""
    bar = TokenUsageBar()
    bar.set_usage(total_tokens=None, cost_usd=None, context_used_pct=None)
    content = str(bar.render())
    assert "No usage data yet" in content
    assert "Tokens:" not in content
    assert "Cost:" not in content
    assert "Context:" not in content


def test_token_usage_bar_mixed_none_values() -> None:
    """Test with some None and some actual values."""
    bar = TokenUsageBar()
    bar.set_usage(total_tokens=1000, cost_usd=None, context_used_pct=50.0)
    content = str(bar.render())
    assert "turn" in content
    assert "context" in content


async def test_line3_visible_when_mounted() -> None:
    """Mounted regression pin: the panel's box must fit all three lines.

    The adversarial review caught ``height: 3`` + ``border-top`` clipping the
    cost line to a 2-line content area (cost telemetry invisible). Rendered
    text asserts line 3 exists; the mounted region asserts the box fits it.
    """

    from textual.app import App

    class _BarApp(App[None]):
        def compose(self):
            yield TokenUsageBar(id="bar")

    app = _BarApp()
    async with app.run_test() as pilot:
        bar = app.query_one("#bar", TokenUsageBar)
        bar.set_usage(total_tokens=1234, cost_usd=0.05, session_cost_usd=0.02)
        await pilot.pause()
        content = str(bar.render())
        assert "last turn $0.0500" in content
        assert "session total $0.0200" in content
        assert bar.region.height >= 4, (
            f"panel box {bar.region.height} lines cannot show 3 content lines "
            "plus the border (the clipping regression)"
        )
