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

"""Unit tests for the TUI session bar (SessionInfo dataclass and SessionBar)."""

from __future__ import annotations

from victor.ui.tui.session_bar import SessionBar, SessionInfo, TokenUsage, _format_tokens


class TestFormatTokens:
    """Tests for the _format_tokens helper."""

    def test_small_values_shown_as_is(self) -> None:
        assert _format_tokens(0) == "0"
        assert _format_tokens(500) == "500"
        assert _format_tokens(999) == "999"

    def test_thousands_use_k_suffix(self) -> None:
        assert _format_tokens(1000) == "1.0K"
        assert _format_tokens(1500) == "1.5K"
        assert _format_tokens(200000) == "200.0K"


class TestSessionInfoDefaults:
    """Verify SessionInfo dataclass defaults."""

    def test_defaults(self) -> None:
        info = SessionInfo()
        assert info.session_id == "—"
        assert info.mode == "—"
        assert info.model == "—"
        assert info.provider == "—"
        assert info.profile == "—"
        assert info.budget_used == 0
        assert info.budget_total is None
        assert info.context_pct is None
        assert info.context_used is None
        assert info.context_total is None
        assert info.token_usage.prompt_tokens == 0
        assert info.cost_usd is None


class TestTokenUsage:
    """Verify TokenUsage dataclass."""

    def test_defaults(self) -> None:
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.reasoning_tokens == 0
        assert usage.cached_tokens == 0
        assert usage.total_tokens == 0

    def test_to_dict(self) -> None:
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            reasoning_tokens=20,
            cached_tokens=10,
            total_tokens=180,
        )
        result = usage.to_dict()
        assert result == {
            "prompt": 100,
            "completion": 50,
            "reasoning": 20,
            "cached": 10,
            "total": 180,
        }


class TestBuildLine:
    """Verify the session bar line rendering."""

    def test_budget_with_total(self) -> None:
        info = SessionInfo(budget_used=5, budget_total=25)
        line = SessionBar._build_line(info)
        # Should contain budget info
        assert "5/25" in line

    def test_budget_without_total(self) -> None:
        info = SessionInfo(budget_used=5, budget_total=None)
        line = SessionBar._build_line(info)
        # Should NOT show budget when total is None (tool budget not set)
        assert "budget" not in line.lower()

    def test_context_with_used_and_total(self) -> None:
        info = SessionInfo(
            context_used=12500,
            context_total=200000,
            context_pct=6.25,
        )
        line = SessionBar._build_line(info)
        # Should contain context info with percentage
        assert "12.5K/200.0K" in line
        assert "6.2%" in line

    def test_context_pct_only(self) -> None:
        info = SessionInfo(context_pct=42.5)
        line = SessionBar._build_line(info)
        # Should contain percentage
        assert "42.5%" in line

    def test_context_color_coding_low(self) -> None:
        """Context < 70% should be green."""
        info = SessionInfo(context_used=50000, context_total=200000, context_pct=25.0)
        line = SessionBar._build_line(info)
        assert "[green]" in line

    def test_context_color_coding_medium(self) -> None:
        """Context 70-90% should be yellow."""
        info = SessionInfo(context_used=160000, context_total=200000, context_pct=80.0)
        line = SessionBar._build_line(info)
        assert "[yellow]" in line

    def test_context_color_coding_high(self) -> None:
        """Context > 90% should be red."""
        info = SessionInfo(context_used=190000, context_total=200000, context_pct=95.0)
        line = SessionBar._build_line(info)
        assert "[red]" in line

    def test_token_usage_display(self) -> None:
        """Token usage should be displayed with bright_magenta."""
        info = SessionInfo(
            token_usage=TokenUsage(
                prompt_tokens=1000,
                completion_tokens=500,
                reasoning_tokens=200,
                cached_tokens=100,
                total_tokens=1800,
            )
        )
        line = SessionBar._build_line(info)
        # Values are wrapped in bright_magenta markup
        assert "prompt=[bright_magenta]1.0K[/]" in line
        assert "completion=[bright_magenta]500[/]" in line
        assert "reasoning=[bright_magenta]200[/]" in line
        assert "cached=[bright_magenta]100[/]" in line
        assert "[bright_magenta]" in line

    def test_cost_display(self) -> None:
        """Cost should be displayed with bright_yellow."""
        info = SessionInfo(cost_usd=0.0123)
        line = SessionBar._build_line(info)
        assert "$0.0123" in line
        assert "[bright_yellow]" in line

    def test_static_info_uses_white(self) -> None:
        """Static info (session, mode, profile, model, provider) should use white."""
        info = SessionInfo(
            session_id="abc12345",
            mode="build",
            profile="test-profile",
            model="gpt-4",
            provider="openai",
        )
        line = SessionBar._build_line(info)
        # Check that static info is rendered with white color
        assert "[white]" in line
        assert "abc12345" in line
        assert "build" in line
        assert "test-profile" in line
        assert "gpt-4" in line
        assert "openai" in line

    def test_session_id_truncation(self) -> None:
        """Long session IDs should be truncated to 8 characters."""
        info = SessionInfo(session_id="abcdefghij123456")
        line = SessionBar._build_line(info)
        assert "abcdefgh" in line
        assert "abcdefghij123456" not in line

    def test_no_context(self) -> None:
        info = SessionInfo()
        line = SessionBar._build_line(info)
        # Should not contain context info
        assert "context:" not in line

    def test_no_token_usage(self) -> None:
        info = SessionInfo()
        line = SessionBar._build_line(info)
        # Should not contain token breakdown
        assert "tokens:" not in line
