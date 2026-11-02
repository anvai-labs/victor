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

"""Unit tests for the TUI sidebar (AgentState dataclass and AgentStatePanel)."""

from __future__ import annotations

from victor.ui.tui.sidebar import AgentState, AgentStatePanel, _format_tokens


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


class TestAgentStateDefaults:
    """Verify AgentState dataclass defaults."""

    def test_defaults(self) -> None:
        state = AgentState()
        assert state.session_id == "—"
        assert state.mode == "—"
        assert state.model == "—"
        assert state.provider == "—"
        assert state.cwd == "—"
        assert state.budget_used == 0
        assert state.budget_total is None
        assert state.context_pct is None
        assert state.context_used is None
        assert state.context_total is None


class TestBuildTable:
    """Verify the sidebar table rendering."""

    def test_budget_with_total(self) -> None:
        state = AgentState(budget_used=5, budget_total=25)
        table = AgentStatePanel._build_table(state)
        # Table should have rows for session, mode, model, provider, cwd, budget
        assert len(table.rows) == 6

    def test_budget_without_total(self) -> None:
        state = AgentState(budget_used=5, budget_total=None)
        table = AgentStatePanel._build_table(state)
        assert len(table.rows) == 6

    def test_context_with_used_and_total(self) -> None:
        state = AgentState(
            context_used=12500,
            context_total=200000,
            context_pct=6.0,
        )
        table = AgentStatePanel._build_table(state)
        # session, mode, model, provider, cwd, budget, context
        assert len(table.rows) == 7

    def test_context_pct_only(self) -> None:
        state = AgentState(context_pct=42.0)
        table = AgentStatePanel._build_table(state)
        assert len(table.rows) == 7

    def test_no_context(self) -> None:
        state = AgentState()
        table = AgentStatePanel._build_table(state)
        # session, mode, model, provider, cwd, budget (no context row)
        assert len(table.rows) == 6
