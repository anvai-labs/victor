"""TDD tests for the RL budget-signal persistence adapter.

Phase 2 of RL-driven tool-budget calibration: a read-only adapter that loads
:class:`ToolBudgetSignals` and :class:`DecisionOutcomeAggregate` from the
global Victor database (``~/.victor/victor.db``) using the *existing* data
access idiom (``db.cursor()`` + ``Tables.*`` constants + ``dict(row)``).

Design (reuse-first, no new SQLite path):
* The adapter is a thin reader; it never writes and never owns a connection.
* ``db`` is injected for testability and lazily defaults to the canonical
  ``get_database()`` accessor — matching the idiom in
  ``victor/framework/rl/learners/tool_selector.py``.
* Missing/empty tables degrade to empty signals (cold-start safe), mirroring
  the try/except guards used elsewhere in the RL layer.

Schemas (from ``victor/core/schema.py``):
    rl_tool_q:      tool_name, q_value, selection_count, success_count
    decision_outcome: success, quality_score, attributed_reward
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from victor.framework.rl.budget_calibration import (
    DecisionOutcomeAggregate,
    ToolBudgetSignals,
)
from victor.framework.rl.budget_signal_reader import BudgetSignalReader

# ---------------------------------------------------------------------------
# Test doubles mirroring the db.cursor() / fetchall() / dict(row) idiom
# ---------------------------------------------------------------------------


class FakeDB:
    """Minimal DB handle with cursor()."""

    def __init__(
        self,
        tool_rows: List[Dict[str, Any]],
        outcome_rows: List[Dict[str, Any]],
        unified_rows: List[Dict[str, Any]] | None = None,
    ):
        self._tool_rows = tool_rows
        self._outcome_rows = outcome_rows
        self._unified_rows = unified_rows or []

    def cursor(self) -> "_RoutingCursor":
        # Hand back a fresh cursor seeded by the *next* execute's intent.
        # We disambiguate via the query string the reader emits.
        c = _RoutingCursor(self._tool_rows, self._outcome_rows, self._unified_rows)
        return c  # type: ignore[return-value]


class _Row:
    """Row wrapper that supports dict(row), matching sqlite3.Row."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key: str):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)


class _RoutingCursor:
    """Returns tool rows for rl_tool_q queries, outcome rows otherwise."""

    def __init__(self, tool_rows, outcome_rows, unified_rows=None):
        self._tool = tool_rows
        self._outcome = outcome_rows
        self._unified = unified_rows or []

    def execute(self, query: str, params: Any = ()):
        if "rl_q_value" in query:
            self._current = self._unified
        elif "rl_tool_q" in query:
            self._current = self._tool
        else:
            self._current = self._outcome
        return self

    def fetchall(self):
        return [_Row(r) for r in self._current]


# ---------------------------------------------------------------------------
# ToolBudgetSignals loading
# ---------------------------------------------------------------------------

TOOL_ROWS = [
    {
        "tool_name": "read",
        "q_value": 0.918,
        "selection_count": 6189,
        "success_count": 5855,
    },
    {
        "tool_name": "shell",
        "q_value": 0.360,
        "selection_count": 271,
        "success_count": 120,
    },
]


UNIFIED_ROWS = [
    {
        "tool_name": "code",
        "q_value": 0.42,
        "selection_count": 6330,
        "success_count": 3100,
    },
    {
        "tool_name": "git",
        "q_value": 0.61,
        "selection_count": 489,
        "success_count": 290,
    },
]


class TestLoadToolSignalsUnified:
    """P7: the reader must consume the LIVE unified tables (rl_q_value +
    rl_task_stat), not the legacy rl_tool_q frozen at the v0.7.0 migration."""

    def test_load_tool_signals_reads_unified_tables(self):
        db = FakeDB(tool_rows=TOOL_ROWS, outcome_rows=[], unified_rows=UNIFIED_ROWS)
        signals = BudgetSignalReader().load_tool_signals(db=db)
        by_name = {s.tool_name: s for s in signals}
        assert "code" in by_name, "unified rows must win over legacy"
        assert by_name["code"].selection_count == 6330
        assert abs(by_name["code"].success_rate - 3100 / 6330) < 1e-9
        assert "read" not in by_name, "legacy rows must not be mixed in when unified has data"

    def test_load_tool_signals_falls_back_to_legacy_when_unified_empty(self):
        db = FakeDB(tool_rows=TOOL_ROWS, outcome_rows=[], unified_rows=[])
        signals = BudgetSignalReader().load_tool_signals(db=db)
        names = {s.tool_name for s in signals}
        assert names == {"read", "shell"}


class TestLoadToolSignals:
    def test_loads_tools_from_rows(self):
        db = FakeDB(TOOL_ROWS, [])
        signals = BudgetSignalReader().load_tool_signals(db=db)  # type: ignore[arg-type]
        assert len(signals) == 2
        by_name = {s.tool_name: s for s in signals}
        assert by_name["read"].q_value == pytest.approx(0.918)
        assert by_name["read"].success_rate == pytest.approx(5855 / 6189, abs=0.01)
        assert by_name["shell"].selection_count == 271

    def test_empty_table_returns_empty_tuple(self):
        db = FakeDB([], [])
        assert BudgetSignalReader().load_tool_signals(db=db) == ()  # type: ignore[arg-type]

    def test_clamps_q_value_into_range(self):
        # Defends against out-of-range data (e.g. legacy/0.0 default rows).
        rows = [
            {
                "tool_name": "x",
                "q_value": 1.5,
                "selection_count": 10,
                "success_count": 8,
            }
        ]
        db = FakeDB(rows, [])
        signals = BudgetSignalReader().load_tool_signals(db=db)  # type: ignore[arg-type]
        assert all(0.0 <= s.q_value <= 1.0 for s in signals)

    def test_missing_success_count_treated_as_zero(self):
        rows = [{"tool_name": "y", "q_value": 0.5, "selection_count": 100}]
        db = FakeDB(rows, [])
        signals = BudgetSignalReader().load_tool_signals(db=db)  # type: ignore[arg-type]
        assert signals[0].success_rate == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------------------
# DecisionOutcomeAggregate loading
# ---------------------------------------------------------------------------

OUTCOME_ROWS = [
    {"success": 1, "quality_score": 0.9, "attributed_reward": 1.0},
    {"success": 1, "quality_score": 0.8, "attributed_reward": 1.0},
    {"success": 0, "quality_score": 0.1, "attributed_reward": 0.05},
    {"success": 0, "quality_score": 0.0, "attributed_reward": 0.0},
]


class TestLoadDecisionOutcomes:
    def test_aggregates_success_failure_reward(self):
        db = FakeDB([], OUTCOME_ROWS)
        agg = BudgetSignalReader().load_decision_outcomes(db=db)  # type: ignore[arg-type]
        assert agg.total == 4
        assert agg.successes == 2
        assert agg.failures == 2
        # mean of (1.0, 1.0, 0.05, 0.0) = 0.5125
        assert agg.mean_reward == pytest.approx(0.5125, abs=0.01)

    def test_empty_table_is_cold_start_safe(self):
        db = FakeDB([], [])
        agg = BudgetSignalReader().load_decision_outcomes(db=db)  # type: ignore[arg-type]
        assert agg.total == 0
        assert agg.is_reliable() is False
        assert agg.failure_rate == 0.0

    def test_treats_null_success_as_failure(self):
        rows = [{"success": None, "quality_score": None, "attributed_reward": None}]
        db = FakeDB([], rows)
        agg = BudgetSignalReader().load_decision_outcomes(db=db)  # type: ignore[arg-type]
        assert agg.total == 1
        assert agg.successes == 0
        assert agg.failures == 1
        assert agg.mean_reward == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------------------
# End-to-end: reader feeds the calibrator
# ---------------------------------------------------------------------------


class TestReaderFeedsCalibrator:
    def test_full_pipeline_produces_recommendation(self):
        from victor.config.tool_settings import ToolSettings

        from victor.framework.rl.budget_calibration import BudgetCalibrator

        db = FakeDB(TOOL_ROWS, OUTCOME_ROWS)
        reader = BudgetSignalReader()
        tools = reader.load_tool_signals(db=db)  # type: ignore[arg-type]
        decisions = reader.load_decision_outcomes(db=db)  # type: ignore[arg-type]
        rec = BudgetCalibrator().recommend(
            tools=tools, decisions=decisions, baseline=ToolSettings()
        )
        # 50% failure rate, real tool signals -> should produce a calibrated budget.
        assert rec.recommended_tool_call_budget > 0
        assert "read" in rec.relief_eligible_tools
        assert rec.confidence > 0.0
