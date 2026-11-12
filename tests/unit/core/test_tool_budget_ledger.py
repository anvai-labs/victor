"""Unit tests for the run-scoped ToolBudgetLedger (seam-duplication audit Cluster 1).

One object per run owns (ceiling, used, scope). Enforcement sites read
through the ledger when one is bound; ``retarget`` moves the ceiling
without resetting usage, and ``reset`` preserves the interactive-turn
replenish semantics.
"""

import pytest

from victor.core.tool_budget_ledger import ToolBudgetLedger


class TestToolBudgetLedgerSemantics:
    def test_initial_state(self):
        ledger = ToolBudgetLedger(134)

        assert ledger.ceiling == 134
        assert ledger.used == 0
        assert ledger.remaining == 134
        assert ledger.is_exhausted is False
        assert ledger.scope == "run"

    def test_custom_scope(self):
        ledger = ToolBudgetLedger(10, scope="turn")

        assert ledger.scope == "turn"

    def test_consume_advances_used_and_remaining(self):
        ledger = ToolBudgetLedger(134)

        ledger.consume()
        ledger.consume(4)

        assert ledger.used == 5
        assert ledger.remaining == 129
        assert ledger.is_exhausted is False

    def test_consume_rejects_negative_amount(self):
        ledger = ToolBudgetLedger(10)

        with pytest.raises(ValueError):
            ledger.consume(-1)

    def test_exhausted_at_ceiling(self):
        ledger = ToolBudgetLedger(3)
        ledger.consume(3)

        assert ledger.used == 3
        assert ledger.remaining == 0
        assert ledger.is_exhausted is True

    def test_over_consume_clamps_remaining_at_zero(self):
        ledger = ToolBudgetLedger(3)
        ledger.consume(5)

        assert ledger.used == 5
        assert ledger.remaining == 0
        assert ledger.is_exhausted is True

    def test_zero_ceiling_is_exhausted_from_the_start(self):
        ledger = ToolBudgetLedger(0)

        assert ledger.remaining == 0
        assert ledger.is_exhausted is True

    def test_construction_rejects_negative_ceiling(self):
        with pytest.raises(ValueError):
            ToolBudgetLedger(-1)


class TestToolBudgetLedgerRetarget:
    """``retarget`` is the caller re-funding seam: the ceiling moves, usage persists."""

    def test_retarget_moves_ceiling_keeps_used(self):
        ledger = ToolBudgetLedger(134)
        ledger.consume(20)

        ledger.retarget(200)

        assert ledger.ceiling == 200
        assert ledger.used == 20
        assert ledger.remaining == 180
        assert ledger.is_exhausted is False

    def test_retarget_below_used_reports_exhausted(self):
        ledger = ToolBudgetLedger(134)
        ledger.consume(50)

        ledger.retarget(10)

        assert ledger.ceiling == 10
        assert ledger.used == 50
        assert ledger.is_exhausted is True

    def test_retarget_rejects_negative_ceiling(self):
        ledger = ToolBudgetLedger(10)

        with pytest.raises(ValueError):
            ledger.retarget(-5)


class TestToolBudgetLedgerReset:
    """``reset`` preserves interactive-turn replenish semantics."""

    def test_reset_clears_used_keeps_ceiling(self):
        ledger = ToolBudgetLedger(134)
        ledger.consume(130)

        ledger.reset()

        assert ledger.ceiling == 134
        assert ledger.used == 0
        assert ledger.remaining == 134
        assert ledger.is_exhausted is False
