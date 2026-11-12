"""Mirror tests for runtime_overrides budget funding (consolidation PR 2).

extend_budget_funding is the single funding seam: with a ledger-hosting
pipeline it binds/retargets the run-scoped ledger (funding survives turn
end, undo restores the previous binding); without one it falls back to the
legacy per-turn owners exactly as before."""

from types import SimpleNamespace

import pytest

from victor.agent.services.runtime_overrides import (
    OverrideRestorationError,
    apply_overrides,
    extend_budget_funding,
    restore_overrides,
)
from victor.agent.tool_pipeline import ToolPipeline
from victor.core.tool_budget_ledger import ToolBudgetLedger


class TestLedgerFunding:
    def _owners_with_ledger_pipeline(self):
        ledger = ToolBudgetLedger(134)
        tool_pipeline = object.__new__(ToolPipeline)
        tool_pipeline._budget_ledger = ledger
        pipeline = SimpleNamespace(_tool_pipeline=tool_pipeline)
        fallback = SimpleNamespace(_tool_pipeline=pipeline._tool_pipeline)
        return pipeline, fallback, ledger

    def test_funds_fresh_ledger_without_prior_usage_subtraction(self):
        pipeline, fallback, ledger = self._owners_with_ledger_pipeline()
        changes = []
        undo = extend_budget_funding(changes, pipeline, fallback, 134)
        assert ledger.ceiling == 134
        assert ledger.used == 0, "prior session usage must not subtract"
        assert not changes, "ledger mode collects no legacy per-turn owners"
        assert undo is not None

    def test_undo_restores_previous_binding(self):
        pipeline, fallback, ledger = self._owners_with_ledger_pipeline()
        ledger.consume(3)
        changes = []
        undo = extend_budget_funding(changes, pipeline, fallback, 20)
        assert pipeline._tool_pipeline._budget_ledger.ceiling == 20
        undo()
        assert (
            pipeline._tool_pipeline._budget_ledger.ceiling == 134
        ), "retarget keeps used; undo restores prior ceiling"
        assert pipeline._tool_pipeline._budget_ledger is ledger
        assert ledger.used == 3

    @pytest.mark.parametrize("previous_ceiling", [None, 10])
    @pytest.mark.parametrize("failure_stage", ["getter", "setter", "restore"])
    def test_failed_apply_restores_funding(self, previous_ceiling, failure_stage):
        tool_pipeline = object.__new__(ToolPipeline)
        previous = None if previous_ceiling is None else ToolBudgetLedger(previous_ceiling)
        if previous is not None:
            previous.consume(3)
        tool_pipeline._budget_ledger = previous
        owner = SimpleNamespace(_tool_pipeline=tool_pipeline)
        undo = extend_budget_funding([], owner, owner, 20)
        state = {"value": 1}

        def getter():
            if failure_stage == "getter":
                raise RuntimeError("getter failed")
            return state["value"]

        def setter(value):
            state["value"] = value
            if value == 2 or failure_stage == "restore":
                raise RuntimeError("setter failed")

        error = OverrideRestorationError if failure_stage == "restore" else RuntimeError
        with pytest.raises(error):
            apply_overrides([("owner", getter, setter, 2)], extra_undo=undo)
        assert tool_pipeline._budget_ledger is previous
        if previous is not None:
            assert previous.ceiling == 10
            assert previous.used == 3
        assert state["value"] == 1

    def test_successful_apply_keeps_funding_after_turn_restore(self):
        owner, fallback, ledger = self._owners_with_ledger_pipeline()
        ledger.consume(3)
        undo = extend_budget_funding([], owner, fallback, 20)
        snapshot = apply_overrides([], extra_undo=undo)
        restore_overrides(snapshot)
        assert owner._tool_pipeline._budget_ledger is ledger
        assert ledger.ceiling == 20
        assert ledger.used == 3

    def test_ledger_absent_falls_back_to_legacy_owners(self):
        orchestrator = SimpleNamespace(tool_budget=25, _tool_pipeline=None)
        fallback = SimpleNamespace(tool_budget=25, _tool_pipeline=None)
        changes = []
        undo = extend_budget_funding(changes, orchestrator, fallback, 134)
        assert undo is None
        assert changes, "legacy mode collects per-turn owner changes"

    def test_apply_overrides_runs_extra_undo_on_change_failure(self):
        undo_calls = []

        def ok_setter(value):
            pass

        def failing_setter(value):
            if value == 60:
                raise RuntimeError("boom")
            # restore path passes the previous value (2) and must succeed,
            # so the rollback completes and extra_undo runs before the raise

        def undo():
            undo_calls.append(1)

        changes = [
            ("owner", (lambda: 1), ok_setter, 50),
            ("late", (lambda: 2), failing_setter, 60),
        ]
        with __import__("pytest").raises(RuntimeError):
            apply_overrides(changes, extra_undo=undo)
        assert undo_calls == [1], "failed apply must still run the funding undo"
        assert undo_calls == [1], "failed apply must still run the funding undo"
