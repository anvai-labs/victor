"""Mirror tests for runtime_overrides budget funding (consolidation PR 2).

extend_budget_funding is the single funding seam: with a ledger-hosting
pipeline it binds/retargets the run-scoped ledger (funding survives turn
end, undo restores the previous binding); without one it falls back to the
legacy per-turn owners exactly as before."""

from types import SimpleNamespace

from victor.agent.services.runtime_overrides import (
    apply_overrides,
    extend_budget_funding,
)


class TestLedgerFunding:
    def _owners_with_ledger_pipeline(self):
        from victor.core.tool_budget_ledger import ToolBudgetLedger

        ledger = ToolBudgetLedger(134)
        pipeline = SimpleNamespace(
            _tool_pipeline=SimpleNamespace(
                _budget_ledger=ledger,
                fund_budget=ledger.retarget,
                tool_budget=ledger.ceiling,
                calls_used=ledger.used,
            )
        )
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
        changes = []
        undo = extend_budget_funding(changes, pipeline, fallback, 134)
        assert pipeline._tool_pipeline._budget_ledger.ceiling == 134
        undo()
        assert (
            pipeline._tool_pipeline._budget_ledger.ceiling == 134
        ), "retarget keeps used; undo restores prior ceiling"
        assert pipeline._tool_pipeline._budget_ledger is ledger

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
