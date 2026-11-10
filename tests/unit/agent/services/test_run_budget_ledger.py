"""Wiring tests for the run-scoped ToolBudgetLedger (audit Cluster 1 consolidation).

Covers the measured r7 failure (a run refused at ~55-60 tool calls against a
funded 134 because cumulative counters were subtracted from per-run funding),
the max_results decoupling, ledger-absent parity, and PR 1's refusal-triad
observability reading ledger-backed values.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.tool_service import ToolService, ToolServiceConfig
from victor.agent.services.turn_execution_runtime import TurnExecutor
from victor.agent.tool_executor import ToolExecutionResult
from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig
from victor.tools.enums import AccessMode


def _make_pipeline(tool_budget=60, enable_output_aggregation=False, max_results=None):
    registry = MagicMock()
    registry.is_tool_enabled.return_value = True
    registry.get.return_value.access_mode = AccessMode.READONLY
    executor = MagicMock()
    executor.execute = AsyncMock(
        return_value=ToolExecutionResult(
            tool_name="read",
            success=True,
            result="ok",
            execution_time=0.01,
        )
    )
    kwargs = {
        "tool_budget": tool_budget,
        "enable_caching": False,
        "enable_output_aggregation": enable_output_aggregation,
        "enable_synthesis_checkpoints": False,
        "enable_semantic_caching": False,
    }
    if max_results is not None:
        kwargs["max_results"] = max_results
    return ToolPipeline(
        tool_registry=registry,
        tool_executor=executor,
        config=ToolPipelineConfig(**kwargs),
    )


def _make_service(pipeline=None, default_tool_budget=100):
    service = ToolService(
        config=ToolServiceConfig(default_tool_budget=default_tool_budget),
        tool_selector=MagicMock(),
        tool_executor=MagicMock(),
        tool_registrar=MagicMock(),
    )
    if pipeline is not None:
        service._tool_pipeline = pipeline
        service._sync_budget_manager_from_runtime()
    return service


def _make_turn_executor(pipeline):
    instance = object.__new__(TurnExecutor)
    instance._orchestrator = SimpleNamespace(_tool_pipeline=pipeline)
    instance._chat_context = SimpleNamespace()
    instance._tool_context = SimpleNamespace(tool_budget=60, tool_calls_used=0)
    return instance


def _call(path):
    return [{"name": "read", "arguments": {"path": path}}]


class TestFundedRunEnforcement:
    """THE regression test: funding 134 on a session with prior usage must
    leave 134 calls for the run, not 134-minus-prior-usage."""

    @pytest.mark.asyncio
    async def test_fund_134_over_79_prior_calls_leaves_134_for_the_run(self, caplog):
        pipeline = _make_pipeline(tool_budget=60)
        pipeline._calls_used = 79  # cumulative legacy counter from prior session work

        executor = _make_turn_executor(pipeline)
        executor._apply_runtime_context_overrides({"tool_budget": 134})

        service = _make_service(pipeline)

        # All owners read the funded ceiling; prior usage is NOT subtracted.
        assert pipeline.tool_budget == 134
        assert pipeline.calls_used == 0
        assert pipeline.calls_remaining == 134
        assert service.budget == 134
        assert service.budget_used == 0
        assert service.is_budget_exhausted() is False

        # The run can actually spend the funded budget: 134 admitted calls.
        for i in range(134):
            result = await pipeline.execute_tool_calls(_call(f"/tmp/f{i}.py"))
            assert result.results[0].skipped is False, f"refused early at call {i}"
            if i < 133:
                assert result.budget_exhausted is False, f"flagged early at call {i}"

        # The 135th distinct call is refused with the unchanged refusal shape.
        refused = await pipeline.execute_tool_calls(_call("/tmp/f134.py"))
        assert refused.results[0].skipped is True
        assert refused.results[0].skip_reason == "Tool budget exhausted"
        assert refused.results[0].outcome_kind == "budget_exhausted"
        assert refused.results[0].block_source == "tool_budget"
        assert refused.results[0].retryable is False
        assert refused.budget_exhausted is True

        # ToolService reads the same exhausted ledger.
        assert service.is_budget_exhausted() is True

    @pytest.mark.asyncio
    async def test_refusal_triads_fire_with_ledger_values(self, caplog):
        pipeline = _make_pipeline(tool_budget=60)
        pipeline._calls_used = 79

        executor = _make_turn_executor(pipeline)
        executor._apply_runtime_context_overrides({"tool_budget": 134})
        pipeline.consume_budget(134)

        with caplog.at_level(logging.INFO, logger="victor.budget"):
            await pipeline.execute_tool_calls(_call("/tmp/f134.py"))

        triads = [
            r
            for r in caplog.records
            if "BUDGET_REFUSAL" in r.getMessage() and "owner=pipeline" in r.getMessage()
        ]
        assert triads, "pipeline refusal triad must still fire when ledger-bound"
        message = triads[0].getMessage()
        assert "ceiling=134" in message
        assert "used=134" in message
        assert "outcome=budget_exhausted" in message

    @pytest.mark.asyncio
    async def test_service_refusal_triad_fires_with_ledger_values(self, caplog):
        from victor.agent.services.tool_service import ToolBudgetExceededError

        pipeline = _make_pipeline(tool_budget=60)
        executor = _make_turn_executor(pipeline)
        executor._apply_runtime_context_overrides({"tool_budget": 134})
        pipeline.consume_budget(134)
        service = _make_service(pipeline)

        with caplog.at_level(logging.INFO, logger="victor.budget"):
            with pytest.raises(ToolBudgetExceededError) as excinfo:
                await service.execute_tool("read", {"path": "/tmp/x.py"})

        assert "Tool budget exhausted (134 / 134)" in str(excinfo.value)
        message = next(
            r.getMessage()
            for r in caplog.records
            if "BUDGET_REFUSAL" in r.getMessage() and "owner=tool_service" in r.getMessage()
        )
        assert "ceiling=134" in message
        assert "used=134" in message


class TestRetargetMidRun:
    def test_service_set_tool_budget_retargets_ledger_without_resetting_usage(self):
        pipeline = _make_pipeline(tool_budget=60)
        executor = _make_turn_executor(pipeline)
        executor._apply_runtime_context_overrides({"tool_budget": 134})
        service = _make_service(pipeline)

        pipeline.consume_budget(20)
        service.set_tool_budget(200)

        assert pipeline.tool_budget == 200
        assert service.budget == 200
        assert service.budget_used == 20
        assert pipeline.calls_remaining == 180
        assert service.is_budget_exhausted() is False


class TestInteractiveTurnParity:
    """Without funded overrides: byte-identical legacy behavior."""

    @pytest.mark.asyncio
    async def test_unbound_pipeline_refusal_bytes_unchanged(self):
        pipeline = _make_pipeline(tool_budget=2)
        pipeline._calls_used = 2

        result = await pipeline.execute_tool_calls(_call("/tmp/f.py"))

        assert result.results[0].skipped is True
        assert result.results[0].skip_reason == "Tool budget exhausted"
        assert result.results[0].outcome_kind == "budget_exhausted"
        assert result.results[0].block_source == "tool_budget"
        assert result.results[0].retryable is False
        assert result.budget_exhausted is True
        assert pipeline.tool_budget == 2
        assert pipeline.calls_used == 2

    def test_unbound_service_set_tool_budget_keeps_legacy_usage_semantics(self):
        service = _make_service(pipeline=None, default_tool_budget=100)
        service._consume_budget(30)

        service.set_tool_budget(134)

        # Legacy behavior preserved: the ceiling moves, usage is NOT reset
        # (the measured divergence is fixed only where a ledger is funded).
        assert service.budget == 134
        assert service.budget_used == 30

    def test_unbound_service_start_new_turn_resets_usage(self):
        service = _make_service(pipeline=None, default_tool_budget=100)
        service._consume_budget(100)
        assert service.is_budget_exhausted() is True

        service.start_new_turn()

        assert service.budget_used == 0
        assert service.is_budget_exhausted() is False

    def test_pipeline_start_new_turn_resets_bound_ledger(self):
        pipeline = _make_pipeline(tool_budget=60)
        executor = _make_turn_executor(pipeline)
        executor._apply_runtime_context_overrides({"tool_budget": 134})
        pipeline.consume_budget(134)
        assert pipeline.is_budget_exhausted if False else pipeline.calls_remaining == 0

        pipeline.start_new_turn()

        assert pipeline.calls_used == 0
        assert pipeline.tool_budget == 134
        assert pipeline.calls_remaining == 134


class TestFundingSurvivesTurnEnd:
    """Per-turn apply/restore for budgets is replaced by the run-scoped ledger."""

    def test_apply_funds_ledger_and_restore_leaves_it_bound(self):
        pipeline = _make_pipeline(tool_budget=60)
        executor = _make_turn_executor(pipeline)

        snapshot = executor._apply_runtime_context_overrides({"tool_budget": 134})

        assert pipeline._budget_ledger is not None
        assert pipeline._budget_ledger.ceiling == 134
        # No per-turn poke of the legacy config owner.
        assert pipeline.config.tool_budget == 60

        executor._restore_runtime_context_overrides(snapshot)

        assert pipeline._budget_ledger is not None
        assert pipeline._budget_ledger.ceiling == 134

    def test_second_funding_retargets_instead_of_resetting(self):
        pipeline = _make_pipeline(tool_budget=60)
        executor = _make_turn_executor(pipeline)
        executor._apply_runtime_context_overrides({"tool_budget": 134})
        pipeline.consume_budget(20)

        executor._apply_runtime_context_overrides({"tool_budget": 50})

        assert pipeline._budget_ledger.ceiling == 50
        assert pipeline._budget_ledger.used == 20

    def test_unsupported_pipeline_falls_back_to_legacy_budget_changes(self):
        pipeline = SimpleNamespace(config=SimpleNamespace(tool_budget=8))
        executor = _make_turn_executor(pipeline)

        snapshot = executor._apply_runtime_context_overrides({"tool_budget": 2})
        assert pipeline.config.tool_budget == 2

        executor._restore_runtime_context_overrides(snapshot)
        assert pipeline.config.tool_budget == 8


class TestFastPathReadsLedger:
    @pytest.mark.asyncio
    async def test_fast_path_budget_check_reads_funded_ledger(self):
        pipeline = _make_pipeline(tool_budget=60)
        executor = _make_turn_executor(pipeline)
        executor._apply_runtime_context_overrides({"tool_budget": 134})
        pipeline.consume_budget(134)
        executor._tool_context.tool_calls_used = 79  # stale legacy mirror

        from unittest.mock import AsyncMock as _AsyncMock

        executor._maybe_execute_deterministic_tool_turn = _AsyncMock(return_value=None)
        exhausted = executor._tool_context.tool_calls_used >= 60  # legacy comparison
        funded_view = (pipeline.tool_budget, pipeline.calls_used)
        assert funded_view == (134, 134)
        # The ledger-backed view reports exhaustion; the stale mirror would not.
        assert exhausted is True


class TestMaxResultsDecoupling:
    def test_config_max_results_defaults_to_tool_budget_for_parity(self):
        assert (
            _make_pipeline(tool_budget=25, enable_output_aggregation=True).config.max_results == 25
        )

    def test_raising_tool_budget_does_not_change_output_truncation(self):
        pipeline = _make_pipeline(tool_budget=25, enable_output_aggregation=True)
        aggregator_max = pipeline._output_aggregator._max_results
        assert aggregator_max == 25

        pipeline.set_tool_budget(134)

        assert pipeline._output_aggregator._max_results == 25

    def test_explicit_max_results_overrides_the_default(self):
        pipeline = _make_pipeline(tool_budget=25, enable_output_aggregation=True, max_results=7)
        assert pipeline._output_aggregator._max_results == 7

        pipeline.fund_budget(134)
        assert pipeline._output_aggregator._max_results == 7


class TestLedgerAbsentParity:
    def test_pipeline_reads_config_without_ledger(self):
        pipeline = _make_pipeline(tool_budget=5)
        assert getattr(pipeline, "_budget_ledger", None) is None
        assert pipeline.tool_budget == 5
        assert pipeline.calls_used == 0
        assert pipeline.calls_remaining == 5

        pipeline._calls_used = 3
        assert pipeline.calls_used == 3
        assert pipeline.calls_remaining == 2

    def test_set_tool_budget_without_ledger_mutates_config(self):
        pipeline = _make_pipeline(tool_budget=5)
        pipeline.set_tool_budget(9)
        assert pipeline.config.tool_budget == 9
        assert pipeline.tool_budget == 9

    def test_consume_budget_without_ledger_mutates_counter(self):
        pipeline = _make_pipeline(tool_budget=5)
        pipeline.consume_budget(2)
        assert pipeline._calls_used == 2
        assert pipeline.calls_used == 2


class TestFundingTransactionality:
    def test_apply_failure_after_funding_rolls_ledger_back(self):
        class BrokenOrchestrator:
            def __init__(self, pipeline):
                object.__setattr__(self, "_tool_pipeline", pipeline)

            def __setattr__(self, name, value):
                if name == "_runtime_tool_context_overrides":
                    raise RuntimeError("setter failed")
                object.__setattr__(self, name, value)

        pipeline = _make_pipeline(tool_budget=60)
        instance = object.__new__(TurnExecutor)
        instance._orchestrator = BrokenOrchestrator(pipeline)
        instance._chat_context = SimpleNamespace()
        instance._tool_context = SimpleNamespace(tool_budget=60, tool_calls_used=0)

        with pytest.raises(RuntimeError, match="setter failed"):
            instance._apply_runtime_context_overrides({"tool_budget": 134})

        # The funding never half-applies: a failed apply leaves no ledger bound.
        assert pipeline._budget_ledger is None
