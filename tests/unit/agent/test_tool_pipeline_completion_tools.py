"""Tests for caller-declared completion-critical tool exemptions in ToolPipeline.

A consumer whose terminal tool performs completion (a proposal/submit call)
must never have that call suppressed by the pipeline's spin-adjacent skip
paths — budget exhaustion, repeated-failing-call signature tracking, and
semantic redundancy — or its run can never finish (no journal row, the
model's submit never executes). Empty declaration keeps every skip path
byte-identical to before.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.tool_executor import ToolExecutionResult
from victor.agent.tool_pipeline import PipelineExecutionResult, ToolPipeline, ToolPipelineConfig
from victor.tools.enums import AccessMode


@pytest.fixture
def mock_tool_registry():
    registry = MagicMock()
    registry.is_tool_enabled.return_value = True
    registry.get.return_value.access_mode = AccessMode.READONLY
    return registry


@pytest.fixture
def mock_tool_executor():
    executor = MagicMock()
    executor.execute = AsyncMock(
        return_value=ToolExecutionResult(
            tool_name="submit",
            success=True,
            result="submitted",
            execution_time=0.01,
        )
    )
    return executor


def make_pipeline(mock_tool_registry, mock_tool_executor, completion_tools=(), budget=5):
    config = ToolPipelineConfig(
        tool_budget=budget,
        enable_caching=False,
        enable_output_aggregation=False,
        enable_synthesis_checkpoints=False,
        enable_semantic_caching=False,
        completion_tool_names=completion_tools,
    )
    return ToolPipeline(
        tool_registry=mock_tool_registry,
        tool_executor=mock_tool_executor,
        config=config,
    )


async def _fail_once_then_succeed(mock_tool_executor):
    """First execution of the watched tool fails (records the signature)."""
    results = [
        ToolExecutionResult(
            tool_name="submit",
            success=False,
            result="validation error",
            execution_time=0.01,
        ),
        ToolExecutionResult(
            tool_name="submit",
            success=True,
            result="submitted",
            execution_time=0.01,
        ),
    ]
    mock_tool_executor.execute = AsyncMock(side_effect=results)


class TestCompletionToolExemption:
    async def test_no_declaration_budget_exhaustion_unchanged(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Parity: without a declaration, an exhausted budget still skips."""
        pipeline = make_pipeline(mock_tool_registry, mock_tool_executor)
        pipeline._calls_used = pipeline.config.tool_budget
        result = await pipeline.execute_tool_calls([{"name": "submit", "arguments": {}}])
        assert result.results[0].skipped is True
        assert result.results[0].outcome_kind == "budget_exhausted"
        mock_tool_executor.execute.assert_not_awaited()

    async def test_declared_completion_tool_executes_at_exhausted_budget(
        self, mock_tool_registry, mock_tool_executor
    ):
        """The measured failure: the terminal call was the only call in the
        batch and was skipped at 0ms on an exhausted budget, so the run's
        deliverable never executed. A declared completion tool executes."""
        pipeline = make_pipeline(
            mock_tool_registry, mock_tool_executor, completion_tools=("submit",)
        )
        pipeline._calls_used = pipeline.config.tool_budget
        result = await pipeline.execute_tool_calls([{"name": "submit", "arguments": {}}])
        assert result.results[0].skipped is False
        assert result.results[0].success is True
        mock_tool_executor.execute.assert_awaited()
        # An executed call means the batch is not fully blocked for spin
        # detection.
        assert pipeline.last_batch_all_skipped is False
        assert pipeline.last_batch_effectively_blocked is False

    async def test_declared_completion_tool_executes_after_failed_signature(
        self, mock_tool_registry, mock_tool_executor
    ):
        """A retried terminal call that previously failed with the same
        arguments must still execute (consumer-side guard refusals are the
        consumer's own feedback, not a framework termination)."""
        await _fail_once_then_succeed(mock_tool_executor)
        pipeline = make_pipeline(
            mock_tool_registry, mock_tool_executor, completion_tools=("submit",)
        )
        first = await pipeline.execute_tool_calls([{"name": "submit", "arguments": {}}])
        assert first.results[0].success is False  # real failure, recorded

        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="submit",
                success=True,
                result="submitted",
                execution_time=0.01,
            )
        )
        second = await pipeline.execute_tool_calls([{"name": "submit", "arguments": {}}])
        assert second.results[0].skipped is False
        assert second.results[0].success is True

    async def test_undeclared_tool_still_skipped_after_failed_signature(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Parity: signature tracking still skips non-declared repeats."""
        results = [
            ToolExecutionResult(
                tool_name="search", success=False, result="boom", execution_time=0.01
            ),
        ]
        mock_tool_executor.execute = AsyncMock(side_effect=results * 1)
        pipeline = make_pipeline(
            mock_tool_registry, mock_tool_executor, completion_tools=("submit",)
        )
        await pipeline.execute_tool_calls([{"name": "search", "arguments": {}}])
        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="search", success=True, result="ok", execution_time=0.01
            )
        )
        second = await pipeline.execute_tool_calls([{"name": "search", "arguments": {}}])
        assert second.results[0].skipped is True
        assert "Repeated failing call" in second.results[0].skip_reason

    async def test_mixed_batch_completion_executes_spinning_call_blocked(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Mixed batch: the spinning (already-failed) call stays skipped while
        the declared completion call in the same batch executes."""
        await _fail_once_then_succeed(mock_tool_executor)
        pipeline = make_pipeline(
            mock_tool_registry, mock_tool_executor, completion_tools=("submit",)
        )
        await pipeline.execute_tool_calls([{"name": "submit", "arguments": {}}])  # records sig

        mock_tool_executor.execute = AsyncMock(
            return_value=ToolExecutionResult(
                tool_name="submit",
                success=True,
                result="submitted",
                execution_time=0.01,
            )
        )
        batch = await pipeline.execute_tool_calls(
            [
                {"name": "submit", "arguments": {}},  # matches the failed signature
                {"name": "search", "arguments": {"query": "fresh"}},
            ]
        )
        assert batch.results[0].success is True
        assert batch.results[0].skipped is False
        assert pipeline.last_batch_all_skipped is False

    async def test_mixed_batch_undeclared_completion_blocked_spinning_executes(
        self, mock_tool_registry, mock_tool_executor
    ):
        """Parity in a mixed batch: with an exhausted budget and no declared
        completion tool, both calls stay skipped (legacy behavior, including
        the retryable distinction between the internal and tail skips)."""
        pipeline = make_pipeline(mock_tool_registry, mock_tool_executor, budget=1)
        pipeline._calls_used = 1
        batch = await pipeline.execute_tool_calls(
            [
                {"name": "search", "arguments": {"query": "a"}},
                {"name": "read", "arguments": {"path": "/tmp/x"}},
            ]
        )
        assert batch.results[0].skipped is True
        assert batch.results[0].outcome_kind == "budget_exhausted"
        assert batch.results[0].retryable is False  # internal defensive check
        assert batch.results[1].skipped is True
        assert batch.results[1].outcome_kind == "budget_exhausted"
        assert batch.results[1].retryable is True  # dispatch tail-skip
        assert pipeline.last_batch_effectively_blocked is True

    async def test_declared_completion_tool_dispatches_past_budget_in_batch(
        self, mock_tool_registry, mock_tool_executor
    ):
        """A declared completion call tail-positioned past the admitted count
        is dispatched, not tail-skipped."""
        pipeline = make_pipeline(
            mock_tool_registry, mock_tool_executor, completion_tools=("submit",), budget=1
        )
        pipeline._calls_used = 1
        batch = await pipeline.execute_tool_calls(
            [
                {"name": "search", "arguments": {"query": "a"}},
                {"name": "submit", "arguments": {}},
            ]
        )
        assert batch.results[1].skipped is False
        assert batch.results[1].success is True

    async def test_mixed_batch_spin_counting_sees_executed_completion_call(
        self, mock_tool_registry, mock_tool_executor
    ):
        """A batch whose declared completion call executed must not flip the
        pipeline's fully-blocked flags the spin detector counts."""
        pipeline = make_pipeline(
            mock_tool_registry, mock_tool_executor, completion_tools=("submit",)
        )
        pipeline._calls_used = pipeline.config.tool_budget
        batch = await pipeline.execute_tool_calls(
            [
                {"name": "search", "arguments": {"query": "a"}},
                {"name": "submit", "arguments": {}},
            ]
        )
        executed = [r for r in batch.results if not r.skipped]
        assert len(executed) == 1
        assert isinstance(batch, PipelineExecutionResult)
        assert pipeline.last_batch_effectively_blocked is False
