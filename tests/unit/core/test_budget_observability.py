"""Budget-refusal observability (consolidation PR 1).

Every budget enforcement owner records a structured (owner, ceiling, used)
triad at refusal time so budget divergence (e.g. a run refused at ~55-60
calls against a funded 134) is diagnosable from logs alone. Pure
observability: no refusal string, outcome kind, or control-flow change.
"""

import logging
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.tool_executor import ToolExecutionResult
from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig
from victor.core.budget_observability import record_budget_refusal
from victor.tools.enums import AccessMode

LINE_RE = re.compile(
    r"^BUDGET_REFUSAL "
    r"owner=(?P<owner>\S+) "
    r"ceiling=(?P<ceiling>\d+) "
    r"used=(?P<used>\d+) "
    r"tool=(?P<tool>\S+) "
    r"outcome=(?P<outcome>\S+)"
)


def _refusal_records(caplog):
    return [r for r in caplog.records if LINE_RE.match(r.getMessage())]


# ---------------------------------------------------------------------------
# (a) helper emits a parseable line with the correct fields
# ---------------------------------------------------------------------------


def test_record_budget_refusal_emits_parseable_line(caplog):
    with caplog.at_level("INFO", logger="victor.budget"):
        record_budget_refusal(
            "pipeline",
            25,
            25,
            tool_name="read_file",
            outcome_kind="budget_exhausted",
        )

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == "INFO"
    match = LINE_RE.match(record.getMessage())
    assert match, record.getMessage()
    assert match.group("owner") == "pipeline"
    assert match.group("ceiling") == "25"
    assert match.group("used") == "25"
    assert match.group("tool") == "read_file"
    assert match.group("outcome") == "budget_exhausted"


def test_record_budget_refusal_stable_shape_without_optionals(caplog):
    with caplog.at_level("INFO", logger="victor.budget"):
        record_budget_refusal("streaming", 200, 200)

    message = caplog.records[0].getMessage()
    match = LINE_RE.match(message)
    assert match, message
    assert match.group("owner") == "streaming"
    assert match.group("tool") == "unknown"
    assert match.group("outcome") == "none"


def test_record_budget_refusal_extra_appends_key_value_pairs(caplog):
    with caplog.at_level("INFO", logger="victor.budget"):
        record_budget_refusal(
            "pipeline",
            134,
            60,
            tool_name="grep",
            outcome_kind="budget_exhausted",
            extra={"queued": 3},
        )

    message = caplog.records[0].getMessage()
    assert LINE_RE.match(message), message
    assert message.endswith("queued=3")


# ---------------------------------------------------------------------------
# (b) pipeline budget-exhausted skip records owner='pipeline'
# ---------------------------------------------------------------------------


def _mock_pipeline(tool_budget):
    registry = MagicMock()
    registry.is_tool_enabled.return_value = True
    registry.get.return_value.access_mode = AccessMode.READONLY
    executor = MagicMock()
    executor.execute = AsyncMock(
        return_value=ToolExecutionResult(
            tool_name="read",
            success=True,
            result="file contents",
            execution_time=0.01,
        )
    )
    config = ToolPipelineConfig(
        tool_budget=tool_budget,
        enable_caching=False,
        enable_output_aggregation=False,
        enable_synthesis_checkpoints=False,
        enable_semantic_caching=False,
    )
    return ToolPipeline(tool_registry=registry, tool_executor=executor, config=config)


def test_pipeline_budget_exhausted_skip_records_triad(caplog):
    pipeline = _mock_pipeline(tool_budget=3)
    pipeline._calls_used = 3

    with caplog.at_level("INFO", logger="victor.budget"):
        result = pipeline._budget_exhausted_skip({"name": "read_file", "arguments": {}})

    refusals = _refusal_records(caplog)
    assert len(refusals) == 1
    match = LINE_RE.match(refusals[0].getMessage())
    assert match.group("owner") == "pipeline"
    assert match.group("ceiling") == "3"
    assert match.group("used") == "3"
    assert match.group("tool") == "read_file"
    assert match.group("outcome") == "budget_exhausted"
    # Behavior parity: the skip result itself is unchanged.
    assert result.skipped
    assert result.outcome_kind == "budget_exhausted"
    assert result.skip_reason == "Tool budget exhausted — call not executed"
    assert result.retryable


@pytest.mark.asyncio
async def test_pipeline_batch_admission_skip_records_triad(caplog):
    """The dispatch-time budget check records the pipeline's accounting."""
    pipeline = _mock_pipeline(tool_budget=2)
    pipeline._calls_used = 2

    with caplog.at_level("INFO", logger="victor.budget"):
        result = await pipeline.execute_tool_calls([{"name": "read", "arguments": {}}])

    refusals = _refusal_records(caplog)
    # Two records: the batch-admission tail-skip (a named call) and the
    # post-batch budget_exhausted flag (batch-level, no tool name).
    assert len(refusals) == 2
    tail_skip = LINE_RE.match(refusals[0].getMessage())
    assert tail_skip.group("owner") == "pipeline"
    assert tail_skip.group("ceiling") == "2"
    assert tail_skip.group("used") == "2"
    assert tail_skip.group("tool") == "read"
    assert tail_skip.group("outcome") == "budget_exhausted"
    flag = LINE_RE.match(refusals[1].getMessage())
    assert flag.group("owner") == "pipeline"
    assert flag.group("ceiling") == "2"
    assert flag.group("used") == "2"
    assert flag.group("outcome") == "none"
    # Behavior parity: defensive-fallback skip, not the tail-skip variant.
    assert result.results[0].outcome_kind == "budget_exhausted"
    assert result.results[0].retryable is False


@pytest.mark.asyncio
async def test_pipeline_budget_exhausted_flag_records_triad(caplog):
    """The post-batch budget_exhausted flag records the pipeline's accounting."""
    pipeline = _mock_pipeline(tool_budget=1)
    pipeline._calls_used = 0

    with caplog.at_level("INFO", logger="victor.budget"):
        result = await pipeline.execute_tool_calls([{"name": "read", "arguments": {}}])

    assert result.budget_exhausted is True
    refusals = _refusal_records(caplog)
    assert refusals, "budget_exhausted flag must record a refusal triad"
    match = LINE_RE.match(refusals[-1].getMessage())
    assert match.group("owner") == "pipeline"
    assert match.group("ceiling") == "1"
    assert match.group("used") == "1"


# ---------------------------------------------------------------------------
# (c) tool_service exhaustion records owner='tool_service'
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_service_exhaustion_records_triad(caplog):
    from victor.agent.services.tool_service import ToolBudgetExceededError, ToolService

    service = ToolService.__new__(ToolService)
    service._logger = logging.getLogger(__name__)
    with caplog.at_level("INFO", logger="victor.budget"):
        with (
            patch.object(ToolService, "is_budget_exhausted", return_value=True),
            patch.object(ToolService, "_hydrate_tool_if_missing"),
            patch.object(ToolService, "budget", new_callable=lambda: property(lambda self: 100)),
            patch.object(
                ToolService, "budget_used", new_callable=lambda: property(lambda self: 100)
            ),
        ):
            with pytest.raises(ToolBudgetExceededError):
                await service.execute_tool("bash", {})

    refusals = _refusal_records(caplog)
    assert len(refusals) == 1
    match = LINE_RE.match(refusals[0].getMessage())
    assert match.group("owner") == "tool_service"
    assert match.group("ceiling") == "100"
    assert match.group("used") == "100"
    assert match.group("tool") == "bash"
    assert match.group("outcome") == "budget_exhausted"
