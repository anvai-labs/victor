"""Focused tests for ToolService.process_tool_results."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from victor.agent.services.tool_service import (
    ToolResultContext,
    ToolService,
    ToolServiceConfig,
)
from victor.observability.analytics.logger import UsageLogger
from victor.observability.analytics.trace_enrichment import TraceEnricher


@dataclass
class FakeCallResult:
    """Minimal stand-in for PipelineCallResult."""

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    execution_time_ms: float = 100.0
    skipped: bool = False
    skip_reason: Optional[str] = None
    outcome_kind: Optional[str] = None
    block_source: Optional[str] = None
    retryable: Optional[bool] = None
    user_message: Optional[str] = None
    tool_call_id: Optional[str] = None


@dataclass
class FakePipelineResult:
    """Minimal stand-in for PipelineExecutionResult."""

    results: List[FakeCallResult] = field(default_factory=list)


def _make_service() -> ToolService:
    return ToolService(
        config=ToolServiceConfig(),
        tool_selector=MagicMock(),
        tool_executor=MagicMock(),
        tool_registrar=MagicMock(),
    )


def _make_ctx(**overrides) -> ToolResultContext:
    defaults = {
        "executed_tools": [],
        "observed_files": set(),
        "failed_tool_signatures": set(),
        "shown_tool_errors": set(),
        "record_tool_execution": MagicMock(),
        "conversation_state": MagicMock(),
        "unified_tracker": MagicMock(),
        "usage_logger": MagicMock(spec=["log_event"]),
        "add_message": MagicMock(),
        "format_tool_output": MagicMock(),
        "console": MagicMock(),
        "presentation": MagicMock(),
        "task_type": "analysis",
    }
    defaults.update(overrides)
    return ToolResultContext(**defaults)


def test_safe_default_pruning_runs_on_formatted_read_output():
    service = _make_service()
    formatted_output = "\n".join(
        [
            '<TOOL_OUTPUT tool="read" path="src/main.py">',
            "═══ ACTUAL FILE CONTENT: src/main.py ═══",
            "[File: src/main.py]",
            "[Lines 1-219 of 219]",
            "[Size: 4096 bytes]",
        ]
        + [f"{i}\tline {i}" for i in range(1, 220)]
        + [
            "═══ END OF FILE: src/main.py ═══",
            "</TOOL_OUTPUT>",
        ]
    )
    ctx = _make_ctx(format_tool_output=MagicMock(return_value=formatted_output))
    pipeline_result = FakePipelineResult(
        results=[
            FakeCallResult(
                tool_name="read",
                success=True,
                result="raw result",
                arguments={"path": "src/main.py"},
                tool_call_id="call_read_1",
            )
        ]
    )

    with patch(
        "victor.config.tool_settings.get_tool_settings",
        return_value=MagicMock(
            tool_output_preview_enabled=True,
            tool_output_pruning_enabled=True,
            tool_output_pruning_safe_only=True,
        ),
    ):
        results = service.process_tool_results(pipeline_result, ctx)

    assert results[0]["was_pruned"] is True
    assert results[0]["result"] != formatted_output
    assert "[PRUNED PREVIEW:" in results[0]["result"]
    assert results[0]["full_result"] == formatted_output
    assert results[0]["content"] == formatted_output
    ctx.add_message.assert_called_once_with(
        "tool",
        results[0]["content"],
        name="read",
        tool_call_id="call_read_1",
    )


def test_no_regression_for_safety_critical_diff_like_output():
    service = _make_service()
    diff_like_output = (
        "=== TOOL OUTPUT: read ===\n"
        "Path: /tmp/patch.txt\n"
        "--- BEGIN CONTENT ---\n"
        "@@ -10,4 +10,4 @@\n"
        "-old_value\n"
        "+new_value\n"
        "Traceback (most recent call last):\n"
        "RuntimeError: boom\n"
        "--- END CONTENT ---"
    )
    ctx = _make_ctx(format_tool_output=MagicMock(return_value=diff_like_output))
    pipeline_result = FakePipelineResult(
        results=[
            FakeCallResult(
                tool_name="read",
                success=True,
                result={"content": diff_like_output},
                arguments={"path": "/tmp/patch.txt"},
                tool_call_id="call_read_2",
            )
        ]
    )

    with patch(
        "victor.config.tool_settings.get_tool_settings",
        return_value=MagicMock(
            tool_output_preview_enabled=True,
            tool_output_pruning_enabled=True,
            tool_output_pruning_safe_only=True,
        ),
    ):
        results = service.process_tool_results(pipeline_result, ctx)

    assert results[0]["was_pruned"] is False
    assert results[0]["content"] == diff_like_output
    assert results[0]["result"] == diff_like_output
    assert results[0]["full_result"] == diff_like_output


def test_mutating_tools_keep_full_display_output_even_when_preview_pruning_enabled():
    service = _make_service()
    formatted_output = "\n".join(
        [
            '<TOOL_OUTPUT tool="write" path="src/main.py">',
            "Wrote src/main.py",
            "Created 220 lines of code",
            "</TOOL_OUTPUT>",
        ]
    )
    ctx = _make_ctx(format_tool_output=MagicMock(return_value=formatted_output))
    pipeline_result = FakePipelineResult(
        results=[
            FakeCallResult(
                tool_name="write",
                success=True,
                result="raw result",
                arguments={"path": "src/main.py", "content": "print('hello')"},
                tool_call_id="call_write_1",
            )
        ]
    )

    with patch(
        "victor.config.tool_settings.get_tool_settings",
        return_value=MagicMock(
            tool_output_preview_enabled=True,
            tool_output_pruning_enabled=True,
            tool_output_pruning_safe_only=True,
        ),
    ):
        results = service.process_tool_results(pipeline_result, ctx)

    assert results[0]["was_pruned"] is False
    assert results[0]["result"] == formatted_output
    assert results[0]["full_result"] == formatted_output
    assert results[0]["content"] == formatted_output


def test_semantic_failure_preserves_follow_up_suggestions():
    service = _make_service()
    ctx = _make_ctx(format_tool_output=MagicMock(return_value="formatted error"))
    pipeline_result = FakePipelineResult(
        results=[
            FakeCallResult(
                tool_name="graph",
                success=True,
                result={
                    "success": False,
                    "error": "Unsupported graph mode: hubs",
                    "metadata": {
                        "follow_up_suggestions": [
                            {
                                "command": 'graph(mode="overview", path=".", top_k=10)',
                                "description": "Use a supported overview mode.",
                            }
                        ]
                    },
                },
                arguments={"mode": "hubs"},
                tool_call_id="call_graph_1",
            )
        ]
    )

    results = service.process_tool_results(pipeline_result, ctx)

    assert results[0]["success"] is False
    assert results[0]["error"] == "Unsupported graph mode: hubs"
    assert results[0]["follow_up_suggestions"] == [
        {
            "command": 'graph(mode="overview", path=".", top_k=10)',
            "description": "Use a supported overview mode.",
        }
    ]


def test_successful_tool_result_injects_structured_fallback_diagnostics():
    service = _make_service()
    ctx = _make_ctx(format_tool_output=MagicMock(return_value="No matches found"))
    pipeline_result = FakePipelineResult(
        results=[
            FakeCallResult(
                tool_name="code_search",
                success=True,
                result={
                    "success": True,
                    "results": [],
                    "count": 0,
                    "mode": "literal",
                    "fallback": "semantic_index_provider_unavailable",
                    "metadata": {
                        "fallback_guidance": (
                            "Semantic search was unavailable and literal fallback found no matches."
                        ),
                    },
                },
                arguments={"query": "Arc immutable readonly", "mode": "semantic"},
                tool_call_id="call_search_1",
            )
        ]
    )

    results = service.process_tool_results(pipeline_result, ctx)

    assert results[0]["tool_diagnostics"] == [
        "fallback=semantic_index_provider_unavailable",
        "Semantic search was unavailable and literal fallback found no matches.",
    ]
    assert "Tool diagnostics:" in results[0]["content"]
    assert "fallback=semantic_index_provider_unavailable" in results[0]["content"]


def test_skipped_tool_uses_skip_reason_instead_of_unknown_error():
    service = _make_service()
    ctx = _make_ctx(format_tool_output=MagicMock(return_value="formatted skip"))
    pipeline_result = FakePipelineResult(
        results=[
            FakeCallResult(
                tool_name="read",
                success=False,
                result=None,
                error=None,
                arguments={"path": "victor/core/container.py"},
                skipped=True,
                skip_reason="Repeated failing call with same arguments",
                tool_call_id="call_read_skip_1",
            )
        ]
    )

    results = service.process_tool_results(pipeline_result, ctx)

    assert results[0]["success"] is False
    assert results[0]["error"] == "Repeated failing call with same arguments"
    assert "Unknown error" not in results[0]["result"]
    ctx.console.print.assert_called_once()
    assert "Tool call skipped" in ctx.console.print.call_args[0][0]


def test_skipped_tool_prefers_structured_user_message_and_metadata():
    service = _make_service()
    ctx = _make_ctx(format_tool_output=MagicMock(return_value="formatted skip"))
    pipeline_result = FakePipelineResult(
        results=[
            FakeCallResult(
                tool_name="read",
                success=False,
                result={"error": "internal duplicate-read block"},
                error=None,
                arguments={"path": "victor/core/container.py"},
                skipped=True,
                skip_reason="Duplicate file read within session",
                outcome_kind="duplicate_read",
                block_source="session_read_dedup",
                retryable=True,
                user_message=(
                    "File 'victor/core/container.py' was already read with the same "
                    "offset/limit. Choose a different range, file, or tool."
                ),
                tool_call_id="call_read_skip_2",
            )
        ]
    )

    results = service.process_tool_results(pipeline_result, ctx)

    assert results[0]["success"] is False
    assert results[0]["error"].startswith("File 'victor/core/container.py' was already read")
    assert results[0]["outcome_kind"] == "duplicate_read"
    assert results[0]["block_source"] == "session_read_dedup"
    assert results[0]["retryable"] is True
    assert results[0]["user_message"].startswith("File 'victor/core/container.py'")
    ctx.console.print.assert_called_once()
    assert "already read with the same offset/limit" in ctx.console.print.call_args[0][0]


def test_post_processing_failure_still_emits_tool_response():
    service = _make_service()
    ctx = _make_ctx(format_tool_output=MagicMock(side_effect=RuntimeError("format failed")))
    pipeline_result = FakePipelineResult(
        results=[
            FakeCallResult(
                tool_name="metrics",
                success=True,
                result={"summary": "complexity report"},
                arguments={"path": "victor/framework/graph.py"},
                tool_call_id="call_metrics_1",
            )
        ]
    )

    results = service.process_tool_results(pipeline_result, ctx)

    assert results[0]["success"] is False
    assert results[0]["tool_call_id"] == "call_metrics_1"
    assert results[0]["outcome_kind"] == "tool_result_processing_failed"
    assert "Tool result unavailable for 'metrics'" in results[0]["content"]
    ctx.add_message.assert_called_once_with(
        "tool",
        results[0]["content"],
        name="metrics",
        tool_call_id="call_metrics_1",
        persist_synchronously=True,
    )


def test_post_processing_failure_survives_tool_message_write_failure():
    service = _make_service()
    ctx = _make_ctx(
        format_tool_output=MagicMock(side_effect=RuntimeError("format failed")),
        add_message=MagicMock(side_effect=RuntimeError("conversation write failed")),
    )
    pipeline_result = FakePipelineResult(
        results=[
            FakeCallResult(
                tool_name="metrics",
                success=True,
                result={"summary": "complexity report"},
                arguments={"path": "victor/framework/graph.py"},
                tool_call_id="call_metrics_2",
            )
        ]
    )

    results = service.process_tool_results(pipeline_result, ctx)

    assert results[0]["success"] is False
    assert results[0]["tool_call_id"] == "call_metrics_2"
    assert results[0]["outcome_kind"] == "tool_result_processing_failed"
    assert "Tool result unavailable for 'metrics'" in results[0]["content"]


def test_recursive_tool_result_is_logged_without_breaking_metrics_post_processing(
    tmp_path,
):
    class RecursiveResult:
        def __init__(self) -> None:
            self.summary = "complexity report"
            self.self_ref = self

    service = _make_service()
    usage_logger = TraceEnricher(
        UsageLogger(log_file=tmp_path / "usage_log.jsonl", enabled=True),
        enabled=True,
    )
    ctx = _make_ctx(
        usage_logger=usage_logger,
        format_tool_output=MagicMock(return_value="formatted metrics"),
    )
    pipeline_result = FakePipelineResult(
        results=[
            FakeCallResult(
                tool_name="metrics",
                success=True,
                result=RecursiveResult(),
                arguments={"path": "victor/framework/graph.py"},
                tool_call_id="call_metrics_recursive_1",
            )
        ]
    )

    results = service.process_tool_results(pipeline_result, ctx)

    assert results[0]["success"] is True
    assert results[0]["tool_call_id"] == "call_metrics_recursive_1"
    assert results[0]["content"] == "formatted metrics"
    ctx.add_message.assert_called_once_with(
        "tool",
        "formatted metrics",
        name="metrics",
        tool_call_id="call_metrics_recursive_1",
    )


# ---------------------------------------------------------------------------
# Telemetry truth (P2): the tool_result EVENT must reflect string-marker errors
# while the LLM-visible content path stays untouched.
# ---------------------------------------------------------------------------


def _logged_tool_result_event(ctx):
    calls = [c for c in ctx.usage_logger.log_event.call_args_list if c.args[0] == "tool_result"]
    assert calls, "no tool_result event logged"
    return calls[-1].args[1]


def test_tool_result_event_error_marker_string_logs_success_false():
    service = _make_service()
    ctx = _make_ctx()
    error_text = "### ❌ ERROR\nArgument parsing error: unrecognized arguments: -rn"
    results = service.process_tool_results(
        FakePipelineResult(results=[FakeCallResult("code", True, result=error_text)]),
        ctx,
    )
    event = _logged_tool_result_event(ctx)
    assert event["success"] is False
    assert event["outcome_kind"] == "tool_error"
    assert event["error_detail"].startswith("Argument parsing error")
    # LLM-visible content untouched: the SUCCESS formatting branch still received
    # the full corrective text (the event demotion must not reroute the LLM path)
    assert any(
        "### ❌ ERROR" in str(call) for call in ctx.format_tool_output.call_args_list
    ), "success-branch formatter never saw the corrective text"
    assert results, "tool results should still be produced"


def test_tool_result_event_warning_marker_keeps_success_true_with_outcome_kind():
    service = _make_service()
    ctx = _make_ctx()
    warn_text = "### ⚠️ SHELL OPERATOR NOT SUPPORTED\nuse the shell tool"
    service.process_tool_results(
        FakePipelineResult(results=[FakeCallResult("code", True, result=warn_text)]),
        ctx,
    )
    event = _logged_tool_result_event(ctx)
    assert event["success"] is True
    assert event["outcome_kind"] == "warning"


def test_tool_result_event_plain_string_unchanged():
    service = _make_service()
    ctx = _make_ctx()
    service.process_tool_results(
        FakePipelineResult(results=[FakeCallResult("read", True, result="file contents here")]),
        ctx,
    )
    event = _logged_tool_result_event(ctx)
    assert event["success"] is True
    assert event.get("outcome_kind") in (None, "")


def test_tool_result_event_dict_semantic_failure_still_false():
    service = _make_service()
    ctx = _make_ctx()
    service.process_tool_results(
        FakePipelineResult(
            results=[FakeCallResult("shell", True, result={"success": False, "error": "boom"})]
        ),
        ctx,
    )
    event = _logged_tool_result_event(ctx)
    assert event["success"] is False


def test_tool_result_event_existing_outcome_kind_not_clobbered():
    service = _make_service()
    ctx = _make_ctx()
    service.process_tool_results(
        FakePipelineResult(
            results=[
                FakeCallResult(
                    "read",
                    True,
                    result="### ⚠️ warn",
                    outcome_kind="duplicate_read",
                )
            ]
        ),
        ctx,
    )
    event = _logged_tool_result_event(ctx)
    assert event["outcome_kind"] == "duplicate_read"


def test_interrupted_publication_requires_a_persistence_callback():
    import pytest

    ctx = _make_ctx(add_message=None, require_persistence=True)
    ctx.format_tool_output.return_value = "result"
    with pytest.raises(RuntimeError, match="durable message persistence"):
        _make_service().process_tool_results(
            FakePipelineResult(
                results=[
                    FakeCallResult(
                        tool_name="write", success=True, result="done", tool_call_id="done"
                    )
                ]
            ),
            ctx,
        )
