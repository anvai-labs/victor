"""Unit tests for the pure task-report metadata helpers (ADR-019 extraction)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

from victor.agent.task_report_metadata import (
    build_compaction_metadata,
    build_continuation_metadata,
    resolve_task_type,
)

# ── resolve_task_type ─────────────────────────────────────────────


def test_resolve_prefers_unified_task_type() -> None:
    stream_ctx = SimpleNamespace(unified_task_type=SimpleNamespace(value="qa"))
    assert resolve_task_type(stream_ctx, None, ()) == "qa"


def test_resolve_falls_back_to_coarse_type() -> None:
    stream_ctx = SimpleNamespace(unified_task_type=None, coarse_task_type="coding")
    assert resolve_task_type(stream_ctx, None, ()) == "coding"


def test_resolve_uses_tracker_then_candidates() -> None:
    tracker = SimpleNamespace(task_type="review")
    assert resolve_task_type(None, tracker, ("x",)) == "review"
    assert resolve_task_type(None, None, (None, "", "fromcandidate")) == "fromcandidate"


def test_resolve_defaults_when_nothing_available() -> None:
    assert resolve_task_type(None, None, (None, "")) == "default"


# ── build_compaction_metadata ─────────────────────────────────────


def test_compaction_empty_when_no_signals() -> None:
    meta = build_compaction_metadata(None, None)
    assert meta["occurred"] is False
    assert meta["summary"] == ""
    assert meta["messages_removed"] == 0
    assert meta["saved_tokens"] == 0


def test_compaction_marks_occurred_from_summary() -> None:
    stream_ctx = SimpleNamespace(compaction_summary="trimmed 3 msgs")
    meta = build_compaction_metadata(stream_ctx, None)
    assert meta["occurred"] is True
    assert meta["summary"] == "trimmed 3 msgs"


def test_compaction_reads_saved_tokens_from_context_service() -> None:
    class _Ctx:
        def get_performance_metrics(self) -> Dict[str, Any]:
            return {"last_compaction_saved_tokens": 512}

    meta = build_compaction_metadata(None, _Ctx())
    assert meta["saved_tokens"] == 512
    assert meta["occurred"] is True


def test_compaction_survives_context_service_error() -> None:
    class _Boom:
        def get_performance_metrics(self) -> Dict[str, Any]:
            raise RuntimeError("nope")

    meta = build_compaction_metadata(None, _Boom())
    assert meta["saved_tokens"] == 0  # error swallowed, defaults used


# ── build_continuation_metadata ───────────────────────────────────


def test_continuation_empty_without_stream_ctx() -> None:
    assert build_continuation_metadata(None) == {}


def test_continuation_collects_and_bounds_fields() -> None:
    stream_ctx = SimpleNamespace(
        task_intent="  ship the feature  ",
        plan_steps=[f"step{i}" for i in range(10)],  # >6 → truncated
        intent_log=[{"a": i} for i in range(10)],  # last 6 kept
        resume_summary="resuming",
        degraded_resume_state=True,
        build_continuation_ledger=lambda **kw: "LEDGER",
    )
    meta = build_continuation_metadata(stream_ctx)
    assert meta["task_intent"] == "ship the feature"
    assert len(meta["plan_steps"]) == 6
    assert len(meta["intent_log"]) == 6
    assert meta["resume_summary"] == "resuming"
    assert meta["degraded_resume_state"] is True
    assert meta["continuation_ledger"] == "LEDGER"


def test_continuation_ledger_falls_back_to_noarg_call() -> None:
    def _ledger_no_kwargs() -> str:
        return "NOARG"

    stream_ctx = SimpleNamespace(build_continuation_ledger=_ledger_no_kwargs)
    meta = build_continuation_metadata(stream_ctx)
    assert meta["continuation_ledger"] == "NOARG"
