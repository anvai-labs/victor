# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the extracted trace-analysis module and its re-export shim."""

from victor.framework.rl.learners import trace_analysis as ta
from victor.framework.rl.learners.trace_analysis import (
    ExecutionTrace,
    ToolCallTrace,
    TraceZone,
    analyze_capability_gaps,
    classify_trace_zone,
    format_failing_exemplars,
    get_failure_hint,
    score_trace_quality,
)


def _trace(**kw):
    base = {
        "session_id": "s",
        "task_type": "action",
        "provider": "ollama",
        "model": "qwen",
        "tool_calls": 4,
        "tool_failures": {},
        "success": True,
        "completion_score": 0.9,
        "tokens_used": 100,
    }
    base.update(kw)
    return ExecutionTrace(**base)


class TestReExportBackCompat:
    def test_symbols_are_reexported_from_prompt_optimizer(self):
        # The learner module must keep re-exporting every relocated name, and
        # the class objects must be identical (isinstance across both paths).
        from victor.framework.rl.learners import prompt_optimizer as po

        for name in (
            "ExecutionTrace",
            "ToolCallTrace",
            "HarnessVerdict",
            "TraceZone",
            "CapabilityGap",
            "FAILURE_HINTS",
            "FAILURE_TO_CAPABILITY",
            "get_failure_hint",
            "classify_trace_zone",
            "score_trace_quality",
            "analyze_capability_gaps",
            "format_failing_exemplars",
        ):
            assert getattr(po, name) is getattr(ta, name), name


class TestZoneClassification:
    def test_recovery_success_despite_failures(self):
        t = _trace(success=True, tool_failures={"edit_mismatch": 1}, completion_score=0.8)
        assert classify_trace_zone(t) is TraceZone.RECOVERY

    def test_failure_low_score(self):
        t = _trace(success=False, completion_score=0.2)
        assert classify_trace_zone(t) is TraceZone.FAILURE

    def test_success_clean(self):
        t = _trace(success=True, tool_failures={}, completion_score=0.9)
        assert classify_trace_zone(t) is TraceZone.SUCCESS


class TestAnalysisHelpers:
    def test_capability_gaps_from_failures(self):
        traces = [
            _trace(success=False, completion_score=0.1, tool_failures={"edit_mismatch": 3})
            for _ in range(2)
        ]
        gaps = analyze_capability_gaps(traces)
        assert gaps and gaps[0].capability == "edit_precision"
        assert gaps[0].failure_count == 6

    def test_failure_hint_lookup(self):
        assert "old_str" in get_failure_hint("edit_mismatch")
        assert get_failure_hint("nonexistent") == ""

    def test_quality_score_bounded(self):
        assert 0.0 <= score_trace_quality(_trace(tool_calls=6)) <= 1.0

    def test_format_failing_exemplars_empty_when_clean(self):
        assert format_failing_exemplars([_trace()]) == ""

    def test_format_failing_exemplars_renders_bad_calls(self):
        t = _trace(
            success=False,
            completion_score=0.1,
            tool_call_details=[
                ToolCallTrace(
                    tool_name="edit",
                    arguments_summary="foo.py",
                    error_detail="old_str not found",
                    success=False,
                )
            ],
        )
        out = format_failing_exemplars([t])
        assert "Failing exemplars" in out
        assert "edit(foo.py)" in out
        assert "old_str not found" in out
