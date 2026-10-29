"""Tests for active trace quality filtering (MemReader-inspired)."""

from unittest.mock import MagicMock


class TestTraceQualityScoring:
    def test_function_exists(self):
        from victor.framework.rl.learners.trace_analysis import score_trace_quality

        assert callable(score_trace_quality)

    def test_rich_trace_high_score(self):
        from victor.framework.rl.learners.trace_analysis import score_trace_quality

        detail = MagicMock()
        detail.result_summary = "Found function definition"
        detail.error_detail = ""
        detail.reasoning_before = "I need to find the auth module"
        trace = MagicMock()
        trace.tool_calls = 6
        trace.tool_call_details = [detail] * 5
        trace.tool_failures = {"edit_mismatch": 2}
        trace.success = False
        assert score_trace_quality(trace) >= 0.7

    def test_trivial_trace_low_score(self):
        from victor.framework.rl.learners.trace_analysis import score_trace_quality

        trace = MagicMock()
        trace.tool_calls = 1
        trace.tool_call_details = []
        trace.tool_failures = {}
        trace.success = True
        assert score_trace_quality(trace) < 0.3

    def test_abandoned_session_zero(self):
        from victor.framework.rl.learners.trace_analysis import score_trace_quality

        trace = MagicMock()
        trace.tool_calls = 0
        trace.tool_call_details = []
        trace.tool_failures = {}
        trace.success = False
        assert score_trace_quality(trace) == 0.0

    def test_well_categorized_failures_boost(self):
        from victor.framework.rl.learners.trace_analysis import score_trace_quality

        trace = MagicMock()
        trace.tool_calls = 3
        trace.tool_call_details = []
        trace.tool_failures = {"edit_mismatch": 2, "file_not_found": 1}
        trace.success = False
        assert score_trace_quality(trace) >= 0.3

    def test_success_without_details_moderate(self):
        from victor.framework.rl.learners.trace_analysis import score_trace_quality

        trace = MagicMock()
        trace.tool_calls = 4
        trace.tool_call_details = []
        trace.tool_failures = {}
        trace.success = True
        score = score_trace_quality(trace)
        assert 0.2 <= score <= 0.6

    def test_threshold_constant_exists(self):
        from victor.framework.rl.learners.trace_analysis import TRACE_QUALITY_THRESHOLD

        assert isinstance(TRACE_QUALITY_THRESHOLD, float)
        assert 0.0 < TRACE_QUALITY_THRESHOLD < 1.0
