"""Tests for GEPA semantic trace zones (PRIME-inspired)."""

from unittest.mock import MagicMock


class TestTraceZoneEnum:
    def test_zone_enum_exists(self):
        from victor.framework.rl.learners.prompt_optimizer import TraceZone

        assert hasattr(TraceZone, "SUCCESS")
        assert hasattr(TraceZone, "FAILURE")
        assert hasattr(TraceZone, "RECOVERY")

    def test_zone_values(self):
        from victor.framework.rl.learners.prompt_optimizer import TraceZone

        assert TraceZone.SUCCESS.value == "successful_strategies"
        assert TraceZone.FAILURE.value == "failure_patterns"
        assert TraceZone.RECOVERY.value == "recovery_patterns"


class TestTraceZoneClassification:
    def _make_trace(self, score=0.5, success=True, has_failures=False, has_recovery=False):
        trace = MagicMock()
        trace.completion_score = score
        trace.success = success
        trace.tool_failures = {"edit_mismatch": 2} if has_failures else {}
        if has_recovery:
            trace.success = True
            trace.tool_failures = {"edit_mismatch": 1}
            trace.completion_score = 0.7
        return trace

    def test_high_score_success_zone(self):
        from victor.framework.rl.learners.prompt_optimizer import (
            classify_trace_zone,
            TraceZone,
        )

        trace = self._make_trace(score=0.85, success=True)
        assert classify_trace_zone(trace) == TraceZone.SUCCESS

    def test_low_score_failure_zone(self):
        from victor.framework.rl.learners.prompt_optimizer import (
            classify_trace_zone,
            TraceZone,
        )

        trace = self._make_trace(score=0.3, success=False, has_failures=True)
        assert classify_trace_zone(trace) == TraceZone.FAILURE

    def test_recovery_zone(self):
        from victor.framework.rl.learners.prompt_optimizer import (
            classify_trace_zone,
            TraceZone,
        )

        trace = self._make_trace(has_recovery=True)
        assert classify_trace_zone(trace) == TraceZone.RECOVERY

    def test_medium_score_defaults_to_success(self):
        from victor.framework.rl.learners.prompt_optimizer import (
            classify_trace_zone,
            TraceZone,
        )

        trace = self._make_trace(score=0.6, success=True)
        assert classify_trace_zone(trace) == TraceZone.SUCCESS


class TestZonedFormatting:
    def _make_trace(self, sid, score, success, failures=None):
        trace = MagicMock()
        trace.session_id = sid
        trace.completion_score = score
        trace.success = success
        trace.task_type = "coding"
        trace.provider = "ollama"
        trace.model = "qwen3"
        trace.tool_calls = 5
        trace.tool_failures = failures or {}
        trace.tool_call_details = []
        trace.tokens_used = 1000
        return trace

    def test_formatted_output_has_zone_headers(self):
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy

        traces = [
            self._make_trace("s1", 0.9, True),
            self._make_trace("s2", 0.2, False, {"edit_mismatch": 3}),
        ]

        result = GEPAServiceStrategy._format_traces_as_asi(traces)
        assert "SUCCESSFUL STRATEGIES" in result
        assert "FAILURE PATTERNS" in result

    def test_empty_zone_omitted(self):
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy

        traces = [self._make_trace("s1", 0.9, True)]

        result = GEPAServiceStrategy._format_traces_as_asi(traces)
        assert "SUCCESSFUL STRATEGIES" in result
        assert "FAILURE PATTERNS" not in result

    def test_recovery_zone_detected(self):
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy

        traces = [self._make_trace("s1", 0.7, True, {"edit_mismatch": 1})]

        result = GEPAServiceStrategy._format_traces_as_asi(traces)
        assert "RECOVERY PATTERNS" in result
