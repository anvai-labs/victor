"""Tests for GEPA capability gap analysis (TRACE-inspired)."""

from unittest.mock import MagicMock


class TestCapabilityGapAnalysis:
    def _make_trace(self, success=False, failures=None, details=None):
        trace = MagicMock()
        trace.success = success
        trace.completion_score = 0.8 if success else 0.3
        trace.tool_failures = failures or {}
        trace.tool_call_details = details or []
        return trace

    def _make_detail(self, success=True, error=""):
        d = MagicMock()
        d.success = success
        d.error_detail = error
        return d

    def test_function_exists(self):
        from victor.framework.rl.learners.trace_analysis import analyze_capability_gaps

        assert callable(analyze_capability_gaps)

    def test_dominant_gap_identified(self):
        from victor.framework.rl.learners.trace_analysis import analyze_capability_gaps

        traces = [
            self._make_trace(success=False, failures={"edit_mismatch": 8, "file_not_found": 2})
        ]
        gaps = analyze_capability_gaps(traces)
        assert len(gaps) >= 1
        assert gaps[0].capability == "edit_precision"
        assert gaps[0].failure_rate > 0.5

    def test_multiple_gaps_ranked(self):
        from victor.framework.rl.learners.trace_analysis import analyze_capability_gaps

        traces = [
            self._make_trace(
                success=False,
                failures={"edit_mismatch": 5, "file_not_found": 3, "timeout": 2},
            )
        ]
        gaps = analyze_capability_gaps(traces)
        assert len(gaps) >= 2
        counts = [g.failure_count for g in gaps]
        assert counts == sorted(counts, reverse=True)

    def test_no_failures_returns_empty(self):
        from victor.framework.rl.learners.trace_analysis import analyze_capability_gaps

        traces = [self._make_trace(success=True, failures={})]
        assert analyze_capability_gaps(traces) == []

    def test_example_errors_collected(self):
        from victor.framework.rl.learners.trace_analysis import analyze_capability_gaps

        detail = self._make_detail(success=False, error="old_str not found in auth.py")
        traces = [self._make_trace(success=False, failures={"edit_mismatch": 1}, details=[detail])]
        gaps = analyze_capability_gaps(traces)
        assert len(gaps) >= 1
        assert any("old_str" in e for e in gaps[0].example_errors)

    def test_max_3_gaps_returned(self):
        from victor.framework.rl.learners.trace_analysis import analyze_capability_gaps

        traces = [
            self._make_trace(
                success=False,
                failures={
                    "edit_mismatch": 5,
                    "file_not_found": 4,
                    "timeout": 3,
                    "search_no_results": 2,
                    "shell_error": 1,
                },
            )
        ]
        assert len(analyze_capability_gaps(traces)) <= 3

    def test_capability_mapping_exists(self):
        from victor.framework.rl.learners.trace_analysis import FAILURE_TO_CAPABILITY, FAILURE_HINTS

        for category in FAILURE_HINTS:
            assert category in FAILURE_TO_CAPABILITY, f"Missing mapping for {category}"


class TestCapabilityGapDataclass:
    def test_dataclass_exists(self):
        from victor.framework.rl.learners.trace_analysis import CapabilityGap

        gap = CapabilityGap(
            capability="edit_precision",
            failure_rate=0.65,
            failure_count=13,
            example_errors=["err"],
        )
        assert gap.capability == "edit_precision"


class TestGapReportFormatting:
    def test_format_function_exists(self):
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy

        assert hasattr(GEPAServiceStrategy, "_format_gap_report")

    def test_gap_report_contains_capability_name(self):
        from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy
        from victor.framework.rl.learners.trace_analysis import CapabilityGap

        gaps = [CapabilityGap("edit_precision", 0.65, 13, ["old_str not found"])]
        report = GEPAServiceStrategy._format_gap_report(gaps)
        assert "edit_precision" in report
        assert "65" in report
