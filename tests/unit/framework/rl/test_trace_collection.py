# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Characterization tests for the extracted trace_collection helpers.

These pin the exact behaviour that moved out of PromptOptimizerLearner, and
assert the learner's delegating methods still produce identical results.
"""

import pytest

from victor.framework.rl.learners.prompt_optimizer import PromptOptimizerLearner
from victor.framework.rl.learners.trace_analysis import (
    ExecutionTrace,
    HarnessVerdict,
    ToolCallTrace,
)
from victor.framework.rl.learners.trace_collection import (
    absorb_run_kind,
    absorb_session_identity,
    categorize_failure,
    merge_traces,
    normalize_provider_label,
    score_session,
    scope_traces_to_provider,
)


def _trace(session_id="s", provider="ollama", details=None, completion_score=0.5):
    return ExecutionTrace(
        session_id=session_id,
        task_type="action",
        provider=provider,
        model="m",
        tool_calls=len(details or []),
        tool_failures={},
        success=True,
        completion_score=completion_score,
        tokens_used=0,
        tool_call_details=details or [],
    )


class TestCategorizeFailure:
    @pytest.mark.parametrize(
        "error,expected",
        [
            ("File not found: /x", "file_not_found"),
            ("cannot read directory", "read_directory"),
            ("Permission denied", "permission_denied"),
            # No "file"/"path" here, so the earlier file_not_found check is
            # skipped and this correctly falls through to edit_mismatch.
            ("old_str not found", "edit_mismatch"),
            ("match found 3 times", "edit_ambiguous"),
            ("syntax error after edit", "edit_syntax"),
            ("tool not found", "tool_not_found"),
            ("request timed out", "timeout"),
            ("no matches for query", "search_no_results"),
            ("test failed: assertion", "test_failure"),
            ("command failed", "shell_error"),
            ("tool raised an error", "tool_error"),
            ("something weird", "other"),
        ],
    )
    def test_mapping(self, error, expected):
        assert categorize_failure(error) == expected
        # Delegation matches the free function.
        assert PromptOptimizerLearner._categorize_failure(error) == expected


class TestScoreSession:
    def test_harness_verdict_wins(self):
        v = HarnessVerdict(completion_score=0.42, success=True, task_id="t", benchmark="mbpp")
        assert score_session(v, failure_rate=0.9) == (0.42, True, "harness")

    def test_proxy_when_no_verdict(self):
        assert score_session(None, 0.0) == (1.0, True, "tool_failure_proxy")
        score, success, src = score_session(None, 0.5)
        assert src == "tool_failure_proxy" and success is False
        assert score == pytest.approx(0.25)


class TestProviderIdentity:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("MoonshotProvider", "moonshot"),
            ("SandhiOllamaProvider", "ollama"),
            ("MoonshotCompatProvider", "moonshot"),
            ("", ""),
        ],
    )
    def test_normalize(self, raw, expected):
        assert normalize_provider_label(raw) == expected

    def test_absorb_identity_first_wins(self):
        session = {"provider": "", "model": ""}
        absorb_session_identity(session, {"provider": "MoonshotProvider", "model": "k2"})
        assert session == {"provider": "moonshot", "model": "k2"}
        # Second event does not override.
        absorb_session_identity(session, {"provider": "OllamaProvider", "model": "q"})
        assert session == {"provider": "moonshot", "model": "k2"}

    def test_absorb_run_kind_first_wins(self):
        session = {"run_kind": ""}
        absorb_run_kind(session, {"run_kind": "benchmark"})
        assert session["run_kind"] == "benchmark"
        absorb_run_kind(session, {"run_kind": "interactive"})
        assert session["run_kind"] == "benchmark"


class TestScopeAndMerge:
    def test_scope_falls_back_when_too_few(self):
        traces = [_trace(provider="ollama"), _trace(provider="moonshot")]
        # Only one moonshot trace, min 5 -> falls back to full pool.
        out = scope_traces_to_provider(traces, "moonshot", min_traces=5)
        assert out == traces

    def test_scope_restricts_when_enough(self):
        traces = [_trace(session_id=f"m{i}", provider="moonshot") for i in range(5)]
        traces += [_trace(session_id="o", provider="ollama")]
        out = scope_traces_to_provider(traces, "moonshot", min_traces=5)
        assert len(out) == 5
        assert all(t.provider == "moonshot" for t in out)

    def test_merge_prefers_richer_and_dedupes(self):
        thin = _trace(session_id="dup", details=[], completion_score=0.3)
        rich = _trace(
            session_id="dup",
            details=[ToolCallTrace(tool_name="read")],
            completion_score=0.9,
        )
        merged = merge_traces([thin], [rich])
        assert len(merged) == 1
        assert merged[0].tool_call_details  # the richer one won
