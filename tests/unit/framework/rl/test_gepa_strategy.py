# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Characterization tests for the extracted GEPAStrategy + re-export shim.

Exercises the deterministic (no-LLM) heuristic paths by disabling the provider
so ``_call_llm`` returns None, then asserts the heuristic reflect/mutate/merge
behaviour is unchanged by the move.
"""

from victor.framework.rl.learners import gepa_strategy as gs
from victor.framework.rl.learners.gepa_strategy import GEPAStrategy
from victor.framework.rl.learners.trace_analysis import ExecutionTrace, ToolCallTrace


def _heuristic_strategy():
    strat = GEPAStrategy()
    # Disable the provider so _call_llm() returns None -> heuristic fallback,
    # making the strategy fully deterministic and offline for the test.
    strat._provider_name = None
    return strat


def _trace(**kw):
    base = {
        "session_id": "abcdef123456",
        "task_type": "action",
        "provider": "ollama",
        "model": "qwen",
        "tool_calls": 4,
        "tool_failures": {},
        "success": True,
        "completion_score": 0.9,
        "tokens_used": 500,
    }
    base.update(kw)
    return ExecutionTrace(**base)


class TestReExportBackCompat:
    def test_symbols_reexported_with_identity(self):
        from victor.framework.rl.learners import prompt_optimizer as po

        assert po.GEPAStrategy is gs.GEPAStrategy
        assert po.PromptOptimizationStrategy is gs.PromptOptimizationStrategy


class TestHeuristicReflect:
    def test_reflection_reports_aggregate_and_failures(self):
        traces = [
            _trace(success=False, completion_score=0.2, tool_failures={"file_not_found": 3}),
            _trace(success=True, tool_failures={}),
        ]
        out = _heuristic_strategy().reflect(traces, "GROUNDING_RULES", "base prompt")
        assert "Analysis of 2 execution traces" in out
        assert "Success rate: 1/2" in out
        assert "file_not_found: 3" in out
        # No provider -> no LLM augmentation appended.
        assert "LLM Reflection" not in out

    def test_reflection_includes_failing_exemplars(self):
        t = _trace(
            success=False,
            completion_score=0.1,
            tool_failures={"edit_mismatch": 1},
            tool_call_details=[
                ToolCallTrace(
                    tool_name="edit",
                    arguments_summary="foo.py",
                    error_detail="old_str not found",
                    success=False,
                )
            ],
        )
        out = _heuristic_strategy().reflect([t], "GROUNDING_RULES", "base")
        assert "Failing exemplars" in out
        assert "old_str not found" in out


class TestHeuristicMutate:
    def test_appends_failure_specific_guidance(self):
        reflection = "Top failure categories:\n- file_not_found: 5\n- timeout: 2"
        out = _heuristic_strategy().mutate("Base.", reflection, "GROUNDING_RULES")
        assert "Verify file paths with ls()" in out
        assert "Keep tool calls focused" in out
        assert out.startswith("Base.")

    def test_no_guidance_leaves_text_unchanged(self):
        out = _heuristic_strategy().mutate("Base.", "no known patterns here", "X")
        assert out == "Base."


class TestHeuristicMerge:
    def test_dedupes_lines_across_candidates(self):
        a = "- rule one\n- rule two"
        b = "- rule two\n- rule three"
        out = _heuristic_strategy().merge(a, b, "GROUNDING_RULES")
        assert out.count("rule two") == 1
        assert "rule one" in out and "rule three" in out
