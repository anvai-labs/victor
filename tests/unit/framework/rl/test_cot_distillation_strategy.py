# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the CoT distillation strategy.

The load-bearing property is *faithfulness*: when a trace carries tool-call
detail, the distilled scaffold must reflect that trace's real trajectory, not
a fixed template.
"""

from victor.framework.rl.learners.prompt_optimizer import ExecutionTrace, ToolCallTrace
from victor.framework.rl.learners.strategies.cot_distillation_strategy import (
    CoTDistillationStrategy,
)


def _strong_trace(details, *, failures=None, tool_calls=5, score=0.9):
    return ExecutionTrace(
        session_id="s1",
        task_type="action",
        provider="anthropic",
        model="sonnet",
        tool_calls=tool_calls,
        tool_failures=failures or {},
        success=True,
        completion_score=score,
        tokens_used=1200,
        tool_call_details=details,
    )


class TestCoTFaithfulness:
    def test_scaffold_follows_the_real_tool_trajectory(self):
        details = [
            ToolCallTrace(
                tool_name="code_search",
                reasoning_before="Locate the failing module.",
                success=True,
            ),
            ToolCallTrace(tool_name="read", success=True),
            ToolCallTrace(tool_name="read", success=True),  # consecutive → collapsed
            ToolCallTrace(
                tool_name="edit",
                reasoning_before="Apply the fix to the handler",
                success=True,
            ),
            ToolCallTrace(tool_name="run_tests", success=True),
        ]
        strategy = CoTDistillationStrategy()

        out = strategy.reflect([_strong_trace(details)], "FEW_SHOT_EXAMPLES", "base")

        # Real tool names appear, phrased with their phase verb.
        assert "DISCOVER with code_search" in out
        assert "EDIT with edit" in out
        assert "VERIFY with run_tests" in out
        # Recorded reasoning is carried into the step.
        assert "Locate the failing module" in out
        # Consecutive identical tools collapse to a single step.
        assert out.count("READ with read") == 1
        # Header advertises the honest basis.
        assert "observed trace" in out
        # Order is preserved.
        assert out.index("code_search") < out.index("edit") < out.index("run_tests")

    def test_failed_calls_are_not_distilled(self):
        details = [
            ToolCallTrace(tool_name="code_search", success=True),
            ToolCallTrace(tool_name="shell", success=False),  # failed → skipped
            ToolCallTrace(tool_name="edit", success=True),
        ]
        strategy = CoTDistillationStrategy()

        out = strategy.reflect([_strong_trace(details)], "FEW_SHOT_EXAMPLES", "base")

        assert "with shell" not in out
        assert "DISCOVER with code_search" in out
        assert "EDIT with edit" in out

    def test_recovery_step_only_for_failures_actually_hit(self):
        details = [
            ToolCallTrace(tool_name="read", success=True),
            ToolCallTrace(tool_name="edit", success=True),
        ]
        strategy = CoTDistillationStrategy()

        with_mismatch = strategy.reflect(
            [_strong_trace(details, failures={"edit_mismatch": 2})],
            "FEW_SHOT_EXAMPLES",
            "base",
        )
        without = strategy.reflect([_strong_trace(details)], "FEW_SHOT_EXAMPLES", "base")

        assert "re-read the file and copy old_str exactly" in with_mismatch
        assert "re-read the file and copy old_str exactly" not in without

    def test_steps_are_capped_at_max_steps(self):
        details = [
            ToolCallTrace(tool_name=name, success=True)
            for name in ("code_search", "read", "graph", "edit", "run_tests", "git")
        ]
        strategy = CoTDistillationStrategy(max_steps=3)

        out = strategy.reflect([_strong_trace(details, tool_calls=6)], "X", "base")

        step_lines = [line for line in out.splitlines() if line[:2] in ("1.", "2.", "3.", "4.")]
        assert len(step_lines) == 3


class TestCoTFallback:
    def test_generic_scaffold_when_no_tool_detail(self):
        strategy = CoTDistillationStrategy()
        trace = _strong_trace([], tool_calls=6)  # no tool_call_details

        out = strategy.reflect([trace], "FEW_SHOT_EXAMPLES", "base")

        assert "DISCOVER: Use code_search" in out
        assert "the success profile" in out
        # tools > 5 → verify step present in the fallback template
        assert "VERIFY" in out


class TestCoTGating:
    def test_no_output_without_strong_traces(self):
        weak = ExecutionTrace(
            session_id="s",
            task_type="action",
            provider="ollama",
            model="qwen",
            tool_calls=1,
            tool_failures={},
            success=False,
            completion_score=0.1,
            tokens_used=100,
        )
        assert CoTDistillationStrategy().reflect([weak], "X", "base") == ""

    def test_no_transfer_when_target_already_close(self):
        details = [ToolCallTrace(tool_name="read", success=True)]
        source = _strong_trace(details, score=0.9)
        source.provider = "anthropic"
        target = _strong_trace(details, score=0.85)
        target.provider = "ollama"

        # Gap 0.05 < default min_score_gap 0.15 → nothing to transfer.
        out = CoTDistillationStrategy().reflect(
            [source, target], "X", "base", target_provider="ollama"
        )
        assert out == ""

    def test_mutate_appends_reflection(self):
        strategy = CoTDistillationStrategy()
        assert strategy.mutate("base", "scaffold", "X") == "base\n\nscaffold"
        assert strategy.mutate("base", "", "X") == "base"
