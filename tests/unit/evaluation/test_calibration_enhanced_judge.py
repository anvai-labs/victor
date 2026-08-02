"""Tests for the EnhancedCompletionEvaluator calibration adapter (EVR-3, ADR-009).

The adapter's contract: production perception + a live-shaped final-turn
TurnResult, binary completion score, deterministic, zero LLM calls.
"""

from pathlib import Path

from victor.evaluation.calibration_corpus import default_corpus
from victor.evaluation.calibration_enhanced_judge import make_enhanced_judge
from victor.evaluation.judge_calibration_harness import (
    JudgeCalibrationHarness,
    Transcript,
    TranscriptStep,
    alternating_scripted_executor,
)


def _transcript(final: str, *, tools: int = 0) -> Transcript:
    steps = tuple(TranscriptStep(kind="tool", content=f"edit file {i}") for i in range(tools)) + (
        TranscriptStep(kind="message", content="working"),
    )
    return Transcript(steps=steps, final_message=final)


class TestEnhancedJudgeAdapter:
    def test_returns_binary_scores(self, tmp_path: Path):
        judge = make_enhanced_judge()
        score = judge("Create a file named hello.txt", _transcript("Done.", tools=1), tmp_path)
        assert score in (0.0, 1.0)

    def test_deterministic_across_calls(self, tmp_path: Path):
        judge = make_enhanced_judge()
        args = ("Fix the bug in utils.py", _transcript("Done — fixed.", tools=2), tmp_path)
        assert judge(*args) == judge(*args)

    def test_runs_on_the_full_harness(self, tmp_path: Path):
        # End-to-end: the adapter must survive every family of the real corpus
        # and produce a valid report (this is the parity-measurement path).
        harness = JudgeCalibrationHarness(default_corpus(variants=1))
        report = harness.run(
            alternating_scripted_executor(period=2),
            make_enhanced_judge(),
            workspace_root=tmp_path,
            keep_workspaces=True,
        )
        assert len(report.samples) == 6  # one per family
        assert all(s.judged in (0.0, 1.0) for s in report.samples)
        # The gate decision must be computable (not an exception path).
        assert isinstance(report.gate_decision.trusted, bool)
