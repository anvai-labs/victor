"""TDD tests for AgenticLoop DECIDE-phase wiring — Wave A1.

Qualifies score-velocity boundaries. Actual driver wiring and cross-run isolation
are exercised in test_agentic_loop_verification rather than source inspection or
copies of the decision implementation.
"""

from __future__ import annotations


class TestBackslideGuardWired:
    """The DECIDE block must call _apply_backslide_guard on COMPLETE decisions."""

    def test_backslide_guard_downgrades_complete_with_negative_velocity(self):
        """Guard downgrades COMPLETE to CONTINUE when score drops significantly."""
        from victor.framework.agentic_loop import AgenticLoop
        from victor.framework.evaluation_nodes import (
            EvaluationDecision,
            EvaluationResult,
        )

        loop = AgenticLoop.__new__(AgenticLoop)
        loop._score_history = [0.99]  # previous score was very high

        evaluation = EvaluationResult(
            decision=EvaluationDecision.COMPLETE,
            score=0.75,  # dropped significantly: velocity = 0.75 - 0.99 = -0.24 < -0.10
            reason="test",
        )
        result = loop._apply_backslide_guard(evaluation)
        assert result.decision != EvaluationDecision.COMPLETE
        assert result.decision == EvaluationDecision.CONTINUE

    def test_backslide_guard_allows_complete_with_positive_velocity(self):
        """Guard preserves COMPLETE when score improves from previous turn."""
        from victor.framework.agentic_loop import AgenticLoop
        from victor.framework.evaluation_nodes import (
            EvaluationDecision,
            EvaluationResult,
        )

        loop = AgenticLoop.__new__(AgenticLoop)
        loop._score_history = [0.70]  # previous score was lower

        evaluation = EvaluationResult(
            decision=EvaluationDecision.COMPLETE,
            score=0.90,  # improved: velocity = +0.20 > 0
            reason="test",
        )
        result = loop._apply_backslide_guard(evaluation)
        assert result.decision == EvaluationDecision.COMPLETE

    def test_score_history_grows_after_each_guard_call(self):
        """_apply_backslide_guard appends score to _score_history on each call."""
        from victor.framework.agentic_loop import AgenticLoop
        from victor.framework.evaluation_nodes import (
            EvaluationDecision,
            EvaluationResult,
        )

        loop = AgenticLoop.__new__(AgenticLoop)
        loop._score_history = []

        eval1 = EvaluationResult(decision=EvaluationDecision.COMPLETE, score=0.8, reason="")
        loop._apply_backslide_guard(eval1)
        assert len(loop._score_history) == 1

        eval2 = EvaluationResult(decision=EvaluationDecision.COMPLETE, score=0.85, reason="")
        loop._apply_backslide_guard(eval2)
        assert len(loop._score_history) == 2

    def test_guard_preserves_complete_on_first_turn(self):
        """With empty score_history (first turn), guard should not downgrade."""
        from victor.framework.agentic_loop import AgenticLoop
        from victor.framework.evaluation_nodes import (
            EvaluationDecision,
            EvaluationResult,
        )

        loop = AgenticLoop.__new__(AgenticLoop)
        loop._score_history = []

        evaluation = EvaluationResult(
            decision=EvaluationDecision.COMPLETE,
            score=0.95,
            reason="done",
        )
        result = loop._apply_backslide_guard(evaluation)
        assert result.decision == EvaluationDecision.COMPLETE
