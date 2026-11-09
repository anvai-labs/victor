"""FEP-0043 guided termination: while the caller declared completion
objectives that remain unmet and iteration budget remains, loop-terminating
heuristics degrade to guidance and never terminate the run early.

Covered heuristics:
- spin escalation (enhanced evaluator + legacy evaluation branch): FAIL on
  consecutive fully-blocked batches / stuck-without-tools becomes CONTINUE
  guidance;
- adaptive plateau early-exit: PLATEAU is suppressed while budget remains;
- the spin detector's fully-blocked threshold is configurable per session.

Default-off parity: no declaration (and the budget-exhausted marker) leaves
every decision byte-identical to the ungoverned behavior. The iteration cap
plus the existing bounded criteria-gate yields still guarantee termination.
"""

from unittest.mock import MagicMock

import pytest

from victor.agent.services.turn_execution_runtime import TurnResult
from victor.agent.turn_policy import SpinDetector, SpinDetectorConfig, SpinState
from victor.framework.agentic_loop import AgenticLoop
from victor.framework.criteria_gate import criteria_unmet_with_budget
from victor.framework.enhanced_completion_evaluation import EnhancedCompletionEvaluator
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.providers.base import CompletionResponse

UNMET_STATE = {"criteria": {"findings": 30}, "findings": ["a", "b"]}


class TestCriteriaUnmetWithBudget:
    def test_declared_unmet_with_budget_is_true(self):
        assert criteria_unmet_with_budget(UNMET_STATE) is True

    def test_declared_met_is_false(self):
        state = {"criteria": {"findings": 2}, "findings": ["a", "b"]}
        assert criteria_unmet_with_budget(state) is False

    def test_no_declaration_is_false(self):
        assert criteria_unmet_with_budget({"findings": ["a"]}) is False
        assert criteria_unmet_with_budget({}) is False
        assert criteria_unmet_with_budget(None) is False

    def test_budget_exhausted_is_false_even_when_unmet(self):
        state = {**UNMET_STATE, "_criteria_budget_exhausted": True}
        assert criteria_unmet_with_budget(state) is False

    def test_marker_only_absent_is_unmet(self):
        assert criteria_unmet_with_budget({"complete_on_key": "admitted"}) is True

    def test_marker_only_recorded_is_met(self):
        state = {"complete_on_key": "admitted", "admitted": True}
        assert criteria_unmet_with_budget(state) is False


class _MockSpinDetector:
    """Minimal spin detector stand-in matching the evaluator's duck reads."""

    def __init__(self, consecutive_all_blocked=0, consecutive_no_tool_turns=0):
        self.state = SpinState.TERMINATED
        self.consecutive_all_blocked = consecutive_all_blocked
        self.consecutive_no_tool_turns = consecutive_no_tool_turns
        self.total_tool_calls = 5


def _turn():
    return TurnResult(response=CompletionResponse(content="", role="assistant"))


class TestEnhancedEvaluatorGuidedSpin:
    @pytest.fixture
    def evaluator(self):
        return EnhancedCompletionEvaluator(
            enable_requirement_validation=True,
            enable_completion_scoring=True,
            enable_context_keywords=True,
            completion_threshold=0.80,
        )

    @pytest.mark.asyncio
    async def test_no_criteria_all_blocked_fails_unchanged(self, evaluator):
        result = await evaluator.evaluate(
            perception=None,
            action_result=_turn(),
            state={},
            spin_detector=_MockSpinDetector(consecutive_all_blocked=4),
        )
        assert result.decision == EvaluationDecision.FAIL
        assert "Spin detected" in result.reason
        assert result.score == 0.1

    @pytest.mark.asyncio
    async def test_unmet_criteria_degrades_all_blocked_fail_to_continue(self, evaluator):
        result = await evaluator.evaluate(
            perception=None,
            action_result=_turn(),
            state=dict(UNMET_STATE),
            spin_detector=_MockSpinDetector(consecutive_all_blocked=4),
        )
        assert result.decision == EvaluationDecision.CONTINUE
        assert "Spin detected" in result.reason
        assert "unmet" in result.reason

    @pytest.mark.asyncio
    async def test_budget_exhausted_all_blocked_fails_unchanged(self, evaluator):
        state = {**UNMET_STATE, "_criteria_budget_exhausted": True}
        result = await evaluator.evaluate(
            perception=None,
            action_result=_turn(),
            state=state,
            spin_detector=_MockSpinDetector(consecutive_all_blocked=4),
        )
        assert result.decision == EvaluationDecision.FAIL

    @pytest.mark.asyncio
    async def test_marker_only_unmet_degrades_fail_to_continue(self, evaluator):
        state = {"complete_on_key": "admitted"}
        result = await evaluator.evaluate(
            perception=None,
            action_result=_turn(),
            state=state,
            spin_detector=_MockSpinDetector(consecutive_all_blocked=4),
        )
        assert result.decision == EvaluationDecision.CONTINUE

    @pytest.mark.asyncio
    async def test_unmet_criteria_degrades_stuck_fail_to_continue(self, evaluator):
        result = await evaluator.evaluate(
            perception=None,
            action_result=_turn(),
            state=dict(UNMET_STATE),
            spin_detector=_MockSpinDetector(consecutive_no_tool_turns=5),
        )
        assert result.decision == EvaluationDecision.CONTINUE
        assert "Agent stuck" in result.reason

    @pytest.mark.asyncio
    async def test_no_criteria_stuck_fails_unchanged(self, evaluator):
        result = await evaluator.evaluate(
            perception=None,
            action_result=_turn(),
            state={},
            spin_detector=_MockSpinDetector(consecutive_no_tool_turns=5),
        )
        assert result.decision == EvaluationDecision.FAIL
        assert "Agent stuck" in result.reason


def make_loop():
    loop = AgenticLoop(
        orchestrator=MagicMock(spec=[]),
        max_iterations=64,
        enable_fulfillment_check=False,
        enable_adaptive_iterations=True,
        config={"enable_topology_routing": False, "disable_enhanced_completion": True},
    )
    return loop


class TestLegacyEvaluationBranchGuidedSpin:
    @pytest.mark.asyncio
    async def test_no_criteria_legacy_spin_fails_unchanged(self):
        loop = make_loop()
        loop.spin_detector.consecutive_all_blocked = 4
        result = await loop._evaluate_core(None, _turn(), {})
        assert result.decision == EvaluationDecision.FAIL
        assert "Spin detected" in result.reason

    @pytest.mark.asyncio
    async def test_unmet_criteria_legacy_spin_degrades_to_continue(self):
        loop = make_loop()
        loop.spin_detector.consecutive_all_blocked = 4
        result = await loop._evaluate_core(None, _turn(), dict(UNMET_STATE))
        assert result.decision == EvaluationDecision.CONTINUE
        assert "Spin detected" in result.reason

    @pytest.mark.asyncio
    async def test_budget_exhausted_legacy_spin_fails_unchanged(self):
        loop = make_loop()
        loop.spin_detector.consecutive_all_blocked = 4
        state = {**UNMET_STATE, "_criteria_budget_exhausted": True}
        result = await loop._evaluate_core(None, _turn(), state)
        assert result.decision == EvaluationDecision.FAIL


class TestAdaptivePlateauGuidance:
    def _plateau_scores(self, loop):
        loop._progress_scores = [0.5, 0.5, 0.5, 0.5]

    def test_plateau_fires_without_declaration(self):
        loop = make_loop()
        self._plateau_scores(loop)
        evaluation = EvaluationResult(
            decision=EvaluationDecision.CONTINUE, score=0.5, reason="r", metadata={}
        )
        decision = loop._check_adaptive_termination(4, evaluation, state={})
        assert decision is not None
        assert decision.value == "plateau"

    def test_plateau_suppressed_while_criteria_unmet_with_budget(self):
        loop = make_loop()
        self._plateau_scores(loop)
        evaluation = EvaluationResult(
            decision=EvaluationDecision.CONTINUE, score=0.5, reason="r", metadata={}
        )
        decision = loop._check_adaptive_termination(4, evaluation, state=dict(UNMET_STATE))
        assert decision is None

    def test_plateau_not_suppressed_when_budget_exhausted(self):
        loop = make_loop()
        self._plateau_scores(loop)
        evaluation = EvaluationResult(
            decision=EvaluationDecision.CONTINUE, score=0.5, reason="r", metadata={}
        )
        state = {**UNMET_STATE, "_criteria_budget_exhausted": True}
        decision = loop._check_adaptive_termination(4, evaluation, state=state)
        assert decision is not None
        assert decision.value == "plateau"


class TestSpinDetectorThresholdKnob:
    def test_default_threshold_matches_legacy_constant(self):
        detector = SpinDetector()
        assert detector.config.max_consecutive_all_blocked == 4
        detector.consecutive_all_blocked = 4
        assert detector.state == SpinState.TERMINATED

    def test_raised_threshold_defers_termination(self):
        detector = SpinDetector(config=SpinDetectorConfig(max_consecutive_all_blocked=8))
        detector.consecutive_all_blocked = 4
        assert detector.state == SpinState.BLOCKED
        detector.consecutive_all_blocked = 8
        assert detector.state == SpinState.TERMINATED
