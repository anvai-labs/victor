"""TerminationAuthority — one decision object for the loop-terminating
heuristics (consolidation PR 5, seam-duplication audit Cluster 3).

While the caller declared completion objectives that remain unmet and
iteration budget remains (``criteria_check``), spin and plateau outcomes
return ``action='guide'``: the caller injects the guided CONTINUE (spin) or
logs the suppression (plateau) instead of terminating. Without a
declaration — or once the per-iteration budget marker flips — every
decision is byte-identical to the pre-consolidation behavior (exact reason
strings, 0.1/0.2 scores, PLATEAU/exit paths).

The authority also folds the loop's ``_criteria_budget_exhausted`` writes
into one per-iteration hook (``observe_iteration``) and exposes the shared
guided-suppression predicate for the streaming blocked-streak seam.

Termination remains guaranteed without any of this: the iteration cap, the
existing bounded criteria-gate yields, and the LoopGuards wall-clock.
"""

from unittest.mock import MagicMock

import pytest

from victor.agent.services.turn_execution_runtime import TurnResult
from victor.agent.streaming.context import StreamingChatContext
from victor.agent.streaming.handler import StreamingChatHandler
from victor.agent.turn_policy import SpinState
from victor.config.settings import Settings
from victor.framework.agentic_loop import AgenticLoop
from victor.framework.enhanced_completion_evaluation import EnhancedCompletionEvaluator
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.framework.termination_authority import (
    GUIDED_SUPPRESSION_SUFFIX,
    TerminationAuthority,
)
from victor.providers.base import CompletionResponse

UNMET_STATE = {"criteria": {"findings": 30}, "findings": ["a", "b"]}
MET_STATE = {"criteria": {"findings": 2}, "findings": ["a", "b"]}
EXHAUSTED_STATE = {**UNMET_STATE, "_criteria_budget_exhausted": True}

# Golden byte-exact legacy strings (pre-consolidation branch output).
SPIN_ALL_BLOCKED_FAIL = "Spin detected: 4 consecutive fully-blocked tool batches"
SPIN_ALL_BLOCKED_GUIDED = SPIN_ALL_BLOCKED_FAIL + GUIDED_SUPPRESSION_SUFFIX
SPIN_STUCK_FAIL = "Agent stuck: 5 turns without tool calls"
SPIN_STUCK_GUIDED = SPIN_STUCK_FAIL + GUIDED_SUPPRESSION_SUFFIX
SPIN_REPETITION_FAIL = (
    "Repetitive loop detected: the model repeated the same tool calls " "for 3 consecutive turns"
)
SPIN_REPETITION_GUIDED = SPIN_REPETITION_FAIL + GUIDED_SUPPRESSION_SUFFIX
PLATEAU_SUPPRESSED_LOG = (
    "Adaptive plateau suppressed: caller-declared completion criteria are "
    "unmet with iteration budget remaining"
)


def unmet_authority() -> TerminationAuthority:
    return TerminationAuthority.for_state(dict(UNMET_STATE))


def unguided_authority() -> TerminationAuthority:
    return TerminationAuthority.for_state({})


# ============================================================================
# Authority truth table
# ============================================================================


class TestGuidedPredicateTruthTable:
    def test_no_declaration_is_unguided(self):
        assert TerminationAuthority.for_state({}).guided is False
        assert TerminationAuthority.for_state(None).guided is False
        assert TerminationAuthority.for_state({"findings": ["a"]}).guided is False

    def test_unmet_declared_with_budget_is_guided(self):
        assert unmet_authority().guided is True

    def test_declared_met_is_unguided(self):
        assert TerminationAuthority.for_state(dict(MET_STATE)).guided is False

    def test_budget_exhausted_is_unguided_even_when_unmet(self):
        assert TerminationAuthority.for_state(dict(EXHAUSTED_STATE)).guided is False

    def test_custom_criteria_check_is_respected(self):
        authority = TerminationAuthority(criteria_check=lambda: True)
        assert authority.guided is True
        assert TerminationAuthority(criteria_check=lambda: False).guided is False


class TestSpinDecisionTruthTable:
    def test_no_criteria_all_blocked_terminates_with_legacy_bytes(self):
        decision = unguided_authority().spin_decision(consecutive_all_blocked=4)
        assert decision.action == "terminate"
        assert decision.reason == SPIN_ALL_BLOCKED_FAIL
        assert decision.score == 0.1

    def test_no_criteria_repetition_terminates_with_legacy_bytes(self):
        decision = unguided_authority().spin_decision(repetition_count=3)
        assert decision.action == "terminate"
        assert decision.reason == SPIN_REPETITION_FAIL
        assert decision.score == 0.1

    def test_no_criteria_stuck_terminates_with_legacy_bytes(self):
        decision = unguided_authority().spin_decision(consecutive_no_tool_turns=5)
        assert decision.action == "terminate"
        assert decision.reason == SPIN_STUCK_FAIL
        assert decision.score == 0.2

    def test_guided_all_blocked_degrades_to_continue_bytes(self):
        decision = unmet_authority().spin_decision(consecutive_all_blocked=4)
        assert decision.action == "guide"
        assert decision.reason == SPIN_ALL_BLOCKED_GUIDED
        assert decision.score == 0.2

    def test_guided_repetition_degrades_to_continue_bytes(self):
        decision = unmet_authority().spin_decision(repetition_count=3)
        assert decision.action == "guide"
        assert decision.reason == SPIN_REPETITION_GUIDED
        assert decision.score == 0.2

    def test_guided_stuck_degrades_to_continue_bytes(self):
        decision = unmet_authority().spin_decision(consecutive_no_tool_turns=5)
        assert decision.action == "guide"
        assert decision.reason == SPIN_STUCK_GUIDED
        assert decision.score == 0.2

    def test_budget_exhausted_flips_suppression_off(self):
        authority = TerminationAuthority.for_state(dict(EXHAUSTED_STATE))
        decision = authority.spin_decision(consecutive_all_blocked=4)
        assert decision.action == "terminate"
        assert decision.reason == SPIN_ALL_BLOCKED_FAIL
        assert decision.score == 0.1

    def test_all_blocked_takes_precedence_over_repetition(self):
        decision = unmet_authority().spin_decision(consecutive_all_blocked=4, repetition_count=5)
        assert decision.reason == SPIN_ALL_BLOCKED_GUIDED

    def test_repetition_takes_precedence_over_stuck(self):
        decision = unmet_authority().spin_decision(repetition_count=3, consecutive_no_tool_turns=9)
        assert decision.reason == SPIN_REPETITION_GUIDED

    def test_repetition_threshold_boundary_matches_legacy_off_by_one(self):
        """Legacy fires at repetition_count >= threshold - 1."""
        below = unguided_authority().spin_decision(repetition_count=1, repetition_threshold=3)
        at = unguided_authority().spin_decision(repetition_count=2, repetition_threshold=3)
        assert below.action == "terminate" and below.reason == SPIN_STUCK_FAIL.replace("5", "0")
        assert at.action == "terminate"
        assert at.reason == (
            "Repetitive loop detected: the model repeated the same tool calls "
            "for 2 consecutive turns"
        )


class TestPlateauDecisionTruthTable:
    def test_no_criteria_plateau_terminates(self):
        decision = unguided_authority().plateau_decision(
            progress_scores=[0.5, 0.5, 0.5, 0.5],
            iteration=4,
            plateau_window=3,
            plateau_tolerance=0.02,
            min_plateau_iteration=3,
        )
        assert decision.action == "terminate"

    def test_guided_plateau_returns_suppression_log_line(self):
        decision = unmet_authority().plateau_decision(
            progress_scores=[0.5, 0.5, 0.5, 0.5],
            iteration=4,
            plateau_window=3,
            plateau_tolerance=0.02,
            min_plateau_iteration=3,
        )
        assert decision.action == "guide"
        assert decision.reason == PLATEAU_SUPPRESSED_LOG

    def test_budget_exhausted_plateau_terminates(self):
        decision = TerminationAuthority.for_state(dict(EXHAUSTED_STATE)).plateau_decision(
            progress_scores=[0.5, 0.5, 0.5, 0.5],
            iteration=4,
            plateau_window=3,
            plateau_tolerance=0.02,
            min_plateau_iteration=3,
        )
        assert decision.action == "terminate"

    def test_insufficient_history_decides(self):
        decision = unmet_authority().plateau_decision(
            progress_scores=[0.5, 0.5],
            iteration=4,
            plateau_window=3,
            plateau_tolerance=0.02,
            min_plateau_iteration=3,
        )
        assert decision.action == "decide"

    def test_iteration_below_minimum_decides(self):
        decision = unmet_authority().plateau_decision(
            progress_scores=[0.5, 0.5, 0.5],
            iteration=2,
            plateau_window=3,
            plateau_tolerance=0.02,
            min_plateau_iteration=3,
        )
        assert decision.action == "decide"

    def test_improving_scores_decide(self):
        decision = unmet_authority().plateau_decision(
            progress_scores=[0.4, 0.5, 0.7],
            iteration=4,
            plateau_window=3,
            plateau_tolerance=0.02,
            min_plateau_iteration=3,
        )
        assert decision.action == "decide"


class TestBudgetMarkerFolding:
    def test_observe_iteration_marks_final_funded_iteration(self):
        authority = unguided_authority()
        state = dict(UNMET_STATE)
        authority.observe_iteration(state, iteration=1, effective_max=6)
        assert state["_criteria_budget_exhausted"] is False
        authority.observe_iteration(state, iteration=6, effective_max=6)
        assert state["_criteria_budget_exhausted"] is True

    def test_observe_iteration_flip_disables_guidance(self):
        state = dict(UNMET_STATE)
        authority = TerminationAuthority.for_state(state)
        assert authority.guided is True
        authority.observe_iteration(state, iteration=6, effective_max=6)
        assert authority.guided is False

    def test_reset_budget_marker_clears_flag(self):
        authority = unguided_authority()
        state = dict(EXHAUSTED_STATE)
        authority.reset_budget_marker(state)
        assert state["_criteria_budget_exhausted"] is False


class TestSharedGuidedSuppressionPredicate:
    """The streaming blocked-streak seam consults the same predicate."""

    def test_no_declaration_returns_none(self):
        assert TerminationAuthority.guided_suppression_check({}) is None
        assert TerminationAuthority.guided_suppression_check(None) is None
        assert TerminationAuthority.guided_suppression_check({"other": 1}) is None

    def test_declared_unmet_returns_true_predicate(self):
        check = TerminationAuthority.guided_suppression_check(dict(UNMET_STATE))
        assert callable(check)
        assert check() is True

    def test_declared_marker_only_unmet_returns_true_predicate(self):
        check = TerminationAuthority.guided_suppression_check({"complete_on_key": "done"})
        assert check() is True

    def test_budget_exhausted_returns_false_predicate(self):
        check = TerminationAuthority.guided_suppression_check(dict(EXHAUSTED_STATE))
        assert check() is False


# ============================================================================
# Golden parity at the former suppression sites
# ============================================================================


def make_loop():
    loop = AgenticLoop(
        orchestrator=MagicMock(spec=[]),
        max_iterations=64,
        enable_fulfillment_check=False,
        enable_adaptive_iterations=True,
        config={"enable_topology_routing": False, "disable_enhanced_completion": True},
    )
    return loop


def _turn():
    return TurnResult(response=CompletionResponse(content="", role="assistant"))


class _MockSpinDetector:
    def __init__(self, consecutive_all_blocked=0, consecutive_no_tool_turns=0):
        self.state = SpinState.TERMINATED
        self.consecutive_all_blocked = consecutive_all_blocked
        self.consecutive_no_tool_turns = consecutive_no_tool_turns
        self.total_tool_calls = 5
        self._repetition_count = 0
        self.REPETITION_THRESHOLD = 3


class TestLegacySpinBranchGoldenParity:
    @pytest.mark.asyncio
    async def test_no_criteria_spin_fail_bytes_unchanged(self):
        loop = make_loop()
        loop.spin_detector.consecutive_all_blocked = 4
        result = await loop._evaluate_core(None, _turn(), {})
        assert result.decision == EvaluationDecision.FAIL
        assert result.score == 0.1
        assert result.reason == SPIN_ALL_BLOCKED_FAIL

    @pytest.mark.asyncio
    async def test_guided_spin_continue_bytes_unchanged(self):
        loop = make_loop()
        loop.spin_detector.consecutive_all_blocked = 4
        result = await loop._evaluate_core(None, _turn(), dict(UNMET_STATE))
        assert result.decision == EvaluationDecision.CONTINUE
        assert result.score == 0.2
        assert result.reason == SPIN_ALL_BLOCKED_GUIDED

    @pytest.mark.asyncio
    async def test_budget_exhaustion_boundary_flips_suppression_off(self):
        loop = make_loop()
        loop.spin_detector.consecutive_all_blocked = 4
        result = await loop._evaluate_core(None, _turn(), dict(EXHAUSTED_STATE))
        assert result.decision == EvaluationDecision.FAIL
        assert result.score == 0.1
        assert result.reason == SPIN_ALL_BLOCKED_FAIL

    @pytest.mark.asyncio
    async def test_guided_stuck_continue_bytes_unchanged(self):
        loop = make_loop()
        loop.spin_detector.consecutive_no_tool_turns = 5
        result = await loop._evaluate_core(None, _turn(), dict(UNMET_STATE))
        assert result.decision == EvaluationDecision.CONTINUE
        assert result.score == 0.2
        assert result.reason == SPIN_STUCK_GUIDED

    @pytest.mark.asyncio
    async def test_no_criteria_stuck_fail_bytes_unchanged(self):
        loop = make_loop()
        loop.spin_detector.consecutive_no_tool_turns = 5
        result = await loop._evaluate_core(None, _turn(), {})
        assert result.decision == EvaluationDecision.FAIL
        assert result.score == 0.2
        assert result.reason == SPIN_STUCK_FAIL

    @pytest.mark.asyncio
    async def test_guided_repetition_continue_bytes_unchanged(self):
        loop = make_loop()
        loop.spin_detector._repetition_count = 3
        result = await loop._evaluate_core(None, _turn(), dict(UNMET_STATE))
        assert result.decision == EvaluationDecision.CONTINUE
        assert result.score == 0.2
        assert result.reason == SPIN_REPETITION_GUIDED


class _MockEnhancedSpinDetector(_MockSpinDetector):
    pass


class TestEnhancedEvaluatorGoldenParity:
    @pytest.fixture
    def evaluator(self):
        return EnhancedCompletionEvaluator(
            enable_requirement_validation=True,
            enable_completion_scoring=True,
            enable_context_keywords=True,
            completion_threshold=0.80,
        )

    @pytest.mark.asyncio
    async def test_no_criteria_all_blocked_fail_bytes_unchanged(self, evaluator):
        result = await evaluator.evaluate(
            perception=None,
            action_result=_turn(),
            state={},
            spin_detector=_MockEnhancedSpinDetector(consecutive_all_blocked=4),
        )
        assert result.decision == EvaluationDecision.FAIL
        assert result.score == 0.1
        assert result.reason == SPIN_ALL_BLOCKED_FAIL

    @pytest.mark.asyncio
    async def test_guided_all_blocked_continue_bytes_unchanged(self, evaluator):
        result = await evaluator.evaluate(
            perception=None,
            action_result=_turn(),
            state=dict(UNMET_STATE),
            spin_detector=_MockEnhancedSpinDetector(consecutive_all_blocked=4),
        )
        assert result.decision == EvaluationDecision.CONTINUE
        assert result.score == 0.2
        assert result.reason == SPIN_ALL_BLOCKED_GUIDED

    @pytest.mark.asyncio
    async def test_budget_exhaustion_boundary_flips_suppression_off(self, evaluator):
        result = await evaluator.evaluate(
            perception=None,
            action_result=_turn(),
            state=dict(EXHAUSTED_STATE),
            spin_detector=_MockEnhancedSpinDetector(consecutive_all_blocked=4),
        )
        assert result.decision == EvaluationDecision.FAIL
        assert result.reason == SPIN_ALL_BLOCKED_FAIL

    @pytest.mark.asyncio
    async def test_guided_stuck_continue_bytes_unchanged(self, evaluator):
        result = await evaluator.evaluate(
            perception=None,
            action_result=_turn(),
            state=dict(UNMET_STATE),
            spin_detector=_MockEnhancedSpinDetector(consecutive_no_tool_turns=5),
        )
        assert result.decision == EvaluationDecision.CONTINUE
        assert result.score == 0.2
        assert result.reason == SPIN_STUCK_GUIDED


class TestAdaptivePlateauGoldenParity:
    def _loop_with_scores(self):
        loop = make_loop()
        loop._progress_scores = [0.5, 0.5, 0.5, 0.5]
        return loop

    def _evaluation(self):
        return EvaluationResult(
            decision=EvaluationDecision.CONTINUE, score=0.5, reason="r", metadata={}
        )

    def test_no_criteria_plateau_decision_unchanged(self):
        loop = self._loop_with_scores()
        decision = loop._check_adaptive_termination(4, self._evaluation(), state={})
        assert decision is not None
        assert decision.value == "plateau"

    def test_guided_plateau_still_suppressed(self):
        loop = self._loop_with_scores()
        decision = loop._check_adaptive_termination(4, self._evaluation(), state=dict(UNMET_STATE))
        assert decision is None

    def test_budget_exhaustion_boundary_flips_suppression_off(self):
        loop = self._loop_with_scores()
        decision = loop._check_adaptive_termination(
            4, self._evaluation(), state=dict(EXHAUSTED_STATE)
        )
        assert decision is not None
        assert decision.value == "plateau"


# ============================================================================
# Streaming blocked-streak seam (minimum scope)
# ============================================================================


class TestStreamingBlockedStreakGuidedSuppression:
    @pytest.fixture
    def handler(self):
        adder = MagicMock()
        settings = Settings(analytics_enabled=False, use_semantic_tool_selection=False)
        return StreamingChatHandler(settings=settings, message_adder=adder), adder

    def _ctx(self, consecutive=3, total=3):
        return StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=consecutive,
            total_blocked_attempts=total,
        )

    def test_no_guided_check_force_bytes_unchanged(self, handler):
        result, adder = handler
        ctx = self._ctx()
        outcome = result.check_blocked_threshold(
            ctx, all_blocked=True, consecutive_limit=4, total_limit=10
        )
        assert outcome is not None
        assert outcome.clear_tool_calls is True
        adder.add_message.assert_called()

    def test_unguided_check_force_bytes_unchanged(self, handler):
        result, _ = handler
        ctx = self._ctx()
        outcome = result.check_blocked_threshold(
            ctx,
            all_blocked=True,
            consecutive_limit=4,
            total_limit=10,
            guided_check=lambda: False,
        )
        assert outcome is not None

    def test_guided_check_suppresses_consecutive_force(self, handler):
        result, adder = handler
        ctx = self._ctx()
        outcome = result.check_blocked_threshold(
            ctx,
            all_blocked=True,
            consecutive_limit=4,
            total_limit=10,
            guided_check=lambda: True,
        )
        assert outcome is None
        adder.add_message.assert_not_called()

    def test_total_limit_backstop_survives_guided_check(self, handler):
        """Termination stays guaranteed: the total-blocked backstop is not gated."""
        result, _ = handler
        ctx = self._ctx(consecutive=0, total=20)
        outcome = result.check_blocked_threshold(
            ctx,
            all_blocked=False,
            consecutive_limit=4,
            total_limit=6,
            guided_check=lambda: True,
        )
        assert outcome is not None
