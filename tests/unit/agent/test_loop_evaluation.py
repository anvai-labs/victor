"""Unit tests for the hybrid loop termination evaluator package."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from victor.agent.continuation_contract import (
    ContinuationActionType,
    ContinuationStatePatch,
)
from victor.agent.loop_evaluation import (
    AgenticLoopEvaluator,
    HybridLoopEvaluator,
    LegacyEvaluator,
    LoopContext,
    LoopDecision,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_intent(intent_type="other"):
    m = MagicMock()
    m.intent = intent_type
    m.confidence = 0.8
    return m


def _base_ctx(**overrides) -> LoopContext:
    defaults = {
        "user_message": "hi",
        "task_type": "general",
        "is_analysis_task": False,
        "is_action_task": False,
        "is_direct_response": False,
        "full_content": "Hello! How can I help?",
        "content_length": 22,
        "mentioned_tools": [],
        "intent_result": _make_intent(),
        "iteration": 1,
        "continuation_prompts": 0,
        "asking_input_prompts": 0,
        "max_prompts_summary_requested": False,
        "force_tool_execution_attempts": 0,
        "synthesis_nudge_count": 0,
        "quality_score": 0.0,
        "task_completion_signals": {},
        "one_shot_mode": False,
        "compaction_occurred": False,
        "compaction_messages_removed": 0,
        "degraded_resume_state": False,
        "resume_summary": "",
        "settings": MagicMock(),
        "rl_coordinator": None,
        "provider_name": "ollama",
        "model": "test",
        "tool_budget": 20,
        "unified_tracker_config": {
            "max_total_iterations": 50,
            "max_continuation_prompts": 6,
        },
    }
    defaults.update(overrides)
    return LoopContext(**defaults)


# =============================================================================
# LoopDecision.to_directive()
# =============================================================================


class TestLoopDecisionToDirective:
    def test_finish_action_converts(self):
        decision = LoopDecision(action=ContinuationActionType.FINISH, reason="done")
        d = decision.to_directive()
        assert d.action is ContinuationActionType.FINISH
        assert d.reason == "done"
        assert d.message is None

    def test_message_preserved(self):
        decision = LoopDecision(
            action=ContinuationActionType.PROMPT_TOOL_CALL,
            reason="continue",
            message="Please use a tool.",
        )
        d = decision.to_directive()
        assert d.message == "Please use a tool."

    def test_state_patch_max_prompts_flag(self):
        decision = LoopDecision(
            action=ContinuationActionType.REQUEST_SUMMARY,
            reason="budget",
            set_max_prompts_summary_requested=True,
        )
        d = decision.to_directive()
        assert d.state_patch.max_prompts_summary_requested is True

    def test_state_patch_continuation_count(self):
        patch = ContinuationStatePatch(continuation_prompts=3)
        decision = LoopDecision(
            action=ContinuationActionType.PROMPT_TOOL_CALL,
            reason="nudge",
            state_patch=patch,
        )
        d = decision.to_directive()
        assert d.state_patch.continuation_prompts == 3


# =============================================================================
# LegacyEvaluator
# =============================================================================


class TestLegacyEvaluator:
    def test_delegates_to_continuation_strategy(self):
        from victor.agent.continuation_contract import ContinuationDirective

        evaluator = LegacyEvaluator()
        ctx = _base_ctx(is_direct_response=True, content_length=10)

        mock_directive = MagicMock(spec=ContinuationDirective)
        mock_directive.action = ContinuationActionType.FINISH
        mock_directive.reason = "delegated"
        mock_directive.message = None
        mock_directive.state_patch = ContinuationStatePatch()
        mock_directive.extracted_call = None
        mock_directive.mentioned_tools = None

        with patch("victor.agent.continuation_strategy.ContinuationStrategy") as MockStrategy:
            instance = MockStrategy.return_value
            instance.determine_continuation_action.return_value = mock_directive

            decision = evaluator.evaluate(ctx)

        assert decision.action is ContinuationActionType.FINISH
        assert decision.source == "legacy"
        instance.determine_continuation_action.assert_called_once()

    def test_passes_all_ctx_params(self):
        from victor.agent.continuation_contract import ContinuationDirective

        evaluator = LegacyEvaluator()
        ctx = _base_ctx(
            continuation_prompts=2,
            provider_name="anthropic",
            model="claude-3",
            is_analysis_task=True,
        )

        mock_directive = MagicMock(spec=ContinuationDirective)
        mock_directive.action = ContinuationActionType.FINISH
        mock_directive.reason = "ok"
        mock_directive.message = None
        mock_directive.state_patch = ContinuationStatePatch()
        mock_directive.extracted_call = None
        mock_directive.mentioned_tools = None

        with patch("victor.agent.continuation_strategy.ContinuationStrategy") as MockStrategy:
            instance = MockStrategy.return_value
            instance.determine_continuation_action.return_value = mock_directive
            evaluator.evaluate(ctx)

        call_kwargs = instance.determine_continuation_action.call_args[1]
        assert call_kwargs["continuation_prompts"] == 2
        assert call_kwargs["provider_name"] == "anthropic"
        assert call_kwargs["model"] == "claude-3"
        assert call_kwargs["is_analysis_task"] is True


# =============================================================================
# AgenticLoopEvaluator
# =============================================================================


class TestAgenticLoopEvaluator:
    def test_direct_response_finishes_immediately(self):
        evaluator = AgenticLoopEvaluator()
        ctx = _base_ctx(is_direct_response=True, content_length=50)
        decision = evaluator.evaluate(ctx)
        assert decision.action is ContinuationActionType.FINISH
        assert decision.source == "agentic_loop"

    def test_no_content_nudges_once(self):
        evaluator = AgenticLoopEvaluator()
        ctx = _base_ctx(content_length=0, full_content="", continuation_prompts=0)
        decision = evaluator.evaluate(ctx)
        assert decision.action is ContinuationActionType.PROMPT_TOOL_CALL

    def test_no_content_after_two_nudges_requests_summary(self):
        evaluator = AgenticLoopEvaluator()
        ctx = _base_ctx(content_length=0, full_content="", continuation_prompts=2)
        decision = evaluator.evaluate(ctx)
        assert decision.action is ContinuationActionType.REQUEST_SUMMARY

    def test_mentioned_tools_returns_low_confidence_finish(self):
        """Mentioned tools → confidence=0.0 so hybrid falls back to legacy."""
        evaluator = AgenticLoopEvaluator()
        ctx = _base_ctx(mentioned_tools=["read", "ls"])
        decision = evaluator.evaluate(ctx)
        assert decision.action is ContinuationActionType.FINISH
        assert decision.confidence == 0.0

    def test_high_quality_score_completes(self):
        evaluator = AgenticLoopEvaluator()
        ctx = _base_ctx(quality_score=0.9, content_length=100)
        decision = evaluator.evaluate(ctx)
        assert decision.action is ContinuationActionType.FINISH

    def test_general_task_with_content_finishes(self):
        """CONTINUE → FINISH for non-analysis, non-action tasks."""
        evaluator = AgenticLoopEvaluator()
        ctx = _base_ctx(
            quality_score=0.0,  # No explicit score
            content_length=200,
            is_analysis_task=False,
            is_action_task=False,
        )
        decision = evaluator.evaluate(ctx)
        assert decision.action is ContinuationActionType.FINISH

    def test_analysis_task_continues(self):
        """Analysis task with moderate score continues the loop."""
        evaluator = AgenticLoopEvaluator()
        ctx = _base_ctx(
            quality_score=0.5,
            content_length=100,
            is_analysis_task=True,
        )
        decision = evaluator.evaluate(ctx)
        assert decision.action is ContinuationActionType.PROMPT_TOOL_CALL

    def test_plateau_triggers_summary(self):
        """Stalled quality score (plateau) requests a summary."""
        evaluator = AgenticLoopEvaluator()
        ctx = _base_ctx(quality_score=0.5, is_analysis_task=True, continuation_prompts=1)
        # Seed the history with a plateau
        evaluator._score_history = [0.5, 0.5, 0.5]
        ctx2 = _base_ctx(quality_score=0.5, is_analysis_task=True, continuation_prompts=1)
        decision = evaluator.evaluate(ctx2)
        assert decision.action is ContinuationActionType.REQUEST_SUMMARY

    def test_score_history_accumulated_across_calls(self):
        """Score history grows with each evaluate call."""
        evaluator = AgenticLoopEvaluator()
        for score in [0.4, 0.5, 0.6]:
            ctx = _base_ctx(quality_score=score, is_analysis_task=True)
            evaluator.evaluate(ctx)
        assert len(evaluator._score_history) == 3

    def test_infer_score_general_task_with_content(self):
        """Inferred score for general task with content should be >= 0.5."""
        evaluator = AgenticLoopEvaluator()
        ctx = _base_ctx(quality_score=0.0, content_length=50)
        score = evaluator._infer_score(ctx)
        assert score >= 0.5

    def test_infer_score_empty_content(self):
        """No content → low inferred score."""
        evaluator = AgenticLoopEvaluator()
        ctx = _base_ctx(quality_score=0.0, content_length=0, full_content="")
        score = evaluator._infer_score(ctx)
        assert score < 0.5


# =============================================================================
# HybridLoopEvaluator
# =============================================================================


class TestHybridLoopEvaluator:
    def test_legacy_path_when_flag_off(self):
        hybrid = HybridLoopEvaluator()
        hybrid._use_agentic = False  # Bypass flag resolution.

        ctx = _base_ctx()
        mock_decision = LoopDecision(
            action=ContinuationActionType.FINISH, reason="legacy", source="legacy"
        )

        with patch.object(hybrid._get_legacy(), "evaluate", return_value=mock_decision):
            decision = hybrid.evaluate(ctx)

        assert decision.source == "legacy"

    def test_agentic_path_when_flag_on(self):
        hybrid = HybridLoopEvaluator()
        hybrid._use_agentic = True

        ctx = _base_ctx(is_direct_response=True, content_length=10)
        # AgenticLoopEvaluator.evaluate should return FINISH for direct response.
        decision = hybrid.evaluate(ctx)
        assert decision.action is ContinuationActionType.FINISH
        assert decision.source == "agentic_loop"

    def test_fallback_to_legacy_on_zero_confidence(self):
        """confidence=0.0 from agentic triggers legacy fallback."""
        hybrid = HybridLoopEvaluator()
        hybrid._use_agentic = True

        legacy_decision = LoopDecision(
            action=ContinuationActionType.FINISH,
            reason="legacy_handled",
            source="legacy",
        )

        with patch.object(
            hybrid._get_agentic(),
            "evaluate",
            return_value=LoopDecision(
                action=ContinuationActionType.FINISH, reason="delegate", confidence=0.0
            ),
        ):
            with patch.object(hybrid._get_legacy(), "evaluate", return_value=legacy_decision):
                decision = hybrid.evaluate(_base_ctx(mentioned_tools=["read"]))

        assert decision.source == "legacy"
        assert decision.reason == "legacy_handled"

    def test_flag_cached_after_first_resolve(self):
        """_use_agentic is populated after the first evaluate call."""
        hybrid = HybridLoopEvaluator()
        assert hybrid._use_agentic is None  # Not yet resolved.

        dummy_decision = LoopDecision(
            action=ContinuationActionType.FINISH, reason="cached_test", source="legacy"
        )
        with patch.object(hybrid, "_get_legacy") as mock_get_legacy:
            mock_get_legacy.return_value.evaluate.return_value = dummy_decision
            hybrid.evaluate(_base_ctx())

        assert hybrid._use_agentic is not None  # Now cached.

    def test_is_feature_enabled_called_once_across_evaluates(self):
        """is_feature_enabled is invoked exactly once; subsequent calls use cache."""
        hybrid = HybridLoopEvaluator()
        dummy_decision = LoopDecision(
            action=ContinuationActionType.FINISH, reason="x", source="legacy"
        )
        with patch("victor.core.feature_flags.is_feature_enabled", return_value=False) as mock_flag:
            with patch.object(hybrid, "_get_legacy") as mock_get_legacy:
                mock_get_legacy.return_value.evaluate.return_value = dummy_decision
                hybrid.evaluate(_base_ctx())
                hybrid.evaluate(_base_ctx())

        mock_flag.assert_called_once()
