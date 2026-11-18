"""AgenticLoopEvaluator — PERCEIVE→EVALUATE→DECIDE termination backend.

This evaluator implements the same decision contract as ``LegacyEvaluator``
but uses the research-paper-grounded evaluation primitives from
``victor.framework.evaluation_nodes`` (progress_tracking_evaluator,
convergence_evaluator) and will eventually delegate to the async
``FulfillmentDetector`` once the chat loop is fully async.

Active when ``USE_STATEGRAPH_AGENTIC_LOOP`` is ON.

Design notes
------------
* **Quality score as progress signal**: ``LoopContext.quality_score`` (0–1,
  from runtime intelligence scoring) is used as the input to
  ``progress_tracking_evaluator``.  When the loop is fully async, the score
  will come from ``FulfillmentDetector.check_fulfillment()``; for now the
  last known quality score from ``stream_ctx`` is used.

* **Direct-response fast path**: When ``ctx.is_direct_response`` is True and
  the model has already produced content, we immediately FINISH — matching
  the ``_should_finish_direct_response`` guard in ``ContinuationStrategy``
  but without the multi-thousand-line decision tree.

* **FulfillmentDetector hook point** (async, future): marked with TODO below.
  Once ``chat_stream_executor`` is refactored to be fully async, replace the
  heuristic quality score with::

      result = await fulfillment_detector.check_fulfillment(
          task_type=_coerce_task_type(ctx.task_type),
          criteria={...},
          context={...},
      )
      score = result.score

* **State maintenance**: A single ``AgenticLoopEvaluator`` instance is
  created per ``HybridLoopEvaluator``, which is created per
  ``IntentClassificationHandler``.  Score history is maintained across calls
  to enable plateau detection.
"""

from __future__ import annotations

import logging
from typing import List

from victor.agent.continuation_contract import (
    ContinuationActionType,
    ContinuationStatePatch,
)
from victor.agent.loop_evaluation.protocol import (
    LoopContext,
    LoopDecision,
    LoopEvaluator,
)
from victor.framework.evaluation_nodes import EvaluationDecision

logger = logging.getLogger(__name__)

# Thresholds — calibrated to match empirical quality score distributions.
_COMPLETE_THRESHOLD = 0.85  # Quality score ≥ this → task complete
_PLATEAU_WINDOW = 3  # Iterations to detect plateau
_PLATEAU_TOLERANCE = 0.02  # Minimum improvement to not be a plateau


class AgenticLoopEvaluator(LoopEvaluator):
    """PERCEIVE→EVALUATE→DECIDE termination evaluator using EvaluationNode primitives.

    Maintains score history across calls to enable plateau detection without
    requiring the caller to pass history explicitly.
    """

    def __init__(self) -> None:
        self._score_history: List[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, ctx: LoopContext) -> LoopDecision:
        """Evaluate the current loop state and return a termination decision.

        Decision order (mirrors PERCEIVE→EVALUATE→DECIDE cycle):

        1. PERCEIVE: Classify the turn (direct response? has content?).
        2. EVALUATE: Score progress via quality signal + plateau detection.
        3. DECIDE: Map EvaluationDecision to ContinuationActionType.
        """
        # ---- PERCEIVE ---------------------------------------------------------

        # A model question needs the legacy classifier’s clarification rules.
        # In particular, one-shot mode must return control instead of treating
        # a direct-response prompt as satisfied merely because content exists.
        intent = getattr(ctx.intent_result, "intent", None)
        intent_value = getattr(intent, "value", intent)
        if intent_value == "asking_input":
            return LoopDecision(
                action=ContinuationActionType.FINISH,
                reason="asking_input_delegated_to_legacy",
                confidence=0.0,
                source="agentic_loop",
            )

        # Direct-response fast path: model answered, no tools needed.
        if ctx.is_direct_response and ctx.content_length > 0:
            logger.info(
                "[agentic-eval] direct_response satisfied (len=%d) — FINISH",
                ctx.content_length,
            )
            return self._finish("direct-response-complete", confidence=0.99)

        # No content and no tool calls: model stalled — nudge once.
        if ctx.content_length == 0:
            if ctx.continuation_prompts >= 2:
                return self._request_summary("no_content_after_nudges")
            return self._prompt_tool_call(
                "No content produced — nudging model to act",
                continuation_prompts=ctx.continuation_prompts,
            )

        # Mentioned tools but didn't call them — let legacy handle hallucination.
        # (AgenticLoop delegates hallucination recovery to LegacyEvaluator for now.)
        if ctx.mentioned_tools:
            return self._finish(
                "mentioned_tools_delegated_to_legacy",
                confidence=0.0,  # Signal to hybrid to fall back.
            )

        # ---- EVALUATE (quality / progress signal) --------------------------
        # TODO(async-fulfillment): Replace with FulfillmentDetector call once
        # chat_stream_executor is fully async:
        #   result = await fulfillment_detector.check_fulfillment(
        #       task_type=_coerce_task_type(ctx.task_type),
        #       criteria={...},
        #       context={"full_content": ctx.full_content, ...},
        #   )
        #   score = result.score
        score = ctx.quality_score if ctx.quality_score > 0.0 else self._infer_score(ctx)
        self._score_history.append(score)
        iteration = len(self._score_history)

        eval_result = self._run_evaluators(score, iteration)

        # ---- DECIDE ----------------------------------------------------------
        return self._map_decision(eval_result, ctx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_score(self, ctx: LoopContext) -> float:
        """Infer a quality score when runtime intelligence hasn't scored yet.

        Heuristic: penalise long responses on general tasks (model padding),
        reward short crisp responses.  Analysis/action tasks get a baseline
        continuation score so we don't terminate too early.
        """
        if ctx.is_analysis_task or ctx.is_action_task:
            # Multi-step tasks: assume partial progress, let plateau detect stalling.
            return 0.5 + min(0.1 * ctx.iteration, 0.3)
        # General task: a non-empty first response is likely complete.
        return 0.7 if ctx.content_length > 0 else 0.3

    def _run_evaluators(self, score: float, iteration: int) -> EvaluationDecision:
        """Apply progress tracking + convergence logic to the score history."""

        # Check completion.
        if score >= _COMPLETE_THRESHOLD:
            return EvaluationDecision.COMPLETE

        # Plateau detection.
        if len(self._score_history) >= _PLATEAU_WINDOW:
            recent = self._score_history[-_PLATEAU_WINDOW:]
            improvement = max(recent) - min(recent)
            if improvement < _PLATEAU_TOLERANCE:
                logger.info(
                    "[agentic-eval] plateau after %d iters (improvement=%.3f) — FAIL",
                    iteration,
                    improvement,
                )
                return EvaluationDecision.FAIL

        # Check regression vs previous turn.
        if len(self._score_history) >= 2:
            delta = self._score_history[-1] - self._score_history[-2]
            if delta < -0.1:
                return EvaluationDecision.RETRY

        return EvaluationDecision.CONTINUE

    def _map_decision(self, decision: EvaluationDecision, ctx: LoopContext) -> LoopDecision:
        """Map EvaluationDecision to a LoopDecision with continuation action."""

        if decision == EvaluationDecision.COMPLETE:
            logger.info("[agentic-eval] COMPLETE — finishing turn")
            return self._finish("agentic_eval_complete", confidence=0.95)

        if decision == EvaluationDecision.FAIL:
            # Plateau or persistent failure — request summary.
            if ctx.continuation_prompts >= 1:
                return self._request_summary("agentic_eval_plateau_or_fail")
            # First plateau: nudge once before requesting summary.
            return self._prompt_tool_call(
                "Progress plateau detected — nudging for concrete action",
                continuation_prompts=ctx.continuation_prompts,
            )

        if decision == EvaluationDecision.RETRY:
            # Regression: try again unless budget is exhausted.
            max_prompts = ctx.unified_tracker_config.get("max_continuation_prompts", 3)
            if ctx.continuation_prompts >= max_prompts:
                return self._request_summary("agentic_eval_regression_budget_exceeded")
            return self._prompt_tool_call(
                "Quality regression detected — retrying",
                continuation_prompts=ctx.continuation_prompts,
            )

        # CONTINUE: analysis/action tasks keep going; general tasks finish.
        if not ctx.is_analysis_task and not ctx.is_action_task:
            logger.info("[agentic-eval] general task CONTINUE → FINISH (single-turn)")
            return self._finish("agentic_eval_general_turn_complete", confidence=0.9)

        max_prompts = ctx.unified_tracker_config.get("max_continuation_prompts", 6)
        if ctx.continuation_prompts >= max_prompts:
            return self._request_summary("agentic_eval_max_prompts_reached")

        return self._prompt_tool_call(
            "Continuing agentic task",
            continuation_prompts=ctx.continuation_prompts,
        )

    # ------------------------------------------------------------------
    # Result factories
    # ------------------------------------------------------------------

    def _finish(self, reason: str, confidence: float = 1.0) -> LoopDecision:
        return LoopDecision(
            action=ContinuationActionType.FINISH,
            reason=reason,
            confidence=confidence,
            source="agentic_loop",
        )

    def _request_summary(self, reason: str) -> LoopDecision:
        patch = ContinuationStatePatch(max_prompts_summary_requested=True)
        return LoopDecision(
            action=ContinuationActionType.REQUEST_SUMMARY,
            reason=reason,
            message=(
                "Please provide a summary of your findings/work so far. Conclude your response."
            ),
            source="agentic_loop",
            state_patch=patch,
            set_max_prompts_summary_requested=True,
        )

    def _prompt_tool_call(self, reason: str, continuation_prompts: int = 0) -> LoopDecision:
        patch = ContinuationStatePatch(continuation_prompts=continuation_prompts + 1)
        return LoopDecision(
            action=ContinuationActionType.PROMPT_TOOL_CALL,
            reason=reason,
            message="Continue. Use appropriate tools if needed.",
            source="agentic_loop",
            state_patch=patch,
        )
