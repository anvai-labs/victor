# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Enhanced Completion Evaluation - Requirement-driven completion detection.

This module provides enhanced completion evaluation for the AgenticLoop by
integrating:
1. RequirementValidator - validates against extracted requirements
2. CompletionScorer - combines multiple signals into unified score
3. ContextAwareKeywordDetector - task-type-specific keyword detection

This replaces and enhances the existing _evaluate() method in AgenticLoop
with requirement-driven, multi-signal completion detection.

Design Principles:
1. Requirement-driven: Completion based on what was asked
2. Multi-signal fusion: Combines diverse signals for robustness
3. Backward compatible: Falls back to legacy behavior when components unavailable
4. Explainable: Clear logging of why completion was triggered

Based on research from:
- arXiv:2603.07379 - Agentic RAG Taxonomy (requirement extraction)
- arXiv:2604.07415 - SubSearch intermediate reward design
- arXiv:2601.21268 - Meta-evaluation without ground truth

Example:
    from victor.framework.enhanced_completion_evaluation import (
        EnhancedCompletionEvaluator,
        EvaluationDecision,
        EvaluationResult,
    )

    evaluator = EnhancedCompletionEvaluator()

    result = await evaluator.evaluate(
        perception=perception,
        action_result=turn_result,
        state=state,
        fulfillment_detector=fulfillment,
    )

    if result.decision == EvaluationDecision.COMPLETE:
        print("Task complete!")
"""

from __future__ import annotations

import enum
import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.framework.completion_scorer import (
    CompletionScorer,
    CompletionScore,
    CompletionSignal,
    TaskType,
)
from victor.framework.fulfillment import TaskType as FulfillmentTaskType
from victor.framework.context_aware_keyword_detector import (
    ContextAwareKeywordDetector,
)
from victor.framework.evaluation_nodes import EvaluationDecision
from victor.framework.runtime_evaluation_policy import (
    CompletionCalibration,
    RuntimeEvaluationPolicy,
)
from victor.framework.requirement_validator import (
    RequirementValidator,
    RequirementResult,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.agent.services.turn_execution_runtime import TurnResult
    from victor.framework.agentic_loop import SpinState
    from victor.framework.fulfillment import FulfillmentDetector
    from victor.framework.perception_integration import Perception


class EnhancedCompletionEvaluator:
    """Enhanced completion evaluator with requirement-driven detection.

    This evaluator provides algorithmic completion detection:
    - Uses PerceptionIntegration to understand task requirements
    - Validates completion against extracted requirements
    - Combines multiple signals (fulfillment, keywords, confidence)
    - Provides unified, explainable completion decisions

    Usage:
        evaluator = EnhancedCompletionEvaluator()

        result = await evaluator.evaluate(
            perception=perception,
            action_result=turn_result,
            state=state,
            fulfillment_detector=fulfillment,
            spin_detector=spin_detector,
        )
    """

    def __init__(
        self,
        enable_requirement_validation: bool = True,
        enable_completion_scoring: bool = True,
        enable_context_keywords: bool = True,
        completion_threshold: Optional[float] = None,
        enable_calibrated_completion: Optional[bool] = None,
        evaluation_policy: Optional[RuntimeEvaluationPolicy] = None,
        fuser_config: Optional[Any] = None,
    ):
        """Initialize enhanced evaluator.

        Args:
            enable_requirement_validation: Enable requirement-driven validation
            enable_completion_scoring: Enable multi-signal scoring
            enable_context_keywords: Enable context-aware keyword detection
            completion_threshold: Threshold for completion decision (0.0-1.0).
                When omitted, uses the shared runtime evaluation policy threshold.
            enable_calibrated_completion: Whether to require execution support
                before accepting strong completion scores. Defaults to feature flag.
        """
        self.enable_requirement_validation = enable_requirement_validation
        self.enable_completion_scoring = enable_completion_scoring
        self.enable_context_keywords = enable_context_keywords
        policy = evaluation_policy or RuntimeEvaluationPolicy()
        if completion_threshold is not None and completion_threshold != policy.completion_threshold:
            policy = replace(policy, completion_threshold=completion_threshold)
        self._evaluation_policy = policy
        self.completion_threshold = self._evaluation_policy.completion_threshold
        if enable_calibrated_completion is None:
            try:
                from victor.core.feature_flags import (
                    FeatureFlag,
                    get_feature_flag_manager,
                )

                enable_calibrated_completion = get_feature_flag_manager().is_enabled(
                    FeatureFlag.USE_CALIBRATED_COMPLETION
                )
            except Exception:
                enable_calibrated_completion = False
        self.enable_calibrated_completion = enable_calibrated_completion

        # Fuser config (Wave G: typed config for CompletionSignalFuser)
        self._fuser_config = fuser_config

        # Initialize components
        self.requirement_validator = RequirementValidator()
        self.completion_scorer = CompletionScorer(default_threshold=self.completion_threshold)
        self.keyword_detector = ContextAwareKeywordDetector()

    async def evaluate(
        self,
        perception: Optional[Perception],
        action_result: Any,
        state: Dict[str, Any],
        fulfillment_detector: Optional[Any] = None,
        spin_detector: Optional[Any] = None,
    ) -> Any:
        """Evaluate completion with enhanced detection.

        Args:
            perception: Perception from PerceptionIntegration
            action_result: TurnResult with response/tool results
            state: Current state dictionary
            fulfillment_detector: Optional FulfillmentDetector instance
            spin_detector: Optional SpinDetector instance

        Returns:
            EvaluationResult with decision and rationale
        """
        from victor.framework.evaluation_nodes import EvaluationResult
        from victor.agent.services.turn_execution_runtime import TurnResult

        # === PRIORITY 1: Spin Detection (always check) ===
        if spin_detector is not None:
            spin_result = self._check_spin_detection(spin_detector, action_result)
            if spin_result is not None:
                return spin_result

        # === PRIORITY 2: Q&A Shortcut (fast path for questions) ===
        if isinstance(action_result, TurnResult):
            qa_result = self._check_qa_shortcut(action_result)
            if qa_result is not None:
                return qa_result

        # === PRIORITY 3: Enhanced Completion Detection ===
        if self.enable_completion_scoring and perception is not None:
            enhanced_result = await self._evaluate_enhanced(
                perception=perception,
                action_result=action_result,
                state=state,
                fulfillment_detector=fulfillment_detector,
            )
            if enhanced_result is not None:
                return enhanced_result

        # === PRIORITY 4: Legacy Fallback ===
        return self._evaluate_legacy(
            perception=perception,
            action_result=action_result,
            state=state,
        )

    async def _evaluate_enhanced(
        self,
        perception: Perception,
        action_result: Any,
        state: Dict[str, Any],
        fulfillment_detector: Optional[Any],
    ) -> Optional[Any]:
        """Enhanced evaluation with requirement-driven detection.

        Returns None if enhanced evaluation cannot be applied,
        allowing fallback to legacy logic.
        """
        from victor.framework.evaluation_nodes import EvaluationResult

        try:
            # Step 1: Validate against requirements
            requirement_result = None
            if self.enable_requirement_validation:
                requirements = getattr(perception, "requirements", None)
                if requirements:
                    requirement_result = self.requirement_validator.validate_completion(
                        requirements=requirements,
                        action_result=action_result,
                        context=state,
                    )

                    # Fast fail: Critical requirements not met
                    if (
                        requirement_result is not None
                        and not requirement_result.is_satisfied
                        and len(requirement_result.critical_gaps) > 0
                    ):
                        return EvaluationResult(
                            decision=EvaluationDecision.CONTINUE,
                            score=requirement_result.satisfaction_score,
                            reason=f"Critical requirements not met: {requirement_result.summary}",
                        )

            # Step 2: Check fulfillment (task-specific validation)
            fulfillment_result = None
            if fulfillment_detector is not None and hasattr(perception, "task_type"):
                try:
                    task_type = self._map_to_task_type(perception)
                    fulfillment_result = await fulfillment_detector.check_fulfillment(
                        task_type=self._map_to_fulfillment_task_type(task_type),
                        criteria=state.get("criteria", {}),
                        context=state,
                    )

                    # Fast success: Task fulfilled
                    if (
                        fulfillment_result is not None
                        and hasattr(fulfillment_result, "is_fulfilled")
                        and fulfillment_result.is_fulfilled
                    ):
                        return EvaluationResult(
                            decision=EvaluationDecision.COMPLETE,
                            score=fulfillment_result.score,
                            reason=f"Task fulfilled: {fulfillment_result.reason}",
                        )
                except Exception as e:
                    logger.warning(f"Fulfillment check failed: {e}")

            # Step 3: Detect completion signals (context-aware keywords)
            keyword_result = None
            if self.enable_context_keywords:
                response = self._extract_response(action_result)
                if response:
                    task_type = self._map_to_task_type(perception)
                    requirements = getattr(perception, "requirements", None)

                    keyword_result = self.keyword_detector.detect_completion(
                        response=response,
                        task_type=task_type,
                        requirements=requirements,
                    )

                    # Fast fail: Model requesting continuation
                    if keyword_result.is_continuation_request:
                        return EvaluationResult(
                            decision=EvaluationDecision.CONTINUE,
                            score=0.6,
                            reason="Model offered continuation - awaiting user direction",
                        )

            # Step 4: Calculate unified completion score
            completion_score = self.completion_scorer.calculate_completion_score(
                requirement_result=requirement_result,
                fulfillment_result=fulfillment_result,
                keyword_result=keyword_result,
                perception=perception,
                task_type=self._map_to_task_type(perception),
            )

            # Step 4b: Fuse the four component signals through CompletionSignalFuser.
            # The fuser owns weighted aggregation + velocity tracking so the scorer's
            # per-component breakdown is used as authoritative signal inputs.
            from victor.framework.completion_signal_fuser import CompletionSignalFuser

            fuser = CompletionSignalFuser(config=self._fuser_config)
            fused = fuser.fuse(
                fulfillment=completion_score.fulfillment_score,
                requirement=completion_score.requirement_score,
                keyword=completion_score.keyword_score,
                confidence=completion_score.confidence_score,
                score_history=list(getattr(self, "_score_history", [])),
            )
            # Use the fused score as the composite signal; fall back to scorer total
            # if the fused score is not meaningfully different (guard against regressions).
            fused_score = fused.score

            threshold = completion_score.threshold
            score = fused_score
            metadata: Dict[str, Any] = {
                "fused_velocity": fused.velocity,
                "fused_signals": fused.signals_used,
            }

            if self.enable_calibrated_completion:
                calibration = self._calibrate_completion(
                    completion_score=completion_score,
                    perception=perception,
                    action_result=action_result,
                    state=state,
                    requirement_result=requirement_result,
                    keyword_result=keyword_result,
                )
                metadata["calibration"] = calibration.to_metadata()
                threshold = calibration.threshold
                score = calibration.calibrated_score

                if calibration.requires_additional_support:
                    return self._evaluation_policy.build_completion_evaluation(
                        score=score,
                        threshold=threshold,
                        metadata=metadata,
                        requires_additional_support=True,
                    )

            # Step 5: Make decision based on score
            return self._evaluation_policy.build_completion_evaluation(
                score=score,
                threshold=threshold,
                metadata=metadata,
            )

        except Exception as e:
            logger.warning(f"Enhanced evaluation failed: {e}, falling back to legacy")
            return None

    def _check_spin_detection(self, spin_detector: Any, action_result: Any = None) -> Optional[Any]:
        """Check for spin detection (agent stuck).

        Returns EvaluationResult if spin detected, None otherwise.
        When consecutive_no_tool_turns triggered, yields COMPLETE instead of FAIL
        if the model already provided a substantial response — that means it finished
        analysis and is delivering its answer, not that it is stuck.
        """
        from victor.framework.evaluation_nodes import EvaluationResult
        from victor.framework.agentic_loop import SpinState

        if not hasattr(spin_detector, "state"):
            return None

        spin_state = spin_detector.state

        if spin_state == SpinState.TERMINATED:
            if hasattr(spin_detector, "consecutive_all_blocked"):
                if spin_detector.consecutive_all_blocked > 0:
                    return EvaluationResult(
                        decision=EvaluationDecision.FAIL,
                        score=0.1,
                        reason=(
                            f"Spin detected: {spin_detector.consecutive_all_blocked} "
                            "consecutive fully-blocked tool batches"
                        ),
                    )

            if hasattr(spin_detector, "consecutive_no_tool_turns"):
                had_prior_tools = getattr(spin_detector, "total_tool_calls", 0) > 0
                response_text = (
                    self._extract_response(action_result) if action_result is not None else ""
                ) or ""
                response_substantial = len(
                    response_text.strip()
                ) > 100 and not self._is_intent_only_response(response_text)

                if had_prior_tools and response_substantial:
                    # Model used tools earlier then delivered a prose response — it
                    # gathered data and is now answering.
                    return EvaluationResult(
                        decision=EvaluationDecision.COMPLETE,
                        score=0.75,
                        reason=(
                            f"Model used tools earlier then provided a "
                            f"{len(response_text)}-char response — treating as complete"
                        ),
                    )

                if not had_prior_tools and response_substantial:
                    # Model made ZERO tool calls throughout but generated substantial
                    # content each turn.  This is a knowledge-generation task (e.g.
                    # create a checklist / best-practices guide) where tool usage is
                    # not expected.  Firing the spin-stuck penalty is a false positive.
                    return EvaluationResult(
                        decision=EvaluationDecision.COMPLETE,
                        score=0.70,
                        reason=(
                            f"Model generated {len(response_text)}-char knowledge "
                            f"response without tools — knowledge generation task complete"
                        ),
                    )

                return EvaluationResult(
                    decision=EvaluationDecision.FAIL,
                    score=0.2,
                    reason=(
                        f"Agent stuck: {spin_detector.consecutive_no_tool_turns} "
                        "turns without tool calls"
                    ),
                )

        return None

    def _is_intent_only_response(self, response_text: str) -> bool:
        """Return True when the response is pure future-intent narration.

        Phrases like "I'll now read...", "Let me now analyze..." describe
        planned actions rather than completed work.  Treating them as final
        answers causes the loop to exit prematurely without doing any analysis.

        Two checks are applied (kept in sync with
        ``AgenticLoop._is_intent_only_response``):
          1. Leading-sentence prefix check (legacy behavior) so responses that
             start with intent but contain substantive findings still pass.
          2. Meta-deliberation density check across the FULL response. Catches
             the failure mode where the model narrates imminent action
             ("Executing now", "Going now", "Calling now", "Making the call",
             "no more deliberation") without ever invoking a tool. Such
             narration must NOT be treated as a complete answer or the loop
             exits before any tool runs. Only fires when there is no
             substantive payload (no code blocks / result-like content).
        """
        if not response_text:
            return False
        first_line = response_text.strip().split("\n")[0].strip().lower()
        intent_prefixes = (
            "i'll now ",
            "i'll ",
            "i will now ",
            "i will ",
            "let me now ",
            "let me ",
            "now i'll ",
            "now i will ",
            "i'm going to ",
            "i am going to ",
            "i'm now ",
            "i am now ",
            "next, i'll ",
            "next i'll ",
        )
        if any(first_line.startswith(p) for p in intent_prefixes):
            return True

        # Meta-deliberation narration density check (full response).
        # Real findings usually carry a payload (a fenced code block or a
        # tool-result-style table). Narration-only responses do not, so we
        # gate the density signal on the absence of such payloads.
        if "```" in response_text:
            return False
        lowered = response_text.lower()
        if lowered.count("|") >= 3 and "---" in lowered:
            return False  # Markdown table — looks like a result dump, not narration

        deliberation_markers = (
            "executing now",
            "executing.",
            "going now",
            "going.",
            "calling now",
            "calling.",
            "running now",
            "running.",
            "making the call",
            "making the request",
            "let me make the call",
            "no more deliberation",
            "stop the meta-deliberation",
            "stop deliberating",
            "done deliberating",
            "just execute",
            "executing the",
            "polling",
            "no sleep",
            "pure status read",
            "going. (",
            "done. (",
            "final. (",
            "(no sleep)",
            "(no more deliberation)",
            "(will act on results",
            "(finally.)",
            "(stop. calling.)",
        )
        marker_hits = sum(1 for m in deliberation_markers if m in lowered)
        # 3+ distinct imminent-action markers without a payload is strong
        # evidence of meta-deliberation narration, not a real answer.
        return marker_hits >= 3

    def _check_qa_shortcut(self, action_result: Any) -> Optional[Any]:
        """Check for Q&A shortcut (model answered without tools).

        Returns EvaluationResult if Q&A shortcut applies, None otherwise.
        """
        from victor.framework.evaluation_nodes import EvaluationResult
        from victor.agent.services.turn_execution_runtime import TurnResult

        if not isinstance(action_result, TurnResult):
            return None

        turn: TurnResult = action_result

        # Q&A shortcut: model answered without tools on a question task
        if (
            hasattr(turn, "is_qa_response")
            and turn.is_qa_response
            and hasattr(turn, "has_content")
            and turn.has_content
        ):
            return EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=0.9,
                reason="Q&A shortcut: accepted direct answer",
            )

        return None

    def _evaluate_legacy(
        self,
        perception: Optional[Perception],
        action_result: Any,
        state: Dict[str, Any],
    ) -> Any:
        """Legacy evaluation fallback.

        This preserves the original AgenticLoop._evaluate() behavior
        when enhanced components are not available.
        """
        from victor.framework.evaluation_nodes import EvaluationResult
        from victor.agent.services.turn_execution_runtime import TurnResult

        # Legacy: Tool execution tracking
        if isinstance(action_result, TurnResult):
            turn: TurnResult = action_result

            # Model used tools = normal progress
            if (
                hasattr(turn, "has_tool_calls")
                and turn.has_tool_calls
                and hasattr(turn, "successful_tool_count")
            ):
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=0.5 + min(turn.successful_tool_count * 0.1, 0.3),
                    reason=f"Tools executed: {turn.successful_tool_count} ok, "
                    f"{turn.failed_tool_count} failed",
                )

            # Detect final answers — but skip pure intent narration ("I'll now read…")
            response_text = turn.content or ""
            if (
                not turn.has_tool_calls
                and turn.has_content
                and len(response_text.strip()) > 100
                and not self._is_intent_only_response(response_text)
            ):
                return EvaluationResult(
                    decision=EvaluationDecision.COMPLETE,
                    score=0.8,
                    reason=f"Model provided substantial response ({len(response_text)} chars)",
                )

            # No tools, no content = needs retry
            if not turn.has_tool_calls:
                return EvaluationResult(
                    decision=EvaluationDecision.CONTINUE,
                    score=0.3,
                    reason="No tools used yet — giving another chance",
                )

        # Legacy: Confidence-based fallback
        if perception is not None and hasattr(perception, "confidence"):
            return self._evaluation_policy.get_confidence_evaluation(perception.confidence)

        # Default: continue with low confidence
        return EvaluationResult(
            decision=EvaluationDecision.CONTINUE,
            score=0.4,
            reason="No clear completion signal - continuing",
        )

    def _map_to_task_type(self, perception: Perception) -> TaskType:
        """Map perception to TaskType for completion detection.

        Uses TaskTypeRegistry.to_completion_task_type() as the SINGLE SOURCE OF TRUTH.
        This ensures consistency across all systems and eliminates duplicate mappings.
        """
        from victor.framework.task_types import TaskTypeRegistry

        # Check if perception has task_type attribute
        if hasattr(perception, "task_type"):
            try:
                # Handle various types that task_type might be
                task_type_value = perception.task_type
                if isinstance(task_type_value, str):
                    perception_task_type = task_type_value
                elif isinstance(task_type_value, enum.Enum):
                    perception_task_type = str(task_type_value.value)
                else:
                    # For unexpected types (Pydantic models, etc.), try to extract string
                    perception_task_type = str(task_type_value)

                # Use the canonical registry - SINGLE SOURCE OF TRUTH
                registry = TaskTypeRegistry.get_instance()
                result = registry.to_completion_task_type(perception_task_type)

                if result is not None:
                    return result

                # If no match found, log with more detail and return UNKNOWN
                logger.debug(
                    f"No TaskType mapping found for '{perception_task_type}' "
                    f"(original: {task_type_value}, type: {type(task_type_value).__name__})"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to map perception.task_type to TaskType: {e}, "
                    f"task_type value: {repr(perception.task_type)}"
                )

        # Map from ActionIntent
        if hasattr(perception, "intent"):
            from victor.agent.action_authorizer import ActionIntent

            intent_to_type = {
                ActionIntent.WRITE_ALLOWED: TaskType.CODE_GENERATION,
                ActionIntent.AMBIGUOUS: TaskType.CODE_GENERATION,
                ActionIntent.DISPLAY_ONLY: TaskType.SEARCH,
                ActionIntent.READ_ONLY: TaskType.ANALYSIS,
            }

            intent = perception.intent
            if intent in intent_to_type:
                return intent_to_type[intent]

        # Default: unknown
        return TaskType.UNKNOWN

    def _map_to_fulfillment_task_type(self, task_type: TaskType) -> FulfillmentTaskType:
        """Convert completion task types to the shared fulfillment task enum."""
        if isinstance(task_type, FulfillmentTaskType):
            return task_type

        raw_value = getattr(task_type, "value", None)
        if isinstance(raw_value, str):
            for candidate in FulfillmentTaskType:
                if candidate.value == raw_value:
                    return candidate

        return FulfillmentTaskType.UNKNOWN

    def _extract_response(self, action_result: Any) -> Optional[str]:
        """Extract response text from action_result."""
        if hasattr(action_result, "response"):
            response = action_result.response
            if isinstance(response, str):
                return response
            if hasattr(response, "content") and isinstance(response.content, str):
                return response.content
        if hasattr(action_result, "content") and isinstance(action_result.content, str):
            return action_result.content
        if isinstance(action_result, str):
            return action_result
        return None

    def _calibrate_completion(
        self,
        completion_score: CompletionScore,
        perception: Perception,
        action_result: Any,
        state: Dict[str, Any],
        requirement_result: Optional[RequirementResult],
        keyword_result: Optional[CompletionSignal],
    ) -> CompletionCalibration:
        """Adjust raw completion score using support and execution signals."""
        evidence_score, reasons = self._estimate_evidence_support(
            perception=perception,
            action_result=action_result,
            state=state,
        )
        return self._evaluation_policy.calibrate_completion(
            raw_score=completion_score.total_score,
            evidence_score=evidence_score,
            threshold=completion_score.threshold,
            continuation_requested=bool(
                keyword_result is not None and keyword_result.is_continuation_request
            ),
            requirements_satisfied=bool(
                requirement_result is None or requirement_result.is_satisfied
            ),
            reasons=reasons,
        )

    def _estimate_evidence_support(
        self,
        perception: Perception,
        action_result: Any,
        state: Dict[str, Any],
    ) -> tuple[float, List[str]]:
        """Estimate how well the current answer is supported by execution evidence."""
        reasons: List[str] = []
        task_type = self._map_to_task_type(perception)

        if self._state_has_artifacts(state):
            reasons.append("state_artifacts_present")
            return 0.90, reasons

        if getattr(action_result, "has_tool_calls", False):
            successful = max(0, int(getattr(action_result, "successful_tool_count", 0) or 0))
            total = int(getattr(action_result, "tool_calls_count", 0) or 0)
            if total <= 0 and hasattr(action_result, "tool_calls"):
                total = len(action_result.tool_calls or [])
            if total <= 0 and hasattr(action_result, "tool_results"):
                total = len(action_result.tool_results or [])
            total = max(total, successful, 1)
            reasons.append("tool_backed_execution")
            return min(1.0, 0.65 + 0.25 * (successful / total)), reasons

        if getattr(action_result, "is_qa_response", False) and task_type in {
            TaskType.SEARCH,
            TaskType.ANALYSIS,
            TaskType.DOCUMENTATION,
        }:
            reasons.append("qa_shortcut_allowed")
            return 0.75, reasons

        reasons.append("direct_answer_without_execution_evidence")
        if getattr(perception, "requirements", None):
            reasons.append("requirements_present")
        return 0.30, reasons

    def _state_has_artifacts(self, state: Dict[str, Any]) -> bool:
        """Detect durable evidence that work was executed."""
        if state.get("files_modified"):
            return True
        if state.get("tests_passed") or state.get("tests_total"):
            return True
        if state.get("source_count", 0) > 0:
            return True
        sources = state.get("sources")
        if isinstance(sources, list) and len(sources) > 0:
            return True
        return False
