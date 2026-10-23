# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical runtime evaluation-policy object for live agent execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, List, Mapping, Optional

# RuntimeEvaluationFeedback now lives in victor_contracts (the shared definition
# layer). Re-exported here so existing imports from this module keep working and
# so victor.evaluation no longer has to import victor.framework at module scope.
from victor_contracts.runtime_evaluation import RuntimeEvaluationFeedback

__all__ = ["RuntimeEvaluationFeedback"]


@dataclass(frozen=True)
class ClarificationDecision:
    """Typed clarification policy outcome for a perception result."""

    requires_clarification: bool = False
    reason: Optional[str] = None
    prompt: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Return compatibility mapping for existing perception callers."""
        return {
            "needs_clarification": self.requires_clarification,
            "clarification_reason": self.reason,
            "clarification_prompt": self.prompt,
        }


@dataclass
class CompletionCalibration:
    """Post-processing view of raw completion score grounded in execution support."""

    raw_score: float
    calibrated_score: float
    evidence_score: float
    threshold: float
    requires_additional_support: bool
    support_penalty: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize for logging and benchmark traces."""
        return {
            "raw_score": round(self.raw_score, 4),
            "calibrated_score": round(self.calibrated_score, 4),
            "evidence_score": round(self.evidence_score, 4),
            "threshold": round(self.threshold, 4),
            "support_penalty": round(self.support_penalty, 4),
            "requires_additional_support": self.requires_additional_support,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class RuntimeEvaluationPolicy:
    """Shared thresholds and wording for runtime evaluation decisions."""

    clarification_confidence_threshold: float = 0.45
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.5
    completion_threshold: float = 0.8
    enhanced_progress_threshold: float = 0.5
    low_confidence_retry_limit: int = 3
    calibrated_completion_raw_weight: float = 0.75
    calibrated_completion_evidence_weight: float = 0.25
    continuation_request_penalty: float = 0.10
    unsupported_requirement_penalty: float = 0.10
    minimum_supported_evidence_score: float = 0.75

    default_clarification_prompt: str = (
        "Please clarify the target file, component, or bug before I continue."
    )
    fallback_clarification_reason: str = "task details are incomplete"
    empty_request_reason: str = "request is empty"
    empty_request_prompt: str = "What would you like me to do?"
    underspecified_target_reason: str = "target artifact or scope is underspecified"
    underspecified_target_prompt: str = "Which file, component, or bug should I target first?"
    confirmation_reason: str = "task intent requires confirmation"
    confirmation_prompt: str = (
        "Should I modify files directly, or keep this read-only and provide guidance first?"
    )

    high_confidence_reason: str = "High confidence in perception"
    medium_confidence_reason: str = "Medium confidence - continue"
    low_confidence_reason: str = "Low confidence - retry"
    completion_requires_support_reason: str = (
        "Completion score is high but needs stronger execution support"
    )
    completion_success_reason_template: str = (
        "Requirements satisfied: {score:.2f} >= {threshold:.2f}"
    )
    completion_progress_reason_template: str = "Progress: {score:.2f} (threshold: {threshold:.2f})"
    completion_retry_reason_template: str = "Insufficient progress: {score:.2f}"
    retry_exhausted_reason_template: str = (
        "Low confidence retry budget exhausted after {retry_count} retries"
    )

    def __post_init__(self) -> None:
        for name, val in [
            ("calibrated_completion_raw_weight", self.calibrated_completion_raw_weight),
            (
                "calibrated_completion_evidence_weight",
                self.calibrated_completion_evidence_weight,
            ),
        ]:
            if val < 0.0:
                raise ValueError(f"{name} must be >= 0.0, got {val}")
        total = self.calibrated_completion_raw_weight + self.calibrated_completion_evidence_weight
        if total <= 0.0:
            raise ValueError(
                f"calibrated_completion_raw_weight + calibrated_completion_evidence_weight "
                f"must be > 0.0, got {total}"
            )

    @classmethod
    def from_config(
        cls,
        config: Optional[Mapping[str, Any]] = None,
    ) -> "RuntimeEvaluationPolicy":
        """Build a policy object from a flat runtime config mapping."""
        if not config:
            return cls()

        allowed = {field.name for field in fields(cls)}
        overrides = {key: value for key, value in config.items() if key in allowed}
        return cls(**overrides)

    def to_config(self) -> Dict[str, Any]:
        """Return config-compatible representation of the policy."""
        return asdict(self)

    def with_overrides(self, **overrides: Any) -> "RuntimeEvaluationPolicy":
        """Return a cloned policy with non-None overrides applied."""
        allowed = {field.name for field in fields(self)}
        filtered = {
            key: value for key, value in overrides.items() if key in allowed and value is not None
        }
        if not filtered:
            return self
        return replace(self, **filtered)

    def with_feedback(
        self,
        feedback: Optional[RuntimeEvaluationFeedback],
    ) -> "RuntimeEvaluationPolicy":
        """Return a cloned policy with runtime-calibration feedback applied."""
        if feedback is None:
            return self
        return self.with_overrides(
            completion_threshold=feedback.completion_threshold,
            enhanced_progress_threshold=feedback.enhanced_progress_threshold,
            minimum_supported_evidence_score=feedback.minimum_supported_evidence_score,
        )

    def empty_request_decision(self, confidence: float = 0.0) -> ClarificationDecision:
        """Return the canonical clarification response for empty requests."""
        return ClarificationDecision(
            requires_clarification=True,
            reason=self.empty_request_reason,
            prompt=self.empty_request_prompt,
            confidence=confidence,
        )

    def underspecified_target_decision(self, confidence: float) -> ClarificationDecision:
        """Return the canonical clarification response for missing targets."""
        return ClarificationDecision(
            requires_clarification=True,
            reason=self.underspecified_target_reason,
            prompt=self.underspecified_target_prompt,
            confidence=confidence,
        )

    def confirmation_required_decision(self, confidence: float) -> ClarificationDecision:
        """Return the canonical clarification response for confirmation-needed tasks."""
        return ClarificationDecision(
            requires_clarification=True,
            reason=self.confirmation_reason,
            prompt=self.confirmation_prompt,
            confidence=confidence,
        )

    def get_clarification_decision(self, perception: Optional[Any]) -> ClarificationDecision:
        """Normalize clarification policy into one typed runtime decision."""
        if not getattr(perception, "needs_clarification", False):
            return ClarificationDecision(
                requires_clarification=False,
                confidence=float(getattr(perception, "confidence", 0.0) or 0.0),
            )

        reason = (
            getattr(perception, "clarification_reason", None) or self.fallback_clarification_reason
        )
        prompt = (
            getattr(perception, "clarification_prompt", None) or self.default_clarification_prompt
        )
        confidence = float(getattr(perception, "confidence", 0.0) or 0.0)
        return ClarificationDecision(
            requires_clarification=True,
            reason=reason,
            prompt=prompt,
            confidence=confidence,
        )

    def get_confidence_evaluation(self, confidence: float) -> Any:
        """Emit the canonical confidence-band evaluation without mutating retry state."""
        from victor.framework.evaluation_nodes import (
            EvaluationDecision,
            EvaluationResult,
        )

        if confidence >= self.high_confidence_threshold:
            return EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=confidence,
                reason=self.high_confidence_reason,
            )

        if confidence >= self.medium_confidence_threshold:
            return EvaluationResult(
                decision=EvaluationDecision.CONTINUE,
                score=confidence,
                reason=self.medium_confidence_reason,
            )

        return EvaluationResult(
            decision=EvaluationDecision.RETRY,
            score=confidence,
            reason=self.low_confidence_reason,
        )

    def calibrate_completion(
        self,
        *,
        raw_score: float,
        evidence_score: float,
        threshold: Optional[float] = None,
        continuation_requested: bool = False,
        requirements_satisfied: bool = True,
        reasons: Optional[List[str]] = None,
    ) -> CompletionCalibration:
        """Adjust a completion score using shared runtime support heuristics."""
        support_penalty = 0.0
        resolved_reasons = list(reasons or [])

        if continuation_requested:
            support_penalty += self.continuation_request_penalty
            resolved_reasons.append("continuation_requested")

        if not requirements_satisfied and evidence_score < self.minimum_supported_evidence_score:
            support_penalty += self.unsupported_requirement_penalty
            resolved_reasons.append("requirements_not_fully_satisfied")

        raw_weight = max(self.calibrated_completion_raw_weight, 0.0)
        evidence_weight = max(self.calibrated_completion_evidence_weight, 0.0)
        total_weight = raw_weight + evidence_weight

        weighted_score = (
            (raw_score * raw_weight) + (evidence_score * evidence_weight)
        ) / total_weight
        resolved_threshold = self.completion_threshold if threshold is None else threshold
        calibrated_score = max(0.0, min(1.0, weighted_score - support_penalty))
        requires_additional_support = (
            raw_score >= resolved_threshold and calibrated_score < resolved_threshold
        )

        return CompletionCalibration(
            raw_score=raw_score,
            calibrated_score=calibrated_score,
            evidence_score=evidence_score,
            threshold=resolved_threshold,
            requires_additional_support=requires_additional_support,
            support_penalty=support_penalty,
            reasons=resolved_reasons,
        )

    def build_completion_evaluation(
        self,
        *,
        score: float,
        threshold: float,
        metadata: Optional[Dict[str, Any]] = None,
        requires_additional_support: bool = False,
    ) -> Any:
        """Create the canonical enhanced-completion evaluation result."""
        from victor.framework.evaluation_nodes import (
            EvaluationDecision,
            EvaluationResult,
        )

        evaluation_metadata = dict(metadata or {})
        if requires_additional_support:
            return EvaluationResult(
                decision=EvaluationDecision.CONTINUE,
                score=score,
                reason=self.completion_requires_support_reason,
                metadata=evaluation_metadata,
            )

        if score >= threshold:
            return EvaluationResult(
                decision=EvaluationDecision.COMPLETE,
                score=score,
                reason=self.completion_success_reason_template.format(
                    score=score,
                    threshold=threshold,
                ),
                metadata=evaluation_metadata,
            )

        if score >= self.enhanced_progress_threshold:
            return EvaluationResult(
                decision=EvaluationDecision.CONTINUE,
                score=score,
                reason=self.completion_progress_reason_template.format(
                    score=score,
                    threshold=threshold,
                ),
                metadata=evaluation_metadata,
            )

        return EvaluationResult(
            decision=EvaluationDecision.RETRY,
            score=score,
            reason=self.completion_retry_reason_template.format(
                score=score,
                threshold=threshold,
            ),
            metadata=evaluation_metadata,
        )

    def apply_retry_budget(
        self,
        evaluation: Any,
        state: Dict[str, Any],
        *,
        retry_limit: Optional[int] = None,
        low_confidence_threshold: Optional[float] = None,
    ) -> Any:
        """Apply the canonical retry-budget policy to low-confidence retry results."""
        from victor.framework.evaluation_nodes import (
            EvaluationDecision,
            EvaluationResult,
        )

        if getattr(evaluation, "should_complete", False) or getattr(
            evaluation, "should_continue", False
        ):
            if "low_confidence_retries" in state:
                state["low_confidence_retries"] = 0
            return evaluation

        threshold = (
            self.medium_confidence_threshold
            if low_confidence_threshold is None
            else low_confidence_threshold
        )
        if not getattr(evaluation, "should_retry", False) or evaluation.score >= threshold:
            return evaluation

        effective_retry_limit = (
            self.low_confidence_retry_limit if retry_limit is None else retry_limit
        )
        effective_retry_limit = max(int(effective_retry_limit), 0)
        retry_count = int(state.get("low_confidence_retries", 0))
        if retry_count >= effective_retry_limit:
            return EvaluationResult(
                decision=EvaluationDecision.FAIL,
                score=evaluation.score,
                reason=self.retry_exhausted_reason_template.format(retry_count=retry_count),
                metrics=dict(evaluation.metrics),
                metadata={
                    **dict(evaluation.metadata),
                    "low_confidence_retry_exhausted": True,
                    "low_confidence_retries": retry_count,
                    "low_confidence_retry_limit": effective_retry_limit,
                },
            )

        retry_count += 1
        state["low_confidence_retries"] = retry_count
        return EvaluationResult(
            decision=EvaluationDecision.RETRY,
            score=evaluation.score,
            reason=evaluation.reason,
            metrics=dict(evaluation.metrics),
            metadata={
                **dict(evaluation.metadata),
                "low_confidence_retries": retry_count,
                "low_confidence_retry_limit": effective_retry_limit,
            },
        )

    def evaluate_confidence_progress(
        self,
        confidence: float,
        state: Dict[str, Any],
        *,
        retry_limit: Optional[int] = None,
    ) -> Any:
        """Apply the canonical confidence-band policy for live-loop evaluation."""
        evaluation = self.get_confidence_evaluation(confidence)

        if getattr(evaluation, "should_complete", False) or getattr(
            evaluation, "should_continue", False
        ):
            if "low_confidence_retries" in state:
                state["low_confidence_retries"] = 0
            return evaluation

        return self.apply_retry_budget(
            evaluation,
            state,
            retry_limit=retry_limit,
            low_confidence_threshold=self.medium_confidence_threshold,
        )
