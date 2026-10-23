"""RuntimeEvaluationPolicy at its core home: every decision branch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from victor_contracts.runtime_evaluation import RuntimeEvaluationFeedback

from victor.core.runtime_evaluation_policy import (
    ClarificationDecision,
    CompletionCalibration,
    RuntimeEvaluationPolicy,
)
from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult


@pytest.fixture
def policy():
    return RuntimeEvaluationPolicy()


class TestValueObjects:
    def test_clarification_mapping_uses_legacy_keys(self):
        decision = ClarificationDecision(True, "why", "what?", 0.4)
        assert decision.to_dict() == {
            "needs_clarification": True,
            "clarification_reason": "why",
            "clarification_prompt": "what?",
        }

    def test_calibration_metadata_is_rounded(self):
        cal = CompletionCalibration(0.123456, 0.2, 0.3, 0.8, True, 0.1, ["r"])
        meta = cal.to_metadata()
        assert meta["raw_score"] == 0.1235 and meta["reasons"] == ["r"]
        assert meta["requires_additional_support"] is True


class TestConstruction:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"calibrated_completion_raw_weight": -0.1},
            {"calibrated_completion_evidence_weight": -1.0},
            {"calibrated_completion_raw_weight": 0.0, "calibrated_completion_evidence_weight": 0.0},
        ],
    )
    def test_invalid_weights_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            RuntimeEvaluationPolicy(**kwargs)

    def test_empty_config_and_unknown_keys(self, policy):
        assert RuntimeEvaluationPolicy.from_config(None) == policy
        assert RuntimeEvaluationPolicy.from_config({"nope": 1}) == policy
        assert policy.to_config()["completion_threshold"] == 0.8

    def test_overrides_skip_none_and_unknown(self, policy):
        assert policy.with_overrides(completion_threshold=None, bogus=1) is policy
        assert policy.with_overrides(completion_threshold=0.9).completion_threshold == 0.9

    def test_feedback_applies_thresholds(self, policy):
        assert policy.with_feedback(None) is policy
        tuned = policy.with_feedback(
            RuntimeEvaluationFeedback(
                completion_threshold=0.7,
                enhanced_progress_threshold=0.4,
                minimum_supported_evidence_score=0.6,
            )
        )
        assert (
            tuned.completion_threshold,
            tuned.enhanced_progress_threshold,
            tuned.minimum_supported_evidence_score,
        ) == (0.7, 0.4, 0.6)


class TestClarification:
    def test_canned_decisions(self, policy):
        assert policy.empty_request_decision().prompt == policy.empty_request_prompt
        assert policy.underspecified_target_decision(0.3).reason == (
            policy.underspecified_target_reason
        )
        confirm = policy.confirmation_required_decision(0.5)
        assert confirm.requires_clarification and confirm.confidence == 0.5

    def test_no_clarification_needed(self, policy):
        decision = policy.get_clarification_decision(SimpleNamespace(confidence=0.7))
        assert decision == ClarificationDecision(False, confidence=0.7)
        assert policy.get_clarification_decision(None).confidence == 0.0

    def test_missing_reason_and_prompt_fall_back(self, policy):
        perception = SimpleNamespace(needs_clarification=True, confidence=None)
        decision = policy.get_clarification_decision(perception)
        assert decision.reason == policy.fallback_clarification_reason
        assert decision.prompt == policy.default_clarification_prompt
        assert decision.confidence == 0.0


class TestConfidenceBands:
    @pytest.mark.parametrize(
        "confidence,expected",
        [
            (0.9, EvaluationDecision.COMPLETE),
            (0.6, EvaluationDecision.CONTINUE),
            (0.2, EvaluationDecision.RETRY),
        ],
    )
    def test_bands(self, policy, confidence, expected):
        assert policy.get_confidence_evaluation(confidence).decision is expected

    def test_progress_resets_retries_on_success(self, policy):
        state = {"low_confidence_retries": 2}
        assert policy.evaluate_confidence_progress(0.9, state).should_complete
        assert state["low_confidence_retries"] == 0

    def test_progress_counts_then_exhausts_retries(self, policy):
        state: dict = {}
        for expected in (1, 2):
            result = policy.evaluate_confidence_progress(0.1, state, retry_limit=2)
            assert result.should_retry and state["low_confidence_retries"] == expected
        exhausted = policy.evaluate_confidence_progress(0.1, state, retry_limit=2)
        assert exhausted.decision is EvaluationDecision.FAIL
        assert exhausted.metadata["low_confidence_retry_exhausted"] is True


class TestRetryBudget:
    def test_passes_through_non_retry_and_confident_retry(self, policy):
        complete = EvaluationResult(EvaluationDecision.COMPLETE, 0.9, "ok")
        state = {"low_confidence_retries": 3}
        assert policy.apply_retry_budget(complete, state) is complete
        assert state["low_confidence_retries"] == 0
        failed = EvaluationResult(EvaluationDecision.FAIL, 0.1, "x")
        assert policy.apply_retry_budget(failed, {}) is failed
        confident = EvaluationResult(EvaluationDecision.RETRY, 0.6, "x")
        assert policy.apply_retry_budget(confident, {}) is confident

    def test_negative_limit_exhausts_immediately(self, policy):
        low = EvaluationResult(EvaluationDecision.RETRY, 0.1, "x", metrics={"m": 1})
        result = policy.apply_retry_budget(low, {}, retry_limit=-5)
        assert result.decision is EvaluationDecision.FAIL
        assert result.metrics == {"m": 1}
        assert result.metadata["low_confidence_retry_limit"] == 0


class TestCompletion:
    def test_penalties_and_support_flag(self, policy):
        cal = policy.calibrate_completion(
            raw_score=0.9,
            evidence_score=0.2,
            continuation_requested=True,
            requirements_satisfied=False,
            reasons=["seed"],
        )
        assert cal.support_penalty == pytest.approx(0.2)
        assert cal.reasons == ["seed", "continuation_requested", "requirements_not_fully_satisfied"]
        assert cal.requires_additional_support is True

    def test_explicit_threshold_and_clamping(self, policy):
        cal = policy.calibrate_completion(raw_score=0.0, evidence_score=0.0, threshold=0.5)
        assert cal.threshold == 0.5 and cal.calibrated_score == 0.0
        assert cal.requires_additional_support is False

    @pytest.mark.parametrize(
        "score,support,expected",
        [
            (0.95, True, EvaluationDecision.CONTINUE),
            (0.95, False, EvaluationDecision.COMPLETE),
            (0.6, False, EvaluationDecision.CONTINUE),
            (0.1, False, EvaluationDecision.RETRY),
        ],
    )
    def test_build_completion_evaluation(self, policy, score, support, expected):
        result = policy.build_completion_evaluation(
            score=score,
            threshold=0.8,
            metadata={"k": "v"},
            requires_additional_support=support,
        )
        assert result.decision is expected and result.metadata == {"k": "v"}
