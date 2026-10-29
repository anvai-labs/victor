"""Tests for TaskCompletionDetector with LLM decision service augmentation."""

from unittest.mock import MagicMock


from victor.agent.decisions.schemas import DecisionType, TaskCompletionDecision
from victor.agent.services.protocols.decision_service import DecisionResult
from victor.agent.task_completion import (
    CompletionConfidence,
    TaskCompletionDetector,
)
from victor.core.completion_markers import FILE_DONE_MARKER, TASK_DONE_MARKER


def _make_decision_service(
    is_complete: bool = True,
    confidence: float = 0.9,
    phase: str = "done",
    source: str = "llm",
):
    """Create a mock decision service that returns a predetermined result."""
    service = MagicMock()
    result = DecisionResult(
        decision_type=DecisionType.TASK_COMPLETION,
        result=TaskCompletionDecision(is_complete=is_complete, confidence=confidence, phase=phase),
        source=source,
        confidence=confidence,
    )
    service.decide_sync.return_value = result
    return service


class TestTaskCompletionWithoutService:
    """Verify existing behavior is unchanged when no service is provided."""

    def test_no_service_default(self):
        detector = TaskCompletionDetector()
        assert detector._decision_service is None

    def test_active_signal_still_high(self):
        detector = TaskCompletionDetector()
        detector.analyze_response(f"{FILE_DONE_MARKER} Created cache.py")
        assert detector.get_completion_confidence() == CompletionConfidence.HIGH

    def test_plain_done_no_longer_triggers_active_completion(self):
        detector = TaskCompletionDetector()
        detector.analyze_response("DONE: Task complete")

        assert detector.get_completion_confidence() != CompletionConfidence.HIGH
        assert detector._state.active_signal_detected is False

    def test_passive_signal_still_low(self):
        detector = TaskCompletionDetector()
        detector.analyze_response("The file has been created successfully")
        assert detector.get_completion_confidence() == CompletionConfidence.LOW

    def test_no_signal_still_none(self):
        detector = TaskCompletionDetector()
        detector.analyze_response("Let me think about this")
        assert detector.get_completion_confidence() == CompletionConfidence.NONE

    def test_framework_completion_signal_reused_for_structured_code_output(self):
        detector = TaskCompletionDetector()
        detector.analyze_intent("Create a cache_manager.py file")

        detector.analyze_response(
            "Here is the implementation:\n\n```python\nclass CacheManager:\n    pass\n```"
        )

        assert detector._last_framework_signal is not None
        assert detector._last_framework_signal.has_complete_code is True
        assert detector.get_completion_confidence() == CompletionConfidence.LOW

    def test_framework_continuation_signal_detected_for_contextual_phrase(self):
        detector = TaskCompletionDetector()
        detector.analyze_intent("Analyze the architecture")

        detector.analyze_response("Let me know if you'd like me to compare more tradeoffs.")

        assert detector._state.continuation_requests >= 1


class TestTaskCompletionWithLLMService:
    """Test LLM augmentation in task completion detection."""

    def test_runtime_intelligence_delegates_completion_decision(self):
        """RuntimeIntelligenceService is preferred over direct decision service wiring."""
        runtime_intelligence = MagicMock()
        runtime_intelligence.decide_sync.return_value = DecisionResult(
            decision_type=DecisionType.TASK_COMPLETION,
            result=TaskCompletionDecision(is_complete=True, confidence=0.9, phase="done"),
            source="llm",
            confidence=0.9,
        )
        detector = TaskCompletionDetector(runtime_intelligence=runtime_intelligence)

        assert detector.get_completion_confidence() == CompletionConfidence.MEDIUM
        runtime_intelligence.decide_sync.assert_called_once()

    def test_llm_upgrades_none_to_medium(self):
        """When heuristic is NONE but LLM says complete, upgrades to MEDIUM."""
        service = _make_decision_service(is_complete=True, confidence=0.9, phase="done")
        detector = TaskCompletionDetector(decision_service=service)

        # No heuristic signals at all
        assert detector.get_completion_confidence() == CompletionConfidence.MEDIUM
        service.decide_sync.assert_called_once()

    def test_llm_low_confidence_no_upgrade(self):
        """When LLM confidence is below 0.7, no upgrade happens."""
        service = _make_decision_service(is_complete=True, confidence=0.5, phase="done")
        detector = TaskCompletionDetector(decision_service=service)

        assert detector.get_completion_confidence() == CompletionConfidence.NONE

    def test_llm_says_not_complete(self):
        """When LLM says not complete, stays NONE."""
        service = _make_decision_service(is_complete=False, confidence=0.9, phase="working")
        detector = TaskCompletionDetector(decision_service=service)

        assert detector.get_completion_confidence() == CompletionConfidence.NONE

    def test_llm_not_called_when_heuristic_high(self):
        """LLM is never consulted when heuristic confidence is HIGH."""
        service = _make_decision_service()
        detector = TaskCompletionDetector(decision_service=service)
        detector.analyze_response(f"{TASK_DONE_MARKER} Task is complete")

        result = detector.get_completion_confidence()
        assert result == CompletionConfidence.HIGH
        # decide_sync may have been called in analyze_response but should not
        # override the HIGH result from get_completion_confidence

    def test_llm_analyze_response_detects_completion(self):
        """LLM detects completion in analyze_response when heuristics miss it."""
        service = _make_decision_service(is_complete=True, confidence=0.9, phase="done")
        detector = TaskCompletionDetector(decision_service=service)

        # Ambiguous response with no clear signals
        detector.analyze_response("I have completed all the requested changes to the codebase.")

        # LLM should have added a completion signal
        assert "llm:task_complete" in detector._state.completion_signals

    def test_llm_analyze_response_detects_stuck(self):
        """LLM detects stuck phase and increments continuation requests."""
        service = _make_decision_service(is_complete=False, confidence=0.8, phase="stuck")
        detector = TaskCompletionDetector(decision_service=service)

        detector.analyze_response("I'm not sure what to do next. Let me think about this.")
        assert detector._state.continuation_requests >= 1

    def test_llm_failure_graceful(self):
        """Service failure doesn't break detection."""
        service = MagicMock()
        service.decide_sync.side_effect = RuntimeError("LLM unavailable")
        detector = TaskCompletionDetector(decision_service=service)

        # Should not raise
        detector.analyze_response("Some response text")
        assert detector.get_completion_confidence() == CompletionConfidence.NONE

    def test_llm_not_called_when_active_signal_exists(self):
        """LLM is not consulted when active signal already detected."""
        service = _make_decision_service()
        detector = TaskCompletionDetector(decision_service=service)

        detector.analyze_response(f"{TASK_DONE_MARKER} Implementation complete")

        # Active signal was detected, LLM should NOT be called
        # (the analyze_response returns early on active signal)
        service.decide_sync.assert_not_called()

    def test_budget_exhausted_source(self):
        """Budget exhausted returns heuristic fallback."""
        service = MagicMock()
        service.decide_sync.return_value = DecisionResult(
            decision_type=DecisionType.TASK_COMPLETION,
            result=None,
            source="budget_exhausted",
            confidence=0.0,
        )
        detector = TaskCompletionDetector(decision_service=service)
        detector.analyze_response("Some text without signals")

        # budget_exhausted means no LLM result, should be NONE
        assert detector.get_completion_confidence() == CompletionConfidence.NONE


class TestTaskCompletionLocalClassifierGate:
    """FEP-0012: the shipped local classifier may signal completion only when
    ``local_classifier_completion_signal`` is opted in (default off). The head
    predicts reward buckets; a confident 'pass' -> is_complete."""

    def test_local_classifier_rejected_by_default(self):
        """source=local_classifier is ignored for completion when the flag is off."""
        service = _make_decision_service(
            is_complete=True, confidence=0.9, phase="done", source="local_classifier"
        )
        detector = TaskCompletionDetector(decision_service=service)
        # Default setting is off -> local_classifier does NOT upgrade confidence.
        assert detector.get_completion_confidence() == CompletionConfidence.NONE

    def test_local_classifier_accepted_when_enabled(self, monkeypatch):
        class _Enabled:
            local_classifier_completion_signal = True

        monkeypatch.setattr("victor.config.decision_settings.DecisionServiceSettings", _Enabled)
        service = _make_decision_service(
            is_complete=True, confidence=0.9, phase="done", source="local_classifier"
        )
        detector = TaskCompletionDetector(decision_service=service)
        assert detector.get_completion_confidence() == CompletionConfidence.MEDIUM

    def test_local_classifier_low_confidence_no_upgrade(self, monkeypatch):
        class _Enabled:
            local_classifier_completion_signal = True

        monkeypatch.setattr("victor.config.decision_settings.DecisionServiceSettings", _Enabled)
        service = _make_decision_service(
            is_complete=True, confidence=0.5, phase="done", source="local_classifier"
        )
        detector = TaskCompletionDetector(decision_service=service)
        assert detector.get_completion_confidence() == CompletionConfidence.NONE

    def test_local_classifier_says_not_complete(self, monkeypatch):
        class _Enabled:
            local_classifier_completion_signal = True

        monkeypatch.setattr("victor.config.decision_settings.DecisionServiceSettings", _Enabled)
        service = _make_decision_service(
            is_complete=False,
            confidence=0.9,
            phase="working",
            source="local_classifier",
        )
        detector = TaskCompletionDetector(decision_service=service)
        assert detector.get_completion_confidence() == CompletionConfidence.NONE
