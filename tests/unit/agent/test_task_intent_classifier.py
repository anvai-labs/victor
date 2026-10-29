# Copyright 2025 Vijaykumar Singh
# Licensed under the Apache License, Version 2.0

"""Tests for LLM-based task intent classifier.

TDD tests verifying that TaskTypeDecision (with deliverables) correctly
drives the completion detector's expected deliverables, replacing the
regex-only analyze_intent() with LLM classification + regex fallback.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock


from victor.agent.decisions.schemas import DecisionType, TaskTypeDecision
from victor.agent.task_completion import DeliverableType, TaskCompletionDetector

# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestTaskTypeDecisionSchema:
    """Test the enhanced TaskTypeDecision schema with deliverables."""

    def test_action_with_file_modified(self):
        d = TaskTypeDecision(
            task_type="action",
            confidence=0.9,
            deliverables=["file_modified"],
        )
        assert d.task_type == "action"
        assert d.deliverables == ["file_modified"]

    def test_analysis_with_analysis_provided(self):
        d = TaskTypeDecision(
            task_type="analysis",
            confidence=0.85,
            deliverables=["analysis_provided"],
        )
        assert d.deliverables == ["analysis_provided"]

    def test_generation_with_multiple_deliverables(self):
        d = TaskTypeDecision(
            task_type="generation",
            confidence=0.75,
            deliverables=["file_created", "code_executed"],
        )
        assert len(d.deliverables) == 2

    def test_empty_deliverables_allowed(self):
        """Empty deliverables = infer from task_type (backwards compat)."""
        d = TaskTypeDecision(task_type="edit", confidence=0.8)
        assert d.deliverables == []

    def test_all_task_types(self):
        for tt in ["analysis", "action", "generation", "search", "edit"]:
            d = TaskTypeDecision(task_type=tt, confidence=0.8)
            assert d.task_type == tt

    def test_all_deliverable_types(self):
        for dt in [
            "file_created",
            "file_modified",
            "analysis_provided",
            "answer_provided",
            "plan_provided",
            "code_executed",
        ]:
            d = TaskTypeDecision(task_type="action", confidence=0.8, deliverables=[dt])
            assert dt in d.deliverables


# ---------------------------------------------------------------------------
# Regex fallback tests (no LLM service)
# ---------------------------------------------------------------------------


class TestRegexIntentFallback:
    """Test regex-based fallback classification in analyze_intent."""

    def _classify(self, message: str) -> List[DeliverableType]:
        detector = TaskCompletionDetector()
        return detector.analyze_intent(message)

    def test_fix_intent(self):
        assert DeliverableType.FILE_MODIFIED in self._classify("Fix the bug in auth.py")

    def test_create_intent(self):
        assert DeliverableType.FILE_CREATED in self._classify("Create a new cache_manager.py")

    def test_analyze_intent(self):
        assert DeliverableType.ANALYSIS_PROVIDED in self._classify("Analyze the architecture")

    def test_explain_intent(self):
        assert DeliverableType.ANSWER_PROVIDED in self._classify("Explain how providers work")

    def test_test_intent(self):
        assert DeliverableType.CODE_EXECUTED in self._classify("Run the unit tests")

    def test_empty_message(self):
        assert self._classify("") == []

    def test_swe_bench_issue_text(self):
        """Complex issue text — regex may miss it, LLM classifier handles it."""
        result = self._classify(
            "Modeling's `separability_matrix` does not compute separability "
            "correctly for nested CompoundModels."
        )
        # Document: regex returns empty for complex issue text
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# LLM classifier integration tests (mocked decision service)
# ---------------------------------------------------------------------------


def _mock_decision_service(task_type_result: TaskTypeDecision):
    """Create a mock decision service returning the given TaskTypeDecision."""
    from victor.agent.services.protocols.decision_service import DecisionResult

    service = MagicMock()
    service.decide_sync.return_value = DecisionResult(
        decision_type=DecisionType.TASK_TYPE_CLASSIFICATION,
        result=task_type_result,
        source="llm",
        confidence=task_type_result.confidence,
        latency_ms=50.0,
        tokens_used=30,
    )
    return service


class TestLLMIntentClassification:
    """Test LLM-based classification via the decision service."""

    def test_action_sets_file_modified(self):
        service = _mock_decision_service(
            TaskTypeDecision(task_type="action", confidence=0.9, deliverables=["file_modified"])
        )
        detector = TaskCompletionDetector(decision_service=service)
        result = detector.analyze_intent("Fix the separability_matrix bug")
        assert DeliverableType.FILE_MODIFIED in result

    def test_analysis_sets_analysis_provided(self):
        service = _mock_decision_service(
            TaskTypeDecision(
                task_type="analysis",
                confidence=0.85,
                deliverables=["analysis_provided"],
            )
        )
        detector = TaskCompletionDetector(decision_service=service)
        result = detector.analyze_intent("Review the auth module")
        assert DeliverableType.ANALYSIS_PROVIDED in result

    def test_generation_sets_file_created(self):
        service = _mock_decision_service(
            TaskTypeDecision(task_type="generation", confidence=0.8, deliverables=["file_created"])
        )
        detector = TaskCompletionDetector(decision_service=service)
        result = detector.analyze_intent("Create a new cache manager")
        assert DeliverableType.FILE_CREATED in result

    def test_search_sets_answer_provided(self):
        service = _mock_decision_service(
            TaskTypeDecision(task_type="search", confidence=0.85, deliverables=["answer_provided"])
        )
        detector = TaskCompletionDetector(decision_service=service)
        result = detector.analyze_intent("Find where auth tokens are stored")
        assert DeliverableType.ANSWER_PROVIDED in result

    def test_edit_sets_file_modified(self):
        service = _mock_decision_service(
            TaskTypeDecision(task_type="edit", confidence=0.9, deliverables=["file_modified"])
        )
        detector = TaskCompletionDetector(decision_service=service)
        result = detector.analyze_intent("Refactor the provider layer")
        assert DeliverableType.FILE_MODIFIED in result

    def test_multiple_deliverables(self):
        service = _mock_decision_service(
            TaskTypeDecision(
                task_type="generation",
                confidence=0.8,
                deliverables=["file_created", "code_executed"],
            )
        )
        detector = TaskCompletionDetector(decision_service=service)
        result = detector.analyze_intent("Create and test a new API endpoint")
        assert DeliverableType.FILE_CREATED in result
        assert DeliverableType.CODE_EXECUTED in result

    def test_empty_deliverables_infers_from_task_type(self):
        """When LLM returns no deliverables, infer from task_type."""
        service = _mock_decision_service(
            TaskTypeDecision(task_type="action", confidence=0.9, deliverables=[])
        )
        detector = TaskCompletionDetector(decision_service=service)
        result = detector.analyze_intent("Fix the login bug")
        # Should infer FILE_MODIFIED from task_type="action"
        assert DeliverableType.FILE_MODIFIED in result

    def test_low_confidence_falls_back_to_regex(self):
        """Low confidence LLM result should fall through to regex."""
        from victor.agent.services.protocols.decision_service import DecisionResult

        service = MagicMock()
        service.decide_sync.return_value = DecisionResult(
            decision_type=DecisionType.TASK_TYPE_CLASSIFICATION,
            result=TaskTypeDecision(
                task_type="analysis", confidence=0.3, deliverables=["analysis_provided"]
            ),
            source="llm",
            confidence=0.3,
            latency_ms=50.0,
            tokens_used=30,
        )
        detector = TaskCompletionDetector(decision_service=service)
        result = detector.analyze_intent("fix the bug in auth.py")
        # Regex fallback should detect "fix" → FILE_MODIFIED
        assert DeliverableType.FILE_MODIFIED in result

    def test_decision_service_error_falls_back(self):
        """Decision service error → regex fallback."""
        service = MagicMock()
        service.decide_sync.side_effect = Exception("LLM unavailable")
        detector = TaskCompletionDetector(decision_service=service)
        result = detector.analyze_intent("fix the login bug")
        assert DeliverableType.FILE_MODIFIED in result

    def test_no_service_uses_regex(self):
        """No decision service → pure regex."""
        detector = TaskCompletionDetector(decision_service=None)
        result = detector.analyze_intent("update the README")
        assert DeliverableType.FILE_MODIFIED in result

    def test_swe_bench_issue_classified_correctly(self):
        """Complex SWE-bench issue → LLM classifies as action with file_modified."""
        service = _mock_decision_service(
            TaskTypeDecision(task_type="action", confidence=0.92, deliverables=["file_modified"])
        )
        detector = TaskCompletionDetector(decision_service=service)
        issue = (
            "Modeling's `separability_matrix` does not compute separability "
            "correctly for nested CompoundModels. Consider the following model:\n"
            "```python\nfrom astropy.modeling import models as m\n```"
        )
        result = detector.analyze_intent(issue)
        assert DeliverableType.FILE_MODIFIED in result


# ---------------------------------------------------------------------------
# Completion detector integration
# ---------------------------------------------------------------------------


class TestCompletionWithClassification:
    """Test that classification correctly controls should_stop()."""

    def test_analysis_stops_when_analysis_delivered(self):
        detector = TaskCompletionDetector()
        detector._state.expected_deliverables = [DeliverableType.ANALYSIS_PROVIDED]
        detector._state.completed_deliverables.append(
            MagicMock(type=DeliverableType.ANALYSIS_PROVIDED)
        )
        assert detector.should_stop() is True

    def test_fix_does_not_stop_without_file_edit(self):
        detector = TaskCompletionDetector()
        detector._state.expected_deliverables = [DeliverableType.FILE_MODIFIED]
        detector._state.completion_signals.add("passive:in summary")
        assert detector.should_stop() is False

    def test_fix_stops_after_file_edit(self):
        detector = TaskCompletionDetector()
        detector._state.expected_deliverables = [DeliverableType.FILE_MODIFIED]
        detector._state.completed_deliverables.append(MagicMock(type=DeliverableType.FILE_MODIFIED))
        assert detector.should_stop() is True

    def test_explain_stops_on_answer(self):
        detector = TaskCompletionDetector()
        detector._state.expected_deliverables = [DeliverableType.ANSWER_PROVIDED]
        detector._state.completed_deliverables.append(
            MagicMock(type=DeliverableType.ANSWER_PROVIDED)
        )
        assert detector.should_stop() is True

    def test_empty_deliverables_stops_on_active_signal(self):
        detector = TaskCompletionDetector()
        detector._state.expected_deliverables = []
        detector._state.active_signal_detected = True
        detector._state.completion_signals.add("active:done:")
        assert detector.should_stop() is True

    def test_task_type_to_deliverable_inference(self):
        """When LLM returns task_type but no deliverables, inference should work."""
        # Map: action/edit → FILE_MODIFIED, generation → FILE_CREATED,
        #       analysis/search → ANALYSIS_PROVIDED
        from victor.agent.task_completion import _infer_deliverables_from_task_type

        assert DeliverableType.FILE_MODIFIED in _infer_deliverables_from_task_type("action")
        assert DeliverableType.FILE_MODIFIED in _infer_deliverables_from_task_type("edit")
        assert DeliverableType.FILE_CREATED in _infer_deliverables_from_task_type("generation")
        assert DeliverableType.ANALYSIS_PROVIDED in _infer_deliverables_from_task_type("analysis")
        assert DeliverableType.ANSWER_PROVIDED in _infer_deliverables_from_task_type("search")
