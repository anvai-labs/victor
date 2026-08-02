"""Tests for LLM decision schemas."""

import pytest
from pydantic import ValidationError

from victor.agent.decisions.schemas import (
    ContinuationDecision,
    DecisionType,
    ErrorClassDecision,
    IntentDecision,
    LoopDetection,
    QuestionTypeDecision,
    TaskCompletionDecision,
    TaskTypeDecision,
)


class TestDecisionType:
    """Tests for DecisionType enum."""

    def test_all_types_exist(self):
        assert DecisionType.TASK_COMPLETION.value == "task_completion"
        assert DecisionType.INTENT_CLASSIFICATION.value == "intent_classification"
        assert DecisionType.TASK_TYPE_CLASSIFICATION.value == "task_type_classification"
        assert DecisionType.QUESTION_CLASSIFICATION.value == "question_classification"
        assert DecisionType.LOOP_DETECTION.value == "loop_detection"
        assert DecisionType.ERROR_CLASSIFICATION.value == "error_classification"
        assert DecisionType.CONTINUATION_ACTION.value == "continuation_action"

    def test_decision_type_count(self):
        assert len(DecisionType) == 15  # + TURN_AUDIT (EVR-6 per-turn judge)


class TestTaskCompletionDecision:
    """Tests for TaskCompletionDecision schema."""

    def test_valid(self):
        d = TaskCompletionDecision(is_complete=True, confidence=0.9, phase="done")
        assert d.is_complete is True
        assert d.confidence == 0.9
        assert d.phase == "done"

    def test_all_phases(self):
        for phase in ("working", "finalizing", "done", "stuck"):
            d = TaskCompletionDecision(is_complete=False, confidence=0.5, phase=phase)
            assert d.phase == phase

    def test_invalid_phase(self):
        with pytest.raises(ValidationError):
            TaskCompletionDecision(is_complete=True, confidence=0.5, phase="invalid")

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            TaskCompletionDecision(is_complete=True, confidence=1.5, phase="done")
        with pytest.raises(ValidationError):
            TaskCompletionDecision(is_complete=True, confidence=-0.1, phase="done")


class TestIntentDecision:
    """Tests for IntentDecision schema."""

    def test_valid_intents(self):
        for intent in ("continuation", "completion", "asking_input", "stuck_loop"):
            d = IntentDecision(intent=intent, confidence=0.8)
            assert d.intent == intent

    def test_invalid_intent(self):
        with pytest.raises(ValidationError):
            IntentDecision(intent="unknown", confidence=0.5)


class TestTaskTypeDecision:
    """Tests for TaskTypeDecision schema."""

    def test_valid_types(self):
        for task_type in ("analysis", "action", "generation", "search", "edit"):
            d = TaskTypeDecision(task_type=task_type, confidence=0.7)
            assert d.task_type == task_type


class TestQuestionTypeDecision:
    """Tests for QuestionTypeDecision schema."""

    def test_valid_types(self):
        for q_type in ("rhetorical", "continuation", "clarification", "info"):
            d = QuestionTypeDecision(question_type=q_type, confidence=0.6)
            assert d.question_type == q_type


class TestLoopDetection:
    """Tests for LoopDetection schema."""

    def test_loop_detected(self):
        d = LoopDetection(is_loop=True, loop_type="stalling")
        assert d.is_loop is True
        assert d.loop_type == "stalling"

    def test_no_loop(self):
        d = LoopDetection(is_loop=False, loop_type="none")
        assert d.is_loop is False

    def test_all_loop_types(self):
        for loop_type in ("stalling", "circular", "repetition", "none"):
            d = LoopDetection(is_loop=True, loop_type=loop_type)
            assert d.loop_type == loop_type


class TestErrorClassDecision:
    """Tests for ErrorClassDecision schema."""

    def test_valid_types(self):
        for error_type in ("permanent", "transient", "retryable"):
            d = ErrorClassDecision(error_type=error_type, confidence=0.9)
            assert d.error_type == error_type


class TestContinuationDecision:
    """Tests for ContinuationDecision schema."""

    def test_valid_actions(self):
        for action in (
            "continue_asking_input",
            "continue_with_synthesis_hint",
            "execute_extracted_tool",
            "finish",
            "force_tool_execution",
            "prompt_tool_call",
            "request_completion",
            "request_summary",
            "return_to_user",
        ):
            d = ContinuationDecision(action=action, reason="test reason")
            assert d.action == action

    def test_reason_max_length(self):
        d = ContinuationDecision(action="finish", reason="x" * 100)
        assert len(d.reason) == 100

    def test_reason_too_long(self):
        with pytest.raises(ValidationError):
            ContinuationDecision(action="finish", reason="x" * 101)

    def test_json_roundtrip(self):
        d = ContinuationDecision(action="finish", reason="task done")
        data = d.model_dump()
        d2 = ContinuationDecision.model_validate(data)
        assert d == d2
