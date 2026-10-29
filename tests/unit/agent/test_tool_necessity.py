"""Tests for meta-cognitive tool necessity decision (HDPO-inspired)."""

import pytest
from pydantic import ValidationError


class TestToolNecessitySchema:
    def test_schema_exists(self):
        from victor.agent.decisions.schemas import ToolNecessityDecision

        d = ToolNecessityDecision(requires_tools=True, confidence=0.9)
        assert d.requires_tools is True
        assert d.confidence == 0.9

    def test_confidence_bounded(self):
        from victor.agent.decisions.schemas import ToolNecessityDecision

        with pytest.raises(ValidationError):
            ToolNecessityDecision(requires_tools=True, confidence=1.5)

    def test_decision_type_exists(self):
        from victor.agent.decisions.schemas import DecisionType

        assert hasattr(DecisionType, "TOOL_NECESSITY")


class TestToolNecessityPrompt:
    def test_prompt_registered(self):
        from victor.agent.decisions.prompts import DECISION_PROMPTS
        from victor.agent.decisions.schemas import DecisionType

        assert DecisionType.TOOL_NECESSITY in DECISION_PROMPTS

    def test_prompt_has_required_fields(self):
        from victor.agent.decisions.prompts import DECISION_PROMPTS
        from victor.agent.decisions.schemas import DecisionType

        prompt = DECISION_PROMPTS[DecisionType.TOOL_NECESSITY]
        assert prompt.system
        assert prompt.user_template
        assert "{message_excerpt}" in prompt.user_template
        assert prompt.max_tokens <= 40


class TestIsQuestionOnlyWithConfidence:
    def test_pure_qa_high_confidence(self):
        from victor.agent.services.turn_execution_runtime import TurnExecutor

        is_qa, conf = TurnExecutor._is_question_only_scored("What is Python?")
        assert is_qa is True
        assert conf >= 0.8

    def test_action_task_high_confidence(self):
        from victor.agent.services.turn_execution_runtime import TurnExecutor

        is_qa, conf = TurnExecutor._is_question_only_scored("Fix the bug in main.py")
        assert is_qa is False
        assert conf >= 0.8

    def test_ambiguous_lower_confidence(self):
        from victor.agent.services.turn_execution_runtime import TurnExecutor

        is_qa, conf = TurnExecutor._is_question_only_scored("Show me how to fix the auth module")
        # "Show me" prefix triggers QA but "fix" is an action word — ambiguous
        assert conf < 0.8

    def test_short_question_high_confidence(self):
        from victor.agent.services.turn_execution_runtime import TurnExecutor

        is_qa, conf = TurnExecutor._is_question_only_scored("What is 2+2?")
        assert is_qa is True
        assert conf >= 0.9

    def test_long_action_request(self):
        from victor.agent.services.turn_execution_runtime import TurnExecutor

        is_qa, conf = TurnExecutor._is_question_only_scored(
            "Create a new REST API endpoint for user authentication with JWT tokens"
        )
        assert is_qa is False

    def test_backward_compatible_bool(self):
        """Original _is_question_only still works as bool."""
        from victor.agent.services.turn_execution_runtime import TurnExecutor

        assert TurnExecutor._is_question_only("What is Python?") is True
        assert TurnExecutor._is_question_only("Fix the bug") is False
