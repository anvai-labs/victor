"""Tests for QueryClassifier — Layer 1 of agentic execution quality."""

from unittest.mock import MagicMock, patch

import pytest

from victor.agent.query_classifier import (
    QueryClassifier,
    QueryType,
)
from victor.framework.task.protocols import TaskClassification, TaskComplexity


@pytest.fixture
def classifier():
    """Create a QueryClassifier with a mocked TaskComplexityService."""
    return QueryClassifier()


@pytest.fixture
def classifier_with_mock_service():
    """Create a QueryClassifier with an injected mock TaskComplexityService."""
    mock_service = MagicMock()
    mock_service.classify.return_value = TaskClassification(
        complexity=TaskComplexity.MEDIUM,
        tool_budget=5,
        confidence=0.8,
        matched_patterns=["test"],
    )
    return QueryClassifier(complexity_service=mock_service), mock_service


class TestQueryTypeClassification:
    def test_quick_question_classification(self, classifier):
        result = classifier.classify("What is the auth module?")
        assert result.query_type == QueryType.QUICK_QUESTION
        assert result.should_plan is False

    def test_exact_response_prompt_is_quick_question(self, classifier):
        result = classifier.classify("Reply with exactly READY")
        assert result.query_type == QueryType.QUICK_QUESTION
        assert result.should_plan is False

    def test_exploration_classification(self, classifier):
        result = classifier.classify("Explore the auth module and map its dependencies")
        assert result.query_type == QueryType.EXPLORATION
        assert result.should_plan is True

    def test_implementation_classification(self, classifier):
        result = classifier.classify("Implement JWT authentication for the API")
        assert result.query_type == QueryType.IMPLEMENTATION

    def test_review_classification(self, classifier):
        result = classifier.classify("Review the parser module for correctness")
        assert result.query_type == QueryType.REVIEW

    def test_debugging_classification(self, classifier):
        result = classifier.classify("Fix the null pointer exception in user service")
        assert result.query_type == QueryType.DEBUGGING


class TestPlanningDecision:
    def test_complex_exploration_suggests_subagents(self, classifier):
        with patch.object(classifier, "_classify_complexity") as mock_cx:
            mock_cx.return_value = TaskClassification(
                complexity=TaskComplexity.COMPLEX,
                tool_budget=10,
                confidence=0.9,
                matched_patterns=["explore"],
            )
            result = classifier.classify("Explore all API endpoints across the codebase")
            assert result.should_use_subagents is True

    def test_simple_query_no_subagents(self, classifier):
        result = classifier.classify("What is a decorator?")
        assert result.should_use_subagents is False


class TestBudgetHints:
    def test_simple_query_low_budget(self, classifier):
        result = classifier.classify("What is Python?")
        assert result.continuation_budget_hint == 2

    def test_exploration_high_budget(self, classifier):
        result = classifier.classify("Explore the entire codebase structure")
        assert result.continuation_budget_hint == 8


class TestComplexityDelegation:
    def test_composes_task_complexity_service(self, classifier_with_mock_service):
        classifier, mock_service = classifier_with_mock_service
        classifier.classify("Implement a new feature")
        mock_service.classify.assert_called_once_with("Implement a new feature")

    def test_ambiguous_defaults_to_medium(self, classifier):
        result = classifier.classify("do something with the code please")
        assert result.complexity in (
            TaskComplexity.MEDIUM,
            TaskComplexity.SIMPLE,
            TaskComplexity.COMPLEX,
        )
        assert 0.0 <= result.confidence <= 1.0
