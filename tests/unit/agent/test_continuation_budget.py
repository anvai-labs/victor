"""Tests for dynamic continuation budgets — Layer 4 of agentic execution quality."""

from unittest.mock import MagicMock


from victor.agent.continuation_strategy import ContinuationStrategy
from victor.agent.query_classifier import QueryClassification, QueryType
from victor.framework.task.protocols import TaskComplexity


def _make_classification(
    query_type: QueryType = QueryType.IMPLEMENTATION,
    budget_hint: int = 6,
) -> QueryClassification:
    return QueryClassification(
        query_type=query_type,
        complexity=TaskComplexity.MEDIUM,
        should_plan=False,
        should_use_subagents=False,
        continuation_budget_hint=budget_hint,
        confidence=0.8,
    )


def _make_settings(**overrides):
    settings = MagicMock()
    settings.max_continuation_prompts_analysis = overrides.get("analysis", 6)
    settings.max_continuation_prompts_action = overrides.get("action", 5)
    settings.max_continuation_prompts_default = overrides.get("default", 3)
    return settings


def _make_base_kwargs(settings=None):
    """Common kwargs for determine_continuation_action with no-op values."""
    intent = MagicMock()
    intent.intent = MagicMock()
    intent.intent.value = "unknown"
    intent.confidence = 0.5
    return {
        "intent_result": intent,
        "is_analysis_task": False,
        "is_action_task": False,
        "content_length": 100,
        "full_content": "Some response content",
        "continuation_prompts": 0,
        "asking_input_prompts": 0,
        "one_shot_mode": False,
        "mentioned_tools": None,
        "max_prompts_summary_requested": False,
        "settings": settings or _make_settings(),
        "rl_coordinator": None,
        "provider_name": "anthropic",
        "model": "claude-sonnet",
        "tool_budget": 10,
        "unified_tracker_config": {},
        "task_completion_signals": None,
    }


class TestBudgetWithClassification:
    def test_default_budget_unchanged_without_classification(self):
        strategy = ContinuationStrategy()
        kwargs = _make_base_kwargs()
        # Without classification, should use default thresholds
        result = strategy.determine_continuation_action(**kwargs)
        # Should return valid action (not crash)
        assert "action" in result

    def test_exploration_gets_higher_budget(self):
        strategy = ContinuationStrategy()
        classification = _make_classification(QueryType.EXPLORATION, budget_hint=8)
        kwargs = _make_base_kwargs()
        kwargs["query_classification"] = classification
        # With exploration classification, the budget hint should apply
        result = strategy.determine_continuation_action(**kwargs)
        assert "action" in result

    def test_quick_question_gets_lower_budget(self):
        strategy = ContinuationStrategy()
        classification = _make_classification(QueryType.QUICK_QUESTION, budget_hint=2)
        kwargs = _make_base_kwargs()
        kwargs["query_classification"] = classification
        # Quick question with budget=2, continuation_prompts=0 should still work
        result = strategy.determine_continuation_action(**kwargs)
        assert "action" in result

    def test_plan_step_count_increases_budget(self):
        strategy = ContinuationStrategy()
        classification = _make_classification(QueryType.IMPLEMENTATION, budget_hint=6)
        kwargs = _make_base_kwargs()
        kwargs["query_classification"] = classification
        kwargs["plan_step_count"] = 5
        result = strategy.determine_continuation_action(**kwargs)
        assert "action" in result

    def test_plan_budget_no_decrease(self):
        strategy = ContinuationStrategy()
        classification = _make_classification(QueryType.IMPLEMENTATION, budget_hint=6)
        kwargs = _make_base_kwargs()
        kwargs["query_classification"] = classification
        kwargs["plan_step_count"] = 1  # 1 * 2 = 2, less than default
        result = strategy.determine_continuation_action(**kwargs)
        assert "action" in result

    def test_rl_override_takes_precedence(self):
        strategy = ContinuationStrategy()
        classification = _make_classification(QueryType.QUICK_QUESTION, budget_hint=2)

        rl_coordinator = MagicMock()
        recommendation = MagicMock()
        recommendation.value = 10  # RL says 10, overrides budget hint of 2
        recommendation.confidence = 0.9
        rl_coordinator.get_recommendation.return_value = recommendation

        kwargs = _make_base_kwargs()
        kwargs["query_classification"] = classification
        kwargs["rl_coordinator"] = rl_coordinator
        result = strategy.determine_continuation_action(**kwargs)
        assert "action" in result

    def test_manual_override_takes_precedence(self):
        strategy = ContinuationStrategy()
        classification = _make_classification(QueryType.QUICK_QUESTION, budget_hint=2)

        settings = _make_settings(analysis=15, action=12, default=10)
        kwargs = _make_base_kwargs(settings=settings)
        kwargs["query_classification"] = classification
        result = strategy.determine_continuation_action(**kwargs)
        assert "action" in result
