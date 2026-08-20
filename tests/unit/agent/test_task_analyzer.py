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

"""Tests for the unified TaskAnalyzer."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from victor.agent.task_analyzer import (
    TaskAnalyzer,
    TaskAnalysis,
    get_task_analyzer,
    reset_task_analyzer,
)
from victor.framework.task.complexity import TaskComplexity
from victor.agent.action_authorizer import ActionIntent
from victor.core.container import reset_container


class TestTaskAnalysis:
    """Tests for TaskAnalysis dataclass."""

    def test_is_simple(self):
        """Test is_simple property."""
        analysis = TaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=2,
            complexity_confidence=0.9,
        )
        assert analysis.is_simple
        assert not analysis.is_complex

    def test_is_complex(self):
        """Test is_complex property."""
        analysis = TaskAnalysis(
            complexity=TaskComplexity.COMPLEX,
            tool_budget=20,
            complexity_confidence=0.9,
        )
        assert analysis.is_complex
        assert not analysis.is_simple

    def test_is_generation(self):
        """Test is_generation property."""
        analysis = TaskAnalysis(
            complexity=TaskComplexity.GENERATION,
            tool_budget=5,
            complexity_confidence=0.9,
        )
        assert analysis.is_generation
        assert not analysis.is_simple

    def test_should_force_completion(self):
        """Test should_force_completion method."""
        analysis = TaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=3,
            complexity_confidence=0.9,
        )
        assert not analysis.should_force_completion(2)
        assert analysis.should_force_completion(3)
        assert analysis.should_force_completion(5)

    def test_has_team_suggestion_no_coordination(self):
        """Test has_team_suggestion when no coordination suggestion."""
        analysis = TaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=3,
            complexity_confidence=0.9,
        )
        assert not analysis.has_team_suggestion

    def test_has_team_suggestion_with_empty_team_rec(self):
        """Test has_team_suggestion with empty team recommendations."""
        mock_coordination = MagicMock()
        mock_coordination.team_recommendations = []

        analysis = TaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=3,
            complexity_confidence=0.9,
            coordination_suggestion=mock_coordination,
        )
        assert not analysis.has_team_suggestion

    def test_has_team_suggestion_with_team_rec(self):
        """Test has_team_suggestion with team recommendations."""
        mock_coordination = MagicMock()
        mock_coordination.team_recommendations = [MagicMock()]

        analysis = TaskAnalysis(
            complexity=TaskComplexity.COMPLEX,
            tool_budget=10,
            complexity_confidence=0.9,
            coordination_suggestion=mock_coordination,
        )
        assert analysis.has_team_suggestion

    def test_should_spawn_team_no_coordination(self):
        """Test should_spawn_team when no coordination suggestion."""
        analysis = TaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=3,
            complexity_confidence=0.9,
        )
        assert not analysis.should_spawn_team

    def test_should_spawn_team_with_suggestion(self):
        """Test should_spawn_team with coordination suggestion."""
        mock_coordination = MagicMock()
        mock_coordination.should_spawn_team = True

        analysis = TaskAnalysis(
            complexity=TaskComplexity.COMPLEX,
            tool_budget=10,
            complexity_confidence=0.9,
            coordination_suggestion=mock_coordination,
        )
        assert analysis.should_spawn_team

    def test_primary_team_no_coordination(self):
        """Test primary_team when no coordination suggestion."""
        analysis = TaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=3,
            complexity_confidence=0.9,
        )
        assert analysis.primary_team is None

    def test_primary_team_with_suggestion(self):
        """Test primary_team with coordination suggestion."""
        mock_team_rec = MagicMock()
        mock_team_rec.team_name = "coding_team"

        mock_coordination = MagicMock()
        mock_coordination.primary_team = mock_team_rec

        analysis = TaskAnalysis(
            complexity=TaskComplexity.COMPLEX,
            tool_budget=10,
            complexity_confidence=0.9,
            coordination_suggestion=mock_coordination,
        )
        assert analysis.primary_team == "coding_team"

    def test_primary_team_none_suggestion(self):
        """Test primary_team when no primary team recommendation."""
        mock_coordination = MagicMock()
        mock_coordination.primary_team = None

        analysis = TaskAnalysis(
            complexity=TaskComplexity.SIMPLE,
            tool_budget=3,
            complexity_confidence=0.9,
            coordination_suggestion=mock_coordination,
        )
        assert analysis.primary_team is None


class TestTaskAnalyzer:
    """Tests for TaskAnalyzer class."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_task_analyzer()

    def teardown_method(self):
        """Reset global state after each test."""
        reset_task_analyzer()

    def test_analyze_simple_query(self):
        """Test analyzing a simple query."""
        analyzer = TaskAnalyzer()
        result = analyzer.analyze("What files are in this project?")

        assert isinstance(result, TaskAnalysis)
        assert result.complexity in [TaskComplexity.SIMPLE, TaskComplexity.MEDIUM]
        assert result.tool_budget > 0

    def test_analyze_complex_query(self):
        """Test analyzing a complex query."""
        analyzer = TaskAnalyzer()
        result = analyzer.analyze("Refactor the authentication system to use JWT tokens")

        assert isinstance(result, TaskAnalysis)
        # Complex tasks should have higher budget
        assert result.tool_budget >= 5

    def test_analyze_write_request(self):
        """Test analyzing a write request."""
        analyzer = TaskAnalyzer()
        result = analyzer.analyze("Create a new file called hello.py")

        assert isinstance(result, TaskAnalysis)
        assert result.action_intent in [
            ActionIntent.WRITE_ALLOWED,
            ActionIntent.AMBIGUOUS,
        ]

    def test_analyze_display_request(self):
        """Test analyzing a display-only request."""
        analyzer = TaskAnalyzer()
        result = analyzer.analyze("Show me the contents of README.md")

        assert isinstance(result, TaskAnalysis)
        # Display requests typically don't authorize writes
        assert result.action_intent in [
            ActionIntent.DISPLAY_ONLY,
            ActionIntent.READ_ONLY,
            ActionIntent.AMBIGUOUS,
        ]

    def test_classify_complexity(self):
        """Test quick complexity classification."""
        analyzer = TaskAnalyzer()
        result = analyzer.classify_complexity("List all Python files")

        assert result.complexity in [TaskComplexity.SIMPLE, TaskComplexity.MEDIUM]

    def test_check_write_authorization(self):
        """Test quick write authorization check."""
        analyzer = TaskAnalyzer()

        # Explicit write request with file path
        result = analyzer.check_write_authorization("Create and save a new file at hello.py")
        assert isinstance(result, bool)

        # Display request - should not authorize writes
        result2 = analyzer.check_write_authorization("Show me the contents of README.md")
        assert isinstance(result2, bool)

    def test_unified_classifier_uses_runtime_intelligence_when_configured(self):
        """TaskAnalyzer should build its unified classifier on the canonical runtime service."""
        runtime_intelligence = MagicMock()
        analyzer = TaskAnalyzer(runtime_intelligence=runtime_intelligence)

        classifier = analyzer.unified_classifier

        assert classifier._runtime_intelligence is runtime_intelligence

    def test_is_simple_query(self):
        """Test is_simple_query helper."""
        analyzer = TaskAnalyzer()

        # Returns boolean
        result = analyzer.is_simple_query("What is X?")
        assert isinstance(result, bool)

        # Complex query should not be simple
        result2 = analyzer.is_simple_query(
            "Analyze the entire codebase and refactor all authentication modules"
        )
        assert isinstance(result2, bool)

    def test_get_tool_budget(self):
        """Test get_tool_budget helper."""
        analyzer = TaskAnalyzer()

        budget = analyzer.get_tool_budget("Analyze the entire codebase")
        assert isinstance(budget, int)
        assert budget > 0

    def test_lazy_loading(self):
        """Test that classifiers are lazily loaded."""
        analyzer = TaskAnalyzer()

        # Initially none are loaded
        assert analyzer._complexity_classifier is None
        assert analyzer._action_authorizer is None

        # Access triggers loading
        _ = analyzer.complexity_classifier
        assert analyzer._complexity_classifier is not None

        _ = analyzer.action_authorizer
        assert analyzer._action_authorizer is not None

    def test_set_coordination_advisor(self):
        """Test setting the canonical coordination advisor."""
        analyzer = TaskAnalyzer()
        advisor = MagicMock()

        analyzer.set_coordination_advisor(advisor)

        assert analyzer._coordination_advisor is advisor

    def test_set_coordinator_warns_and_updates_coordination_advisor(self):
        """Deprecated coordinator setter should map to the canonical advisor surface."""
        analyzer = TaskAnalyzer()
        advisor = MagicMock()

        with pytest.warns(
            DeprecationWarning,
            match="TaskAnalyzer.set_coordinator\\(\\.\\.\\.\\) is deprecated",
        ):
            analyzer.set_coordinator(advisor)

        assert analyzer._coordination_advisor is advisor

    def test_set_coordination_runtime(self):
        """Test setting the canonical coordination runtime."""
        analyzer = TaskAnalyzer()
        coordination_runtime = MagicMock()

        analyzer.set_coordination_runtime(coordination_runtime)

        assert analyzer._coordination_runtime is coordination_runtime

    def test_classify_task_keywords(self):
        """Test classify_task_keywords method."""
        analyzer = TaskAnalyzer()

        # Analysis task
        result = analyzer.classify_task_keywords("What files are in this project?")
        assert isinstance(result, dict)
        assert "is_analysis_task" in result
        assert "is_action_task" in result
        assert "needs_execution" in result
        assert "coarse_task_type" in result

    def test_analyze_with_suggestions_no_coordinator(self):
        """Test analyze_with_suggestions without coordinator."""
        analyzer = TaskAnalyzer()

        result = analyzer.analyze_with_suggestions("Refactor this code", mode="build")

        assert isinstance(result, TaskAnalysis)
        # No coordinator means no suggestions
        assert not result.has_team_suggestion

    def test_analyze_with_suggestions_uses_coordination_runtime_when_available(self):
        """TaskAnalyzer should prefer the service-owned coordination runtime when bound."""
        coordination_runtime = MagicMock()
        analyzer = TaskAnalyzer(
            coordination_runtime=coordination_runtime,
            runtime_subject=SimpleNamespace(name="runtime"),
        )
        suggestion = MagicMock()
        suggestion.has_team_suggestion = True
        suggestion.team_recommendations = [MagicMock()]
        suggestion.action.value = "auto_spawn"

        coordination_runtime.suggest_for_task.return_value = suggestion

        result = analyzer.analyze_with_suggestions("Refactor this code", mode="build")

        assert result.coordination_suggestion is suggestion
        coordination_runtime.suggest_for_task.assert_called_once_with(
            runtime_subject=analyzer._runtime_subject,
            coordination_advisor=None,
            task_type=result.unified_task_type.value.lower(),
            complexity=result.complexity.value.lower(),
            mode="build",
        )

    def test_analyze_with_suggestions_auto_provisions_coordination_runtime(self):
        """TaskAnalyzer should auto-provision the service runtime for runtime-subject paths."""
        analyzer = TaskAnalyzer(runtime_subject=SimpleNamespace(name="runtime"))
        suggestion = MagicMock()
        suggestion.has_team_suggestion = True
        suggestion.team_recommendations = [MagicMock()]
        suggestion.action.value = "auto_spawn"

        with patch(
            "victor.agent.services.coordination_advisor_runtime.CoordinationAdvisorRuntime"
        ) as runtime_cls:
            runtime_cls.return_value.suggest_for_task.return_value = suggestion
            result = analyzer.analyze_with_suggestions("Refactor this code", mode="build")

        assert result.coordination_suggestion is suggestion
        assert analyzer._coordination_runtime is runtime_cls.return_value
        runtime_cls.return_value.suggest_for_task.assert_called_once_with(
            runtime_subject=analyzer._runtime_subject,
            coordination_advisor=None,
            task_type=result.unified_task_type.value.lower(),
            complexity=result.complexity.value.lower(),
            mode="build",
        )

    def test_analyze_with_suggestions_uses_coordination_advisor_when_available(self):
        """TaskAnalyzer should route advisor-only usage through the service runtime."""
        advisor = MagicMock()
        suggestion = MagicMock()
        suggestion.has_team_suggestion = False
        analyzer = TaskAnalyzer(coordination_advisor=advisor)

        with patch(
            "victor.agent.services.coordination_advisor_runtime.CoordinationAdvisorRuntime"
        ) as runtime_cls:
            runtime_cls.return_value.suggest_for_task.return_value = suggestion
            result = analyzer.analyze_with_suggestions("Refactor this code", mode="plan")

        assert result.coordination_suggestion is suggestion
        runtime_cls.return_value.suggest_for_task.assert_called_once_with(
            runtime_subject=None,
            coordination_advisor=advisor,
            task_type=result.unified_task_type.value.lower(),
            complexity=result.complexity.value.lower(),
            mode="plan",
        )

    def test_init_with_deprecated_coordinator_warns_and_maps_to_coordination_advisor(
        self,
    ):
        """Deprecated constructor argument should forward to coordination_advisor."""
        advisor = MagicMock()

        with pytest.warns(
            DeprecationWarning,
            match="TaskAnalyzer\\(coordinator=\\.\\.\\.\\) is deprecated",
        ):
            analyzer = TaskAnalyzer(coordinator=advisor)

        assert analyzer._coordination_advisor is advisor

    def test_analyze_with_history_context(self):
        """Test analyze with proper history context."""
        analyzer = TaskAnalyzer()

        # History should be list of message dicts
        result = analyzer.analyze(
            "Continue the task",
            context={"history": [{"role": "user", "content": "Previous message"}]},
        )

        assert isinstance(result, TaskAnalysis)

    def test_analyze_with_task_type_enabled(self):
        """Test analyze with task type classification."""
        analyzer = TaskAnalyzer()

        result = analyzer.analyze(
            "Refactor this code",
            include_task_type=True,
        )

        assert isinstance(result, TaskAnalysis)
        # Task type may or may not be set depending on if classifier is available
        assert isinstance(result.task_type_confidence, float)

    def test_analyze_with_intent_enabled(self):
        """Test analyze with intent classification."""
        analyzer = TaskAnalyzer()

        result = analyzer.analyze(
            "Show me the file",
            include_intent=True,
        )

        assert isinstance(result, TaskAnalysis)
        # Intent may or may not be set depending on if classifier is available


class TestGlobalAnalyzer:
    """Tests for global analyzer management."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_task_analyzer()

    def teardown_method(self):
        """Reset global state after each test."""
        reset_task_analyzer()

    def test_get_task_analyzer(self):
        """Test getting global analyzer."""
        analyzer = get_task_analyzer()
        assert isinstance(analyzer, TaskAnalyzer)

    def test_get_task_analyzer_same_instance(self):
        """Test that get_task_analyzer returns same instance."""
        analyzer1 = get_task_analyzer()
        analyzer2 = get_task_analyzer()
        assert analyzer1 is analyzer2

    def test_reset_task_analyzer(self):
        """Test resetting global analyzer."""
        reset_container()
        reset_task_analyzer()
        analyzer1 = get_task_analyzer()
        reset_container()
        reset_task_analyzer()
        analyzer2 = get_task_analyzer()

        assert analyzer1 is not analyzer2
