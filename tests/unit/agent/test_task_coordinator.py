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

"""Unit tests for TaskCoordinator.

Tests task preparation, intent detection, and task-specific guidance.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Any

from victor.agent.action_authorizer import ActionIntent, IntentClassification
from victor.agent.services.task_runtime import TaskCoordinator
from victor.agent.conversation.history_metadata import build_internal_history_metadata
from victor.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = Mock(spec=Settings)
    settings.temperature = 0.7
    settings.tool_budget = 15
    return settings


@pytest.fixture
def mock_task_analyzer():
    """Create mock TaskAnalyzer."""
    analyzer = MagicMock()

    # Configure complexity classification
    complexity_result = Mock()
    complexity_result.complexity = Mock(value="moderate")
    complexity_result.confidence = 0.8
    complexity_result.prompt_hint = "Test hint"
    analyzer.classify_complexity.return_value = complexity_result

    # Configure intent detection
    intent_result = Mock()
    intent_result.intent = Mock(value="write_allowed")
    intent_result.prompt_guard = None
    analyzer.detect_intent.return_value = intent_result

    return analyzer


@pytest.fixture
def mock_unified_tracker():
    """Create mock UnifiedTaskTracker."""
    tracker = MagicMock()
    tracker.config = {"max_total_iterations": 50}
    tracker.set_tool_budget = Mock()
    tracker._sticky_user_budget = False
    return tracker


@pytest.fixture
def mock_conversation_controller():
    """Create mock ConversationController."""
    controller = MagicMock()
    controller.add_message = Mock()
    return controller


@pytest.fixture
def mock_prompt_builder():
    """Create mock PromptBuilder."""
    builder = MagicMock()
    builder.prompt_contributors = []
    return builder


@pytest.fixture
def task_coordinator(
    mock_task_analyzer,
    mock_unified_tracker,
    mock_prompt_builder,
    mock_settings,
):
    """Create TaskCoordinator with mocked dependencies."""
    return TaskCoordinator(
        task_analyzer=mock_task_analyzer,
        unified_tracker=mock_unified_tracker,
        prompt_builder=mock_prompt_builder,
        settings=mock_settings,
    )


class TestTaskPreparation:
    """Tests for task preparation operations."""

    def test_prepare_task_basic(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Test basic task preparation."""
        unified_type = Mock(value="edit")
        message = "Fix the bug in test.py"

        classification, budget = task_coordinator.prepare_task(
            message, unified_type, mock_conversation_controller
        )

        # Verify task analyzer was called
        mock_task_analyzer.classify_complexity.assert_called_once_with(message)

        # Verify classification result returned
        assert classification is not None
        assert isinstance(budget, int)

    def test_prepare_task_promotes_simple_edit_request_to_action_budget(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Write-authorized edit tasks should not be flattened to SIMPLE budgets."""
        from victor.framework.task import TaskComplexity

        complexity_result = Mock()
        complexity_result.complexity = TaskComplexity.SIMPLE
        complexity_result.confidence = 0.62
        complexity_result.matched_patterns = []
        mock_task_analyzer.classify_complexity.return_value = complexity_result

        intent_result = Mock()
        intent_result.intent = ActionIntent.WRITE_ALLOWED
        intent_result.confidence = 0.91
        intent_result.prompt_guard = None
        mock_task_analyzer.detect_intent.return_value = intent_result

        classification, budget = task_coordinator.prepare_task(
            "Address them comprehensively.",
            Mock(value="edit"),
            mock_conversation_controller,
        )

        assert classification.complexity == TaskComplexity.ACTION
        assert budget >= 50
        mock_task_analyzer.detect_intent.assert_called_with("Address them comprehensively.")


class TestContinuationComplexityCarryForward:
    """A continuation turn must not shrink the budget of the work being continued.

    Regression: "continue next steps" was classified SIMPLE and slashed the tool
    budget 200 → 20, starving the follow-up turn (session codingagent-363cca81).
    """

    @staticmethod
    def _classification(complexity, confidence=0.9):
        result = Mock()
        result.complexity = complexity
        result.confidence = confidence
        result.matched_patterns = []
        return result

    def test_bare_continuation_carries_forward_prior_budget(
        self,
        task_coordinator,
        mock_task_analyzer,
        mock_unified_tracker,
        mock_conversation_controller,
    ):
        from victor.framework.task import DEFAULT_BUDGETS, TaskComplexity

        # Turn 1: an ACTION-complexity task (budget 200).
        mock_task_analyzer.classify_complexity.return_value = self._classification(
            TaskComplexity.ACTION
        )
        task_coordinator.prepare_task(
            "Implement the plan across the web server and CLI",
            Mock(value="edit"),
            mock_conversation_controller,
        )

        # Turn 2: a bare continuation, classified SIMPLE from its own text.
        mock_task_analyzer.classify_complexity.return_value = self._classification(
            TaskComplexity.SIMPLE, confidence=0.5
        )
        classification, budget = task_coordinator.prepare_task(
            "continue next steps",
            Mock(value="design"),
            mock_conversation_controller,
        )

        assert classification.complexity == TaskComplexity.ACTION
        assert budget == DEFAULT_BUDGETS[TaskComplexity.ACTION]
        assert "continuation_complexity_carry_forward" in classification.matched_patterns
        # The SIMPLE budget-slashing branch must not have run.
        mock_unified_tracker.set_tool_budget.assert_not_called()

    def test_non_continuation_turn_still_reclassifies(
        self,
        task_coordinator,
        mock_task_analyzer,
        mock_unified_tracker,
        mock_conversation_controller,
    ):
        from victor.framework.task import TaskComplexity

        mock_task_analyzer.classify_complexity.return_value = self._classification(
            TaskComplexity.ACTION
        )
        task_coordinator.prepare_task(
            "Implement the plan", Mock(value="edit"), mock_conversation_controller
        )

        mock_task_analyzer.classify_complexity.return_value = self._classification(
            TaskComplexity.SIMPLE, confidence=0.8
        )
        intent_result = Mock()
        intent_result.intent = ActionIntent.READ_ONLY
        intent_result.confidence = 0.8
        mock_task_analyzer.detect_intent.return_value = intent_result

        classification, _budget = task_coordinator.prepare_task(
            "list the files in victor/ui", Mock(value="search"), mock_conversation_controller
        )

        assert classification.complexity == TaskComplexity.SIMPLE
        mock_unified_tracker.set_tool_budget.assert_called_once()

    def test_continuation_never_raises_a_larger_fresh_classification(
        self,
        task_coordinator,
        mock_task_analyzer,
        mock_conversation_controller,
    ):
        """Carry-forward is monotone: it never shrinks a bigger fresh budget."""
        from victor.framework.task import DEFAULT_BUDGETS, TaskComplexity

        mock_task_analyzer.classify_complexity.return_value = self._classification(
            TaskComplexity.SIMPLE
        )
        task_coordinator.prepare_task(
            "show me the readme", Mock(value="search"), mock_conversation_controller
        )

        # Continuation with a payload that itself classifies as ACTION keeps ACTION.
        mock_task_analyzer.classify_complexity.return_value = self._classification(
            TaskComplexity.ACTION
        )
        classification, budget = task_coordinator.prepare_task(
            "continue and also refactor the session store",
            Mock(value="edit"),
            mock_conversation_controller,
        )

        assert classification.complexity == TaskComplexity.ACTION
        assert budget == DEFAULT_BUDGETS[TaskComplexity.ACTION]

    def test_first_turn_without_prior_classification_is_unchanged(
        self,
        task_coordinator,
        mock_task_analyzer,
        mock_unified_tracker,
        mock_conversation_controller,
    ):
        from victor.framework.task import TaskComplexity

        mock_task_analyzer.classify_complexity.return_value = self._classification(
            TaskComplexity.SIMPLE, confidence=0.8
        )
        intent_result = Mock()
        intent_result.intent = ActionIntent.READ_ONLY
        intent_result.confidence = 0.8
        mock_task_analyzer.detect_intent.return_value = intent_result

        classification, _budget = task_coordinator.prepare_task(
            "continue", Mock(value="search"), mock_conversation_controller
        )

        assert classification.complexity == TaskComplexity.SIMPLE
        mock_unified_tracker.set_tool_budget.assert_called_once()

    def test_prepare_task_promotes_remediation_followup_even_for_design_task_type(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Write-authorized remediation follow-ups should not stay on SIMPLE/design budgets."""
        from victor.framework.task import TaskComplexity

        complexity_result = Mock()
        complexity_result.complexity = TaskComplexity.SIMPLE
        complexity_result.confidence = 0.58
        complexity_result.matched_patterns = []
        mock_task_analyzer.classify_complexity.return_value = complexity_result

        intent_result = Mock()
        intent_result.intent = ActionIntent.WRITE_ALLOWED
        intent_result.confidence = 0.88
        intent_result.prompt_guard = None
        mock_task_analyzer.detect_intent.return_value = intent_result

        classification, budget = task_coordinator.prepare_task(
            "continue to address the remaining findings and suggestions",
            Mock(value="design"),
            mock_conversation_controller,
        )

        assert classification.complexity == TaskComplexity.ACTION
        assert budget >= 50

    def test_prepare_task_with_task_hint(self, task_coordinator, mock_conversation_controller):
        """Test task preparation with task hint injection."""
        from unittest.mock import patch

        unified_type = Mock(value="edit")
        message = "Refactor the code"

        with patch(
            "victor.agent.prompt_builder.get_task_type_hint",
            return_value="Test task hint",
        ):
            task_coordinator.prepare_task(message, unified_type, mock_conversation_controller)

            # Verify hint was injected as a user message with [TASK-HINT:] prefix
            mock_conversation_controller.add_message.assert_called()
            calls = [
                call
                for call in mock_conversation_controller.add_message.call_args_list
                if len(call[0]) >= 2 and call[0][0] == "user" and "[TASK-HINT:" in str(call[0][1])
            ]
            assert len(calls) > 0
            assert calls[0].kwargs["metadata"] == build_internal_history_metadata("task_hint")

    def test_prepare_task_skips_task_hint_for_direct_response(
        self, task_coordinator, mock_conversation_controller
    ):
        """Direct-response prompts should not receive hidden task-hint injections."""
        from unittest.mock import patch

        unified_type = Mock(value="general")

        with patch(
            "victor.agent.prompt_builder.get_task_type_hint",
            return_value="Test task hint",
        ):
            task_coordinator.prepare_task(
                "Reply with exactly READY",
                unified_type,
                mock_conversation_controller,
            )

        hint_calls = [
            call
            for call in mock_conversation_controller.add_message.call_args_list
            if len(call[0]) >= 2 and call[0][0] == "user" and "[TASK-HINT:" in str(call[0][1])
        ]
        assert hint_calls == []


class TestIntentDetection:
    """Tests for intent detection operations."""

    def test_apply_intent_guard_write_allowed(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Test intent guard with WRITE_ALLOWED intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Configure mock for WRITE_ALLOWED
        intent_result = Mock()
        intent_result.intent = ActionIntent.WRITE_ALLOWED
        intent_result.prompt_guard = None
        mock_task_analyzer.detect_intent.return_value = intent_result

        message = "Create a new feature"
        task_coordinator.apply_intent_guard(message, mock_conversation_controller)

        # Verify intent detection was called
        mock_task_analyzer.detect_intent.assert_called_once_with(message)

        # Verify current intent was set
        assert task_coordinator.current_intent == ActionIntent.WRITE_ALLOWED

        # No prompt guard should be added for WRITE_ALLOWED
        # (can't assert this directly without checking call args)

    def test_apply_intent_guard_display_only(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Test intent guard with DISPLAY_ONLY intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Configure mock for DISPLAY_ONLY
        intent_result = Mock()
        intent_result.intent = ActionIntent.DISPLAY_ONLY
        intent_result.prompt_guard = "READ ONLY MODE: Do not modify files."
        mock_task_analyzer.detect_intent.return_value = intent_result

        message = "Show me the code"
        task_coordinator.apply_intent_guard(message, mock_conversation_controller)

        # The guard is now delivered via the prompt builder's contextual guidance
        # ("INTENT CONSTRAINT: ..."), not injected as a visible user message.
        mock_conversation_controller.add_message.assert_not_called()
        assert (
            task_coordinator.prompt_builder.contextual_guidance
            == "INTENT CONSTRAINT: READ ONLY MODE: Do not modify files."
        )

        # Verify current intent was set
        assert task_coordinator.current_intent == ActionIntent.DISPLAY_ONLY

    def test_apply_intent_guard_skips_guard_for_direct_response(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Direct-response prompts should not receive hidden prompt-guard injections."""
        from victor.agent.action_authorizer import ActionIntent

        intent_result = Mock()
        intent_result.intent = ActionIntent.DISPLAY_ONLY
        intent_result.prompt_guard = "READ ONLY MODE: Do not modify files."
        mock_task_analyzer.detect_intent.return_value = intent_result

        task_coordinator.apply_intent_guard(
            "Reply with exactly READY",
            mock_conversation_controller,
        )

        mock_conversation_controller.add_message.assert_not_called()
        assert task_coordinator.current_intent == ActionIntent.DISPLAY_ONLY

    def test_apply_intent_guard_read_only(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Test intent guard with READ_ONLY intent."""
        from victor.agent.action_authorizer import ActionIntent

        # Configure mock for READ_ONLY
        intent_result = Mock()
        intent_result.intent = ActionIntent.READ_ONLY
        intent_result.prompt_guard = "This is read-only analysis."
        mock_task_analyzer.detect_intent.return_value = intent_result

        message = "Analyze the codebase"
        task_coordinator.apply_intent_guard(message, mock_conversation_controller)

        # The guard is delivered via prompt builder contextual guidance, not add_message.
        mock_conversation_controller.add_message.assert_not_called()
        assert (
            task_coordinator.prompt_builder.contextual_guidance
            == "INTENT CONSTRAINT: This is read-only analysis."
        )

    def test_apply_intent_guard_continuation_carries_forward_write_intent(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Continuation follow-ups should preserve prior write intent when underspecified."""
        task_coordinator._current_intent = ActionIntent.WRITE_ALLOWED
        mock_task_analyzer.detect_intent.return_value = IntentClassification(
            intent=ActionIntent.DISPLAY_ONLY,
            confidence=0.2,
            matched_signals=[],
            safe_actions=set(),
            prompt_guard="READ ONLY MODE: Do not modify files.",
        )

        task_coordinator.apply_intent_guard(
            "continue with the remaining files",
            mock_conversation_controller,
        )

        assert task_coordinator.current_intent == ActionIntent.WRITE_ALLOWED
        mock_conversation_controller.add_message.assert_not_called()

    def test_apply_intent_guard_continuation_carries_forward_read_only_intent(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Continuation follow-ups should preserve prior read-only intent when appropriate."""
        task_coordinator._current_intent = ActionIntent.READ_ONLY
        mock_task_analyzer.detect_intent.return_value = IntentClassification(
            intent=ActionIntent.DISPLAY_ONLY,
            confidence=0.2,
            matched_signals=[],
            safe_actions=set(),
            prompt_guard="READ ONLY MODE: Do not modify files.",
        )

        task_coordinator.apply_intent_guard(
            "continue reviewing the code",
            mock_conversation_controller,
        )

        assert task_coordinator.current_intent == ActionIntent.READ_ONLY
        # Carried-forward read-only intent guard is delivered via contextual guidance.
        mock_conversation_controller.add_message.assert_not_called()
        guidance = task_coordinator.prompt_builder.contextual_guidance
        assert guidance is not None
        assert guidance.startswith("INTENT CONSTRAINT:")
        assert "read-only query" in guidance

    def test_apply_intent_guard_bare_continue_uses_prior_intent(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """A bare continuation command should resume the prior intent, not reset it."""
        task_coordinator._current_intent = ActionIntent.READ_ONLY
        mock_task_analyzer.detect_intent.return_value = IntentClassification(
            intent=ActionIntent.WRITE_ALLOWED,
            confidence=0.9,
            matched_signals=["continuation_keyword"],
            safe_actions=set(),
            prompt_guard="",
        )

        task_coordinator.apply_intent_guard("continue", mock_conversation_controller)

        assert task_coordinator.current_intent == ActionIntent.READ_ONLY
        # Prior read-only intent is resumed and its guard delivered via contextual guidance.
        mock_conversation_controller.add_message.assert_not_called()
        assert task_coordinator.prompt_builder.contextual_guidance is not None
        assert task_coordinator.prompt_builder.contextual_guidance.startswith("INTENT CONSTRAINT:")

    def test_apply_intent_guard_injects_database_task_hint_for_explicit_sqlite_request(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Explicit database inspection requests should get a concrete task hint."""
        from victor.agent.action_authorizer import ActionIntent

        intent_result = Mock()
        intent_result.intent = ActionIntent.DISPLAY_ONLY
        intent_result.prompt_guard = "READ ONLY MODE: Do not modify files."
        mock_task_analyzer.detect_intent.return_value = intent_result

        task_coordinator.apply_intent_guard(
            "review the sqllite db directly and inspect the schema",
            mock_conversation_controller,
        )

        # The database-inspection hint is appended to contextual guidance, not injected
        # as a separate [TASK-HINT:] user message.
        guidance = task_coordinator.prompt_builder.contextual_guidance
        assert guidance is not None
        assert "DATABASE INSPECTION" in guidance
        assert "Prefer the db tool first" in guidance


class TestTaskGuidance:
    """Tests for task-specific guidance."""

    def test_apply_task_guidance_analysis_task(
        self, task_coordinator, mock_conversation_controller
    ):
        """Test task guidance for analysis tasks."""
        unified_type = Mock(value="analyze")
        message = "Analyze the project structure"

        # Set initial temperature
        initial_temperature = 0.7
        task_coordinator.temperature = initial_temperature

        task_coordinator.apply_task_guidance(
            user_message=message,
            unified_task_type=unified_type,
            is_analysis_task=True,
            is_action_task=False,
            needs_execution=False,
            max_exploration_iterations=50,
            conversation_controller=mock_conversation_controller,
        )

        # Note: Temperature for analysis is handled by the unified resolver (ADR-013)
        # at the provider seam, not by scalar mutation in apply_task_guidance.
        # The temperature remains unchanged here - the policy is applied elsewhere.
        assert task_coordinator.temperature == initial_temperature

        # Verify system messages were added
        assert mock_conversation_controller.add_message.call_count >= 2

    def test_apply_task_guidance_action_task(self, task_coordinator, mock_conversation_controller):
        """Test task guidance for action tasks."""
        unified_type = Mock(value="create")
        message = "Create a new script"

        task_coordinator.apply_task_guidance(
            user_message=message,
            unified_task_type=unified_type,
            is_analysis_task=False,
            is_action_task=True,
            needs_execution=True,
            max_exploration_iterations=10,
            conversation_controller=mock_conversation_controller,
        )

        # Verify system message was added for action task
        mock_conversation_controller.add_message.assert_called()
        calls = mock_conversation_controller.add_message.call_args_list
        assert any("action-oriented" in str(call).lower() for call in calls)

    def test_apply_task_guidance_analysis_tool_budget(
        self, task_coordinator, mock_conversation_controller
    ):
        """Test that analysis tasks increase tool budget."""
        unified_type = Mock(value="analyze")

        # Set initial tool budget
        task_coordinator.tool_budget = 15

        task_coordinator.apply_task_guidance(
            user_message="Analyze all files",
            unified_task_type=unified_type,
            is_analysis_task=True,
            is_action_task=False,
            needs_execution=False,
            max_exploration_iterations=100,
            conversation_controller=mock_conversation_controller,
        )

        # Verify tool budget was increased for analysis
        assert task_coordinator.tool_budget >= 200


class TestTaskCoordinatorProperties:
    """Tests for TaskCoordinator properties."""

    def test_temperature_property(self, task_coordinator):
        """Test temperature property getter/setter."""
        task_coordinator.temperature = 0.9
        assert task_coordinator.temperature == 0.9

    def test_tool_budget_property(self, task_coordinator):
        """Test tool_budget property getter/setter."""
        task_coordinator.tool_budget = 25
        assert task_coordinator.tool_budget == 25

    def test_observed_files_property(self, task_coordinator):
        """Test observed_files property getter/setter."""
        files = ["test.py", "main.py"]
        task_coordinator.observed_files = files
        assert task_coordinator.observed_files == files

    def test_current_intent_property(
        self, task_coordinator, mock_task_analyzer, mock_conversation_controller
    ):
        """Test current_intent property."""
        from victor.agent.action_authorizer import ActionIntent

        # Set intent via apply_intent_guard
        intent_result = Mock()
        intent_result.intent = ActionIntent.READ_ONLY
        intent_result.prompt_guard = "Guard"
        mock_task_analyzer.detect_intent.return_value = intent_result

        task_coordinator.apply_intent_guard("test message", mock_conversation_controller)

        assert task_coordinator.current_intent == ActionIntent.READ_ONLY


class TestTaskCoordinatorInitialization:
    """Tests for TaskCoordinator initialization."""

    def test_initialization_with_valid_dependencies(
        self,
        mock_task_analyzer,
        mock_unified_tracker,
        mock_prompt_builder,
        mock_settings,
    ):
        """Test successful initialization with valid dependencies."""
        coordinator = TaskCoordinator(
            task_analyzer=mock_task_analyzer,
            unified_tracker=mock_unified_tracker,
            prompt_builder=mock_prompt_builder,
            settings=mock_settings,
        )

        assert coordinator.task_analyzer is mock_task_analyzer
        assert coordinator.unified_tracker is mock_unified_tracker
        assert coordinator.prompt_builder is mock_prompt_builder
        assert coordinator.settings is mock_settings

    def test_set_reminder_manager(self, task_coordinator):
        """Test setting reminder manager."""
        reminder_manager = Mock()
        task_coordinator.set_reminder_manager(reminder_manager)

        assert task_coordinator._reminder_manager is reminder_manager
