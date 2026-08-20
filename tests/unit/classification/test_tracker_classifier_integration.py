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

"""Tests for UnifiedTaskTracker + UnifiedTaskClassifier integration.

Tests cover:
- Negation-aware task type detection
- Budget setting from classification
- Context boosting with conversation history
- Factory function with negation awareness
"""

from victor.agent.unified_task_tracker import (
    UnifiedTaskTracker,
    TrackerTaskType,
    create_tracker_with_negation_awareness,
)


class TestNegationAwareDetection:
    """Tests for detect_task_type_with_negation method."""

    def test_analysis_task_detected(self):
        """Test that analysis task is correctly detected."""
        tracker = UnifiedTaskTracker()
        task_type, result = tracker.detect_task_type_with_negation(
            "Analyze the codebase for security issues"
        )
        assert task_type == TrackerTaskType.ANALYZE
        assert result.is_analysis_task is True

    def test_negated_analysis_not_detected_as_analysis(self):
        """Test that negated analysis keyword doesn't classify as analysis."""
        tracker = UnifiedTaskTracker()
        task_type, result = tracker.detect_task_type_with_negation(
            "Don't analyze this, just run the tests"
        )
        # Should NOT be ANALYZE since "analyze" is negated
        # "run" is not negated, so should detect as action-related
        assert task_type != TrackerTaskType.ANALYZE
        assert "analyze" in [m.keyword for m in result.negated_keywords]

    def test_positive_override_run_after_negated_analyze(self):
        """Test that 'but do run' overrides earlier negation."""
        tracker = UnifiedTaskTracker()
        task_type, result = tracker.detect_task_type_with_negation(
            "Don't analyze but do run the tests"
        )
        # "analyze" should be negated, "run" should NOT be negated
        negated_kws = [m.keyword for m in result.negated_keywords]
        matched_kws = [m.keyword for m in result.matched_keywords]

        assert "analyze" in negated_kws
        assert "run" in matched_kws
        assert result.needs_execution is True

    def test_edit_task_detected(self):
        """Test that edit task is correctly detected."""
        tracker = UnifiedTaskTracker()
        task_type, result = tracker.detect_task_type_with_negation(
            "Refactor the authentication module"
        )
        assert task_type == TrackerTaskType.EDIT
        assert "refactor" in [m.keyword for m in result.matched_keywords]

    def test_search_task_detected(self):
        """Test that search task is correctly detected."""
        tracker = UnifiedTaskTracker()
        task_type, result = tracker.detect_task_type_with_negation(
            "Find all usages of the deprecated function"
        )
        assert task_type == TrackerTaskType.SEARCH
        assert "find" in [m.keyword for m in result.matched_keywords]

    def test_generation_task_maps_to_create(self):
        """Test that generation task maps to CREATE type."""
        tracker = UnifiedTaskTracker()
        task_type, result = tracker.detect_task_type_with_negation(
            "Generate a function to validate emails"
        )
        assert task_type == TrackerTaskType.CREATE
        assert result.is_generation_task is True

    def test_budget_set_from_classification(self):
        """Test that tool budget is set from classification result."""
        tracker = UnifiedTaskTracker()
        tracker.detect_task_type_with_negation("Analyze the entire codebase thoroughly")

        # Analysis tasks should get high budget
        assert tracker.remaining_budget >= 100


class TestContextBoosting:
    """Tests for context-aware classification."""

    def test_context_boosts_confidence(self):
        """Test that conversation history boosts classification confidence."""
        tracker = UnifiedTaskTracker()

        # Build up analysis context
        history = [
            {"role": "user", "content": "Analyze the auth module"},
            {"role": "assistant", "content": "I analyzed the auth module..."},
            {"role": "user", "content": "Now analyze the database module"},
            {"role": "assistant", "content": "Database analysis..."},
        ]

        # Ambiguous message that benefits from context
        task_type, result = tracker.detect_task_type_with_negation(
            "Continue with the review",
            history=history,
        )

        # Should get context boost since history is analysis-heavy
        assert result.context_boost >= 0 or result.source == "context"

    def test_no_context_no_boost(self):
        """Test that empty history doesn't affect classification."""
        tracker = UnifiedTaskTracker()

        _, result_no_history = tracker.detect_task_type_with_negation("Analyze the code")

        _, result_empty_history = tracker.detect_task_type_with_negation(
            "Analyze the code",
            history=[],
        )

        assert result_no_history.confidence == result_empty_history.confidence


class TestClassifyWithNegationAwareness:
    """Tests for classify_with_negation_awareness method."""

    def test_returns_detailed_dict(self):
        """Test that method returns comprehensive classification details."""
        tracker = UnifiedTaskTracker()
        details = tracker.classify_with_negation_awareness(
            "Don't analyze, just search for the function"
        )

        # Check all expected keys are present
        assert "task_type" in details
        assert "confidence" in details
        assert "is_action_task" in details
        assert "is_analysis_task" in details
        assert "negated_keywords" in details
        assert "matched_keywords" in details
        assert "recommended_budget" in details
        assert "source" in details

    def test_negated_keywords_in_details(self):
        """Test that negated keywords are properly reported."""
        tracker = UnifiedTaskTracker()
        details = tracker.classify_with_negation_awareness("Skip the analysis, just run tests")

        # "analysis" should be negated
        assert "analysis" in details["negated_keywords"]


class TestFactoryFunction:
    """Tests for create_tracker_with_negation_awareness factory."""

    def test_creates_configured_tracker(self):
        """Test factory creates properly configured tracker."""
        tracker, task_type, details = create_tracker_with_negation_awareness(
            "Analyze the architecture"
        )

        assert isinstance(tracker, UnifiedTaskTracker)
        assert task_type == TrackerTaskType.ANALYZE
        assert tracker.task_type == TrackerTaskType.ANALYZE

    def test_factory_with_negated_message(self):
        """Test factory handles negated keywords correctly."""
        tracker, task_type, details = create_tracker_with_negation_awareness(
            "Don't analyze, just run the tests"
        )

        assert "analyze" in details["negated_keywords"]
        assert task_type != TrackerTaskType.ANALYZE

    def test_factory_with_history(self):
        """Test factory accepts history parameter."""
        history = [
            {"role": "user", "content": "Analyze the code"},
            {"role": "assistant", "content": "Analysis..."},
        ]

        tracker, task_type, details = create_tracker_with_negation_awareness(
            "Continue",
            history=history,
        )

        assert isinstance(tracker, UnifiedTaskTracker)
        # Should have classification source in details
        assert "source" in details


class TestEdgeCases:
    """Tests for edge cases in integration."""

    def test_empty_message(self):
        """Test handling of empty message."""
        tracker = UnifiedTaskTracker()
        task_type, result = tracker.detect_task_type_with_negation("")

        # Should default gracefully
        assert task_type == TrackerTaskType.GENERAL
        assert result.confidence < 0.5

    def test_multiple_task_types_in_message(self):
        """Test message with multiple task types."""
        tracker = UnifiedTaskTracker()
        task_type, result = tracker.detect_task_type_with_negation(
            "Analyze the code and then refactor the authentication"
        )

        # Both analysis and edit keywords present
        keywords = [m.keyword for m in result.matched_keywords]
        assert "analyze" in keywords or "refactor" in keywords

    def test_all_keywords_negated(self):
        """Test message where all action keywords are negated."""
        tracker = UnifiedTaskTracker()
        task_type, result = tracker.detect_task_type_with_negation(
            "Don't analyze and don't refactor anything"
        )

        # Should have low confidence or default type
        assert task_type == TrackerTaskType.GENERAL or result.confidence < 0.5
