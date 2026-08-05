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

"""Unit tests for UnifiedTaskClassifier.

Tests cover:
- Basic keyword classification
- Negation detection
- Confidence scoring
- Contextual classification
- Ensemble voting
- Edge cases and backward compatibility
"""

from unittest.mock import MagicMock

import pytest
from victor.agent.decisions.schemas import (
    DecisionType,
    TaskCategoryType,
    TaskTypeDecision,
)
from victor.agent.services.protocols.decision_service import DecisionResult
from victor.agent.unified_classifier import (
    ClassificationResult,
    ClassifierTaskType,
    KeywordMatch,
    UnifiedTaskClassifier,
    _is_keyword_negated,
    _find_keywords_with_positions,
    _calculate_category_score,
    ACTION_KEYWORDS,
    ANALYSIS_KEYWORDS,
    classify_task,
    get_unified_classifier,
)


class TestKeywordMatching:
    """Tests for basic keyword matching functionality."""

    def test_find_action_keywords(self):
        """Test finding action keywords in message."""
        matches = _find_keywords_with_positions("Run the tests and deploy", ACTION_KEYWORDS)
        keywords = [m.keyword for m in matches]
        assert "run" in keywords
        assert "deploy" in keywords

    def test_find_analysis_keywords(self):
        """Test finding analysis keywords in message."""
        matches = _find_keywords_with_positions("Analyze the codebase", ANALYSIS_KEYWORDS)
        keywords = [m.keyword for m in matches]
        assert "analyze" in keywords

    def test_case_insensitive_matching(self):
        """Test that matching is case insensitive."""
        matches = _find_keywords_with_positions("ANALYZE the CODE", ANALYSIS_KEYWORDS)
        assert len(matches) > 0
        assert any(m.keyword == "analyze" for m in matches)

    def test_word_boundary_matching(self):
        """Test that keywords match on word boundaries."""
        # "creation" should not match "create"
        matches = _find_keywords_with_positions("The creation of the file", [("create", 1.0)])
        assert len(matches) == 0

        # "create" should match
        matches = _find_keywords_with_positions("Create a new file", [("create", 1.0)])
        assert len(matches) == 1


class TestNegationDetection:
    """Tests for negation detection functionality."""

    def test_dont_negation(self):
        """Test 'don't' negation pattern."""
        assert _is_keyword_negated("Don't analyze this", "analyze", 7)
        assert _is_keyword_negated("don't run the tests", "run", 7)

    def test_do_not_negation(self):
        """Test 'do not' negation pattern."""
        assert _is_keyword_negated("Do not analyze the code", "analyze", 10)
        assert _is_keyword_negated("Please do not execute this", "execute", 18)

    def test_skip_negation(self):
        """Test 'skip' negation pattern."""
        assert _is_keyword_negated("Skip the review please", "review", 9)

    def test_without_negation(self):
        """Test 'without' negation pattern."""
        assert _is_keyword_negated("Without analyzing deeply", "analyzing", 8)

    def test_no_negation_positive_case(self):
        """Test that non-negated keywords are not flagged."""
        assert not _is_keyword_negated("Please analyze the code", "analyze", 7)
        assert not _is_keyword_negated("Run the tests now", "run", 0)

    def test_negation_window_limit(self):
        """Test that negation detection respects window size."""
        # Negation too far from keyword
        long_text = "Don't " + "x" * 50 + " analyze"
        position = long_text.index("analyze")
        assert not _is_keyword_negated(long_text, "analyze", position)

    def test_positive_override_but_do(self):
        """Test 'but do' overrides earlier negation."""
        text = "Don't analyze but do run the tests"
        run_pos = text.index("run")
        assert not _is_keyword_negated(text, "run", run_pos)
        # But "analyze" should still be negated
        analyze_pos = text.index("analyze")
        assert _is_keyword_negated(text, "analyze", analyze_pos)

    def test_positive_override_just(self):
        """Test ', just' overrides earlier negation."""
        text = "Don't analyze, just run the tests"
        run_pos = text.index("run")
        assert not _is_keyword_negated(text, "run", run_pos)

    def test_positive_override_instead(self):
        """Test ', instead' overrides earlier negation."""
        text = "Don't analyze; instead deploy the app"
        deploy_pos = text.index("deploy")
        assert not _is_keyword_negated(text, "deploy", deploy_pos)


class TestCategoryScoring:
    """Tests for category score calculation."""

    def test_simple_scoring(self):
        """Test basic score calculation."""
        matches = [
            KeywordMatch("create", "action", 0, False, 1.0),
            KeywordMatch("run", "action", 10, False, 0.9),
        ]
        score, count = _calculate_category_score(matches)
        assert score == 1.9
        assert count == 2

    def test_negated_reduces_score(self):
        """Test that negated keywords reduce score."""
        matches = [
            KeywordMatch("create", "action", 0, False, 1.0),
            KeywordMatch("analyze", "analysis", 10, True, 1.0),  # Negated
        ]
        score, count = _calculate_category_score(matches)
        assert score == 0.5  # 1.0 - (1.0 * 0.5)
        assert count == 1  # Only non-negated counted

    def test_all_negated_zero_score(self):
        """Test that all negated returns zero score."""
        matches = [
            KeywordMatch("analyze", "analysis", 0, True, 1.0),
        ]
        score, count = _calculate_category_score(matches)
        assert score == 0.0
        assert count == 0


class TestUnifiedTaskClassifier:
    """Tests for UnifiedTaskClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create a fresh classifier instance."""
        return UnifiedTaskClassifier(enable_semantic=False)

    def test_classify_action_task(self, classifier):
        """Test classification of action tasks."""
        result = classifier.classify("Run the tests")
        assert result.task_type == ClassifierTaskType.ACTION
        assert result.is_action_task is True
        assert result.needs_execution is True

    def test_classify_analysis_task(self, classifier):
        """Test classification of analysis tasks."""
        result = classifier.classify("Analyze the codebase for issues")
        assert result.task_type == ClassifierTaskType.ANALYSIS
        assert result.is_analysis_task is True
        assert result.temperature_adjustment > 0

    def test_classify_generation_task(self, classifier):
        """Test classification of generation tasks."""
        result = classifier.classify("Generate a function to sort arrays")
        assert result.task_type == ClassifierTaskType.GENERATION
        assert result.is_generation_task is True
        assert result.is_action_task is True  # Generation implies action

    def test_classify_search_task(self, classifier):
        """Test classification of search tasks."""
        result = classifier.classify("Find all usages of the function")
        assert result.task_type == ClassifierTaskType.SEARCH

    def test_classify_edit_task(self, classifier):
        """Test classification of edit tasks."""
        result = classifier.classify("Refactor the authentication module")
        assert result.task_type == ClassifierTaskType.EDIT

    def test_classify_default_task(self, classifier):
        """Test classification of ambiguous messages."""
        result = classifier.classify("Hello, how are you?")
        assert result.task_type == ClassifierTaskType.DEFAULT
        assert result.confidence < 0.5

    def test_negation_prevents_classification(self, classifier):
        """Test that negated keywords don't trigger classification."""
        result = classifier.classify("Don't analyze this, just show the files")
        # Should not be classified as analysis due to negation
        assert result.is_analysis_task is False or result.negated_keywords

    def test_position_based_precedence(self, classifier):
        """Test position-based priority when both analysis and action present."""
        # When action keyword ("create") appears AFTER analysis ("analyze"),
        # action takes precedence since it's the end goal
        result = classifier.classify("Analyze the code and create a report")
        # Both analysis and action keywords present
        assert result.is_analysis_task is True
        assert result.is_action_task is True
        # Action (generation) should win since "create" appears last
        assert result.task_type == ClassifierTaskType.GENERATION

        # When analysis appears last, it should win
        result2 = classifier.classify("Create initial draft and analyze it thoroughly")
        assert result2.is_analysis_task is True
        assert result2.is_action_task is True
        assert result2.task_type == ClassifierTaskType.ANALYSIS

    def test_confidence_scoring(self, classifier):
        """Test that confidence scores are reasonable."""
        # Strong signal
        result = classifier.classify("Analyze the entire codebase thoroughly")
        assert result.confidence > 0.5

        # Weak signal
        result = classifier.classify("Maybe look at something")
        assert result.confidence < 0.7

    def test_tool_budget_recommendation(self, classifier):
        """Test tool budget recommendations."""
        # Analysis gets high budget
        result = classifier.classify("Analyze the architecture")
        assert result.recommended_tool_budget >= 100

        # Generation gets low budget
        result = classifier.classify("Generate a simple function")
        assert result.recommended_tool_budget <= 20

    def test_runtime_intelligence_upgrades_low_confidence_classification(self):
        """RuntimeIntelligenceService should own low-confidence task-type decisions."""
        runtime_intelligence = MagicMock()
        runtime_intelligence.decide_sync.return_value = DecisionResult(
            decision_type=DecisionType.TASK_TYPE_CLASSIFICATION,
            result=TaskTypeDecision(
                task_type=TaskCategoryType.ANALYSIS,
                confidence=0.92,
                deliverables=[],
            ),
            source="llm",
            confidence=0.92,
        )
        classifier = UnifiedTaskClassifier(
            enable_semantic=False,
            runtime_intelligence=runtime_intelligence,
        )

        result = classifier.classify("Maybe look at something", use_cache=False)

        assert result.task_type == ClassifierTaskType.ANALYSIS
        assert result.source == "llm"
        runtime_intelligence.decide_sync.assert_called_once()


class TestContextualClassification:
    """Tests for context-aware classification."""

    @pytest.fixture
    def classifier(self):
        return UnifiedTaskClassifier(enable_semantic=False)

    def test_context_boost_consistent_type(self, classifier):
        """Test that context boosts confidence for consistent task types."""
        history = [
            {"role": "user", "content": "Analyze the auth module"},
            {"role": "assistant", "content": "I analyzed it..."},
            {"role": "user", "content": "Now analyze the database module"},
        ]
        result = classifier.classify_with_context("Continue the review", history)
        # Should get context boost since history is analysis-heavy
        assert result.context_boost >= 0 or result.source == "context"

    def test_context_switches_default_to_dominant(self, classifier):
        """Test that context can switch DEFAULT to dominant type."""
        history = [
            {"role": "user", "content": "Analyze the codebase"},
            {"role": "assistant", "content": "Analysis..."},
            {"role": "user", "content": "Analyze the tests"},
            {"role": "assistant", "content": "Test analysis..."},
            {"role": "user", "content": "Analyze the docs"},
        ]
        # Ambiguous message that would be DEFAULT
        result = classifier.classify_with_context("Continue", history)
        # With strong analysis context, should lean toward analysis
        assert result.context_signals or result.task_type != ClassifierTaskType.DEFAULT

    def test_no_context_no_boost(self, classifier):
        """Test that empty history doesn't affect classification."""
        result1 = classifier.classify("Analyze the code")
        result2 = classifier.classify_with_context("Analyze the code", [])
        assert result1.confidence == result2.confidence


class TestLegacyCompatibility:
    """Tests for backward compatibility with _classify_task_keywords."""

    @pytest.fixture
    def classifier(self):
        return UnifiedTaskClassifier(enable_semantic=False)

    def test_to_legacy_dict_action(self, classifier):
        """Test legacy dict format for action tasks."""
        result = classifier.classify("Run the tests")
        legacy = result.to_legacy_dict()
        assert "is_action_task" in legacy
        assert "is_analysis_task" in legacy
        assert "needs_execution" in legacy
        assert "coarse_task_type" in legacy
        assert legacy["is_action_task"] is True
        assert legacy["coarse_task_type"] == "action"

    def test_to_legacy_dict_analysis(self, classifier):
        """Test legacy dict format for analysis tasks."""
        result = classifier.classify("Analyze the code")
        legacy = result.to_legacy_dict()
        assert legacy["is_analysis_task"] is True
        assert legacy["coarse_task_type"] == "analysis"

    def test_to_legacy_dict_default(self, classifier):
        """Test legacy dict format for default tasks."""
        result = classifier.classify("Hello")
        legacy = result.to_legacy_dict()
        assert legacy["coarse_task_type"] == "default"

    def test_to_legacy_dict_includes_new_fields(self, classifier):
        """Test that legacy dict includes new fields too."""
        result = classifier.classify("Analyze the code")
        legacy = result.to_legacy_dict()
        assert "confidence" in legacy
        assert "source" in legacy
        assert "task_type" in legacy


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_unified_classifier_singleton(self):
        """Test that get_unified_classifier returns same instance."""
        c1 = get_unified_classifier()
        c2 = get_unified_classifier()
        assert c1 is c2

    def test_classify_task_convenience(self):
        """Test the classify_task convenience function."""
        result = classify_task("Analyze the code")
        assert isinstance(result, ClassificationResult)
        assert result.task_type == ClassifierTaskType.ANALYSIS


class TestCaching:
    """Tests for classification caching functionality."""

    @pytest.fixture
    def classifier(self):
        return UnifiedTaskClassifier(enable_semantic=False)

    def test_cache_hit(self, classifier):
        """Test that repeated classifications are cached."""
        message = "Analyze the code thoroughly"

        # First call - cache miss
        result1 = classifier.classify(message)
        stats1 = classifier.get_cache_stats()
        assert stats1["cache_misses"] == 1
        assert stats1["cache_hits"] == 0

        # Second call - cache hit
        result2 = classifier.classify(message)
        stats2 = classifier.get_cache_stats()
        assert stats2["cache_hits"] == 1
        assert stats2["cache_misses"] == 1

        # Results should be the same
        assert result1.task_type == result2.task_type
        assert result1.confidence == result2.confidence

    def test_cache_disabled(self, classifier):
        """Test that cache can be bypassed."""
        message = "Run the tests"

        # First call with cache
        classifier.classify(message)

        # Second call with cache disabled - doesn't touch cache
        classifier.classify(message, use_cache=False)
        stats = classifier.get_cache_stats()

        # Should have 0 hits and 1 miss (disabled call doesn't touch cache)
        assert stats["cache_misses"] == 1
        assert stats["cache_hits"] == 0

        # Third call with cache - should hit cache
        classifier.classify(message)
        stats = classifier.get_cache_stats()
        assert stats["cache_hits"] == 1

    def test_cache_clear(self, classifier):
        """Test cache clearing."""
        classifier.classify("Analyze the code")
        classifier.classify("Run the tests")

        stats = classifier.get_cache_stats()
        assert stats["cache_size"] == 2

        classifier.clear_cache()
        stats = classifier.get_cache_stats()
        assert stats["cache_size"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0

    def test_cache_stats(self, classifier):
        """Test cache statistics reporting."""
        stats = classifier.get_cache_stats()
        assert "cache_size" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "hit_rate" in stats
        assert "max_size" in stats
        assert "ttl_seconds" in stats

    def test_different_messages_not_cached_together(self, classifier):
        """Test that different messages have separate cache entries."""
        classifier.classify("Analyze the code")
        classifier.classify("Run the tests")

        stats = classifier.get_cache_stats()
        assert stats["cache_size"] == 2
        assert stats["cache_misses"] == 2


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def classifier(self):
        return UnifiedTaskClassifier(enable_semantic=False)

    def test_empty_message(self, classifier):
        """Test classification of empty message."""
        result = classifier.classify("")
        assert result.task_type == ClassifierTaskType.DEFAULT
        assert result.confidence < 0.5

    def test_very_long_message(self, classifier):
        """Test classification of very long message."""
        long_msg = "Analyze " + "the code " * 1000
        result = classifier.classify(long_msg)
        assert result.task_type == ClassifierTaskType.ANALYSIS

    def test_special_characters(self, classifier):
        """Test classification with special characters."""
        result = classifier.classify("Analyze @#$%^& the code!")
        assert result.task_type == ClassifierTaskType.ANALYSIS

    def test_multiple_keywords_same_category(self, classifier):
        """Test message with multiple keywords from same category."""
        result = classifier.classify("Analyze, review, and audit the codebase")
        assert result.task_type == ClassifierTaskType.ANALYSIS
        assert len(result.matched_keywords) >= 3

    def test_conflicting_keywords(self, classifier):
        """Test message with conflicting keyword categories."""
        result = classifier.classify("Don't analyze but do run the tests")
        # "analyze" is negated, "run" is not
        assert result.needs_execution is True

    def test_question_patterns(self, classifier):
        """Test analysis classification via question patterns."""
        result = classifier.classify("How does the authentication work?")
        assert result.task_type == ClassifierTaskType.ANALYSIS

    def test_multi_word_keyword(self, classifier):
        """Test matching of multi-word keywords."""
        result = classifier.classify("Do a full analysis of the entire codebase")
        assert "entire codebase" in [
            m.keyword for m in result.matched_keywords
        ] or "full analysis" in [m.keyword for m in result.matched_keywords]
