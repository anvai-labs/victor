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

"""Tests for QuestionTypeClassifier."""

import pytest

from victor.storage.embeddings.question_classifier import (
    QuestionType,
    QuestionTypeClassifier,
    QuestionClassificationResult,
    classify_question,
    should_auto_continue,
)


@pytest.fixture
def classifier():
    """Create a fresh classifier instance."""
    QuestionTypeClassifier.reset_instance()
    return QuestionTypeClassifier.get_instance()


class TestQuestionClassificationResult:
    """Tests for QuestionClassificationResult dataclass."""

    def test_should_auto_continue_for_rhetorical(self):
        """Rhetorical questions with high confidence should auto-continue."""
        result = QuestionClassificationResult(
            question_type=QuestionType.RHETORICAL,
            confidence=0.85,
            matched_pattern="test",
        )
        assert result.should_auto_continue is True

    def test_should_auto_continue_for_continuation(self):
        """Continuation questions with high confidence should auto-continue."""
        result = QuestionClassificationResult(
            question_type=QuestionType.CONTINUATION,
            confidence=0.90,
            matched_pattern="test",
        )
        assert result.should_auto_continue is True

    def test_should_not_auto_continue_low_confidence(self):
        """Low confidence should not auto-continue even for rhetorical."""
        result = QuestionClassificationResult(
            question_type=QuestionType.RHETORICAL,
            confidence=0.50,
            matched_pattern="test",
        )
        assert result.should_auto_continue is False

    def test_should_not_auto_continue_for_clarification(self):
        """Clarification questions should not auto-continue."""
        result = QuestionClassificationResult(
            question_type=QuestionType.CLARIFICATION,
            confidence=0.90,
            matched_pattern="test",
        )
        assert result.should_auto_continue is False

    def test_should_not_auto_continue_for_information(self):
        """Information questions should not auto-continue."""
        result = QuestionClassificationResult(
            question_type=QuestionType.INFORMATION,
            confidence=0.95,
            matched_pattern="test",
        )
        assert result.should_auto_continue is False


class TestQuestionTypeClassifier:
    """Tests for QuestionTypeClassifier class."""

    def test_singleton_pattern(self, classifier):
        """Test that singleton returns same instance."""
        instance1 = QuestionTypeClassifier.get_instance()
        instance2 = QuestionTypeClassifier.get_instance()
        assert instance1 is instance2

    def test_reset_instance(self):
        """Test that reset_instance creates new instance."""
        instance1 = QuestionTypeClassifier.get_instance()
        QuestionTypeClassifier.reset_instance()
        instance2 = QuestionTypeClassifier.get_instance()
        assert instance1 is not instance2

    # Continuation questions
    def test_classify_should_i_continue(self, classifier):
        """Should I continue is a continuation question."""
        result = classifier.classify("Should I continue with the implementation?")
        assert result.question_type == QuestionType.CONTINUATION
        assert result.confidence >= 0.9
        assert result.should_auto_continue is True

    def test_classify_shall_i_proceed(self, classifier):
        """Shall I proceed is a continuation question."""
        result = classifier.classify("Shall I proceed with the next step?")
        assert result.question_type == QuestionType.CONTINUATION
        assert result.confidence >= 0.9
        assert result.should_auto_continue is True

    def test_classify_would_you_like_me_to_continue(self, classifier):
        """Would you like me to continue is a continuation question."""
        result = classifier.classify("Would you like me to continue with this approach?")
        assert result.question_type == QuestionType.CONTINUATION
        assert result.confidence >= 0.9
        assert result.should_auto_continue is True

    def test_classify_do_you_want_me_to_proceed(self, classifier):
        """Do you want me to proceed is a continuation question."""
        result = classifier.classify("Do you want me to proceed with the changes?")
        assert result.question_type == QuestionType.CONTINUATION
        assert result.confidence >= 0.8
        assert result.should_auto_continue is True

    def test_classify_ready_to_continue(self, classifier):
        """Ready to continue is a continuation question."""
        result = classifier.classify("Are you ready to continue?")
        assert result.question_type == QuestionType.CONTINUATION
        assert result.confidence >= 0.8
        assert result.should_auto_continue is True

    # Rhetorical questions
    def test_classify_does_this_look_good(self, classifier):
        """Does this look good is rhetorical."""
        result = classifier.classify("I've added the function. Does this look good?")
        assert result.question_type == QuestionType.RHETORICAL
        assert result.confidence >= 0.8
        assert result.should_auto_continue is True

    def test_classify_make_sense(self, classifier):
        """Make sense is rhetorical."""
        result = classifier.classify("The implementation uses a factory pattern. Make sense?")
        assert result.question_type == QuestionType.RHETORICAL
        assert result.confidence >= 0.8
        assert result.should_auto_continue is True

    def test_classify_is_this_what_you_wanted(self, classifier):
        """Is this what you wanted is rhetorical."""
        result = classifier.classify("Here's the updated code. Is this what you had in mind?")
        assert result.question_type == QuestionType.RHETORICAL
        assert result.confidence >= 0.8
        assert result.should_auto_continue is True

    # Clarification questions - need user input
    def test_classify_which_option(self, classifier):
        """Which option needs user input."""
        result = classifier.classify("Which option do you prefer for the database?")
        assert result.question_type == QuestionType.CLARIFICATION
        assert result.confidence >= 0.8
        assert result.should_auto_continue is False

    def test_classify_what_should_name(self, classifier):
        """What should we name needs user input."""
        result = classifier.classify("What should I name the new function?")
        assert result.question_type == QuestionType.CLARIFICATION
        assert result.confidence >= 0.8
        assert result.should_auto_continue is False

    def test_classify_where_should_put(self, classifier):
        """Where should I put needs user input."""
        result = classifier.classify("Where should I put the configuration file?")
        assert result.question_type == QuestionType.CLARIFICATION
        assert result.confidence >= 0.8
        assert result.should_auto_continue is False

    def test_classify_how_would_you_like(self, classifier):
        """How would you like needs user input."""
        result = classifier.classify("How would you like me to handle errors?")
        assert result.question_type == QuestionType.CLARIFICATION
        assert result.confidence >= 0.8
        assert result.should_auto_continue is False

    # Information questions - need specific user data
    def test_classify_api_key(self, classifier):
        """API key question needs user input."""
        result = classifier.classify("What's your API key for the service?")
        assert result.question_type == QuestionType.INFORMATION
        assert result.confidence >= 0.9
        assert result.should_auto_continue is False

    def test_classify_credentials(self, classifier):
        """Credentials question needs user input."""
        result = classifier.classify("What is the username and password for the database?")
        assert result.question_type == QuestionType.INFORMATION
        assert result.confidence >= 0.9
        assert result.should_auto_continue is False

    def test_classify_database_url(self, classifier):
        """Database URL question needs user input."""
        result = classifier.classify("What's the database connection URL?")
        assert result.question_type == QuestionType.INFORMATION
        assert result.confidence >= 0.8
        assert result.should_auto_continue is False

    # Edge cases
    def test_classify_empty_text(self, classifier):
        """Empty text should return unknown."""
        result = classifier.classify("")
        assert result.question_type == QuestionType.UNKNOWN
        assert result.confidence == 0.0

    def test_classify_no_question_mark(self, classifier):
        """Text without question mark should return unknown."""
        result = classifier.classify("I will continue with the implementation.")
        assert result.question_type == QuestionType.UNKNOWN
        assert result.matched_pattern == "no_question_mark"

    def test_classify_unrecognized_question(self, classifier):
        """Unrecognized question defaults to clarification (conservative)."""
        result = classifier.classify("What do you think about this approach?")
        assert result.question_type == QuestionType.CLARIFICATION
        assert result.confidence == 0.5  # Default low confidence
        assert result.should_auto_continue is False


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_classify_question_convenience(self):
        """Test classify_question convenience function."""
        QuestionTypeClassifier.reset_instance()
        result = classify_question("Should I continue?")
        assert result.question_type == QuestionType.CONTINUATION

    def test_should_auto_continue_convenience(self):
        """Test should_auto_continue convenience function."""
        QuestionTypeClassifier.reset_instance()
        assert should_auto_continue("Should I continue with this?") is True
        assert should_auto_continue("What's your API key?") is False
        assert should_auto_continue("Which file should I use?") is False
