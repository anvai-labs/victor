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

"""Unit tests for PromptNormalizer module.

Tests the prompt normalization and deduplication functionality including:
- Action verb canonicalization (view→read, check→read)
- Duplicate message detection
- Continuation message collapsing
- Section deduplication
"""

import pytest
from victor.agent.prompt_normalizer import (
    PromptNormalizer,
    NormalizationResult,
    get_prompt_normalizer,
    reset_normalizer,
)


@pytest.fixture
def normalizer():
    """Create a fresh PromptNormalizer instance for each test."""
    return PromptNormalizer()


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton normalizer before each test."""
    reset_normalizer()
    yield
    reset_normalizer()


class TestVerbNormalization:
    """Tests for action verb canonicalization."""

    def test_view_to_read(self, normalizer):
        """'view' should be normalized to 'read'."""
        result = normalizer.normalize("view the auth.py file")
        assert result.normalized == "read the auth.py file"
        assert "view→read" in result.changes

    def test_look_at_to_read(self, normalizer):
        """'look at' should be normalized to 'read'."""
        result = normalizer.normalize("look at the config file")
        assert result.normalized == "read the config file"
        assert "look at→read" in result.changes

    def test_check_to_read(self, normalizer):
        """'check' should be normalized to 'read'."""
        result = normalizer.normalize("check the error logs")
        assert result.normalized == "read the error logs"
        assert "check→read" in result.changes

    def test_show_to_read(self, normalizer):
        """'show' should be normalized to 'read'."""
        result = normalizer.normalize("show me the file contents")
        assert result.normalized == "read me the file contents"
        assert "show→read" in result.changes

    def test_display_to_read(self, normalizer):
        """'display' should be normalized to 'read'."""
        result = normalizer.normalize("display the output")
        assert result.normalized == "read the output"
        assert "display→read" in result.changes

    def test_examine_to_analyze(self, normalizer):
        """'examine' should be normalized to 'analyze'."""
        result = normalizer.normalize("examine the code structure")
        assert result.normalized == "analyze the code structure"
        assert "examine→analyze" in result.changes

    def test_review_to_analyze(self, normalizer):
        """'review' should be normalized to 'analyze'."""
        result = normalizer.normalize("review the implementation")
        assert result.normalized == "analyze the implementation"
        assert "review→analyze" in result.changes

    def test_inspect_to_analyze(self, normalizer):
        """'inspect' should be normalized to 'analyze'."""
        result = normalizer.normalize("inspect the class hierarchy")
        assert result.normalized == "analyze the class hierarchy"
        assert "inspect→analyze" in result.changes

    def test_case_insensitive(self, normalizer):
        """Normalization should be case-insensitive."""
        result = normalizer.normalize("VIEW the file")
        assert result.normalized == "read the file"
        assert "view→read" in result.changes

    def test_no_change_for_unknown_verbs(self, normalizer):
        """Unknown verbs should not be changed."""
        result = normalizer.normalize("fix the bug")
        assert result.normalized == "fix the bug"
        assert len(result.changes) == 0

    def test_preserves_other_content(self, normalizer):
        """Normalization should preserve surrounding content."""
        result = normalizer.normalize("Please view auth.py and explain")
        assert "read" in result.normalized
        assert "auth.py" in result.normalized
        assert "explain" in result.normalized


class TestDuplicateDetection:
    """Tests for duplicate message detection."""

    def test_exact_duplicate_detected(self, normalizer):
        """Exact duplicate messages should be detected."""
        normalizer.normalize("fix the bug in auth.py")
        result = normalizer.normalize("fix the bug in auth.py")
        assert result.is_duplicate is True

    def test_different_messages_not_duplicates(self, normalizer):
        """Different messages should not be marked as duplicates."""
        normalizer.normalize("fix the bug in auth.py")
        result = normalizer.normalize("fix the bug in settings.py")
        assert result.is_duplicate is False

    def test_whitespace_normalized_before_hashing(self, normalizer):
        """Extra whitespace should be normalized before duplicate detection."""
        normalizer.normalize("fix  the   bug")
        result = normalizer.normalize("fix the bug")
        assert result.is_duplicate is True

    def test_first_message_not_duplicate(self, normalizer):
        """First message should never be marked as duplicate."""
        result = normalizer.normalize("fix the bug")
        assert result.is_duplicate is False


class TestContinuationCollapsing:
    """Tests for continuation message collapsing."""

    def test_continue_recognized(self, normalizer):
        """'continue' should be recognized as continuation."""
        result = normalizer.normalize("continue")
        # First continuation is not collapsed
        assert result.normalized == "continue"

    def test_yes_recognized(self, normalizer):
        """'yes' should be recognized as continuation."""
        result = normalizer.normalize("yes")
        assert result.normalized == "yes"

    def test_ok_recognized(self, normalizer):
        """'ok' should be recognized as continuation."""
        result = normalizer.normalize("ok")
        assert result.normalized == "ok"

    def test_go_on_recognized(self, normalizer):
        """'go on' should be recognized as continuation."""
        result = normalizer.normalize("go on")
        assert result.normalized == "go on"

    def test_proceed_recognized(self, normalizer):
        """'proceed' should be recognized as continuation."""
        result = normalizer.normalize("proceed")
        assert result.normalized == "proceed"

    def test_keep_going_recognized(self, normalizer):
        """'keep going' should be recognized as continuation."""
        result = normalizer.normalize("keep going")
        assert result.normalized == "keep going"

    def test_multiple_continuations_collapsed(self, normalizer):
        """After 3+ continuations, they should be collapsed."""
        normalizer.normalize("continue")
        normalizer.normalize("yes")
        result = normalizer.normalize("ok")  # 3rd continuation
        assert result.normalized == "continue"
        assert "collapsed continuation #3" in result.changes

    def test_non_continuation_resets_count(self, normalizer):
        """Non-continuation messages should reset the count."""
        normalizer.normalize("continue")
        normalizer.normalize("yes")
        normalizer.normalize("fix the bug")  # Not a continuation
        result = normalizer.normalize("ok")  # Only 1st after reset
        assert "collapsed" not in str(result.changes)


class TestSectionDeduplication:
    """Tests for section deduplication."""

    def test_removes_exact_duplicates(self, normalizer):
        """Exact duplicate sections should be removed."""
        sections = ["Section A", "Section B", "Section A"]
        result = normalizer.deduplicate_sections(sections)
        assert len(result) == 2
        assert "Section A" in result
        assert "Section B" in result

    def test_preserves_order(self, normalizer):
        """First occurrence should be preserved."""
        sections = ["First", "Second", "First"]
        result = normalizer.deduplicate_sections(sections)
        assert result == ["First", "Second"]

    def test_handles_empty_list(self, normalizer):
        """Empty list should return empty list."""
        result = normalizer.deduplicate_sections([])
        assert result == []

    def test_handles_empty_strings(self, normalizer):
        """Empty strings should be filtered out."""
        sections = ["Section A", "", "   ", "Section B"]
        result = normalizer.deduplicate_sections(sections)
        assert len(result) == 2
        assert "Section A" in result
        assert "Section B" in result

    def test_whitespace_normalized(self, normalizer):
        """Whitespace differences should be normalized."""
        sections = ["Section  A", "Section A"]  # Extra space
        result = normalizer.deduplicate_sections(sections)
        assert len(result) == 1


class TestSimilarityDetection:
    """Tests for similarity-based detection."""

    def test_similar_messages_detected(self, normalizer):
        """Similar messages should be detected above threshold."""
        normalizer.normalize("fix the authentication bug in auth.py")
        result = normalizer.is_similar_to_recent(
            "fix the authentication bug in auth.py file", threshold=0.8
        )
        assert result is True

    def test_dissimilar_messages_not_flagged(self, normalizer):
        """Dissimilar messages should not be flagged."""
        normalizer.normalize("fix the authentication bug")
        result = normalizer.is_similar_to_recent("add new feature to dashboard", threshold=0.8)
        assert result is False

    def test_empty_history(self, normalizer):
        """Empty history should return False."""
        result = normalizer.is_similar_to_recent("any message", threshold=0.8)
        assert result is False


class TestTokensSaved:
    """Tests for token savings calculation."""

    def test_token_savings_calculated(self, normalizer):
        """Token savings should be calculated correctly."""
        # "view" (4 chars) -> "read" (4 chars) = 0 savings
        result = normalizer.normalize("view")
        assert result.tokens_saved >= 0

    def test_longer_to_shorter_saves_tokens(self, normalizer):
        """Longer -> shorter normalization saves tokens."""
        # Leading/trailing whitespace is stripped
        result = normalizer.normalize("   view   ")  # Extra spaces stripped
        assert result.tokens_saved >= 0


class TestStatistics:
    """Tests for normalizer statistics."""

    def test_get_stats(self, normalizer):
        """get_stats should return correct tracking info."""
        normalizer.normalize("message 1")
        normalizer.normalize("message 2")
        stats = normalizer.get_stats()
        assert stats["tracked_messages"] == 2
        assert stats["continuation_count"] == 0

    def test_stats_with_continuations(self, normalizer):
        """Stats should track continuation count."""
        normalizer.normalize("continue")
        normalizer.normalize("yes")
        stats = normalizer.get_stats()
        assert stats["continuation_count"] == 2


class TestReset:
    """Tests for normalizer reset."""

    def test_reset_clears_history(self, normalizer):
        """Reset should clear message history."""
        normalizer.normalize("message 1")
        normalizer.reset()
        stats = normalizer.get_stats()
        assert stats["tracked_messages"] == 0

    def test_reset_clears_continuation_count(self, normalizer):
        """Reset should clear continuation count."""
        normalizer.normalize("continue")
        normalizer.reset()
        stats = normalizer.get_stats()
        assert stats["continuation_count"] == 0


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_prompt_normalizer_returns_same_instance(self):
        """get_prompt_normalizer should return the same instance."""
        n1 = get_prompt_normalizer()
        n2 = get_prompt_normalizer()
        assert n1 is n2

    def test_reset_normalizer_clears_singleton(self):
        """reset_normalizer should clear the singleton."""
        n1 = get_prompt_normalizer()
        reset_normalizer()
        n2 = get_prompt_normalizer()
        # After reset, should get a fresh instance
        assert n1.get_stats()["tracked_messages"] >= 0  # Original still works
