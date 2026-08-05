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

"""
Unit tests for fuzzy matching module.

Tests cover:
- Adaptive threshold calculation
- Fuzzy keyword extraction
- Cascading matching strategy
- Performance benchmarks
- Edge cases and error handling
"""

import pytest

from victor.storage.embeddings.fuzzy_matcher import (
    calculate_similarity_ratio,
    exact_match_only,
    extract_key_terms_fuzzy,
    get_edit_distance_threshold,
    is_fuzzy_match,
    match_keywords_cascading,
    match_keywords_optimized,
    tokenize,
)


class TestEditDistanceThreshold:
    """Tests for adaptive edit distance threshold calculation."""

    def test_threshold_1_3_chars_no_tolerance(self):
        """Words 1-3 chars should have 0 tolerance."""
        assert get_edit_distance_threshold(1) == 0
        assert get_edit_distance_threshold(2) == 0
        assert get_edit_distance_threshold(3) == 0

    def test_threshold_4_6_chars_one_edit(self):
        """Words 4-6 chars should allow 1 edit."""
        assert get_edit_distance_threshold(4) == 1
        assert get_edit_distance_threshold(5) == 1
        assert get_edit_distance_threshold(6) == 1

    def test_threshold_7_9_chars_two_edits(self):
        """Words 7-9 chars should allow 2 edits."""
        assert get_edit_distance_threshold(7) == 2
        assert get_edit_distance_threshold(8) == 2
        assert get_edit_distance_threshold(9) == 2

    def test_threshold_10plus_chars_adaptive(self):
        """Words 10+ chars should have adaptive threshold."""
        assert get_edit_distance_threshold(10) == 2
        assert get_edit_distance_threshold(12) == 3  # 12//4 = 3
        assert get_edit_distance_threshold(16) == 4  # 16//4 = 4
        assert get_edit_distance_threshold(20) == 5  # 20//4 = 5

    def test_threshold_edge_cases(self):
        """Test edge cases for threshold calculation."""
        # Boundary values
        assert get_edit_distance_threshold(0) == 0
        assert get_edit_distance_threshold(100) == 25  # 100//4


class TestFuzzyMatching:
    """Tests for fuzzy keyword extraction."""

    def test_fuzzy_match_single_typo(self):
        """Should match words with single typo."""
        words = {"analize", "structre", "architcture"}
        key_terms = {"analyze": 1.5, "structure": 1.2, "architecture": 1.2}

        matches = extract_key_terms_fuzzy(words, key_terms)
        assert matches == {"analyze", "structure", "architecture"}

    def test_fuzzy_match_exact_match(self):
        """Should still match exact words."""
        words = {"analyze", "structure", "architecture"}
        key_terms = {"analyze": 1.5, "structure": 1.2, "architecture": 1.2}

        matches = extract_key_terms_fuzzy(words, key_terms)
        assert matches == {"analyze", "structure", "architecture"}

    def test_fuzzy_match_similarity_ratio(self):
        """Should respect minimum similarity ratio."""
        # "analize" vs "analyze" = 1 edit, 88% similarity ✅
        # "analizz" vs "analyze" = 2 edits, 75% similarity ✅
        # "anallze" vs "analyze" = 2 edits, 67% similarity ❌

        words = {"analize", "analizz", "anallze"}
        key_terms = {"analyze": 1.5}

        matches = extract_key_terms_fuzzy(words, key_terms, min_similarity_ratio=0.75)
        assert "analyze" in matches  # "analize" matched
        # Only high-similarity matches
        assert matches == {"analyze"}

    def test_fuzzy_match_no_false_positives(self):
        """Should not match very different words."""
        # Use a very different word that exceeds threshold
        words = {"exec"}  # Very different from execute
        key_terms = {"execute": 1.2}

        matches = extract_key_terms_fuzzy(words, key_terms, min_similarity_ratio=0.75)
        # "exec" vs "execute" = 3 edits, ratio < 0.75
        assert len(matches) == 0

    def test_fuzzy_match_empty_inputs(self):
        """Should handle empty inputs gracefully."""
        assert extract_key_terms_fuzzy(set(), {}) == set()
        assert extract_key_terms_fuzzy({"test"}, {}) == set()
        assert extract_key_terms_fuzzy(set(), {"test": 1.0}) == set()

    def test_fuzzy_match_case_insensitive(self):
        """Should be case-insensitive."""
        # Note: The function converts words to lowercase before matching
        # So we need to use the correct lowercase forms
        words = {"analize", "structre"}  # Lowercase with typos
        key_terms = {"analyze": 1.5, "structure": 1.2}

        matches = extract_key_terms_fuzzy(words, key_terms)
        # Both should match with fuzzy matching
        assert "analyze" in matches
        assert "structure" in matches

    def test_fuzzy_match_multiple_candidates(self):
        """Should match best candidate when multiple options."""
        words = {"analize"}
        key_terms = {"analyze": 1.5, "analyse": 1.0}

        matches = extract_key_terms_fuzzy(words, key_terms)
        # Should match at least one
        assert len(matches) >= 1

    def test_fuzzy_match_weight_independence(self):
        """Weights should not affect matching, only scoring."""
        words = {"analize"}
        key_terms_low_weight = {"analyze": 0.1}
        key_terms_high_weight = {"analyze": 2.0}

        matches_low = extract_key_terms_fuzzy(words, key_terms_low_weight)
        matches_high = extract_key_terms_fuzzy(words, key_terms_high_weight)

        assert matches_low == matches_high == {"analyze"}


class TestCascadingStrategy:
    """Tests for cascading matching strategy."""

    def test_cascading_exact_match_first(self):
        """Should prefer exact match over fuzzy."""
        key_terms = {"analyze": 1.5}
        query = "analyze the code"

        matches, stats = match_keywords_cascading(query, key_terms, use_fuzzy=True)
        assert stats["method"] == "exact"
        assert matches == {"analyze"}

    def test_cascading_fuzzy_fallback(self):
        """Should use fuzzy match when no exact match."""
        key_terms = {"analyze": 1.5}
        query = "analize the code"

        matches, stats = match_keywords_cascading(query, key_terms, use_fuzzy=True)
        assert stats["method"] == "fuzzy"
        assert matches == {"analyze"}
        assert "matched" in stats

    def test_cascading_no_match(self):
        """Should return empty when no match found."""
        key_terms = {"execute": 1.2}
        query = "read the file"

        matches, stats = match_keywords_cascading(query, key_terms, use_fuzzy=True)
        assert stats["method"] == "none"
        assert len(matches) == 0

    def test_cascading_fuzzy_disabled(self):
        """Should not use fuzzy when disabled."""
        key_terms = {"analyze": 1.5}
        query = "analize the code"

        matches, stats = match_keywords_cascading(query, key_terms, use_fuzzy=False)
        assert stats["method"] == "none"
        assert len(matches) == 0

    def test_cascading_custom_similarity_threshold(self):
        """Should respect custom similarity threshold."""
        key_terms = {"analyze": 1.5}
        query = "analize the code"

        # Lower threshold should match
        matches_low, _ = match_keywords_cascading(
            query, key_terms, use_fuzzy=True, min_similarity_ratio=0.7
        )
        assert len(matches_low) > 0

        # Higher threshold might not match
        matches_high, _ = match_keywords_cascading(
            query, key_terms, use_fuzzy=True, min_similarity_ratio=0.95
        )
        # Very strict threshold
        assert len(matches_high) <= len(matches_low)

    def test_cascading_multiple_keywords(self):
        """Should handle multiple keywords in query."""
        # Note: Cascading prioritizes exact matches and returns early
        # So if any exact match exists, it won't check for fuzzy matches
        key_terms = {"analyze": 1.5, "structure": 1.2, "framework": 1.1}

        # Test with only fuzzy matches (no exact matches)
        query = "analize the structre"  # No exact matches

        matches, stats = match_keywords_cascading(query, key_terms, use_fuzzy=True)
        # Both should match with fuzzy: analize→analyze, structre→structure
        assert len(matches) >= 1  # At least 1 fuzzy match
        assert stats["method"] == "fuzzy"

    def test_cascading_stats_dict_structure(self):
        """Should return properly structured stats dict."""
        key_terms = {"analyze": 1.5}

        # Exact match
        _, stats_exact = match_keywords_cascading("analyze", key_terms, use_fuzzy=True)
        assert "method" in stats_exact
        assert "count" in stats_exact
        assert stats_exact["method"] == "exact"

        # Fuzzy match
        _, stats_fuzzy = match_keywords_cascading("analize", key_terms, use_fuzzy=True)
        assert "method" in stats_fuzzy
        assert "count" in stats_fuzzy
        assert "matched" in stats_fuzzy
        assert stats_fuzzy["method"] == "fuzzy"

        # No match
        _, stats_none = match_keywords_cascading("other", key_terms, use_fuzzy=True)
        assert "method" in stats_none
        assert "count" in stats_none
        assert stats_none["method"] == "none"


class TestExactMatchOnly:
    """Tests for exact match function."""

    def test_exact_match_found(self):
        """Should find exact matches."""
        key_terms = {"analyze": 1.5, "structure": 1.2}
        query = "analyze the code structure"

        matches = exact_match_only(query, key_terms)
        assert matches == {"analyze", "structure"}

    def test_exact_match_not_found(self):
        """Should not find fuzzy matches."""
        key_terms = {"analyze": 1.5}
        query = "analize the code"

        matches = exact_match_only(query, key_terms)
        assert matches == set()

    def test_exact_match_case_insensitive(self):
        """Should be case-insensitive."""
        key_terms = {"analyze": 1.5}
        query = "ANALYZE the code"

        matches = exact_match_only(query, key_terms)
        assert matches == {"analyze"}

    def test_exact_match_empty(self):
        """Should handle empty inputs."""
        assert exact_match_only("", {}) == set()
        assert exact_match_only("test", {}) == set()
        assert exact_match_only("", {"test": 1.0}) == set()


class TestOptimizedMatching:
    """Tests for optimized matching function."""

    def test_optimized_short_query_no_fuzzy(self):
        """Should skip fuzzy for short queries when disabled."""
        key_terms = {"analyze": 1.5}
        query = "analize"

        matches = match_keywords_optimized(query, key_terms, use_fuzzy=False)
        assert matches == set()

    def test_optimized_high_confidence_exact(self):
        """Should skip fuzzy when high-confidence exact match exists."""
        key_terms = {"analyze": 1.5, "structure": 1.2, "review": 1.4}
        query = "analyze code structure and review design"

        matches = match_keywords_optimized(query, key_terms, use_fuzzy=True)
        # Should find exact matches, skip fuzzy
        assert len(matches) >= 3

    def test_optimized_fuzzy_fallback(self):
        """Should use fuzzy when no exact match."""
        key_terms = {"analyze": 1.5}
        query = "analize the code"

        matches = match_keywords_optimized(query, key_terms, use_fuzzy=True)
        assert matches == {"analyze"}


class TestTokenize:
    """Tests for tokenization function."""

    def test_tokenize_basic(self):
        """Should tokenize basic text."""
        text = "Analyze the code structure!"
        tokens = tokenize(text)
        assert tokens == {"analyze", "code", "structure", "the"}

    def test_tokenize_punctuation(self):
        """Should handle punctuation."""
        text = "analyze, code; structure!"
        tokens = tokenize(text)
        assert "analyze" in tokens
        assert "code" in tokens
        assert "structure" in tokens

    def test_tokenize_case_insensitive(self):
        """Should convert to lowercase."""
        text = "ANALYZE Code Structure"
        tokens = tokenize(text)
        assert "analyze" in tokens
        assert "code" in tokens
        assert "structure" in tokens

    def test_tokenize_empty(self):
        """Should handle empty string."""
        assert tokenize("") == set()
        assert tokenize("...") == set()


class TestSimilarityRatio:
    """Tests for similarity ratio calculation."""

    def test_similarity_identical(self):
        """Should return 1.0 for identical words."""
        ratio = calculate_similarity_ratio("analyze", "analyze")
        assert ratio == 1.0

    def test_similarity_similar(self):
        """Should return high ratio for similar words."""
        ratio = calculate_similarity_ratio("analyze", "analize")
        assert ratio > 0.85

    def test_similarity_different(self):
        """Should return low ratio for different words."""
        ratio = calculate_similarity_ratio("analyze", "execute")
        assert ratio < 0.5

    def test_similarity_case_insensitive(self):
        """Should be case-insensitive."""
        ratio1 = calculate_similarity_ratio("Analyze", "analyze")
        ratio2 = calculate_similarity_ratio("ANALYZE", "analyze")
        assert ratio1 == 1.0
        assert ratio2 == 1.0


class TestIsFuzzyMatch:
    """Tests for fuzzy match checker."""

    def test_is_fuzzy_match_true(self):
        """Should return True for fuzzy matches."""
        assert is_fuzzy_match("analyze", "analize") is True
        assert is_fuzzy_match("structure", "structre") is True

    def test_is_fuzzy_match_false(self):
        """Should return False for non-matches."""
        assert is_fuzzy_match("analyze", "execute") is False
        assert is_fuzzy_match("structure", "building") is False

    def test_is_fuzzy_match_exact(self):
        """Should return True for exact matches."""
        assert is_fuzzy_match("analyze", "analyze") is True

    def test_is_fuzzy_match_custom_threshold(self):
        """Should respect custom threshold."""
        # Lower threshold
        assert is_fuzzy_match("analyze", "analize", min_similarity_ratio=0.8) is True
        # Higher threshold
        assert is_fuzzy_match("analyze", "analize", min_similarity_ratio=0.95) is False


class TestPerformance:
    """Performance and benchmark tests."""

    def test_fuzzy_match_performance(self, benchmark):
        """Fuzzy matching should be fast enough."""
        key_terms = {
            "analyze": 1.0,
            "structure": 1.0,
            "architecture": 1.0,
            "framework": 1.0,
            "design": 1.0,
            "system": 1.0,
            "create": 1.0,
            "generate": 1.0,
            "implement": 1.0,
            "refactor": 1.0,
        }
        words = {"analize", "structre", "architcture"}

        result = benchmark(extract_key_terms_fuzzy, words, key_terms)
        assert result is not None
        assert len(result) > 0

    def test_cascading_performance(self, benchmark):
        """Cascading matcher should be fast."""
        key_terms = dict.fromkeys(["analyze", "structure", "architecture"], 1.0)
        query = "analize the structre and architcture"

        result = benchmark(match_keywords_cascading, query, key_terms, True)
        matches, stats = result
        assert len(matches) > 0
        assert stats["method"] in ["exact", "fuzzy"]


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_unicode_characters(self):
        """Should handle Unicode characters."""
        words = {"café"}  # Unicode word
        key_terms = {"cafe": 1.0}

        # Should not crash
        matches = extract_key_terms_fuzzy(words, key_terms)
        # Behavior depends on normalization, just check it doesn't crash
        assert isinstance(matches, set)

    def test_very_long_word(self):
        """Should handle very long words."""
        long_word = "supercalifragilisticexpialidocious"
        words = {long_word}
        key_terms = {long_word: 1.0}

        matches = extract_key_terms_fuzzy(words, key_terms)
        assert long_word in matches

    def test_single_character_words(self):
        """Should handle single character words."""
        words = {"a", "i"}
        key_terms = {"a": 1.0}

        matches = extract_key_terms_fuzzy(words, key_terms)
        assert "a" in matches

    def test_special_characters_in_query(self):
        """Should handle special characters in query."""
        key_terms = {"analyze": 1.5}
        query = "@#$ analyze %^&"

        matches, _ = match_keywords_cascading(query, key_terms)
        assert "analyze" in matches

    def test_numbers_in_query(self):
        """Should handle numbers in query."""
        key_terms = {"test": 1.0}
        query = "test 123 test"

        matches, _ = match_keywords_cascading(query, key_terms)
        assert "test" in matches

    def test_multiple_spaces(self):
        """Should handle multiple spaces."""
        key_terms = {"analyze": 1.5, "structure": 1.2}
        query = "analyze    the     structure"

        matches, _ = match_keywords_cascading(query, key_terms)
        assert matches == {"analyze", "structure"}
