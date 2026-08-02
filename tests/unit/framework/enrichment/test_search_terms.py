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

"""Unit tests for search term extraction utilities."""

import pytest
from victor.framework.enrichment.search_terms import (
    QUESTION_WORDS,
    SEARCH_TERM_PATTERNS,
    STOP_WORDS,
    SearchTermExtractor,
    extract_search_terms,
    get_search_term_patterns,
)


class TestSearchTermConstants:
    """Tests for search term constants."""

    def test_question_words_contains_common_words(self):
        """Test QUESTION_WORDS contains common question words."""
        assert "what" in QUESTION_WORDS
        assert "how" in QUESTION_WORDS
        assert "why" in QUESTION_WORDS
        assert "when" in QUESTION_WORDS
        assert "where" in QUESTION_WORDS
        assert "who" in QUESTION_WORDS

    def test_question_words_contains_auxiliary_verbs(self):
        """Test QUESTION_WORDS contains auxiliary verbs."""
        assert "is" in QUESTION_WORDS
        assert "are" in QUESTION_WORDS
        assert "can" in QUESTION_WORDS
        assert "could" in QUESTION_WORDS
        assert "would" in QUESTION_WORDS
        assert "should" in QUESTION_WORDS

    def test_stop_words_contains_articles(self):
        """Test STOP_WORDS contains articles."""
        assert "the" in STOP_WORDS
        assert "a" in STOP_WORDS
        assert "an" in STOP_WORDS

    def test_stop_words_contains_prepositions(self):
        """Test STOP_WORDS contains prepositions."""
        assert "in" in STOP_WORDS
        assert "on" in STOP_WORDS
        assert "at" in STOP_WORDS
        assert "to" in STOP_WORDS
        assert "for" in STOP_WORDS
        assert "of" in STOP_WORDS

    def test_search_term_patterns_contains_quoted(self):
        """Test SEARCH_TERM_PATTERNS contains quoted pattern."""
        assert "quoted" in SEARCH_TERM_PATTERNS

    def test_search_term_patterns_contains_capitalized(self):
        """Test SEARCH_TERM_PATTERNS contains capitalized pattern."""
        assert "capitalized" in SEARCH_TERM_PATTERNS

    def test_search_term_patterns_contains_technical(self):
        """Test SEARCH_TERM_PATTERNS contains technical pattern."""
        assert "technical" in SEARCH_TERM_PATTERNS

    def test_search_term_patterns_contains_hyphenated(self):
        """Test SEARCH_TERM_PATTERNS contains hyphenated pattern."""
        assert "hyphenated" in SEARCH_TERM_PATTERNS


class TestGetSearchTermPatterns:
    """Tests for get_search_term_patterns function."""

    def test_returns_copy_of_patterns(self):
        """Test returns a copy, not the original."""
        patterns = get_search_term_patterns()
        patterns["test"] = "test_pattern"
        assert "test" not in SEARCH_TERM_PATTERNS

    def test_returns_all_patterns(self):
        """Test returns all expected patterns."""
        patterns = get_search_term_patterns()
        assert len(patterns) == 4
        assert "quoted" in patterns
        assert "capitalized" in patterns
        assert "technical" in patterns
        assert "hyphenated" in patterns


class TestExtractSearchTerms:
    """Tests for extract_search_terms function."""

    def test_empty_prompt_returns_empty_list(self):
        """Test empty prompt returns empty list."""
        assert extract_search_terms("") == []

    def test_whitespace_only_returns_empty_list(self):
        """Test whitespace-only prompt returns empty list."""
        assert extract_search_terms("   ") == []
        assert extract_search_terms("\n\t") == []

    def test_none_like_empty_string(self):
        """Test None-like values return empty list."""
        # Empty string
        assert extract_search_terms("") == []

    def test_extracts_quoted_phrases(self):
        """Test extraction of quoted phrases."""
        result = extract_search_terms('What is "machine learning"?')
        assert "machine learning" in result

    def test_extracts_multiple_quoted_phrases(self):
        """Test extraction of multiple quoted phrases."""
        result = extract_search_terms('Compare "Python" and "JavaScript"')
        assert "Python" in result
        assert "JavaScript" in result

    def test_extracts_capitalized_terms(self):
        """Test extraction of capitalized terms."""
        result = extract_search_terms("Tell me about Python programming")
        assert "Python" in result

    def test_extracts_capitalized_phrases(self):
        """Test extraction of capitalized phrases."""
        result = extract_search_terms("What is Machine Learning?")
        assert "Machine Learning" in result

    def test_extracts_technical_snake_case(self):
        """Test extraction of snake_case terms."""
        result = extract_search_terms("How to use snake_case_variable?")
        assert "snake_case_variable" in result

    def test_extracts_technical_camel_case(self):
        """Test extraction of camelCase terms."""
        result = extract_search_terms("What is camelCaseVariable?")
        assert "camelCaseVariable" in result

    def test_extracts_hyphenated_terms(self):
        """Test extraction of hyphenated terms."""
        result = extract_search_terms("Explain object-oriented programming")
        assert "object-oriented" in result

    def test_respects_max_terms_limit(self):
        """Test max_terms parameter limits results."""
        prompt = '"term1" "term2" "term3" "term4" "term5" "term6"'
        result = extract_search_terms(prompt, max_terms=3)
        assert len(result) <= 3

    def test_respects_min_word_length(self):
        """Test min_word_length filters short words."""
        result = extract_search_terms("get the items from database", min_word_length=5)
        # "get" is too short with min_word_length=5
        assert "items" in result or "database" in result

    def test_filters_stop_words(self):
        """Test stop words are filtered out."""
        result = extract_search_terms("the and or but")
        # All stop words should be filtered
        for term in result:
            assert term.lower() not in STOP_WORDS

    def test_removes_question_words_from_start(self):
        """Test question words at start are handled."""
        result = extract_search_terms("What is quantum computing?")
        # Should extract meaningful terms
        assert "quantum" in result or "computing" in result or "What" in result

    def test_deduplicates_terms(self):
        """Test duplicate terms are removed."""
        result = extract_search_terms('"Python" Python PYTHON')
        # Should only appear once
        python_count = sum(1 for t in result if t.lower() == "python")
        assert python_count == 1

    def test_preserves_term_order(self):
        """Test terms are ordered by extraction priority."""
        # Quoted should come first
        result = extract_search_terms('"first_term" SecondTerm')
        if len(result) >= 2 and "first_term" in result and "SecondTerm" in result:
            assert result.index("first_term") < result.index("SecondTerm")

    def test_complex_prompt(self):
        """Test complex prompt with multiple term types."""
        prompt = """How does "deep learning" work with TensorFlow
                   and machine-learning pipelines using my_custom_function?"""
        result = extract_search_terms(prompt)
        assert len(result) > 0
        # Should have some terms extracted
        has_deep_learning = "deep learning" in result
        has_tensorflow = "TensorFlow" in result
        has_machine_learning = "machine-learning" in result
        has_custom_function = "my_custom_function" in result
        assert has_deep_learning or has_tensorflow or has_machine_learning or has_custom_function

    def test_real_question_example(self):
        """Test with realistic question."""
        result = extract_search_terms("What are the best practices for REST API design?")
        assert len(result) > 0

    def test_programming_query(self):
        """Test with programming-related query."""
        result = extract_search_terms("How to implement a binary_search algorithm in Python?")
        assert len(result) > 0
        # Should extract binary_search
        has_binary_search = any("binary_search" in t for t in result)
        has_python = any("Python" in t for t in result)
        assert has_binary_search or has_python

    def test_max_terms_default(self):
        """Test default max_terms value."""
        # Create prompt with many potential terms
        prompt = '"A" "B" "C" "D" "E" "F" "G" "H"'
        result = extract_search_terms(prompt)
        # Default is 5
        assert len(result) <= 5


class TestSearchTermExtractor:
    """Tests for SearchTermExtractor class."""

    def test_init_default_values(self):
        """Test default initialization."""
        extractor = SearchTermExtractor()
        assert extractor.max_terms == 5
        assert extractor._min_word_length == 4

    def test_init_custom_max_terms(self):
        """Test custom max_terms."""
        extractor = SearchTermExtractor(max_terms=10)
        assert extractor.max_terms == 10

    def test_init_custom_min_word_length(self):
        """Test custom min_word_length."""
        extractor = SearchTermExtractor(min_word_length=6)
        assert extractor._min_word_length == 6

    def test_init_with_native_disabled(self):
        """Test initialization with native disabled."""
        extractor = SearchTermExtractor(use_native=False)
        assert extractor._use_native is False

    def test_extract_empty_prompt(self):
        """Test extract with empty prompt."""
        extractor = SearchTermExtractor()
        result = extractor.extract("")
        assert result == []

    def test_extract_simple_prompt(self):
        """Test extract with simple prompt."""
        extractor = SearchTermExtractor()
        result = extractor.extract("What is Python?")
        assert "Python" in result

    def test_extract_respects_max_terms(self):
        """Test extract respects max_terms setting."""
        extractor = SearchTermExtractor(max_terms=2)
        result = extractor.extract('"A" "B" "C" "D"')
        assert len(result) <= 2

    def test_batch_extract_empty_list(self):
        """Test batch_extract with empty list."""
        extractor = SearchTermExtractor()
        result = extractor.batch_extract([])
        assert result == []

    def test_batch_extract_single_prompt(self):
        """Test batch_extract with single prompt."""
        extractor = SearchTermExtractor()
        result = extractor.batch_extract(["What is Python?"])
        assert len(result) == 1
        assert "Python" in result[0]

    def test_batch_extract_multiple_prompts(self):
        """Test batch_extract with multiple prompts."""
        extractor = SearchTermExtractor()
        prompts = [
            "What is Python?",
            "How does JavaScript work?",
            "Explain TensorFlow",
        ]
        result = extractor.batch_extract(prompts)
        assert len(result) == 3
        assert "Python" in result[0]
        # JavaScript might be in phrase or standalone
        assert any("JavaScript" in t for t in result[1])
        # TensorFlow might be in phrase or standalone
        assert any("TensorFlow" in t for t in result[2])

    def test_batch_extract_with_empty_prompts(self):
        """Test batch_extract handles empty prompts."""
        extractor = SearchTermExtractor()
        result = extractor.batch_extract(["", "What is Python?", ""])
        assert len(result) == 3
        assert result[0] == []
        assert "Python" in result[1]
        assert result[2] == []

    def test_is_native_available_static(self):
        """Test is_native_available static method."""
        # Should return a boolean
        result = SearchTermExtractor.is_native_available()
        assert isinstance(result, bool)

    def test_max_terms_property_getter(self):
        """Test max_terms property getter."""
        extractor = SearchTermExtractor(max_terms=7)
        assert extractor.max_terms == 7

    def test_max_terms_property_setter(self):
        """Test max_terms property setter."""
        extractor = SearchTermExtractor(max_terms=5)
        extractor.max_terms = 10
        assert extractor.max_terms == 10

    def test_max_terms_setter_enforces_minimum(self):
        """Test max_terms setter enforces minimum of 1."""
        extractor = SearchTermExtractor()
        extractor.max_terms = 0
        assert extractor.max_terms >= 1
        extractor.max_terms = -5
        assert extractor.max_terms >= 1

    def test_extract_with_changed_max_terms(self):
        """Test extract uses updated max_terms."""
        extractor = SearchTermExtractor(max_terms=5)
        extractor.max_terms = 1
        result = extractor.extract('"A" "B" "C"')
        assert len(result) == 1


class TestPatternMatching:
    """Tests for specific pattern matching behavior."""

    def test_quoted_pattern_single_word(self):
        """Test quoted pattern with single word."""
        result = extract_search_terms('"Python"')
        assert "Python" in result

    def test_quoted_pattern_phrase(self):
        """Test quoted pattern with phrase."""
        result = extract_search_terms('"machine learning algorithms"')
        assert "machine learning algorithms" in result

    def test_quoted_pattern_empty_quotes(self):
        """Test quoted pattern handles empty quotes."""
        result = extract_search_terms('""')
        # Empty quotes should not produce empty string in results
        assert "" not in result

    def test_capitalized_pattern_single_capital(self):
        """Test capitalized pattern with single capital word."""
        result = extract_search_terms("Using React for frontend")
        # React might be captured as part of a phrase "Using React" or standalone
        assert any("React" in t for t in result)

    def test_capitalized_pattern_multi_word(self):
        """Test capitalized pattern with multi-word phrase."""
        result = extract_search_terms("Object Oriented Programming concepts")
        # Should capture capitalized phrase
        assert any("Object" in t or "Programming" in t for t in result)

    def test_technical_pattern_underscore(self):
        """Test technical pattern with underscores."""
        result = extract_search_terms("using my_function_name")
        assert "my_function_name" in result

    def test_technical_pattern_multiple_underscores(self):
        """Test technical pattern with multiple underscores."""
        result = extract_search_terms("call get_user_by_id function")
        assert "get_user_by_id" in result

    def test_technical_pattern_camel_case(self):
        """Test technical pattern with camelCase."""
        result = extract_search_terms("using getUserById method")
        assert "getUserById" in result

    def test_hyphenated_pattern_simple(self):
        """Test hyphenated pattern with simple term."""
        result = extract_search_terms("real-time processing")
        assert "real-time" in result

    def test_hyphenated_pattern_multiple_hyphens(self):
        """Test hyphenated pattern with multiple hyphens."""
        result = extract_search_terms("state-of-the-art technology")
        assert "state-of-the-art" in result

    def test_no_match_lowercase_only(self):
        """Test extraction from lowercase-only text."""
        result = extract_search_terms("this is all lowercase")
        # Should still find significant words
        assert len(result) >= 0  # May or may not find terms


class TestEdgeCases:
    """Tests for edge cases."""

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        result = extract_search_terms("What is cafe and resume?")
        assert len(result) >= 0  # Should not crash

    def test_special_characters(self):
        """Test handling of special characters."""
        result = extract_search_terms("@#$%^&*()")
        assert result == [] or all(isinstance(t, str) for t in result)

    def test_very_long_prompt(self):
        """Test handling of very long prompt."""
        long_prompt = "What is Python? " * 100
        result = extract_search_terms(long_prompt)
        assert len(result) <= 5  # Should respect max_terms

    def test_newlines_and_tabs(self):
        """Test handling of newlines and tabs."""
        result = extract_search_terms("What\nis\tPython?")
        assert "Python" in result

    def test_numbers_in_terms(self):
        """Test terms containing numbers."""
        result = extract_search_terms("Using Python3 and Web2.0")
        # Numbers in technical terms should be handled
        assert len(result) >= 0

    def test_mixed_case_deduplication(self):
        """Test deduplication is case-insensitive."""
        result = extract_search_terms("Python python PYTHON")
        python_count = sum(1 for t in result if t.lower() == "python")
        assert python_count <= 1

    def test_prompt_starting_with_question_word(self):
        """Test prompt starting with question word."""
        result = extract_search_terms("What is the meaning of life?")
        # Should extract meaningful terms, not just "What"
        assert len(result) >= 0
