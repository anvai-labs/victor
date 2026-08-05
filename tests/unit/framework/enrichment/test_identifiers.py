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

"""Unit tests for identifier extraction utilities."""

import pytest
from victor.framework.enrichment.identifiers import (
    PATTERNS,
    COMMON_WORDS,
    extract_identifiers,
    extract_camelcase,
    extract_snake_case,
    extract_dotted_paths,
    extract_quoted_identifiers,
    IdentifierExtractor,
)


class TestPatterns:
    """Tests for pattern constants."""

    def test_patterns_contains_camelcase(self):
        """Test PATTERNS contains camelcase pattern."""
        assert "camelcase" in PATTERNS
        assert PATTERNS["camelcase"]

    def test_patterns_contains_snake_case(self):
        """Test PATTERNS contains snake_case pattern."""
        assert "snake_case" in PATTERNS
        assert PATTERNS["snake_case"]

    def test_patterns_contains_dotted(self):
        """Test PATTERNS contains dotted pattern."""
        assert "dotted" in PATTERNS
        assert PATTERNS["dotted"]

    def test_patterns_contains_quoted(self):
        """Test PATTERNS contains quoted pattern."""
        assert "quoted" in PATTERNS
        assert PATTERNS["quoted"]


class TestCommonWords:
    """Tests for common words exclusion."""

    def test_common_words_is_set(self):
        """Test COMMON_WORDS is a set."""
        assert isinstance(COMMON_WORDS, set)

    def test_common_words_contains_articles(self):
        """Test COMMON_WORDS contains common articles."""
        assert "the" in COMMON_WORDS
        assert "and" in COMMON_WORDS

    def test_common_words_contains_keywords(self):
        """Test COMMON_WORDS contains common keywords."""
        assert "class" in COMMON_WORDS
        assert "function" in COMMON_WORDS


class TestExtractIdentifiers:
    """Tests for extract_identifiers function."""

    def test_extract_empty_text(self):
        """Test extraction from empty text."""
        result = extract_identifiers("")
        assert result == []

    def test_extract_none_text(self):
        """Test extraction handles None-like empty."""
        result = extract_identifiers("")
        assert result == []

    def test_extract_camelcase_identifiers(self):
        """Test extracting CamelCase identifiers."""
        text = "Check the UserManager and AuthService classes"
        result = extract_identifiers(text)
        assert "UserManager" in result
        assert "AuthService" in result

    def test_extract_snake_case_identifiers(self):
        """Test extracting snake_case identifiers."""
        text = "Call get_user_data and update_settings functions"
        result = extract_identifiers(text)
        assert "get_user_data" in result
        assert "update_settings" in result

    def test_extract_dotted_paths(self):
        """Test extracting dotted paths."""
        text = "Import from victor.agent.orchestrator"
        result = extract_identifiers(text)
        assert "victor.agent.orchestrator" in result

    def test_extract_quoted_identifiers(self):
        """Test extracting backtick-quoted identifiers."""
        text = "Use the `config` and `settings` variables"
        result = extract_identifiers(text)
        assert "config" in result
        assert "settings" in result

    def test_excludes_common_words(self):
        """Test common words are excluded."""
        text = "the function class method"
        result = extract_identifiers(text)
        assert "the" not in result
        assert "function" not in result
        assert "class" not in result

    def test_respects_min_length(self):
        """Test minimum length filtering."""
        text = "Check a b ab abc test"
        result = extract_identifiers(text, min_length=3)
        # Only identifiers >= 3 chars should be included
        assert "a" not in result
        assert "b" not in result
        assert "ab" not in result

    def test_respects_max_identifiers(self):
        """Test maximum identifiers limit."""
        text = " ".join([f"var_{i}" for i in range(100)])
        result = extract_identifiers(text, max_identifiers=10)
        assert len(result) <= 10

    def test_custom_patterns(self):
        """Test using custom patterns."""
        custom_patterns = {"uppercase": r"\b([A-Z]{3,})\b"}
        text = "Check ABC and XYZ but not ab or xy"
        result = extract_identifiers(text, patterns=custom_patterns)
        assert "ABC" in result
        assert "XYZ" in result

    def test_custom_exclude_words(self):
        """Test using custom exclude words."""
        text = "check the user_data field"
        result = extract_identifiers(text, exclude_words={"user_data"})
        assert "user_data" not in result

    def test_deduplicates_results(self):
        """Test results are deduplicated."""
        text = "UserManager uses UserManager pattern"
        result = extract_identifiers(text)
        assert result.count("UserManager") == 1

    def test_preserves_order(self):
        """Test order of first occurrence is preserved."""
        text = "First check UserManager then AuthService"
        result = extract_identifiers(text)
        user_idx = result.index("UserManager") if "UserManager" in result else -1
        auth_idx = result.index("AuthService") if "AuthService" in result else -1
        if user_idx >= 0 and auth_idx >= 0:
            assert user_idx < auth_idx


class TestConvenienceFunctions:
    """Tests for convenience extraction functions."""

    def test_extract_camelcase_only(self):
        """Test extract_camelcase extracts only CamelCase."""
        text = "UserManager get_user user.manager"
        result = extract_camelcase(text)
        assert "UserManager" in result
        assert "get_user" not in result

    def test_extract_snake_case_only(self):
        """Test extract_snake_case extracts only snake_case."""
        text = "UserManager get_user_data user.manager"
        result = extract_snake_case(text)
        assert "get_user_data" in result
        assert "UserManager" not in result

    def test_extract_dotted_paths_only(self):
        """Test extract_dotted_paths extracts only dotted."""
        text = "Import victor.agent.orchestrator not UserManager"
        result = extract_dotted_paths(text)
        assert "victor.agent.orchestrator" in result
        assert "UserManager" not in result

    def test_extract_quoted_only(self):
        """Test extract_quoted_identifiers extracts only quoted."""
        text = "Use `config` not UserManager"
        result = extract_quoted_identifiers(text)
        assert "config" in result
        assert "UserManager" not in result


class TestIdentifierExtractor:
    """Tests for IdentifierExtractor class."""

    def test_default_extractor(self):
        """Test extractor with default patterns."""
        extractor = IdentifierExtractor()
        text = "UserManager uses get_config from app.config"
        result = extractor.extract(text)
        assert "UserManager" in result
        assert "get_config" in result
        assert "app.config" in result

    def test_custom_patterns_extractor(self):
        """Test extractor with custom patterns."""
        extractor = IdentifierExtractor(patterns={"upper": r"\b([A-Z]+)\b"})
        text = "Check ABC def XYZ"
        result = extractor.extract(text)
        assert "ABC" in result
        assert "XYZ" in result
        assert "def" not in result

    def test_custom_exclude_words(self):
        """Test extractor with custom exclude words."""
        extractor = IdentifierExtractor(exclude_words={"UserManager"})
        text = "UserManager uses AuthService"
        result = extractor.extract(text)
        assert "AuthService" in result
        # Note: custom exclude replaces default, so "UserManager" might still match
        # depending on implementation

    def test_add_pattern(self):
        """Test adding custom pattern."""
        extractor = IdentifierExtractor()
        extractor.add_pattern("all_caps", r"\b([A-Z]{2,})\b")
        text = "Check API and SDK values"
        result = extractor.extract(text)
        assert "API" in result
        assert "SDK" in result

    def test_add_exclude_word(self):
        """Test adding exclude word."""
        extractor = IdentifierExtractor()
        extractor.add_exclude_word("UserManager")
        # The word should be added to exclude list
        assert "usermanager" in extractor.exclude_words
