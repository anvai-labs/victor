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

"""Comprehensive unit tests for text_normalizer module.

Tests cover:
- All 5 normalization functions
- All 5 preset methods
- Edge cases: empty strings, special characters, unicode
- Error handling: ValueError for invalid inputs
- Coverage goal: >95%
"""

import pytest

from victor.core.utils.text_normalizer import (
    TextNormalizationPresets,
    normalize_for_filename,
    normalize_for_git_branch,
    normalize_for_test_filename,
    sanitize_class_name,
    slugify,
)


class TestNormalizeForGitBranch:
    """Tests for normalize_for_git_branch function."""

    def test_basic_normalization(self):
        """Test basic feature name normalization."""
        assert normalize_for_git_branch("User Authentication") == "user-authentication"
        assert normalize_for_git_branch("Data Processor") == "data-processor"
        assert normalize_for_git_branch("API Controller") == "api-controller"

    def test_with_prefix(self):
        """Test normalization with prefix."""
        assert normalize_for_git_branch("User Auth", prefix="feature/") == "feature/user-auth"
        assert normalize_for_git_branch("Memory Leak", prefix="bugfix/") == "bugfix/memory-leak"
        assert normalize_for_git_branch("Hot Fix", prefix="hotfix/") == "hotfix/hot-fix"

    def test_prefix_without_trailing_slash(self):
        """Test that prefix without trailing slash gets normalized correctly."""
        # Prefix should not have double slash
        result = normalize_for_git_branch("Test", prefix="feature/")
        assert result == "feature/test"
        assert "//" not in result

    def test_whitespace_normalization(self):
        """Test that multiple spaces get collapsed to single hyphen."""
        assert normalize_for_git_branch("  Multiple   Spaces  ") == "multiple-spaces"
        assert normalize_for_git_branch("One  Two   Three    Four") == "one-two-three-four"

    def test_special_character_removal(self):
        """Test that special characters are removed."""
        assert normalize_for_git_branch("Special@#$Characters") == "specialcharacters"
        assert normalize_for_git_branch("Test!Name!Here") == "testnamehere"
        # Underscores are converted to hyphens in git branch names
        assert normalize_for_git_branch("API_v2_Endpoint") == "api-v2-endpoint"

    def test_lowercase_conversion(self):
        """Test that output is always lowercase."""
        assert normalize_for_git_branch("UPPERCASE") == "uppercase"
        assert normalize_for_git_branch("MixedCase") == "mixedcase"
        assert normalize_for_git_branch("camelCase") == "camelcase"

    def test_hyphen_normalization(self):
        """Test that multiple consecutive hyphens get collapsed."""
        assert normalize_for_git_branch("test--name") == "test-name"
        assert normalize_for_git_branch("many---hyphens") == "many-hyphens"

    def test_leading_trailing_hyphens_removed(self):
        """Test that leading and trailing hyphens are removed."""
        assert normalize_for_git_branch("-test-") == "test"
        assert normalize_for_git_branch("--leading") == "leading"
        assert normalize_for_git_branch("trailing--") == "trailing"

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_for_git_branch("")

        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_for_git_branch("   ")

    def test_only_special_characters_raises_error(self):
        """Test that string with only special characters raises ValueError."""
        with pytest.raises(ValueError, match="no valid characters"):
            normalize_for_git_branch("@#$%^&*()")

        with pytest.raises(ValueError, match="no valid characters"):
            normalize_for_git_branch("---")

    def test_numeric_preservation(self):
        """Test that numbers are preserved."""
        assert normalize_for_git_branch("API Version 2") == "api-version-2"
        assert normalize_for_git_branch("Test 123") == "test-123"
        assert normalize_for_git_branch("v2.0 API") == "v20-api"

    def test_real_world_examples(self):
        """Test with real-world feature names."""
        assert (
            normalize_for_git_branch("User Authentication System") == "user-authentication-system"
        )
        assert normalize_for_git_branch("Add OAuth2 Support") == "add-oauth2-support"
        assert normalize_for_git_branch("Fix Memory Leak") == "fix-memory-leak"


class TestNormalizeForFilename:
    """Tests for normalize_for_filename function."""

    def test_basic_normalization(self):
        """Test basic filename normalization."""
        assert normalize_for_filename("Data Processor") == "data_processor"
        assert normalize_for_filename("User Auth") == "user_auth"
        assert normalize_for_filename("API Controller") == "api_controller"

    def test_with_extension(self):
        """Test normalization with file extension."""
        assert normalize_for_filename("Data Processor", extension=".py") == "data_processor.py"
        assert normalize_for_filename("Test File", extension=".txt") == "test_file.txt"

    def test_extension_without_dot(self):
        """Test that extension without dot gets normalized correctly."""
        result = normalize_for_filename("Test", extension="py")
        assert result == "test.py"

    def test_hyphen_to_underscore(self):
        """Test that hyphens are converted to underscores."""
        assert normalize_for_filename("test-name") == "test_name"
        assert normalize_for_filename("API-Controller") == "api_controller"

    def test_whitespace_normalization(self):
        """Test that whitespace becomes underscores."""
        assert normalize_for_filename("  Multiple   Spaces  ") == "multiple_spaces"
        assert normalize_for_filename("One  Two   Three") == "one_two_three"

    def test_special_character_removal(self):
        """Test that special characters are removed."""
        assert normalize_for_filename("Special@#$Characters") == "specialcharacters"
        assert normalize_for_filename("Test!File!Name") == "testfilename"

    def test_underscore_preservation(self):
        """Test that existing underscores are preserved."""
        assert normalize_for_filename("test_file") == "test_file"
        assert normalize_for_filename("my_test_file") == "my_test_file"

    def test_underscore_normalization(self):
        """Test that multiple underscores get collapsed."""
        assert normalize_for_filename("test__file") == "test_file"
        assert normalize_for_filename("many___underscores") == "many_underscores"

    def test_leading_trailing_underscores_removed(self):
        """Test that leading and trailing underscores are removed."""
        assert normalize_for_filename("_test_") == "test"
        assert normalize_for_filename("__leading") == "leading"
        assert normalize_for_filename("trailing__") == "trailing"

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_for_filename("")

        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_for_filename("   ")

    def test_only_special_characters_raises_error(self):
        """Test that string with only special characters raises ValueError."""
        with pytest.raises(ValueError, match="no valid characters"):
            normalize_for_filename("@#$%^&*()")

    def test_lowercase_conversion(self):
        """Test that output is always lowercase."""
        assert normalize_for_filename("UPPERCASE") == "uppercase"
        assert normalize_for_filename("CamelCase") == "camelcase"


class TestSlugify:
    """Tests for slugify function."""

    def test_basic_slugification(self):
        """Test basic slug generation."""
        assert slugify("How to Use Victor CLI") == "how-to-use-victor-cli"
        assert slugify("Python Programming 101") == "python-programming-101"
        assert slugify("Getting Started") == "getting-started"

    def test_custom_delimiter(self):
        """Test slug generation with custom delimiter."""
        assert slugify("API Controller", delimiter="_") == "api_controller"
        assert slugify("Test Name Here", delimiter="_") == "test_name_here"
        assert slugify("Multiple Words Together", delimiter="_") == "multiple_words_together"

    def test_default_delimiter(self):
        """Test that default delimiter is hyphen."""
        assert slugify("Test Name") == "test-name"
        assert slugify("Multiple Words") == "multiple-words"

    def test_whitespace_normalization(self):
        """Test that whitespace gets replaced with delimiter."""
        assert slugify("  Multiple   Spaces  ") == "multiple-spaces"
        assert slugify("Tabs\tAnd\nNewlines") == "tabs-and-newlines"

    def test_special_character_removal(self):
        """Test that special characters are removed."""
        assert slugify("Special@#$Characters") == "specialcharacters"
        assert slugify("What's Up?") == "whats-up"

    def test_delimiter_normalization(self):
        """Test that multiple consecutive delimiters get collapsed."""
        assert slugify("test---name") == "test-name"
        assert slugify("many___underscores", delimiter="_") == "many_underscores"

    def test_leading_trailing_delimiters_removed(self):
        """Test that leading and trailing delimiters are removed."""
        assert slugify("-test-") == "test"
        assert slugify("--leading") == "leading"
        assert slugify("trailing--") == "trailing"

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            slugify("")

        with pytest.raises(ValueError, match="cannot be empty"):
            slugify("   ")

    def test_only_special_characters_raises_error(self):
        """Test that string with only special characters raises ValueError."""
        with pytest.raises(ValueError, match="no valid characters"):
            slugify("@#$%^&*()")

    def test_numeric_preservation(self):
        """Test that numbers are preserved."""
        assert slugify("Version 2.0") == "version-20"
        assert slugify("Top 10 Tips") == "top-10-tips"


class TestSanitizeClassName:
    """Tests for sanitize_class_name function."""

    def test_basic_conversion(self):
        """Test basic conversion to PascalCase."""
        assert sanitize_class_name("data processor") == "DataProcessor"
        assert sanitize_class_name("user auth") == "UserAuth"
        assert sanitize_class_name("API controller") == "APIController"

    def test_hyphen_to_pascalcase(self):
        """Test that hyphens are handled correctly."""
        assert sanitize_class_name("user-authentication-module") == "UserAuthenticationModule"
        # Acronyms are preserved correctly
        assert sanitize_class_name("API-controller") == "APIController"

    def test_underscore_to_pascalcase(self):
        """Test that underscores are handled correctly."""
        assert sanitize_class_name("data_processor") == "DataProcessor"
        assert sanitize_class_name("user_auth_module") == "UserAuthModule"

    def test_whitespace_to_pascalcase(self):
        """Test that whitespace is handled correctly."""
        assert sanitize_class_name("  multiple   spaces  ") == "MultipleSpaces"
        assert sanitize_class_name("one  two   three") == "OneTwoThree"

    def test_special_character_removal(self):
        """Test that special characters are removed."""
        assert sanitize_class_name("Special@#$Characters") == "SpecialCharacters"
        assert sanitize_class_name("Test!Name!Here") == "TestNameHere"

    def test_already_pascalcase(self):
        """Test that already PascalCase strings are preserved."""
        assert sanitize_class_name("DataProcessor") == "DataProcessor"
        assert sanitize_class_name("APIController") == "APIController"

    def test_mixed_case_input(self):
        """Test that mixed case input is handled correctly."""
        assert sanitize_class_name("data processor") == "DataProcessor"
        # Capitalizes first letter, preserves the rest
        assert sanitize_class_name("DATA PROCESSOR") == "DATAPROCESSOR"
        assert sanitize_class_name("DaTa PrOcEsSoR") == "DaTaPrOcEsSoR"

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_class_name("")

        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_class_name("   ")

    def test_only_special_characters_raises_error(self):
        """Test that string with only special characters raises ValueError."""
        with pytest.raises(ValueError, match="no valid characters"):
            sanitize_class_name("@#$%^&*()")

    def test_numeric_preservation(self):
        """Test that numbers are preserved."""
        assert sanitize_class_name("API version 2") == "APIVersion2"
        assert sanitize_class_name("test 123") == "Test123"


class TestNormalizeForTestFilename:
    """Tests for normalize_for_test_filename function."""

    def test_basic_normalization(self):
        """Test basic test filename generation."""
        assert normalize_for_test_filename("User Authentication") == "test_user_authentication.py"
        assert normalize_for_test_filename("Data Processor") == "test_data_processor.py"

    def test_always_has_test_prefix(self):
        """Test that output always starts with 'test_'."""
        result = normalize_for_test_filename("Some Feature")
        assert result.startswith("test_")
        assert not result.startswith("test_test_")

    def test_always_has_py_extension(self):
        """Test that output always ends with '.py'."""
        result = normalize_for_test_filename("Some Feature")
        assert result.endswith(".py")

    def test_lowercase_conversion(self):
        """Test that output is always lowercase."""
        assert normalize_for_test_filename("UPPERCASE") == "test_uppercase.py"
        assert normalize_for_test_filename("CamelCase") == "test_camelcase.py"

    def test_special_character_handling(self):
        """Test that special characters are handled correctly."""
        assert normalize_for_test_filename("Special@#$Characters") == "test_specialcharacters.py"

    def test_whitespace_normalization(self):
        """Test that whitespace becomes underscores."""
        assert normalize_for_test_filename("  Multiple   Spaces  ") == "test_multiple_spaces.py"

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_for_test_filename("")

    def test_real_world_examples(self):
        """Test with real-world feature names."""
        assert (
            normalize_for_test_filename("User Authentication Module")
            == "test_user_authentication_module.py"
        )
        assert (
            normalize_for_test_filename("OAuth2 Implementation") == "test_oauth2_implementation.py"
        )


class TestTextNormalizationPresets:
    """Tests for TextNormalizationPresets class."""

    def test_git_feature_branch(self):
        """Test git_feature_branch preset."""
        assert (
            TextNormalizationPresets.git_feature_branch("User Authentication")
            == "feature/user-authentication"
        )
        assert (
            TextNormalizationPresets.git_feature_branch("Data Processor")
            == "feature/data-processor"
        )

    def test_git_bugfix_branch(self):
        """Test git_bugfix_branch preset."""
        assert TextNormalizationPresets.git_bugfix_branch("Memory Leak") == "bugfix/memory-leak"
        assert TextNormalizationPresets.git_bugfix_branch("Crash Fix") == "bugfix/crash-fix"

    def test_python_source_file(self):
        """Test python_source_file preset."""
        assert TextNormalizationPresets.python_source_file("Data Processor") == "data_processor.py"
        assert TextNormalizationPresets.python_source_file("User Auth") == "user_auth.py"

    def test_python_test_file(self):
        """Test python_test_file preset."""
        assert TextNormalizationPresets.python_test_file("User Auth") == "test_user_auth.py"
        assert (
            TextNormalizationPresets.python_test_file("Data Processor") == "test_data_processor.py"
        )

    def test_url_slug(self):
        """Test url_slug preset."""
        assert TextNormalizationPresets.url_slug("How to Use Victor") == "how-to-use-victor"
        assert TextNormalizationPresets.url_slug("Getting Started") == "getting-started"


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        # Unicode characters that should be removed
        assert normalize_for_git_branch("Test™©®") == "test"  # Special unicode removed
        assert slugify("Hello World") == "hello-world"  # ASCII works fine

    def test_very_long_names(self):
        """Test handling of very long names."""
        long_name = "This Is A Very Long Feature Name With Many Words"
        result = normalize_for_git_branch(long_name)
        assert "this-is-a-very-long-feature-name-with-many-words" == result

    def test_single_character(self):
        """Test handling of single character."""
        assert normalize_for_git_branch("A") == "a"
        assert normalize_for_filename("A") == "a"
        assert slugify("A") == "a"

    def test_numbers_only(self):
        """Test handling of numbers-only strings."""
        assert normalize_for_git_branch("123") == "123"
        assert normalize_for_filename("123") == "123"
        assert slugify("123") == "123"

    def test_already_normalized_input(self):
        """Test that already normalized input is handled correctly."""
        assert normalize_for_git_branch("already-normalized") == "already-normalized"
        assert normalize_for_filename("already_normalized") == "already_normalized"
        assert slugify("already-slugified") == "already-slugified"


class TestErrorMessages:
    """Tests for error messages and validation."""

    def test_empty_string_error_message(self):
        """Test that empty string gives helpful error message."""
        with pytest.raises(ValueError, match="Feature name cannot be empty"):
            normalize_for_git_branch("")

        with pytest.raises(ValueError, match="Filename cannot be empty"):
            normalize_for_filename("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            slugify("")

    def test_invalid_characters_error_message(self):
        """Test that invalid characters give helpful error message."""
        with pytest.raises(ValueError, match="no valid characters"):
            normalize_for_git_branch("@#$")

        with pytest.raises(ValueError, match="no valid characters"):
            normalize_for_filename("---")
