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

"""Canonical text normalization utilities for consistent naming conventions.

This module provides centralized text transformation functions for generating
consistent names across different contexts:
- Git branch names (lowercase, hyphens, optional prefix)
- Python filenames (snake_case)
- URL slugs (hyphen-separated)
- Python class names (PascalCase)
- Test filenames (test_ prefix)

Design Pattern: Strategy Pattern
==================================
Each normalization function is a pure strategy function that can be used
independently or composed with other transformations.

SOLID Compliance:
- SRP: Each function has a single, well-defined purpose
- OCP: New normalization patterns can be added without modifying existing
- LSP: All functions are pure and substitutable
- ISP: Minimal interfaces (each function takes str, returns str)
- DIP: No dependencies on concrete implementations, only stdlib

Usage Examples:

    # Normalize for git branch
    branch_name = normalize_for_git_branch("User Authentication", prefix="feature/")
    # Returns: "feature/user-authentication"

    # Normalize for Python filename
    filename = normalize_for_filename("Data Processor", extension=".py")
    # Returns: "data_processor.py"

    # Normalize for URL slug
    slug = slugify("How to Use Victor CLI", delimiter="-")
    # Returns: "how-to-use-victor-cli"

    # Normalize for class name
    class_name = sanitize_class_name("data processor")
    # Returns: "DataProcessor"

    # Normalize for test filename
    test_file = normalize_for_test_filename("User Authentication")
    # Returns: "test_user_authentication.py"

Why This Utility Exists:
-------------------------
Before this refactor, text normalization logic was duplicated across:
- new_feature_workflow.py: 107 lines with hard-coded sanitization
- Various modules: Ad-hoc string manipulation throughout codebase

This utility extracts text normalization as a shared concern, following DRY
principles and ensuring consistent naming conventions across the codebase.

Performance Characteristics:
---------------------------
- O(n) where n is string length
- Uses compiled regex patterns for efficiency
- No external dependencies beyond stdlib
- Pure functions (no side effects, easy to test)
"""

import re
from typing import Optional

# Pre-compile regex patterns for performance
# Pattern to match non-alphanumeric characters (except hyphens/underscores)
_NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-zA-Z0-9\-_]")

# Pattern to match whitespace and special characters
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Pattern to match non-word characters
_NON_WORD_PATTERN = re.compile(r"[^a-zA-Z0-9_]")


def normalize_for_git_branch(name: str, prefix: Optional[str] = None) -> str:
    """Normalize a feature name for use as a git branch name.

    Git branch names should:
    - Be lowercase
    - Use hyphens instead of spaces
    - Contain only alphanumeric characters and hyphens
    - Optionally have a prefix (e.g., "feature/", "bugfix/")

    Args:
        name: The feature name to normalize (e.g., "User Authentication")
        prefix: Optional prefix to add (e.g., "feature/", "bugfix/")

    Returns:
        Normalized branch name (e.g., "feature/user-authentication")

    Raises:
        ValueError: If name is empty or contains only special characters

    Examples:
        >>> normalize_for_git_branch("User Authentication")
        'user-authentication'
        >>> normalize_for_git_branch("User Authentication", prefix="feature/")
        'feature/user-authentication'
        >>> normalize_for_git_branch("  Multiple   Spaces  ")
        'multiple-spaces'
        >>> normalize_for_git_branch("Special@#$Characters")
        'specialcharacters'
    """
    if not name or not name.strip():
        raise ValueError("Feature name cannot be empty")

    # Convert to lowercase
    normalized = name.lower()

    # Replace underscores with hyphens (git convention)
    normalized = normalized.replace("_", "-")

    # Replace whitespace with hyphens
    normalized = _WHITESPACE_PATTERN.sub("-", normalized)

    # Remove non-alphanumeric characters except hyphens
    normalized = _NON_ALPHANUMERIC_PATTERN.sub("", normalized)

    # Collapse multiple consecutive hyphens
    normalized = re.sub(r"-+", "-", normalized)

    # Remove leading/trailing hyphens
    normalized = normalized.strip("-")

    if not normalized:
        raise ValueError(f"Feature name '{name}' contains no valid characters for a branch name")

    # Add prefix if provided
    if prefix:
        # Ensure prefix doesn't end with a slash if we add one
        normalized = f"{prefix.rstrip('/')}/{normalized}"

    return normalized


def normalize_for_filename(name: str, extension: Optional[str] = None) -> str:
    """Normalize a name for use as a Python filename (snake_case).

    Python filenames should:
    - Be lowercase
    - Use underscores instead of spaces or hyphens
    - Contain only alphanumeric characters and underscores
    - Optionally have a file extension

    Args:
        name: The name to normalize (e.g., "Data Processor")
        extension: Optional file extension (e.g., ".py", ".txt")

    Returns:
        Normalized filename (e.g., "data_processor.py")

    Raises:
        ValueError: If name is empty or contains only special characters

    Examples:
        >>> normalize_for_filename("Data Processor")
        'data_processor'
        >>> normalize_for_filename("Data Processor", extension=".py")
        'data_processor.py'
        >>> normalize_for_filename("UserAuthenticationModule")
        'userauthenticationmodule'
        >>> normalize_for_filename("API-Controller")
        'api_controller'
    """
    if not name or not name.strip():
        raise ValueError("Filename cannot be empty")

    # Convert to lowercase
    normalized = name.lower()

    # Replace hyphens with underscores
    normalized = normalized.replace("-", "_")

    # Replace whitespace with underscores
    normalized = _WHITESPACE_PATTERN.sub("_", normalized)

    # Remove non-alphanumeric characters except underscores
    normalized = _NON_WORD_PATTERN.sub("", normalized)

    # Collapse multiple consecutive underscores
    normalized = re.sub(r"_+", "_", normalized)

    # Remove leading/trailing underscores
    normalized = normalized.strip("_")

    if not normalized:
        raise ValueError(f"Name '{name}' contains no valid characters for a filename")

    # Add extension if provided
    if extension:
        if not extension.startswith("."):
            extension = f".{extension}"
        normalized = f"{normalized}{extension}"

    return normalized


def slugify(text: str, delimiter: str = "-") -> str:
    """Convert text to a URL-friendly slug.

    Slugs should:
    - Be lowercase
    - Use specified delimiter (default: hyphen)
    - Contain only alphanumeric characters and the delimiter
    - Not have leading/trailing delimiters

    Args:
        text: The text to slugify (e.g., "How to Use Victor CLI")
        delimiter: Delimiter character to use (default: "-")

    Returns:
        URL-friendly slug (e.g., "how-to-use-victor-cli")

    Raises:
        ValueError: If text is empty or contains only special characters

    Examples:
        >>> slugify("How to Use Victor CLI")
        'how-to-use-victor-cli'
        >>> slugify("Python Programming 101")
        'python-programming-101'
        >>> slugify("API Controller", delimiter="_")
        'api_controller'
        >>> slugify("  Extra   Spaces  ")
        'extra-spaces'
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    # Convert to lowercase
    normalized = text.lower()

    # Replace whitespace with delimiter
    normalized = _WHITESPACE_PATTERN.sub(delimiter, normalized)

    # Remove non-alphanumeric characters except delimiter
    if delimiter == "-":
        # For hyphen delimiter, allow hyphens
        normalized = _NON_ALPHANUMERIC_PATTERN.sub("", normalized)
    else:
        # For other delimiters, remove all non-alphanumeric
        pattern = re.compile(rf"[^a-zA-Z0-9{re.escape(delimiter)}]")
        normalized = pattern.sub("", normalized)

    # Collapse multiple consecutive delimiters
    delimiter_pattern = re.compile(f"{re.escape(delimiter)}+")
    normalized = delimiter_pattern.sub(delimiter, normalized)

    # Remove leading/trailing delimiters
    normalized = normalized.strip(delimiter)

    if not normalized:
        raise ValueError(f"Text '{text}' contains no valid characters for a slug")

    return normalized


def sanitize_class_name(name: str) -> str:
    """Convert a name to PascalCase for use as a Python class name.

    Class names should:
    - Be in PascalCase (each word capitalized)
    - Contain only alphanumeric characters
    - Not have spaces, hyphens, or underscores

    Args:
        name: The name to convert (e.g., "data processor")

    Returns:
        PascalCase class name (e.g., "DataProcessor")

    Raises:
        ValueError: If name is empty or contains only special characters

    Examples:
        >>> sanitize_class_name("data processor")
        'DataProcessor'
        >>> sanitize_class_name("user-authentication-module")
        'UserAuthenticationModule'
        >>> sanitize_class_name("API controller")
        'APIController'
        >>> sanitize_class_name("XML parser")
        'XMLParser'
    """
    if not name or not name.strip():
        raise ValueError("Class name cannot be empty")

    # Remove any non-alphanumeric characters (keep spaces and word boundaries)
    normalized = re.sub(r"[^a-zA-Z0-9\s_\-]", "", name)

    # Split on whitespace, underscores, and hyphens
    words = re.split(r"[\s_\-]+", normalized.strip())

    # Capitalize first letter of each word, preserve the rest
    words = [word[0].upper() + word[1:] if word else "" for word in words if word]

    if not words:
        raise ValueError(f"Name '{name}' contains no valid characters for a class name")

    # Join words together
    return "".join(words)


def normalize_for_test_filename(feature_name: str) -> str:
    """Normalize a feature name for use as a test filename.

    Test filenames should:
    - Start with "test_"
    - Be lowercase
    - Use underscores instead of spaces
    - Contain only alphanumeric characters and underscores
    - End with ".py"

    Args:
        feature_name: The feature name to normalize (e.g., "User Authentication")

    Returns:
        Normalized test filename (e.g., "test_user_authentication.py")

    Raises:
        ValueError: If feature_name is empty or contains only special characters

    Examples:
        >>> normalize_for_test_filename("User Authentication")
        'test_user_authentication.py'
        >>> normalize_for_test_filename("Data Processor")
        'test_data_processor.py'
        >>> normalize_for_test_filename("API-Controller")
        'test_api_controller.py'
        >>> normalize_for_test_filename("  Multiple   Words  ")
        'test_multiple_words.py'
    """
    if not feature_name or not feature_name.strip():
        raise ValueError("Feature name cannot be empty")

    # Use normalize_for_filename to get snake_case
    base_name = normalize_for_filename(feature_name)

    # Prepend "test_" prefix
    return f"test_{base_name}.py"


class TextNormalizationPresets:
    """Predefined normalization configurations for common use cases.

    These presets provide semantic names for common text normalization patterns.
    All preset methods are static and return normalized strings.
    """

    @staticmethod
    def git_feature_branch(name: str) -> str:
        """Normalize for git feature branch with 'feature/' prefix.

        Use for: Creating feature branches from feature descriptions.

        Example:
            >>> TextNormalizationPresets.git_feature_branch("User Authentication")
            'feature/user-authentication'
        """
        return normalize_for_git_branch(name, prefix="feature/")

    @staticmethod
    def git_bugfix_branch(name: str) -> str:
        """Normalize for git bugfix branch with 'bugfix/' prefix.

        Use for: Creating bugfix branches from bug descriptions.

        Example:
            >>> TextNormalizationPresets.git_bugfix_branch("Memory Leak")
            'bugfix/memory-leak'
        """
        return normalize_for_git_branch(name, prefix="bugfix/")

    @staticmethod
    def python_source_file(name: str) -> str:
        """Normalize for Python source file with .py extension.

        Use for: Creating Python module filenames.

        Example:
            >>> TextNormalizationPresets.python_source_file("Data Processor")
            'data_processor.py'
        """
        return normalize_for_filename(name, extension=".py")

    @staticmethod
    def python_test_file(name: str) -> str:
        """Normalize for Python test file with test_ prefix and .py extension.

        Use for: Creating test module filenames.

        Example:
            >>> TextNormalizationPresets.python_test_file("User Auth")
            'test_user_auth.py'
        """
        return normalize_for_test_filename(name)

    @staticmethod
    def url_slug(name: str) -> str:
        """Normalize for URL slug with hyphens.

        Use for: Creating URL-friendly slugs for web routes or documentation.

        Example:
            >>> TextNormalizationPresets.url_slug("How to Use Victor")
            'how-to-use-victor'
        """
        return slugify(name, delimiter="-")


__all__ = [
    "normalize_for_git_branch",
    "normalize_for_filename",
    "slugify",
    "sanitize_class_name",
    "normalize_for_test_filename",
    "TextNormalizationPresets",
]
