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

"""Identifier extraction utilities for enrichment.

This module provides configurable pattern-based extraction of identifiers
from text, consolidating previously duplicated logic across verticals.

Supports:
- CamelCase identifiers (class names)
- snake_case identifiers (function names)
- Dotted paths (module.function)
- Backtick-quoted identifiers

Example:
    identifiers = extract_identifiers(
        "Check the UserManager class and get_user() function",
        patterns=PATTERNS
    )
    # Returns: ["UserManager", "get_user"]
"""

import re
from typing import Dict, List, Optional, Set

# Default identifier patterns
PATTERNS: Dict[str, str] = {
    "camelcase": r"\b([A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)+)\b",
    "snake_case": r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b",
    "dotted": r"\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\b",
    "quoted": r"`([a-zA-Z_][a-zA-Z0-9_]*)`",
}

# Common words to exclude from identifier extraction
COMMON_WORDS: Set[str] = {
    "the",
    "and",
    "for",
    "not",
    "but",
    "with",
    "from",
    "this",
    "that",
    "have",
    "has",
    "had",
    "are",
    "was",
    "were",
    "been",
    "being",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "may",
    "might",
    "must",
    "need",
    "want",
    "like",
    "just",
    "only",
    "also",
    "even",
    "true",
    "false",
    "none",
    "null",
    "undefined",
    "class",
    "function",
    "method",
    "variable",
    "const",
    "let",
    "var",
    "def",
    "import",
}


def extract_identifiers(
    text: str,
    patterns: Optional[Dict[str, str]] = None,
    min_length: int = 2,
    max_identifiers: int = 50,
    exclude_words: Optional[Set[str]] = None,
) -> List[str]:
    """Extract identifiers from text using configurable patterns.

    Args:
        text: Text to extract identifiers from
        patterns: Dict of pattern_name -> regex_pattern.
                  Defaults to PATTERNS.
        min_length: Minimum identifier length
        max_identifiers: Maximum number of identifiers to return
        exclude_words: Words to exclude. Defaults to COMMON_WORDS.

    Returns:
        List of unique identifiers, preserving order of first occurrence

    Example:
        >>> extract_identifiers("Use UserManager.get_user() for auth")
        ['UserManager', 'get_user', 'UserManager.get_user']
    """
    if not text:
        return []

    patterns = patterns or PATTERNS
    exclude = exclude_words if exclude_words is not None else COMMON_WORDS

    found: List[str] = []

    for pattern_name, pattern in patterns.items():
        try:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) >= min_length and match.lower() not in exclude:
                    found.append(match)
        except re.error:
            # Skip invalid patterns
            continue

    # Deduplicate while preserving order
    seen: Set[str] = set()
    result: List[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result[:max_identifiers]


def extract_camelcase(text: str, min_length: int = 2) -> List[str]:
    """Extract CamelCase identifiers (class names).

    Args:
        text: Text to search
        min_length: Minimum identifier length

    Returns:
        List of CamelCase identifiers
    """
    return extract_identifiers(
        text,
        patterns={"camelcase": PATTERNS["camelcase"]},
        min_length=min_length,
    )


def extract_snake_case(text: str, min_length: int = 2) -> List[str]:
    """Extract snake_case identifiers (function names).

    Args:
        text: Text to search
        min_length: Minimum identifier length

    Returns:
        List of snake_case identifiers
    """
    return extract_identifiers(
        text,
        patterns={"snake_case": PATTERNS["snake_case"]},
        min_length=min_length,
    )


def extract_dotted_paths(text: str) -> List[str]:
    """Extract dotted paths (module.function).

    Args:
        text: Text to search

    Returns:
        List of dotted path identifiers
    """
    return extract_identifiers(
        text,
        patterns={"dotted": PATTERNS["dotted"]},
        min_length=3,  # At least "a.b"
    )


def extract_quoted_identifiers(text: str) -> List[str]:
    """Extract backtick-quoted identifiers.

    Args:
        text: Text to search

    Returns:
        List of quoted identifiers (without backticks)
    """
    return extract_identifiers(
        text,
        patterns={"quoted": PATTERNS["quoted"]},
        min_length=1,
    )


class IdentifierExtractor:
    """Configurable identifier extractor with custom patterns.

    Allows vertical-specific pattern configuration while maintaining
    a consistent extraction interface.

    Example:
        extractor = IdentifierExtractor(
            patterns={"custom": r"\\b(custom_\\w+)\\b"},
            exclude_words={"custom_ignore"}
        )
        identifiers = extractor.extract("Found custom_value here")
    """

    def __init__(
        self,
        patterns: Optional[Dict[str, str]] = None,
        exclude_words: Optional[Set[str]] = None,
        min_length: int = 2,
        max_identifiers: int = 50,
    ) -> None:
        """Initialize extractor with configuration.

        Args:
            patterns: Custom pattern dict or None for defaults
            exclude_words: Words to exclude or None for defaults
            min_length: Minimum identifier length
            max_identifiers: Maximum identifiers to return
        """
        self.patterns = patterns or PATTERNS.copy()
        self.exclude_words = exclude_words or COMMON_WORDS.copy()
        self.min_length = min_length
        self.max_identifiers = max_identifiers

    def extract(self, text: str) -> List[str]:
        """Extract identifiers using configured patterns.

        Args:
            text: Text to extract from

        Returns:
            List of unique identifiers
        """
        return extract_identifiers(
            text,
            patterns=self.patterns,
            min_length=self.min_length,
            max_identifiers=self.max_identifiers,
            exclude_words=self.exclude_words,
        )

    def add_pattern(self, name: str, pattern: str) -> None:
        """Add a custom pattern.

        Args:
            name: Pattern name
            pattern: Regex pattern string
        """
        self.patterns[name] = pattern

    def add_exclude_word(self, word: str) -> None:
        """Add a word to exclude list.

        Args:
            word: Word to exclude (case-insensitive)
        """
        self.exclude_words.add(word.lower())


__all__ = [
    "PATTERNS",
    "COMMON_WORDS",
    "extract_identifiers",
    "extract_camelcase",
    "extract_snake_case",
    "extract_dotted_paths",
    "extract_quoted_identifiers",
    "IdentifierExtractor",
]
