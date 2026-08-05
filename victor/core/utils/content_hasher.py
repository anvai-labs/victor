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

"""Shared content hashing utility with configurable normalization.

This utility provides consistent hashing for different deduplication needs:
- OutputDeduplicator: Removes repetitive LLM text blocks
- ToolDeduplicationTracker: Prevents redundant tool API calls

The key insight is that both need hashing, but with different normalization.

Design Pattern: Strategy Pattern
================================
The ContentHasher uses configurable normalization strategies to support
different deduplication use cases while maintaining a single, well-tested
hashing implementation.

SOLID Compliance:
- SRP: Single class handles content hashing with normalization
- OCP: New normalization strategies can be added via HasherPresets
- LSP: All presets return ContentHasher instances
- ISP: Minimal interface (hash, hash_dict, hash_list, hash_block)
- DIP: Uses native accelerator protocol when available

Usage Examples:

    # For text deduplication (normalize whitespace, case-insensitive)
    text_hasher = ContentHasher(
        normalize_whitespace=True,
        case_insensitive=True,
        hash_length=12
    )
    hash1 = text_hasher.hash("Hello  World")
    hash2 = text_hasher.hash("hello world")
    assert hash1 == hash2  # Same due to normalization

    # For exact tool call matching (no normalization)
    tool_hasher = ContentHasher(
        normalize_whitespace=False,
        case_insensitive=False,
        hash_length=16
    )
    call_hash = tool_hasher.hash_dict({"tool": "read_file", "path": "/foo"})

Why This Utility Exists:
-----------------------
Before this refactor, OutputDeduplicator and ToolDeduplicationTracker had
duplicate hashing logic (~50 LOC each). This utility extracts the shared
concern (content hashing) while allowing each domain to configure normalization
appropriate to its needs.

Performance Characteristics:
---------------------------
- O(n) where n is content length
- SHA-256 for cryptographic quality (collision resistance)
- Configurable hash length for memory efficiency
- Uses Rust native normalize_block() when available (10-50x faster)
- No external dependencies beyond stdlib
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional

# Try to import native acceleration
_NATIVE_NORMALIZE_AVAILABLE = False
_native_normalize_block = None

try:
    from victor.processing.native import normalize_block as _native_normalize_block
    from victor.processing.native import is_native_available

    _NATIVE_NORMALIZE_AVAILABLE = is_native_available()
except ImportError:
    pass


class ContentHasher:
    """Shared content hashing with configurable normalization.

    This utility provides a consistent hashing mechanism for deduplication
    with pluggable normalization strategies.

    Attributes:
        normalize_whitespace: Collapse multiple whitespace to single space
        case_insensitive: Convert to lowercase before hashing
        hash_length: Number of hex chars to return from hash (1-64)
        remove_punctuation: Remove trailing punctuation before hashing
    """

    def __init__(
        self,
        normalize_whitespace: bool = True,
        case_insensitive: bool = False,
        hash_length: int = 16,
        remove_punctuation: bool = False,
    ):
        """Initialize hasher with normalization options.

        Args:
            normalize_whitespace: If True, collapse multiple whitespace to
                                 single space. Useful for fuzzy text matching
                                 where "hello  world" == "hello world".
            case_insensitive: If True, convert to lowercase before hashing.
                             Useful for case-insensitive deduplication.
            hash_length: Number of hexadecimal chars to return (1-64).
                        Shorter hashes use less memory but have higher
                        collision probability. 12-16 is typical.
            remove_punctuation: If True, remove trailing punctuation (.,;:).
                               Useful for text where "Hello." == "Hello"

        Raises:
            ValueError: If hash_length is not in range [1, 64]
        """
        if not 1 <= hash_length <= 64:
            raise ValueError(f"hash_length must be 1-64, got {hash_length}")

        self.normalize_whitespace = normalize_whitespace
        self.case_insensitive = case_insensitive
        self.hash_length = hash_length
        self.remove_punctuation = remove_punctuation

    def hash(self, content: str) -> str:
        """Generate hash for content with optional normalization.

        The normalization pipeline applies transformations in this order:
        1. Whitespace normalization (if enabled) - uses Rust when available
        2. Punctuation removal (if enabled)
        3. Case conversion (if enabled)
        4. SHA-256 hashing
        5. Truncation to hash_length

        Args:
            content: Content to hash (any string)

        Returns:
            Hexadecimal hash string of specified length, or empty string
            if content is empty/None

        Example:
            >>> hasher = ContentHasher(normalize_whitespace=True)
            >>> hasher.hash("hello  world")
            'a591a6d40bf420...'
            >>> hasher.hash("hello world")
            'a591a6d40bf420...'  # Same hash
        """
        if not content:
            return ""

        normalized = content

        # Apply normalization pipeline
        if self.normalize_whitespace:
            # Use Rust native normalization when available (10-50x faster)
            if (
                _NATIVE_NORMALIZE_AVAILABLE
                and _native_normalize_block is not None
                and self.remove_punctuation
            ):
                # Native normalize_block handles whitespace + punctuation in one pass
                normalized = _native_normalize_block(normalized)
            else:
                # Pure Python fallback
                normalized = re.sub(r"\s+", " ", normalized.strip())
                if self.remove_punctuation:
                    normalized = normalized.rstrip(".,;:")
        elif self.remove_punctuation:
            # Only punctuation removal (no whitespace normalization)
            normalized = normalized.rstrip(".,;:")

        if self.case_insensitive:
            normalized = normalized.lower()

        # Generate SHA-256 hash
        hash_obj = hashlib.sha256(normalized.encode("utf-8"))
        return hash_obj.hexdigest()[: self.hash_length]

    def hash_dict(self, data: Dict[str, Any]) -> str:
        """Hash dictionary by sorting keys and hashing JSON representation.

        This method ensures that dictionary order doesn't affect the hash:
        {"a": 1, "b": 2} produces the same hash as {"b": 2, "a": 1}.

        Args:
            data: Dictionary to hash (must be JSON-serializable)

        Returns:
            Hexadecimal hash string

        Raises:
            TypeError: If data contains non-serializable objects

        Example:
            >>> hasher = ContentHasher()
            >>> hasher.hash_dict({"tool": "read_file", "path": "/foo"})
            'c9f0f895fb98ab...'
        """
        # Sort keys for deterministic ordering
        # Use default=str to handle non-JSON types (e.g., Path objects)
        serialized = json.dumps(data, sort_keys=True, default=str)
        return self.hash(serialized)

    def hash_list(self, items: List[Any]) -> str:
        """Hash list by sorting items and hashing string representation.

        Args:
            items: List to hash (items must be sortable and stringable)

        Returns:
            Hexadecimal hash string

        Example:
            >>> hasher = ContentHasher()
            >>> hasher.hash_list(["foo", "bar", "baz"])
            '5f4dcc3b5aa76...'
        """
        # Sort items for deterministic ordering
        serialized = str(sorted(str(item) for item in items))
        return self.hash(serialized)

    def hash_block(self, block: str, min_length: int = 0) -> Optional[str]:
        """Hash a content block with optional minimum length check.

        This is a convenience method for block-based deduplication where
        short blocks should be skipped.

        Args:
            block: Content block to hash
            min_length: Minimum block length to hash (0 = no minimum)

        Returns:
            Hash string if block meets minimum length, None otherwise

        Example:
            >>> hasher = ContentHasher()
            >>> hasher.hash_block("short", min_length=10)
            None  # Block too short
            >>> hasher.hash_block("long enough content", min_length=10)
            'e3b0c44298fc1...'
        """
        if min_length > 0 and len(block.strip()) < min_length:
            return None
        return self.hash(block)

    def normalize(self, content: str) -> str:
        """Normalize content without hashing (for inspection/debugging).

        Applies the same normalization pipeline as hash() but returns
        the normalized string instead of its hash. Useful for debugging
        and testing normalization behavior.

        Args:
            content: Content to normalize

        Returns:
            Normalized content string

        Example:
            >>> hasher = ContentHasher(normalize_whitespace=True, case_insensitive=True)
            >>> hasher.normalize("Hello  World")
            'hello world'
        """
        if not content:
            return ""

        normalized = content

        # Apply normalization pipeline (same as hash())
        if self.normalize_whitespace:
            if (
                _NATIVE_NORMALIZE_AVAILABLE
                and _native_normalize_block is not None
                and self.remove_punctuation
            ):
                normalized = _native_normalize_block(normalized)
            else:
                normalized = re.sub(r"\s+", " ", normalized.strip())
                if self.remove_punctuation:
                    normalized = normalized.rstrip(".,;:")
        elif self.remove_punctuation:
            normalized = normalized.rstrip(".,;:")

        if self.case_insensitive:
            normalized = normalized.lower()

        return normalized


# Predefined hasher configurations for common use cases
class HasherPresets:
    """Predefined hasher configurations for common use cases.

    These presets provide semantic names for common hashing patterns.
    """

    @staticmethod
    def text_fuzzy() -> ContentHasher:
        """Hasher for fuzzy text matching (whitespace + case insensitive).

        Use for: Text deduplication where whitespace and case don't matter.
        Example: "Hello  World" == "hello world"
        """
        return ContentHasher(
            normalize_whitespace=True,
            case_insensitive=True,
            hash_length=12,
            remove_punctuation=True,
        )

    @staticmethod
    def text_strict() -> ContentHasher:
        """Hasher for strict text matching (whitespace normalized only).

        Use for: Text deduplication where case matters but whitespace doesn't.
        Example: "Hello  World" == "Hello World" but != "hello world"
        """
        return ContentHasher(
            normalize_whitespace=True,
            case_insensitive=False,
            hash_length=12,
            remove_punctuation=False,
        )

    @staticmethod
    def exact_match() -> ContentHasher:
        """Hasher for exact matching (no normalization).

        Use for: Tool call deduplication, API signatures, exact content matching.
        Example: "Hello  World" != "Hello World"
        """
        return ContentHasher(
            normalize_whitespace=False,
            case_insensitive=False,
            hash_length=16,
            remove_punctuation=False,
        )

    @staticmethod
    def query_semantic() -> ContentHasher:
        """Hasher for semantic query matching (whitespace + case insensitive).

        Use for: Search query deduplication where "Tool Registration" should
        match "tool registration".
        """
        return ContentHasher(
            normalize_whitespace=True,
            case_insensitive=True,
            hash_length=16,
            remove_punctuation=False,
        )


__all__ = ["ContentHasher", "HasherPresets"]
