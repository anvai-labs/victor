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

"""Tests for ContentHasher utility.

This test suite verifies that ContentHasher provides consistent hashing
with configurable normalization for deduplication use cases.
"""

import pytest
from victor.core.utils.content_hasher import ContentHasher, HasherPresets


class TestContentHasherBasic:
    """Test basic ContentHasher functionality."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        hasher = ContentHasher()

        assert hasher.normalize_whitespace is True
        assert hasher.case_insensitive is False
        assert hasher.hash_length == 16
        assert hasher.remove_punctuation is False

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        hasher = ContentHasher(
            normalize_whitespace=False,
            case_insensitive=True,
            hash_length=12,
            remove_punctuation=True,
        )

        assert hasher.normalize_whitespace is False
        assert hasher.case_insensitive is True
        assert hasher.hash_length == 12
        assert hasher.remove_punctuation is True

    def test_hash_length_validation(self):
        """Test that hash_length is validated."""
        with pytest.raises(ValueError, match="hash_length must be 1-64"):
            ContentHasher(hash_length=0)

        with pytest.raises(ValueError, match="hash_length must be 1-64"):
            ContentHasher(hash_length=65)

    def test_hash_basic(self):
        """Test basic hashing."""
        hasher = ContentHasher()
        hash1 = hasher.hash("hello world")
        hash2 = hasher.hash("hello world")

        assert hash1 == hash2
        assert len(hash1) == 16  # default hash_length

    def test_hash_empty_content(self):
        """Test hashing empty content."""
        hasher = ContentHasher()

        assert hasher.hash("") == ""
        assert hasher.hash(None) == ""

    def test_hash_length_respected(self):
        """Test that hash_length is respected."""
        hasher_short = ContentHasher(hash_length=8)
        hasher_long = ContentHasher(hash_length=32)

        hash_short = hasher_short.hash("test")
        hash_long = hasher_long.hash("test")

        assert len(hash_short) == 8
        assert len(hash_long) == 32
        # Short hash should be prefix of long hash
        assert hash_long.startswith(hash_short)


class TestWhitespaceNormalization:
    """Test whitespace normalization."""

    def test_whitespace_normalization_enabled(self):
        """Test that whitespace normalization works."""
        hasher = ContentHasher(normalize_whitespace=True)

        hash1 = hasher.hash("hello  world")
        hash2 = hasher.hash("hello world")
        hash3 = hasher.hash("hello\t\nworld")

        # All should be normalized to same hash
        assert hash1 == hash2 == hash3

    def test_whitespace_normalization_disabled(self):
        """Test that whitespace is preserved when normalization disabled."""
        hasher = ContentHasher(normalize_whitespace=False)

        hash1 = hasher.hash("hello  world")
        hash2 = hasher.hash("hello world")

        # Different whitespace should produce different hashes
        assert hash1 != hash2

    def test_leading_trailing_whitespace(self):
        """Test that leading/trailing whitespace is handled."""
        hasher = ContentHasher(normalize_whitespace=True)

        hash1 = hasher.hash("  hello world  ")
        hash2 = hasher.hash("hello world")

        # Should be normalized to same hash
        assert hash1 == hash2


class TestCaseSensitivity:
    """Test case sensitivity."""

    def test_case_sensitive(self):
        """Test case sensitive hashing."""
        hasher = ContentHasher(case_insensitive=False)

        hash1 = hasher.hash("Hello")
        hash2 = hasher.hash("hello")

        assert hash1 != hash2

    def test_case_insensitive(self):
        """Test case insensitive hashing."""
        hasher = ContentHasher(case_insensitive=True)

        hash1 = hasher.hash("Hello")
        hash2 = hasher.hash("hello")
        hash3 = hasher.hash("HELLO")

        assert hash1 == hash2 == hash3


class TestPunctuationRemoval:
    """Test punctuation removal."""

    def test_punctuation_removal_enabled(self):
        """Test that trailing punctuation is removed."""
        hasher = ContentHasher(remove_punctuation=True)

        hash1 = hasher.hash("Hello")
        hash2 = hasher.hash("Hello.")
        hash3 = hasher.hash("Hello,")
        hash4 = hasher.hash("Hello;")
        hash5 = hasher.hash("Hello:")

        # All should match
        assert hash1 == hash2 == hash3 == hash4 == hash5

    def test_punctuation_removal_disabled(self):
        """Test that punctuation is preserved."""
        hasher = ContentHasher(remove_punctuation=False)

        hash1 = hasher.hash("Hello")
        hash2 = hasher.hash("Hello.")

        assert hash1 != hash2

    def test_punctuation_middle_preserved(self):
        """Test that punctuation in middle is preserved."""
        hasher = ContentHasher(remove_punctuation=True)

        hash1 = hasher.hash("Hello, world")
        hash2 = hasher.hash("Hello world")

        # Comma in middle should not be removed
        assert hash1 != hash2


class TestDictHashing:
    """Test dictionary hashing."""

    def test_hash_dict_basic(self):
        """Test basic dictionary hashing."""
        hasher = ContentHasher()

        dict1 = {"a": 1, "b": 2}
        hash1 = hasher.hash_dict(dict1)

        assert len(hash1) == 16
        assert hash1  # Non-empty

    def test_hash_dict_order_independent(self):
        """Test that dictionary order doesn't affect hash."""
        hasher = ContentHasher()

        dict1 = {"a": 1, "b": 2, "c": 3}
        dict2 = {"c": 3, "a": 1, "b": 2}
        dict3 = {"b": 2, "c": 3, "a": 1}

        hash1 = hasher.hash_dict(dict1)
        hash2 = hasher.hash_dict(dict2)
        hash3 = hasher.hash_dict(dict3)

        assert hash1 == hash2 == hash3

    def test_hash_dict_different_values(self):
        """Test that different values produce different hashes."""
        hasher = ContentHasher()

        dict1 = {"a": 1, "b": 2}
        dict2 = {"a": 1, "b": 3}

        hash1 = hasher.hash_dict(dict1)
        hash2 = hasher.hash_dict(dict2)

        assert hash1 != hash2

    def test_hash_dict_nested(self):
        """Test hashing nested dictionaries."""
        hasher = ContentHasher()

        dict1 = {"a": 1, "b": {"c": 2, "d": 3}}
        dict2 = {"b": {"d": 3, "c": 2}, "a": 1}

        hash1 = hasher.hash_dict(dict1)
        hash2 = hasher.hash_dict(dict2)

        assert hash1 == hash2

    def test_hash_dict_with_non_json_types(self):
        """Test hashing dict with non-JSON types using default=str."""
        from pathlib import Path

        hasher = ContentHasher()

        dict1 = {"path": Path("/foo/bar")}
        hash1 = hasher.hash_dict(dict1)

        assert hash1  # Should not raise error


class TestListHashing:
    """Test list hashing."""

    def test_hash_list_basic(self):
        """Test basic list hashing."""
        hasher = ContentHasher()

        list1 = ["foo", "bar", "baz"]
        hash1 = hasher.hash_list(list1)

        assert len(hash1) == 16
        assert hash1

    def test_hash_list_order_independent(self):
        """Test that list order is normalized."""
        hasher = ContentHasher()

        list1 = ["foo", "bar", "baz"]
        list2 = ["baz", "foo", "bar"]
        list3 = ["bar", "baz", "foo"]

        hash1 = hasher.hash_list(list1)
        hash2 = hasher.hash_list(list2)
        hash3 = hasher.hash_list(list3)

        # Lists are sorted before hashing, so order shouldn't matter
        assert hash1 == hash2 == hash3

    def test_hash_list_different_items(self):
        """Test that different items produce different hashes."""
        hasher = ContentHasher()

        list1 = ["foo", "bar"]
        list2 = ["foo", "baz"]

        hash1 = hasher.hash_list(list1)
        hash2 = hasher.hash_list(list2)

        assert hash1 != hash2


class TestHashBlock:
    """Test hash_block convenience method."""

    def test_hash_block_no_min_length(self):
        """Test hash_block with no minimum length."""
        hasher = ContentHasher()

        hash1 = hasher.hash_block("short", min_length=0)
        assert hash1  # Should return hash

    def test_hash_block_meets_min_length(self):
        """Test hash_block when content meets minimum length."""
        hasher = ContentHasher()

        hash1 = hasher.hash_block("long enough content here", min_length=10)
        assert hash1

    def test_hash_block_below_min_length(self):
        """Test hash_block when content is below minimum length."""
        hasher = ContentHasher()

        hash1 = hasher.hash_block("short", min_length=20)
        assert hash1 is None

    def test_hash_block_strips_before_checking(self):
        """Test that hash_block strips whitespace before checking length."""
        hasher = ContentHasher()

        # "  short  " is 10 chars but "short" is 5
        hash1 = hasher.hash_block("  short  ", min_length=10)
        assert hash1 is None


class TestHasherPresets:
    """Test HasherPresets configurations."""

    def test_text_fuzzy_preset(self):
        """Test text_fuzzy preset."""
        hasher = HasherPresets.text_fuzzy()

        assert hasher.normalize_whitespace is True
        assert hasher.case_insensitive is True
        assert hasher.hash_length == 12
        assert hasher.remove_punctuation is True

        # Verify it works for fuzzy matching
        hash1 = hasher.hash("Hello  World.")
        hash2 = hasher.hash("hello world")
        assert hash1 == hash2

    def test_text_strict_preset(self):
        """Test text_strict preset."""
        hasher = HasherPresets.text_strict()

        assert hasher.normalize_whitespace is True
        assert hasher.case_insensitive is False
        assert hasher.hash_length == 12
        assert hasher.remove_punctuation is False

        # Whitespace normalized but case preserved
        hash1 = hasher.hash("Hello  World")
        hash2 = hasher.hash("Hello World")
        hash3 = hasher.hash("hello world")

        assert hash1 == hash2
        assert hash1 != hash3  # Case matters

    def test_exact_match_preset(self):
        """Test exact_match preset."""
        hasher = HasherPresets.exact_match()

        assert hasher.normalize_whitespace is False
        assert hasher.case_insensitive is False
        assert hasher.hash_length == 16
        assert hasher.remove_punctuation is False

        # Everything must match exactly
        hash1 = hasher.hash("Hello  World")
        hash2 = hasher.hash("Hello World")
        assert hash1 != hash2

    def test_query_semantic_preset(self):
        """Test query_semantic preset."""
        hasher = HasherPresets.query_semantic()

        assert hasher.normalize_whitespace is True
        assert hasher.case_insensitive is True
        assert hasher.hash_length == 16
        assert hasher.remove_punctuation is False

        # Case and whitespace normalized
        hash1 = hasher.hash("Tool  Registration")
        hash2 = hasher.hash("tool registration")
        assert hash1 == hash2


class TestIntegrationWithDeduplicators:
    """Integration tests verifying ContentHasher works with deduplicators."""

    def test_output_deduplicator_uses_content_hasher(self):
        """Test that OutputDeduplicator uses ContentHasher."""
        from victor.agent.output_deduplicator import OutputDeduplicator

        dedup = OutputDeduplicator(min_block_length=20)

        # Verify hasher is configured correctly
        assert hasattr(dedup, "_hasher")
        assert isinstance(dedup._hasher, ContentHasher)
        assert dedup._hasher.normalize_whitespace is True
        assert dedup._hasher.case_insensitive is True
        assert dedup._hasher.hash_length == 12

        # Verify deduplication works
        text = "This is a block with enough content.\n\nThis is a block with enough content."
        result = dedup.process(text)

        assert result.count("This is a block with enough content.") == 1

    def test_streaming_deduplicator_uses_content_hasher(self):
        """Test that StreamingDeduplicator uses ContentHasher."""
        from victor.agent.output_deduplicator import StreamingDeduplicator

        dedup = StreamingDeduplicator(min_block_length=20)

        # Verify hasher is configured correctly
        assert hasattr(dedup, "_hasher")
        assert isinstance(dedup._hasher, ContentHasher)

    def test_tool_deduplication_tracker_uses_content_hasher(self):
        """Test that ToolDeduplicationTracker uses ContentHasher."""
        from victor.agent.tool_call_tracker import (
            ToolCallTracker as ToolDeduplicationTracker,
        )

        tracker = ToolDeduplicationTracker()

        # Verify hasher is configured correctly
        assert hasattr(tracker, "_hasher")
        assert isinstance(tracker._hasher, ContentHasher)
        assert tracker._hasher.normalize_whitespace is False
        assert tracker._hasher.case_insensitive is False
        assert tracker._hasher.hash_length == 16

        # Verify exact duplicate detection works
        tracker.add_call("read_file", {"path": "/foo.py"})
        is_dup = tracker.is_redundant("read_file", {"path": "/foo.py"})
        assert is_dup is True

        # Different path should not be duplicate
        is_dup2 = tracker.is_redundant("read_file", {"path": "/bar.py"})
        assert is_dup2 is False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_hash_very_long_content(self):
        """Test hashing very long content."""
        hasher = ContentHasher()

        long_content = "a" * 100000
        hash1 = hasher.hash(long_content)

        assert len(hash1) == 16

    def test_hash_unicode_content(self):
        """Test hashing Unicode content."""
        hasher = ContentHasher()

        hash1 = hasher.hash("Hello 世界")
        hash2 = hasher.hash("Hello 世界")

        assert hash1 == hash2

    def test_hash_special_characters(self):
        """Test hashing special characters."""
        hasher = ContentHasher()

        hash1 = hasher.hash("!@#$%^&*()")
        hash2 = hasher.hash("!@#$%^&*()")

        assert hash1 == hash2

    def test_hash_multiline_content(self):
        """Test hashing multiline content."""
        hasher = ContentHasher(normalize_whitespace=True)

        hash1 = hasher.hash("line1\nline2\nline3")
        hash2 = hasher.hash("line1 line2 line3")

        # With whitespace normalization, newlines should become spaces
        assert hash1 == hash2


class TestPerformance:
    """Test performance characteristics."""

    def test_hashing_is_fast(self):
        """Test that hashing is fast enough for production use."""
        import time

        hasher = ContentHasher()

        start = time.time()
        for i in range(1000):
            hasher.hash(f"content {i}")
        elapsed = time.time() - start

        # 1000 hashes should complete in under 0.5 seconds
        assert elapsed < 0.5

    def test_dict_hashing_is_fast(self):
        """Test that dictionary hashing is fast."""
        import time

        hasher = ContentHasher()

        test_dict = {"key1": "value1", "key2": "value2", "key3": "value3"}

        start = time.time()
        for _ in range(1000):
            hasher.hash_dict(test_dict)
        elapsed = time.time() - start

        # 1000 dict hashes should complete in under 0.5 seconds
        assert elapsed < 0.5
