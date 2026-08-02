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

"""Comprehensive unit tests for ToolCache."""

import pytest

from victor.storage.cache.tool_cache import ToolCache, _hash_args
from victor.storage.cache.config import CacheConfig


class TestHashArgs:
    """Tests for _hash_args function."""

    def test_hash_simple_dict(self):
        """Test hashing a simple dictionary."""
        args = {"key": "value", "number": 42}
        result = _hash_args(args)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA256 produces 64-char hex

    def test_hash_deterministic(self):
        """Test that hashing is deterministic."""
        args = {"query": "test", "limit": 10}
        hash1 = _hash_args(args)
        hash2 = _hash_args(args)
        assert hash1 == hash2

    def test_hash_order_independent(self):
        """Test that key order doesn't affect hash."""
        args1 = {"a": 1, "b": 2}
        args2 = {"b": 2, "a": 1}
        assert _hash_args(args1) == _hash_args(args2)

    def test_hash_different_values(self):
        """Test that different values produce different hashes."""
        args1 = {"query": "test1"}
        args2 = {"query": "test2"}
        assert _hash_args(args1) != _hash_args(args2)

    def test_hash_complex_types(self):
        """Test hashing with complex types."""
        args = {"list": [1, 2, 3], "nested": {"a": "b"}}
        result = _hash_args(args)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_unhashable_fallback(self):
        """Test fallback for unhashable types."""

        # Create an object that json.dumps can't handle
        class Unhashable:
            def __str__(self):
                return "unhashable"

        args = {"obj": Unhashable()}
        result = _hash_args(args)
        assert isinstance(result, str)


class TestToolCacheInit:
    """Tests for ToolCache initialization."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        cache = ToolCache(ttl=60)
        assert cache.ttl == 60
        assert cache.allowlist == set()
        assert cache._path_index == {}

    def test_init_with_allowlist(self):
        """Test initialization with allowlist."""
        allowlist = ["read_file", "code_search"]
        cache = ToolCache(ttl=60, allowlist=allowlist)
        assert cache.allowlist == {"read_file", "code_search"}

    def test_init_with_config(self):
        """Test initialization with custom cache config."""
        config = CacheConfig(memory_max_size=100)
        cache = ToolCache(ttl=60, cache_config=config)
        assert cache.cache is not None


class TestToolCacheGetSet:
    """Tests for ToolCache get/set operations."""

    def test_get_not_in_allowlist(self):
        """Test get returns None for tools not in allowlist."""
        cache = ToolCache(ttl=60, allowlist=["allowed_tool"])
        result = cache.get("not_allowed", {"arg": "value"})
        assert result is None

    def test_set_not_in_allowlist(self):
        """Test set is no-op for tools not in allowlist."""
        cache = ToolCache(ttl=60, allowlist=["allowed_tool"])
        cache.set("not_allowed", {"arg": "value"}, "result")
        # Should not be cached
        result = cache.get("not_allowed", {"arg": "value"})
        assert result is None

    def test_set_and_get_allowed_tool(self):
        """Test set and get for allowed tools."""
        cache = ToolCache(ttl=60, allowlist=["read_file"])
        args = {"path": "/test/file.py"}
        value = {"content": "file contents"}

        cache.set("read_file", args, value)
        result = cache.get("read_file", args)

        assert result == value

    def test_get_miss_returns_none(self):
        """Test get returns None for cache miss."""
        cache = ToolCache(ttl=60, allowlist=["read_file"])
        result = cache.get("read_file", {"path": "/nonexistent"})
        assert result is None

    def test_path_indexing(self):
        """Test that paths are indexed for invalidation."""
        cache = ToolCache(ttl=60, allowlist=["read_file"])
        args = {"path": "/test/file.py"}
        cache.set("read_file", args, "content")

        assert "/test/file.py" in cache._path_index
        assert len(cache._path_index["/test/file.py"]) == 1

    def test_multiple_paths_indexing(self):
        """Test indexing with multiple paths."""
        cache = ToolCache(ttl=60, allowlist=["batch_process"])
        args = {"paths": ["/path1.py", "/path2.py"]}
        cache.set("batch_process", args, "result")

        assert "/path1.py" in cache._path_index
        assert "/path2.py" in cache._path_index

    def test_root_path_indexing(self):
        """Test indexing with root argument."""
        cache = ToolCache(ttl=60, allowlist=["code_search"])
        args = {"query": "test", "root": "/project/src"}
        cache.set("code_search", args, "results")

        assert "/project/src" in cache._path_index


class TestToolCacheClear:
    """Tests for ToolCache clearing operations."""

    def test_clear_all(self):
        """Test clearing all cache entries."""
        cache = ToolCache(ttl=60, allowlist=["read_file"])
        cache.set("read_file", {"path": "/file1.py"}, "content1")
        cache.set("read_file", {"path": "/file2.py"}, "content2")

        cache.clear_all()

        assert cache.get("read_file", {"path": "/file1.py"}) is None
        assert cache.get("read_file", {"path": "/file2.py"}) is None
        assert cache._path_index == {}

    def test_clear_namespaces(self):
        """Test clearing specific namespaces."""
        cache = ToolCache(ttl=60, allowlist=["read_file", "code_search"])
        cache.set("read_file", {"path": "/file.py"}, "content")
        cache.set("code_search", {"query": "test"}, "results")

        cache.clear_namespaces(["read_file"])

        # read_file should be cleared
        assert cache.get("read_file", {"path": "/file.py"}) is None
        # code_search should still exist
        assert cache.get("code_search", {"query": "test"}) == "results"


class TestToolCacheInvalidation:
    """Tests for ToolCache invalidation."""

    def test_invalidate_paths(self):
        """Test invalidating entries by path."""
        cache = ToolCache(ttl=60, allowlist=["read_file"])
        cache.set("read_file", {"path": "/file.py"}, "content")

        cache.invalidate_paths(["/file.py"])

        assert cache.get("read_file", {"path": "/file.py"}) is None
        assert "/file.py" not in cache._path_index

    def test_invalidate_paths_multiple(self):
        """Test invalidating multiple paths."""
        cache = ToolCache(ttl=60, allowlist=["read_file"])
        cache.set("read_file", {"path": "/file1.py"}, "content1")
        cache.set("read_file", {"path": "/file2.py"}, "content2")
        cache.set("read_file", {"path": "/file3.py"}, "content3")

        cache.invalidate_paths(["/file1.py", "/file2.py"])

        assert cache.get("read_file", {"path": "/file1.py"}) is None
        assert cache.get("read_file", {"path": "/file2.py"}) is None
        assert cache.get("read_file", {"path": "/file3.py"}) == "content3"

    def test_invalidate_paths_fallback(self):
        """Test invalidation fallback when paths not indexed."""
        cache = ToolCache(ttl=60, allowlist=["read_file"])
        cache.set("read_file", {"path": "/file.py"}, "content")

        # Invalidate a different path - should fall back to clearing allowlist
        cache.invalidate_paths(["/other.py"])

        # Original entry should be cleared by fallback
        assert cache.get("read_file", {"path": "/file.py"}) is None

    def test_invalidate_by_tool(self):
        """Test invalidating all entries for a specific tool."""
        cache = ToolCache(ttl=60, allowlist=["read_file", "code_search"])
        cache.set("read_file", {"path": "/file1.py"}, "content1")
        cache.set("read_file", {"path": "/file2.py"}, "content2")
        cache.set("code_search", {"query": "test"}, "results")

        cache.invalidate_by_tool("read_file")

        # read_file entries should be cleared
        assert cache.get("read_file", {"path": "/file1.py"}) is None
        assert cache.get("read_file", {"path": "/file2.py"}) is None
        # code_search should remain
        assert cache.get("code_search", {"query": "test"}) == "results"

    def test_invalidate_by_tool_cleans_path_index(self):
        """Test that invalidate_by_tool cleans up path index."""
        cache = ToolCache(ttl=60, allowlist=["read_file", "code_search"])
        cache.set("read_file", {"path": "/file.py"}, "content")
        cache.set("code_search", {"query": "test", "root": "/file.py"}, "results")

        # Both tools reference /file.py
        assert "/file.py" in cache._path_index
        assert len(cache._path_index["/file.py"]) == 2

        cache.invalidate_by_tool("read_file")

        # Path index should only have code_search entry now
        assert "/file.py" in cache._path_index
        assert len(cache._path_index["/file.py"]) == 1
        assert all("read_file:" not in ref for ref in cache._path_index["/file.py"])


class TestToolCacheKey:
    """Tests for cache key generation."""

    def test_key_generation(self):
        """Test that key is tuple of (tool_name, args_hash)."""
        cache = ToolCache(ttl=60, allowlist=["test_tool"])
        name, hashed = cache._key("test_tool", {"arg": "value"})

        assert name == "test_tool"
        assert isinstance(hashed, str)
        assert len(hashed) == 64

    def test_key_consistency(self):
        """Test that same args produce same key."""
        cache = ToolCache(ttl=60, allowlist=["test_tool"])
        key1 = cache._key("test_tool", {"a": 1, "b": 2})
        key2 = cache._key("test_tool", {"b": 2, "a": 1})

        assert key1 == key2


class TestToolCacheEdgeCases:
    """Tests for edge cases."""

    def test_empty_allowlist(self):
        """Test cache with empty allowlist."""
        cache = ToolCache(ttl=60)

        cache.set("any_tool", {"arg": "value"}, "result")
        result = cache.get("any_tool", {"arg": "value"})

        assert result is None

    def test_empty_args(self):
        """Test caching with empty args."""
        cache = ToolCache(ttl=60, allowlist=["no_args_tool"])

        cache.set("no_args_tool", {}, "result")
        result = cache.get("no_args_tool", {})

        assert result == "result"

    def test_none_value(self):
        """Test caching None value."""
        cache = ToolCache(ttl=60, allowlist=["null_result"])

        cache.set("null_result", {"arg": "value"}, None)
        cache.get("null_result", {"arg": "value"})

        # None can be a valid cached value
        # Behavior depends on underlying cache implementation

    def test_large_value(self):
        """Test caching large values."""
        cache = ToolCache(ttl=60, allowlist=["large_result"])
        large_value = "x" * 100000  # 100KB string

        cache.set("large_result", {"arg": "value"}, large_value)
        result = cache.get("large_result", {"arg": "value"})

        assert result == large_value

    def test_root_already_in_path_values(self):
        """Test that root is not duplicated in path_values (covers line 57)."""
        cache = ToolCache(ttl=60, allowlist=["code_search"])
        # path == root, so root shouldn't be added again
        args = {"path": "/project", "root": "/project", "query": "test"}
        cache.set("code_search", args, "results")

        # Only one entry for /project in path_index
        assert "/project" in cache._path_index
        # There should be only one reference, not duplicate
        assert len(cache._path_index["/project"]) == 1

    def test_hash_args_json_exception(self):
        """Test _hash_args fallback when json.dumps fails (covers lines 13-14)."""

        # Create an object that raises an exception during JSON serialization
        class BadObject:
            def __str__(self):
                return "bad_object_string"

            def __repr__(self):
                return "BadObject()"

        args = {"obj": BadObject()}
        # Should fall back to str() representation
        result = _hash_args(args)
        assert isinstance(result, str)
        assert len(result) == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
