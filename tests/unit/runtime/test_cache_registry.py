"""Tests for unified CacheRegistry (OBS-1)."""

import pytest

from victor.runtime.cache_registry import (
    CacheCategory,
    CacheEntry,
    CacheRegistry,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset singleton between tests."""
    CacheRegistry.reset()
    yield
    CacheRegistry.reset()


class TestCacheRegistryBasics:
    def test_singleton(self):
        r1 = CacheRegistry.get_instance()
        r2 = CacheRegistry.get_instance()
        assert r1 is r2

    def test_register_and_list(self):
        reg = CacheRegistry.get_instance()
        reg.register("cache_a", {}, category="tool")
        reg.register("cache_b", {}, category="embedding")
        assert sorted(reg.list_caches()) == ["cache_a", "cache_b"]

    def test_unregister(self):
        reg = CacheRegistry.get_instance()
        reg.register("temp", {})
        assert reg.unregister("temp") is True
        assert reg.unregister("temp") is False
        assert reg.list_caches() == []


class TestCacheInvalidation:
    def test_invalidate_all(self):
        reg = CacheRegistry.get_instance()
        cleared = []
        reg.register("a", None, invalidate_fn=lambda: cleared.append("a"))
        reg.register("b", None, invalidate_fn=lambda: cleared.append("b"))
        count = reg.invalidate_all()
        assert count == 2
        assert sorted(cleared) == ["a", "b"]

    def test_invalidate_category(self):
        reg = CacheRegistry.get_instance()
        cleared = []
        reg.register(
            "tool1",
            None,
            category="tool",
            invalidate_fn=lambda: cleared.append("tool1"),
        )
        reg.register(
            "emb1",
            None,
            category="embedding",
            invalidate_fn=lambda: cleared.append("emb1"),
        )
        count = reg.invalidate_category("tool")
        assert count == 1
        assert cleared == ["tool1"]

    def test_invalidate_by_name(self):
        reg = CacheRegistry.get_instance()
        cleared = []
        reg.register("specific", None, invalidate_fn=lambda: cleared.append("hit"))
        assert reg.invalidate_by_name("specific") is True
        assert cleared == ["hit"]
        assert reg.invalidate_by_name("nonexistent") is False

    def test_invalidate_uses_cache_clear_method(self):
        """Cache with .clear() method should work without explicit invalidate_fn."""
        reg = CacheRegistry.get_instance()
        cache = {"key": "value"}
        reg.register("dict_cache", cache)
        assert reg.invalidate_by_name("dict_cache") is True
        assert cache == {}

    def test_invalidate_tolerates_failure(self):
        """A failing invalidation should not crash the registry."""
        reg = CacheRegistry.get_instance()

        def boom():
            raise RuntimeError("fail")

        reg.register("bad", None, invalidate_fn=boom)
        reg.register("good", None, invalidate_fn=lambda: None)
        count = reg.invalidate_all()
        assert count == 1  # good succeeded, bad failed


class TestCacheRegistryStatus:
    def test_status_counts(self):
        reg = CacheRegistry.get_instance()
        reg.register("a", {"x": 1}, category="tool")
        reg.register("b", {"y": 2, "z": 3}, category="tool")
        reg.register("c", {}, category="embedding")

        status = reg.get_status()
        assert status.total_caches == 3
        assert status.caches_by_category == {"tool": 2, "embedding": 1}

    def test_status_size_tracking(self):
        reg = CacheRegistry.get_instance()
        reg.register("sized", {"a": 1, "b": 2}, category="general")
        status = reg.get_status()
        sized_entry = [e for e in status.entries if e["name"] == "sized"][0]
        assert sized_entry["size"] == 2

    def test_empty_status(self):
        reg = CacheRegistry.get_instance()
        status = reg.get_status()
        assert status.total_caches == 0
        assert status.total_known_size == 0


class TestCacheEntry:
    def test_invalidate_with_clear(self):
        cache = [1, 2, 3]
        entry = CacheEntry(name="test", cache=cache, category=CacheCategory.GENERAL)
        assert entry.invalidate() is True
        assert cache == []

    def test_invalidate_with_custom_fn(self):
        called = []
        entry = CacheEntry(
            name="test",
            cache=None,
            category=CacheCategory.GENERAL,
            invalidate_fn=lambda: called.append(True),
        )
        assert entry.invalidate() is True
        assert called == [True]

    def test_get_size_with_len(self):
        entry = CacheEntry(name="test", cache={"a": 1, "b": 2}, category=CacheCategory.GENERAL)
        assert entry.get_size() == 2

    def test_get_size_returns_none_for_unsized(self):
        entry = CacheEntry(name="test", cache=object(), category=CacheCategory.GENERAL)
        assert entry.get_size() is None
