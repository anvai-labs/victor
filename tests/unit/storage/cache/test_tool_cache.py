"""Owned connection cleanup preserves persisted tool-cache data."""

import sqlite3
from unittest.mock import MagicMock

import pytest

from victor.storage.cache.config import CacheConfig
from victor.storage.cache.tool_cache import ToolCache


def test_close_releases_handle_and_preserves_persisted_entries(tmp_path):
    config = CacheConfig(disk_path=tmp_path, enable_memory=False)
    cache = ToolCache(60, ["read"], config, MagicMock())
    cache.set("read", {"path": "a.py"}, "persisted")
    db = cache.cache._disk_cache._db
    cache.close()
    cache.close()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        db.execute("SELECT 1")
    reopened = ToolCache(60, ["read"], config, MagicMock())
    try:
        assert reopened.get("read", {"path": "a.py"}) == "persisted"
    finally:
        reopened.close()


def test_owned_cache_close_failure_is_not_hidden(tmp_path):
    cache = ToolCache(60, [], CacheConfig(disk_path=tmp_path), MagicMock())
    disk = cache.cache._disk_cache
    real_close = disk.close
    disk.close = MagicMock(side_effect=OSError("disk close failed"))
    try:
        with pytest.raises(OSError, match="disk close failed"):
            cache.close()
    finally:
        real_close()
