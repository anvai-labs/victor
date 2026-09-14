"""Persistent cache round trips and hostile on-disk input regressions."""

import os
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from victor.storage.cache.json_disk_cache import JsonDiskCache
from victor.storage.cache.config import CacheConfig
from victor.storage.cache.tiered_cache import TieredCache


class HostileValue:
    def __reduce__(self):
        return (os.system, ("touch should-never-be-created",))


@pytest.mark.parametrize(
    "value",
    [True, 0, 3.5, "hello", b"\x00\xff", [1, "a"], (2, b"x"), {1: (2, 3), "tag": ["bytes", "x"]}],
)
def test_round_trip_survives_reopen(tmp_path, value):
    cache = JsonDiskCache(str(tmp_path), 100_000)
    cache.set("key", value, expire=60)
    cache.close()
    cache = JsonDiskCache(str(tmp_path), 100_000)
    try:
        assert cache.get("key") == value
    finally:
        cache.close()


def test_hostile_legacy_cache_is_never_migrated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    legacy = tmp_path / "cache.db"
    with sqlite3.connect(legacy) as db:
        db.execute("CREATE TABLE cache (key TEXT, value BLOB)")
        db.execute("INSERT INTO cache VALUES (?, ?)", ("key", pickle.dumps(HostileValue())))
    original = legacy.read_bytes()
    cache = JsonDiskCache(str(tmp_path), 100_000)
    try:
        assert cache.get("key") is None
        cache.set("safe", {"ok": True})
        assert cache.get("safe") == {"ok": True}
        assert legacy.read_bytes() == original
        assert not Path("should-never-be-created").exists()
    finally:
        cache.close()


def test_malicious_or_malformed_current_entry_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = JsonDiskCache(str(tmp_path), 100_000)
    try:
        for payload in [pickle.dumps(HostileValue()), '["python", "os.system"]', "{broken"]:
            with sqlite3.connect(cache.path) as db:
                db.execute(
                    "INSERT OR REPLACE INTO entries VALUES (?, ?, NULL, 0, 0)", ("key", payload)
                )
            assert cache.get("key") is None
            assert "key" not in cache.iterkeys()
        assert not Path("should-never-be-created").exists()
        with pytest.raises(TypeError, match="Unsupported"):
            cache.set("hostile", HostileValue())
    finally:
        cache.close()


def test_ttl_and_eviction_leave_live_entries(tmp_path):
    cache = JsonDiskCache(str(tmp_path), 32)
    try:
        cache.set("expired", 1, expire=-1)
        assert cache.get("expired") is None
        cache.set("first", "a" * 10)
        cache.set("second", "b" * 10)
        assert cache.get("first") is None
        assert cache.get("second") == "b" * 10
        with pytest.raises(ValueError, match="size limit"):
            cache.set("too-large", "x" * 100)
        assert cache.get("second") == "b" * 10
    finally:
        cache.close()


def test_two_connections_and_threads_do_not_lose_writes(tmp_path):
    caches = [JsonDiskCache(str(tmp_path), 100_000) for _ in range(2)]
    try:

        def write(i):
            caches[i % 2].set(f"key{i}", {"value": i})

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(write, range(60)))
        assert len(caches[0]) == 60
        for i in range(60):
            assert caches[1].get(f"key{i}") == {"value": i}
        caches[0].clear()
        assert len(caches[1]) == 0
    finally:
        for cache in caches:
            cache.close()


@pytest.mark.parametrize("replacement", [HostileValue(), "x" * 1000])
def test_memory_only_replacement_cannot_resurrect_old_disk_value(tmp_path, replacement):
    config = CacheConfig(disk_path=tmp_path, disk_max_size=100, memory_max_size=1)
    cache = TieredCache(config)
    try:
        assert cache.set("key", {"version": "old"})
        assert cache.set("key", replacement)
        assert cache.get("key") is replacement
        cache.set("other", "evicts key from memory")
        assert cache.get("key") is None
    finally:
        cache.close()
    reopened = TieredCache(config)
    try:
        assert reopened.get("key") is None
    finally:
        reopened.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission checks")
def test_cache_files_are_private_and_symlinks_are_rejected(tmp_path):
    cache = JsonDiskCache(str(tmp_path / "private"), 100_000)
    try:
        assert cache.directory.stat().st_mode & 0o777 == 0o700
        assert cache.path.stat().st_mode & 0o777 == 0o600
        directory_link = tmp_path / "link"
        directory_link.symlink_to(cache.directory, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            JsonDiskCache(str(directory_link), 100_000)
    finally:
        cache.close()
    other = tmp_path / "other"
    other.mkdir()
    (other / "cache-v2.sqlite3").symlink_to(cache.path)
    with pytest.raises(ValueError, match="symlink"):
        JsonDiskCache(str(other), 100_000)


@pytest.mark.skipif(os.name != "posix", reason="POSIX hardlink and FIFO checks")
def test_preseeded_hardlink_cannot_modify_another_database(tmp_path):
    external = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(external) as db:
        db.execute("CREATE TABLE important (value TEXT)")
        db.execute("INSERT INTO important VALUES ('preserve me')")
    external.chmod(0o644)
    original = external.read_bytes()
    directory = tmp_path / "cache"
    directory.mkdir()
    os.link(external, directory / "cache-v2.sqlite3")
    with pytest.raises(ValueError, match="one link"):
        JsonDiskCache(str(directory), 100_000)
    assert external.read_bytes() == original
    assert external.stat().st_mode & 0o777 == 0o644


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO checks")
def test_nonregular_database_is_rejected_without_blocking(tmp_path):
    os.mkfifo(tmp_path / "cache-v2.sqlite3")
    with pytest.raises(ValueError, match="regular file"):
        JsonDiskCache(str(tmp_path), 100_000)


def test_overwrite_delete_expiry_and_reopen_preserve_size_budget(tmp_path):
    cache = JsonDiskCache(str(tmp_path), 40)
    try:
        cache.set("a", "x" * 20)
        cache.set("a", "x")
        cache.set("b", "y" * 20)
        assert cache.get("a") == "x"  # Replacement released the old allocation.
        cache.delete("b")
        cache.set("expired", "z" * 20, expire=-1)
        cache.set("c", "w" * 20)
        assert cache.get("a") == "x"
    finally:
        cache.close()
    cache = JsonDiskCache(str(tmp_path), 40)
    try:
        cache.set("d", "v" * 20)
        assert cache.get("c") is None
        assert cache.get("d") == "v" * 20
        cache.clear()
        cache.set("whole", "a" * 30)
        assert cache.get("whole") == "a" * 30
    finally:
        cache.close()
