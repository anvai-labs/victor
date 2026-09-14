"""Private SQLite cache that persists data without executing Python deserializers.

The versioned database never opens or migrates legacy diskcache files. Unsupported
objects can still use the memory tier; disk persistence accepts only data values.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sqlite3
import stat
import threading
import time
from typing import Any


def _encode(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is bytes:
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if type(value) in (list, tuple):
        return [type(value).__name__, [_encode(item) for item in value]]
    if type(value) is dict:
        return ["dict", [[_encode(k), _encode(v)] for k, v in value.items()]]
    raise TypeError(f"Unsupported persistent cache type: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Invalid cache value")
    tag, payload = value
    if tag == "bytes" and isinstance(payload, str):
        return base64.b64decode(payload, validate=True)
    if tag in ("list", "tuple") and isinstance(payload, list):
        items = [_decode(item) for item in payload]
        return tuple(items) if tag == "tuple" else items
    if tag == "dict" and isinstance(payload, list):
        return {_decode(k): _decode(v) for k, v in payload}
    raise ValueError("Unknown cache value type")


class JsonDiskCache:
    """Bounded persistent data cache with TTL and process-safe SQLite writes."""

    def __init__(self, directory: str, size_limit: int) -> None:
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError("Cache directory must not be a symlink")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name == "posix":
            if self.directory.stat().st_uid != os.getuid():
                raise PermissionError("Cache directory must belong to the current user")
            self.directory.chmod(0o700)
        self.path = self.directory / "cache-v2.sqlite3"
        if self.path.is_symlink():
            raise ValueError("Cache database must not be a symlink")
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(self.path, flags, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("Cache database must be a regular file with one link")
            if os.name == "posix":
                if info.st_uid != os.getuid():
                    raise PermissionError("Cache database must belong to the current user")
                os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        self._lock = threading.RLock()
        self._limit = size_limit
        self._db = sqlite3.connect(self.path, timeout=15, check_same_thread=False)
        self._db.execute("PRAGMA trusted_schema=OFF")
        self._db.execute("PRAGMA journal_mode=DELETE")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS entries "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL, expires REAL, "
            "updated REAL NOT NULL, size INTEGER NOT NULL)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS expiry ON entries(expires)")
        self._db.execute("CREATE INDEX IF NOT EXISTS age ON entries(updated, key)")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache_size (id INTEGER PRIMARY KEY CHECK (id=1), "
            "bytes INTEGER NOT NULL)"
        )
        # Reconcile once on open. Writes maintain this counter transactionally;
        # they never rescan cached payloads to enforce the payload budget.
        self._db.execute("BEGIN IMMEDIATE")
        self._db.execute(
            "INSERT OR REPLACE INTO cache_size SELECT 1, coalesce(sum(size), 0) FROM entries"
        )
        self._db.commit()

    def _expire(self) -> None:
        self._db.execute("BEGIN IMMEDIATE")
        now = time.time()
        expired = self._db.execute(
            "SELECT coalesce(sum(size), 0) FROM entries WHERE expires <= ?", (now,)
        ).fetchone()[0]
        self._db.execute("DELETE FROM entries WHERE expires <= ?", (now,))
        if expired:
            self._db.execute("UPDATE cache_size SET bytes = bytes - ? WHERE id=1", (expired,))

    def _delete(self, key: str) -> bool:
        row = self._db.execute("SELECT size FROM entries WHERE key = ?", (key,)).fetchone()
        if row is None:
            return False
        self._db.execute("DELETE FROM entries WHERE key = ?", (key,))
        self._db.execute("UPDATE cache_size SET bytes = bytes - ? WHERE id=1", (row[0],))
        return True

    def get(self, key: str) -> Any:
        with self._lock, self._db:
            self._expire()
            row = self._db.execute("SELECT value FROM entries WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            try:
                return _decode(json.loads(row[0]))
            except (ValueError, TypeError, RecursionError):
                self._delete(key)
                return None

    def set(self, key: str, value: Any, expire: float | None = None) -> bool:
        try:
            encoded = json.dumps(_encode(value), allow_nan=False, separators=(",", ":"))
            size = len(encoded.encode("utf-8")) + len(key.encode("utf-8"))
            if size > self._limit:
                raise ValueError("Cache value exceeds disk size limit")
        except (TypeError, ValueError, RecursionError):
            # A newer memory-only value must never reveal the superseded disk
            # value after memory eviction or restart. Commit invalidation before
            # propagating the persistence failure to the tiered cache.
            self.delete(key)
            raise
        now = time.time()
        expires = None if expire is None else now + expire
        with self._lock, self._db:
            self._expire()
            old = self._db.execute("SELECT size FROM entries WHERE key = ?", (key,)).fetchone()
            self._db.execute(
                "INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?, ?)",
                (key, encoded, expires, now, size),
            )
            self._db.execute(
                "UPDATE cache_size SET bytes = bytes + ? WHERE id=1",
                (size - (old[0] if old else 0),),
            )
            total = self._db.execute("SELECT bytes FROM cache_size WHERE id=1").fetchone()[0]
            while total > self._limit:
                oldest, old_size = self._db.execute(
                    "SELECT key, size FROM entries ORDER BY updated, key LIMIT 1"
                ).fetchone()
                self._delete(oldest)
                total -= old_size
        return True

    def delete(self, key: str) -> bool:
        with self._lock, self._db:
            # Acquire SQLite's write lock before reading the entry size.
            self._expire()
            return self._delete(key)

    def __delitem__(self, key: str) -> None:
        if not self.delete(key):
            raise KeyError(key)

    def clear(self) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM entries")
            self._db.execute("UPDATE cache_size SET bytes = 0 WHERE id=1")

    def iterkeys(self) -> list[str]:
        with self._lock, self._db:
            self._expire()
            return [row[0] for row in self._db.execute("SELECT key FROM entries")]

    def __len__(self) -> int:
        with self._lock, self._db:
            self._expire()
            return int(self._db.execute("SELECT count(*) FROM entries").fetchone()[0])

    def volume(self) -> int:
        with self._lock:
            return self.path.stat().st_size

    def close(self) -> None:
        with self._lock:
            self._db.close()
