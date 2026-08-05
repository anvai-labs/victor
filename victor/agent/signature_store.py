# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Persistent failed tool signature storage for cross-session learning.

This module provides SQLite-based persistence for failed tool call signatures,
enabling the agent to learn from past failures across sessions:
- Avoid repeating known-failing tool calls
- Track failure patterns for debugging
- Expire old signatures to allow retrying after fixes

Design Principles:
- SQLite for lightweight, file-based persistence
- TTL-based expiration for automatic cleanup
- Thread-safe concurrent access
- Minimal memory footprint (signatures stored on disk)

Database:
    Uses the unified database at ~/.victor/victor.db via victor.core.database.
"""

from __future__ import annotations

import hashlib
from victor.core.json_utils import json_dumps
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from victor.core.database import get_database
from victor.core.schema import Tables

logger = logging.getLogger(__name__)

# Default TTL for failed signatures (7 days)
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60

# Schema version for migrations
SCHEMA_VERSION = 1


@dataclass
class FailedSignature:
    """Record of a failed tool call."""

    tool_name: str
    args_hash: str
    error_message: str
    failure_count: int
    first_seen: float
    last_seen: float
    expires_at: float

    @property
    def is_expired(self) -> bool:
        """Check if signature has expired."""
        return time.time() > self.expires_at


class SignatureStore:
    """SQLite-backed store for failed tool call signatures.

    Provides persistent storage for tool call failures, enabling cross-session
    learning to avoid repeating known-failing calls.

    Example:
        store = SignatureStore(Path("~/.victor/signatures.db"))

        # Check before executing
        if store.is_known_failure("read", {"path": "/nonexistent"}):
            print("This call is known to fail")

        # Record a failure
        store.record_failure("read", {"path": "/nonexistent"}, "File not found")

        # Get failure stats
        stats = store.get_stats()
    """

    @classmethod
    def _get_schema(cls) -> str:
        """Get schema SQL with table constants."""
        return f"""
        CREATE TABLE IF NOT EXISTS {Tables.UI_FAILED_CALL} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            args_hash TEXT NOT NULL,
            args_json TEXT,
            error_message TEXT,
            failure_count INTEGER DEFAULT 1,
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL,
            expires_at REAL NOT NULL,
            UNIQUE(tool_name, args_hash)
        );

        CREATE INDEX IF NOT EXISTS idx_failed_call_tool ON {Tables.UI_FAILED_CALL}(tool_name);
        CREATE INDEX IF NOT EXISTS idx_failed_call_expires ON {Tables.UI_FAILED_CALL}(expires_at);
        CREATE INDEX IF NOT EXISTS idx_failed_call_lookup ON {Tables.UI_FAILED_CALL}(tool_name, args_hash);

        CREATE TABLE IF NOT EXISTS {Tables.SYS_SCHEMA_VERSION} (
            version INTEGER PRIMARY KEY
        );
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_signatures: int = 10000,
    ):
        """Initialize signature store.

        Args:
            db_path: Path to SQLite database - legacy, now uses unified database
            ttl_seconds: Time-to-live for signatures
            max_signatures: Maximum signatures to store (oldest pruned first)
        """
        # Use unified database from victor.core.database
        self._db_manager = get_database()
        self.db_path = self._db_manager.db_path
        self.ttl_seconds = ttl_seconds
        self.max_signatures = max_signatures
        self._lock = threading.RLock()
        self._local = threading.local()

        # In-memory cache for frequent lookups
        self._cache: Dict[str, float] = {}  # signature -> expires_at
        self._cache_ttl = 300  # 5 minutes
        self._cache_time = 0.0

        # Ensure database exists
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection from unified database manager.

        Yields:
            SQLite connection
        """
        try:
            yield self._db_manager.get_connection()
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            with self._get_connection() as conn:
                conn.executescript(self._get_schema())

                # Check/set schema version
                cursor = conn.execute(f"SELECT version FROM {Tables.SYS_SCHEMA_VERSION} LIMIT 1")
                row = cursor.fetchone()
                if row is None:
                    conn.execute(
                        f"INSERT INTO {Tables.SYS_SCHEMA_VERSION} (version) VALUES (?)",
                        (SCHEMA_VERSION,),
                    )
                conn.commit()

    def _compute_hash(self, args: Dict[str, Any]) -> str:
        """Compute deterministic hash of arguments.

        Args:
            args: Tool arguments

        Returns:
            SHA256 hash of normalized arguments
        """
        try:
            # Normalize by sorting keys
            normalized = json_dumps(args, sort_keys=True, default=str)
        except Exception:
            normalized = str(args)

        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def _make_signature_key(self, tool_name: str, args_hash: str) -> str:
        """Create cache key for signature."""
        return f"{tool_name}:{args_hash}"

    def _refresh_cache(self) -> None:
        """Refresh in-memory cache from database."""
        now = time.time()
        if now - self._cache_time < self._cache_ttl:
            return

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    f"SELECT tool_name, args_hash, expires_at FROM {Tables.UI_FAILED_CALL} "
                    "WHERE expires_at > ?",
                    (now,),
                )
                self._cache = {
                    self._make_signature_key(row["tool_name"], row["args_hash"]): row["expires_at"]
                    for row in cursor
                }
                self._cache_time = now

    def is_known_failure(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> bool:
        """Check if a tool call is known to fail.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            True if this exact call has failed before
        """
        args_hash = self._compute_hash(args)
        key = self._make_signature_key(tool_name, args_hash)

        # Check cache first
        self._refresh_cache()
        if key in self._cache:
            if self._cache[key] > time.time():
                return True

        # Fall back to database
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    f"SELECT expires_at FROM {Tables.UI_FAILED_CALL} "
                    "WHERE tool_name = ? AND args_hash = ? AND expires_at > ?",
                    (tool_name, args_hash, time.time()),
                )
                row = cursor.fetchone()
                if row:
                    self._cache[key] = row["expires_at"]
                    return True

        return False

    def record_failure(
        self,
        tool_name: str,
        args: Dict[str, Any],
        error_message: str,
        custom_ttl: Optional[int] = None,
    ) -> None:
        """Record a failed tool call.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            error_message: Error message from failure
            custom_ttl: Optional custom TTL for this signature
        """
        args_hash = self._compute_hash(args)
        now = time.time()
        ttl = custom_ttl if custom_ttl is not None else self.ttl_seconds
        expires_at = now + ttl

        with self._lock:
            with self._get_connection() as conn:
                # Upsert: insert or update
                conn.execute(
                    f"""
                    INSERT INTO {Tables.UI_FAILED_CALL}
                        (tool_name, args_hash, args_json, error_message,
                         failure_count, first_seen, last_seen, expires_at)
                    VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                    ON CONFLICT(tool_name, args_hash) DO UPDATE SET
                        error_message = excluded.error_message,
                        failure_count = failure_count + 1,
                        last_seen = excluded.last_seen,
                        expires_at = excluded.expires_at
                    """,
                    (
                        tool_name,
                        args_hash,
                        json_dumps(args, default=str),
                        error_message[:500],  # Limit error message length
                        now,
                        now,
                        expires_at,
                    ),
                )
                conn.commit()

            # Update cache
            key = self._make_signature_key(tool_name, args_hash)
            self._cache[key] = expires_at

        # Prune if needed
        self._maybe_prune()

    def clear_signature(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> bool:
        """Clear a specific failure signature (e.g., after a fix).

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            True if signature was found and deleted
        """
        args_hash = self._compute_hash(args)

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    f"DELETE FROM {Tables.UI_FAILED_CALL} WHERE tool_name = ? AND args_hash = ?",
                    (tool_name, args_hash),
                )
                conn.commit()
                deleted = cursor.rowcount > 0

            # Update cache
            key = self._make_signature_key(tool_name, args_hash)
            self._cache.pop(key, None)

        return deleted

    def clear_tool(self, tool_name: str) -> int:
        """Clear all signatures for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Number of signatures deleted
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    f"DELETE FROM {Tables.UI_FAILED_CALL} WHERE tool_name = ?",
                    (tool_name,),
                )
                conn.commit()
                deleted = cursor.rowcount

            # Clear from cache
            keys_to_delete = [k for k in self._cache if k.startswith(f"{tool_name}:")]
            for key in keys_to_delete:
                del self._cache[key]

        return deleted

    def clear_all(self) -> int:
        """Clear all signatures.

        Returns:
            Number of signatures deleted
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(f"DELETE FROM {Tables.UI_FAILED_CALL}")
                conn.commit()
                deleted = cursor.rowcount

            self._cache.clear()

        return deleted

    def cleanup_expired(self) -> int:
        """Remove expired signatures.

        Returns:
            Number of signatures removed
        """
        now = time.time()
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    f"DELETE FROM {Tables.UI_FAILED_CALL} WHERE expires_at < ?",
                    (now,),
                )
                conn.commit()
                deleted = cursor.rowcount

            # Clear expired from cache
            self._cache = {k: v for k, v in self._cache.items() if v > now}

        if deleted > 0:
            logger.debug(f"Cleaned up {deleted} expired signatures")

        return deleted

    def _maybe_prune(self) -> None:
        """Prune oldest signatures if over limit."""
        with self._get_connection() as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {Tables.UI_FAILED_CALL}")
            count = cursor.fetchone()[0]

            if count > self.max_signatures:
                # Delete oldest 10%
                to_delete = max(count - self.max_signatures, count // 10)
                conn.execute(
                    f"""
                    DELETE FROM {Tables.UI_FAILED_CALL} WHERE id IN (
                        SELECT id FROM {Tables.UI_FAILED_CALL}
                        ORDER BY last_seen ASC LIMIT ?
                    )
                    """,
                    (to_delete,),
                )
                conn.commit()
                logger.debug(f"Pruned {to_delete} old signatures")

    def get_failures(
        self,
        tool_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[FailedSignature]:
        """Get recorded failures.

        Args:
            tool_name: Optional filter by tool name
            limit: Maximum records to return

        Returns:
            List of FailedSignature records
        """
        with self._get_connection() as conn:
            if tool_name:
                cursor = conn.execute(
                    f"SELECT * FROM {Tables.UI_FAILED_CALL} WHERE tool_name = ? "
                    "ORDER BY last_seen DESC LIMIT ?",
                    (tool_name, limit),
                )
            else:
                cursor = conn.execute(
                    f"SELECT * FROM {Tables.UI_FAILED_CALL} ORDER BY last_seen DESC LIMIT ?",
                    (limit,),
                )

            return [
                FailedSignature(
                    tool_name=row["tool_name"],
                    args_hash=row["args_hash"],
                    error_message=row["error_message"],
                    failure_count=row["failure_count"],
                    first_seen=row["first_seen"],
                    last_seen=row["last_seen"],
                    expires_at=row["expires_at"],
                )
                for row in cursor
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get signature store statistics.

        Returns:
            Dictionary with store statistics
        """
        with self._get_connection() as conn:
            # Total count
            cursor = conn.execute(f"SELECT COUNT(*) FROM {Tables.UI_FAILED_CALL}")
            total = cursor.fetchone()[0]

            # Active (non-expired) count
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM {Tables.UI_FAILED_CALL} WHERE expires_at > ?",
                (time.time(),),
            )
            active = cursor.fetchone()[0]

            # By tool
            cursor = conn.execute(
                f"SELECT tool_name, COUNT(*), SUM(failure_count) "
                f"FROM {Tables.UI_FAILED_CALL} GROUP BY tool_name"
            )
            by_tool = {row[0]: {"signatures": row[1], "total_failures": row[2]} for row in cursor}

            # Most failing
            cursor = conn.execute(
                f"SELECT tool_name, args_hash, failure_count "
                f"FROM {Tables.UI_FAILED_CALL} ORDER BY failure_count DESC LIMIT 5"
            )
            most_failing = [{"tool": row[0], "hash": row[1], "count": row[2]} for row in cursor]

        return {
            "total_signatures": total,
            "active_signatures": active,
            "expired_signatures": total - active,
            "by_tool": by_tool,
            "most_failing": most_failing,
            "cache_size": len(self._cache),
            "db_path": str(self.db_path),
        }

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Global store instance
_global_store: Optional[SignatureStore] = None


def get_signature_store(db_path: Optional[Path] = None) -> SignatureStore:
    """Get global signature store instance.

    Args:
        db_path: Optional custom database path

    Returns:
        SignatureStore instance
    """
    global _global_store

    if _global_store is None:
        _global_store = SignatureStore(db_path)

    return _global_store


def reset_signature_store() -> None:
    """Reset global signature store (for testing)."""
    global _global_store

    if _global_store:
        _global_store.close()
        _global_store = None
