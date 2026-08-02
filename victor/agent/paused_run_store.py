# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Store of durably-paused single-agent runs (FEP-0029).

When a turn parks on a policy ASK, the turn boundary records a :class:`PausedRun` keyed by an opaque
``run_id`` and surfaces that token on the ``awaiting_approval`` result. Two interchangeable backends
implement :class:`PausedRunStoreProtocol`:

- :class:`ProjectDbPausedRunStore` (default) — a ``project.db``-backed ``paused_run`` table so pauses
  **survive a process restart** (Phase 2). Self-manages its table (idempotent ``CREATE TABLE IF NOT
  EXISTS``) against the project database, mirroring ``ConversationStore``; thread-local connection.
- :class:`InMemoryPausedRunStore` — a process-local map (Phase 1), used for tests and as a fallback
  when no project database is available.

The transcript itself is *not* stored here — it already lives durably in ``ConversationStore`` keyed
by ``session_id``; a paused run only records the pending gated tool + approval request that a resume
needs on top of the transcript.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

#: FEP-0029: a pending pause older than this (seconds) is considered stale — the resume seam expires
#: it (opportunistic GC) rather than acting on a day-old approval. Generous by default (24h).
DEFAULT_PAUSE_TTL_SECONDS: float = 24 * 60 * 60


@dataclass
class PausedRun:
    """A single durable pause point awaiting a human approval decision (FEP-0029)."""

    run_id: str
    session_id: Optional[str]
    agent_id: Optional[str]
    approval_request: Dict[str, Any]
    # Best-effort record of the gated action (tool_name + arguments), read from the approval
    # request context — Phase 3 uses this to faithfully replay the persisted call on resume.
    pending_tool: Optional[Dict[str, Any]] = None
    status: str = "awaiting_approval"  # 'awaiting_approval' | 'resumed' | 'cancelled'
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PausedRunStoreProtocol(Protocol):
    """The interchangeable interface implemented by both paused-run backends (FEP-0029)."""

    def save(
        self,
        *,
        session_id: Optional[str],
        agent_id: Optional[str],
        approval_request: Dict[str, Any],
        pending_tool: Optional[Dict[str, Any]] = None,
        created_at: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str: ...

    def get(self, run_id: str) -> Optional[PausedRun]: ...

    def mark_resumed(self, run_id: str) -> bool: ...

    def list_pending(self) -> List[PausedRun]: ...

    def expire_pending(self, *, max_age_seconds: float, now: Optional[float] = None) -> int: ...

    def purge(self, *, before: float) -> int: ...

    def clear(self) -> None: ...


class InMemoryPausedRunStore:
    """Process-local registry of paused runs (Phase 1 / fallback / tests). Thread-safe."""

    def __init__(self) -> None:
        self._runs: Dict[str, PausedRun] = {}
        self._lock = threading.Lock()

    def save(
        self,
        *,
        session_id: Optional[str],
        agent_id: Optional[str],
        approval_request: Dict[str, Any],
        pending_tool: Optional[Dict[str, Any]] = None,
        created_at: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a pause and return its opaque ``run_id``."""
        run_id = uuid.uuid4().hex
        run = PausedRun(
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            approval_request=approval_request,
            pending_tool=pending_tool,
            created_at=created_at,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._runs[run_id] = run
        return run_id

    def get(self, run_id: str) -> Optional[PausedRun]:
        with self._lock:
            return self._runs.get(run_id)

    def mark_resumed(self, run_id: str) -> bool:
        """Mark a run resumed (single-use). Returns False if unknown or already resumed."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.status != "awaiting_approval":
                return False
            run.status = "resumed"
            return True

    def list_pending(self) -> List[PausedRun]:
        with self._lock:
            return [r for r in self._runs.values() if r.status == "awaiting_approval"]

    def expire_pending(self, *, max_age_seconds: float, now: Optional[float] = None) -> int:
        """Mark pending runs older than ``max_age_seconds`` as ``expired``. Returns the count."""
        cutoff = (time.time() if now is None else now) - max_age_seconds
        expired = 0
        with self._lock:
            for run in self._runs.values():
                # created_at == 0 means "unset" — never expire on a missing timestamp.
                if run.status == "awaiting_approval" and 0 < run.created_at < cutoff:
                    run.status = "expired"
                    expired += 1
        return expired

    def purge(self, *, before: float) -> int:
        """Delete terminal (non-pending) runs created before ``before``. Returns the count."""
        with self._lock:
            drop = [
                rid
                for rid, r in self._runs.items()
                if r.status != "awaiting_approval" and r.created_at < before
            ]
            for rid in drop:
                del self._runs[rid]
        return len(drop)

    def clear(self) -> None:
        """Test hook: drop all records."""
        with self._lock:
            self._runs.clear()


_PAUSED_RUN_DDL = """
CREATE TABLE IF NOT EXISTS paused_run (
    run_id           TEXT PRIMARY KEY,
    session_id       TEXT,
    agent_id         TEXT,
    approval_request TEXT NOT NULL,
    pending_tool     TEXT,
    status           TEXT NOT NULL DEFAULT 'awaiting_approval',
    created_at       REAL,
    resumed_at       REAL,
    metadata         TEXT
)
"""

_PAUSED_RUN_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_paused_run_session "
    "ON paused_run(session_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_paused_run_status ON paused_run(status)",
)


class ProjectDbPausedRunStore:
    """``project.db``-backed paused-run store (FEP-0029 Phase 2) — survives a process restart.

    Self-manages the ``paused_run`` table (idempotent ``CREATE TABLE IF NOT EXISTS``) against the
    project database, mirroring ``ConversationStore`` (the pause is a property of a project-scoped
    conversation). Uses a thread-local connection. JSON-encodes the approval request / pending tool /
    metadata. Accepts an explicit ``db_path`` so tests can point at a temporary database.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            from victor.config.settings import get_project_paths

            db_path = get_project_paths().project_db
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._write_lock = threading.Lock()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=60.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute(_PAUSED_RUN_DDL)
            for index_sql in _PAUSED_RUN_INDEXES:
                conn.execute(index_sql)
            conn.commit()
            self._local.conn = conn
        return conn

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> PausedRun:
        return PausedRun(
            run_id=row["run_id"],
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            approval_request=json.loads(row["approval_request"] or "{}"),
            pending_tool=json.loads(row["pending_tool"]) if row["pending_tool"] else None,
            status=row["status"],
            created_at=row["created_at"] or 0.0,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def save(
        self,
        *,
        session_id: Optional[str],
        agent_id: Optional[str],
        approval_request: Dict[str, Any],
        pending_tool: Optional[Dict[str, Any]] = None,
        created_at: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        run_id = uuid.uuid4().hex
        with self._write_lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO paused_run (run_id, session_id, agent_id, approval_request, "
                "pending_tool, status, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    session_id,
                    agent_id,
                    json.dumps(approval_request or {}),
                    json.dumps(pending_tool) if pending_tool is not None else None,
                    "awaiting_approval",
                    created_at,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()
        return run_id

    def get(self, run_id: str) -> Optional[PausedRun]:
        row = (
            self._conn().execute("SELECT * FROM paused_run WHERE run_id = ?", (run_id,)).fetchone()
        )
        return self._row_to_run(row) if row is not None else None

    def mark_resumed(self, run_id: str) -> bool:
        """Mark a run resumed (single-use). Returns False if unknown or already resumed."""
        with self._write_lock:
            conn = self._conn()
            cur = conn.execute(
                "UPDATE paused_run SET status = 'resumed', resumed_at = ? "
                "WHERE run_id = ? AND status = 'awaiting_approval'",
                (0.0, run_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def list_pending(self) -> List[PausedRun]:
        rows = (
            self._conn()
            .execute(
                "SELECT * FROM paused_run WHERE status = 'awaiting_approval' ORDER BY created_at"
            )
            .fetchall()
        )
        return [self._row_to_run(r) for r in rows]

    def expire_pending(self, *, max_age_seconds: float, now: Optional[float] = None) -> int:
        """Mark pending runs older than ``max_age_seconds`` as ``expired``. Returns the count."""
        cutoff = (time.time() if now is None else now) - max_age_seconds
        with self._write_lock:
            conn = self._conn()
            cur = conn.execute(
                "UPDATE paused_run SET status = 'expired' "
                "WHERE status = 'awaiting_approval' AND created_at > 0 AND created_at < ?",
                (cutoff,),
            )
            conn.commit()
            return cur.rowcount

    def purge(self, *, before: float) -> int:
        """Delete terminal (non-pending) runs created before ``before``. Returns the count."""
        with self._write_lock:
            conn = self._conn()
            cur = conn.execute(
                "DELETE FROM paused_run WHERE status != 'awaiting_approval' AND created_at < ?",
                (before,),
            )
            conn.commit()
            return cur.rowcount

    def clear(self) -> None:
        """Test hook: drop all records."""
        with self._write_lock:
            conn = self._conn()
            conn.execute("DELETE FROM paused_run")
            conn.commit()


_store: Optional[PausedRunStoreProtocol] = None
_store_lock = threading.Lock()


def get_paused_run_store() -> PausedRunStoreProtocol:
    """Return the process-wide paused-run store (FEP-0029).

    Defaults to the durable ``project.db``-backed store so pauses survive a restart; falls back to
    the in-memory store when no project database is resolvable (e.g. an ephemeral/non-project run).
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                try:
                    _store = ProjectDbPausedRunStore()
                except Exception:  # pragma: no cover - defensive: no project DB → in-memory
                    _store = InMemoryPausedRunStore()
    return _store


def set_paused_run_store(store: PausedRunStoreProtocol) -> None:
    """Override the process-wide store (tests / embedding). Pass a fresh store to isolate state."""
    global _store
    with _store_lock:
        _store = store


def reset_paused_run_store() -> None:
    """Clear the process-wide store singleton so the next call re-resolves it."""
    global _store
    with _store_lock:
        _store = None


def record_pause_from_approval(
    request: Any,
    *,
    session_id: Optional[str],
    agent_id: Optional[str],
    created_at: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Persist a pause from an :class:`ApprovalPause`'s request (FEP-0029). Shared helper.

    Extracts the pending gated tool (name/args) from the approval request's context and writes a
    ``paused_run`` via :func:`get_paused_run_store`. Used by BOTH the turn boundary (a fresh ASK,
    ``message_execution``) and resume continuation (a chained ASK, ``durable_resume``) so the
    extraction + store write live in one place. Returns ``(run_id, approval_request_dict)``.
    """
    req_dict: Dict[str, Any] = request.to_dict() if hasattr(request, "to_dict") else {}
    ctx = getattr(request, "context", {}) or {}
    tool_name = ctx.get("tool_name") or ctx.get("tool")
    pending_tool = (
        {"tool_name": tool_name, "arguments": ctx.get("arguments") or ctx.get("args")}
        if tool_name
        else None
    )
    run_id = get_paused_run_store().save(
        session_id=session_id,
        agent_id=agent_id,
        approval_request=req_dict,
        pending_tool=pending_tool,
        created_at=created_at,
        metadata=metadata,
    )
    return run_id, req_dict
