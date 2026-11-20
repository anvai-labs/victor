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

"""Preference and episodic long-term memory store (FEP-0048, FOU-A2a).

Two content classes behind one provider:

- **Preference**: stable directives ("always use uv, never pip") with a scope
  (``user`` or ``project``), provenance, and confidence. Persisted until
  explicitly forgotten.
- **Episode**: timestamped event summaries ("renamed the service to X") with
  session/thread linkage. Project scope only in this increment.

Two-database model (docs/architecture.md): user-scope preferences live in the
GLOBAL database (``~/.victor/victor.db`` via ``get_database()``); project-scope
preferences and all episodes live in the PROJECT database
(``./.victor/project.db`` via ``get_project_database()``). Physical database
separation is the primary scope-isolation boundary — a project query never
opens the global connection and vice versa — and every query additionally
filters on ``scope`` as defense in depth.

Safety: content is scanned with the canonical secret scanner
(``victor.security.safety.secrets.detect_secrets``) and hard-capped at
``MAX_MEMORY_CONTENT_BYTES``. Secret-shaped or oversized content is refused
with a typed error and never stored.

Ranking interop: ``search`` returns ``MemoryResult`` entries shaped exactly
like the other adapters', so ``UnifiedMemoryCoordinator`` ranking, dedup, and
transfer-hint machinery consume this store unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from victor.core.database import DatabaseManager, get_database, get_project_database
from victor.core.schema import Schema, Tables
from victor.storage.memory.unified import (
    MemoryProvenance,
    MemoryQuery,
    MemoryResult,
    MemoryType,
)

logger = logging.getLogger(__name__)

#: Hard cap on stored memory content. Preferences and episodes are directives
#: and summaries, not documents; oversized payloads are refused on the write
#: path so a runaway consolidation hook cannot flood the context budget.
MAX_MEMORY_CONTENT_BYTES = 4096

#: Episodes older than this many days rank toward zero (decay, not deletion —
#: the forgetting policy is explicit forget, per FEP-0048's open question).
EPISODE_RECENCY_HORIZON_DAYS = 90.0

_PREFERENCE_STABILITY_BASE = 0.2
_PREFERENCE_OVERLAP_WEIGHT = 0.6
_PREFERENCE_CONFIDENCE_WEIGHT = 0.2
_EPISODE_OVERLAP_WEIGHT = 0.8
_EPISODE_RECENCY_WEIGHT = 0.2

_PREFERENCE_TABLE = Tables.MEMORY_PREFERENCE
_EPISODE_TABLE = Tables.MEMORY_EPISODE

_PREFERENCE_COLUMNS = "id, scope, content, source, confidence, created_at, updated_at, metadata"
_EPISODE_COLUMNS = "id, scope, summary, session_id, thread_id, timestamp, metadata"

_PREFERENCE_INSERT = (
    f"INSERT INTO {_PREFERENCE_TABLE}" f" ({_PREFERENCE_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)
_EPISODE_INSERT = (
    f"INSERT INTO {_EPISODE_TABLE}" f" ({_EPISODE_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)"
)


class MemoryScope(str, Enum):
    """Scope of a stored memory; decides which database holds the row."""

    USER = "user"
    PROJECT = "project"


class MemoryWriteRefused(ValueError):
    """Base error for content the store refuses to persist.

    Raised instead of storing so callers (tool, consolidation hook) can
    surface a typed, user-actionable refusal.
    """

    def __init__(self, message: str, *, content_preview: str = ""):
        super().__init__(message)
        self.content_preview = content_preview[:80]


class OversizedContentRefused(MemoryWriteRefused):
    """Content exceeds the configured byte cap and is not stored."""

    def __init__(self, size_bytes: int, cap_bytes: int = MAX_MEMORY_CONTENT_BYTES):
        super().__init__(
            f"memory content of {size_bytes} bytes exceeds the "
            f"{cap_bytes}-byte cap; store a summary instead"
        )


class SecretContentRefused(MemoryWriteRefused):
    """Content matches secret-shaped patterns and is never stored."""

    def __init__(self, secret_types: List[str], *, content_preview: str = ""):
        super().__init__(
            "refusing to store secret-shaped content "
            f"(matched: {', '.join(sorted(set(secret_types)))}); "
            "store a reference or the rotation status instead",
            content_preview=content_preview,
        )
        self.secret_types = list(secret_types)


class EmptyContentMatchRefused(ValueError):
    """``forget(content_match=...)`` with an empty/whitespace-only match.

    A contains-match on the empty string matches EVERY preference in the
    selected scope, so an empty form value or model argument would erase an
    entire scope. Rejected in the store before any transaction opens
    (Codex review finding 13); escape-coding LIKE wildcards does not address
    this input.
    """

    def __init__(self) -> None:
        super().__init__(
            "content_match must contain a non-whitespace substring; an empty "
            "match would delete every preference in the scope"
        )


class CrossPrincipalAccessRefused(MemoryWriteRefused):
    """An operation addressed the store as a principal that owns nothing here.

    Store-level defense-in-depth under the enforced single-principal posture
    (Codex review finding 4): when a principal context is present — bound at
    construction or supplied per call — addressing the store as a DIFFERENT
    principal is refused before any database work. Full per-principal
    ownership is the paused A4a+ design (VAS-06).
    """

    def __init__(self, store_principal: str, caller_principal: str) -> None:
        super().__init__(
            f"principal {caller_principal!r} may not address memory bound to "
            f"principal {store_principal!r}: cross-principal memory addressing "
            "is refused until owner isolation lands (VAS-06 / FOU-A4a)"
        )
        self.store_principal = store_principal
        self.caller_principal = caller_principal


@dataclass(frozen=True)
class Preference:
    """A stable directive captured at user or project scope."""

    id: str
    scope: MemoryScope
    content: str
    source: str = "user"
    confidence: float = 1.0
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Episode:
    """A timestamped event summary linked to its session/thread."""

    id: str
    scope: MemoryScope
    summary: str
    session_id: str = ""
    thread_id: str = ""
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PreferenceEpisodicStore:
    """SQLite-backed preference and episodic memory store.

    Implements the ``MemoryProviderProtocol`` read surface (``memory_type``,
    async ``search``/``store``/``get``, ``is_available``) so adapters can
    register it with ``UnifiedMemoryCoordinator``; the typed write helpers
    (``set_preference``, ``record_episode``) are the intended write path and
    the protocol ``store`` routes through them.

    Args:
        global_db: Global database manager (user scope). Defaults to the
            ``get_database()`` singleton.
        project_db: Project database manager (project scope). Defaults to
            ``get_project_database(project_path)``.
        project_path: Project root used when ``project_db`` is not injected.
        max_content_bytes: Content cap override (tests / stricter policies).
        principal_id: Optional principal context bound to this store. When
            set, operations naming a DIFFERENT principal (``on_behalf_of``)
            are refused with ``CrossPrincipalAccessRefused`` before any
            database work — defense-in-depth under the enforced
            single-principal posture (Codex review finding 4); leave unset
            for the single-operator deployments the posture sanctions.
    """

    def __init__(
        self,
        global_db: Optional[DatabaseManager] = None,
        project_db: Optional[DatabaseManager] = None,
        project_path: Optional[Any] = None,
        max_content_bytes: int = MAX_MEMORY_CONTENT_BYTES,
        principal_id: Optional[str] = None,
    ):
        self._global_db = global_db if global_db is not None else get_database()
        self._project_db = (
            project_db if project_db is not None else get_project_database(project_path)
        )
        self._max_content_bytes = max_content_bytes
        self._principal_id = principal_id
        self._ensure_tables()

    @property
    def principal_id(self) -> Optional[str]:
        """The principal context bound to this store (None = unbound)."""
        return self._principal_id

    def _guard_principal(self, on_behalf_of: Optional[str]) -> None:
        """Reject cross-principal addressing when a principal context exists.

        Fires only when BOTH the store is bound and the call names a
        principal, and they differ — checked before any transaction opens.
        """
        if (
            on_behalf_of is not None
            and self._principal_id is not None
            and on_behalf_of != self._principal_id
        ):
            raise CrossPrincipalAccessRefused(self._principal_id, on_behalf_of)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        """Self-ensure the memory tables in both databases.

        The DDL lives in ``Schema`` and runs with the standard schema lists;
        this explicit pass covers existing databases without a schema-version
        bump (same lazy pattern as the RL core tables). Idempotent.
        """
        for db, statements in self._schema_statements():
            with db.transaction() as conn:
                for statement in statements:
                    conn.executescript(statement)

    def _schema_statements(self) -> List[Tuple[DatabaseManager, Tuple[str, ...]]]:
        return [
            (self._global_db, (Schema.MEMORY_PREFERENCE, Schema.MEMORY_PREFERENCE_INDEXES)),
            (
                self._project_db,
                (
                    Schema.MEMORY_PREFERENCE,
                    Schema.MEMORY_PREFERENCE_INDEXES,
                    Schema.MEMORY_EPISODE,
                    Schema.MEMORY_EPISODE_INDEXES,
                ),
            ),
        ]

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    def set_preference(
        self,
        content: str,
        scope: MemoryScope = MemoryScope.USER,
        *,
        source: str = "user",
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        now: Optional[float] = None,
        on_behalf_of: Optional[str] = None,
    ) -> Preference:
        """Store a stable directive.

        Refuses (``MemoryWriteRefused`` subclasses) and stores nothing when
        the content is secret-shaped or exceeds the size cap.

        Args:
            on_behalf_of: Optional caller principal; refused when it differs
                from the store's bound principal context.

        Returns:
            The stored ``Preference``.
        """
        self._guard_principal(on_behalf_of)
        self.validate_content(content)
        self._check_scope(scope)

        timestamp = now if now is not None else time.time()
        pref = Preference(
            id=f"pref_{uuid.uuid4().hex[:16]}",
            scope=scope,
            content=content,
            source=source,
            confidence=max(0.0, min(1.0, confidence)),
            created_at=timestamp,
            updated_at=timestamp,
            metadata=metadata or {},
        )
        db = self._db_for(scope)
        with db.transaction() as conn:
            conn.execute(
                _PREFERENCE_INSERT,
                (
                    pref.id,
                    pref.scope.value,
                    pref.content,
                    pref.source,
                    pref.confidence,
                    pref.created_at,
                    pref.updated_at,
                    json.dumps(pref.metadata),
                ),
            )
        logger.debug("Stored preference %s (scope=%s)", pref.id, scope.value)
        return pref

    def get_preferences(
        self,
        scope: MemoryScope,
        *,
        limit: int = 100,
        on_behalf_of: Optional[str] = None,
    ) -> List[Preference]:
        """List preferences for exactly one scope.

        Rows of the other scope are never returned: the query runs against the
        scope's own database and filters ``scope = ?``.

        Args:
            on_behalf_of: Optional caller principal; refused when it differs
                from the store's bound principal context.
        """
        self._guard_principal(on_behalf_of)
        self._check_scope(scope)
        db = self._db_for(scope)
        rows = db.query(
            f"""
            SELECT {_PREFERENCE_COLUMNS}
            FROM {_PREFERENCE_TABLE}
            WHERE scope = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (scope.value, limit),
        )
        return [self._row_to_preference(row) for row in rows]

    def forget(
        self,
        *,
        preference_id: Optional[str] = None,
        content_match: Optional[str] = None,
        scope: Optional[MemoryScope] = None,
        on_behalf_of: Optional[str] = None,
    ) -> int:
        """Forget preferences by id and/or by content substring match.

        Args:
            preference_id: Delete this exact preference id.
            content_match: Delete preferences whose content contains this
                substring (LIKE-escaped; ``%``/``_`` are literal). Empty and
                whitespace-only matches are REFUSED
                (``EmptyContentMatchRefused``) — a contains-match on the
                empty string would erase the entire scope.
            scope: Restrict the deletion to one scope. Required when
                ``content_match`` is given so a vague match can never sweep
                across both databases.
            on_behalf_of: Optional caller principal; refused when it differs
                from the store's bound principal context.

        Returns:
            Number of rows deleted.
        """
        self._guard_principal(on_behalf_of)

        if preference_id is None and content_match is None:
            raise ValueError("forget requires preference_id or content_match")

        # Codex finding 13: the empty contains-match matches EVERY row in the
        # scope — reject the input before any transaction opens.
        if content_match is not None and not content_match.strip():
            raise EmptyContentMatchRefused()

        if preference_id is not None:
            return self._forget_by_id(preference_id, scope)
        assert content_match is not None  # narrowed above
        if scope is None:
            raise ValueError("forget by content_match requires an explicit scope")
        self._check_scope(scope)
        db = self._db_for(scope)
        with db.transaction() as conn:
            cursor = conn.execute(
                f"""
                DELETE FROM {_PREFERENCE_TABLE}
                WHERE scope = ? AND content LIKE ? ESCAPE '\\'
                """,
                (scope.value, _like_escape_contains(content_match)),
            )
            return max(0, cursor.rowcount)

    def _forget_by_id(self, preference_id: str, scope: Optional[MemoryScope]) -> int:
        """Delete by id, probing only the permitted scope databases."""
        scopes = [scope] if scope is not None else list(MemoryScope)
        deleted = 0
        for candidate in scopes:
            self._check_scope(candidate)
            db = self._db_for(candidate)
            with db.transaction() as conn:
                cursor = conn.execute(
                    f"DELETE FROM {_PREFERENCE_TABLE} WHERE id = ?", (preference_id,)
                )
                deleted += max(0, cursor.rowcount)
            if deleted:
                break
        return deleted

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------

    def record_episode(
        self,
        summary: str,
        *,
        session_id: str = "",
        thread_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        on_behalf_of: Optional[str] = None,
    ) -> Episode:
        """Record a project-scope event summary linked to its session/thread.

        Refuses secret-shaped or oversized summaries; nothing is stored.

        Args:
            on_behalf_of: Optional caller principal; refused when it differs
                from the store's bound principal context.
        """
        self._guard_principal(on_behalf_of)
        self.validate_content(summary)

        ts = timestamp if timestamp is not None else time.time()
        episode = Episode(
            id=f"epis_{uuid.uuid4().hex[:16]}",
            scope=MemoryScope.PROJECT,
            summary=summary,
            session_id=session_id,
            thread_id=thread_id,
            timestamp=ts,
            metadata=metadata or {},
        )
        with self._project_db.transaction() as conn:
            conn.execute(
                _EPISODE_INSERT,
                (
                    episode.id,
                    episode.scope.value,
                    episode.summary,
                    episode.session_id,
                    episode.thread_id,
                    episode.timestamp,
                    json.dumps(episode.metadata),
                ),
            )
        logger.debug("Recorded episode %s (thread=%s)", episode.id, episode.thread_id)
        return episode

    def episodes_for_thread(
        self,
        thread_id: str,
        *,
        limit: int = 100,
    ) -> List[Episode]:
        """Episodes for a thread, newest first (project database only)."""
        rows = self._project_db.query(
            f"""
            SELECT {_EPISODE_COLUMNS}
            FROM {_EPISODE_TABLE}
            WHERE thread_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (thread_id, limit),
        )
        return [self._row_to_episode(row) for row in rows]

    def recent_episodes(self, *, limit: int = 20) -> List[Episode]:
        """Most recent episodes across threads, newest first (project db).

        The listing surface for the user-facing memory views (the /memory
        slash command and the GET /memory API route); thread-scoped queries
        use ``episodes_for_thread``.
        """
        return self._recent_episodes(limit=limit)

    # ------------------------------------------------------------------
    # MemoryProviderProtocol read path
    # ------------------------------------------------------------------

    @property
    def memory_type(self) -> MemoryType:
        """Primary lane; episodes are surfaced by the EPISODE-lane adapter."""
        return MemoryType.PREFERENCE

    def search(self, query: MemoryQuery) -> List[MemoryResult]:
        """Search preferences and episodes, ranked for the coordinator.

        Scope handling: ``query.filters["scope"]`` pins the lane — ``"user"``
        queries only the global database, ``"project"`` only the project
        database. Unscoped queries search both (a federated query is not a
        project-scoped query), but results carry their scope in metadata so
        downstream injection can filter.

        Lane handling (Codex finding 19): ``query.filters["kind"]`` selects
        the content lane BEFORE reading — ``"preference"`` skips the episode
        table entirely and ``"episode"`` skips the preference tables. The
        adapters push their lane down here so each scan reads only the rows
        its caller can surface.

        Synchronous by design: callers on the event loop go through the
        adapter's ``asyncio.to_thread`` hop, never inline.

        Returns:
            ``MemoryResult`` entries sorted by relevance (descending), each
            with provenance stamped for dedup/merge tracing.
        """
        scope_filter = self._scope_filter(query)
        lane_filter = self._lane_filter(query)

        want_preferences = lane_filter in (None, "preference")
        want_episodes = lane_filter in (None, "episode")

        results: List[MemoryResult] = []
        if scope_filter in (None, MemoryScope.USER):
            if want_preferences:
                results.extend(self._search_preferences(query, MemoryScope.USER))
        if scope_filter in (None, MemoryScope.PROJECT):
            if want_preferences:
                results.extend(self._search_preferences(query, MemoryScope.PROJECT))
            if want_episodes:
                results.extend(self._search_episodes(query))

        threshold = max(query.min_relevance, 1e-6)
        results = [r for r in results if r.relevance >= threshold]
        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[: query.limit]

    @staticmethod
    def _lane_filter(query: MemoryQuery) -> Optional[str]:
        """The ``kind`` lane filter, normalized (``None`` = both lanes)."""
        raw = (query.filters or {}).get("kind")
        if raw is None:
            return None
        lane = str(raw).lower()
        if lane not in ("preference", "episode"):
            return None
        return lane

    def _search_preferences(self, query: MemoryQuery, scope: MemoryScope) -> List[MemoryResult]:
        """Search one scope's preference rows in that scope's database."""
        prefs = self.get_preferences(scope, limit=500)
        terms = _query_terms(query.query)
        results: List[MemoryResult] = []
        for pref in prefs:
            overlap = _term_overlap(terms, pref.content)
            if terms and overlap <= 0.0:
                continue
            relevance = min(
                1.0,
                _PREFERENCE_STABILITY_BASE
                + _PREFERENCE_OVERLAP_WEIGHT * overlap
                + _PREFERENCE_CONFIDENCE_WEIGHT * pref.confidence,
            )
            results.append(
                MemoryResult(
                    source=MemoryType.PREFERENCE,
                    content=pref.content,
                    relevance=relevance,
                    id=pref.id,
                    metadata={
                        "kind": "preference",
                        "scope": pref.scope.value,
                        "source": pref.source,
                        "confidence": pref.confidence,
                        **pref.metadata,
                    },
                    timestamp=pref.updated_at,
                )
            )
        return results

    def _search_episodes(self, query: MemoryQuery) -> List[MemoryResult]:
        """Search episode rows in the project database."""
        thread_filter = (query.filters or {}).get("thread_id")
        if thread_filter:
            episodes = self.episodes_for_thread(str(thread_filter), limit=500)
        else:
            episodes = self._recent_episodes(limit=500)
        terms = _query_terms(query.query)
        now = time.time()
        results: List[MemoryResult] = []
        for episode in episodes:
            overlap = _term_overlap(terms, episode.summary)
            if terms and overlap <= 0.0:
                continue
            age_days = max(0.0, (now - episode.timestamp) / 86400.0)
            recency = max(0.0, 1.0 - age_days / EPISODE_RECENCY_HORIZON_DAYS)
            relevance = min(
                1.0,
                _EPISODE_OVERLAP_WEIGHT * overlap + _EPISODE_RECENCY_WEIGHT * recency,
            )
            results.append(
                MemoryResult(
                    source=MemoryType.EPISODE,
                    content=episode.summary,
                    relevance=relevance,
                    id=episode.id,
                    metadata={
                        "kind": "episode",
                        "scope": episode.scope.value,
                        "session_id": episode.session_id,
                        "thread_id": episode.thread_id,
                        **episode.metadata,
                    },
                    timestamp=episode.timestamp,
                )
            )
        return results

    def _recent_episodes(self, *, limit: int = 500) -> List[Episode]:
        rows = self._project_db.query(
            f"""
            SELECT {_EPISODE_COLUMNS}
            FROM {_EPISODE_TABLE}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [self._row_to_episode(row) for row in rows]

    # ------------------------------------------------------------------
    # Protocol conveniences (store/get/is_available)
    # ------------------------------------------------------------------

    async def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Protocol write path routing to the typed helpers.

        ``metadata["kind"]`` selects the lane (``"preference"`` default,
        ``"episode"``); refusals propagate as ``MemoryWriteRefused``.
        """
        metadata = metadata or {}
        kind = metadata.get("kind", "preference")
        if kind == "episode":
            self.record_episode(
                str(value),
                session_id=metadata.get("session_id", ""),
                thread_id=metadata.get("thread_id", ""),
                metadata=metadata.get("data"),
            )
        else:
            scope = MemoryScope(metadata.get("scope", MemoryScope.USER.value))
            self.set_preference(
                str(value),
                scope=scope,
                source=metadata.get("source", "user"),
                confidence=metadata.get("confidence", 1.0),
                metadata=metadata.get("data"),
            )

    async def get(self, key: str) -> Optional[MemoryResult]:
        """Fetch one memory by id across both lanes/scopes.

        The synchronous SQLite lookups run on a worker thread (Codex finding
        19): protocol consumers await this on the event loop, so the DB work
        is offloaded and the loop never blocks on it.
        """
        return await asyncio.to_thread(self._get_by_id_sync, key)

    def _get_by_id_sync(self, key: str) -> Optional[MemoryResult]:
        """Synchronous id lookup across both lanes/scopes (off-loop callee)."""
        for scope in MemoryScope:
            rows = self._db_for(scope).query(
                f"""
                SELECT {_PREFERENCE_COLUMNS}
                FROM {_PREFERENCE_TABLE}
                WHERE id = ?
                """,
                (key,),
            )
            if rows:
                pref = self._row_to_preference(rows[0])
                return MemoryResult(
                    source=MemoryType.PREFERENCE,
                    content=pref.content,
                    relevance=pref.confidence,
                    id=pref.id,
                    metadata={
                        "kind": "preference",
                        "scope": pref.scope.value,
                        "source": pref.source,
                        "confidence": pref.confidence,
                    },
                    timestamp=pref.updated_at,
                )
        rows = self._project_db.query(
            f"""
            SELECT {_EPISODE_COLUMNS}
            FROM {_EPISODE_TABLE}
            WHERE id = ?
            """,
            (key,),
        )
        if rows:
            episode = self._row_to_episode(rows[0])
            return MemoryResult(
                source=MemoryType.EPISODE,
                content=episode.summary,
                relevance=1.0,
                id=episode.id,
                metadata={
                    "kind": "episode",
                    "scope": episode.scope.value,
                    "session_id": episode.session_id,
                    "thread_id": episode.thread_id,
                },
                timestamp=episode.timestamp,
            )
        return None

    def is_available(self) -> bool:
        """Both databases are reachable and the tables exist."""
        try:
            self._global_db.query_one(f"SELECT 1 FROM {_PREFERENCE_TABLE} LIMIT 1")
            self._project_db.query_one(f"SELECT 1 FROM {_EPISODE_TABLE} LIMIT 1")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def validate_content(self, content: str) -> None:
        """Validate content against the write-path policy without storing.

        Raises:
            OversizedContentRefused: content exceeds the byte cap.
            SecretContentRefused: content matches secret-shaped patterns.

        Public so write-path callers (tool, consolidation hook) can
        pre-screen candidates — e.g. drop doomed consolidation proposals at
        propose time — under exactly the same policy the store enforces
        before persisting.
        """
        size = len(content.encode("utf-8"))
        if size > self._max_content_bytes:
            raise OversizedContentRefused(size, cap_bytes=self._max_content_bytes)

        secret_types = _scan_for_secrets(content)
        if secret_types:
            raise SecretContentRefused(secret_types, content_preview=content)

    def _scope_filter(self, query: MemoryQuery) -> Optional[MemoryScope]:
        raw = (query.filters or {}).get("scope")
        if raw is None:
            return None
        return MemoryScope(raw)

    @staticmethod
    def _check_scope(scope: MemoryScope) -> None:
        if not isinstance(scope, MemoryScope):
            raise ValueError(f"unknown memory scope: {scope!r}")

    def _db_for(self, scope: MemoryScope) -> DatabaseManager:
        """The single database that owns ``scope`` — the isolation boundary."""
        return self._global_db if scope is MemoryScope.USER else self._project_db

    @staticmethod
    def _row_to_preference(row: Any) -> Preference:
        return Preference(
            id=row["id"],
            scope=MemoryScope(row["scope"]),
            content=row["content"],
            source=row["source"],
            confidence=row["confidence"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    @staticmethod
    def _row_to_episode(row: Any) -> Episode:
        return Episode(
            id=row["id"],
            scope=MemoryScope(row["scope"]),
            summary=row["summary"],
            session_id=row["session_id"],
            thread_id=row["thread_id"],
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def provenance(self, query_id: str, lane: MemoryType) -> MemoryProvenance:
        """Provenance stamp matching the other adapters' convention."""
        return MemoryProvenance(
            store_class=type(self).__name__,
            query_id=query_id,
            lane=lane,
            store_id=None,
        )


def _scan_for_secrets(content: str) -> List[str]:
    """Return the secret types matched in ``content`` (empty when clean).

    Reuses the canonical scanner so memory captures stay aligned with the
    repo's single credential-pattern set (including the Rust-accelerated path
    when available).
    """
    from victor.security.safety.secrets import detect_secrets

    return [match.secret_type for match in detect_secrets(content)]


def _like_escape_contains(text: str) -> str:
    """Escape LIKE wildcards and wrap for a contains-match."""
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _query_terms(query_text: str) -> List[str]:
    return [term for term in query_text.lower().split() if term]


def _term_overlap(terms: List[str], content: str) -> float:
    """Fraction of query terms present in the content (0.0 when no terms)."""
    if not terms:
        return 0.0
    content_lower = content.lower()
    hits = sum(1 for term in terms if term in content_lower)
    return hits / len(terms)


__all__ = [
    "CrossPrincipalAccessRefused",
    "EmptyContentMatchRefused",
    "Episode",
    "EPISODE_RECENCY_HORIZON_DAYS",
    "MAX_MEMORY_CONTENT_BYTES",
    "MemoryScope",
    "MemoryWriteRefused",
    "OversizedContentRefused",
    "Preference",
    "PreferenceEpisodicStore",
    "SecretContentRefused",
]
