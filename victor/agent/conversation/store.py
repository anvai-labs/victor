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

"""
SQLite-based conversation store with token-aware context management.

This module provides persistent conversation state management with:
- SQLite persistence for session recovery
- Token-aware context window management
- Priority-based message pruning
- Semantic relevance scoring for context selection

The primary class is `ConversationStore`.

For simpler alternatives:
- In-memory: see `victor.agent.message_history.MessageHistory`
- JSON file-based: see `victor.agent.session.SessionPersistence`

Usage:
    store = ConversationStore()
    session = store.create_session(project_path="/path/to/project")

    # Add messages
    store.add_message(session.session_id, MessageRole.USER, "Hello")
    store.add_message(session.session_id, MessageRole.ASSISTANT, "Hi there!")

    # Get context for provider
    messages = store.get_context_messages(session.session_id)
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING
import asyncio
import logging
import os
import re
import sqlite3
import time

from victor.core.async_utils import run_sync
from victor.core.json_utils import json_dumps, json_loads
from victor.tools.core_tool_aliases import canonicalize_core_tool_name
import uuid

if TYPE_CHECKING:
    from victor.storage.embeddings.service import EmbeddingService
    from victor.agent.conversation_embedding_store import ConversationEmbeddingStore

logger = logging.getLogger(__name__)

_SQLITE_WRITE_RETRY_DELAYS = (0.05, 0.15, 0.35)


# Canonical enums from conversation/types.py
from victor.agent.conversation.types import (
    ConversationMessage,
    MESSAGE_SOURCE_METADATA_KEY,
    MessagePriority,
    MessageRole,
    MessageSource,
)
from victor.agent.runtime.context import AgentRuntimeContext
from victor.agent.services.context_service import ContextService, ContextServiceConfig

# ML metadata extracted to victor/agent/ml_metadata.py
from victor.agent.ml_metadata import (  # noqa: F401
    ContextSize,
    ModelFamily,
    ModelMetadata,
    ModelSize,
    get_known_model_context,
    get_known_model_params,
    parse_model_metadata,
)

# ConversationMessage is imported from types.py — single canonical definition.
# Previously duplicated here; consolidated to avoid field drift.

# Source → priority overrides (None = fall through to role-based logic).
# Mirrors _SOURCE_ROLE_OVERRIDES in scoring.py; kept separate to avoid a
# cross-module import cycle at definition time.
_SOURCE_PRIORITY_MAP: dict = {
    MessageSource.USER_TYPED: MessagePriority.CRITICAL,
    MessageSource.AGENT_NUDGE: MessagePriority.EPHEMERAL,
    MessageSource.AGENT_CONTINUATION: MessagePriority.EPHEMERAL,
    MessageSource.AGENT_GUIDANCE: MessagePriority.LOW,
    MessageSource.COMPACTION_SUMMARY: MessagePriority.MEDIUM,
}


from victor.agent.conversation.session_model import ConversationSession

CONVERSATION_DB_ENV = "VICTOR_CONVERSATION_DB"


def _db_path_override() -> Optional[Path]:
    """Explicit conversation-DB location from the environment, if set.

    The default location is derived from the *current working directory*
    (``{project}/.victor/project.db``). That silently breaks for any process
    that chdirs: benchmark evaluation runs each task inside a
    ``tempfile.mkdtemp()`` workspace, so their conversations were written into
    a directory that is deleted when the task ends — or, depending on whether
    project paths had already been resolved and cached, leaked into whichever
    repo the run happened to start from. Neither is a place anyone can read
    them back from afterwards.

    Setting this pins the store to a durable, explicit path regardless of cwd.
    Precedence is ``db_path`` argument > this variable > project default, so
    callers that already pass a path are unaffected.
    """
    raw = os.environ.get(CONVERSATION_DB_ENV, "").strip()
    if not raw:
        return None
    try:
        return Path(raw).expanduser()
    except (OSError, ValueError):
        logger.warning("Ignoring unusable %s=%r", CONVERSATION_DB_ENV, raw)
        return None


class ConversationStore:
    """
    SQLite-based conversation store with intelligent context pruning.

    Features:
    - SQLite persistence for session recovery
    - Token-aware context window management
    - Priority-based message pruning
    - Semantic relevance scoring for context selection
    - Normalized schema for efficient ML/RL aggregation queries

    Schema Design:
    - Lookup tables (model_families, model_sizes, context_sizes, providers)
      store categorical values with INTEGER PRIMARY KEYs
    - Sessions table uses INTEGER FK columns for efficient joins and aggregation
    - Indexes on FK columns for fast GROUP BY queries

    For simpler in-memory storage, see MessageHistory.
    For JSON file-based persistence, see SessionPersistence.

    Usage:
        store = ConversationStore()
        session = store.create_session(project_path="/path/to/project")
        store.add_message(session.session_id, MessageRole.USER, "Hello")
        messages = store.get_context_messages(session.session_id)
    """

    # Schema version for migration tracking (semver format: major.minor.patch)
    SCHEMA_VERSION = "0.3.0"  # Aligned with migration 0.3.0 (normalized schema)

    def __init__(
        self,
        db_path: Optional[Path] = None,
        max_context_tokens: int = 100000,
        response_reserve: int = 4096,
        chars_per_token: int = 4,
    ):
        """Initialize the conversation memory manager.

        Args:
            db_path: Path to SQLite database. Defaults to {project}/.victor/project.db
            max_context_tokens: Maximum tokens in context window
            response_reserve: Tokens reserved for response
            chars_per_token: Approximate characters per token for estimation
        """
        from victor.config.settings import get_project_paths

        paths = get_project_paths()
        self.db_path = db_path or _db_path_override() or paths.project_db
        self.max_context_tokens = max_context_tokens
        self.response_reserve = response_reserve
        self.chars_per_token = chars_per_token

        # Migration check: rename conversation.db to project.db if it exists
        legacy_db = paths.project_victor_dir / "conversation.db"
        if not db_path and legacy_db.exists() and not self.db_path.exists():
            try:
                legacy_db.rename(self.db_path)
                logger.info(f"Migrated legacy database {legacy_db} to {self.db_path}")
            except Exception as e:
                logger.warning(f"Failed to migrate legacy database: {e}")

        # In-memory session cache
        self._sessions: Dict[str, ConversationSession] = {}

        # ID lookup caches for normalized tables (populated lazily)
        self._model_family_ids: Dict[str, int] = {}
        self._model_size_ids: Dict[str, int] = {}
        self._context_size_ids: Dict[str, int] = {}
        self._provider_ids: Dict[str, int] = {}

        # Write serialization locks to prevent concurrent database lock errors
        import asyncio
        import threading

        self._write_lock_async = asyncio.Lock()  # For async operations
        self._write_lock_sync = threading.Lock()  # For sync operations

        # Thread-local connection pool for better performance
        self._local = threading.local()

        # Instantiate sub-managers
        from victor.agent.conversation.store_schema import ConversationStoreSchema
        from victor.agent.conversation.store_session import ConversationStoreSession
        from victor.agent.conversation.store_trace import ConversationStoreTrace
        from victor.agent.conversation.store_messages import ConversationStoreMessages

        self._schema = ConversationStoreSchema(self)
        self._sessions_mgr = ConversationStoreSession(self)
        self._traces = ConversationStoreTrace(self)
        self._messages = ConversationStoreMessages(self)

        # Initialize database
        self._init_database()

        logger.debug(
            f"ConversationStore initialized. DB: {self.db_path}, Max tokens: {max_context_tokens}"
        )

    def _get_connection(self) -> sqlite3.Connection:
        """Get a SQLite connection with optimized settings for concurrent access.

        Uses thread-local connection caching (simple pool) to avoid connection overhead.

        Returns a cached connection with:
        - 60-second timeout for handling concurrent access
        - WAL mode for better read/write concurrency
        - Busy timeout for automatic retries when locked
        - Optimized pragmas for performance

        Thread-local caching provides:
        - Reuse connections within the same thread
        - Different threads get different connections (thread-safe)
        - Automatic cleanup when thread ends
        """
        # Check if thread-local connection exists and is valid
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                # Test if connection is still alive
                self._local.conn.execute("SELECT 1")
                return self._local.conn
            except sqlite3.Error:
                # Connection is stale, close and recreate
                try:
                    self._local.conn.close()
                except Exception:
                    pass
                self._local.conn = None

        # Create new connection
        conn = sqlite3.connect(self.db_path, timeout=60.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        # Configure connection for optimal performance
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 60000")  # 60 second busy timeout
        conn.execute("PRAGMA foreign_keys = ON")

        # Cache in thread-local storage
        self._local.conn = conn

        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections with automatic cleanup.

        Yields:
            Thread-local SQLite connection with Row factory

        Example:
            with self._connection() as conn:
                conn.execute("INSERT INTO ...")
                conn.execute("UPDATE ...")

        The connection is automatically returned to the pool after use.
        """
        yield self._get_connection()

    def _init_database(self) -> None:
        """Initialize SQLite database for persistence with normalized schema."""
        self._schema.init_database()

    def _ensure_lookup_tables(self, conn: sqlite3.Connection) -> None:
        self._schema.ensure_lookup_tables(conn)

    def _ensure_critical_tables(self, conn: sqlite3.Connection) -> None:
        self._schema.ensure_critical_tables(conn)

    def _apply_normalized_schema(self, conn: sqlite3.Connection) -> None:
        self._schema.apply_normalized_schema(conn)

    def _auto_maintenance(self) -> None:
        self._schema.auto_maintenance()

    def _apply_hybrid_compaction_schema(self, conn: sqlite3.Connection) -> None:
        self._schema.apply_hybrid_compaction_schema(conn)

    def _backfill_message_lineage_columns(self, conn: sqlite3.Connection) -> None:
        self._schema.backfill_message_lineage_columns(conn)

    def _populate_fts5_index(self, conn: sqlite3.Connection) -> None:
        self._schema.populate_fts5_index(conn)

    def _populate_lookup_tables(self, conn: sqlite3.Connection) -> None:
        self._schema.populate_lookup_tables(conn)

    def _load_lookup_caches(self, conn: sqlite3.Connection) -> None:
        self._schema.load_lookup_caches(conn)

    def _migrate_sessions_table(self, conn: sqlite3.Connection) -> None:
        self._schema.migrate_sessions_table(conn)

    def _get_or_create_provider_id(
        self,
        conn: sqlite3.Connection,
        provider: Optional[str],
    ) -> Optional[int]:
        return self._schema.get_or_create_provider_id(conn, provider)

    def _get_model_family_id(self, family: Optional[ModelFamily]) -> Optional[int]:
        return self._schema.get_model_family_id(family)

    def _get_model_size_id(self, size: Optional[ModelSize]) -> Optional[int]:
        return self._schema.get_model_size_id(size)

    def _get_context_size_id(self, ctx_size: Optional[ContextSize]) -> Optional[int]:
        return self._schema.get_context_size_id(ctx_size)

    def create_session(
        self,
        session_id: Optional[str] = None,
        project_path: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        profile: Optional[str] = None,
        tool_capable: bool = False,
    ) -> ConversationSession:
        return self._sessions_mgr.create_session(
            session_id=session_id,
            project_path=project_path,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            profile=profile,
            tool_capable=tool_capable,
        )

    def save_session(
        self,
        conversation: Any,
        model: str,
        provider: str,
        profile: str = "default",
        session_id: Optional[str] = None,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        conversation_state: Optional[Any] = None,
        tool_selection_stats: Optional[Dict[str, Any]] = None,
        execution_state: Optional[Any] = None,
        session_ledger: Optional[Any] = None,
        compaction_hierarchy: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save session with rich metadata.

        Args:
            conversation: Conversation object, dict with "messages", or list of messages.
            model: Model identifier.
            provider: Provider identifier.
            profile: User-facing profile name.
            session_id: Existing session id to update, or None for a new session.
            title: Optional session title.
            tags: Optional user-assigned tags.
            conversation_state: ConversationStateMachine (or its dict).
            tool_selection_stats: Optional tool selection statistics dict.
            execution_state: ExecutionState (or its dict).
            session_ledger: SessionLedger (or its dict).
            compaction_hierarchy: Message compaction hierarchy dict.

        Returns:
            The session id.
        """
        return self._sessions_mgr.save_session(
            conversation=conversation,
            model=model,
            provider=provider,
            profile=profile,
            session_id=session_id,
            title=title,
            tags=tags,
            conversation_state=conversation_state,
            tool_selection_stats=tool_selection_stats,
            execution_state=execution_state,
            session_ledger=session_ledger,
            compaction_hierarchy=compaction_hierarchy,
        )

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions_mgr.load_session(session_id)

    def search_sessions(
        self,
        query: str,
        *,
        limit: int = 5,
        project_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self._sessions_mgr.search_sessions(
            query=query,
            limit=limit,
            project_path=project_path,
        )

    def _row_to_session_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row into a dictionary of session metadata.

        Safe for partial SELECTs (e.g. lightweight search queries): missing
        columns default to None instead of raising ``No item with that key``.
        """
        # sqlite3.Row supports the mapping protocol, so dict(row) yields a
        # plain dict we can safely .get() against for any optional column.
        row_map = dict(row)
        meta_raw = row_map.get("metadata")
        meta = json_loads(meta_raw) if meta_raw else {}

        # Parse joined lookup strings to construct dict matching model_metadata expectations
        provider_name = row_map.get("provider_name")
        family_name = row_map.get("model_family_name")
        size_name = row_map.get("model_size_name")
        ctx_name = row_map.get("context_size_name")

        return {
            "session_id": row_map.get("session_id"),
            "created_at": datetime.fromisoformat(row_map["created_at"]),
            "last_activity": datetime.fromisoformat(row_map["last_activity"]),
            "project_path": (
                Path(row_map["project_path"]) if row_map.get("project_path") else None
            ),
            "model": row_map.get("model"),
            "profile": row_map.get("profile"),
            "max_tokens": row_map.get("max_tokens"),
            "reserved_tokens": row_map.get("reserved_tokens"),
            "metadata": meta,
            # Rich metadata surfaced for search/list consumers
            "title": meta.get("title"),
            "tags": meta.get("tags", []),
            # Normalization lookups
            "provider": provider_name,
            "model_family": ModelFamily(family_name) if family_name else None,
            "model_size": ModelSize(size_name) if size_name else None,
            "context_size": ContextSize(ctx_name) if ctx_name else None,
            "model_params_b": row_map.get("model_params_b"),
            "context_tokens": row_map.get("context_tokens"),
            "tool_capable": bool(row_map.get("tool_capable")),
            "is_moe": bool(row_map.get("is_moe")),
            "is_reasoning": bool(row_map.get("is_reasoning")),
            # Tokens/Metrics
            "prompt_tokens": row_map.get("prompt_tokens"),
            "completion_tokens": row_map.get("completion_tokens"),
            "cached_tokens": row_map.get("cached_tokens"),
            "reasoning_tokens": row_map.get("reasoning_tokens"),
            "cost_usd_micros": row_map.get("cost_usd_micros"),
        }

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        return self._sessions_mgr.get_session(session_id)

    def list_sessions(
        self,
        project_path: Optional[str] = None,
        limit: int = 10,
    ) -> List[ConversationSession]:
        return self._sessions_mgr.list_sessions(
            project_path=project_path,
            limit=limit,
        )

    def _add_message_impl(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        priority: Optional[MessagePriority],
        tool_name: Optional[str],
        tool_call_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        tool_calls: Optional[List],
    ) -> ConversationMessage:
        return self._messages._add_message_impl(
            session_id=session_id,
            role=role,
            content=content,
            priority=priority,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata,
            tool_calls=tool_calls,
        )

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        priority: Optional[MessagePriority] = None,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List] = None,
    ) -> ConversationMessage:
        return self._messages.add_message(
            session_id=session_id,
            role=role,
            content=content,
            priority=priority,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata,
            tool_calls=tool_calls,
        )

    def add_system_message(
        self,
        session_id: str,
        content: str,
    ) -> ConversationMessage:
        return self._messages.add_system_message(session_id, content)

    def get_context_messages(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Dict[str, Any]]:
        return self._messages.get_context_messages(
            session_id=session_id,
            max_tokens=max_tokens,
            include_system=include_system,
        )

    def get_messages_for_agent(
        self,
        session_id: str,
        agent_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[ConversationMessage]:
        return self._messages.get_messages_for_agent(
            session_id=session_id,
            agent_id=agent_id,
            limit=limit,
        )

    def load_agent_context_service(
        self,
        runtime_context: AgentRuntimeContext,
        *,
        max_tokens: Optional[int] = None,
    ) -> ContextService:
        return self._messages.load_agent_context_service(
            runtime_context=runtime_context,
            max_tokens=max_tokens,
        )

    def record_compaction_event(
        self,
        *,
        session_id: str,
        strategy: str,
        agent_id: Optional[str] = None,
        messages_removed: int = 0,
        tokens_freed: int = 0,
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._messages.record_compaction_event(
            session_id=session_id,
            strategy=strategy,
            agent_id=agent_id,
            messages_removed=messages_removed,
            tokens_freed=tokens_freed,
            summary=summary,
            metadata=metadata,
        )

    def get_recent_messages(
        self,
        session_id: str,
        count: int = 10,
    ) -> List[ConversationMessage]:
        return self._messages.get_recent_messages(session_id, count)

    def clear_session(self, session_id: str):
        self._sessions_mgr.clear_session(session_id)

    def delete_session(self, session_id: str):
        self._sessions_mgr.delete_session(session_id)

    def vacuum(self) -> Dict[str, Any]:
        """Reclaim unused space in the database.

        Performs incremental auto-vacuum plus a checkpoint to merge WAL file.
        This is useful after bulk deletions to reduce database file size.

        Returns:
            Dictionary with before/after file sizes and space freed
        """
        import os

        before_size = os.path.getsize(self.db_path) if self.db_path.exists() else 0

        with sqlite3.connect(self.db_path) as conn:
            # Run incremental auto-vacuum
            conn.execute("PRAGMA incremental_vacuum")
            # Checkpoint WAL to main database
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        after_size = os.path.getsize(self.db_path) if self.db_path.exists() else 0
        freed = before_size - after_size

        logger.info(
            f"Database vacuum: {before_size / 1024:.1f}KB -> {after_size / 1024:.1f}KB "
            f"(freed {freed / 1024:.1f}KB)"
        )

        return {
            "before_bytes": before_size,
            "after_bytes": after_size,
            "freed_bytes": freed,
        }

    # =========================================================================
    # DATABASE CLEANUP
    # =========================================================================

    # Known test model names that should not exist in production DB
    TEST_MODEL_NAMES = frozenset(
        {
            "test-model",
            "dummy",
            "fake",
            "dummy-model",
            "specific-tool-model",
            "unknown-model",
            "totally-unknown-model-xyz",
            "mock-model",
            "test-provider-model",
            "gpt-oss:20b",
        }
    )

    def cleanup_stale_sessions(
        self,
        max_age_days: int = 30,
        purge_test_models: bool = True,
        purge_empty: bool = True,
    ) -> Dict[str, int]:
        return self._sessions_mgr.cleanup_stale_sessions(
            max_age_days=max_age_days,
            purge_test_models=purge_test_models,
            purge_empty=purge_empty,
        )

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics including file size and record counts.

        Returns:
            Dictionary with database stats
        """
        import os

        file_size = os.path.getsize(self.db_path) if self.db_path.exists() else 0

        with sqlite3.connect(self.db_path) as conn:
            session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

            # Get WAL mode info
            journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

        return {
            "file_path": str(self.db_path),
            "file_size_bytes": file_size,
            "file_size_kb": file_size / 1024,
            "session_count": session_count,
            "message_count": message_count,
            "journal_mode": journal_mode,
        }

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        return self._sessions_mgr.get_session_stats(session_id)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _get_or_create_session(self, session_id: str) -> ConversationSession:
        """Get existing session or create new one."""
        session = self.get_session(session_id)
        if session is None:
            session = self.create_session(session_id=session_id)
        return session

    def _determine_priority(
        self,
        role: MessageRole,
        tool_name: Optional[str],
        source: Optional["MessageSource"] = None,
    ) -> MessagePriority:
        """Determine message priority based on role, tool context, and source origin."""
        # Source-based overrides take precedence over role-based logic.
        if source is not None:
            priority = _SOURCE_PRIORITY_MAP.get(source)
            if priority is not None:
                return priority
            # AGENT_RESPONSE, LEDGER_RENDER, SYSTEM_INJECTED, UNKNOWN → fall through

        canonical_tool_name = canonicalize_core_tool_name(tool_name) if tool_name else None

        if role == MessageRole.SYSTEM:
            return MessagePriority.CRITICAL

        if role == MessageRole.USER:
            return MessagePriority.HIGH

        if role == MessageRole.ASSISTANT:
            return MessagePriority.HIGH

        if role == MessageRole.TOOL:
            # File contents and search results are valuable context
            if canonical_tool_name in ("read", "code_search", "ls"):
                return MessagePriority.HIGH
            return MessagePriority.MEDIUM

        if role == MessageRole.TOOL_CALL:
            return MessagePriority.MEDIUM

        return MessagePriority.MEDIUM

    def _estimate_tokens(self, content: str) -> int:
        return self._traces.estimate_tokens(content)

    def _build_trace_metadata(
        self,
        role: MessageRole,
        content: str,
        tool_name: Optional[str],
        tool_call_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        tool_calls: Optional[List],
    ) -> Dict[str, Any]:
        return self._traces.build_trace_metadata(
            role=role,
            content=content,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata,
            tool_calls=tool_calls,
        )

    def _is_execution_trace_message(
        self,
        role: MessageRole,
        tool_name: Optional[str],
        tool_call_id: Optional[str],
        tool_calls: Optional[List],
    ) -> bool:
        return self._traces.is_execution_trace_message(
            role=role,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
        )

    def _build_execution_trace_text(
        self,
        role: MessageRole,
        content: str,
        tool_name: Optional[str],
        tool_call_id: Optional[str],
        tool_calls: Optional[List],
    ) -> str:
        return self._traces.build_execution_trace_text(
            role=role,
            content=content,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _extract_trace_attribute(content: str, attribute_name: str) -> Optional[str]:
        from victor.agent.conversation.store_trace import ConversationStoreTrace

        return ConversationStoreTrace.extract_trace_attribute(content, attribute_name)

    @staticmethod
    def _trace_tokens(text: str) -> List[str]:
        from victor.agent.conversation.store_trace import ConversationStoreTrace

        return ConversationStoreTrace.trace_tokens(text)

    def get_message_trace_kind(self, message: ConversationMessage) -> str:
        return self._traces.get_message_trace_kind(message)

    def get_message_trace_text(
        self,
        message: ConversationMessage,
        trace_kind: Optional[str] = None,
    ) -> str:
        return self._traces.get_message_trace_text(message, trace_kind)

    def _score_execution_trace(
        self,
        query: str,
        trace_text: str,
    ) -> float:
        return self._traces.score_execution_trace(query, trace_text)

    def get_dual_trace_relevant_messages(
        self,
        session_id: str,
        query: str,
        semantic_limit: int = 5,
        execution_limit: int = 3,
        min_similarity: float = 0.3,
    ) -> Dict[str, List[Tuple[ConversationMessage, float]]]:
        return self._traces.get_dual_trace_relevant_messages(
            session_id=session_id,
            query=query,
            semantic_limit=semantic_limit,
            execution_limit=execution_limit,
            min_similarity=min_similarity,
        )

    async def aget_dual_trace_relevant_messages(
        self,
        session_id: str,
        query: str,
        semantic_limit: int = 5,
        execution_limit: int = 3,
        min_similarity: float = 0.3,
    ) -> Dict[str, List[Tuple[ConversationMessage, float]]]:
        return await self._traces.aget_dual_trace_relevant_messages(
            session_id=session_id,
            query=query,
            semantic_limit=semantic_limit,
            execution_limit=execution_limit,
            min_similarity=min_similarity,
        )

    def _get_execution_trace_matches(
        self,
        session_id: str,
        query: str,
        limit: int,
    ) -> List[Tuple[ConversationMessage, float]]:
        """Retrieve execution traces using a lightweight lexical scorer."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY timestamp DESC
                """,
                (session_id,),
            ).fetchall()

        scored_messages: List[Tuple[ConversationMessage, float]] = []
        for row in rows:
            message = self._message_from_row(row)
            if self.get_message_trace_kind(message) != "execution":
                continue

            trace_text = self.get_message_trace_text(message, "execution")
            score = self._score_execution_trace(query, trace_text)
            if score <= 0:
                continue
            scored_messages.append((message, score))

        scored_messages.sort(key=lambda item: item[1], reverse=True)
        return scored_messages[:limit]

    def _score_messages(
        self,
        messages: List[ConversationMessage],
    ) -> List[tuple[ConversationMessage, float]]:
        return self._messages._score_messages(messages)

    def _prune_context(self, session: ConversationSession):
        self._messages._prune_context(session)

    def _delete_messages_from_db(self, session_id: str, message_ids: List[str]) -> None:
        self._messages._delete_messages_from_db(session_id, message_ids)

    def _persist_session(self, session: ConversationSession):
        """Persist session to database using normalized FK columns."""
        with sqlite3.connect(self.db_path) as conn:
            # Get or create provider ID
            provider_id = self._get_or_create_provider_id(conn, session.provider)
            session_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }

            # Get FK IDs from cached lookups
            model_family_id = self._get_model_family_id(session.model_family)
            model_size_id = self._get_model_size_id(session.model_size)
            context_size_id = self._get_context_size_id(session.context_size)

            # Build metadata dict with both legacy and new rich fields
            metadata = {
                "active_files": session.active_files,
                "tool_usage_count": session.tool_usage_count,
                # Rich session metadata
                "title": session.title,
                "tags": session.tags,
                # State persistence
                "conversation_state": session.conversation_state,
                "execution_state": session.execution_state,
                "session_ledger": session.session_ledger,
                "compaction_hierarchy": session.compaction_hierarchy,
                # Preview messages (converted to dicts for JSON serialization)
                "preview_messages": [
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                        "timestamp": (msg.timestamp.isoformat() if msg.timestamp else None),
                        "token_count": msg.token_count,
                        "priority": msg.priority.value if msg.priority else 50,
                        "tool_name": msg.tool_name,
                        "tool_call_id": msg.tool_call_id,
                        "metadata": msg.metadata if msg.metadata else {},
                    }
                    for msg in session.preview_messages
                ],
            }

            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, created_at, last_activity, project_path,
                 model, profile, max_tokens, reserved_tokens, metadata,
                 provider_id, model_family_id, model_size_id, context_size_id,
                 model_params_b, context_tokens, tool_capable, is_moe, is_reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.created_at.isoformat(),
                    session.last_activity.isoformat(),
                    session.project_path,
                    session.model,
                    session.profile,
                    session.max_tokens,
                    session.reserved_tokens,
                    json_dumps(metadata),
                    # Normalized FK columns
                    provider_id,
                    model_family_id,
                    model_size_id,
                    context_size_id,
                    # Numeric columns
                    session.model_params_b,
                    session.context_tokens,
                    1 if session.tool_capable else 0,
                    1 if session.is_moe else 0,
                    1 if session.is_reasoning else 0,
                ),
            )
            if "provider" in session_columns:
                conn.execute(
                    "UPDATE sessions SET provider = ? WHERE session_id = ?",
                    (session.provider, session.session_id),
                )

    # Max chars for tool output content stored in SQLite.
    # Full content is only needed during the active session (in-memory).
    # Historical records keep a truncated version to save space.
    _TOOL_OUTPUT_STORE_LIMIT = 8000

    def _sanitize_metadata_for_json(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata dict for JSON serialization.

        Converts non-JSON-serializable objects to safe string representations.
        This handles edge cases like Ellipsis, Path objects, etc.

        Args:
            metadata: Raw metadata dict

        Returns:
            Sanitized metadata dict that is JSON-serializable
        """
        if not metadata:
            return {}

        def _sanitize_value(value: Any) -> Any:
            """Recursively sanitize a value for JSON serialization."""
            # Handle Ellipsis (...)
            if value is ...:
                return "<ellipsis>"
            # Handle Path objects
            elif hasattr(value, "__fspath__"):  # Path-like objects
                return str(value)
            # Handle types that can't be JSON serialized
            elif isinstance(value, type(lambda: None)):  # Functions/lambdas
                return f"<function: {getattr(value, '__name__', 'lambda')}>"
            # Handle objects with __dict__
            elif hasattr(value, "__dict__") and not isinstance(
                value, (str, bytes, dict, list, tuple, int, float, bool)
            ):
                return f"<object: {value.__class__.__name__}>"
            # Recursively handle lists and dicts
            elif isinstance(value, dict):
                return {k: _sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [_sanitize_value(v) for v in value]
            else:
                return value

        try:
            # Try to serialize as-is first
            from victor.core.json_utils import json_dumps

            json_dumps(metadata)
            return metadata
        except (TypeError, ValueError):
            # If that fails, sanitize the metadata
            logger.debug(f"Sanitizing metadata for JSON serialization: {list(metadata.keys())}")
            return {k: _sanitize_value(v) for k, v in metadata.items()}

    def _persist_message(self, session_id: str, message: ConversationMessage):
        """Persist message to database.

        IMPORTANT: Uses self._write_lock_sync to serialize all write
        operations and prevent "database is locked" errors from concurrent access.

        Tool output messages (tool_call, tool_result, and user-role
        TOOL_OUTPUT blocks) are truncated to _TOOL_OUTPUT_STORE_LIMIT
        chars when stored, since full content is only needed during the
        active session where it lives in memory.
        """
        content = message.content
        # Truncate large tool outputs for storage
        if len(content) > self._TOOL_OUTPUT_STORE_LIMIT and (
            message.role in (MessageRole.TOOL_CALL, MessageRole.TOOL)
            or (message.role == MessageRole.USER and content.startswith("<TOOL_OUTPUT"))
        ):
            content = (
                content[: self._TOOL_OUTPUT_STORE_LIMIT]
                + f"\n\n[... truncated from {len(message.content)} chars "
                f"for storage]"
            )

        # Merge tool_calls into metadata for persistence so assistant
        # messages that requested tool calls can be fully reconstructed.
        meta = dict(message.metadata) if message.metadata else {}
        if message.tool_calls:
            meta["tool_calls"] = message.tool_calls

        # Sanitize metadata for JSON serialization (handles Ellipsis, Path objects, etc.)
        meta = self._sanitize_metadata_for_json(meta)
        agent_id = meta.get("agent_id")
        parent_session_id = meta.get("parent_session_id")
        team_id = meta.get("team_id")
        member_id = meta.get("member_id")
        plan_id = meta.get("plan_id")
        plan_step_id = meta.get("plan_step_id")

        def _write(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT OR REPLACE INTO messages
                (id, session_id, role, content, timestamp, token_count,
                 priority, tool_name, tool_call_id, metadata,
                 agent_id, parent_session_id, team_id, member_id, plan_id, plan_step_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    session_id,
                    message.role.value,
                    content,
                    message.timestamp.isoformat(),
                    message.token_count,
                    message.priority.value,
                    message.tool_name,
                    message.tool_call_id,
                    json_dumps(meta),
                    agent_id,
                    parent_session_id,
                    team_id,
                    member_id,
                    plan_id,
                    plan_step_id,
                ),
            )

        self._with_locked_write_retry(_write, operation="persist_message")

    def _update_session_activity(self, session_id: str):
        """Update session last activity timestamp with write serialization."""

        def _write(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )

        self._with_locked_write_retry(_write, operation="update_session_activity")

    @staticmethod
    def _is_sqlite_locked_error(exc: sqlite3.OperationalError) -> bool:
        message = str(exc).lower()
        return "database is locked" in message or "database table is locked" in message

    def _with_locked_write_retry(self, write_fn: Any, *, operation: str) -> None:
        """Run a serialized SQLite write with bounded retry for transient lock contention."""
        attempts = len(_SQLITE_WRITE_RETRY_DELAYS) + 1
        for attempt in range(attempts):
            try:
                # CRITICAL: Use lock to serialize writes in this process. The retry handles
                # external processes or stale SQLite locks that survive busy_timeout.
                with self._write_lock_sync:
                    with self._get_connection() as conn:
                        write_fn(conn)
                return
            except sqlite3.OperationalError as exc:
                if not self._is_sqlite_locked_error(exc) or attempt == attempts - 1:
                    raise
                delay = _SQLITE_WRITE_RETRY_DELAYS[attempt]
                logger.debug(
                    "SQLite write lock during %s; retrying in %.2fs (%d/%d)",
                    operation,
                    delay,
                    attempt + 1,
                    attempts - 1,
                )
                time.sleep(delay)

    def update_session_token_usage(
        self,
        session_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        cost_usd_micros: int = 0,
    ) -> None:
        """Persist cumulative API token usage for a session.

        Accumulates (+=) into the existing row so repeated calls are additive.
        Safe to call after every turn — uses INSERT OR IGNORE + UPDATE pattern
        to handle sessions not yet persisted to DB.

        Args:
            session_id: Session to update
            prompt_tokens: Actual prompt tokens from API response
            completion_tokens: Actual completion tokens
            cached_tokens: Cache-hit tokens (reduces effective cost)
            reasoning_tokens: Extended reasoning tokens (if applicable)
            cost_usd_micros: Cost in USD micros (1e-6 USD)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE sessions
                    SET prompt_tokens     = COALESCE(prompt_tokens, 0)     + ?,
                        completion_tokens = COALESCE(completion_tokens, 0) + ?,
                        cached_tokens     = COALESCE(cached_tokens, 0)     + ?,
                        reasoning_tokens  = COALESCE(reasoning_tokens, 0)  + ?,
                        cost_usd_micros   = COALESCE(cost_usd_micros, 0)   + ?,
                        last_activity     = ?
                    WHERE session_id = ?
                    """,
                    (
                        prompt_tokens,
                        completion_tokens,
                        cached_tokens,
                        reasoning_tokens,
                        cost_usd_micros,
                        __import__("datetime").datetime.now().isoformat(),
                        session_id,
                    ),
                )
        except Exception as e:
            logger.debug("update_session_token_usage skipped: %s", e)

    def update_message_token_count(
        self,
        session_id: str,
        message_id: str,
        actual_token_count: int,
    ) -> None:
        """Update an individual message's token count with actual API value.

        Replaces the estimated token count set during add_message() with the
        actual token count from the API response. This enables accurate per-
        message token tracking for analytics and cost estimation.

        Args:
            session_id: Session containing the message
            message_id: Message to update
            actual_token_count: Actual token count from API response
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE messages
                    SET token_count = ?
                    WHERE session_id = ? AND message_id = ?
                    """,
                    (actual_token_count, session_id, message_id),
                )
                conn.commit()

            # Also update in-memory session if loaded
            session = self._sessions.get(session_id)
            if session:
                for msg in session.messages:
                    if msg.id == message_id:
                        # Adjust session total: remove estimate, add actual
                        session.current_tokens -= msg.token_count
                        msg.token_count = actual_token_count
                        session.current_tokens += actual_token_count
                        break
        except Exception as e:
            logger.debug("update_message_token_count skipped: %s", e)

    def get_session_token_stats(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get comprehensive token statistics for a session.

        Returns token usage, cost estimates, cache hit rate, and reasoning
        token usage from the sessions table.

        Args:
            session_id: Session to query

        Returns:
            Dict with keys: prompt_tokens, completion_tokens, cached_tokens,
            reasoning_tokens, total_tokens, cache_hit_rate, cost_usd_micros,
            or None if session not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT prompt_tokens, completion_tokens, cached_tokens, reasoning_tokens, cost_usd_micros
                    FROM sessions
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                prompt_tokens = row["prompt_tokens"] or 0
                completion_tokens = row["completion_tokens"] or 0
                cached_tokens = row["cached_tokens"] or 0
                reasoning_tokens = row["reasoning_tokens"] or 0
                cost_usd_micros = row["cost_usd_micros"] or 0

                total_tokens = prompt_tokens + completion_tokens
                cache_hit_rate = (cached_tokens / prompt_tokens * 100) if prompt_tokens > 0 else 0.0

                return {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens,
                    "cache_hit_rate": round(cache_hit_rate, 2),
                    "cost_usd_micros": cost_usd_micros,
                    "cost_usd": cost_usd_micros / 1_000_000,
                }
        except Exception as e:
            logger.debug("get_session_token_stats error: %s", e)
            return None

    def get_total_token_usage(
        self,
        hours: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get aggregate token usage across all sessions.

        Useful for monitoring overall usage and cost estimation.

        Args:
            hours: Only include sessions from the last N hours. None = all time.

        Returns:
            Dict with aggregate statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if hours is not None:
                    cursor = conn.execute(
                        """
                        SELECT
                            COUNT(*) as session_count,
                            COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
                            COALESCE(SUM(completion_tokens), 0) as total_completion_tokens,
                            COALESCE(SUM(cached_tokens), 0) as total_cached_tokens,
                            COALESCE(SUM(reasoning_tokens), 0) as total_reasoning_tokens,
                            COALESCE(SUM(cost_usd_micros), 0) as total_cost_usd_micros
                        FROM sessions
                        WHERE datetime(last_activity) >= datetime('now', '-' || ? || ' hours')
                        """,
                        (hours,),
                    )
                else:
                    cursor = conn.execute("""
                        SELECT
                            COUNT(*) as session_count,
                            COALESCE(SUM(prompt_tokens), 0) as total_prompt_tokens,
                            COALESCE(SUM(completion_tokens), 0) as total_completion_tokens,
                            COALESCE(SUM(cached_tokens), 0) as total_cached_tokens,
                            COALESCE(SUM(reasoning_tokens), 0) as total_reasoning_tokens,
                            COALESCE(SUM(cost_usd_micros), 0) as total_cost_usd_micros
                        FROM sessions
                        """)

                row = cursor.fetchone()
                total_prompt = row["total_prompt_tokens"] or 0
                total_completion = row["total_completion_tokens"] or 0
                total_cached = row["total_cached_tokens"] or 0

                return {
                    "session_count": row["session_count"] or 0,
                    "total_prompt_tokens": total_prompt,
                    "total_completion_tokens": total_completion,
                    "total_cached_tokens": total_cached,
                    "total_reasoning_tokens": row["total_reasoning_tokens"] or 0,
                    "total_tokens": total_prompt + total_completion,
                    "total_cost_usd_micros": row["total_cost_usd_micros"] or 0,
                    "total_cost_usd": (row["total_cost_usd_micros"] or 0) / 1_000_000,
                    "cache_hit_rate": (
                        round(total_cached / total_prompt * 100, 2) if total_prompt > 0 else 0.0
                    ),
                }
        except Exception as e:
            logger.debug("get_total_token_usage error: %s", e)
            return {
                "session_count": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cached_tokens": 0,
                "total_reasoning_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd_micros": 0,
                "total_cost_usd": 0.0,
                "cache_hit_rate": 0.0,
            }

    # -----------------------------------------------------------------
    # Async variants — offload blocking SQLite I/O to thread pool
    # Use these from async contexts to avoid blocking the event loop.
    # -----------------------------------------------------------------

    async def add_message_async(
        self,
        session_id: str,
        role: "MessageRole",
        content: str,
        priority: Optional["MessagePriority"] = None,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List] = None,
    ) -> "ConversationMessage":
        """Async variant of add_message with serialized writes."""
        return await self._messages.add_message_async(
            session_id=session_id,
            role=role,
            content=content,
            priority=priority,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata,
            tool_calls=tool_calls,
        )

    def _load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load session from database with JOINs to lookup tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # JOIN with lookup tables to get enum names
            row = conn.execute(
                """
                SELECT
                    s.*,
                    p.name AS provider_name,
                    mf.name AS model_family_name,
                    ms.name AS model_size_name,
                    cs.name AS context_size_name
                FROM sessions s
                LEFT JOIN providers p ON s.provider_id = p.id
                LEFT JOIN model_families mf ON s.model_family_id = mf.id
                LEFT JOIN model_sizes ms ON s.model_size_id = ms.id
                LEFT JOIN context_sizes cs ON s.context_size_id = cs.id
                WHERE s.session_id = ?
                """,
                (session_id,),
            ).fetchone()

            if not row:
                return None

            session = self._session_from_row(row)

            # Load messages
            messages = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY timestamp
                """,
                (session_id,),
            ).fetchall()

            for msg_row in messages:
                session.messages.append(self._message_from_row(msg_row))

            session.current_tokens = sum(m.token_count for m in session.messages)

            return session

    def _session_from_row(self, row: sqlite3.Row) -> ConversationSession:
        """Create session from database row with normalized FK lookups."""
        metadata = json_loads(row["metadata"] or "{}")
        row_keys = row.keys()

        # Get provider from joined table or NULL
        provider = row["provider_name"] if "provider_name" in row_keys else None

        # Parse model family from joined lookup table
        model_family = None
        family_name = row["model_family_name"] if "model_family_name" in row_keys else None
        if family_name:
            try:
                model_family = ModelFamily(family_name)
            except ValueError:
                model_family = ModelFamily.UNKNOWN

        # Parse model size from joined lookup table
        model_size = None
        size_name = row["model_size_name"] if "model_size_name" in row_keys else None
        if size_name:
            try:
                model_size = ModelSize(size_name)
            except ValueError:
                model_size = ModelSize.MEDIUM

        # Parse context size from joined lookup table
        context_size = None
        ctx_name = row["context_size_name"] if "context_size_name" in row_keys else None
        if ctx_name:
            try:
                context_size = ContextSize(ctx_name)
            except ValueError:
                context_size = ContextSize.MEDIUM

        # Load preview messages from metadata
        preview_messages_data = metadata.get("preview_messages", [])
        preview_messages = []
        for pm in preview_messages_data:
            if isinstance(pm, dict):
                preview_msg = ConversationMessage(
                    role=MessageRole(pm.get("role", "user")),
                    content=pm.get("content", ""),
                    timestamp=(
                        datetime.fromisoformat(pm["timestamp"])
                        if pm.get("timestamp")
                        else datetime.now()
                    ),
                    token_count=pm.get("token_count", 0),
                    priority=MessagePriority(pm.get("priority", 50)),
                    tool_name=pm.get("tool_name"),
                    tool_call_id=pm.get("tool_call_id"),
                    metadata=pm.get("metadata", {}),
                )
                preview_messages.append(preview_msg)

        return ConversationSession(
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_activity=datetime.fromisoformat(row["last_activity"]),
            project_path=row["project_path"],
            provider=provider,
            model=row["model"],
            max_tokens=row["max_tokens"],
            reserved_tokens=row["reserved_tokens"],
            active_files=metadata.get("active_files", []),
            tool_usage_count=metadata.get("tool_usage_count", 0),
            profile=row["profile"] if "profile" in row_keys else None,
            model_family=model_family,
            model_size=model_size,
            model_params_b=(row["model_params_b"] if "model_params_b" in row_keys else None),
            context_size=context_size,
            context_tokens=(row["context_tokens"] if "context_tokens" in row_keys else None),
            tool_capable=(bool(row["tool_capable"]) if "tool_capable" in row_keys else False),
            is_moe=bool(row["is_moe"]) if "is_moe" in row_keys else False,
            is_reasoning=(bool(row["is_reasoning"]) if "is_reasoning" in row_keys else False),
            # Rich session metadata (from metadata column)
            title=metadata.get("title"),
            tags=metadata.get("tags", []),
            # State persistence (from metadata column)
            conversation_state=metadata.get("conversation_state"),
            execution_state=metadata.get("execution_state"),
            session_ledger=metadata.get("session_ledger"),
            compaction_hierarchy=metadata.get("compaction_hierarchy"),
            # Preview messages (loaded from metadata)
            preview_messages=preview_messages,
        )

    def _message_from_row(self, row: sqlite3.Row) -> ConversationMessage:
        """Create message from database row."""
        metadata = json_loads(row["metadata"] or "{}")
        row_keys = row.keys()
        for key in (
            "agent_id",
            "parent_session_id",
            "team_id",
            "member_id",
            "plan_id",
            "plan_step_id",
        ):
            if key in row_keys and row[key] is not None:
                metadata.setdefault(key, row[key])
        return ConversationMessage(
            id=row["id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            token_count=row["token_count"],
            priority=MessagePriority(row["priority"]),
            tool_name=row["tool_name"],
            tool_call_id=row["tool_call_id"],
            metadata=metadata,
        )

    @staticmethod
    def _generate_session_id() -> str:
        """Generate unique session ID."""
        return f"session_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _generate_message_id() -> str:
        """Generate unique message ID."""
        return f"msg_{uuid.uuid4().hex[:12]}"

    # =========================================================================
    # SEMANTIC RETRIEVAL METHODS
    # For enhanced context compaction using embeddings
    # =========================================================================

    def set_embedding_service(self, service: "EmbeddingService") -> None:
        """Set the embedding service for semantic operations.

        Args:
            service: EmbeddingService instance for computing embeddings
        """
        self._embedding_service = service

    def set_embedding_store(self, store: "ConversationEmbeddingStore") -> None:
        """Set the LanceDB embedding store for efficient vector search.

        When set, message embeddings are synced to LanceDB on add_message(),
        and semantic retrieval uses LanceDB vector search instead of
        on-the-fly embedding computation.

        Args:
            store: ConversationEmbeddingStore instance
        """
        self._embedding_store = store
        logger.info("ConversationStore: LanceDB embedding store configured")

    def get_semantically_relevant_messages(
        self,
        session_id: str,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        exclude_recent: int = 5,
    ) -> List[Tuple[ConversationMessage, float]]:
        """Retrieve messages semantically relevant to a query from the full history.

        This enables enhanced context compaction by finding relevant historical
        messages that may have been pruned from in-memory context.

        Uses LanceDB vector search when available (O(log n)), falling back to
        on-the-fly embedding computation (O(n)) if not configured.

        Args:
            session_id: Session to search in
            query: Query text to find similar messages for
            limit: Maximum messages to return
            min_similarity: Minimum cosine similarity threshold (0-1)
            exclude_recent: Skip N most recent messages (already in context)

        Returns:
            List of (message, similarity_score) tuples sorted by similarity
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Cannot call get_semantically_relevant_messages from async context. "
                "Use await aget_semantically_relevant_messages(...) instead."
            )

        return run_sync(
            self.aget_semantically_relevant_messages(
                session_id=session_id,
                query=query,
                limit=limit,
                min_similarity=min_similarity,
                exclude_recent=exclude_recent,
            )
        )

    async def aget_semantically_relevant_messages(
        self,
        session_id: str,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        exclude_recent: int = 5,
    ) -> List[Tuple[ConversationMessage, float]]:
        """Async variant of semantic message retrieval for runtime code paths."""
        # Try LanceDB vector search first (much faster)
        if hasattr(self, "_embedding_store") and self._embedding_store is not None:
            return await self._aget_relevant_messages_via_lancedb(
                session_id, query, limit, min_similarity, exclude_recent
            )

        # Fall back to on-the-fly embedding (slower but works without LanceDB)
        return await asyncio.to_thread(
            self._get_relevant_messages_via_embedding,
            session_id,
            query,
            limit,
            min_similarity,
            exclude_recent,
        )

    def _get_relevant_messages_via_lancedb(
        self,
        session_id: str,
        query: str,
        limit: int,
        min_similarity: float,
        exclude_recent: int,
    ) -> List[Tuple[ConversationMessage, float]]:
        """Retrieve relevant messages using LanceDB vector search.

        Much faster than on-the-fly embedding - O(log n) vs O(n).
        """
        return run_sync(
            self._aget_relevant_messages_via_lancedb(
                session_id=session_id,
                query=query,
                limit=limit,
                min_similarity=min_similarity,
                exclude_recent=exclude_recent,
            )
        )

    def _get_excluded_message_ids(self, session_id: str, exclude_recent: int) -> List[str]:
        """Fetch recent message IDs that should be excluded from semantic search."""
        # Get recent message IDs to exclude
        exclude_ids: List[str] = []
        if exclude_recent > 0:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                recent_rows = conn.execute(
                    """
                    SELECT id FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, exclude_recent),
                ).fetchall()
                exclude_ids = [row["id"] for row in recent_rows]
        return exclude_ids

    async def _aget_relevant_messages_via_lancedb(
        self,
        session_id: str,
        query: str,
        limit: int,
        min_similarity: float,
        exclude_recent: int,
    ) -> List[Tuple[ConversationMessage, float]]:
        """Retrieve relevant messages using the async embedding store path."""
        exclude_ids = self._get_excluded_message_ids(session_id, exclude_recent)

        search_results = await self._embedding_store.search_similar(
            query=query,
            session_id=session_id,
            limit=limit,
            min_similarity=min_similarity,
            exclude_message_ids=exclude_ids,
        )

        # Fetch full messages from SQLite for the matching IDs
        if not search_results:
            return []

        result_ids = [r.message_id for r in search_results]
        similarity_map = {r.message_id: r.similarity for r in search_results}

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" * len(result_ids))
            rows = conn.execute(
                f"""
                SELECT * FROM messages
                WHERE id IN ({placeholders})
                """,
                result_ids,
            ).fetchall()

        # Build result list with similarity scores
        results: List[Tuple[ConversationMessage, float]] = []
        for row in rows:
            message = self._message_from_row(row)
            similarity = similarity_map.get(message.id, 0.0)
            results.append((message, similarity))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _get_relevant_messages_via_embedding(
        self,
        session_id: str,
        query: str,
        limit: int,
        min_similarity: float,
        exclude_recent: int,
    ) -> List[Tuple[ConversationMessage, float]]:
        """Retrieve relevant messages using on-the-fly embedding computation.

        Fallback when LanceDB is not available. Slower but works without setup.
        """
        if not hasattr(self, "_embedding_service") or self._embedding_service is None:
            logger.warning("No embedding service set for semantic retrieval")
            return []

        try:
            # Load all messages from session (excluding recent ones already in context)
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    """,
                    (session_id,),
                ).fetchall()

            if not rows or len(rows) <= exclude_recent:
                return []

            # Skip recent messages (already in active context)
            historical_rows = rows[exclude_recent:]

            # Get query embedding
            query_embedding = self._embedding_service.embed_text_sync(query[:2000])

            # Compute embeddings and similarities for historical messages
            scored_messages: List[Tuple[ConversationMessage, float]] = []

            # Collect candidate messages + embeddings, then score them against the
            # query in ONE native crossing (batch_cosine_similarity) instead of
            # paying the FFI + observability cost per row. The query vector is
            # converted once, not once per row.
            query_vec = query_embedding.tolist()
            candidates: List[Tuple[ConversationMessage, List[float]]] = []
            for row in historical_rows:
                message = self._message_from_row(row)

                # Skip very short messages
                if len(message.content) < 20:
                    continue

                msg_embedding = self._embedding_service.embed_text_sync(message.content[:2000])
                candidates.append((message, msg_embedding.tolist()))

            if candidates:
                similarities = self._batch_cosine_similarity(
                    query_vec, [emb for _, emb in candidates]
                )
                for (message, _), similarity in zip(candidates, similarities):
                    if similarity >= min_similarity:
                        scored_messages.append((message, similarity))

            # Sort by similarity (descending) and limit
            scored_messages.sort(key=lambda x: x[1], reverse=True)
            return scored_messages[:limit]

        except Exception as e:
            logger.warning(f"Semantic retrieval failed: {e}")
            return []

    def get_relevant_summaries(
        self,
        session_id: str,
        query: str,
        limit: int = 3,
        min_similarity: float = 0.25,
    ) -> List[Tuple[str, float]]:
        """Retrieve context summaries relevant to current query.

        Searches the context_summaries table for summaries that might
        contain relevant historical context.

        Args:
            session_id: Session to search in
            query: Query text to find relevant summaries for
            limit: Maximum summaries to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (summary_text, similarity_score) tuples
        """
        if not hasattr(self, "_embedding_service") or self._embedding_service is None:
            return []

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT summary, created_at FROM context_summaries
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT 20
                    """,
                    (session_id,),
                ).fetchall()

            if not rows:
                return []

            query_embedding = self._embedding_service.embed(query[:2000])
            scored_summaries: List[Tuple[str, float]] = []

            # Collect candidate summaries + embeddings, then score them in ONE
            # native crossing instead of one cosine_similarity call per row.
            candidates: List[Tuple[str, List[float]]] = []
            for row in rows:
                summary = row["summary"]
                if len(summary) < 10:
                    continue

                summary_embedding = self._embedding_service.embed(summary[:2000])
                candidates.append((summary, summary_embedding))

            if candidates:
                similarities = self._batch_cosine_similarity(
                    query_embedding, [emb for _, emb in candidates]
                )
                for (summary, _), similarity in zip(candidates, similarities):
                    if similarity >= min_similarity:
                        scored_summaries.append((summary, similarity))

            scored_summaries.sort(key=lambda x: x[1], reverse=True)
            return scored_summaries[:limit]

        except Exception as e:
            logger.warning(f"Summary retrieval failed: {e}")
            return []

    def store_compaction_summary(
        self,
        session_id: str,
        summary: str,
        messages_summarized: List[str],
    ) -> None:
        """Store a compaction summary for later retrieval.

        When context is compacted, store a summary of what was removed
        so it can be retrieved later if needed.

        Args:
            session_id: Session the summary belongs to
            summary: Summary text describing compacted content
            messages_summarized: List of message IDs that were summarized
        """
        if not summary or len(summary) < 5:
            return

        token_count = len(summary) // self.chars_per_token

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO context_summaries
                (id, session_id, summary, token_count, messages_summarized, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"sum_{uuid.uuid4().hex[:12]}",
                    session_id,
                    summary,
                    token_count,
                    json_dumps(messages_summarized),
                    datetime.now().isoformat(),
                ),
            )

        logger.debug(f"Stored compaction summary for session {session_id}")

    def store_compaction_summary_enhanced(
        self,
        session_id: str,
        summary_xml: str,
        summary_text: str,
        summary_json: Dict[str, Any],
        messages_summarized: List[str],
        strategy_used: str,
        complexity_score: float,
        tokens_saved: int,
        duration_ms: int,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> str:
        """Store enhanced compaction summary with dual formats.

        Stores both XML (machine-readable) and natural language formats,
        along with metadata for analytics and monitoring.

        Args:
            session_id: Session the summary belongs to
            summary_xml: Machine-readable XML format
            summary_text: Natural language format
            summary_json: Structured summary data as dict
            messages_summarized: List of message IDs that were summarized
            strategy_used: 'rule_based', 'llm_based', or 'hybrid'
            complexity_score: 0.0-1.0 complexity score
            tokens_saved: Estimated tokens saved
            duration_ms: Compaction duration in milliseconds
            llm_provider: LLM provider used (if applicable)
            llm_model: LLM model used (if applicable)
            success: Whether compaction succeeded
            error_message: Error message if failed

        Returns:
            Summary ID
        """
        summary_id = f"sum_{uuid.uuid4().hex[:12]}"

        with sqlite3.connect(self.db_path) as conn:
            # Determine format
            if summary_xml and summary_text:
                summary_format = "both"
            elif summary_xml:
                summary_format = "xml"
            else:
                summary_format = "natural"

            # Store enhanced summary
            conn.execute(
                """
                INSERT INTO context_summaries (
                    id, session_id, summary_format, summary_xml, summary_text,
                    summary_json, messages_summarized, messages_summarized_json,
                    strategy_used, complexity_score, estimated_tokens_saved,
                    token_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary_id,
                    session_id,
                    summary_format,
                    summary_xml,
                    summary_text,
                    json_dumps(summary_json),
                    json_dumps(messages_summarized),
                    json_dumps(messages_summarized),
                    strategy_used,
                    complexity_score,
                    tokens_saved,
                    len(summary_xml or summary_text) // self.chars_per_token,
                    datetime.now().isoformat(),
                ),
            )

            # Log to compaction_history for analytics
            history_id = f"hist_{uuid.uuid4().hex[:12]}"
            conn.execute(
                """
                INSERT INTO compaction_history (
                    id, session_id, strategy_used, message_count_before,
                    message_count_after, token_count_before, token_count_after,
                    duration_ms, llm_provider, llm_model, success, error_message,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    history_id,
                    session_id,
                    strategy_used,
                    len(messages_summarized),  # message_count_before
                    0,  # message_count_after (not tracked here)
                    tokens_saved
                    + len(summary_xml or summary_text)
                    // self.chars_per_token,  # token_count_before (estimated)
                    len(summary_xml or summary_text) // self.chars_per_token,  # token_count_after
                    duration_ms,
                    llm_provider,
                    llm_model,
                    success,
                    error_message,
                    datetime.now().isoformat(),
                ),
            )

        logger.debug(
            f"Stored enhanced compaction summary for session {session_id} "
            f"(strategy={strategy_used}, tokens_saved={tokens_saved})"
        )

        return summary_id

    def get_compaction_summaries(
        self,
        session_id: str,
        format_preference: str = "both",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve compaction summaries with format preference.

        Args:
            session_id: Session to retrieve summaries for
            format_preference: 'xml', 'natural', or 'both'
            limit: Maximum number of summaries to retrieve

        Returns:
            List of summary dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            if format_preference == "xml":
                cursor = conn.execute(
                    """
                    SELECT id, summary_xml as summary, strategy_used,
                           complexity_score, estimated_tokens_saved, created_at
                    FROM context_summaries
                    WHERE session_id = ? AND summary_xml IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )
            elif format_preference == "natural":
                cursor = conn.execute(
                    """
                    SELECT id, summary_text as summary, strategy_used,
                           complexity_score, estimated_tokens_saved, created_at
                    FROM context_summaries
                    WHERE session_id = ? AND summary_text IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )
            else:  # both
                cursor = conn.execute(
                    """
                    SELECT id, summary_xml, summary_text, strategy_used,
                           complexity_score, estimated_tokens_saved, created_at
                    FROM context_summaries
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )

            summaries = []
            for row in cursor.fetchall():
                summary = {
                    "id": row[0],
                    "summary_xml": row[1] if format_preference != "natural" else None,
                    "summary_text": (row[2] if format_preference == "natural" else row[2]),
                    "strategy_used": row[3],
                    "complexity_score": row[4],
                    "estimated_tokens_saved": row[5],
                    "created_at": row[6],
                }
                if format_preference == "both":
                    summary["summary_xml"] = row[1]
                    summary["summary_text"] = row[2]
                    summary["strategy_used"] = row[3]
                summaries.append(summary)

            return summaries

    def get_compaction_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve compaction history for analytics.

        Args:
            session_id: Session to retrieve history for
            limit: Maximum number of events to retrieve

        Returns:
            List of compaction event dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, session_id, strategy_used, message_count_before,
                       message_count_after, token_count_before, token_count_after,
                       duration_ms, llm_provider, llm_model, success, error_message,
                       created_at
                FROM compaction_history
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            )

            history = []
            for row in cursor.fetchall():
                history.append(
                    {
                        "id": row[0],
                        "session_id": row[1],
                        "strategy_used": row[2],
                        "message_count_before": row[3],
                        "message_count_after": row[4],
                        "token_count_before": row[5],
                        "token_count_after": row[6],
                        "duration_ms": row[7],
                        "llm_provider": row[8],
                        "llm_model": row[9],
                        "success": bool(row[10]),
                        "error_message": row[11],
                        "created_at": row[12],
                    }
                )

            return history

    def store_compaction_history(
        self,
        session_id: str,
        strategy_used: str,
        message_count_before: int,
        message_count_after: int,
        token_count_before: int,
        token_count_after: int,
        duration_ms: int,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> str:
        """Store compaction event to history table for analytics.

        Args:
            session_id: Session ID
            strategy_used: Compaction strategy used (rule_based, llm_based, hybrid)
            message_count_before: Message count before compaction
            message_count_after: Message count after compaction
            token_count_before: Estimated token count before compaction
            token_count_after: Token count after compaction
            duration_ms: Duration of compaction in milliseconds
            llm_provider: LLM provider used (if applicable)
            llm_model: LLM model used (if applicable)
            success: Whether compaction succeeded
            error_message: Error message if failed

        Returns:
            History event ID
        """
        history_id = f"hist_{uuid.uuid4().hex[:12]}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO compaction_history (
                    id, session_id, strategy_used, message_count_before,
                    message_count_after, token_count_before, token_count_after,
                    duration_ms, llm_provider, llm_model, success, error_message,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    history_id,
                    session_id,
                    strategy_used,
                    message_count_before,
                    message_count_after,
                    token_count_before,
                    token_count_after,
                    duration_ms,
                    llm_provider,
                    llm_model,
                    success,
                    error_message,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        logger.debug(
            f"Stored compaction history event {history_id} for session {session_id} "
            f"(strategy={strategy_used}, duration_ms={duration_ms}, success={success})"
        )

        return history_id

    def get_compaction_statistics(
        self,
        session_id: Optional[str] = None,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """Get aggregate compaction statistics.

        Args:
            session_id: Optional session ID to filter by (None = all sessions)
            limit: Maximum number of events to analyze

        Returns:
            Dictionary with aggregate statistics:
            - total_compactions: Total number of compaction events
            - success_rate: Overall success rate (0.0-1.0)
            - avg_duration_ms: Average duration in milliseconds
            - avg_tokens_saved: Average tokens saved
            - strategy_counts: Count per strategy
            - strategy_success_rates: Success rate per strategy
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Build query with optional session filter
            session_filter = "WHERE session_id = ?" if session_id else ""
            session_params = (session_id,) if session_id else ()

            # Get total statistics
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total_compactions,
                    AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(duration_ms) as avg_duration_ms,
                    AVG(token_count_before - token_count_after) as avg_tokens_saved
                FROM compaction_history
                {session_filter}
                LIMIT ?
                """,
                session_params + (limit,),
            ).fetchone()

            # Get counts per strategy
            strategy_rows = conn.execute(
                f"""
                SELECT
                    strategy_used,
                    COUNT(*) as count,
                    AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(duration_ms) as avg_duration_ms,
                    AVG(token_count_before - token_count_after) as avg_tokens_saved
                FROM compaction_history
                {session_filter}
                GROUP BY strategy_used
                LIMIT ?
                """,
                session_params + (limit,),
            ).fetchall()

            strategy_counts = {}
            strategy_success_rates = {}
            strategy_durations = {}
            strategy_savings = {}

            for row in strategy_rows:
                strategy = row["strategy_used"]
                strategy_counts[strategy] = row["count"]
                strategy_success_rates[strategy] = row["success_rate"]
                strategy_durations[strategy] = row["avg_duration_ms"]
                strategy_savings[strategy] = row["avg_tokens_saved"]

            return {
                "total_compactions": row["total_compactions"] if row else 0,
                "success_rate": row["success_rate"] if row else 0.0,
                "avg_duration_ms": row["avg_duration_ms"] if row else 0.0,
                "avg_tokens_saved": row["avg_tokens_saved"] if row else 0.0,
                "strategy_counts": strategy_counts,
                "strategy_success_rates": strategy_success_rates,
                "strategy_durations": strategy_durations,
                "strategy_savings": strategy_savings,
            }

    def get_compaction_performance_trends(
        self,
        session_id: Optional[str] = None,
        hours_back: int = 24,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get time-series performance trends for compaction.

        Args:
            session_id: Optional session ID to filter by (None = all sessions)
            hours_back: Number of hours to look back (default: 24)
            limit: Maximum number of events to return

        Returns:
            List of compaction events with timestamp, ordered by time
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Calculate cutoff time
            from datetime import timedelta

            cutoff_time = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()

            # Build query with optional session filter
            session_filter = "AND session_id = ?" if session_id else ""
            session_params = (session_id,) if session_id else ()

            rows = conn.execute(
                f"""
                SELECT
                    id, session_id, strategy_used, message_count_before,
                    message_count_after, token_count_before, token_count_after,
                    duration_ms, success, error_message, created_at
                FROM compaction_history
                WHERE created_at >= ?
                {session_filter}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (cutoff_time,) + session_params + (limit,),
            ).fetchall()

            trends = []
            for row in rows:
                trends.append(
                    {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "strategy_used": row["strategy_used"],
                        "message_count_before": row["message_count_before"],
                        "message_count_after": row["message_count_after"],
                        "token_count_before": row["token_count_before"],
                        "token_count_after": row["token_count_after"],
                        "tokens_saved": row["token_count_before"] - row["token_count_after"],
                        "duration_ms": row["duration_ms"],
                        "success": bool(row["success"]),
                        "error_message": row["error_message"],
                        "created_at": row["created_at"],
                    }
                )

            return trends

    def get_strategy_comparison(
        self,
        limit: int = 1000,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare performance across compaction strategies.

        Args:
            limit: Maximum number of events to analyze per strategy

        Returns:
            Dictionary mapping strategy names to their performance metrics
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            rows = conn.execute(
                """
                SELECT
                    strategy_used,
                    COUNT(*) as total_compactions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_compactions,
                    AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(duration_ms) as avg_duration_ms,
                    MIN(duration_ms) as min_duration_ms,
                    MAX(duration_ms) as max_duration_ms,
                    AVG(token_count_before - token_count_after) as avg_tokens_saved,
                    MIN(token_count_before - token_count_after) as min_tokens_saved,
                    MAX(token_count_before - token_count_after) as max_tokens_saved
                FROM compaction_history
                GROUP BY strategy_used
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            comparison = {}
            for row in rows:
                strategy = row["strategy_used"]
                comparison[strategy] = {
                    "total_compactions": row["total_compactions"],
                    "successful_compactions": row["successful_compactions"],
                    "success_rate": row["success_rate"],
                    "avg_duration_ms": row["avg_duration_ms"],
                    "min_duration_ms": row["min_duration_ms"],
                    "max_duration_ms": row["max_duration_ms"],
                    "avg_tokens_saved": row["avg_tokens_saved"],
                    "min_tokens_saved": row["min_tokens_saved"],
                    "max_tokens_saved": row["max_tokens_saved"],
                }

            return comparison

    def get_historical_tool_results(
        self,
        session_id: str,
        tool_names: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[ConversationMessage]:
        """Retrieve historical tool results from the session.

        Useful for finding relevant previous tool outputs that might
        inform the current task.

        Args:
            session_id: Session to search in
            tool_names: Optional list of tool names to filter by
            limit: Maximum results to return

        Returns:
            List of tool result messages
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if tool_names:
                placeholders = ",".join("?" * len(tool_names))
                rows = conn.execute(
                    f"""
                    SELECT * FROM messages
                    WHERE session_id = ?
                    AND role = 'tool'
                    AND tool_name IN ({placeholders})
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, *tool_names, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE session_id = ?
                    AND role = 'tool'
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, limit),
                ).fetchall()

            return [self._message_from_row(row) for row in rows]

    def search_messages_fts(
        self,
        session_id: str,
        query: str,
        limit: int = 20,
        roles: Optional[List[str]] = None,
    ) -> List[ConversationMessage]:
        """Search messages using FTS5 full-text search.

        Uses SQLite FTS5 for O(log n) keyword search instead of O(n) linear scan.
        This is ~20x faster than linear content matching for large conversations.

        Args:
            session_id: Session to search in
            query: Search query (supports FTS5 syntax: AND, OR, NOT, phrases)
            limit: Maximum results to return
            roles: Optional list of roles to filter by (user, assistant, tool_result)

        Returns:
            List of matching messages ordered by relevance (BM25 rank)

        Example:
            # Simple keyword search
            results = memory.search_messages_fts(session_id, "authentication error")

            # FTS5 phrase search
            results = memory.search_messages_fts(session_id, '"login failed"')

            # FTS5 boolean query
            results = memory.search_messages_fts(session_id, "error AND NOT warning")
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Build the query based on role filtering
            if roles:
                placeholders = ",".join("?" * len(roles))
                sql = f"""
                    SELECT m.*, bm25(messages_fts) AS rank
                    FROM messages_fts fts
                    JOIN messages m ON fts.message_id = m.id
                    WHERE fts.session_id = ?
                    AND messages_fts MATCH ?
                    AND m.role IN ({placeholders})
                    ORDER BY rank
                    LIMIT ?
                """
                params = (session_id, query, *roles, limit)
            else:
                sql = """
                    SELECT m.*, bm25(messages_fts) AS rank
                    FROM messages_fts fts
                    JOIN messages m ON fts.message_id = m.id
                    WHERE fts.session_id = ?
                    AND messages_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """
                params = (session_id, query, limit)

            try:
                rows = conn.execute(sql, params).fetchall()
                return [self._message_from_row(row) for row in rows]
            except sqlite3.OperationalError as e:
                # FTS5 may not be available in all SQLite builds
                logger.warning(f"FTS5 search failed, falling back to LIKE: {e}")
                return self._fallback_content_search(conn, session_id, query, limit, roles)

    def _fallback_content_search(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        query: str,
        limit: int,
        roles: Optional[List[str]] = None,
    ) -> List[ConversationMessage]:
        """Fallback content search using LIKE when FTS5 is unavailable."""
        # Simple tokenization for LIKE search
        terms = query.split()
        like_patterns = [f"%{term}%" for term in terms]

        if roles:
            role_placeholders = ",".join("?" * len(roles))
            where_clause = f"session_id = ? AND role IN ({role_placeholders})"
            params = [session_id, *roles]
        else:
            where_clause = "session_id = ?"
            params = [session_id]

        # Build AND conditions for each term
        for pattern in like_patterns:
            where_clause += " AND content LIKE ?"
            params.append(pattern)

        params.append(limit)

        rows = conn.execute(
            f"""
            SELECT * FROM messages
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [self._message_from_row(row) for row in rows]

    def rebuild_fts_index(self, session_id: Optional[str] = None) -> int:
        """Rebuild FTS5 index from messages table.

        Useful after schema migrations or to fix corrupted indexes.

        Args:
            session_id: Optional session to rebuild (all if None)

        Returns:
            Number of messages indexed
        """
        with sqlite3.connect(self.db_path) as conn:
            # Clear existing FTS content
            if session_id:
                conn.execute(
                    """
                    INSERT INTO messages_fts(messages_fts, rowid, content, message_id, session_id)
                    SELECT 'delete', rowid, content, id, session_id
                    FROM messages WHERE session_id = ?
                    """,
                    (session_id,),
                )
                # Rebuild for specific session
                cursor = conn.execute(
                    """
                    INSERT INTO messages_fts(rowid, content, message_id, session_id)
                    SELECT rowid, content, id, session_id
                    FROM messages WHERE session_id = ?
                    """,
                    (session_id,),
                )
            else:
                # Full rebuild - delete all then reinsert
                conn.execute("DELETE FROM messages_fts")
                cursor = conn.execute("""
                    INSERT INTO messages_fts(rowid, content, message_id, session_id)
                    SELECT rowid, content, id, session_id FROM messages
                    """)

            count = cursor.rowcount
            conn.commit()
            logger.info(f"Rebuilt FTS5 index: {count} messages indexed")
            return count

    # =========================================================================
    # ML/RL AGGREGATION METHODS
    # Efficient queries using normalized FK columns for training data extraction
    # =========================================================================

    def get_provider_stats(self) -> List[Dict[str, Any]]:
        """Get session statistics grouped by provider.

        Returns aggregated stats for ML/RL analysis of provider performance.
        Uses normalized provider_id FK for efficient GROUP BY.

        Returns:
            List of dicts with provider, session_count, total_messages,
            avg_messages_per_session, tool_capable_pct
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    p.name AS provider,
                    COUNT(DISTINCT s.session_id) AS session_count,
                    COUNT(m.id) AS total_messages,
                    ROUND(COUNT(m.id) * 1.0 / COUNT(DISTINCT s.session_id), 1)
                        AS avg_messages_per_session,
                    ROUND(AVG(s.tool_capable) * 100, 1)
                        AS tool_capable_pct
                FROM sessions s
                LEFT JOIN providers p ON s.provider_id = p.id
                LEFT JOIN messages m ON s.session_id = m.session_id
                WHERE p.name IS NOT NULL
                GROUP BY p.name
                ORDER BY session_count DESC
                """).fetchall()

            return [dict(row) for row in rows]

    def get_model_family_stats(self) -> List[Dict[str, Any]]:
        """Get session statistics grouped by model family.

        Returns aggregated stats for ML/RL analysis of model architecture performance.
        Uses normalized model_family_id FK for efficient GROUP BY.

        Returns:
            List of dicts with model_family, session_count, total_messages,
            avg_params_b, tool_capable_pct, moe_pct, reasoning_pct
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    mf.name AS model_family,
                    COUNT(DISTINCT s.session_id) AS session_count,
                    COUNT(m.id) AS total_messages,
                    ROUND(AVG(s.model_params_b), 1) AS avg_params_b,
                    ROUND(AVG(s.tool_capable) * 100, 1) AS tool_capable_pct,
                    ROUND(AVG(s.is_moe) * 100, 1) AS moe_pct,
                    ROUND(AVG(s.is_reasoning) * 100, 1) AS reasoning_pct
                FROM sessions s
                LEFT JOIN model_families mf ON s.model_family_id = mf.id
                LEFT JOIN messages m ON s.session_id = m.session_id
                WHERE mf.name IS NOT NULL
                GROUP BY mf.name
                ORDER BY session_count DESC
                """).fetchall()

            return [dict(row) for row in rows]

    def get_model_size_stats(self) -> List[Dict[str, Any]]:
        """Get session statistics grouped by model size category.

        Returns aggregated stats for ML/RL analysis of model size impact.
        Uses normalized model_size_id FK for efficient GROUP BY.

        Returns:
            List of dicts with model_size, session_count, total_messages,
            avg_context_tokens, tool_capable_pct
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT
                    ms.name AS model_size,
                    COUNT(DISTINCT s.session_id) AS session_count,
                    COUNT(m.id) AS total_messages,
                    ROUND(AVG(s.context_tokens), 0) AS avg_context_tokens,
                    ROUND(AVG(s.tool_capable) * 100, 1) AS tool_capable_pct
                FROM sessions s
                LEFT JOIN model_sizes ms ON s.model_size_id = ms.id
                LEFT JOIN messages m ON s.session_id = m.session_id
                WHERE ms.name IS NOT NULL
                GROUP BY ms.name
                ORDER BY
                    CASE ms.name
                        WHEN 'tiny' THEN 1
                        WHEN 'small' THEN 2
                        WHEN 'medium' THEN 3
                        WHEN 'large' THEN 4
                        WHEN 'xlarge' THEN 5
                        WHEN 'xxlarge' THEN 6
                    END
                """).fetchall()

            return [dict(row) for row in rows]

    def get_rl_training_data(
        self,
        limit: int = 1000,
        min_messages: int = 2,
    ) -> List[Dict[str, Any]]:
        """Get comprehensive session data for RL training.

        Returns fully denormalized session data with all ML-friendly features
        for training reinforcement learning models on conversation quality.

        Args:
            limit: Maximum sessions to return
            min_messages: Minimum messages per session to include

        Returns:
            List of dicts with all normalized columns plus computed metrics:
            - provider, model, profile
            - model_family, model_size, context_size
            - model_params_b, context_tokens
            - tool_capable, is_moe, is_reasoning
            - message_count, user_messages, assistant_messages, tool_messages
            - session_duration_minutes
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    s.session_id,
                    s.model,
                    s.profile,
                    p.name AS provider,
                    mf.name AS model_family,
                    ms.name AS model_size,
                    cs.name AS context_size,
                    s.model_params_b,
                    s.context_tokens,
                    s.tool_capable,
                    s.is_moe,
                    s.is_reasoning,
                    COUNT(m.id) AS message_count,
                    SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) AS user_messages,
                    SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) AS assistant_messages,
                    SUM(CASE WHEN m.role IN ('tool_call', 'tool')
                        THEN 1 ELSE 0 END) AS tool_messages,
                    ROUND(
                        (JULIANDAY(MAX(m.timestamp)) - JULIANDAY(MIN(m.timestamp))) * 24 * 60,
                        1
                    ) AS session_duration_minutes
                FROM sessions s
                LEFT JOIN providers p ON s.provider_id = p.id
                LEFT JOIN model_families mf ON s.model_family_id = mf.id
                LEFT JOIN model_sizes ms ON s.model_size_id = ms.id
                LEFT JOIN context_sizes cs ON s.context_size_id = cs.id
                LEFT JOIN messages m ON s.session_id = m.session_id
                GROUP BY s.session_id
                HAVING COUNT(m.id) >= ?
                ORDER BY s.last_activity DESC
                LIMIT ?
                """,
                (min_messages, limit),
            ).fetchall()

            return [dict(row) for row in rows]

    def get_aggregation_summary(self) -> Dict[str, Any]:
        """Get high-level aggregation summary for ML/RL dashboard.

        Returns:
            Dict with counts and breakdowns for quick overview
        """
        with sqlite3.connect(self.db_path) as conn:
            # Total counts
            session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

            # Provider breakdown
            provider_counts = conn.execute("""
                SELECT p.name, COUNT(*) AS count
                FROM sessions s
                JOIN providers p ON s.provider_id = p.id
                GROUP BY p.name
                ORDER BY count DESC
                LIMIT 5
                """).fetchall()

            # Model family breakdown
            family_counts = conn.execute("""
                SELECT mf.name, COUNT(*) AS count
                FROM sessions s
                JOIN model_families mf ON s.model_family_id = mf.id
                GROUP BY mf.name
                ORDER BY count DESC
                LIMIT 5
                """).fetchall()

            # Tool capability stats
            tool_stats = conn.execute("""
                SELECT
                    SUM(CASE WHEN tool_capable = 1 THEN 1 ELSE 0 END) AS tool_capable,
                    SUM(CASE WHEN tool_capable = 0 THEN 1 ELSE 0 END) AS not_tool_capable
                FROM sessions
                """).fetchone()

            return {
                "total_sessions": session_count,
                "total_messages": message_count,
                "top_providers": [(row[0], row[1]) for row in provider_counts],
                "top_families": [(row[0], row[1]) for row in family_counts],
                "tool_capable_sessions": tool_stats[0] or 0,
                "not_tool_capable_sessions": tool_stats[1] or 0,
            }

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Uses Rust-accelerated implementation with NumPy fallback.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            from victor.processing.native import cosine_similarity

            return cosine_similarity(vec1, vec2)
        except Exception as e:
            logger.debug("Cosine similarity computation failed, returning 0.0: %s", e)
        return 0.0

    def _batch_cosine_similarity(
        self, query: List[float], corpus: List[List[float]]
    ) -> List[float]:
        """Cosine similarity of ``query`` against many vectors in one crossing.

        Amortises the native FFI + dispatch cost over the whole corpus instead
        of paying it per pair. Falls back to per-item scoring on any error so
        behaviour matches :meth:`_cosine_similarity`.
        """
        if not corpus:
            return []
        try:
            from victor.processing.native import batch_cosine_similarity

            return batch_cosine_similarity(query, corpus)
        except Exception as e:
            logger.debug("Batch cosine similarity failed, scoring per-item: %s", e)
            return [self._cosine_similarity(query, vec) for vec in corpus]


# Convenience function to get global store
_global_store: Optional[ConversationStore] = None


def get_conversation_manager() -> ConversationStore:
    """Get the global conversation store.

    Note: Function name kept for backward compatibility.
    """
    global _global_store
    if _global_store is None:
        _global_store = ConversationStore()
    return _global_store
