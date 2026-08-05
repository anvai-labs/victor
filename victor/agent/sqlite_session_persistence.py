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

"""SQLite-based session persistence for Victor conversations.

.. deprecated:: 0.7.0
    This module is DEPRECATED and will be removed in version 0.10.0.
    Use ``ConversationStore`` from ``victor.agent.conversation.store`` instead.

Migration guide:
    # Old API (deprecated):
    from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence
    persistence = get_sqlite_session_persistence()
    sessions = persistence.list_sessions()
    session = persistence.load_session(session_id)

    # New API (canonical):
    from victor.agent.conversation.store import ConversationStore
    store = ConversationStore()
    sessions = store.list_sessions()
    session = store.load_session(session_id)

The ConversationStore provides feature parity and additional capabilities:
- Token-aware context window management
- Priority-based message pruning
- Semantic relevance scoring
- ML/RL-friendly aggregation queries
- Full-text search support
- Rich session metadata (title, tags, state persistence)
"""

from __future__ import annotations

from victor.core.json_utils import json_dumps, json_loads
from json import JSONDecodeError
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    pass  # Reserved for future type imports

logger = logging.getLogger(__name__)


class SQLiteSessionPersistence:
    """SQLite-based session persistence (DEPRECATED).

    .. deprecated:: 0.7.0
        Use ``ConversationStore`` from
        ``victor.agent.conversation.store`` instead. This class
        will be removed in version 0.10.0.

    Stores conversation sessions and messages in the project database,
    providing fast queries and eliminating JSON file duplication.

    Migration guide:
        # Old API:
        from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence
        persistence = get_sqlite_session_persistence()
        sessions = persistence.list_sessions()
        session = persistence.load_session(session_id)

        # New API:
        from victor.agent.conversation.store import ConversationStore
        store = ConversationStore()
        sessions = store.list_sessions()
        session = store.get_session(session_id)

    The ConversationStore provides:
    - Token-aware context window management
    - Priority-based message pruning
    - Semantic relevance scoring
    - ML/RL-friendly aggregation
    - Full-text search support

    Tables Used:
    - sessions: session_id, created_at, last_activity, provider, model, profile, metadata
    - messages: id, session_id, role, content, timestamp, metadata
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize SQLite session persistence.

        Args:
            db_path: Path to project database (default: .victor/project.db)

        .. deprecated:: 0.7.0
            Use ConversationStore from victor.agent.conversation.store instead.
            This class will be removed in version 0.10.0.
        """
        warnings.warn(
            "SQLiteSessionPersistence is deprecated and will be removed in v0.10.0. "
            "Migrate to ConversationStore from victor.agent.conversation.store. "
            "See module docstring for migration guide. "
            "All features have feature parity in ConversationStore.",
            DeprecationWarning,
            stacklevel=2,
        )
        from victor.core.database import get_project_database

        if db_path:
            self._db_path = db_path
        else:
            from victor.config.settings import get_project_paths

            self._db_path = get_project_paths().project_db

        self._db = get_project_database(self._db_path)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Ensure required tables exist."""
        try:
            result = self._db.query(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name IN ('sessions', 'messages')"
            )
            tables = [row[0] for row in result] if result else []

            if "sessions" not in tables or "messages" not in tables:
                logger.warning("Sessions or messages table missing, initializing...")
                self._init_tables()

            self._ensure_schema_compatibility()
        except Exception as e:
            logger.error(f"Error checking tables: {e}")
            self._init_tables()
            self._ensure_schema_compatibility()

    def _init_tables(self) -> None:
        """Initialize sessions and messages tables."""
        from victor.core.schema import Schema

        try:
            # Create sessions table
            self._db.execute(Schema.CONV_SESSION)
            # Create messages table
            self._db.execute(Schema.CONV_MESSAGE)
            logger.debug("Initialized sessions and messages tables")
        except Exception as e:
            logger.error(f"Failed to initialize tables: {e}")

    def _table_columns(self, table_name: str) -> set[str]:
        """Return the current column set for a SQLite table."""
        result = self._db.query(f"PRAGMA table_info({table_name})")
        return {row[1] for row in result} if result else set()

    def _ensure_schema_compatibility(self) -> None:
        """Add compatibility columns expected by the deprecated API."""
        sessions_columns = self._table_columns("sessions")
        messages_columns = self._table_columns("messages")

        if "metadata" not in sessions_columns:
            self._db.execute("ALTER TABLE sessions ADD COLUMN metadata TEXT")
        if "project_path" not in sessions_columns:
            self._db.execute("ALTER TABLE sessions ADD COLUMN project_path TEXT")
        if "provider" not in sessions_columns:
            self._db.execute("ALTER TABLE sessions ADD COLUMN provider TEXT")
        if "model" not in sessions_columns:
            self._db.execute("ALTER TABLE sessions ADD COLUMN model TEXT")
        if "profile" not in sessions_columns:
            self._db.execute("ALTER TABLE sessions ADD COLUMN profile TEXT")
        if "created_at" not in sessions_columns:
            self._db.execute("ALTER TABLE sessions ADD COLUMN created_at TIMESTAMP")
        if "last_activity" not in sessions_columns:
            self._db.execute("ALTER TABLE sessions ADD COLUMN last_activity TIMESTAMP")

        if "metadata" not in messages_columns:
            self._db.execute("ALTER TABLE messages ADD COLUMN metadata TEXT")
        if "tool_name" not in messages_columns:
            self._db.execute("ALTER TABLE messages ADD COLUMN tool_name TEXT")
        if "tool_call_id" not in messages_columns:
            self._db.execute("ALTER TABLE messages ADD COLUMN tool_call_id TEXT")
        if "timestamp" not in messages_columns:
            self._db.execute("ALTER TABLE messages ADD COLUMN timestamp TIMESTAMP")
        if "token_count" not in messages_columns:
            self._db.execute("ALTER TABLE messages ADD COLUMN token_count INTEGER DEFAULT 0")
        if "priority" not in messages_columns:
            self._db.execute("ALTER TABLE messages ADD COLUMN priority INTEGER DEFAULT 0")

    def save_session(
        self,
        conversation: Any,  # MessageHistory or dict
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
        """Save a session to SQLite database.

        Args:
            conversation: MessageHistory instance or dict
            model: Model name used in the session
            provider: Provider name
            profile: Profile name
            session_id: Optional custom session ID (auto-generated if not provided)
            title: Optional custom title (auto-generated if not provided)
            tags: Optional list of tags
            conversation_state: Optional ConversationStateMachine state
            tool_selection_stats: Optional tool usage statistics

        Returns:
            The session ID
        """
        # Convert conversation to dict if needed
        if hasattr(conversation, "to_dict"):
            conversation_data = conversation.to_dict()
        elif isinstance(conversation, dict):
            conversation_data = conversation
        else:
            conversation_data = {"messages": []}

        # Generate session ID if not provided (uses new format: projectroot-base62)
        if not session_id:
            from victor.agent.session_id import generate_session_id

            session_id = generate_session_id()

        # Generate title if not provided
        if not title:
            title = self._generate_title(conversation_data)

        # Get messages
        messages = conversation_data.get("messages", [])
        message_count = len(messages)

        # Timestamps
        now = datetime.now().isoformat()

        # Prepare session data (JSON format for storage)
        session_data = {
            "metadata": {
                "session_id": session_id,
                "created_at": now,
                "updated_at": now,
                "model": model,
                "provider": provider,
                "profile": profile,
                "message_count": message_count,
                "title": title,
                "tags": tags or [],
            },
            "conversation": conversation_data,
            "conversation_state": (
                conversation_state.to_dict()
                if hasattr(conversation_state, "to_dict")
                else conversation_state
            ),
            "tool_selection_stats": tool_selection_stats,
            "execution_state": (
                execution_state.to_dict()
                if hasattr(execution_state, "to_dict")
                else execution_state
            ),
            "session_ledger": (
                session_ledger.to_dict() if hasattr(session_ledger, "to_dict") else session_ledger
            ),
            "compaction_hierarchy": compaction_hierarchy,
        }

        try:
            # Insert or replace session
            self._db.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, created_at, last_activity, project_path, provider, model, profile, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    now,
                    now,
                    str(self._db_path),
                    provider,
                    model,
                    profile,
                    json_dumps(session_data),
                ),
            )

            # Delete existing messages for this session (if updating)
            self._db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

            # Insert messages
            base_time = datetime.now()
            for index, msg in enumerate(messages):
                role = msg.get("role", "")
                content = msg.get("content", "")
                tool_name = msg.get("name")
                tool_call_id = msg.get("tool_call_id")
                metadata = {
                    key: value
                    for key, value in msg.items()
                    if key not in {"role", "content", "name", "tool_call_id"}
                }
                timestamp = (base_time + timedelta(microseconds=index)).isoformat()

                self._db.execute(
                    """INSERT INTO messages
                       (id, session_id, role, content, timestamp, token_count, priority, tool_name, tool_call_id, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"{session_id}:{index:06d}",
                        session_id,
                        role,
                        content,
                        timestamp,
                        len(content.split()) if content else 0,
                        0,
                        tool_name,
                        tool_call_id,
                        json_dumps(metadata) if metadata else None,
                    ),
                )

            logger.debug(f"Saved session {session_id} to SQLite ({message_count} messages)")
            return session_id

        except Exception as e:
            logger.error(f"Failed to save session to SQLite: {e}")
            return ""

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session from SQLite.

        Args:
            session_id: Session ID to load

        Returns:
            Session dictionary or None if not found
        """
        try:
            session_columns = self._table_columns("sessions")
            provider_id_select = "provider_id" if "provider_id" in session_columns else "NULL"
            result = self._db.query(
                f"""SELECT session_id, created_at, last_activity, provider, model, profile, metadata,
                           {provider_id_select}
                    FROM sessions WHERE session_id = ?""",
                (session_id,),
            )

            rows = list(result) if result else []
            if not rows:
                logger.warning(f"Session not found: {session_id}")
                return None

            row = rows[0]
            metadata_payload = {}
            if row[6]:
                try:
                    metadata_payload = json_loads(row[6])
                except JSONDecodeError:
                    logger.warning("Failed to parse session metadata for %s", session_id)

            if self._is_serialized_session_payload(metadata_payload):
                return metadata_payload

            messages = self.get_session_messages(session_id)
            conversation_messages, preview_messages = self._split_preview_messages(messages)
            title = self._generate_title({"messages": conversation_messages})
            session_data = {
                "metadata": {
                    "session_id": row[0],
                    "created_at": row[1],
                    "updated_at": row[2],
                    "model": row[4],
                    "provider": self._resolve_provider_name(row[3], row[7]),
                    "profile": row[5],
                    "message_count": len(conversation_messages),
                    "title": title,
                    "tags": [],
                },
                "conversation": {
                    "messages": conversation_messages,
                    "preview_messages": preview_messages,
                },
                "conversation_state": None,
                "tool_selection_stats": None,
                "execution_state": None,
                "session_ledger": None,
                "compaction_hierarchy": None,
            }

            logger.debug(f"Loaded session {session_id} from SQLite")
            return session_data

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def list_sessions(self, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """List sessions from SQLite.

        Args:
            limit: Maximum number of sessions to return
            offset: Offset for pagination

        Returns:
            List of session dictionaries with metadata
        """
        try:
            session_columns = self._table_columns("sessions")
            provider_id_select = "provider_id" if "provider_id" in session_columns else "NULL"
            result = self._db.query(
                f"""SELECT session_id, provider, model, profile, created_at, last_activity, metadata,
                           {provider_id_select}
                    FROM sessions
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?""",
                (limit, offset),
            )

            sessions = []
            if result:
                for row in result:
                    metadata_payload: Dict[str, Any] = {}
                    if row[6]:
                        try:
                            metadata_payload = json_loads(row[6])
                        except JSONDecodeError:
                            metadata_payload = {}

                    if self._is_serialized_session_payload(metadata_payload):
                        session_metadata = metadata_payload.get("metadata", {})
                        preview_messages = metadata_payload.get("conversation", {}).get(
                            "preview_messages", []
                        )
                        count_result = self._db.query(
                            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                            (row[0],),
                        )
                        message_count = (
                            count_result[0][0]
                            if count_result
                            else session_metadata.get("message_count", 0)
                        )
                        title = session_metadata.get("title", "Untitled Session")
                        updated_at = session_metadata.get("updated_at", row[5])
                    else:
                        session_messages = self.get_session_messages(row[0])
                        conversation_messages, preview_messages = self._split_preview_messages(
                            session_messages
                        )
                        message_count = len(conversation_messages)
                        title = self._generate_title({"messages": conversation_messages})
                        updated_at = row[5]

                    sessions.append(
                        {
                            "session_id": row[0],
                            "title": title,
                            "provider": self._resolve_provider_name(row[1], row[7]),
                            "model": row[2],
                            "profile": row[3],
                            "created_at": row[4],
                            "updated_at": updated_at,
                            "message_count": message_count,
                            "preview_count": (
                                len(preview_messages) if isinstance(preview_messages, list) else 0
                            ),
                        }
                    )

            return sessions

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from SQLite.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted successfully
        """
        try:
            # Delete messages first (foreign key)
            self._db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            # Delete session
            self._db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

            logger.info(f"Deleted session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def search_sessions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search sessions by title or content.

        Args:
            query: Search query string
            limit: Maximum results

        Returns:
            List of matching sessions
        """
        try:
            sessions = []
            lowered_query = query.lower()
            for session in self.list_sessions(limit=100000, offset=0):
                if lowered_query in session["title"].lower():
                    sessions.append(session)
                    continue

                session_data = self.load_session(session["session_id"])
                conversation = session_data.get("conversation", {}) if session_data else {}
                if self._contains_query(
                    conversation.get("messages", []), lowered_query
                ) or self._contains_query(conversation.get("preview_messages", []), lowered_query):
                    sessions.append(session)

                if len(sessions) >= limit:
                    break

            return sessions[:limit]

        except Exception as e:
            logger.error(f"Failed to search sessions: {e}")
            return []

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a session.

        Args:
            session_id: Session ID

        Returns:
            List of message dictionaries
        """
        try:
            result = self._db.query(
                """SELECT role, content, tool_name, tool_call_id, metadata, timestamp
                   FROM messages
                   WHERE session_id = ?
                   ORDER BY timestamp ASC, id ASC""",
                (session_id,),
            )

            messages = []
            if result:
                for row in result:
                    msg = {
                        "role": row[0],
                        "content": row[1],
                        "created_at": row[5],
                    }

                    if row[2]:
                        msg["name"] = row[2]
                    if row[3]:
                        msg["tool_call_id"] = row[3]
                    if row[4]:
                        try:
                            metadata = json_loads(row[4])
                            if isinstance(metadata, dict):
                                msg.update(metadata)
                        except Exception:
                            logger.debug(
                                "Failed to parse message metadata for session %s",
                                session_id,
                            )

                    messages.append(msg)

            return messages

        except Exception as e:
            logger.error(f"Failed to get messages for session {session_id}: {e}")
            return []

    def _resolve_provider_name(self, provider_name: Any, provider_id: Any) -> Optional[str]:
        """Resolve provider name from either legacy text column or normalized provider table."""
        if isinstance(provider_name, str) and provider_name.strip():
            return provider_name

        if provider_id is None or "provider_id" not in self._table_columns("sessions"):
            return provider_name

        try:
            result = self._db.query(
                "SELECT name FROM providers WHERE id = ?",
                (provider_id,),
            )
            if result:
                resolved = result[0][0]
                if isinstance(resolved, str) and resolved.strip():
                    return resolved
        except Exception:
            logger.debug("Failed to resolve provider name for provider_id=%s", provider_id)

        return provider_name

    @staticmethod
    def _is_serialized_session_payload(metadata_payload: Any) -> bool:
        """Return whether metadata contains the legacy full session envelope."""
        return (
            isinstance(metadata_payload, dict)
            and isinstance(metadata_payload.get("metadata"), dict)
            and isinstance(metadata_payload.get("conversation"), dict)
        )

    @staticmethod
    def _is_preview_message_payload(message: Dict[str, Any]) -> bool:
        """Return whether a message row represents a replay-only preview sidecar."""
        if not isinstance(message, dict):
            return False
        return bool(message.get("interactive_preview")) or isinstance(
            message.get("preview_kind"), str
        )

    def _split_preview_messages(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split stored messages into provider-facing and replay-only preview buckets."""
        conversation_messages: List[Dict[str, Any]] = []
        preview_messages: List[Dict[str, Any]] = []

        for message in messages:
            if not self._is_preview_message_payload(message):
                conversation_messages.append(message)
                continue

            preview_metadata = {
                key: value
                for key, value in message.items()
                if key
                not in {
                    "role",
                    "content",
                    "created_at",
                    "name",
                    "tool_call_id",
                    "after_message_index",
                    "interactive_preview",
                }
            }
            after_message_index = message.get("after_message_index")
            if not isinstance(after_message_index, int) or after_message_index < 0:
                after_message_index = len(conversation_messages)

            preview_messages.append(
                {
                    "role": str(message.get("role", "system")),
                    "content": str(message.get("content", "")),
                    "metadata": preview_metadata,
                    "after_message_index": after_message_index,
                }
            )

        return conversation_messages, preview_messages

    def _contains_query(self, value: Any, lowered_query: str) -> bool:
        """Recursively search structured session content for a query string."""
        if isinstance(value, str):
            return lowered_query in value.lower()
        if isinstance(value, dict):
            return any(self._contains_query(item, lowered_query) for item in value.values())
        if isinstance(value, list):
            return any(self._contains_query(item, lowered_query) for item in value)
        return False

    def _generate_title(self, conversation_data: Dict[str, Any]) -> str:
        """Generate a title from the first user message.

        Args:
            conversation_data: Serialized conversation data

        Returns:
            A title string (truncated to 50 chars)
        """
        messages = conversation_data.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = str(msg.get("content", ""))
                # Truncate to 50 chars
                if len(content) > 50:
                    return content[:47] + "..."
                return content
        return "Untitled Session"


def get_sqlite_session_persistence(
    db_path: Optional[Path] = None,
) -> SQLiteSessionPersistence:
    """Get the SQLite session persistence instance (DEPRECATED).

    .. deprecated:: 0.7.0
        Use ``ConversationStore`` from ``victor.agent.conversation.store`` instead.
        This function will be removed in version 0.10.0.

    Args:
        db_path: Optional database path (for testing). If not provided,
                 uses VICTOR_TEST_DB_PATH env var or default path.

    Returns:
        SQLiteSessionPersistence instance

    Migration example:
        # Old:
        from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence
        persistence = get_sqlite_session_persistence()
        sessions = persistence.list_sessions()

        # New:
        from victor.agent.conversation.store import ConversationStore
        store = ConversationStore()
        sessions = store.list_sessions()
    """
    import os

    warnings.warn(
        "get_sqlite_session_persistence() is deprecated. Use ConversationStore from "
        "victor.agent.conversation.store instead. This will be removed in version 0.10.0.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Support test database override via environment variable
    if db_path is None:
        test_db_path = os.environ.get("VICTOR_TEST_DB_PATH")
        if test_db_path:
            db_path = Path(test_db_path)

    return SQLiteSessionPersistence(db_path=db_path)


__all__ = [
    "SQLiteSessionPersistence",
    "get_sqlite_session_persistence",
]
