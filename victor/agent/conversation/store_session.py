# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Session CRUD and management logic for ConversationStore."""

from __future__ import annotations
import logging
import sqlite3
import uuid
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.conversation.store import ConversationStore

from victor.agent.conversation.session_model import ConversationSession
from victor.agent.conversation.types import (
    ConversationMessage,
    MessageRole,
    MessagePriority,
)
from victor.agent.ml_metadata import (
    get_known_model_context,
    get_known_model_params,
    parse_model_metadata,
)

logger = logging.getLogger(__name__)


class ConversationStoreSession:
    """Manages ConversationSession lifecycle and CRUD persistence operations."""

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

    def __init__(self, store: ConversationStore):
        """Initialize session manager.

        Args:
            store: Reference to parent ConversationStore
        """
        self.store = store

    @property
    def db_path(self):
        return self.store.db_path

    @property
    def max_context_tokens(self):
        return self.store.max_context_tokens

    @property
    def response_reserve(self):
        return self.store.response_reserve

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
        """Create a new conversation session."""
        if session_id is None:
            session_id = self.store._generate_session_id()

        model_family = None
        model_size = None
        model_params_b = None
        context_size = None
        context_tokens = None
        is_moe = False
        is_reasoning = False

        if model:
            known_context = get_known_model_context(model)
            known_params = get_known_model_params(model)

            metadata = parse_model_metadata(
                model,
                provider=provider,
                known_context=known_context,
                known_params_b=known_params,
            )

            model_family = metadata.model_family
            model_size = metadata.model_size
            model_params_b = metadata.model_params_b
            context_size = metadata.context_size
            context_tokens = metadata.context_tokens
            is_moe = metadata.is_moe
            is_reasoning = metadata.is_reasoning

        session = ConversationSession(
            session_id=session_id,
            project_path=project_path,
            provider=provider,
            model=model,
            profile=profile,
            max_tokens=max_tokens or self.max_context_tokens,
            reserved_tokens=self.response_reserve,
            model_family=model_family,
            model_size=model_size,
            model_params_b=model_params_b,
            context_size=context_size,
            context_tokens=context_tokens,
            tool_capable=tool_capable,
            is_moe=is_moe,
            is_reasoning=is_reasoning,
        )

        self.store._sessions[session_id] = session
        self.store._persist_session(session)

        fam = model_family.value if model_family else "unknown"
        sz = model_size.value if model_size else "unknown"
        logger.info(f"Created session {session_id} for project: {project_path} [{fam}/{sz}]")

        return session

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
        """Save session with rich metadata."""
        if hasattr(conversation, "to_dict"):
            conversation_data = conversation.to_dict()
            messages = conversation_data.get("messages", [])
            preview_messages = conversation_data.get("preview_messages", [])
        elif isinstance(conversation, dict):
            messages = conversation.get("messages", [])
            preview_messages = conversation.get("preview_messages", [])
        elif isinstance(conversation, list):
            messages = conversation
            preview_messages = []
        else:
            messages = []
            preview_messages = []

        if not session_id:
            session_id = self.store._generate_session_id()

        if not title and messages:
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get("role") == "user":
                        title = msg.get("content", "")[:50]
                        break
                elif hasattr(msg, "role") and msg.role == MessageRole.USER:
                    title = msg.content[:50]
                    break

        conv_state_dict = (
            conversation_state.to_dict()
            if hasattr(conversation_state, "to_dict")
            else conversation_state
        )
        exec_state_dict = (
            execution_state.to_dict() if hasattr(execution_state, "to_dict") else execution_state
        )
        ledger_dict = (
            session_ledger.to_dict() if hasattr(session_ledger, "to_dict") else session_ledger
        )

        session = self.store.get_session(session_id)
        if not session:
            session = self.create_session(
                session_id=session_id,
                provider=provider,
                model=model,
                profile=profile,
            )
        else:
            session.messages.clear()

        session.title = title
        session.tags = tags or []
        session.conversation_state = conv_state_dict
        session.execution_state = exec_state_dict
        session.session_ledger = ledger_dict
        session.compaction_hierarchy = compaction_hierarchy

        for msg in messages:
            if isinstance(msg, dict):
                role = MessageRole(msg["role"])
                content = msg["content"]
                token_count = msg.get("token_count", 0)
                priority = MessagePriority(msg.get("priority", 50))
                tool_name = msg.get("tool_name")
                tool_call_id = msg.get("tool_call_id")
                metadata = msg.get("metadata", {})
                timestamp_str = msg.get("timestamp")
                timestamp = (
                    datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
                )

                conversation_msg = ConversationMessage(
                    role=role,
                    content=content,
                    token_count=token_count,
                    priority=priority,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    metadata=metadata,
                    timestamp=timestamp,
                )
            elif isinstance(msg, ConversationMessage):
                conversation_msg = msg
            else:
                continue

            if conversation_msg not in session.messages:
                session.messages.append(conversation_msg)

        session.preview_messages.clear()
        for preview in preview_messages:
            if not isinstance(preview, dict):
                continue

            metadata = dict(preview.get("metadata") or {})
            after_message_index = preview.get("after_message_index")
            if after_message_index is not None:
                metadata["_victor_preview_after_message_index"] = after_message_index
            timestamp_str = preview.get("timestamp")
            timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()

            session.preview_messages.append(
                ConversationMessage(
                    role=MessageRole(preview.get("role", "system")),
                    content=preview.get("content", ""),
                    token_count=preview.get("token_count", 0),
                    priority=MessagePriority(preview.get("priority", 50)),
                    tool_name=preview.get("tool_name"),
                    tool_call_id=preview.get("tool_call_id"),
                    metadata=metadata,
                    timestamp=timestamp,
                )
            )

        self.store._persist_session(session)
        logger.info(f"Saved session {session_id} with title: {title}")
        return session_id

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session with rich metadata."""
        session = self.store.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None

        conversation_messages = []
        preview_messages = []

        for msg in session.messages:
            if isinstance(msg, ConversationMessage):
                msg_dict = {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "token_count": msg.token_count,
                    "priority": msg.priority.value if msg.priority else None,
                    "tool_name": msg.tool_name,
                    "tool_call_id": msg.tool_call_id,
                    "metadata": msg.metadata if msg.metadata else {},
                }

                if msg.metadata and msg.metadata.get("is_preview"):
                    preview_messages.append(msg_dict)
                else:
                    conversation_messages.append(msg_dict)

        for msg in session.preview_messages:
            if isinstance(msg, ConversationMessage):
                metadata = dict(msg.metadata or {})
                after_message_index = metadata.pop("_victor_preview_after_message_index", None)
                msg_dict = {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "token_count": msg.token_count,
                    "priority": msg.priority.value if msg.priority else None,
                    "tool_name": msg.tool_name,
                    "tool_call_id": msg.tool_call_id,
                    "metadata": metadata,
                }
                if after_message_index is not None:
                    msg_dict["after_message_index"] = after_message_index
                preview_messages.append(msg_dict)

        title = session.title
        if not title and conversation_messages:
            for msg in conversation_messages:
                if msg["role"] == "user":
                    title = msg["content"][:50]
                    break

        if not title:
            title = "Untitled"

        session_data = {
            "metadata": {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.last_activity.isoformat(),
                "model": session.model or "unknown",
                "provider": session.provider or "unknown",
                "profile": session.profile or "default",
                "message_count": len(conversation_messages),
                "title": title,
                "tags": session.tags or [],
            },
            "conversation": {
                "messages": conversation_messages,
                "preview_messages": preview_messages,
            },
            "conversation_state": session.conversation_state,
            "tool_selection_stats": None,
            "execution_state": session.execution_state,
            "session_ledger": session.session_ledger,
            "compaction_hierarchy": session.compaction_hierarchy,
        }

        logger.info(f"Loaded session {session_id} from ConversationStore")
        return session_data

    def search_sessions(
        self,
        query: str,
        limit: int = 10,
        project_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search sessions by title or content."""
        try:
            lowered_query = f"%{query.lower()}%"
            results = []

            with self.store._connection() as conn:
                if project_path:
                    rows = conn.execute(
                        """
                        SELECT DISTINCT
                            s.session_id,
                            s.created_at,
                            s.last_activity,
                            s.model,
                            s.profile,
                            p.name AS provider,
                            s.metadata,
                            NULL as message_count
                        FROM sessions s
                        LEFT JOIN providers p ON s.provider_id = p.id
                        WHERE s.project_path = ?
                          AND (
                              LOWER(json_extract(s.metadata, '$.title')) LIKE LOWER(?)
                              OR LOWER(s.metadata) LIKE LOWER(?)
                          )
                        ORDER BY s.last_activity DESC
                        LIMIT ?
                        """,
                        (project_path, lowered_query, lowered_query, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT DISTINCT
                            s.session_id,
                            s.created_at,
                            s.last_activity,
                            s.model,
                            s.profile,
                            p.name AS provider,
                            s.metadata,
                            NULL as message_count
                        FROM sessions s
                        LEFT JOIN providers p ON s.provider_id = p.id
                        WHERE (
                            LOWER(json_extract(s.metadata, '$.title')) LIKE LOWER(?)
                            OR LOWER(s.metadata) LIKE LOWER(?)
                        )
                        ORDER BY s.last_activity DESC
                        LIMIT ?
                        """,
                        (lowered_query, lowered_query, limit),
                    ).fetchall()

                for row in rows:
                    session_dict = self.store._row_to_session_dict(row)
                    if row["session_id"] in self.store._sessions:
                        session_dict["message_count"] = len(
                            self.store._sessions[row["session_id"]].messages
                        )
                    results.append(session_dict)

                if len(results) >= limit:
                    return results[:limit]

                # FTS5's query parser interprets punctuation such as `/` as
                # syntax. Treat user input as a literal phrase instead.
                fts_query = f'"{query.replace(chr(34), chr(34) * 2)}"'
                found_ids = {r["session_id"] for r in results} if results else set()

                if project_path:
                    content_rows = conn.execute(
                        """
                        SELECT DISTINCT
                            s.session_id,
                            s.created_at,
                            s.last_activity,
                            s.model,
                            s.profile,
                            p.name AS provider,
                            s.metadata
                        FROM sessions s
                        LEFT JOIN providers p ON s.provider_id = p.id
                        INNER JOIN messages_fts fts ON s.session_id = fts.session_id
                        WHERE s.project_path = ?
                          AND messages_fts MATCH ?
                        ORDER BY s.last_activity DESC
                        LIMIT ?
                        """,
                        (project_path, fts_query, limit * 2),
                    ).fetchall()
                else:
                    content_rows = conn.execute(
                        """
                        SELECT DISTINCT
                            s.session_id,
                            s.created_at,
                            s.last_activity,
                            s.model,
                            s.profile,
                            p.name AS provider,
                            s.metadata
                        FROM sessions s
                        LEFT JOIN providers p ON s.provider_id = p.id
                        INNER JOIN messages_fts fts ON s.session_id = fts.session_id
                        WHERE messages_fts MATCH ?
                        ORDER BY s.last_activity DESC
                        LIMIT ?
                        """,
                        (fts_query, limit * 2),
                    ).fetchall()

                for row in content_rows:
                    if row["session_id"] in found_ids:
                        continue
                    if len(results) >= limit:
                        break
                    session_dict = self.store._row_to_session_dict(row)
                    results.append(session_dict)

                if len(results) < limit:
                    for session_id, session in list(self.store._sessions.items()):
                        if project_path and session.project_path != project_path:
                            continue
                        if session_id in found_ids:
                            continue
                        for msg in session.messages:
                            if lowered_query.replace("%", "") in msg.content.lower():
                                session_dict = {
                                    "session_id": session.session_id,
                                    "created_at": session.created_at.isoformat(),
                                    "updated_at": session.last_activity.isoformat(),
                                    "model": session.model or "unknown",
                                    "provider": session.provider or "unknown",
                                    "profile": session.profile or "default",
                                    "message_count": len(session.messages),
                                    "title": session.title or "Untitled",
                                    "tags": session.tags or [],
                                }
                                results.append(session_dict)
                                found_ids.add(session_id)
                                break
                        if len(results) >= limit:
                            break

            return results[:limit]

        except Exception as e:
            logger.error(f"Failed to search sessions: {e}")
            return []

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get a session by ID."""
        if session_id in self.store._sessions:
            return self.store._sessions[session_id]

        session = self.store._load_session(session_id)
        if session:
            self.store._sessions[session_id] = session

        return session

    def list_sessions(
        self,
        project_path: Optional[str] = None,
        limit: int = 10,
    ) -> List[ConversationSession]:
        """List recent sessions with JOINs to lookup tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            base_query = """
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
            """

            if project_path:
                rows = conn.execute(
                    f"""
                    {base_query}
                    WHERE s.project_path = ?
                    ORDER BY s.last_activity DESC
                    LIMIT ?
                    """,
                    (project_path, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    {base_query}
                    ORDER BY s.last_activity DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

            sessions = []
            for row in rows:
                session = self.store._session_from_row(row)
                sessions.append(session)

            return sessions

    def clear_session(self, session_id: str):
        """Clear all messages from a session."""
        session = self.get_session(session_id)
        if session:
            session.messages.clear()
            session.current_tokens = 0

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM messages WHERE session_id = ?",
                    (session_id,),
                )

            logger.info(f"Cleared session {session_id}")

    def delete_session(self, session_id: str):
        """Delete a session and all its messages."""
        if session_id in self.store._sessions:
            del self.store._sessions[session_id]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

        logger.info(f"Deleted session {session_id}")

    def cleanup_stale_sessions(
        self,
        max_age_days: int = 30,
        purge_test_models: bool = True,
        purge_empty: bool = True,
    ) -> Dict[str, int]:
        """Remove stale, test, and empty sessions from SQLite."""
        deleted = {"test_models": 0, "empty": 0, "stale": 0}

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")

                if purge_test_models:
                    placeholders = ",".join("?" * len(self.TEST_MODEL_NAMES))
                    cursor = conn.execute(
                        f"DELETE FROM sessions WHERE model IN ({placeholders})",
                        list(self.TEST_MODEL_NAMES),
                    )
                    deleted["test_models"] = cursor.rowcount

                if purge_empty:
                    cursor = conn.execute(
                        "DELETE FROM sessions WHERE session_id NOT IN "
                        "(SELECT DISTINCT session_id FROM messages)"
                    )
                    deleted["empty"] = cursor.rowcount

                if max_age_days > 0:
                    from datetime import timedelta

                    cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
                    cursor = conn.execute(
                        "DELETE FROM sessions WHERE last_activity < ?",
                        (cutoff,),
                    )
                    deleted["stale"] = cursor.rowcount

            total = sum(deleted.values())
            if total > 0:
                logger.info(
                    f"Cleaned up {total} sessions: "
                    f"{deleted['test_models']} test, "
                    f"{deleted['empty']} empty, "
                    f"{deleted['stale']} stale"
                )
                self.store._sessions.clear()

        except sqlite3.Error as e:
            logger.warning(f"Session cleanup failed: {e}")

        return deleted

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        session = self.get_session(session_id)
        if not session:
            return {}

        role_counts = {}
        for msg in session.messages:
            role_counts[msg.role.value] = role_counts.get(msg.role.value, 0) + 1

        return {
            "session_id": session_id,
            "message_count": len(session.messages),
            "total_tokens": session.current_tokens,
            "available_tokens": session.available_tokens,
            "utilization": session.current_tokens / session.max_tokens,
            "role_distribution": role_counts,
            "tool_usage_count": session.tool_usage_count,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "duration_seconds": (session.last_activity - session.created_at).total_seconds(),
        }
