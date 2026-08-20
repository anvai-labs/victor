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

"""Message operations and pruning logic for ConversationStore."""

from __future__ import annotations
import logging
import sqlite3
import uuid
import json
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.conversation.store import ConversationStore

from victor.agent.conversation.types import (
    ConversationMessage,
    MESSAGE_SOURCE_METADATA_KEY,
    MessagePriority,
    MessageRole,
    MessageSource,
)
from victor.agent.conversation.session_model import ConversationSession
from victor.agent.runtime.context import AgentRuntimeContext
from victor.agent.services.context_service import ContextService, ContextServiceConfig
from victor.core.json_utils import json_dumps, json_loads
from victor.tools.core_tool_aliases import canonicalize_core_tool_name

logger = logging.getLogger(__name__)

# Source → priority overrides (None = fall through to role-based logic).
_SOURCE_PRIORITY_MAP: dict = {
    MessageSource.USER_TYPED: MessagePriority.CRITICAL,
    MessageSource.AGENT_NUDGE: MessagePriority.EPHEMERAL,
    MessageSource.AGENT_CONTINUATION: MessagePriority.EPHEMERAL,
    MessageSource.AGENT_GUIDANCE: MessagePriority.LOW,
    MessageSource.COMPACTION_SUMMARY: MessagePriority.MEDIUM,
}


class ConversationStoreMessages:
    """Manages message operations, persistence, context retrieval, and pruning."""

    def __init__(self, store: ConversationStore):
        """Initialize message manager.

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

    def _determine_priority(
        self,
        role: MessageRole,
        tool_name: Optional[str],
        source: Optional[MessageSource] = None,
    ) -> MessagePriority:
        """Determine message priority based on role, tool context, and source origin."""
        # Source-based overrides take precedence over role-based logic.
        if source is not None:
            priority = _SOURCE_PRIORITY_MAP.get(source)
            if priority is not None:
                return priority

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
        """Shared implementation for adding a message."""
        session = self.store._get_or_create_session(session_id)

        # Auto-determine priority based on role and source (when available)
        if priority is None:
            source_raw = (metadata or {}).get(MESSAGE_SOURCE_METADATA_KEY)
            source: Optional[MessageSource] = None
            if source_raw:
                try:
                    source = MessageSource(source_raw)
                except ValueError:
                    pass
            priority = self._determine_priority(role, tool_name, source=source)

        # Estimate token count
        token_count = self.store._estimate_tokens(content)
        trace_metadata = self.store._build_trace_metadata(
            role=role,
            content=content,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=metadata,
            tool_calls=tool_calls,
        )
        merged_metadata = dict(metadata or {})
        merged_metadata.update(trace_metadata)

        message = ConversationMessage(
            id=self.store._generate_message_id(),
            role=role,
            content=content,
            timestamp=datetime.now(),
            token_count=token_count,
            priority=priority,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            metadata=merged_metadata,
            tool_calls=tool_calls,
        )

        session.messages.append(message)
        session.current_tokens += token_count
        session.last_activity = datetime.now()

        # Track tool usage
        if role in (MessageRole.TOOL_CALL, MessageRole.TOOL):
            session.tool_usage_count += 1

        # Check if pruning is needed
        if session.current_tokens > (session.max_tokens - session.reserved_tokens):
            self._prune_context(session)

        return message

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
        """Add a message to the conversation."""
        # Call shared implementation
        message = self._add_message_impl(
            session_id,
            role,
            content,
            priority,
            tool_name,
            tool_call_id,
            metadata,
            tool_calls,
        )

        # Persist (sync SQLite I/O)
        self._persist_message(session_id, message)
        self._update_session_activity(session_id)

        logger.debug(f"Added {role.value} message to {session_id}. Tokens: {message.token_count}")

        return message

    async def add_message_async(
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
        """Async variant of add_message with serialized writes."""
        # Call shared implementation
        message = self._add_message_impl(
            session_id,
            role,
            content,
            priority,
            tool_name,
            tool_call_id,
            metadata,
            tool_calls,
        )

        # CRITICAL: Serialize async writes with asyncio.Lock
        async with self.store._write_lock_async:
            # Persist (async SQLite I/O - offloaded to thread pool)
            await asyncio.to_thread(self._persist_message, session_id, message)
            await asyncio.to_thread(self._update_session_activity, session_id)

        session = self.store._sessions.get(session_id)
        total_tokens = session.current_tokens if session else 0
        logger.debug(
            "Added %s message to %s (async). Tokens: %d, Total: %d",
            role.value,
            session_id,
            message.token_count,
            total_tokens,
        )

        return message

    def add_system_message(
        self,
        session_id: str,
        content: str,
    ) -> ConversationMessage:
        """Add a system message with CRITICAL priority."""
        return self.add_message(
            session_id,
            MessageRole.SYSTEM,
            content,
            priority=MessagePriority.CRITICAL,
        )

    def get_context_messages(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
        include_system: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get messages formatted for the provider, respecting token limits."""
        session = self.store.get_session(session_id)
        if not session:
            return []

        max_tokens = max_tokens or (session.max_tokens - session.reserved_tokens)

        # Score and select messages
        scored_messages = self._score_messages(session.messages)

        # Select messages within token budget
        selected = []
        token_budget = max_tokens

        for msg, _score in scored_messages:
            if msg.token_count <= token_budget:
                selected.append(msg)
                token_budget -= msg.token_count

        # Sort by timestamp for coherent conversation
        selected.sort(key=lambda m: m.timestamp)

        # Filter system messages if requested
        if not include_system:
            selected = [m for m in selected if m.role != MessageRole.SYSTEM]

        logger.debug(
            f"Selected {len(selected)} messages for context. "
            f"Tokens used: {max_tokens - token_budget}/{max_tokens}"
        )

        return [msg.to_provider_format() for msg in selected]

    def get_messages_for_agent(
        self,
        session_id: str,
        agent_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[ConversationMessage]:
        """Return messages scoped to one agent inside a session."""
        query = """
            SELECT * FROM messages
            WHERE session_id = ? AND agent_id = ?
            ORDER BY timestamp
        """
        params: tuple[Any, ...]
        if limit is not None:
            query += " LIMIT ?"
            params = (session_id, agent_id, int(limit))
        else:
            params = (session_id, agent_id)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [self.store._message_from_row(row) for row in rows]

    def load_agent_context_service(
        self,
        runtime_context: AgentRuntimeContext,
        *,
        max_tokens: Optional[int] = None,
    ) -> ContextService:
        """Hydrate a ContextService from persisted messages for one agent runtime."""
        service = ContextService(
            ContextServiceConfig(max_tokens=max_tokens or self.max_context_tokens)
        )
        for message in self.get_messages_for_agent(
            runtime_context.session_id,
            runtime_context.agent_id,
        ):
            payload: Dict[str, Any] = {
                "role": message.role.value,
                "content": message.content,
            }
            if message.metadata:
                payload["metadata"] = dict(message.metadata)
            service.add_message(payload)
        return service

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
        """Persist a per-agent compaction audit event."""
        metadata = dict(metadata or {})
        event_id = f"compact_{uuid.uuid4().hex[:12]}"
        parent_session_id = metadata.get("parent_session_id")
        team_id = metadata.get("team_id")
        member_id = metadata.get("member_id")
        plan_id = metadata.get("plan_id")
        plan_step_id = metadata.get("plan_step_id")

        def _write(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO compaction_events (
                    id, session_id, agent_id, parent_session_id, team_id, member_id,
                    plan_id, plan_step_id, strategy, messages_removed, tokens_freed,
                    summary, created_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    session_id,
                    agent_id,
                    parent_session_id,
                    team_id,
                    member_id,
                    plan_id,
                    plan_step_id,
                    strategy,
                    int(messages_removed),
                    int(tokens_freed),
                    summary,
                    datetime.now(timezone.utc).isoformat(),
                    json_dumps(self.store._sanitize_metadata_for_json(metadata)),
                ),
            )

        self.store._with_locked_write_retry(_write, operation="record_compaction_event")
        return event_id

    def get_recent_messages(
        self,
        session_id: str,
        count: int = 10,
    ) -> List[ConversationMessage]:
        """Get the most recent messages."""
        session = self.store.get_session(session_id)
        if not session:
            return []

        return session.messages[-count:]

    def _persist_message(self, session_id: str, message: ConversationMessage):
        """Persist message to database."""
        content = message.content
        # Truncate large tool outputs for storage
        if len(content) > self.store._TOOL_OUTPUT_STORE_LIMIT and (
            message.role in (MessageRole.TOOL_CALL, MessageRole.TOOL)
            or (message.role == MessageRole.USER and content.startswith("<TOOL_OUTPUT"))
        ):
            content = (
                content[: self.store._TOOL_OUTPUT_STORE_LIMIT]
                + f"\n\n[... truncated from {len(message.content)} chars "
                f"for storage]"
            )

        # Merge tool_calls into metadata for persistence so assistant
        # messages that requested tool calls can be fully reconstructed.
        meta = dict(message.metadata) if message.metadata else {}
        if message.tool_calls:
            meta["tool_calls"] = message.tool_calls

        # Sanitize metadata for JSON serialization
        meta = self.store._sanitize_metadata_for_json(meta)
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

        self.store._with_locked_write_retry(_write, operation="persist_message")

    def _update_session_activity(self, session_id: str):
        """Update session last activity timestamp."""

        def _write(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )

        self.store._with_locked_write_retry(_write, operation="update_session_activity")

    def _score_messages(
        self,
        messages: List[ConversationMessage],
    ) -> List[tuple[ConversationMessage, float]]:
        """Score messages for context selection."""
        if not messages:
            return []

        from victor.agent.conversation.scoring import score_messages, STORE_WEIGHTS
        from victor.agent.conversation.types import (
            ConversationMessage as CanonicalMessage,
        )

        # Convert store messages to canonical type for scoring
        canonical_msgs = [
            CanonicalMessage(
                role=msg.role.value,
                content=msg.content,
                id=msg.id,
                timestamp=msg.timestamp,
                token_count=msg.token_count,
                priority=msg.priority,
                metadata=msg.metadata,
                tool_name=msg.tool_name,
                tool_call_id=msg.tool_call_id,
            )
            for msg in messages
        ]

        scored = score_messages(canonical_msgs, weights=STORE_WEIGHTS)

        # Map back to original store messages
        canonical_to_store = {id(cm): msg for cm, msg in zip(canonical_msgs, messages)}
        return [(canonical_to_store[id(cm)], s) for cm, s in scored]

    def _prune_context(self, session: ConversationSession):
        """Prune conversation context to fit within token limits."""
        target_tokens = int((session.max_tokens - session.reserved_tokens) * 0.8)

        logger.info(
            f"Pruning session {session.session_id}. "
            f"Current: {session.current_tokens}, Target: {target_tokens}"
        )

        # Separate by priority
        critical = [m for m in session.messages if m.priority == MessagePriority.CRITICAL]
        others = [m for m in session.messages if m.priority != MessagePriority.CRITICAL]

        # Sort others by score
        scored_others = self._score_messages(others)

        # Select within budget
        kept = list(critical)
        kept_ids = {m.id for m in kept}
        current_tokens = sum(m.token_count for m in kept)

        for msg, _ in scored_others:
            if current_tokens + msg.token_count <= target_tokens:
                kept.append(msg)
                kept_ids.add(msg.id)
                current_tokens += msg.token_count

        # Identify pruned messages and delete from SQLite
        pruned_ids = [m.id for m in session.messages if m.id not in kept_ids]
        pruned_count = len(pruned_ids)

        if pruned_ids:
            self._delete_messages_from_db(session.session_id, pruned_ids)

        # Update session
        session.messages = sorted(kept, key=lambda m: m.timestamp)
        session.current_tokens = current_tokens

        logger.info(
            f"Pruned {pruned_count} messages (deleted from DB). "
            f"Remaining: {len(session.messages)}, Tokens: {current_tokens}"
        )

    def _delete_messages_from_db(self, session_id: str, message_ids: List[str]) -> None:
        """Delete pruned messages from SQLite in batches."""
        batch_size = 500
        try:
            with sqlite3.connect(self.db_path) as conn:
                for i in range(0, len(message_ids), batch_size):
                    batch = message_ids[i : i + batch_size]
                    placeholders = ",".join("?" * len(batch))
                    conn.execute(
                        f"DELETE FROM messages WHERE id IN ({placeholders})",
                        batch,
                    )
            logger.debug(
                f"Deleted {len(message_ids)} pruned messages from DB for session {session_id}"
            )
        except sqlite3.Error as e:
            logger.warning(f"Failed to delete pruned messages from DB: {e}")
