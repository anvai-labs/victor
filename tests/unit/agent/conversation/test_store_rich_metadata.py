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

"""Unit tests for ConversationStore rich metadata features."""

import json
import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
import sqlite3

from victor.agent.conversation.migrations import apply_migration_0_3_0
from victor.agent.conversation.store import ConversationStore
from victor.agent.conversation.types import (
    ConversationMessage,
    MessageRole,
    MessagePriority,
)


@pytest.fixture
def temp_store():
    """Create a temporary ConversationStore for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_project.db"

    try:
        store = ConversationStore(db_path=db_path)
        yield store
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestConversationSessionRichMetadata:
    """Test ConversationSession dataclass with rich metadata fields."""

    def test_sqlite_locked_write_retries_once(self, temp_store, monkeypatch):
        """Transient SQLite lock errors should retry before surfacing."""
        monkeypatch.setattr("victor.agent.conversation.store.time.sleep", lambda _delay: None)
        attempts = 0

        def write_fn(conn):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise sqlite3.OperationalError("database is locked")
            conn.execute("SELECT 1")

        temp_store._with_locked_write_retry(write_fn, operation="test")

        assert attempts == 2

    def test_conversation_session_has_title_field(self, temp_store):
        """Test that ConversationSession has title field."""
        session = temp_store.create_session(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
        )

        # Should have title field (default None)
        assert hasattr(session, "title")
        assert session.title is None

        # Can set title
        session.title = "Test Session"
        assert session.title == "Test Session"

    def test_conversation_session_has_tags_field(self, temp_store):
        """Test that ConversationSession has tags field."""
        session = temp_store.create_session(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
        )

        # Should have tags field (default empty list)
        assert hasattr(session, "tags")
        assert session.tags == []

        # Can add tags
        session.tags = ["coding", "debugging"]
        assert session.tags == ["coding", "debugging"]

    def test_conversation_session_has_state_persistence_fields(self, temp_store):
        """Test that ConversationSession has state persistence fields."""
        session = temp_store.create_session(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
        )

        # Should have state fields
        assert hasattr(session, "conversation_state")
        assert hasattr(session, "execution_state")
        assert hasattr(session, "session_ledger")
        assert hasattr(session, "compaction_hierarchy")

        # All default to None
        assert session.conversation_state is None
        assert session.execution_state is None
        assert session.session_ledger is None
        assert session.compaction_hierarchy is None

        # Can set state dicts
        session.conversation_state = {"stage": "IN_PROGRESS"}
        session.execution_state = {"tool_calls": 5}
        assert session.conversation_state == {"stage": "IN_PROGRESS"}
        assert session.execution_state == {"tool_calls": 5}

    def test_conversation_session_has_preview_messages_field(self, temp_store):
        """Test that ConversationSession has preview_messages field."""
        session = temp_store.create_session(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
        )

        # Should have preview_messages field (default empty list)
        assert hasattr(session, "preview_messages")
        assert session.preview_messages == []

        # Can add preview messages
        preview_msg = ConversationMessage(
            role=MessageRole.ASSISTANT,
            content="Preview: file.py",
            metadata={"is_preview": True, "preview_path": "file.py"},
        )
        session.preview_messages.append(preview_msg)
        assert len(session.preview_messages) == 1

    def test_conversation_session_to_dict_includes_rich_metadata(self, temp_store):
        """Test that to_dict() includes rich metadata fields."""
        session = temp_store.create_session(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
        )
        session.title = "Test Title"
        session.tags = ["tag1", "tag2"]
        session.conversation_state = {"stage": "DONE"}
        session.execution_state = {"calls": 10}

        session_dict = session.to_dict()

        # Should include rich metadata
        assert "title" in session_dict
        assert "tags" in session_dict
        assert "conversation_state" in session_dict
        assert "execution_state" in session_dict
        assert "session_ledger" in session_dict
        assert "compaction_hierarchy" in session_dict

        assert session_dict["title"] == "Test Title"
        assert session_dict["tags"] == ["tag1", "tag2"]
        assert session_dict["conversation_state"] == {"stage": "DONE"}
        assert session_dict["execution_state"] == {"calls": 10}


class TestSaveSession:
    """Test ConversationStore.save_session() method."""

    def test_save_session_creates_new_session(self, temp_store):
        """Test that save_session creates a new session."""
        messages = [
            {
                "role": "user",
                "content": "Hello, help me debug this code",
            },
            {
                "role": "assistant",
                "content": "I'll help you debug the code",
            },
        ]

        session_id = temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            profile="default",
        )

        assert session_id is not None
        assert isinstance(session_id, str)

        # Verify session was created
        session = temp_store.get_session(session_id)
        assert session is not None
        assert session.provider == "anthropic"
        assert session.model == "claude-3-5-sonnet-20241022"

    def test_save_session_with_custom_title(self, temp_store):
        """Test that save_session saves custom title."""
        messages = [{"role": "user", "content": "Test message"}]

        session_id = temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            title="Custom Session Title",
        )

        session = temp_store.get_session(session_id)
        assert session.title == "Custom Session Title"

    def test_save_session_generates_title_from_first_user_message(self, temp_store):
        """Test that save_session generates title from first user message."""
        messages = [
            {"role": "user", "content": "Help me implement a binary search tree"},
            {"role": "assistant", "content": "Sure, here's how..."},
        ]

        session_id = temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
        )

        session = temp_store.get_session(session_id)
        # Title should be the full message (37 chars < 50, so not truncated)
        assert session.title == "Help me implement a binary search tree"

    def test_save_session_with_tags(self, temp_store):
        """Test that save_session saves tags."""
        messages = [{"role": "user", "content": "Test"}]

        session_id = temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            tags=["algorithms", "data-structures"],
        )

        session = temp_store.get_session(session_id)
        assert session.tags == ["algorithms", "data-structures"]

    def test_save_session_with_conversation_state(self, temp_store):
        """Test that save_session saves conversation state."""
        messages = [{"role": "user", "content": "Test"}]

        conv_state = {
            "current_stage": "REQUIREMENT_GATHERING",
            "completed_stages": [],
            "last_updated": "2025-01-01T00:00:00",
        }

        session_id = temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            conversation_state=conv_state,
        )

        session = temp_store.get_session(session_id)
        assert session.conversation_state == conv_state

    def test_save_session_with_execution_state(self, temp_store):
        """Test that save_session saves execution state."""
        messages = [{"role": "user", "content": "Test"}]

        exec_state = {
            "tool_calls_used": 5,
            "total_tokens": 1000,
            "last_tool": "code_search",
        }

        session_id = temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            execution_state=exec_state,
        )

        session = temp_store.get_session(session_id)
        assert session.execution_state == exec_state

    def test_save_session_with_session_ledger(self, temp_store):
        """Test that save_session saves session ledger."""
        messages = [{"role": "user", "content": "Test"}]

        ledger = {
            "entries": [
                {
                    "timestamp": "2025-01-01T00:00:00",
                    "event": "tool_call",
                    "tool": "read_file",
                }
            ],
            "high_signals": [],
        }

        session_id = temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            session_ledger=ledger,
        )

        session = temp_store.get_session(session_id)
        assert session.session_ledger == ledger

    def test_save_session_with_compaction_hierarchy(self, temp_store):
        """Test that save_session saves compaction hierarchy."""
        messages = [{"role": "user", "content": "Test"}]

        hierarchy = {
            "active_context": "summary_1",
            "contexts": {
                "summary_1": {"content": "Summary of messages 1-10"},
                "summary_2": {"content": "Summary of messages 11-20"},
            },
        }

        session_id = temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            compaction_hierarchy=hierarchy,
        )

        session = temp_store.get_session(session_id)
        assert session.compaction_hierarchy == hierarchy

    def test_save_session_updates_existing_session(self, temp_store):
        """Test that save_session updates existing session."""
        messages = [{"role": "user", "content": "Initial message"}]

        # Create initial session
        session_id = temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            title="Initial Title",
        )

        # Update session with more messages
        updated_messages = [
            {"role": "user", "content": "Initial message"},
            {"role": "assistant", "content": "Response"},
        ]

        temp_store.save_session(
            conversation={"messages": updated_messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            session_id=session_id,
            title="Updated Title",
        )

        # Verify updates
        session = temp_store.get_session(session_id)
        assert session.title == "Updated Title"
        assert len(session.messages) == 2

    def test_save_session_with_conversation_message_objects(self, temp_store):
        """Test that save_session handles ConversationMessage objects."""
        messages = [
            ConversationMessage(
                role=MessageRole.USER,
                content="Test message",
                timestamp=datetime.now(),
                priority=MessagePriority.MEDIUM,
            )
        ]

        session_id = temp_store.save_session(
            conversation=messages,
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
        )

        session = temp_store.get_session(session_id)
        assert len(session.messages) == 1
        assert session.messages[0].role == MessageRole.USER
        assert session.messages[0].content == "Test message"


class TestLoadSession:
    """Test ConversationStore.load_session() method."""

    def test_load_session_returns_correct_structure(self, temp_store):
        """Test that load_session returns dict with correct structure."""
        messages = [{"role": "user", "content": "Test"}]

        session_id = temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            title="Test Title",
            tags=["test"],
        )

        session_data = temp_store.load_session(session_id)

        # Should be a dict
        assert isinstance(session_data, dict)

        # Should have top-level keys
        assert "metadata" in session_data
        assert "conversation" in session_data
        assert "conversation_state" in session_data
        assert "tool_selection_stats" in session_data
        assert "execution_state" in session_data
        assert "session_ledger" in session_data
        assert "compaction_hierarchy" in session_data

    def test_load_session_metadata_structure(self, temp_store):
        """Test that load_session metadata has correct fields."""
        messages = [{"role": "user", "content": "Test"}]

        session_id = temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            profile="fast",
            title="Test Title",
            tags=["tag1", "tag2"],
        )

        session_data = temp_store.load_session(session_id)
        metadata = session_data["metadata"]

        # Check required fields
        assert metadata["session_id"] == session_id
        assert "created_at" in metadata
        assert "updated_at" in metadata
        assert metadata["model"] == "claude-3-5-sonnet-20241022"
        assert metadata["provider"] == "anthropic"
        assert metadata["profile"] == "fast"
        assert metadata["title"] == "Test Title"
        assert metadata["tags"] == ["tag1", "tag2"]
        assert "message_count" in metadata

    def test_load_session_conversation_structure(self, temp_store):
        """Test that load_session conversation has correct structure."""
        messages = [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant response"},
        ]

        session_id = temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
        )

        session_data = temp_store.load_session(session_id)
        conversation = session_data["conversation"]

        # Should have messages and preview_messages
        assert "messages" in conversation
        assert "preview_messages" in conversation

        # Messages should be preserved
        assert len(conversation["messages"]) == 2
        assert conversation["messages"][0]["role"] == "user"
        assert conversation["messages"][0]["content"] == "User message"

    def test_load_session_separates_preview_messages(self, temp_store):
        """Test that load_session separates preview messages from regular messages."""
        # Create session with preview messages
        session = temp_store.create_session(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
        )

        # Add regular message
        temp_store.add_message(
            session.session_id,
            MessageRole.USER,
            "Regular message",
        )

        # Add preview message
        preview_msg = ConversationMessage(
            role=MessageRole.ASSISTANT,
            content="Preview: code",
            metadata={"is_preview": True, "preview_path": "test.py"},
        )
        session.preview_messages.append(preview_msg)
        temp_store._persist_session(session)

        # Load and verify separation
        session_data = temp_store.load_session(session.session_id)
        conversation = session_data["conversation"]

        # Regular messages should be in messages list
        assert len(conversation["messages"]) == 1
        assert conversation["messages"][0]["content"] == "Regular message"

        # Preview messages should be in preview_messages list
        assert len(conversation["preview_messages"]) == 1
        assert conversation["preview_messages"][0]["content"] == "Preview: code"
        assert conversation["preview_messages"][0]["metadata"]["is_preview"] is True

    def test_load_session_includes_rich_state_fields(self, temp_store):
        """Test that load_session includes rich state fields."""
        conv_state = {"stage": "REQUIREMENT_GATHERING"}
        exec_state = {"tool_calls": 5}
        ledger = {"entries": [{"event": "start"}]}
        hierarchy = {"active": "ctx1"}

        messages = [{"role": "user", "content": "Test"}]

        session_id = temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            conversation_state=conv_state,
            execution_state=exec_state,
            session_ledger=ledger,
            compaction_hierarchy=hierarchy,
        )

        session_data = temp_store.load_session(session_id)

        # Should include all state fields
        assert session_data["conversation_state"] == conv_state
        assert session_data["execution_state"] == exec_state
        assert session_data["session_ledger"] == ledger
        assert session_data["compaction_hierarchy"] == hierarchy


def test_load_session_returns_none_for_nonexistent_session(temp_store):
    """Test that load_session returns None for nonexistent session."""
    session_data = temp_store.load_session("nonexistent_session_id")
    assert session_data is None


def test_messages_schema_has_agent_lineage_columns(temp_store):
    """Messages table should expose first-class multi-agent lineage columns."""
    with temp_store._get_connection() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
        compaction_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(compaction_events)").fetchall()
        }

    assert "agent_id" in columns
    assert "parent_session_id" in columns
    assert "team_id" in columns
    assert "member_id" in columns
    assert "plan_id" in columns
    assert "plan_step_id" in columns
    assert {
        "session_id",
        "agent_id",
        "strategy",
        "messages_removed",
        "tokens_freed",
    }.issubset(compaction_columns)


def test_migration_backfills_agent_lineage_columns_from_metadata(tmp_path):
    """Legacy messages should become queryable by agent/team after migration."""
    db_path = tmp_path / "legacy_project.db"
    metadata = {
        "agent_id": "agent_1",
        "parent_session_id": "parent_session",
        "team_id": "team_1",
        "member_id": "member_1",
        "plan_id": "plan_1",
        "plan_step_id": "step_1",
    }

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                last_activity TIMESTAMP NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                token_count INTEGER NOT NULL DEFAULT 0,
                priority INTEGER NOT NULL DEFAULT 0,
                tool_name TEXT,
                tool_call_id TEXT,
                metadata TEXT
            )
        """)
        conn.execute(
            """
            INSERT INTO messages (
                id, session_id, role, content, timestamp, token_count, priority, metadata
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "msg_1",
                "session_1",
                "user",
                "legacy child message",
                "2026-05-12T13:00:00",
                4,
                0,
                json.dumps(metadata),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    assert apply_migration_0_3_0(str(db_path)) is True
    assert apply_migration_0_3_0(str(db_path)) is False

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM messages WHERE id = ?", ("msg_1",)).fetchone()
        indexes = {
            index_row["name"]
            for index_row in conn.execute("PRAGMA index_list(messages)").fetchall()
        }
    finally:
        conn.close()

    assert row["agent_id"] == "agent_1"
    assert row["parent_session_id"] == "parent_session"
    assert row["team_id"] == "team_1"
    assert row["member_id"] == "member_1"
    assert row["plan_id"] == "plan_1"
    assert row["plan_step_id"] == "step_1"
    assert "idx_messages_session_agent_time" in indexes


def test_add_message_persists_agent_lineage_and_filters_by_agent(temp_store):
    """ConversationStore should isolate messages by session+agent for subagent context."""
    session = temp_store.create_session(project_path="/tmp/project")
    temp_store.add_message(
        session.session_id,
        MessageRole.USER,
        "root message",
        metadata={"agent_id": "root_agent", "team_id": "team_1"},
    )
    temp_store.add_message(
        session.session_id,
        MessageRole.USER,
        "child message",
        metadata={
            "agent_id": "child_agent",
            "parent_session_id": session.session_id,
            "team_id": "team_1",
            "member_id": "member_1",
            "plan_id": "plan_1",
            "plan_step_id": "1",
        },
    )

    child_messages = temp_store.get_messages_for_agent(session.session_id, "child_agent")
    root_messages = temp_store.get_messages_for_agent(session.session_id, "root_agent")

    assert [message.content for message in child_messages] == ["child message"]
    assert [message.content for message in root_messages] == ["root message"]
    assert child_messages[0].metadata["member_id"] == "member_1"
    assert child_messages[0].metadata["plan_step_id"] == "1"


def test_load_agent_context_service_uses_only_agent_messages(temp_store):
    """Agent context hydration should not mix messages from sibling agents."""
    from victor.agent.runtime.context import AgentRuntimeContext

    session = temp_store.create_session(project_path="/tmp/project")
    temp_store.add_message(
        session.session_id,
        MessageRole.USER,
        "root message",
        metadata={"agent_id": "root_agent"},
    )
    temp_store.add_message(
        session.session_id,
        MessageRole.USER,
        "child message",
        metadata={"agent_id": "child_agent"},
    )

    service = temp_store.load_agent_context_service(
        AgentRuntimeContext(
            agent_id="child_agent",
            display_name="Child",
            role="researcher",
            session_id=session.session_id,
        )
    )

    messages = service.get_messages()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "child message"
    assert messages[0]["metadata"]["agent_id"] == "child_agent"


def test_agent_context_service_restartability_from_project_db(tmp_path):
    """Agent context should restore from SQLite using session_id+agent_id after restart."""
    from victor.agent.runtime.context import AgentRuntimeContext

    db_path = tmp_path / "project.db"
    store = ConversationStore(db_path=db_path)
    session = store.create_session(project_path="/tmp/project")
    store.add_message(
        session.session_id,
        MessageRole.USER,
        "child persisted message",
        metadata={
            "agent_id": "child_agent",
            "parent_session_id": "session_root",
            "team_id": "team_1",
            "member_id": "member_1",
        },
    )
    store.add_message(
        session.session_id,
        MessageRole.USER,
        "sibling persisted message",
        metadata={"agent_id": "sibling_agent", "team_id": "team_1"},
    )

    restarted_store = ConversationStore(db_path=db_path)
    service = restarted_store.load_agent_context_service(
        AgentRuntimeContext(
            agent_id="child_agent",
            display_name="Child",
            role="researcher",
            session_id=session.session_id,
            parent_session_id="session_root",
            team_id="team_1",
            member_id="member_1",
        )
    )

    messages = service.get_messages()
    assert [message["content"] for message in messages] == ["child persisted message"]
    assert messages[0]["metadata"]["agent_id"] == "child_agent"
    assert messages[0]["metadata"]["team_id"] == "team_1"


def test_record_compaction_event_persists_agent_lineage(temp_store):
    """Compaction events should be queryable by session+agent for restartability."""
    session = temp_store.create_session(project_path="/tmp/project")

    event_id = temp_store.record_compaction_event(
        session_id=session.session_id,
        agent_id="child_agent",
        strategy="tiered",
        messages_removed=4,
        tokens_freed=1200,
        summary="Compacted old child tool chatter",
        metadata={"team_id": "team_1", "member_id": "member_1", "plan_step_id": "1"},
    )

    with temp_store._get_connection() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM compaction_events WHERE id = ?",
            (event_id,),
        ).fetchone()

    assert row["session_id"] == session.session_id
    assert row["agent_id"] == "child_agent"
    assert row["team_id"] == "team_1"
    assert row["member_id"] == "member_1"
    assert row["plan_step_id"] == "1"
    assert row["messages_removed"] == 4
    assert row["tokens_freed"] == 1200


class TestSearchSessions:
    """Test ConversationStore.search_sessions() method."""

    def test_search_sessions_finds_by_title(self, temp_store):
        """Test that search_sessions finds sessions by title."""
        # Create sessions with different titles
        messages = [{"role": "user", "content": "Test"}]

        temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            title="Binary Search Implementation",
        )

        temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            title="Graph Algorithm",
        )

        # Search for "binary"
        results = temp_store.search_sessions("binary", limit=10)

        assert len(results) == 1
        assert "binary" in results[0]["title"].lower()

    def test_search_sessions_finds_by_message_content(self, temp_store):
        """Test that search_sessions finds sessions by message content."""
        # Create sessions with different content
        temp_store.save_session(
            conversation={
                "messages": [
                    {
                        "role": "user",
                        "content": "Help me implement a binary search tree",
                    }
                ]
            },
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
        )

        temp_store.save_session(
            conversation={
                "messages": [{"role": "user", "content": "Create a graph visualization"}]
            },
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
        )

        # Search for "binary"
        results = temp_store.search_sessions("binary", limit=10)

        assert len(results) == 1

    def test_search_sessions_respects_limit(self, temp_store):
        """Test that search_sessions respects limit parameter."""
        messages = [{"role": "user", "content": "Test message"}]

        # Create 5 sessions
        for i in range(5):
            temp_store.save_session(
                conversation={"messages": messages},
                model="claude-3-5-sonnet-20241022",
                provider="anthropic",
                title=f"Session {i}",
            )

        # Search with limit=3
        results = temp_store.search_sessions("session", limit=3)

        assert len(results) == 3

    def test_search_sessions_filters_by_project_path(self, temp_store):
        """Test that search_sessions works correctly (project filtering not yet implemented)."""
        messages = [{"role": "user", "content": "Test"}]

        # Create sessions in different projects
        temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            title="Project A Session",
        )

        # Create second store for different project
        temp_dir2 = tempfile.mkdtemp()
        try:
            db_path2 = Path(temp_dir2) / "test_project2.db"
            store2 = ConversationStore(db_path=db_path2)

            store2.save_session(
                conversation={"messages": messages},
                model="claude-3-5-sonnet-20241022",
                provider="anthropic",
                title="Project B Session",
            )

            # Search should find sessions (project filtering is a TODO feature)
            results = temp_store.search_sessions("session", limit=10)

            # Should find at least the first project's session
            assert len(results) >= 1
            assert any(r["title"] == "Project A Session" for r in results)

        finally:
            shutil.rmtree(temp_dir2, ignore_errors=True)

    def test_search_sessions_returns_empty_list_for_no_matches(self, temp_store):
        """Test that search_sessions returns empty list when no matches."""
        messages = [{"role": "user", "content": "Test"}]

        temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            title="Completely Different Topic",
        )

        results = temp_store.search_sessions("nonexistent_term_xyz", limit=10)

        assert results == []

    def test_search_sessions_case_insensitive(self, temp_store):
        """Test that search_sessions is case insensitive."""
        messages = [{"role": "user", "content": "Help with Binary Search"}]

        temp_store.save_session(
            conversation={"messages": messages},
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
        )

        # Search with different cases
        results_lower = temp_store.search_sessions("binary", limit=10)
        results_upper = temp_store.search_sessions("BINARY", limit=10)
        results_mixed = temp_store.search_sessions("BiNaRy", limit=10)

        assert len(results_lower) == 1
        assert len(results_upper) == 1
        assert len(results_mixed) == 1
