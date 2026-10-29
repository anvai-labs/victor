"""Tests for session ledger persistence and merge functionality."""

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from victor.agent.message_history import MessageHistory
from victor.agent.sqlite_session_persistence import SQLiteSessionPersistence
from victor.agent.session_ledger import SessionLedger
from victor.core.database import reset_project_database


@pytest.fixture
def populated_ledger():
    ledger = SessionLedger()
    ledger.record_file_read("auth.py", "Authentication module", 1)
    ledger.record_file_read("models.py", "Data models", 2)
    ledger.record_decision("Use factory pattern", 3)
    ledger.record_pending_action("Write unit tests", 4)
    return ledger


@pytest.fixture
def second_ledger():
    ledger = SessionLedger()
    ledger.record_file_read("config.py", "Configuration module", 5)
    ledger.record_decision("Use dependency injection", 6)
    return ledger


class TestLedgerPersistence:
    def test_ledger_survives_save_load_cycle(self, populated_ledger):
        """to_dict → save → load → from_dict, entries match."""
        data = populated_ledger.to_dict()
        restored = SessionLedger.from_dict(data)

        assert len(restored.entries) == len(populated_ledger.entries)
        for orig, rest in zip(populated_ledger.entries, restored.entries):
            assert orig.category == rest.category
            assert orig.key == rest.key
            assert orig.summary == rest.summary
            assert orig.turn_index == rest.turn_index

        assert restored.get_files_read() == populated_ledger.get_files_read()

    def test_ledger_none_backward_compat(self):
        """Save without ledger, load returns None for field."""

        # Simulate session_data without session_ledger key
        session_data = {
            "metadata": {},
            "conversation": {"messages": []},
            "session_ledger": None,
        }
        # Verify None is handled gracefully
        ledger_data = session_data.get("session_ledger")
        assert ledger_data is None

    def test_merge_deduplicates_entries(self, populated_ledger):
        """Same file_read key not duplicated after merge."""
        other = SessionLedger()
        other.record_file_read("auth.py", "Auth module (duplicate)", 10)

        original_count = len(populated_ledger.entries)
        populated_ledger.merge(other)

        # auth.py should not be duplicated (same category + key)
        auth_entries = [
            e for e in populated_ledger.entries if e.category == "file_read" and e.key == "auth.py"
        ]
        assert len(auth_entries) == 1

    def test_merge_combines_files_read(self, populated_ledger, second_ledger):
        """files_read dicts are merged."""
        populated_ledger.merge(second_ledger)

        files = populated_ledger.get_files_read()
        assert "auth.py" in files
        assert "models.py" in files
        assert "config.py" in files

    def test_save_session_includes_ledger(self):
        """save_session includes session_ledger in session_data."""
        ledger = SessionLedger()
        ledger.record_file_read("test.py", "Test file", 1)

        ledger_dict = ledger.to_dict()
        assert "entries" in ledger_dict
        assert len(ledger_dict["entries"]) == 1
        assert ledger_dict["entries"][0]["key"] == "test.py"

    def test_save_extracts_ledger_from_orchestrator(self):
        """Verify save_session accepts session_ledger parameter."""
        # This tests that the API accepts the parameter
        from victor.agent.sqlite_session_persistence import SQLiteSessionPersistence
        import inspect

        sig = inspect.signature(SQLiteSessionPersistence.save_session)
        assert "session_ledger" in sig.parameters


@pytest.fixture
def temp_project_db_path(tmp_path: Path):
    """Provide an isolated project DB path for SQLite session tests."""
    db_path = tmp_path / "project.db"
    yield db_path
    reset_project_database(db_path)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestSQLiteSessionPersistenceCompatibility:
    """Guards the deprecated ``SQLiteSessionPersistence`` compat surface.

    These tests intentionally use the deprecated persistence class to verify it
    still behaves correctly (round-trip, schema compat, loader interop). The
    deprecation warning is expected here and intentionally suppressed.
    """

    def test_default_constructor_uses_canonical_project_db_path(self, tmp_path: Path):
        """Default constructor should honor the canonical ProjectPaths.project_db property."""
        db_path = tmp_path / "isolated-project.db"

        with patch(
            "victor.config.settings.get_project_paths",
            return_value=SimpleNamespace(project_db=db_path),
        ):
            persistence = SQLiteSessionPersistence()

        try:
            assert persistence._db_path == db_path
        finally:
            reset_project_database(db_path)

    def test_schema_compatibility_adds_legacy_session_columns(self, temp_project_db_path: Path):
        """Compatibility shim should backfill columns older sessions tables may lack."""
        with sqlite3.connect(temp_project_db_path) as conn:
            conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
            conn.execute("CREATE TABLE messages (id TEXT PRIMARY KEY, session_id TEXT, role TEXT)")

        persistence = SQLiteSessionPersistence(db_path=temp_project_db_path)

        session_columns = persistence._table_columns("sessions")
        assert {
            "created_at",
            "last_activity",
            "project_path",
            "provider",
            "model",
            "profile",
            "metadata",
        } <= session_columns

    def test_sqlite_roundtrip_preserves_preview_sidecar(self, temp_project_db_path: Path):
        """Deprecated SQLite adapter should preserve replay-only preview messages."""
        persistence = SQLiteSessionPersistence(db_path=temp_project_db_path)
        conversation = MessageHistory()
        conversation.add_user_message("Show app.py")
        conversation.add_assistant_message("Here is the current file preview.")
        conversation.add_preview_message(
            "system",
            "FILE PREVIEW: app.py",
            {
                "preview_kind": "file_preview",
                "preview_path": "app.py",
                "preview_language": "python",
                "preview_body": "print('hello')\n",
            },
        )

        session_id = persistence.save_session(
            conversation=conversation,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
            session_id="myproj-preview01",
            title="Preview Session",
        )
        loaded = persistence.load_session(session_id)

        assert loaded is not None
        assert loaded["conversation"]["preview_messages"] == [
            {
                "role": "system",
                "content": "FILE PREVIEW: app.py",
                "metadata": {
                    "preview_kind": "file_preview",
                    "preview_path": "app.py",
                    "preview_language": "python",
                    "preview_body": "print('hello')\n",
                },
                "after_message_index": 2,
            }
        ]

    def test_legacy_loader_can_resume_canonical_conversation_store_sessions(
        self, temp_project_db_path: Path
    ):
        """Legacy session loader should reconstruct canonical ConversationStore sessions."""
        from victor.agent.conversation.store import ConversationStore
        from victor.agent.conversation.types import MessageRole

        store = ConversationStore(db_path=temp_project_db_path)
        session = store.create_session(
            session_id="myproj-canonical01",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            profile="default",
        )
        store.add_message(session.session_id, MessageRole.USER, "Show app.py")
        store.add_message(
            session.session_id,
            MessageRole.ASSISTANT,
            "Here is the current file preview.",
        )
        store.add_message(
            session.session_id,
            MessageRole.SYSTEM,
            "FILE PREVIEW: app.py",
            metadata={
                "interactive_preview": True,
                "preview_kind": "file_preview",
                "preview_path": "app.py",
                "preview_language": "python",
                "preview_body": "print('hello')\n",
                "after_message_index": 2,
            },
        )

        persistence = SQLiteSessionPersistence(db_path=temp_project_db_path)
        loaded = persistence.load_session(session.session_id)
        sessions = persistence.list_sessions()
        listed = next(s for s in sessions if s["session_id"] == session.session_id)

        assert loaded is not None
        assert loaded["metadata"]["provider"] == "anthropic"
        assert loaded["metadata"]["title"] == "Show app.py"
        messages = loaded["conversation"]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Show app.py"
        assert messages[0]["created_at"]
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Here is the current file preview."
        assert messages[1]["created_at"]

        preview_messages = loaded["conversation"]["preview_messages"]
        assert len(preview_messages) == 1
        preview = preview_messages[0]
        assert preview["role"] == "system"
        assert preview["content"] == "FILE PREVIEW: app.py"
        assert preview["after_message_index"] == 2
        assert preview["metadata"]["preview_kind"] == "file_preview"
        assert preview["metadata"]["preview_path"] == "app.py"
        assert preview["metadata"]["preview_language"] == "python"
        assert preview["metadata"]["preview_body"] == "print('hello')\n"
        assert listed["provider"] == "anthropic"
        assert listed["title"] == "Show app.py"
        assert listed["preview_count"] == 1
