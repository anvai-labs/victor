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

"""Tests for victor sessions CLI command."""

from __future__ import annotations

import os
import pytest
import re
from typer.testing import CliRunner
from pathlib import Path
import tempfile
import json

from victor.agent.message_history import MessageHistory
from victor.agent.conversation.store import ConversationStore
from victor.ui.commands.sessions import sessions_app


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


@pytest.fixture
def runner():
    """Test runner for CLI commands."""
    return CliRunner()


@pytest.fixture
def runner_with_db(temp_db_path):
    """Test runner with test database path in environment."""
    return CliRunner(env={"VICTOR_CONVERSATION_DB": str(temp_db_path)})


@pytest.fixture
def temp_db_path():
    """Temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def sample_persistence(temp_db_path):
    """Create a canonical ConversationStore with sample data."""
    persistence = ConversationStore(db_path=temp_db_path)

    # Create sample sessions
    sessions_data = [
        {
            "session_id": "myproj-9Kx7Z2",
            "title": "CI/CD Pipeline Setup",
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "profile": "default",
            "messages": [
                {"role": "user", "content": "Setup CI/CD"},
                {"role": "assistant", "content": "I'll help you setup CI/CD"},
            ],
        },
        {
            "session_id": "myproj-9Kx8A3B",
            "title": "Unit Tests",
            "provider": "openai",
            "model": "gpt-4",
            "profile": "default",
            "messages": [
                {"role": "user", "content": "Write tests"},
                {"role": "assistant", "content": "Here are the tests"},
            ],
        },
    ]

    for session in sessions_data:
        persistence.save_session(
            conversation={"messages": session["messages"]},
            model=session["model"],
            provider=session["provider"],
            profile=session["profile"],
            session_id=session["session_id"],
            title=session["title"],
        )

    return persistence


class TestSessionsCommand:
    """Test suite for victor sessions command."""

    def test_sessions_list_default(self, runner_with_db, sample_persistence):
        """Test 'victor sessions list' with default limit (10)."""
        result = runner_with_db.invoke(sessions_app, ["list"])
        assert result.exit_code == 0
        # Should show both sessions (Rich tables may truncate long titles)
        assert "CI/CD" in result.stdout
        assert "Unit Tests" in result.stdout or "Unit" in result.stdout
        assert "myproj-9Kx7Z2" in result.stdout
        assert "myproj-9Kx8A3B" in result.stdout
        # Verify table structure
        assert "Session ID" in result.stdout
        assert "Title" in result.stdout
        assert "Previews" in result.stdout

    def test_sessions_list_with_limit(self, runner_with_db, sample_persistence):
        """Test 'victor sessions list --limit 1'."""
        result = runner_with_db.invoke(sessions_app, ["list", "--limit", "1"])
        assert result.exit_code == 0
        # Should show only 1 session (may be either one due to ordering)
        assert ("myproj-9Kx7Z2" in result.stdout) or ("myproj-9Kx8A3B" in result.stdout)

    def test_sessions_list_json(self, runner_with_db, sample_persistence):
        """Test 'victor sessions list --json'."""
        result = runner_with_db.invoke(sessions_app, ["list", "--json"])
        assert result.exit_code == 0

        # Parse JSON output (strip ANSI codes first)
        clean_output = strip_ansi(result.stdout)
        sessions = json.loads(clean_output)
        assert isinstance(sessions, list)
        # Filter to only sessions created by sample_persistence fixture
        sample_sessions = [
            s for s in sessions if s["session_id"] in ["myproj-9Kx7Z2", "myproj-9Kx8A3B"]
        ]
        assert (
            len(sample_sessions) == 2
        ), f"Expected 2 sample sessions but found {len(sample_sessions)}. Total sessions: {len(sessions)}"

        # Verify the sample sessions have the correct data
        for session in sample_sessions:
            assert session["session_id"] in ["myproj-9Kx7Z2", "myproj-9Kx8A3B"]
            assert "title" in session
            assert "model" in session
            assert "provider" in session
            assert session["preview_count"] == 0

    def test_sessions_list_json_includes_preview_count(self, runner_with_db, temp_db_path):
        """Test 'victor sessions list --json' includes preview counts."""
        persistence = ConversationStore(db_path=temp_db_path)
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

        persistence.save_session(
            conversation=conversation,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
            session_id="myproj-preview-list",
            title="Preview List Session",
        )

        result = runner_with_db.invoke(sessions_app, ["list", "--json"])
        assert result.exit_code == 0

        sessions = json.loads(strip_ansi(result.stdout))
        preview_session = next(
            session for session in sessions if session["session_id"] == "myproj-preview-list"
        )
        assert preview_session["preview_count"] == 1

    def test_sessions_list_empty(self, runner_with_db, temp_db_path):
        """Test 'victor sessions list' with empty database."""
        # Don't create any sessions - just verify command works
        # Note: Due to fixture sharing, may see sessions from other tests
        result = runner_with_db.invoke(sessions_app, ["list"])
        assert result.exit_code == 0
        # Just verify the command runs successfully and shows a table
        assert "Session ID" in result.stdout or "No sessions found" in result.stdout

    def test_sessions_search(self, runner_with_db, sample_persistence):
        """Test 'victor sessions search CI/CD'."""
        result = runner_with_db.invoke(sessions_app, ["search", "CI/CD"])
        assert result.exit_code == 0
        # Should find CI/CD session but not Unit Tests
        assert "CI/CD" in result.stdout and "Pipeline" in result.stdout
        assert "Unit Tests" not in result.stdout
        assert "Previews" in result.stdout

    def test_sessions_search_json(self, runner_with_db, sample_persistence):
        """Test 'victor sessions search --json'."""
        result = runner_with_db.invoke(sessions_app, ["search", "CI/CD", "--json"])
        assert result.exit_code == 0

        # Strip ANSI codes before parsing JSON
        clean_output = strip_ansi(result.stdout)
        sessions = json.loads(clean_output)
        assert isinstance(sessions, list)
        assert len(sessions) == 1
        assert sessions[0]["title"] == "CI/CD Pipeline Setup"
        assert sessions[0]["preview_count"] == 0

    def test_sessions_search_finds_preview_messages(self, runner_with_db, temp_db_path):
        """Test 'victor sessions search' matches replay-only preview content."""
        persistence = ConversationStore(db_path=temp_db_path)
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

        persistence.save_session(
            conversation=conversation,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
            session_id="myproj-preview-search",
            title="Preview Search Session",
        )

        result = runner_with_db.invoke(sessions_app, ["search", "hello", "--json"])
        assert result.exit_code == 0

        clean_output = strip_ansi(result.stdout)
        sessions = json.loads(clean_output)
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "myproj-preview-search"
        assert sessions[0]["title"] == "Preview Search Session"
        assert sessions[0]["preview_count"] == 1

    def test_sessions_show(self, runner_with_db, sample_persistence):
        """Test 'victor sessions show <session_id>'."""
        result = runner_with_db.invoke(sessions_app, ["show", "myproj-9Kx7Z2"])
        assert result.exit_code == 0
        # Check session details (may be in different formats)
        assert "CI/CD" in result.stdout and "Pipeline" in result.stdout and "Setup" in result.stdout
        assert "anthropic" in result.stdout
        assert "2" in result.stdout  # Message count

    def test_sessions_show_json(self, runner_with_db, sample_persistence):
        """Test 'victor sessions show --json'."""
        result = runner_with_db.invoke(sessions_app, ["show", "myproj-9Kx7Z2", "--json"])
        assert result.exit_code == 0

        # Strip ANSI codes before parsing JSON
        clean_output = strip_ansi(result.stdout)
        session = json.loads(clean_output)
        assert session["metadata"]["session_id"] == "myproj-9Kx7Z2"
        assert session["metadata"]["title"] == "CI/CD Pipeline Setup"
        assert session["metadata"]["model"] == "claude-sonnet-4-20250514"
        assert session["metadata"]["provider"] == "anthropic"

    def test_sessions_show_json_preserves_preview_messages(self, runner_with_db, temp_db_path):
        """Test 'victor sessions show --json' preserves preview sidecar payloads."""
        persistence = ConversationStore(db_path=temp_db_path)
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

        persistence.save_session(
            conversation=conversation,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
            session_id="myproj-preview01",
            title="Preview Session",
        )

        result = runner_with_db.invoke(sessions_app, ["show", "myproj-preview01", "--json"])
        assert result.exit_code == 0

        clean_output = strip_ansi(result.stdout)
        session = json.loads(clean_output)
        preview = session["conversation"]["preview_messages"]
        assert len(preview) == 1
        assert preview[0]["role"] == "system"
        assert preview[0]["content"] == "FILE PREVIEW: app.py"
        assert preview[0]["after_message_index"] == 2
        assert preview[0]["metadata"] == {
            "preview_kind": "file_preview",
            "preview_path": "app.py",
            "preview_language": "python",
            "preview_body": "print('hello')\n",
        }

    def test_sessions_show_displays_preview_messages(self, runner_with_db, temp_db_path):
        """Test 'victor sessions show' includes preview sidecars in human output."""
        persistence = ConversationStore(db_path=temp_db_path)
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

        persistence.save_session(
            conversation=conversation,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
            session_id="myproj-preview-human",
            title="Preview Human Session",
        )

        result = runner_with_db.invoke(sessions_app, ["show", "myproj-preview-human"])
        assert result.exit_code == 0

        clean_output = strip_ansi(result.stdout)
        assert "Preview Messages:" in clean_output
        assert "FILE PREVIEW: app.py" in clean_output
        assert "print('hello')" in clean_output
        assert "Previews:" in clean_output

    def test_sessions_show_not_found(self, runner_with_db, sample_persistence):
        """Test 'victor sessions show' with non-existent session."""
        result = runner_with_db.invoke(sessions_app, ["show", "nonexistent-12345"])
        assert result.exit_code != 0 or "not found" in result.stdout.lower()

    def test_sessions_delete(self, runner_with_db, sample_persistence):
        """Test 'victor sessions delete <session_id>'."""
        # Delete session
        result = runner_with_db.invoke(sessions_app, ["delete", "myproj-9Kx7Z2", "--yes"])
        assert result.exit_code == 0
        assert "deleted" in result.stdout.lower()

        # Verify it's gone
        result = runner_with_db.invoke(sessions_app, ["show", "myproj-9Kx7Z2"])
        assert "not found" in result.stdout.lower()

    def test_sessions_export(self, runner_with_db, sample_persistence, tmp_path):
        """Test 'victor sessions export'."""
        export_file = tmp_path / "sessions.json"
        result = runner_with_db.invoke(sessions_app, ["export", "--output", str(export_file)])
        assert result.exit_code == 0
        assert export_file.exists()

        # Verify export (exports full session data with nested structure)
        with open(export_file) as f:
            exported = json.load(f)

        # Filter to only sessions created by sample_persistence fixture
        sample_exported = [
            s
            for s in exported
            if s["metadata"]["session_id"] in ["myproj-9Kx7Z2", "myproj-9Kx8A3B"]
        ]
        assert (
            len(sample_exported) == 2
        ), f"Expected 2 sample sessions but found {len(sample_exported)}. Total exported: {len(exported)}"

        # Check metadata structure
        assert sample_exported[0]["metadata"]["session_id"] in [
            "myproj-9Kx7Z2",
            "myproj-9Kx8A3B",
        ]

    def test_sessions_export_preserves_preview_messages(
        self, runner_with_db, temp_db_path, tmp_path
    ):
        """Test 'victor sessions export' preserves replay-only preview sidecar payloads."""
        persistence = ConversationStore(db_path=temp_db_path)
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

        persistence.save_session(
            conversation=conversation,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
            session_id="myproj-preview-export",
            title="Preview Export Session",
        )

        export_file = tmp_path / "preview_sessions.json"
        result = runner_with_db.invoke(sessions_app, ["export", "--output", str(export_file)])
        assert result.exit_code == 0
        assert export_file.exists()

        exported = json.loads(export_file.read_text())
        preview_export = next(
            session
            for session in exported
            if session["metadata"]["session_id"] == "myproj-preview-export"
        )

        preview = preview_export["conversation"]["preview_messages"]
        assert len(preview) == 1
        assert preview[0]["role"] == "system"
        assert preview[0]["content"] == "FILE PREVIEW: app.py"
        assert preview[0]["after_message_index"] == 2
        assert preview[0]["metadata"] == {
            "preview_kind": "file_preview",
            "preview_path": "app.py",
            "preview_language": "python",
            "preview_body": "print('hello')\n",
        }


class TestSessionsChatFlags:
    """Test suite for victor chat --sessions and --sessionid flags."""

    def test_chat_sessions_flag_lists_sessions(self, runner):
        """Test 'victor chat --sessions' lists top 20 sessions."""
        # This will be tested with integration test since it requires full chat setup
        pass

    def test_chat_sessionid_flag_restores_session(self, runner):
        """Test 'victor chat --sessionid <id>' restores session."""
        # This will be tested with integration test since it requires full chat setup
        pass


class TestSessionsClearCommand:
    """Test suite for victor sessions clear command."""

    def test_sessions_clear_all_with_yes_flag(self, runner_with_db, sample_persistence):
        """Test 'victor sessions clear --yes' deletes all sessions."""
        # Verify sessions exist
        persistence = sample_persistence
        sessions_before = persistence.list_sessions(limit=100)
        # Filter to sample sessions
        sample_count = len(
            [s for s in sessions_before if s.session_id in ["myproj-9Kx7Z2", "myproj-9Kx8A3B"]]
        )
        assert sample_count == 2

        # Clear all sessions
        result = runner_with_db.invoke(sessions_app, ["clear", "--yes"])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "Cleared" in clean_output
        assert "session(s) from database" in clean_output

        # Verify sessions are deleted
        sessions_after = ConversationStore(db_path=persistence.db_path).list_sessions(limit=100)
        # Filter to check if sample sessions are gone
        sample_after = [
            s for s in sessions_after if s.session_id in ["myproj-9Kx7Z2", "myproj-9Kx8A3B"]
        ]
        assert len(sample_after) == 0

    def test_sessions_clear_with_prefix(self, runner_with_db, sample_persistence):
        """Test 'victor sessions clear <prefix>' deletes specific sessions."""
        persistence = sample_persistence

        # Clear sessions matching prefix "myproj-9Kx7" (should match myproj-9Kx7Z2)
        result = runner_with_db.invoke(sessions_app, ["clear", "myproj-9Kx7", "--yes"])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "Cleared" in clean_output
        assert "matching prefix 'myproj-9Kx7'" in clean_output

        # Verify session with matching prefix is deleted
        refreshed_store = ConversationStore(db_path=persistence.db_path)
        assert refreshed_store.load_session("myproj-9Kx7Z2") is None

        # Verify other session still exists
        assert refreshed_store.load_session("myproj-9Kx8A3B") is not None

    def test_sessions_clear_prefix_too_short(self, runner_with_db, sample_persistence):
        """Test 'victor sessions clear' with prefix < 6 chars."""
        result = runner_with_db.invoke(sessions_app, ["clear", "short", "--yes"])
        assert result.exit_code != 0
        clean_output = strip_ansi(result.stdout)
        assert "Prefix must be at least 6 characters long" in clean_output

    def test_sessions_clear_prefix_not_found(self, runner_with_db, temp_db_path):
        """Test 'victor sessions clear' with non-matching prefix."""
        result = runner_with_db.invoke(sessions_app, ["clear", "nomatch-999", "--yes"])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "No sessions found matching prefix 'nomatch-999'" in clean_output

    def test_sessions_clear_empty_database(self, runner_with_db, temp_db_path):
        """Test 'victor sessions clear' with empty database."""
        # Clear any existing sessions first
        persistence = ConversationStore(db_path=temp_db_path)
        all_sessions = persistence.list_sessions(limit=10000)
        for session in all_sessions:
            persistence.delete_session(session.session_id)

        # Now test with empty database
        result = runner_with_db.invoke(sessions_app, ["clear", "--yes"])
        assert result.exit_code == 0
        assert "No sessions found" in result.stdout


class TestSessionsListAllFlag:
    """Test suite for victor sessions list --all flag."""

    def test_sessions_list_with_all_flag(self, runner_with_db, sample_persistence):
        """Test 'victor sessions list --all' lists all sessions."""
        # Create additional sessions beyond default limit
        persistence = sample_persistence
        for i in range(15):
            persistence.save_session(
                conversation={"messages": [{"role": "user", "content": f"Test {i}"}]},
                model="claude-sonnet-4-20250514",
                provider="anthropic",
                profile="default",
                title=f"Extra Session {i}",
            )

        # List with default limit
        result_default = runner_with_db.invoke(sessions_app, ["list"])
        assert result_default.exit_code == 0

        # List with --all flag
        result_all = runner_with_db.invoke(sessions_app, ["list", "--all"])
        assert result_all.exit_code == 0

        # --all should show more sessions than default
        # We can't count exact sessions in table output, but can check for specific sessions
        # Note: Table wraps titles across lines, so check for "Extra" which appears in "Extra Session X"
        assert "Extra" in result_all.stdout
