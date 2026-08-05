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

"""Integration tests for victor sessions command with real database."""

from __future__ import annotations

import pytest
import re
from pathlib import Path
import tempfile
import shutil

from victor.agent.conversation.store import ConversationStore
from victor.agent.conversation.types import ConversationMessage, MessageRole
from victor.config.settings import get_project_paths


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


@pytest.fixture
def project_db_path():
    """Get project database path."""
    return get_project_paths().project_db


@pytest.fixture
def backup_db(project_db_path):
    """Backup existing database before tests and provide clean state."""
    backup_path = project_db_path.with_suffix(".db.backup")

    # Backup existing database
    if project_db_path.exists():
        shutil.copy(project_db_path, backup_path)

    # Clear the database for clean test state
    # Also remove WAL and SHM files to avoid I/O errors
    if project_db_path.exists():
        project_db_path.unlink()

    # Clean up SQLite auxiliary files
    for ext in ["-wal", "-shm"]:
        aux_file = project_db_path.with_suffix(f".db{ext}")
        if aux_file.exists():
            try:
                aux_file.unlink()
            except Exception:
                pass  # Ignore errors, might not exist

    yield

    # Restore backup
    if backup_path.exists():
        shutil.copy(backup_path, project_db_path)
        backup_path.unlink()


class TestSessionsCLIIntegration:
    """Integration tests for sessions CLI command."""

    @pytest.mark.integration
    def test_project_db_path_uses_isolated_integration_victor_dir(
        self,
        project_db_path,
        isolated_integration_project_victor_dir,
    ):
        """Integration tests should not touch the repo-local project database."""
        assert project_db_path == isolated_integration_project_victor_dir / "project.db"

    @pytest.mark.integration
    def test_sessions_list_command(self, backup_db):
        """Test victor sessions list command with real database."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Create sample session
        store = ConversationStore()
        session = store.create_session(
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
        )
        store.add_message(
            session.session_id,
            MessageRole.USER,
            "Test",
        )
        session_id = session.session_id

        # Test list command
        result = runner.invoke(sessions_app, ["list"])
        assert result.exit_code == 0
        rendered = strip_ansi(result.stdout)
        assert session_id in rendered
        assert "Saved Sessions (last 1)" in rendered
        assert "Total: 1 session(s)" in rendered

    @pytest.mark.integration
    def test_sessions_show_command(self, backup_db):
        """Test victor sessions show command."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Create sample session
        store = ConversationStore()
        session = store.create_session(
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
        )
        store.add_message(
            session.session_id,
            MessageRole.USER,
            "Test",
        )
        session_id = session.session_id

        # Test show command
        result = runner.invoke(sessions_app, ["show", session_id])
        assert result.exit_code == 0
        # Note: ConversationStore doesn't have title, so we check for session_id instead
        assert session_id in result.stdout

    @pytest.mark.integration
    def test_parallel_sessions_workflow(self, backup_db):
        """Test parallel session creation and retrieval."""
        store = ConversationStore()

        # Create session 1
        session_1 = store.create_session(
            model="ollama:qwen2.5-coder:7b",
            provider="ollama",
            profile="default",
        )
        store.add_message(
            session_1.session_id,
            MessageRole.USER,
            "DevOps task",
        )
        session_id_1 = session_1.session_id

        # Small delay to ensure different timestamp for unique ID
        import time

        time.sleep(0.01)

        # Create session 2
        session_2 = store.create_session(
            model="gpt-4",
            provider="openai",
            profile="default",
        )
        store.add_message(
            session_2.session_id,
            MessageRole.USER,
            "Testing task",
        )
        session_id_2 = session_2.session_id

        # Verify both sessions exist and have different IDs
        assert (
            session_id_1 != session_id_2
        ), f"Session IDs should be unique but got: {session_id_1} == {session_id_2}"

        # Load both sessions
        loaded_session_1 = store.get_session(session_id_1)
        loaded_session_2 = store.get_session(session_id_2)

        assert loaded_session_1 is not None
        assert loaded_session_2 is not None
        assert loaded_session_1.model == "ollama:qwen2.5-coder:7b"
        assert loaded_session_2.model == "gpt-4"

    @pytest.mark.integration
    def test_session_id_parsing(self):
        """Test session ID parsing utility."""
        from victor.agent.session_id import parse_session_id, validate_session_id

        # Valid session ID
        session_id = "myproj-9Kx7Z2"
        assert validate_session_id(session_id) is True

        parts = parse_session_id(session_id)
        assert parts["project_root"] == "myproj"
        assert parts["base62_timestamp"] == "9Kx7Z2"
        assert isinstance(parts["timestamp_ms"], int)
        assert isinstance(parts["timestamp_iso"], str)

        # Invalid session ID
        assert validate_session_id("invalid") is False
        assert validate_session_id("no-separator") is False


class TestChatSessionFlagsIntegration:
    """Integration tests for victor chat --sessions and --sessionid flags."""

    @pytest.mark.integration
    def test_chat_sessions_flag(self, backup_db):
        """Test victor chat --sessions lists sessions."""
        # This requires full orchestrator setup
        # Testing with subprocess would be more appropriate
        pass

    @pytest.mark.integration
    def test_chat_sessionid_flag(self, backup_db):
        """Test victor chat --sessionid restores session."""
        # This requires full orchestrator setup
        # Testing with subprocess would be more appropriate
        pass

    @pytest.mark.integration
    def test_sessions_clear_all(self, backup_db):
        """Test victor sessions clear command to delete all sessions."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Create multiple sample sessions
        store = ConversationStore()
        session_1 = store.create_session(
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
        )
        store.add_message(
            session_1.session_id,
            MessageRole.USER,
            "Test 1",
        )

        # Small delay to ensure different timestamp
        import time

        time.sleep(0.01)

        session_2 = store.create_session(
            model="gpt-4",
            provider="openai",
            profile="default",
        )
        store.add_message(
            session_2.session_id,
            MessageRole.USER,
            "Test 2",
        )

        time.sleep(0.01)

        session_3 = store.create_session(
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
        )
        store.add_message(
            session_3.session_id,
            MessageRole.USER,
            "Test 3",
        )

        # Verify sessions exist
        sessions_before = store.list_sessions(limit=100)
        assert len(sessions_before) >= 3

        # Clear all sessions with --yes flag
        result = runner.invoke(sessions_app, ["clear", "--yes"])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "Cleared" in clean_output
        assert "session(s) from database" in clean_output

        # Verify all sessions are deleted
        sessions_after = store.list_sessions(limit=100)
        assert len(sessions_after) == 0

    @pytest.mark.integration
    def test_sessions_clear_with_prefix(self, backup_db):
        """Test victor sessions clear with prefix to delete specific sessions."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Create sessions with specific prefixes
        store = ConversationStore()

        # Create 3 sessions to increase chance of finding a unique prefix
        session_1 = store.create_session(
            model="ollama:qwen2.5-coder:7b",
            provider="ollama",
            profile="default",
        )
        store.add_message(
            session_1.session_id,
            MessageRole.USER,
            "DevOps task",
        )
        session_id_1 = session_1.session_id

        # Small delay to ensure different timestamp
        import time

        time.sleep(0.05)  # Increased delay

        session_2 = store.create_session(
            model="gpt-4",
            provider="openai",
            profile="default",
        )
        store.add_message(
            session_2.session_id,
            MessageRole.USER,
            "Testing task",
        )
        session_id_2 = session_2.session_id

        time.sleep(0.05)

        session_3 = store.create_session(
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
        )
        store.add_message(
            session_3.session_id,
            MessageRole.USER,
            "Third task",
        )
        session_id_3 = session_3.session_id

        # Verify all sessions exist
        assert store.get_session(session_id_1) is not None
        assert store.get_session(session_id_2) is not None
        assert store.get_session(session_id_3) is not None

        # Find a unique prefix that matches only session_id_1
        # Check against both session_id_2 and session_id_3
        prefix = None
        for prefix_len in range(15, 6, -1):  # Try longer prefixes first
            candidate_prefix = session_id_1[:prefix_len]
            if not session_id_2.startswith(candidate_prefix) and not session_id_3.startswith(
                candidate_prefix
            ):
                prefix = candidate_prefix
                break  # Found a unique prefix

        if not prefix or len(prefix) < 6:
            # If no unique prefix found, skip this test
            pytest.skip(
                f"Could not find unique prefix for test. session_id_1={session_id_1[:20]}, session_id_2={session_id_2[:20]}, session_id_3={session_id_3[:20]}"
            )

        # Verify that the prefix actually matches session_id_1
        assert session_id_1.startswith(
            prefix
        ), f"Prefix {prefix} should match session_id_1 {session_id_1}"
        # Verify that the prefix doesn't match other sessions
        assert not session_id_2.startswith(
            prefix
        ), f"Prefix {prefix} should NOT match session_id_2 {session_id_2}"
        assert not session_id_3.startswith(
            prefix
        ), f"Prefix {prefix} should NOT match session_id_3 {session_id_3}"

        # Clear sessions matching prefix
        result = runner.invoke(sessions_app, ["clear", prefix, "--yes"])
        assert result.exit_code == 0
        assert "Cleared" in result.stdout
        assert f"matching prefix '{prefix}'" in result.stdout

        # Create a new store instance to avoid stale cache issues
        store_after_clear = ConversationStore()

        # Verify session with matching prefix is deleted
        assert store_after_clear.get_session(session_id_1) is None

        # Verify other sessions still exist
        assert store_after_clear.get_session(session_id_2) is not None
        assert store_after_clear.get_session(session_id_3) is not None

    @pytest.mark.integration
    def test_sessions_clear_prefix_too_short(self, backup_db):
        """Test victor sessions clear with prefix shorter than 6 chars."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Try to clear with prefix shorter than 6 characters
        result = runner.invoke(sessions_app, ["clear", "abc", "--yes"])
        assert result.exit_code != 0
        clean_output = strip_ansi(result.stdout)
        assert "Prefix must be at least 6 characters long" in clean_output

    @pytest.mark.integration
    def test_sessions_clear_prefix_not_found(self, backup_db):
        """Test victor sessions clear with non-matching prefix."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Try to clear with prefix that doesn't match any sessions
        result = runner.invoke(sessions_app, ["clear", "nomatch-999", "--yes"])
        assert result.exit_code == 0
        clean_output = strip_ansi(result.stdout)
        assert "No sessions found matching prefix 'nomatch-999'" in clean_output

    @pytest.mark.integration
    def test_sessions_clear_empty_database(self, backup_db):
        """Test victor sessions clear when database is empty."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Ensure database is empty by clearing all sessions
        store = ConversationStore()
        all_sessions = store.list_sessions(limit=10000)
        for session in all_sessions:
            store.delete_session(session.session_id)

        # Verify database is now empty
        sessions = store.list_sessions(limit=100)
        assert len(sessions) == 0

        # Try to clear empty database
        result = runner.invoke(sessions_app, ["clear", "--yes"])
        assert result.exit_code == 0
        assert "No sessions found" in result.stdout

    @pytest.mark.integration
    def test_sessions_list_all(self, backup_db):
        """Test victor sessions list --all command."""
        from typer.testing import CliRunner
        from victor.ui.commands.sessions import sessions_app

        runner = CliRunner()

        # Create more than 10 sessions to test --all flag
        store = ConversationStore()
        import time

        for i in range(15):
            session = store.create_session(
                model="claude-sonnet-4-20250514",
                provider="anthropic",
                profile="default",
            )
            store.add_message(
                session.session_id,
                MessageRole.USER,
                f"Test {i}",
            )
            time.sleep(0.01)  # Ensure unique timestamps

        # List with default limit (should show 10)
        result_default = runner.invoke(sessions_app, ["list"])
        assert result_default.exit_code == 0
        rendered_default = strip_ansi(result_default.stdout)
        assert "Saved Sessions (last 10)" in rendered_default
        assert "Total: 10 session(s)" in rendered_default

        # List with --all flag (should show all 15)
        result_all = runner.invoke(sessions_app, ["list", "--all"])
        assert result_all.exit_code == 0
        rendered_all = strip_ansi(result_all.stdout)
        assert "Saved Sessions (last 15)" in rendered_all
        assert "Total: 15 session(s)" in rendered_all
