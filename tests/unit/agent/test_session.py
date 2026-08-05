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

"""Tests for the session persistence module."""

import tempfile
from pathlib import Path

import pytest

from victor.agent.message_history import MessageHistory
from victor.agent.session import (
    Session,
    SessionPersistence,
    SessionMetadata,
    get_session_manager,
)


@pytest.fixture
def temp_session_dir():
    """Create a temporary directory for session storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def session_manager(temp_session_dir):
    """Create a session manager with temporary storage."""
    return SessionPersistence(session_dir=temp_session_dir)


@pytest.fixture
def sample_conversation():
    """Create a sample conversation manager."""
    conv = MessageHistory(system_prompt="You are a helpful assistant.")
    conv.add_user_message("Hello, how are you?")
    conv.add_assistant_message("I'm doing well, thank you for asking!")
    conv.add_user_message("Can you help me with Python?")
    conv.add_assistant_message("Of course! What would you like to know about Python?")
    return conv


class TestSessionMetadata:
    """Tests for SessionMetadata dataclass."""

    def test_to_dict(self):
        """to_dict should return all fields."""
        metadata = SessionMetadata(
            session_id="test_123",
            created_at="2025-01-01T10:00:00",
            updated_at="2025-01-01T11:00:00",
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
            message_count=5,
            title="Test Session",
            tags=["test", "demo"],
        )
        data = metadata.to_dict()

        assert data["session_id"] == "test_123"
        assert data["model"] == "claude-sonnet-4-20250514"
        assert data["tags"] == ["test", "demo"]

    def test_from_dict(self):
        """from_dict should restore all fields."""
        data = {
            "session_id": "test_456",
            "created_at": "2025-01-02T10:00:00",
            "updated_at": "2025-01-02T11:00:00",
            "model": "gpt-4",
            "provider": "openai",
            "profile": "work",
            "message_count": 10,
            "title": "Work Session",
            "tags": ["work"],
        }
        metadata = SessionMetadata.from_dict(data)

        assert metadata.session_id == "test_456"
        assert metadata.provider == "openai"
        assert metadata.title == "Work Session"


class TestSession:
    """Tests for Session dataclass."""

    def test_to_dict_and_from_dict(self):
        """Session should serialize and deserialize correctly."""
        metadata = SessionMetadata(
            session_id="test_789",
            created_at="2025-01-01T10:00:00",
            updated_at="2025-01-01T11:00:00",
            model="test-model",
            provider="test-provider",
            profile="default",
            message_count=2,
        )
        conversation = {
            "system_prompt": "Test prompt",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        session = Session(
            metadata=metadata,
            conversation=conversation,
            conversation_state={"stage": "exploring"},
        )

        # Serialize
        data = session.to_dict()

        # Deserialize
        restored = Session.from_dict(data)

        assert restored.metadata.session_id == "test_789"
        assert restored.conversation["system_prompt"] == "Test prompt"
        assert restored.conversation_state["stage"] == "exploring"


class TestSessionPersistence:
    """Tests for SessionPersistence class."""

    def test_save_session(self, session_manager, sample_conversation):
        """save_session should create a session file."""
        session_id = session_manager.save_session(
            conversation=sample_conversation,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            profile="default",
        )

        assert session_id is not None
        assert (session_manager.session_dir / f"{session_id}.json").exists()

    def test_save_session_with_custom_title(self, session_manager, sample_conversation):
        """save_session should accept custom title."""
        session_id = session_manager.save_session(
            conversation=sample_conversation,
            model="test-model",
            provider="test-provider",
            title="My Custom Title",
        )

        session = session_manager.load_session(session_id)
        assert session is not None
        assert session.metadata.title == "My Custom Title"

    def test_load_session(self, session_manager, sample_conversation):
        """load_session should restore session data."""
        session_id = session_manager.save_session(
            conversation=sample_conversation,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
        )

        session = session_manager.load_session(session_id)

        assert session is not None
        assert session.metadata.model == "claude-sonnet-4-20250514"
        assert session.metadata.provider == "anthropic"
        assert session.metadata.message_count == 4  # 4 messages (system prompt added lazily)

    def test_load_session_preserves_preview_messages(self, session_manager):
        """load_session should retain replay-only preview sidecar messages."""
        conversation = MessageHistory()
        conversation.add_user_message("Show me config.py")
        conversation.add_assistant_message("Here is the current file preview.")
        conversation.add_preview_message(
            "system",
            "FILE PREVIEW: config.py",
            {
                "preview_kind": "file_preview",
                "preview_path": "config.py",
                "preview_language": "python",
                "preview_body": "SETTING = True\n",
            },
        )

        session_id = session_manager.save_session(
            conversation=conversation,
            model="claude-sonnet-4-20250514",
            provider="anthropic",
        )

        session = session_manager.load_session(session_id)

        assert session is not None
        assert session.conversation["preview_messages"] == [
            {
                "role": "system",
                "content": "FILE PREVIEW: config.py",
                "metadata": {
                    "preview_kind": "file_preview",
                    "preview_path": "config.py",
                    "preview_language": "python",
                    "preview_body": "SETTING = True\n",
                },
                "after_message_index": 2,
            }
        ]

    def test_load_nonexistent_session(self, session_manager):
        """load_session should return None for missing sessions."""
        session = session_manager.load_session("nonexistent_id")
        assert session is None

    def test_list_sessions(self, session_manager, sample_conversation):
        """list_sessions should return saved sessions."""
        # Save multiple sessions
        session_manager.save_session(
            conversation=sample_conversation,
            model="model-1",
            provider="provider-1",
            session_id="session_001",
        )
        session_manager.save_session(
            conversation=sample_conversation,
            model="model-2",
            provider="provider-2",
            session_id="session_002",
        )

        sessions = session_manager.list_sessions()

        assert len(sessions) == 2
        session_ids = [s.session_id for s in sessions]
        assert "session_001" in session_ids
        assert "session_002" in session_ids

    def test_list_sessions_with_limit(self, session_manager, sample_conversation):
        """list_sessions should respect limit parameter."""
        for i in range(5):
            session_manager.save_session(
                conversation=sample_conversation,
                model="test-model",
                provider="test-provider",
                session_id=f"session_{i:03d}",
            )

        sessions = session_manager.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_list_sessions_with_provider_filter(self, session_manager, sample_conversation):
        """list_sessions should filter by provider."""
        session_manager.save_session(
            conversation=sample_conversation,
            model="test-model",
            provider="anthropic",
            session_id="anthropic_session",
        )
        session_manager.save_session(
            conversation=sample_conversation,
            model="test-model",
            provider="openai",
            session_id="openai_session",
        )

        sessions = session_manager.list_sessions(provider="anthropic")

        assert len(sessions) == 1
        assert sessions[0].provider == "anthropic"

    def test_delete_session(self, session_manager, sample_conversation):
        """delete_session should remove session file."""
        session_id = session_manager.save_session(
            conversation=sample_conversation,
            model="test-model",
            provider="test-provider",
        )

        # Verify exists
        assert session_manager.load_session(session_id) is not None

        # Delete
        result = session_manager.delete_session(session_id)
        assert result is True

        # Verify deleted
        assert session_manager.load_session(session_id) is None

    def test_delete_nonexistent_session(self, session_manager):
        """delete_session should return False for missing sessions."""
        result = session_manager.delete_session("nonexistent")
        assert result is False

    def test_update_session(self, session_manager, sample_conversation):
        """update_session should update existing session."""
        session_id = session_manager.save_session(
            conversation=sample_conversation,
            model="test-model",
            provider="test-provider",
        )

        # Add more messages
        sample_conversation.add_user_message("Another question")
        sample_conversation.add_assistant_message("Another answer")

        # Update
        result = session_manager.update_session(session_id, sample_conversation)
        assert result is True

        # Verify update
        session = session_manager.load_session(session_id)
        assert session.metadata.message_count == 6  # Original 4 + 2 new

    def test_get_latest_session(self, session_manager, sample_conversation):
        """get_latest_session should return most recent session."""
        # No sessions initially
        assert session_manager.get_latest_session() is None

        # Add a session
        session_id = session_manager.save_session(
            conversation=sample_conversation,
            model="test-model",
            provider="test-provider",
        )

        latest = session_manager.get_latest_session()
        assert latest is not None
        assert latest.metadata.session_id == session_id

    def test_auto_generate_title(self, session_manager):
        """save_session should auto-generate title from first user message."""
        conv = MessageHistory()
        conv.add_user_message("How do I implement a binary search tree in Python?")
        conv.add_assistant_message("Here's how to implement a BST...")

        session_id = session_manager.save_session(
            conversation=conv,
            model="test-model",
            provider="test-provider",
        )

        session = session_manager.load_session(session_id)
        assert "binary search tree" in session.metadata.title.lower()


class TestGlobalSessionManager:
    """Tests for global session manager functions."""

    def test_get_session_manager_singleton(self):
        """get_session_manager should return same instance."""
        manager1 = get_session_manager()
        manager2 = get_session_manager()
        assert manager1 is manager2

    def test_default_session_dir(self):
        """Default session directory should be {project}/.victor/sessions/."""
        from victor.config.settings import get_project_paths

        manager = get_session_manager()
        expected_dir = get_project_paths().sessions_dir
        assert manager.session_dir == expected_dir
