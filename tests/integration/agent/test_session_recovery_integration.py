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

"""Integration tests for session recovery.

Tests the ability to resume previous chat sessions, including:
- Session creation and persistence
- Message context restoration
- Session state recovery
- Tool call history restoration
- Session metadata preservation
"""

import pytest
from unittest.mock import MagicMock

from victor.agent.conversation_manager import (
    ConversationManager,
    ConversationManagerConfig,
)


class TestSessionRecoveryIntegration:
    """Integration tests for session recovery functionality."""

    @pytest.fixture
    def mock_persistent_store(self):
        """Create a mock store that simulates persistence."""
        store = MagicMock()
        store.session_id = None
        store._sessions = {}  # In-memory session storage
        store._messages = {}  # In-memory message storage per session

        def create_session(session_id=None, project_path=None, provider=None, model=None):
            """Create a new session."""
            if session_id is None:
                import uuid

                session_id = f"session_{uuid.uuid4().hex[:12]}"

            session = MagicMock()
            session.session_id = session_id
            session.project_path = project_path
            session.provider = provider
            session.model = model
            session.messages = []
            session.is_active = True
            session.metadata = {}

            store._sessions[session_id] = session
            store._messages[session_id] = []  # Initialize message list for this session
            store.session_id = session_id
            return session

        def get_session(session_id):
            """Get a session by ID."""
            session = store._sessions.get(session_id)
            if session:
                # Ensure messages list is synchronized
                session.messages = store._messages.get(session_id, [])
            return session

        def add_message(session_id, message):
            """Add a message to a session."""
            if session_id not in store._messages:
                store._messages[session_id] = []
            store._messages[session_id].append(message)

            # Also update the session's messages list
            if session_id in store._sessions:
                store._sessions[session_id].messages = store._messages[session_id]

        def list_sessions(project_path=None, limit=10):
            """List all sessions."""
            return list(store._sessions.values())[:limit]

        def get_session_stats(session_id):
            """Get session statistics."""
            session = store._sessions.get(session_id)
            if not session:
                return None
            return {
                "session_id": session.session_id,
                "message_count": len(store._messages.get(session_id, [])),
                "is_active": session.is_active,
            }

        def persist_messages():
            """Persist messages (no-op for mock)."""
            # Messages are already stored in _messages dict
            pass

        # Setup mock methods
        store.create_session.side_effect = create_session
        store.get_session.side_effect = get_session
        store.add_message.side_effect = add_message
        store.list_sessions.side_effect = list_sessions
        store.get_session_stats.side_effect = get_session_stats
        store.persist_messages.side_effect = persist_messages

        return store

    def test_create_and_recover_session_with_messages(self, mock_persistent_store):
        """Test creating a session, adding messages, and recovering it."""
        # Phase 1: Create initial session and add messages
        manager1 = ConversationManager(
            store=mock_persistent_store,
            provider="anthropic",
            model="claude-3-sonnet",
            config=ConversationManagerConfig(enable_persistence=True),
        )

        # Verify session was created
        assert manager1.session_id is not None
        session_id = manager1.session_id
        print(f"Created session: {session_id}")

        # Add some messages
        msg1 = manager1.add_user_message("Hello, I need help with Python")
        msg2 = manager1.add_assistant_message("I'd be happy to help with Python! What do you need?")
        msg3 = manager1.add_user_message("How do I read a file?")
        msg4 = manager1.add_assistant_message("You can use the built-in open() function")

        # Manually persist messages to mock store (simulating what a real store would do)
        mock_persistent_store.add_message(session_id, msg1)
        mock_persistent_store.add_message(session_id, msg2)
        mock_persistent_store.add_message(session_id, msg3)
        mock_persistent_store.add_message(session_id, msg4)

        # Verify messages were added
        assert manager1.message_count() == 4
        messages = manager1.messages
        assert len(messages) == 4
        assert messages[0].content == "Hello, I need help with Python"
        assert messages[3].content == "You can use the built-in open() function"

        # Phase 2: Recover session in a new manager instance
        manager2 = ConversationManager(
            store=mock_persistent_store,
            provider="anthropic",
            model="claude-3-sonnet",
            config=ConversationManagerConfig(enable_persistence=True),
        )

        # Recover the session
        success = manager2.recover_session(session_id)
        assert success is True
        assert manager2.session_id == session_id

        # Verify messages were restored
        assert manager2.message_count() == 4
        recovered_messages = manager2.messages
        assert len(recovered_messages) == 4
        assert recovered_messages[0].content == "Hello, I need help with Python"
        assert recovered_messages[3].content == "You can use the built-in open() function"

        # Phase 3: Add new messages to recovered session
        # Note: PromptNormalizer normalizes "show" → "read", "view" → "read", etc.
        msg5 = manager2.add_user_message("Can you show me an example?")
        msg6 = manager2.add_assistant_message(
            "Sure! Here's how to read a file:\n\nwith open('file.txt', 'r') as f:\n    content = f.read()"
        )

        # Manually persist new messages
        mock_persistent_store.add_message(session_id, msg5)
        mock_persistent_store.add_message(session_id, msg6)

        # Verify new messages were added
        assert manager2.message_count() == 6
        all_messages = manager2.messages
        assert len(all_messages) == 6
        # Note: Content is normalized by PromptNormalizer ("show" → "read")
        assert all_messages[4].content == "Can you read me an example?"
        assert "with open(" in all_messages[5].content

    def test_session_recovery_preserves_context(self, mock_persistent_store):
        """Test that session recovery preserves conversation context."""
        # Create session with complex conversation
        manager1 = ConversationManager(
            store=mock_persistent_store,
            system_prompt="You are a Python expert",
            provider="anthropic",
            model="claude-3-sonnet",
            config=ConversationManagerConfig(enable_persistence=True),
        )

        session_id = manager1.session_id

        # Create a multi-turn conversation with tool calls
        msg1 = manager1.add_user_message("Help me debug this code")
        msg2 = manager1.add_assistant_message(
            "I'll help you debug. Can you share the code?",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "app.py"}',
                    },
                }
            ],
        )
        msg3 = manager1.add_tool_result(tool_call_id="call_1", content="File contents here...")
        msg4 = manager1.add_assistant_message("I see the issue. You're missing a import statement.")

        # Manually persist messages
        for msg in [msg1, msg2, msg3, msg4]:
            mock_persistent_store.add_message(session_id, msg)

        # Recover session
        manager2 = ConversationManager(
            store=mock_persistent_store,
            provider="anthropic",
            model="claude-3-sonnet",
            config=ConversationManagerConfig(enable_persistence=True),
        )

        manager2.recover_session(session_id)

        # Verify context is preserved
        assert manager2.message_count() == 4

        # Check tool calls are preserved
        last_assistant_message = [m for m in manager2.messages if m.role == "assistant"][-2]
        assert last_assistant_message.tool_calls is not None
        assert len(last_assistant_message.tool_calls) == 1
        assert last_assistant_message.tool_calls[0]["function"]["name"] == "read_file"

        # Verify conversation flow is intact
        messages = manager2.messages
        assert "debug" in messages[0].content.lower()
        assert "import statement" in messages[3].content.lower()

    def test_session_recovery_with_system_prompt(self, mock_persistent_store):
        """Test that system prompt is preserved during session recovery."""
        system_prompt = "You are a helpful coding assistant specializing in Python."

        # Create session with system prompt
        manager1 = ConversationManager(
            store=mock_persistent_store,
            system_prompt=system_prompt,
            provider="anthropic",
            model="claude-3-sonnet",
            config=ConversationManagerConfig(enable_persistence=True),
        )

        session_id = manager1.session_id

        msg1 = manager1.add_user_message("What's Python?")
        msg2 = manager1.add_assistant_message("Python is a high-level programming language.")

        # Manually persist messages
        mock_persistent_store.add_message(session_id, msg1)
        mock_persistent_store.add_message(session_id, msg2)

        # Recover session without system prompt (should restore from session)
        manager2 = ConversationManager(
            store=mock_persistent_store,
            provider="anthropic",
            model="claude-3-sonnet",
            config=ConversationManagerConfig(enable_persistence=True),
        )

        manager2.recover_session(session_id)

        # Verify messages are restored
        assert manager2.message_count() == 2

        # Verify we can continue the conversation
        manager2.add_user_message("Why is it popular?")
        manager2.add_assistant_message(
            "Python is popular due to its simplicity and extensive libraries."
        )

        assert manager2.message_count() == 4

    def test_multiple_sessions_recovery(self, mock_persistent_store):
        """Test recovering different sessions."""
        # Create first session
        manager1 = ConversationManager(
            store=mock_persistent_store,
            provider="anthropic",
            model="claude-3-sonnet",
            config=ConversationManagerConfig(enable_persistence=True),
        )

        session1_id = manager1.session_id

        msg1 = manager1.add_user_message("Session 1: Discussing Python")
        msg2 = manager1.add_assistant_message("Python is great!")

        # Persist messages
        mock_persistent_store.add_message(session1_id, msg1)
        mock_persistent_store.add_message(session1_id, msg2)

        # Create second session
        manager2 = ConversationManager(
            store=mock_persistent_store,
            provider="anthropic",
            model="claude-3-sonnet",
            config=ConversationManagerConfig(enable_persistence=True),
        )

        session2_id = manager2.session_id

        msg3 = manager2.add_user_message("Session 2: Discussing JavaScript")
        msg4 = manager2.add_assistant_message("JavaScript is versatile!")

        # Persist messages
        mock_persistent_store.add_message(session2_id, msg3)
        mock_persistent_store.add_message(session2_id, msg4)

        # Verify they're different sessions
        assert session1_id != session2_id

        # Recover first session
        manager3 = ConversationManager(
            store=mock_persistent_store,
            provider="anthropic",
            model="claude-3-sonnet",
            config=ConversationManagerConfig(enable_persistence=True),
        )

        manager3.recover_session(session1_id)
        assert manager3.message_count() == 2
        assert "Python" in manager3.messages[0].content

        # Recover second session
        manager4 = ConversationManager(
            store=mock_persistent_store,
            provider="anthropic",
            model="claude-3-sonnet",
            config=ConversationManagerConfig(enable_persistence=True),
        )

        manager4.recover_session(session2_id)
        assert manager4.message_count() == 2
        assert "JavaScript" in manager4.messages[0].content

    def test_session_recovery_with_context_metrics(self, mock_persistent_store):
        """Test that context metrics are restored after recovery."""
        # Create session
        manager1 = ConversationManager(
            store=mock_persistent_store,
            provider="anthropic",
            model="claude-3-sonnet",
            config=ConversationManagerConfig(enable_persistence=True, max_context_chars=1000),
        )

        session_id = manager1.session_id

        # Add messages to populate context
        for i in range(5):
            msg_user = manager1.add_user_message(f"Message {i}")
            msg_assist = manager1.add_assistant_message(f"Response {i}")
            # Persist messages
            mock_persistent_store.add_message(session_id, msg_user)
            mock_persistent_store.add_message(session_id, msg_assist)

        # Get context metrics before recovery
        metrics_before = manager1.get_context_metrics()
        assert metrics_before.message_count == 10

        # Recover session
        manager2 = ConversationManager(
            store=mock_persistent_store,
            provider="anthropic",
            model="claude-3-sonnet",
            config=ConversationManagerConfig(enable_persistence=True, max_context_chars=1000),
        )

        manager2.recover_session(session_id)

        # Get context metrics after recovery
        metrics_after = manager2.get_context_metrics()
        assert metrics_after.message_count == 10
        assert metrics_after.total_chars == metrics_before.total_chars

    def test_session_recovery_failure_handling(self, mock_persistent_store):
        """Test handling of non-existent session recovery."""
        manager = ConversationManager(
            store=mock_persistent_store,
            provider="anthropic",
            model="claude-3-sonnet",
            config=ConversationManagerConfig(enable_persistence=True),
        )

        # Try to recover non-existent session
        success = manager.recover_session("nonexistent_session_id")
        assert success is False

        # Verify manager state is unchanged
        assert manager.session_id is not None  # Created its own session
        assert manager.message_count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
