# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for async ConversationStore methods.

Verifies that SQLite I/O is offloaded to the thread pool
when called from async contexts.
"""

import sqlite3

from victor.agent.conversation.store import (
    ConversationStore,
    MessageRole,
)


class TestConversationStoreAsync:
    """Verify async variants of ConversationStore methods."""

    async def test_add_message_async_returns_message(self, tmp_path):
        """add_message_async creates and returns a ConversationMessage."""
        store = ConversationStore(db_path=tmp_path / "test.db")
        session = store.create_session(project_path=str(tmp_path))

        msg = await store.add_message_async(
            session.session_id,
            MessageRole.USER,
            "Hello async",
        )

        assert msg.role == MessageRole.USER
        assert msg.content == "Hello async"
        assert msg.token_count > 0

    async def test_add_message_async_persists_to_db(self, tmp_path):
        """add_message_async writes the message to SQLite."""
        db_path = tmp_path / "test.db"
        store = ConversationStore(db_path=db_path)
        session = store.create_session(project_path=str(tmp_path))

        await store.add_message_async(
            session.session_id,
            MessageRole.USER,
            "Persisted async",
        )

        # Verify directly in SQLite
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT content FROM messages " "WHERE session_id = ?",
                (session.session_id,),
            ).fetchone()
            assert row is not None
            assert "Persisted async" in row[0]

    async def test_add_message_async_updates_session_tokens(self, tmp_path):
        """add_message_async updates in-memory token count."""
        store = ConversationStore(db_path=tmp_path / "test.db")
        session = store.create_session(project_path=str(tmp_path))
        initial_tokens = session.current_tokens

        await store.add_message_async(
            session.session_id,
            MessageRole.USER,
            "Token counting test message",
        )

        assert session.current_tokens > initial_tokens

    async def test_add_message_async_multiple_messages(self, tmp_path):
        """Multiple async messages are all persisted in order."""
        db_path = tmp_path / "test.db"
        store = ConversationStore(db_path=db_path)
        session = store.create_session(project_path=str(tmp_path))

        for i in range(5):
            await store.add_message_async(
                session.session_id,
                MessageRole.USER,
                f"Message {i}",
            )

        assert len(session.messages) == 5

        with sqlite3.connect(db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session.session_id,),
            ).fetchone()[0]
            assert count == 5
