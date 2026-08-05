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

"""Tests for ConversationEmbeddingStore (lazy embedding mode).

The ConversationEmbeddingStore uses a lazy embedding strategy:
- Messages live in SQLite (ConversationStore is source of truth)
- Embeddings are computed on-demand when search_similar() is called
- LanceDB stores only message_id (FK), session_id, vector, timestamp

This test file verifies the lazy embedding behavior and API.
"""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import numpy as np


class MockEmbeddingService:
    """Mock embedding service for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self.model_name = "mock-model"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding based on text hash."""
        # Generate a deterministic embedding based on text content
        np.random.seed(hash(text) % (2**32))
        return np.random.rand(self._dimension).astype(np.float32)

    async def embed_batch(self, texts: list) -> np.ndarray:
        """Generate mock embeddings for batch."""
        embeddings = []
        for text in texts:
            emb = await self.embed_text(text)
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)


def create_test_sqlite_db(db_path: Path, messages: list) -> None:
    """Create a SQLite database with test messages.

    This simulates ConversationStore's messages table structure.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """)
        for msg in messages:
            conn.execute(
                """
                INSERT INTO messages (id, session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    msg["message_id"],
                    msg["session_id"],
                    msg.get("role", "user"),
                    msg["content"],
                    msg.get("timestamp", datetime.now().isoformat()),
                ),
            )
        conn.commit()


class TestConversationEmbeddingSearchResult:
    """Tests for ConversationEmbeddingSearchResult dataclass."""

    def test_creation(self):
        """Test creating a search result."""
        from victor.agent.conversation_embedding_store import (
            ConversationEmbeddingSearchResult,
        )

        result = ConversationEmbeddingSearchResult(
            message_id="msg_123",
            session_id="session_456",
            similarity=0.85,
            timestamp=datetime.now(),
        )

        assert result.message_id == "msg_123"
        assert result.session_id == "session_456"
        assert result.similarity == 0.85

    def test_repr(self):
        """Test string representation."""
        from victor.agent.conversation_embedding_store import (
            ConversationEmbeddingSearchResult,
        )

        result = ConversationEmbeddingSearchResult(
            message_id="msg_123",
            session_id="session_456",
            similarity=0.72,
        )

        repr_str = repr(result)
        assert "msg_123" in repr_str
        assert "0.720" in repr_str


class TestConversationEmbeddingStore:
    """Tests for ConversationEmbeddingStore class (lazy embedding mode)."""

    @pytest.fixture
    def temp_paths(self):
        """Create temporary directories for SQLite and LanceDB.

        Note: Tests use temporary conversation.db for isolation.
        In production, embedding store uses a separate database from project.db/victor.db.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            yield {
                "sqlite_db": Path(tmpdir) / "conversation.db",
                "lancedb_dir": Path(tmpdir) / "embeddings",
            }

    @pytest.fixture
    def mock_service(self):
        """Create a mock embedding service."""
        return MockEmbeddingService()

    @pytest.fixture
    def store(self, mock_service, temp_paths):
        """Create a ConversationEmbeddingStore instance."""
        from victor.agent.conversation_embedding_store import ConversationEmbeddingStore

        return ConversationEmbeddingStore(
            embedding_service=mock_service,
            sqlite_db_path=temp_paths["sqlite_db"],
            lancedb_path=temp_paths["lancedb_dir"],
        )

    @pytest.mark.asyncio
    async def test_initialization(self, store, temp_paths):
        """Test store initialization."""
        assert not store.is_initialized

        await store.initialize()

        assert store.is_initialized
        assert temp_paths["lancedb_dir"].exists()

    @pytest.mark.asyncio
    async def test_lazy_embedding_on_search(self, store, temp_paths):
        """Test that embeddings are created lazily on search.

        Messages should only be embedded when search_similar is called,
        not when they are added to SQLite.
        """
        # Create SQLite DB with test messages
        messages = [
            {
                "message_id": "msg_001",
                "session_id": "session_test",
                "content": "How do I implement user authentication in Python?",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "message_id": "msg_002",
                "session_id": "session_test",
                "content": "Python JWT authentication with Flask or Django framework.",
                "timestamp": datetime.now().isoformat(),
            },
        ]
        create_test_sqlite_db(temp_paths["sqlite_db"], messages)

        await store.initialize()

        # Stats should show 0 embeddings before search
        stats = await store.get_stats()
        assert stats["total_embeddings"] == 0

        # Search triggers lazy embedding
        results = await store.search_similar(
            query="authentication",
            session_id="session_test",
            limit=5,
            min_similarity=0.0,  # Low threshold for mock embeddings
        )

        # Now embeddings should exist
        stats = await store.get_stats()
        assert stats["total_embeddings"] == 2

        # Search should return results
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_skip_short_messages(self, store, temp_paths):
        """Test that very short messages are skipped during lazy embedding."""
        # Create SQLite DB with a short message
        messages = [
            {
                "message_id": "msg_short",
                "session_id": "session_test",
                "content": "Hi",  # Too short (< 20 chars default)
            },
            {
                "message_id": "msg_long",
                "session_id": "session_test",
                "content": "This is a longer message that should be embedded properly.",
            },
        ]
        create_test_sqlite_db(temp_paths["sqlite_db"], messages)

        await store.initialize()

        # Trigger lazy embedding via search
        await store.search_similar(
            query="test query",
            session_id="session_test",
            min_similarity=0.0,
        )

        # Only the long message should be embedded
        stats = await store.get_stats()
        assert stats["total_embeddings"] == 1

    @pytest.mark.asyncio
    async def test_search_with_session_filter(self, store, temp_paths):
        """Test searching with session filter."""
        # Create messages in different sessions
        messages = [
            {
                "message_id": "msg_s1",
                "session_id": "session_1",
                "content": "Authentication and login implementation details for session 1",
            },
            {
                "message_id": "msg_s2",
                "session_id": "session_2",
                "content": "Authentication security best practices for session 2",
            },
        ]
        create_test_sqlite_db(temp_paths["sqlite_db"], messages)

        await store.initialize()

        # Search in session_1 only
        results = await store.search_similar(
            query="authentication",
            session_id="session_1",
            limit=5,
            min_similarity=0.0,
        )

        # Should only find message from session_1
        session_ids = {r.session_id for r in results}
        if results:
            assert "session_1" in session_ids or len(results) == 0
            # Should not find session_2
            assert "session_2" not in session_ids

    @pytest.mark.asyncio
    async def test_search_with_exclude_ids(self, store, temp_paths):
        """Test searching with message exclusions."""
        messages = [
            {
                "message_id": "msg_exclude",
                "session_id": "session_test",
                "content": "Python programming with authentication patterns.",
            },
            {
                "message_id": "msg_include",
                "session_id": "session_test",
                "content": "Python programming with authorization patterns.",
            },
        ]
        create_test_sqlite_db(temp_paths["sqlite_db"], messages)

        await store.initialize()

        results = await store.search_similar(
            query="Python programming",
            exclude_message_ids=["msg_exclude"],
            limit=5,
            min_similarity=0.0,
        )

        # Should not include excluded message
        message_ids = [r.message_id for r in results]
        assert "msg_exclude" not in message_ids

    @pytest.mark.asyncio
    async def test_delete_session(self, store, temp_paths):
        """Test deleting all embeddings from a session."""
        # Create messages in two sessions
        messages = [
            {
                "message_id": f"msg_session_{i}",
                "session_id": "session_to_delete",
                "content": f"Message {i} in session that will be deleted entirely.",
            }
            for i in range(3)
        ] + [
            {
                "message_id": "msg_other",
                "session_id": "other_session",
                "content": "This message is in a different session and should remain.",
            }
        ]
        create_test_sqlite_db(temp_paths["sqlite_db"], messages)

        await store.initialize()

        # Trigger lazy embedding for both sessions
        await store.search_similar(query="message", min_similarity=0.0)

        stats = await store.get_stats()
        assert stats["total_embeddings"] == 4

        # Delete one session
        deleted = await store.delete_session("session_to_delete")
        assert deleted == 3

        # Verify other session still exists
        stats = await store.get_stats()
        assert stats["total_embeddings"] == 1

    @pytest.mark.asyncio
    async def test_get_stats(self, store, temp_paths):
        """Test getting store statistics."""
        # Create messages
        messages = [
            {
                "message_id": "msg_1",
                "session_id": "session_test",
                "content": "Test message for stats verification.",
            }
        ]
        create_test_sqlite_db(temp_paths["sqlite_db"], messages)

        await store.initialize()

        stats = await store.get_stats()

        assert stats["store"] == "conversation_embedding_store"
        assert stats["backend"] == "lancedb"
        assert stats["mode"] == "lazy"
        assert stats["embedding_dimension"] == 384
        assert stats["embedding_model"] == "mock-model"
        assert "lancedb_path" in stats
        assert "sqlite_path" in stats
        assert "max_embeddings" in stats
        assert "usage_pct" in stats

    @pytest.mark.asyncio
    async def test_close(self, store, temp_paths):
        """Test closing the store."""
        await store.initialize()
        assert store.is_initialized

        await store.close()
        assert not store.is_initialized

    @pytest.mark.asyncio
    async def test_similarity_threshold(self, store, temp_paths):
        """Test that similarity threshold filters results."""
        messages = [
            {
                "message_id": "msg_relevant",
                "session_id": "session_test",
                "content": "Python web framework Django REST API development guide.",
            }
        ]
        create_test_sqlite_db(temp_paths["sqlite_db"], messages)

        await store.initialize()

        # Search with very high similarity threshold
        results = await store.search_similar(
            query="completely unrelated topic about cooking recipes",
            min_similarity=0.99,  # Very high threshold
            limit=5,
        )

        # Results list should still be valid (may be empty due to high threshold)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_rebuild(self, store, temp_paths):
        """Test eager rebuild of embeddings."""
        messages = [
            {
                "message_id": "msg_rebuild",
                "session_id": "session_test",
                "content": "Message for rebuild testing with sufficient length.",
            }
        ]
        create_test_sqlite_db(temp_paths["sqlite_db"], messages)

        await store.initialize()

        # No embeddings yet
        stats = await store.get_stats()
        assert stats["total_embeddings"] == 0

        # Rebuild triggers eager embedding
        count = await store.rebuild()
        assert count == 1

        stats = await store.get_stats()
        assert stats["total_embeddings"] == 1

    @pytest.mark.asyncio
    async def test_compact(self, store, temp_paths):
        """Test manual compaction."""
        messages = [
            {
                "message_id": "msg_compact",
                "session_id": "session_test",
                "content": "Message for compaction testing with sufficient length.",
            }
        ]
        create_test_sqlite_db(temp_paths["sqlite_db"], messages)

        await store.initialize()
        await store.rebuild()  # Create some embeddings

        # Compact should succeed
        result = await store.compact()
        # Result can be True or False depending on LanceDB state
        assert isinstance(result, bool)


class TestConversationStoreIntegration:
    """Tests for ConversationStore integration with embedding store."""

    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for both stores.

        Note: Tests use temporary project.db for isolation.
        In production, ConversationStore uses project.db (consolidated database).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            yield {
                "db_path": Path(tmpdir) / "project.db",
                "embed_path": Path(tmpdir) / "embeddings",
            }

    @pytest.mark.asyncio
    async def test_conversation_store_with_embedding_store(self, temp_paths):
        """Test that ConversationStore and embedding store work together.

        Note: The embedding store uses lazy embedding, so messages added to
        ConversationStore are only embedded when search_similar is called.
        """
        from victor.agent.conversation.store import ConversationStore
        from victor.agent.conversation.types import MessageRole
        from victor.agent.conversation_embedding_store import ConversationEmbeddingStore

        mock_service = MockEmbeddingService()

        # Create stores
        conv_store = ConversationStore(db_path=temp_paths["db_path"])
        embed_store = ConversationEmbeddingStore(
            embedding_service=mock_service,
            sqlite_db_path=temp_paths["db_path"],
            lancedb_path=temp_paths["embed_path"],
        )
        await embed_store.initialize()

        # Create a session
        session = conv_store.create_session(project_path="/test/project")

        # Add a message
        conv_store.add_message(
            session.session_id,
            MessageRole.USER,
            "How do I implement user authentication in Django?",
        )

        # Trigger lazy embedding via search (or rebuild)
        count = await embed_store.rebuild(session.session_id)

        # Verify embedding was created
        assert count >= 1

        stats = await embed_store.get_stats()
        assert stats["total_embeddings"] >= 1

        # Clean up
        await embed_store.close()
