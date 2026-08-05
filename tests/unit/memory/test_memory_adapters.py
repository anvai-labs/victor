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

"""Tests for memory adapters.

Tests cover:
- EntityMemoryAdapter
- ConversationMemoryAdapter
- GraphMemoryAdapter
- ToolResultsMemoryAdapter
- Factory functions
"""

import pytest
from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from victor.storage.memory.unified import MemoryQuery, MemoryResult, MemoryType
from victor.storage.memory.adapters import (
    EntityMemoryAdapter,
    ConversationMemoryAdapter,
    GraphMemoryAdapter,
    ToolResultsMemoryAdapter,
    create_entity_adapter,
    create_conversation_adapter,
    create_graph_adapter,
    create_tool_results_adapter,
)

# =============================================================================
# Mock Entity Memory
# =============================================================================


class MockEntity:
    """Mock entity for testing."""

    def __init__(
        self,
        id: str,
        name: str,
        entity_type_value: str = "class",
        description: str = "",
        source: str = "test.py",
        confidence: float = 1.0,
        mentions: int = 1,
    ):
        self.id = id
        self.name = name
        self.entity_type = MagicMock()
        self.entity_type.value = entity_type_value
        self.description = description
        self.attributes = {}
        self.source = source
        self.confidence = confidence
        self.mentions = mentions
        self.first_seen = datetime.now(timezone.utc)
        self.last_seen = datetime.now(timezone.utc)


class MockEntityRelation:
    """Mock entity relation for testing."""

    def __init__(
        self,
        id: str,
        source_id: str,
        target_id: str,
        relation_type_value: str = "imports",
        strength: float = 1.0,
    ):
        self.id = id
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = MagicMock()
        self.relation_type.value = relation_type_value
        self.strength = strength
        self.first_seen = datetime.now(timezone.utc)
        self.last_seen = datetime.now(timezone.utc)


class MockEntityMemory:
    """Mock EntityMemory for testing."""

    def __init__(self):
        self._initialized = True
        self._entities = {}
        self._relations = {}

    async def search(
        self,
        query: str,
        entity_types: Optional[List] = None,
        limit: int = 10,
    ) -> List[MockEntity]:
        results = []
        for entity in self._entities.values():
            if query.lower() in entity.name.lower():
                results.append(entity)
        return results[:limit]

    async def get(self, entity_id: str) -> Optional[MockEntity]:
        return self._entities.get(entity_id)

    async def store(self, entity: MockEntity) -> str:
        self._entities[entity.id] = entity
        return entity.id

    async def get_related(
        self,
        entity_id: str,
        relation_types: Optional[List] = None,
        direction: str = "both",
        limit: int = 20,
    ) -> List[tuple]:
        results = []
        for rel in self._relations.values():
            if rel.source_id == entity_id or rel.target_id == entity_id:
                related_id = rel.target_id if rel.source_id == entity_id else rel.source_id
                if related_id in self._entities:
                    results.append((self._entities[related_id], rel))
        return results[:limit]

    async def store_relation(self, relation: MockEntityRelation) -> str:
        self._relations[relation.id] = relation
        return relation.id


# =============================================================================
# Mock Conversation Memory
# =============================================================================


class MockConversationMessage:
    """Mock conversation message for testing."""

    def __init__(
        self,
        id: str,
        role_value: str,
        content: str,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        priority_value: int = 50,
        token_count: int = 10,
        metadata: Optional[dict] = None,
    ):
        self.id = id
        self.role = MagicMock()
        self.role.value = role_value
        self.content = content
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.priority = MagicMock()
        self.priority.value = priority_value
        self.token_count = token_count
        self.timestamp = datetime.now(timezone.utc)
        self.metadata = metadata or {}


class MockConversationStore:
    """Mock ConversationStore for testing."""

    def __init__(self):
        self._sessions = {}
        self._messages = {}

    def get_semantically_relevant_messages(
        self,
        session_id: str,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.3,
    ) -> List[tuple]:
        if session_id not in self._messages:
            return []

        results = []
        for msg in self._messages[session_id]:
            if query.lower() in msg.content.lower():
                similarity = 0.8 if query.lower() in msg.content.lower() else 0.5
                results.append((msg, similarity))
        return results[:limit]

    def get_recent_messages(
        self,
        session_id: str,
        count: int = 10,
    ) -> List[MockConversationMessage]:
        if session_id not in self._messages:
            return []
        return self._messages[session_id][:count]

    def get_historical_tool_results(
        self,
        session_id: str,
        tool_names: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[MockConversationMessage]:
        if session_id not in self._messages:
            return []

        results = []
        for msg in self._messages[session_id]:
            if msg.role.value == "tool_result":
                if tool_names is None or msg.tool_name in tool_names:
                    results.append(msg)
        return results[:limit]

    def add_message(
        self,
        session_id: str,
        role,
        content: str,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        if session_id not in self._messages:
            self._messages[session_id] = []
        msg = MockConversationMessage(
            id=f"msg_{len(self._messages[session_id])}",
            role_value=role.value if hasattr(role, "value") else role,
            content=content,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )
        self._messages[session_id].append(msg)


# =============================================================================
# Test EntityMemoryAdapter
# =============================================================================


class TestEntityMemoryAdapter:
    """Tests for EntityMemoryAdapter."""

    @pytest.fixture
    def entity_memory(self):
        """Create mock entity memory with sample data."""
        mem = MockEntityMemory()
        mem._entities = {
            "ent_1": MockEntity("ent_1", "UserAuth", "class", "User authentication"),
            "ent_2": MockEntity("ent_2", "AuthMiddleware", "class", "Auth middleware"),
            "ent_3": MockEntity("ent_3", "login", "function", "Login function"),
        }
        return mem

    @pytest.fixture
    def adapter(self, entity_memory):
        """Create adapter with mock memory."""
        return EntityMemoryAdapter(entity_memory)

    def test_memory_type(self, adapter):
        """Test memory type property."""
        assert adapter.memory_type == MemoryType.ENTITY

    @pytest.mark.asyncio
    async def test_search_finds_matching_entities(self, adapter):
        """Test search finds entities by name."""
        query = MemoryQuery(query="Auth", limit=10)
        results = await adapter.search(query)

        assert len(results) == 2
        assert all(r.source == MemoryType.ENTITY for r in results)
        names = [r.content["name"] for r in results]
        assert "UserAuth" in names
        assert "AuthMiddleware" in names

    @pytest.mark.asyncio
    async def test_search_relevance_scoring(self, adapter):
        """Test that relevance scoring works."""
        query = MemoryQuery(query="UserAuth", limit=10)
        results = await adapter.search(query)

        # Exact match should have highest relevance
        assert results[0].content["name"] == "UserAuth"
        assert results[0].relevance == 1.0

    @pytest.mark.asyncio
    async def test_search_no_results(self, adapter):
        """Test search with no matches."""
        query = MemoryQuery(query="nonexistent", limit=10)
        results = await adapter.search(query)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_existing_entity(self, adapter, entity_memory):
        """Test getting an existing entity."""
        result = await adapter.get("ent_1")

        assert result is not None
        assert result.id == "ent_1"
        assert result.content["name"] == "UserAuth"

    @pytest.mark.asyncio
    async def test_get_nonexistent_entity(self, adapter):
        """Test getting a nonexistent entity."""
        result = await adapter.get("nonexistent")
        assert result is None

    def test_is_available(self, adapter):
        """Test availability check."""
        assert adapter.is_available() is True

    def test_is_available_when_not_initialized(self, entity_memory):
        """Test availability when memory not initialized."""
        entity_memory._initialized = False
        adapter = EntityMemoryAdapter(entity_memory)
        assert adapter.is_available() is False


# =============================================================================
# Test ConversationMemoryAdapter
# =============================================================================


class TestConversationMemoryAdapter:
    """Tests for ConversationMemoryAdapter."""

    @pytest.fixture
    def conversation_store(self):
        """Create mock conversation store with sample data."""
        store = MockConversationStore()
        store._messages["session_1"] = [
            MockConversationMessage("msg_1", "user", "Fix the authentication bug"),
            MockConversationMessage("msg_2", "assistant", "I found the auth issue"),
            MockConversationMessage(
                "msg_3",
                "tool_result",
                "File contents here",
                tool_name="read_file",
                tool_call_id="tc_1",
            ),
        ]
        return store

    @pytest.fixture
    def adapter(self, conversation_store):
        """Create adapter with mock store."""
        return ConversationMemoryAdapter(conversation_store, session_id="session_1")

    def test_memory_type(self, adapter):
        """Test memory type property."""
        assert adapter.memory_type == MemoryType.CONVERSATION

    def test_set_session_id(self, adapter):
        """Test setting session ID."""
        adapter.set_session_id("new_session")
        assert adapter._session_id == "new_session"

    @pytest.mark.asyncio
    async def test_search_finds_matching_messages(self, adapter):
        """Test search finds messages by content."""
        query = MemoryQuery(query="auth", limit=10, session_id="session_1")
        results = await adapter.search(query)

        assert len(results) >= 1
        assert all(r.source == MemoryType.CONVERSATION for r in results)

    @pytest.mark.asyncio
    async def test_search_no_session_id(self, conversation_store):
        """Test search without session ID."""
        adapter = ConversationMemoryAdapter(conversation_store)
        query = MemoryQuery(query="auth", limit=10)
        results = await adapter.search(query)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_fallback_to_recent(self, conversation_store):
        """Test fallback to recent messages when semantic search fails."""
        adapter = ConversationMemoryAdapter(conversation_store, session_id="session_1")

        # Search for something that won't match semantically
        query = MemoryQuery(query="authentication", limit=10)
        results = await adapter.search(query)

        # Should fall back to keyword matching in recent messages
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_prefers_async_conversation_store_api(self):
        """Adapter should await the async semantic search API when available."""
        store = MockConversationStore()
        store.aget_semantically_relevant_messages = AsyncMock(
            return_value=[
                (
                    MockConversationMessage(
                        id="msg_async",
                        role_value="assistant",
                        content="Authentication middleware failed",
                    ),
                    0.91,
                )
            ]
        )
        adapter = ConversationMemoryAdapter(store, session_id="session_async")

        results = await adapter.search(MemoryQuery(query="auth", limit=5))

        assert len(results) == 1
        assert results[0].id == "msg_async"
        store.aget_semantically_relevant_messages.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_supports_dual_trace_filters(self, conversation_store):
        """Adapter should expose separate semantic and execution traces when requested."""
        adapter = ConversationMemoryAdapter(conversation_store, session_id="session_1")
        semantic_message = MockConversationMessage(
            "msg_semantic",
            "assistant",
            "Authentication middleware failed during startup.",
            metadata={"memory_trace_kind": "semantic"},
        )
        execution_message = MockConversationMessage(
            "msg_execution",
            "tool",
            '<TOOL_OUTPUT tool="read" path="src/auth.py">...</TOOL_OUTPUT>',
            tool_name="read",
            tool_call_id="call_1",
            metadata={
                "memory_trace_kind": "execution",
                "memory_execution_text": "tool read path src/auth.py config loader",
            },
        )
        conversation_store.aget_dual_trace_relevant_messages = AsyncMock(
            return_value={
                "semantic": [(semantic_message, 0.92)],
                "execution": [(execution_message, 0.81)],
            }
        )
        conversation_store.get_message_trace_text = MagicMock(
            side_effect=lambda message, trace_kind=None: (
                message.metadata.get("memory_execution_text") or message.content
            )
        )

        results = await adapter.search(
            MemoryQuery(
                query="authentication config",
                limit=5,
                session_id="session_1",
                filters={"dual_trace": True},
            )
        )

        assert [result.metadata["trace_kind"] for result in results] == [
            "semantic",
            "execution",
        ]
        assert results[1].content["trace_text"] == "tool read path src/auth.py config loader"
        conversation_store.aget_dual_trace_relevant_messages.assert_awaited_once()

    def test_is_available(self, adapter):
        """Test availability check."""
        assert adapter.is_available() is True


# =============================================================================
# Test GraphMemoryAdapter
# =============================================================================


class TestGraphMemoryAdapter:
    """Tests for GraphMemoryAdapter."""

    @pytest.fixture
    def entity_memory(self):
        """Create mock entity memory with relations."""
        mem = MockEntityMemory()
        mem._entities = {
            "ent_1": MockEntity("ent_1", "UserAuth", "class"),
            "ent_2": MockEntity("ent_2", "AuthService", "class"),
        }
        mem._relations = {
            "rel_1": MockEntityRelation("rel_1", "ent_1", "ent_2", "imports"),
        }
        return mem

    @pytest.fixture
    def adapter(self, entity_memory):
        """Create adapter with mock memory."""
        return GraphMemoryAdapter(entity_memory)

    def test_memory_type(self, adapter):
        """Test memory type property."""
        assert adapter.memory_type == MemoryType.GRAPH

    @pytest.mark.asyncio
    async def test_search_finds_related_entities(self, adapter):
        """Test search finds related entities."""
        query = MemoryQuery(query="ent_1", limit=10)
        results = await adapter.search(query)

        assert len(results) == 1
        assert results[0].source == MemoryType.GRAPH
        assert "entity" in results[0].content
        assert "relation" in results[0].content

    @pytest.mark.asyncio
    async def test_search_no_relations(self, adapter):
        """Test search with no relations."""
        query = MemoryQuery(query="nonexistent", limit=10)
        results = await adapter.search(query)
        assert len(results) == 0

    def test_is_available(self, adapter):
        """Test availability check."""
        assert adapter.is_available() is True


# =============================================================================
# Test ToolResultsMemoryAdapter
# =============================================================================


class TestToolResultsMemoryAdapter:
    """Tests for ToolResultsMemoryAdapter."""

    @pytest.fixture
    def conversation_store(self):
        """Create mock store with tool results."""
        store = MockConversationStore()
        store._messages["session_1"] = [
            MockConversationMessage(
                "msg_1",
                "tool_result",
                "config.py contents here",
                tool_name="read_file",
                tool_call_id="tc_1",
            ),
            MockConversationMessage(
                "msg_2",
                "tool_result",
                "search results for auth",
                tool_name="code_search",
                tool_call_id="tc_2",
            ),
        ]
        return store

    @pytest.fixture
    def adapter(self, conversation_store):
        """Create adapter with mock store."""
        return ToolResultsMemoryAdapter(conversation_store, session_id="session_1")

    def test_memory_type(self, adapter):
        """Test memory type property."""
        assert adapter.memory_type == MemoryType.CODE

    @pytest.mark.asyncio
    async def test_search_finds_tool_results(self, adapter):
        """Test search finds matching tool results."""
        query = MemoryQuery(query="config", limit=10)
        results = await adapter.search(query)

        assert len(results) >= 1
        assert all(r.source == MemoryType.CODE for r in results)

    @pytest.mark.asyncio
    async def test_search_with_tool_filter(self, adapter):
        """Test search with tool name filter."""
        query = MemoryQuery(
            query="",
            limit=10,
            filters={"tool_names": ["read_file"]},
        )
        results = await adapter.search(query)

        assert all(r.content["tool_name"] == "read_file" for r in results)


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_entity_adapter(self):
        """Test creating entity adapter via factory."""
        mem = MockEntityMemory()
        adapter = create_entity_adapter(mem)

        assert isinstance(adapter, EntityMemoryAdapter)
        assert adapter.memory_type == MemoryType.ENTITY

    def test_create_conversation_adapter(self):
        """Test creating conversation adapter via factory."""
        store = MockConversationStore()
        adapter = create_conversation_adapter(store, session_id="test")

        assert isinstance(adapter, ConversationMemoryAdapter)
        assert adapter.memory_type == MemoryType.CONVERSATION

    def test_create_graph_adapter(self):
        """Test creating graph adapter via factory."""
        mem = MockEntityMemory()
        adapter = create_graph_adapter(mem)

        assert isinstance(adapter, GraphMemoryAdapter)
        assert adapter.memory_type == MemoryType.GRAPH

    def test_create_tool_results_adapter(self):
        """Test creating tool results adapter via factory."""
        store = MockConversationStore()
        adapter = create_tool_results_adapter(store, session_id="test")

        assert isinstance(adapter, ToolResultsMemoryAdapter)
        assert adapter.memory_type == MemoryType.CODE
