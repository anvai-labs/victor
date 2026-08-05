# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Entity Memory system.

Tests cover:
- Entity and EntityRelation dataclasses
- EntityMemory 4-tier architecture
- EntityGraph traversal and path finding
- Entity extractors (code, text, composite)
"""

import pytest
from datetime import datetime, timezone

from victor.storage.memory import (
    EntityType,
    Entity,
    EntityRelation,
    RelationType,
    EntityMemory,
    EntityMemoryConfig,
    EntityGraph,
    GraphPath,
    LRUCache,
    CodeEntityExtractor,
    TextEntityExtractor,
    CompositeExtractor,
    ExtractionResult,
    CODE_ENTITY_TYPES,
)


class TestEntity:
    """Tests for Entity dataclass."""

    def test_create_entity(self):
        """Test creating an entity with factory method."""
        entity = Entity.create(
            name="UserAuth",
            entity_type=EntityType.CLASS,
            description="Authentication class",
            source="conversation",
        )

        assert entity.name == "UserAuth"
        assert entity.entity_type == EntityType.CLASS
        assert entity.description == "Authentication class"
        assert entity.id.startswith("ent_")
        assert entity.confidence == 1.0
        assert entity.mentions == 1

    def test_entity_id_generation(self):
        """Test that same name/type generates same ID."""
        entity1 = Entity.create("TestFunc", EntityType.FUNCTION)
        entity2 = Entity.create("TestFunc", EntityType.FUNCTION)
        entity3 = Entity.create("TestFunc", EntityType.CLASS)

        # Same name and type should have same ID
        assert entity1.id == entity2.id
        # Different type should have different ID
        assert entity1.id != entity3.id

    def test_entity_serialization(self):
        """Test entity to_dict/from_dict."""
        entity = Entity.create(
            name="MyClass",
            entity_type=EntityType.CLASS,
            description="A test class",
            attributes={"module": "test"},
            confidence=0.8,
        )

        data = entity.to_dict()
        restored = Entity.from_dict(data)

        assert restored.name == entity.name
        assert restored.entity_type == entity.entity_type
        assert restored.description == entity.description
        assert restored.attributes == entity.attributes
        assert restored.confidence == entity.confidence

    def test_entity_merge(self):
        """Test merging two entity observations."""
        entity1 = Entity.create(
            name="Handler",
            entity_type=EntityType.CLASS,
            description="First observation",
            confidence=0.6,
        )
        entity1.mentions = 2

        entity2 = Entity.create(
            name="Handler",
            entity_type=EntityType.CLASS,
            description="Better description",
            confidence=0.9,
            attributes={"new_attr": "value"},
        )
        entity2.mentions = 3

        merged = entity1.merge_with(entity2)

        # Should use higher confidence description
        assert merged.description == "Better description"
        # Should merge attributes
        assert "new_attr" in merged.attributes
        # Should sum mentions
        assert merged.mentions == 5
        # Should take max confidence
        assert merged.confidence == 0.9


class TestEntityRelation:
    """Tests for EntityRelation dataclass."""

    def test_create_relation(self):
        """Test creating a relation."""
        relation = EntityRelation(
            source_id="ent_abc123",
            target_id="ent_def456",
            relation_type=RelationType.IMPORTS,
            strength=0.8,
        )

        assert relation.source_id == "ent_abc123"
        assert relation.target_id == "ent_def456"
        assert relation.relation_type == RelationType.IMPORTS
        assert relation.strength == 0.8
        assert relation.id.startswith("rel_")

    def test_relation_id_deterministic(self):
        """Test that relation ID is deterministic."""
        rel1 = EntityRelation(
            source_id="ent_a",
            target_id="ent_b",
            relation_type=RelationType.DEPENDS_ON,
        )
        rel2 = EntityRelation(
            source_id="ent_a",
            target_id="ent_b",
            relation_type=RelationType.DEPENDS_ON,
        )

        assert rel1.id == rel2.id

    def test_relation_serialization(self):
        """Test relation to_dict/from_dict."""
        relation = EntityRelation(
            source_id="ent_src",
            target_id="ent_tgt",
            relation_type=RelationType.EXTENDS,
            strength=0.75,
            attributes={"scope": "module"},
        )

        data = relation.to_dict()
        restored = EntityRelation.from_dict(data)

        assert restored.source_id == relation.source_id
        assert restored.target_id == relation.target_id
        assert restored.relation_type == relation.relation_type
        assert restored.strength == relation.strength
        assert restored.attributes == relation.attributes


class TestLRUCache:
    """Tests for LRU cache."""

    def test_cache_put_get(self):
        """Test basic put and get."""
        cache = LRUCache(max_size=3)
        entity = Entity.create("Test", EntityType.FUNCTION)

        cache.put("key1", entity)
        result = cache.get("key1")

        assert result == entity

    def test_cache_eviction(self):
        """Test LRU eviction when max size exceeded."""
        cache = LRUCache(max_size=2)

        e1 = Entity.create("One", EntityType.FUNCTION)
        e2 = Entity.create("Two", EntityType.FUNCTION)
        e3 = Entity.create("Three", EntityType.FUNCTION)

        cache.put("k1", e1)
        cache.put("k2", e2)
        cache.put("k3", e3)  # Should evict k1

        assert cache.get("k1") is None
        assert cache.get("k2") is not None
        assert cache.get("k3") is not None

    def test_cache_access_updates_order(self):
        """Test that accessing an item moves it to end."""
        cache = LRUCache(max_size=2)

        e1 = Entity.create("One", EntityType.FUNCTION)
        e2 = Entity.create("Two", EntityType.FUNCTION)
        e3 = Entity.create("Three", EntityType.FUNCTION)

        cache.put("k1", e1)
        cache.put("k2", e2)
        cache.get("k1")  # Access k1, making k2 least recent
        cache.put("k3", e3)  # Should evict k2

        assert cache.get("k1") is not None  # k1 still there
        assert cache.get("k2") is None  # k2 evicted
        assert cache.get("k3") is not None


class TestEntityMemory:
    """Tests for EntityMemory 4-tier system."""

    @pytest.fixture
    def memory(self):
        """Create in-memory EntityMemory."""
        return EntityMemory(session_id="test_session")

    @pytest.mark.asyncio
    async def test_store_and_get(self, memory):
        """Test storing and retrieving an entity."""
        entity = Entity.create(
            name="TestClass",
            entity_type=EntityType.CLASS,
            description="A test class",
        )

        entity_id = await memory.store(entity)
        retrieved = await memory.get(entity_id)

        assert retrieved is not None
        assert retrieved.name == "TestClass"
        assert retrieved.entity_type == EntityType.CLASS

    @pytest.mark.asyncio
    async def test_search_by_name(self, memory):
        """Test searching entities by name."""
        e1 = Entity.create("UserAuth", EntityType.CLASS)
        e2 = Entity.create("UserProfile", EntityType.CLASS)
        e3 = Entity.create("TokenValidator", EntityType.CLASS)

        await memory.store(e1)
        await memory.store(e2)
        await memory.store(e3)

        results = await memory.search("User")

        assert len(results) == 2
        names = {e.name for e in results}
        assert "UserAuth" in names
        assert "UserProfile" in names

    @pytest.mark.asyncio
    async def test_search_by_type(self, memory):
        """Test searching entities by type."""
        e1 = Entity.create("func1", EntityType.FUNCTION)
        e2 = Entity.create("Class1", EntityType.CLASS)
        e3 = Entity.create("func2", EntityType.FUNCTION)

        await memory.store(e1)
        await memory.store(e2)
        await memory.store(e3)

        results = await memory.search("", entity_types=[EntityType.FUNCTION])

        assert len(results) == 2
        assert all(e.entity_type == EntityType.FUNCTION for e in results)

    @pytest.mark.asyncio
    async def test_store_relation(self, memory):
        """Test storing and retrieving relations."""
        e1 = Entity.create("ClassA", EntityType.CLASS)
        e2 = Entity.create("ClassB", EntityType.CLASS)

        await memory.store(e1)
        await memory.store(e2)

        relation = EntityRelation(
            source_id=e1.id,
            target_id=e2.id,
            relation_type=RelationType.EXTENDS,
        )

        await memory.store_relation(relation)

        related = await memory.get_related(e1.id)
        assert len(related) == 1
        assert related[0][0].id == e2.id
        assert related[0][1].relation_type == RelationType.EXTENDS

    @pytest.mark.asyncio
    async def test_session_entities(self, memory):
        """Test getting session entities."""
        e1 = Entity.create("Entity1", EntityType.CONCEPT)
        e2 = Entity.create("Entity2", EntityType.CONCEPT)

        await memory.store(e1)
        await memory.store(e2)

        session_entities = await memory.get_session_entities()

        assert len(session_entities) == 2

    @pytest.mark.asyncio
    async def test_increment_mentions(self, memory):
        """Test incrementing mention count."""
        entity = Entity.create("TestEntity", EntityType.CONCEPT)
        await memory.store(entity)

        initial = await memory.get(entity.id)
        initial_mentions = initial.mentions

        await memory.increment_mentions(entity.id)

        updated = await memory.get(entity.id)
        # Mentions should increase (exact count depends on merge behavior)
        assert updated.mentions > initial_mentions

    @pytest.mark.asyncio
    async def test_get_stats(self, memory):
        """Test getting memory statistics."""
        await memory.store(Entity.create("E1", EntityType.CLASS))
        await memory.store(Entity.create("E2", EntityType.FUNCTION))

        stats = await memory.get_stats()

        assert stats["session_id"] == "test_session"
        assert stats["short_term_count"] == 2
        assert stats["working_memory_count"] == 2


class TestEntityGraph:
    """Tests for EntityGraph."""

    @pytest.fixture
    def graph(self):
        """Create in-memory entity graph."""
        return EntityGraph(in_memory=True)

    @pytest.mark.asyncio
    async def test_add_entity(self, graph):
        """Test adding entity to graph."""
        entity = Entity.create("TestNode", EntityType.CLASS)

        await graph.add_entity(entity)
        retrieved = await graph.get_entity(entity.id)

        assert retrieved is not None
        assert retrieved.name == "TestNode"

    @pytest.mark.asyncio
    async def test_add_relation(self, graph):
        """Test adding relation to graph."""
        e1 = Entity.create("Source", EntityType.CLASS)
        e2 = Entity.create("Target", EntityType.CLASS)

        await graph.add_entity(e1)
        await graph.add_entity(e2)

        relation = EntityRelation(
            source_id=e1.id,
            target_id=e2.id,
            relation_type=RelationType.DEPENDS_ON,
        )
        await graph.add_relation(relation)

        neighbors = await graph.get_neighbors(e1.id, direction="outgoing")
        assert len(neighbors) == 1
        assert neighbors[0][0].id == e2.id

    @pytest.mark.asyncio
    async def test_get_neighbors_depth(self, graph):
        """Test getting neighbors with depth traversal."""
        # Create chain: A -> B -> C
        e_a = Entity.create("A", EntityType.CLASS)
        e_b = Entity.create("B", EntityType.CLASS)
        e_c = Entity.create("C", EntityType.CLASS)

        await graph.add_entity(e_a)
        await graph.add_entity(e_b)
        await graph.add_entity(e_c)

        await graph.add_relation(
            EntityRelation(
                source_id=e_a.id,
                target_id=e_b.id,
                relation_type=RelationType.DEPENDS_ON,
            )
        )
        await graph.add_relation(
            EntityRelation(
                source_id=e_b.id,
                target_id=e_c.id,
                relation_type=RelationType.DEPENDS_ON,
            )
        )

        # Depth 1 should only get B
        neighbors_d1 = await graph.get_neighbors(e_a.id, depth=1)
        assert len(neighbors_d1) == 1

        # Depth 2 should get B and C
        neighbors_d2 = await graph.get_neighbors(e_a.id, depth=2)
        assert len(neighbors_d2) == 2

    @pytest.mark.asyncio
    async def test_find_paths(self, graph):
        """Test finding paths between entities."""
        e_a = Entity.create("A", EntityType.CLASS)
        e_b = Entity.create("B", EntityType.CLASS)
        e_c = Entity.create("C", EntityType.CLASS)

        await graph.add_entity(e_a)
        await graph.add_entity(e_b)
        await graph.add_entity(e_c)

        await graph.add_relation(
            EntityRelation(
                source_id=e_a.id,
                target_id=e_b.id,
                relation_type=RelationType.DEPENDS_ON,
            )
        )
        await graph.add_relation(
            EntityRelation(
                source_id=e_b.id,
                target_id=e_c.id,
                relation_type=RelationType.DEPENDS_ON,
            )
        )

        paths = await graph.find_paths(e_a.id, e_c.id, max_depth=3)

        assert len(paths) == 1
        assert paths[0].length == 2
        assert paths[0].entities[0].name == "A"
        assert paths[0].entities[1].name == "B"
        assert paths[0].entities[2].name == "C"

    @pytest.mark.asyncio
    async def test_get_subgraph(self, graph):
        """Test extracting a subgraph."""
        e1 = Entity.create("E1", EntityType.CLASS)
        e2 = Entity.create("E2", EntityType.CLASS)
        e3 = Entity.create("E3", EntityType.CLASS)

        await graph.add_entity(e1)
        await graph.add_entity(e2)
        await graph.add_entity(e3)

        await graph.add_relation(
            EntityRelation(
                source_id=e1.id,
                target_id=e2.id,
                relation_type=RelationType.RELATED_TO,
            )
        )

        entities, relations = await graph.get_subgraph({e1.id, e2.id})

        assert len(entities) == 2
        assert len(relations) == 1

    @pytest.mark.asyncio
    async def test_get_stats(self, graph):
        """Test getting graph statistics."""
        e1 = Entity.create("E1", EntityType.CLASS)
        e2 = Entity.create("E2", EntityType.FUNCTION)

        await graph.add_entity(e1)
        await graph.add_entity(e2)

        await graph.add_relation(
            EntityRelation(
                source_id=e1.id,
                target_id=e2.id,
                relation_type=RelationType.CONTAINS,
            )
        )

        stats = await graph.get_stats()

        assert stats.entity_count == 2
        assert stats.relation_count == 1
        assert EntityType.CLASS in stats.entity_type_counts
        assert EntityType.FUNCTION in stats.entity_type_counts

    @pytest.mark.asyncio
    async def test_remove_entity(self, graph):
        """Test removing entity and its relations."""
        e1 = Entity.create("E1", EntityType.CLASS)
        e2 = Entity.create("E2", EntityType.CLASS)

        await graph.add_entity(e1)
        await graph.add_entity(e2)

        await graph.add_relation(
            EntityRelation(
                source_id=e1.id,
                target_id=e2.id,
                relation_type=RelationType.DEPENDS_ON,
            )
        )

        removed = await graph.remove_entity(e1.id)
        assert removed is True

        # Entity should be gone
        assert await graph.get_entity(e1.id) is None

        # Relations should be gone
        stats = await graph.get_stats()
        assert stats.relation_count == 0


class TestCodeEntityExtractor:
    """Tests for CodeEntityExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create code entity extractor."""
        return CodeEntityExtractor()

    @pytest.mark.asyncio
    async def test_extract_function_python(self, extractor):
        """Test extracting Python function definitions."""
        code = """
def authenticate_user(username, password):
    pass

async def fetch_data():
    pass
"""
        result = await extractor.extract(code)

        names = {e.name for e in result.entities}
        assert "authenticate_user" in names
        assert "fetch_data" in names

    @pytest.mark.asyncio
    async def test_extract_class(self, extractor):
        """Test extracting class definitions."""
        code = """
class UserAuthentication:
    pass

class TokenManager(BaseManager):
    pass
"""
        result = await extractor.extract(code)

        names = {e.name for e in result.entities}
        assert "UserAuthentication" in names
        assert "TokenManager" in names

    @pytest.mark.asyncio
    async def test_extract_file_references(self, extractor):
        """Test extracting file references."""
        text = """
Check the implementation in `src/auth/handler.py`.
Also see file: config/settings.json for configuration.
"""
        result = await extractor.extract(text)

        names = {e.name for e in result.entities}
        assert "src/auth/handler.py" in names or "handler.py" in names

    @pytest.mark.asyncio
    async def test_extract_imports(self, extractor):
        """Test extracting module imports."""
        code = """
from victor.storage.memory import EntityMemory
import asyncio
from typing import Dict, List
"""
        result = await extractor.extract(code)

        # Should extract module names
        names = {e.name for e in result.entities}
        assert any("victor" in n.lower() for n in names)


class TestTextEntityExtractor:
    """Tests for TextEntityExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create text entity extractor."""
        return TextEntityExtractor()

    @pytest.mark.asyncio
    async def test_extract_technologies(self, extractor):
        """Test extracting technology mentions."""
        text = """
We're using Python and FastAPI for the backend,
with PostgreSQL for data storage and Redis for caching.
"""
        result = await extractor.extract(text)

        names = {e.name.lower() for e in result.entities}
        assert "python" in names
        assert "fastapi" in names
        assert "postgresql" in names or "postgres" in names

    @pytest.mark.asyncio
    async def test_extract_organizations(self, extractor):
        """Test extracting organization mentions."""
        text = """
Anthropic released Claude, and OpenAI has GPT-4.
Google's Gemini is also available.
"""
        result = await extractor.extract(text)

        # Technologies should be extracted
        names = {e.name.lower() for e in result.entities}
        assert "claude" in names or "anthropic" in names

    @pytest.mark.asyncio
    async def test_extract_concepts(self, extractor):
        """Test extracting concept patterns."""
        text = """
We'll implement the Observer pattern for event handling.
The Singleton approach works well here.
"""
        result = await extractor.extract(text)

        names = {e.name.lower() for e in result.entities}
        # Should extract pattern names
        assert any("observer" in n for n in names) or any("singleton" in n for n in names)


class TestCompositeExtractor:
    """Tests for CompositeExtractor."""

    @pytest.mark.asyncio
    async def test_composite_extraction(self):
        """Test combining multiple extractors."""
        extractor = CompositeExtractor.create_default()

        text = """
class AuthHandler:
    def authenticate(self, user):
        # Uses Python's built-in auth
        pass

We're implementing this with FastAPI and PostgreSQL.
"""
        result = await extractor.extract(text, source="test")

        assert len(result.entities) > 0
        assert result.metadata.get("composite") is True

        # Should have both code and text entities
        types = {e.entity_type for e in result.entities}
        assert EntityType.CLASS in types or EntityType.FUNCTION in types
        assert EntityType.TECHNOLOGY in types

    @pytest.mark.asyncio
    async def test_deduplication(self):
        """Test that duplicate entities are merged."""
        extractor = CompositeExtractor.create_default()

        # Mention Python multiple times
        text = """
Python is great. We use Python for everything.
Python, Python, Python!
"""
        result = await extractor.extract(text)

        # Should have only one Python entity (deduplicated)
        python_entities = [e for e in result.entities if e.name.lower() == "python"]
        assert len(python_entities) <= 1


class TestExtractionResult:
    """Tests for ExtractionResult."""

    def test_merge_results(self):
        """Test merging extraction results."""
        e1 = Entity.create("Entity1", EntityType.CLASS)
        e2 = Entity.create("Entity2", EntityType.FUNCTION)

        result1 = ExtractionResult(entities=[e1], confidence=0.8)
        result2 = ExtractionResult(entities=[e2], confidence=0.9)

        merged = result1.merge(result2)

        assert len(merged.entities) == 2
        assert merged.confidence == 0.8  # Min of confidences

    def test_filter_by_type(self):
        """Test filtering by entity type."""
        entities = [
            Entity.create("Class1", EntityType.CLASS),
            Entity.create("func1", EntityType.FUNCTION),
            Entity.create("Class2", EntityType.CLASS),
        ]

        result = ExtractionResult(entities=entities)
        filtered = result.filter_by_type({EntityType.CLASS})

        assert len(filtered.entities) == 2
        assert all(e.entity_type == EntityType.CLASS for e in filtered.entities)

    def test_filter_by_confidence(self):
        """Test filtering by confidence threshold."""
        e1 = Entity.create("High", EntityType.CLASS)
        e1.confidence = 0.9
        e2 = Entity.create("Low", EntityType.CLASS)
        e2.confidence = 0.3

        result = ExtractionResult(entities=[e1, e2])
        filtered = result.filter_by_confidence(0.5)

        assert len(filtered.entities) == 1
        assert filtered.entities[0].name == "High"


class TestEntityTypeCategories:
    """Tests for entity type category sets."""

    def test_code_entity_types(self):
        """Test CODE_ENTITY_TYPES contains expected types."""
        assert EntityType.FILE in CODE_ENTITY_TYPES
        assert EntityType.FUNCTION in CODE_ENTITY_TYPES
        assert EntityType.CLASS in CODE_ENTITY_TYPES
        assert EntityType.MODULE in CODE_ENTITY_TYPES
        assert EntityType.PERSON not in CODE_ENTITY_TYPES
