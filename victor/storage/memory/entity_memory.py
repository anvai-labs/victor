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

"""4-tier entity memory system.

Implements CrewAI-style entity memory with:
1. Short-term memory: Current session entities
2. Working memory: LRU cache of recently accessed entities
3. Long-term memory: Persistent SQLite storage
4. Entity graph: Relationship tracking and traversal

Example:
    memory = EntityMemory(session_id="session_123")

    # Store entity
    entity = Entity.create("UserAuth", EntityType.CLASS, "Authentication class")
    await memory.store(entity)

    # Retrieve by name or type
    entities = await memory.search("auth", entity_types=[EntityType.CLASS])

    # Get related entities
    related = await memory.get_related("ent_abc123", relation_types=[RelationType.IMPORTS])
"""

import asyncio
import logging
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from victor.storage.memory.entity_types import (
    Entity,
    EntityRelation,
    EntityType,
    RelationType,
)

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache for working memory."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: OrderedDict[str, Entity] = OrderedDict()

    def get(self, key: str) -> Optional[Entity]:
        """Get entity, moving it to end (most recent)."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, entity: Entity) -> None:
        """Add/update entity in cache."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
        self._cache[key] = entity

    def remove(self, key: str) -> bool:
        """Remove entity from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached entities."""
        self._cache.clear()

    def values(self) -> List[Entity]:
        """Get all cached entities."""
        return list(self._cache.values())

    def __len__(self) -> int:
        return len(self._cache)


@dataclass
class EntityMemoryConfig:
    """Configuration for EntityMemory.

    Attributes:
        db_path: Path to SQLite database
        working_memory_size: Max entities in LRU cache
        embedding_enabled: Enable vector embeddings
        auto_extract: Automatically extract entities from messages
        merge_duplicates: Merge entities with same name/type
    """

    db_path: Optional[str] = None
    working_memory_size: int = 100
    embedding_enabled: bool = True
    auto_extract: bool = True
    merge_duplicates: bool = True


class EntityMemory:
    """4-tier entity memory system.

    Tiers:
    1. Short-term: Current session entities (in-memory dict)
    2. Working memory: LRU cache of frequently accessed entities
    3. Long-term: Persistent SQLite storage
    4. Entity graph: Relationship tracking

    The tiers work together:
    - New entities go to short-term first
    - Frequently accessed entities are cached in working memory
    - All entities are persisted to long-term storage
    - Relations are tracked in the entity graph
    """

    def __init__(
        self,
        session_id: str = "default",
        config: Optional[EntityMemoryConfig] = None,
    ):
        """Initialize entity memory.

        Args:
            session_id: Unique session identifier
            config: Memory configuration
        """
        self.session_id = session_id
        self.config = config or EntityMemoryConfig()

        # Tier 1: Short-term memory (current session)
        self._short_term: Dict[str, Entity] = {}

        # Tier 2: Working memory (LRU cache)
        self._working_memory = LRUCache(max_size=self.config.working_memory_size)

        # Tier 3: Long-term memory (SQLite)
        self._db_path = self.config.db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._initialized = False

        # Tier 4: Entity graph (relations)
        self._relations: Dict[str, EntityRelation] = {}

    async def initialize(self) -> None:
        """Initialize the memory storage."""
        if self._initialized:
            return

        if self._db_path:
            await self._init_database()

        self._initialized = True
        logger.info(f"EntityMemory initialized for session {self.session_id}")

    async def _init_database(self) -> None:
        """Initialize SQLite database."""
        db_path = Path(self._db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row

        # Create tables
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                description TEXT,
                attributes TEXT,
                source TEXT,
                confidence REAL DEFAULT 1.0,
                mentions INTEGER DEFAULT 1,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                session_id TEXT
            );

            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                attributes TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id)
            );

            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_entities_session ON entities(session_id);
            CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
        """)
        self._conn.commit()

    async def store(self, entity: Entity) -> str:
        """Store an entity in memory.

        Args:
            entity: Entity to store

        Returns:
            Entity ID
        """
        if not self._initialized:
            await self.initialize()

        # Check for existing entity (merge if configured)
        existing = await self.get(entity.id)
        if existing and self.config.merge_duplicates:
            entity = existing.merge_with(entity)

        # Store in short-term memory
        self._short_term[entity.id] = entity

        # Update working memory
        self._working_memory.put(entity.id, entity)

        # Persist to long-term storage
        if self._conn:
            await self._persist_entity(entity)

        logger.debug(f"Stored entity: {entity.name} ({entity.entity_type.value})")
        return entity.id

    async def _persist_entity(self, entity: Entity) -> None:
        """Persist entity to SQLite."""
        import json

        self._conn.execute(
            """
            INSERT OR REPLACE INTO entities
            (id, name, entity_type, description, attributes, source,
             confidence, mentions, first_seen, last_seen, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity.id,
                entity.name,
                entity.entity_type.value,
                entity.description,
                json.dumps(entity.attributes),
                entity.source,
                entity.confidence,
                entity.mentions,
                entity.first_seen.isoformat(),
                entity.last_seen.isoformat(),
                self.session_id,
            ),
        )
        self._conn.commit()

    async def get(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID.

        Checks tiers in order: short-term → working memory → long-term.

        Args:
            entity_id: Entity ID

        Returns:
            Entity if found, None otherwise
        """
        # Check short-term
        if entity_id in self._short_term:
            return self._short_term[entity_id]

        # Check working memory
        cached = self._working_memory.get(entity_id)
        if cached:
            return cached

        # Check long-term storage
        if self._conn:
            entity = await self._load_entity(entity_id)
            if entity:
                # Promote to working memory
                self._working_memory.put(entity_id, entity)
                return entity

        return None

    async def _load_entity(self, entity_id: str) -> Optional[Entity]:
        """Load entity from SQLite."""
        import json

        cursor = self._conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=EntityType(row["entity_type"]),
            description=row["description"],
            attributes=json.loads(row["attributes"]) if row["attributes"] else {},
            source=row["source"],
            confidence=row["confidence"],
            mentions=row["mentions"],
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
        )

    async def search(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10,
    ) -> List[Entity]:
        """Search for entities by name.

        Args:
            query: Search query (name substring)
            entity_types: Optional filter by types
            limit: Maximum results

        Returns:
            List of matching entities
        """
        if not self._initialized:
            await self.initialize()

        results: List[Entity] = []
        query_lower = query.lower()

        # Search short-term memory first
        for entity in self._short_term.values():
            if query_lower in entity.name.lower():
                if entity_types is None or entity.entity_type in entity_types:
                    results.append(entity)

        # Search long-term if needed
        if len(results) < limit and self._conn:
            type_filter = ""
            params: List[Any] = [f"%{query}%"]

            if entity_types:
                placeholders = ",".join("?" * len(entity_types))
                type_filter = f" AND entity_type IN ({placeholders})"
                params.extend([t.value for t in entity_types])

            params.append(limit)

            cursor = self._conn.execute(
                f"""
                SELECT * FROM entities
                WHERE name LIKE ? {type_filter}
                ORDER BY mentions DESC, last_seen DESC
                LIMIT ?
                """,
                params,
            )

            import json

            for row in cursor:
                # Skip if already in results
                if any(e.id == row["id"] for e in results):
                    continue

                entity = Entity(
                    id=row["id"],
                    name=row["name"],
                    entity_type=EntityType(row["entity_type"]),
                    description=row["description"],
                    attributes=(json.loads(row["attributes"]) if row["attributes"] else {}),
                    source=row["source"],
                    confidence=row["confidence"],
                    mentions=row["mentions"],
                    first_seen=datetime.fromisoformat(row["first_seen"]),
                    last_seen=datetime.fromisoformat(row["last_seen"]),
                )
                results.append(entity)

                if len(results) >= limit:
                    break

        return results[:limit]

    async def get_by_type(
        self,
        entity_type: EntityType,
        limit: int = 50,
    ) -> List[Entity]:
        """Get all entities of a specific type.

        Args:
            entity_type: Type to filter by
            limit: Maximum results

        Returns:
            List of entities
        """
        return await self.search("", entity_types=[entity_type], limit=limit)

    async def store_relation(self, relation: EntityRelation) -> str:
        """Store a relationship between entities.

        Args:
            relation: Relationship to store

        Returns:
            Relation ID
        """
        if not self._initialized:
            await self.initialize()

        # Store in memory
        self._relations[relation.id] = relation

        # Persist to long-term
        if self._conn:
            await self._persist_relation(relation)

        logger.debug(
            f"Stored relation: {relation.source_id} "
            f"-[{relation.relation_type.value}]-> {relation.target_id}"
        )
        return relation.id

    async def _persist_relation(self, relation: EntityRelation) -> None:
        """Persist relation to SQLite."""
        import json

        self._conn.execute(
            """
            INSERT OR REPLACE INTO relations
            (id, source_id, target_id, relation_type, strength,
             attributes, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                relation.id,
                relation.source_id,
                relation.target_id,
                relation.relation_type.value,
                relation.strength,
                json.dumps(relation.attributes),
                relation.first_seen.isoformat(),
                relation.last_seen.isoformat(),
            ),
        )
        self._conn.commit()

    async def get_related(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        direction: str = "outgoing",
        limit: int = 20,
    ) -> List[tuple[Entity, EntityRelation]]:
        """Get entities related to a given entity.

        Args:
            entity_id: Source entity ID
            relation_types: Optional filter by relation types
            direction: "outgoing", "incoming", or "both"
            limit: Maximum results

        Returns:
            List of (entity, relation) tuples
        """
        if not self._initialized:
            await self.initialize()

        results: List[tuple[Entity, EntityRelation]] = []

        # Get relations from memory
        for relation in self._relations.values():
            if direction in ("outgoing", "both") and relation.source_id == entity_id:
                if relation_types is None or relation.relation_type in relation_types:
                    target = await self.get(relation.target_id)
                    if target:
                        results.append((target, relation))

            if direction in ("incoming", "both") and relation.target_id == entity_id:
                if relation_types is None or relation.relation_type in relation_types:
                    source = await self.get(relation.source_id)
                    if source:
                        results.append((source, relation))

        # Load from database if needed
        if len(results) < limit and self._conn:
            if direction in ("outgoing", "both"):
                cursor = self._conn.execute(
                    "SELECT * FROM relations WHERE source_id = ? LIMIT ?",
                    (entity_id, limit),
                )
                for row in cursor:
                    rel = EntityRelation.from_dict(dict(row))
                    if rel.id not in self._relations:
                        if relation_types is None or rel.relation_type in relation_types:
                            target = await self.get(rel.target_id)
                            if target:
                                results.append((target, rel))

            if direction in ("incoming", "both"):
                cursor = self._conn.execute(
                    "SELECT * FROM relations WHERE target_id = ? LIMIT ?",
                    (entity_id, limit),
                )
                for row in cursor:
                    rel = EntityRelation.from_dict(dict(row))
                    if rel.id not in self._relations:
                        if relation_types is None or rel.relation_type in relation_types:
                            source = await self.get(rel.source_id)
                            if source:
                                results.append((source, rel))

        # Sort by relation strength
        results.sort(key=lambda x: x[1].strength, reverse=True)
        return results[:limit]

    async def get_session_entities(self) -> List[Entity]:
        """Get all entities from current session.

        Returns:
            List of session entities
        """
        return list(self._short_term.values())

    async def get_recent_entities(self, limit: int = 20) -> List[Entity]:
        """Get most recently accessed entities.

        Returns:
            List of recent entities from working memory
        """
        return self._working_memory.values()[:limit]

    async def increment_mentions(self, entity_id: str) -> None:
        """Increment mention count for an entity.

        Args:
            entity_id: Entity ID
        """
        entity = await self.get(entity_id)
        if entity:
            entity.mentions += 1
            entity.last_seen = datetime.now(timezone.utc)
            await self.store(entity)

    async def clear_session(self) -> None:
        """Clear short-term memory for current session."""
        self._short_term.clear()
        logger.info(f"Cleared short-term memory for session {self.session_id}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory stats
        """
        stats = {
            "session_id": self.session_id,
            "short_term_count": len(self._short_term),
            "working_memory_count": len(self._working_memory),
            "relations_count": len(self._relations),
        }

        if self._conn:
            cursor = self._conn.execute("SELECT COUNT(*) FROM entities")
            stats["long_term_count"] = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(*) FROM relations")
            stats["stored_relations_count"] = cursor.fetchone()[0]

        return stats

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
