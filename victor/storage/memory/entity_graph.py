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

"""Entity graph for relationship tracking and traversal.

Provides graph-based storage and querying of entity relationships
with support for traversal algorithms and path finding.

Example:
    graph = EntityGraph()

    # Add entities and relationships
    await graph.add_entity(user_entity)
    await graph.add_entity(auth_class)
    await graph.add_relation(EntityRelation(
        source_id=user_entity.id,
        target_id=auth_class.id,
        relation_type=RelationType.DEPENDS_ON,
    ))

    # Find related entities
    related = await graph.get_neighbors(user_entity.id, depth=2)

    # Find paths
    paths = await graph.find_paths(source_id, target_id, max_depth=3)
"""

import logging
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass, field

from victor.core.json_utils import json_dumps, json_loads
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from victor.storage.memory.entity_types import (
    Entity,
    EntityRelation,
    EntityType,
    RelationType,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphPath:
    """Represents a path through the entity graph.

    Attributes:
        entities: Ordered list of entities in the path
        relations: Relations connecting the entities
        total_strength: Sum of relation strengths
    """

    entities: List[Entity] = field(default_factory=list)
    relations: List[EntityRelation] = field(default_factory=list)
    total_strength: float = 0.0

    @property
    def length(self) -> int:
        """Number of edges in the path."""
        return len(self.relations)

    def append(self, entity: Entity, relation: Optional[EntityRelation] = None) -> "GraphPath":
        """Append an entity and optional relation to the path."""
        self.entities.append(entity)
        if relation:
            self.relations.append(relation)
            self.total_strength += relation.strength
        return self


@dataclass
class GraphStats:
    """Statistics about the entity graph."""

    entity_count: int = 0
    relation_count: int = 0
    entity_type_counts: Dict[EntityType, int] = field(default_factory=dict)
    relation_type_counts: Dict[RelationType, int] = field(default_factory=dict)
    avg_connections_per_entity: float = 0.0
    most_connected_entities: List[Tuple[str, int]] = field(default_factory=list)


class EntityGraph:
    """Graph-based entity relationship storage.

    Supports both in-memory and SQLite-backed storage with
    graph traversal algorithms for finding related entities.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        in_memory: bool = True,
    ):
        """Initialize entity graph.

        Args:
            db_path: Path to SQLite database (optional)
            in_memory: Keep graph in memory for fast access
        """
        self._db_path = db_path
        self._in_memory = in_memory
        self._conn: Optional[sqlite3.Connection] = None
        self._initialized = False

        # In-memory adjacency lists
        self._outgoing: Dict[str, List[EntityRelation]] = defaultdict(list)
        self._incoming: Dict[str, List[EntityRelation]] = defaultdict(list)
        self._entities: Dict[str, Entity] = {}

    async def initialize(self) -> None:
        """Initialize the graph storage."""
        if self._initialized:
            return

        if self._db_path:
            await self._init_database()

        self._initialized = True
        logger.info("EntityGraph initialized")

    async def _init_database(self) -> None:
        """Initialize SQLite database for persistence."""
        db_path = Path(self._db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS graph_entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                description TEXT,
                attributes TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS graph_relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                attributes TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES graph_entities(id),
                FOREIGN KEY (target_id) REFERENCES graph_entities(id)
            );

            CREATE INDEX IF NOT EXISTS idx_graph_source ON graph_relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_graph_target ON graph_relations(target_id);
            CREATE INDEX IF NOT EXISTS idx_graph_type ON graph_relations(relation_type);
        """)
        self._conn.commit()

        # Load existing data into memory if in_memory mode
        if self._in_memory:
            await self._load_to_memory()

    async def _load_to_memory(self) -> None:
        """Load database contents into memory."""

        if not self._conn:
            return

        # Load entities
        cursor = self._conn.execute("SELECT * FROM graph_entities")
        for row in cursor:
            entity = Entity(
                id=row["id"],
                name=row["name"],
                entity_type=EntityType(row["entity_type"]),
                description=row["description"],
                attributes=json_loads(row["attributes"]) if row["attributes"] else {},
            )
            self._entities[entity.id] = entity

        # Load relations
        cursor = self._conn.execute("SELECT * FROM graph_relations")
        for row in cursor:
            relation = EntityRelation(
                source_id=row["source_id"],
                target_id=row["target_id"],
                relation_type=RelationType(row["relation_type"]),
                strength=row["strength"],
                attributes=json_loads(row["attributes"]) if row["attributes"] else {},
            )
            self._outgoing[relation.source_id].append(relation)
            self._incoming[relation.target_id].append(relation)

    async def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph.

        Args:
            entity: Entity to add
        """
        if not self._initialized:
            await self.initialize()

        self._entities[entity.id] = entity

        if self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO graph_entities
                (id, name, entity_type, description, attributes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entity.id,
                    entity.name,
                    entity.entity_type.value,
                    entity.description,
                    json_dumps(entity.attributes),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    async def add_relation(self, relation: EntityRelation) -> None:
        """Add a relationship to the graph.

        Args:
            relation: Relation to add
        """
        if not self._initialized:
            await self.initialize()

        # Update in-memory adjacency
        self._outgoing[relation.source_id].append(relation)
        self._incoming[relation.target_id].append(relation)

        if self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO graph_relations
                (id, source_id, target_id, relation_type, strength, attributes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relation.id,
                    relation.source_id,
                    relation.target_id,
                    relation.relation_type.value,
                    relation.strength,
                    json_dumps(relation.attributes),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self._conn.commit()

    async def add_entities_batch(self, entities: List[Entity]) -> None:
        """Add multiple entities in a single transaction (10-50x faster for bulk loads)."""
        if not self._initialized:
            await self.initialize()

        for entity in entities:
            self._entities[entity.id] = entity

        if self._conn and entities:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO graph_entities
                (id, name, entity_type, description, attributes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.id,
                        e.name,
                        e.entity_type.value,
                        e.description,
                        json_dumps(e.attributes),
                        now,
                    )
                    for e in entities
                ],
            )
            self._conn.commit()

    async def add_relations_batch(self, relations: List[EntityRelation]) -> None:
        """Add multiple relations in a single transaction (10-50x faster for bulk loads)."""
        if not self._initialized:
            await self.initialize()

        for relation in relations:
            self._outgoing[relation.source_id].append(relation)
            self._incoming[relation.target_id].append(relation)

        if self._conn and relations:
            now = datetime.now(timezone.utc).isoformat()
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO graph_relations
                (id, source_id, target_id, relation_type, strength, attributes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.id,
                        r.source_id,
                        r.target_id,
                        r.relation_type.value,
                        r.strength,
                        json_dumps(r.attributes),
                        now,
                    )
                    for r in relations
                ],
            )
            self._conn.commit()

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            Entity if found
        """
        return self._entities.get(entity_id)

    async def get_neighbors(
        self,
        entity_id: str,
        relation_types: Optional[Set[RelationType]] = None,
        direction: str = "both",
        depth: int = 1,
    ) -> List[Tuple[Entity, EntityRelation, int]]:
        """Get neighboring entities.

        Args:
            entity_id: Starting entity ID
            relation_types: Filter by relation types
            direction: "outgoing", "incoming", or "both"
            depth: Maximum traversal depth

        Returns:
            List of (entity, relation, depth) tuples
        """
        if not self._initialized:
            await self.initialize()

        results: List[Tuple[Entity, EntityRelation, int]] = []
        visited: Set[str] = {entity_id}
        queue: deque[Tuple[str, int]] = deque([(entity_id, 0)])

        while queue:
            current_id, current_depth = queue.popleft()

            if current_depth >= depth:
                continue

            # Get outgoing neighbors
            if direction in ("outgoing", "both"):
                for relation in self._outgoing.get(current_id, []):
                    if relation_types and relation.relation_type not in relation_types:
                        continue

                    target_id = relation.target_id
                    if target_id not in visited:
                        visited.add(target_id)
                        entity = self._entities.get(target_id)
                        if entity:
                            results.append((entity, relation, current_depth + 1))
                            queue.append((target_id, current_depth + 1))

            # Get incoming neighbors
            if direction in ("incoming", "both"):
                for relation in self._incoming.get(current_id, []):
                    if relation_types and relation.relation_type not in relation_types:
                        continue

                    source_id = relation.source_id
                    if source_id not in visited:
                        visited.add(source_id)
                        entity = self._entities.get(source_id)
                        if entity:
                            results.append((entity, relation, current_depth + 1))
                            queue.append((source_id, current_depth + 1))

        return results

    async def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
        relation_types: Optional[Set[RelationType]] = None,
    ) -> List[GraphPath]:
        """Find all paths between two entities.

        Args:
            source_id: Starting entity ID
            target_id: Target entity ID
            max_depth: Maximum path length
            relation_types: Filter by relation types

        Returns:
            List of paths from source to target
        """
        if not self._initialized:
            await self.initialize()

        paths: List[GraphPath] = []
        source = self._entities.get(source_id)
        target = self._entities.get(target_id)

        if not source or not target:
            return paths

        # BFS for path finding
        queue: deque[GraphPath] = deque()
        initial_path = GraphPath(entities=[source])
        queue.append(initial_path)

        while queue:
            current_path = queue.popleft()

            if current_path.length >= max_depth:
                continue

            current_id = current_path.entities[-1].id

            for relation in self._outgoing.get(current_id, []):
                if relation_types and relation.relation_type not in relation_types:
                    continue

                next_id = relation.target_id
                next_entity = self._entities.get(next_id)

                if not next_entity:
                    continue

                # Avoid cycles
                if any(e.id == next_id for e in current_path.entities):
                    continue

                new_path = GraphPath(
                    entities=current_path.entities.copy(),
                    relations=current_path.relations.copy(),
                    total_strength=current_path.total_strength,
                )
                new_path.append(next_entity, relation)

                if next_id == target_id:
                    paths.append(new_path)
                else:
                    queue.append(new_path)

        # Sort by total strength (strongest paths first)
        paths.sort(key=lambda p: p.total_strength, reverse=True)
        return paths

    async def get_subgraph(
        self,
        entity_ids: Set[str],
        include_relations: bool = True,
    ) -> Tuple[List[Entity], List[EntityRelation]]:
        """Get a subgraph containing specified entities.

        Args:
            entity_ids: Set of entity IDs to include
            include_relations: Include relations between entities

        Returns:
            Tuple of (entities, relations)
        """
        entities = [self._entities[eid] for eid in entity_ids if eid in self._entities]

        relations: List[EntityRelation] = []
        if include_relations:
            for eid in entity_ids:
                for relation in self._outgoing.get(eid, []):
                    if relation.target_id in entity_ids:
                        relations.append(relation)

        return entities, relations

    async def get_stats(self) -> GraphStats:
        """Get graph statistics.

        Returns:
            GraphStats with counts and metrics
        """
        stats = GraphStats()

        stats.entity_count = len(self._entities)

        # Count relations
        relation_count = 0
        for relations in self._outgoing.values():
            relation_count += len(relations)
        stats.relation_count = relation_count

        # Count by entity type
        for entity in self._entities.values():
            if entity.entity_type not in stats.entity_type_counts:
                stats.entity_type_counts[entity.entity_type] = 0
            stats.entity_type_counts[entity.entity_type] += 1

        # Count by relation type
        for relations in self._outgoing.values():
            for relation in relations:
                if relation.relation_type not in stats.relation_type_counts:
                    stats.relation_type_counts[relation.relation_type] = 0
                stats.relation_type_counts[relation.relation_type] += 1

        # Average connections
        if stats.entity_count > 0:
            total_connections = sum(
                len(self._outgoing.get(eid, [])) + len(self._incoming.get(eid, []))
                for eid in self._entities
            )
            stats.avg_connections_per_entity = total_connections / stats.entity_count

        # Most connected entities
        connection_counts = [
            (
                eid,
                len(self._outgoing.get(eid, [])) + len(self._incoming.get(eid, [])),
            )
            for eid in self._entities
        ]
        connection_counts.sort(key=lambda x: x[1], reverse=True)
        stats.most_connected_entities = connection_counts[:10]

        return stats

    async def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and its relations.

        Args:
            entity_id: Entity ID to remove

        Returns:
            True if removed
        """
        if entity_id not in self._entities:
            return False

        # Remove from memory
        del self._entities[entity_id]

        # Remove relations
        if entity_id in self._outgoing:
            del self._outgoing[entity_id]
        if entity_id in self._incoming:
            del self._incoming[entity_id]

        # Remove from other entities' adjacency lists
        for eid in list(self._outgoing.keys()):
            self._outgoing[eid] = [r for r in self._outgoing[eid] if r.target_id != entity_id]
        for eid in list(self._incoming.keys()):
            self._incoming[eid] = [r for r in self._incoming[eid] if r.source_id != entity_id]

        # Remove from database
        if self._conn:
            self._conn.execute(
                "DELETE FROM graph_relations WHERE source_id = ? OR target_id = ?",
                (entity_id, entity_id),
            )
            self._conn.execute(
                "DELETE FROM graph_entities WHERE id = ?",
                (entity_id,),
            )
            self._conn.commit()

        return True

    async def clear(self) -> None:
        """Clear all entities and relations."""
        self._entities.clear()
        self._outgoing.clear()
        self._incoming.clear()

        if self._conn:
            self._conn.execute("DELETE FROM graph_relations")
            self._conn.execute("DELETE FROM graph_entities")
            self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
