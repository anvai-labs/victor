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

"""Memory adapters for integrating existing systems with UnifiedMemoryCoordinator.

This module provides adapter classes that wrap Victor's existing memory systems
to implement the MemoryProviderProtocol, enabling federated search across all
memory backends.

Adapters:
- EntityMemoryAdapter: Wraps EntityMemory for code entities and relationships
- ConversationMemoryAdapter: Wraps ConversationStore for message history
- GraphMemoryAdapter: Wraps EntityGraph for relationship traversal

Example:
    from victor.storage.memory import EntityMemory
    from victor.storage.memory.unified import get_memory_coordinator, MemoryType
    from victor.storage.memory.adapters import EntityMemoryAdapter

    # Create entity memory
    entity_mem = EntityMemory()
    await entity_mem.initialize()

    # Register adapter with coordinator
    coordinator = get_memory_coordinator()
    coordinator.register_provider(EntityMemoryAdapter(entity_mem))

    # Search across all memory
    results = await coordinator.search_all("authentication")
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from dataclasses import replace

from victor.storage.memory.unified import (
    MemoryProviderProtocol,
    MemoryProvenance,
    MemoryQuery,
    MemoryResult,
    MemoryType,
)

if TYPE_CHECKING:
    from victor.storage.memory.entity_memory import EntityMemory
    from victor.storage.memory.entity_graph import EntityGraph
    from victor.agent.conversation.store import ConversationStore

logger = logging.getLogger(__name__)


# =============================================================================
# Entity Memory Adapter
# =============================================================================


class EntityMemoryAdapter:
    """Adapter for EntityMemory to implement MemoryProviderProtocol.

    Wraps the 4-tier entity memory system (short-term, working, long-term, graph)
    to provide unified search across code entities.

    Example:
        entity_mem = EntityMemory(session_id="session_123")
        await entity_mem.initialize()

        adapter = EntityMemoryAdapter(entity_mem)
        results = await adapter.search(MemoryQuery(query="auth"))
    """

    def __init__(self, entity_memory: "EntityMemory"):
        """Initialize adapter.

        Args:
            entity_memory: EntityMemory instance to wrap
        """
        self._memory = entity_memory
        self._available = True

    @property
    def memory_type(self) -> MemoryType:
        """Return ENTITY memory type."""
        return MemoryType.ENTITY

    async def search(self, query: MemoryQuery) -> List[MemoryResult]:
        """Search entity memory.

        Args:
            query: Search query specification

        Returns:
            List of MemoryResult with entity content
        """
        try:
            # Search entities by name
            entities = await self._memory.search(
                query=query.query,
                entity_types=(query.filters.get("entity_types") if query.filters else None),
                limit=query.limit,
            )

            results = []
            for entity in entities:
                # Calculate relevance based on name match
                name_lower = entity.name.lower()
                query_lower = query.query.lower()

                if query_lower == name_lower:
                    relevance = 1.0
                elif query_lower in name_lower:
                    relevance = 0.8
                else:
                    # Partial match
                    relevance = 0.5

                # Boost by confidence and mentions
                relevance *= entity.confidence
                relevance = min(1.0, relevance + (entity.mentions * 0.01))

                results.append(
                    MemoryResult(
                        source=MemoryType.ENTITY,
                        content=self._entity_to_dict(entity),
                        relevance=relevance,
                        id=entity.id,
                        metadata={
                            "entity_type": entity.entity_type.value,
                            "source": entity.source,
                            "mentions": entity.mentions,
                        },
                        timestamp=(entity.last_seen.timestamp() if entity.last_seen else None),
                    )
                )

            provenance = MemoryProvenance(
                store_class=type(self._memory).__name__,
                query_id=query.query_id,
                lane=MemoryType.ENTITY,
                store_id=getattr(self._memory, "session_id", None),
            )
            return [replace(r, provenance=provenance) for r in results]

        except Exception as e:
            logger.warning(f"EntityMemoryAdapter search failed: {e}")
            return []

    async def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store entity in memory.

        Args:
            key: Entity ID (used if value is dict)
            value: Entity or dict with entity data
            metadata: Additional metadata (merged with entity)
        """
        from victor.storage.memory.entity_types import Entity, EntityType

        if isinstance(value, Entity):
            await self._memory.store(value)
        elif isinstance(value, dict):
            # Create entity from dict
            entity_type = EntityType(value.get("entity_type", "other"))
            entity = Entity.create(
                name=value.get("name", key),
                entity_type=entity_type,
                description=value.get("description", ""),
                source=value.get("source"),
                attributes=value.get("attributes", {}),
            )
            if metadata:
                entity.attributes.update(metadata)
            await self._memory.store(entity)

    async def get(self, key: str) -> Optional[MemoryResult]:
        """Get entity by ID.

        Args:
            key: Entity ID

        Returns:
            MemoryResult or None
        """
        entity = await self._memory.get(key)
        if entity is None:
            return None

        return MemoryResult(
            source=MemoryType.ENTITY,
            content=self._entity_to_dict(entity),
            relevance=entity.confidence,
            id=entity.id,
            metadata={
                "entity_type": entity.entity_type.value,
                "source": entity.source,
            },
            timestamp=entity.last_seen.timestamp() if entity.last_seen else None,
        )

    def is_available(self) -> bool:
        """Check if entity memory is available."""
        return self._available and getattr(self._memory, "_initialized", False)

    def _entity_to_dict(self, entity: Any) -> Dict[str, Any]:
        """Convert entity to dictionary for serialization."""
        return {
            "id": entity.id,
            "name": entity.name,
            "entity_type": entity.entity_type.value,
            "description": entity.description,
            "attributes": entity.attributes,
            "source": entity.source,
            "confidence": entity.confidence,
            "mentions": entity.mentions,
        }


# =============================================================================
# Conversation Memory Adapter
# =============================================================================


class ConversationMemoryAdapter:
    """Adapter for ConversationStore to implement MemoryProviderProtocol.

    Wraps the SQLite-based conversation store to provide unified search
    across message history with semantic relevance scoring.

    Example:
        store = ConversationStore()
        session = store.create_session()

        adapter = ConversationMemoryAdapter(store, session.session_id)
        results = await adapter.search(MemoryQuery(query="authentication bug"))
    """

    def __init__(
        self,
        conversation_store: "ConversationStore",
        session_id: Optional[str] = None,
    ):
        """Initialize adapter.

        Args:
            conversation_store: ConversationStore instance to wrap
            session_id: Optional session ID to filter queries
        """
        self._store = conversation_store
        self._session_id = session_id
        self._available = True

    @property
    def memory_type(self) -> MemoryType:
        """Return CONVERSATION memory type."""
        return MemoryType.CONVERSATION

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for queries.

        Args:
            session_id: Session ID to use
        """
        self._session_id = session_id

    async def search(self, query: MemoryQuery) -> List[MemoryResult]:
        """Search conversation history.

        Uses semantic search if available, falls back to keyword matching.

        Args:
            query: Search query specification

        Returns:
            List of MemoryResult with message content
        """
        session_id = query.session_id or self._session_id
        if not session_id:
            logger.warning("ConversationMemoryAdapter: No session_id specified")
            return []

        try:
            results = []
            dual_trace_requested = bool((query.filters or {}).get("dual_trace"))

            if dual_trace_requested and hasattr(self._store, "aget_dual_trace_relevant_messages"):
                dual_trace_results = await self._store.aget_dual_trace_relevant_messages(
                    session_id=session_id,
                    query=query.query,
                    semantic_limit=query.limit,
                    execution_limit=query.limit,
                    min_similarity=query.min_relevance or 0.3,
                )

                for trace_kind in ("semantic", "execution"):
                    for message, similarity in dual_trace_results.get(trace_kind, []):
                        trace_text = (
                            self._store.get_message_trace_text(message, trace_kind)
                            if hasattr(self._store, "get_message_trace_text")
                            else message.content
                        )
                        results.append(
                            MemoryResult(
                                source=MemoryType.CONVERSATION,
                                content={
                                    "role": message.role.value,
                                    "content": message.content,
                                    "tool_name": message.tool_name,
                                    "tool_call_id": message.tool_call_id,
                                    "trace_text": trace_text,
                                },
                                relevance=similarity,
                                id=message.id,
                                metadata={
                                    "role": message.role.value,
                                    "priority": message.priority.value,
                                    "token_count": message.token_count,
                                    "trace_kind": trace_kind,
                                },
                                timestamp=message.timestamp.timestamp(),
                            )
                        )

                provenance = MemoryProvenance(
                    store_class=type(self._store).__name__,
                    query_id=query.query_id,
                    lane=MemoryType.CONVERSATION,
                    store_id=session_id,
                )
                return [replace(r, provenance=provenance) for r in results[: query.limit]]

            # Try semantic search first
            if hasattr(self._store, "aget_semantically_relevant_messages"):
                semantic_results = await self._store.aget_semantically_relevant_messages(
                    session_id=session_id,
                    query=query.query,
                    limit=query.limit,
                    min_similarity=query.min_relevance or 0.3,
                )
            else:
                semantic_results = self._store.get_semantically_relevant_messages(
                    session_id=session_id,
                    query=query.query,
                    limit=query.limit,
                    min_similarity=query.min_relevance or 0.3,
                )

            for message, similarity in semantic_results:
                results.append(
                    MemoryResult(
                        source=MemoryType.CONVERSATION,
                        content={
                            "role": message.role.value,
                            "content": message.content,
                            "tool_name": message.tool_name,
                            "tool_call_id": message.tool_call_id,
                        },
                        relevance=similarity,
                        id=message.id,
                        metadata={
                            "role": message.role.value,
                            "priority": message.priority.value,
                            "token_count": message.token_count,
                        },
                        timestamp=message.timestamp.timestamp(),
                    )
                )

            # If no semantic results, fall back to recent messages with keyword matching
            if not results:
                recent = self._store.get_recent_messages(session_id, count=query.limit * 2)
                query_lower = query.query.lower()

                for message in recent:
                    if query_lower in message.content.lower():
                        results.append(
                            MemoryResult(
                                source=MemoryType.CONVERSATION,
                                content={
                                    "role": message.role.value,
                                    "content": message.content,
                                },
                                relevance=0.5,  # Lower relevance for keyword match
                                id=message.id,
                                metadata={
                                    "role": message.role.value,
                                    "priority": message.priority.value,
                                },
                                timestamp=message.timestamp.timestamp(),
                            )
                        )

            provenance = MemoryProvenance(
                store_class=type(self._store).__name__,
                query_id=query.query_id,
                lane=MemoryType.CONVERSATION,
                store_id=session_id,
            )
            return [replace(r, provenance=provenance) for r in results[: query.limit]]

        except Exception as e:
            logger.warning(f"ConversationMemoryAdapter search failed: {e}")
            return []

    async def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store message in conversation.

        Args:
            key: Message ID (or role if value is string)
            value: Message content or dict
            metadata: Additional metadata
        """
        from victor.agent.conversation.types import MessageRole

        session_id = (metadata or {}).get("session_id") or self._session_id
        if not session_id:
            logger.warning("Cannot store message without session_id")
            return

        if isinstance(value, str):
            # Simple string content - assume user role
            role = MessageRole(key) if key in ["user", "assistant", "system"] else MessageRole.USER
            self._store.add_message(session_id, role, value)
        elif isinstance(value, dict):
            role = MessageRole(value.get("role", "user"))
            content = value.get("content", "")
            self._store.add_message(
                session_id,
                role,
                content,
                tool_name=value.get("tool_name"),
                tool_call_id=value.get("tool_call_id"),
                metadata=metadata,
            )

    async def get(self, key: str) -> Optional[MemoryResult]:
        """Get message by ID.

        Note: ConversationStore doesn't support direct message lookup by ID,
        so this searches recent messages.
        """
        if not self._session_id:
            return None

        recent = self._store.get_recent_messages(self._session_id, count=100)
        for message in recent:
            if message.id == key:
                return MemoryResult(
                    source=MemoryType.CONVERSATION,
                    content={
                        "role": message.role.value,
                        "content": message.content,
                    },
                    relevance=1.0,
                    id=message.id,
                    timestamp=message.timestamp.timestamp(),
                )
        return None

    def is_available(self) -> bool:
        """Check if conversation store is available."""
        return self._available and self._store is not None


# =============================================================================
# Graph Memory Adapter
# =============================================================================


class GraphMemoryAdapter:
    """Adapter for EntityGraph to implement MemoryProviderProtocol.

    Wraps the entity relationship graph to provide unified search
    across entity relationships and traversals.

    Example:
        entity_mem = EntityMemory()
        graph_adapter = GraphMemoryAdapter(entity_mem)

        results = await graph_adapter.search(
            MemoryQuery(query="ent_abc123", filters={"relation_types": ["imports"]})
        )
    """

    def __init__(self, entity_memory: "EntityMemory"):
        """Initialize adapter.

        Args:
            entity_memory: EntityMemory instance (contains entity graph)
        """
        self._memory = entity_memory
        self._available = True

    @property
    def memory_type(self) -> MemoryType:
        """Return GRAPH memory type."""
        return MemoryType.GRAPH

    async def search(self, query: MemoryQuery) -> List[MemoryResult]:
        """Search entity relationships.

        The query is interpreted as an entity ID to find related entities.

        Args:
            query: Search query with entity ID in query field

        Returns:
            List of MemoryResult with related entities
        """
        try:
            # Query is entity ID to find related entities
            entity_id = query.query

            # Get relation type filters if specified
            relation_types = None
            if query.filters and "relation_types" in query.filters:
                from victor.storage.memory.entity_types import RelationType

                relation_types = [RelationType(rt) for rt in query.filters["relation_types"]]

            # Get direction filter
            direction = "both"
            if query.filters and "direction" in query.filters:
                direction = query.filters["direction"]

            related = await self._memory.get_related(
                entity_id=entity_id,
                relation_types=relation_types,
                direction=direction,
                limit=query.limit,
            )

            results = []
            for entity, relation in related:
                results.append(
                    MemoryResult(
                        source=MemoryType.GRAPH,
                        content={
                            "entity": {
                                "id": entity.id,
                                "name": entity.name,
                                "entity_type": entity.entity_type.value,
                                "description": entity.description,
                            },
                            "relation": {
                                "id": relation.id,
                                "type": relation.relation_type.value,
                                "source_id": relation.source_id,
                                "target_id": relation.target_id,
                                "strength": relation.strength,
                            },
                        },
                        relevance=relation.strength,
                        id=f"{relation.id}:{entity.id}",
                        metadata={
                            "relation_type": relation.relation_type.value,
                            "direction": (
                                "outgoing" if relation.source_id == entity_id else "incoming"
                            ),
                        },
                        timestamp=(relation.last_seen.timestamp() if relation.last_seen else None),
                    )
                )

            provenance = MemoryProvenance(
                store_class=type(self._memory).__name__,
                query_id=query.query_id,
                lane=MemoryType.GRAPH,
                store_id=getattr(self._memory, "session_id", None),
            )
            return [replace(r, provenance=provenance) for r in results]

        except Exception as e:
            logger.warning(f"GraphMemoryAdapter search failed: {e}")
            return []

    async def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store relationship in graph.

        Args:
            key: Relation ID
            value: Relation dict with source_id, target_id, relation_type
            metadata: Additional metadata
        """
        from victor.storage.memory.entity_types import EntityRelation, RelationType

        if isinstance(value, dict):
            relation = EntityRelation.create(
                source_id=value["source_id"],
                target_id=value["target_id"],
                relation_type=RelationType(value.get("relation_type", "related_to")),
                strength=value.get("strength", 1.0),
                attributes=value.get("attributes", {}),
            )
            await self._memory.store_relation(relation)

    async def get(self, key: str) -> Optional[MemoryResult]:
        """Get relationship by ID.

        Note: Direct relation lookup not supported, returns None.
        """
        return None

    def is_available(self) -> bool:
        """Check if graph memory is available."""
        return self._available and getattr(self._memory, "_initialized", False)


# =============================================================================
# Tool Results Memory Adapter
# =============================================================================


class ToolResultsMemoryAdapter:
    """Adapter for accessing historical tool results from ConversationStore.

    Provides specialized access to tool call/result messages for context
    about previous tool executions.

    Example:
        store = ConversationStore()
        adapter = ToolResultsMemoryAdapter(store, session_id="session_123")

        # Search for read_file tool results
        results = await adapter.search(
            MemoryQuery(query="config", filters={"tool_names": ["read_file"]})
        )
    """

    def __init__(
        self,
        conversation_store: "ConversationStore",
        session_id: Optional[str] = None,
    ):
        """Initialize adapter.

        Args:
            conversation_store: ConversationStore instance
            session_id: Optional session ID for queries
        """
        self._store = conversation_store
        self._session_id = session_id
        self._available = True

    @property
    def memory_type(self) -> MemoryType:
        """Return CODE memory type (tool results are often code-related)."""
        return MemoryType.CODE

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID for queries."""
        self._session_id = session_id

    async def search(self, query: MemoryQuery) -> List[MemoryResult]:
        """Search tool results.

        Args:
            query: Search query with optional tool_names filter

        Returns:
            List of MemoryResult with tool result content
        """
        session_id = query.session_id or self._session_id
        if not session_id:
            return []

        try:
            # Get tool name filter
            tool_names = None
            if query.filters and "tool_names" in query.filters:
                tool_names = query.filters["tool_names"]

            tool_results = self._store.get_historical_tool_results(
                session_id=session_id,
                tool_names=tool_names,
                limit=query.limit * 2,  # Get more for filtering
            )

            results = []
            query_lower = query.query.lower()

            for message in tool_results:
                # Check if query matches content
                content_lower = message.content.lower()
                if query_lower and query_lower not in content_lower:
                    continue

                # Calculate relevance based on match quality
                if query_lower == content_lower:
                    relevance = 1.0
                elif query_lower in content_lower:
                    # Relevance based on how much of content is the query
                    relevance = min(0.9, len(query_lower) / len(content_lower) + 0.3)
                else:
                    relevance = 0.3

                results.append(
                    MemoryResult(
                        source=MemoryType.CODE,
                        content={
                            "tool_name": message.tool_name,
                            "tool_call_id": message.tool_call_id,
                            "result": message.content,
                        },
                        relevance=relevance,
                        id=message.id,
                        metadata={
                            "tool_name": message.tool_name,
                            "tool_call_id": message.tool_call_id,
                        },
                        timestamp=message.timestamp.timestamp(),
                    )
                )

            # Sort by relevance and limit
            results.sort(key=lambda r: r.relevance, reverse=True)
            return results[: query.limit]

        except Exception as e:
            logger.warning(f"ToolResultsMemoryAdapter search failed: {e}")
            return []

    async def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store tool result (delegates to conversation store)."""
        pass  # Tool results are stored via normal message flow

    async def get(self, key: str) -> Optional[MemoryResult]:
        """Get tool result by ID."""
        return None  # Direct lookup not supported

    def is_available(self) -> bool:
        """Check if tool results are available."""
        return self._available and self._store is not None


# =============================================================================
# Factory Functions
# =============================================================================


def create_entity_adapter(entity_memory: "EntityMemory") -> EntityMemoryAdapter:
    """Create an EntityMemoryAdapter.

    Args:
        entity_memory: EntityMemory instance

    Returns:
        Configured EntityMemoryAdapter
    """
    return EntityMemoryAdapter(entity_memory)


def create_conversation_adapter(
    conversation_store: "ConversationStore",
    session_id: Optional[str] = None,
) -> ConversationMemoryAdapter:
    """Create a ConversationMemoryAdapter.

    Args:
        conversation_store: ConversationStore instance
        session_id: Optional session ID

    Returns:
        Configured ConversationMemoryAdapter
    """
    return ConversationMemoryAdapter(conversation_store, session_id)


def create_graph_adapter(entity_memory: "EntityMemory") -> GraphMemoryAdapter:
    """Create a GraphMemoryAdapter.

    Args:
        entity_memory: EntityMemory instance (contains graph)

    Returns:
        Configured GraphMemoryAdapter
    """
    return GraphMemoryAdapter(entity_memory)


def create_tool_results_adapter(
    conversation_store: "ConversationStore",
    session_id: Optional[str] = None,
) -> ToolResultsMemoryAdapter:
    """Create a ToolResultsMemoryAdapter.

    Args:
        conversation_store: ConversationStore instance
        session_id: Optional session ID

    Returns:
        Configured ToolResultsMemoryAdapter
    """
    return ToolResultsMemoryAdapter(conversation_store, session_id)


__all__ = [
    # Adapters
    "EntityMemoryAdapter",
    "ConversationMemoryAdapter",
    "GraphMemoryAdapter",
    "ToolResultsMemoryAdapter",
    # Factory functions
    "create_entity_adapter",
    "create_conversation_adapter",
    "create_graph_adapter",
    "create_tool_results_adapter",
]
