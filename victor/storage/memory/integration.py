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

"""Integration layer between EntityMemory and ConversationStore.

Provides automatic entity extraction from conversation messages
and entity-enhanced context retrieval.

Example:
    from victor.storage.memory.integration import EntityMemoryIntegration

    integration = EntityMemoryIntegration(entity_memory, conversation_store)
    await integration.process_message("Let's fix the UserAuth class")

    # Entities are automatically extracted
    entities = await integration.get_session_entities()
    # Returns [Entity(name="UserAuth", type=CLASS, ...)]
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from victor.storage.memory.entity_memory import EntityMemory, EntityMemoryConfig
from victor.storage.memory.entity_types import Entity, EntityType
from victor.storage.memory.extractors.composite import (
    CompositeExtractor,
    create_default_extractor,
)

logger = logging.getLogger(__name__)


class EntityMemoryIntegration:
    """Integrates EntityMemory with conversation systems.

    This class provides methods to:
    - Extract entities from messages
    - Store entities with session context
    - Retrieve entities relevant to conversation context
    - Update entity mentions based on conversation flow
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        auto_extract: bool = True,
        min_confidence: float = 0.5,
    ):
        """Initialize entity memory integration.

        Args:
            db_path: Path to entity memory database
            auto_extract: Automatically extract entities from messages
            min_confidence: Minimum confidence for entity storage
        """
        self._entity_memory = EntityMemory(
            config=EntityMemoryConfig(
                db_path=db_path,
                embedding_enabled=True,
                auto_extract=auto_extract,
            )
        )
        self._extractor = create_default_extractor()
        self._min_confidence = min_confidence
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the entity memory system."""
        if self._initialized:
            return
        await self._entity_memory.initialize()
        self._initialized = True

    async def process_message(
        self,
        content: str,
        session_id: str,
        message_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[Entity]:
        """Process a message and extract entities.

        Args:
            content: Message content
            session_id: Session identifier
            message_id: Optional message ID for source tracking
            role: Message role (user, assistant, etc.)

        Returns:
            List of extracted entities
        """
        if not self._initialized:
            await self.initialize()

        # Extract entities from content
        result = await self._extractor.extract(
            content,
            source=f"session:{session_id}:msg:{message_id or 'unknown'}",
            context={"role": role, "session_id": session_id},
        )

        # Filter by confidence
        entities = [e for e in result.entities if e.confidence >= self._min_confidence]

        # Store entities
        for entity in entities:
            # Check for existing entity to update mentions
            existing = await self._entity_memory.get(entity.id)
            if existing:
                await self._entity_memory.increment_mentions(entity.id)
            else:
                await self._entity_memory.store(entity)

        # Store relations
        for relation in result.relations:
            await self._entity_memory.store_relation(relation)

        return entities

    async def get_context_entities(
        self,
        query: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10,
    ) -> List[Entity]:
        """Get entities relevant to a query.

        Args:
            query: Search query or context
            entity_types: Filter by entity types
            limit: Maximum entities to return

        Returns:
            List of relevant entities
        """
        if not self._initialized:
            await self.initialize()

        return await self._entity_memory.search(
            query,
            entity_types=entity_types,
            limit=limit,
        )

    async def get_session_entities(self) -> List[Entity]:
        """Get all entities from current session.

        Returns:
            List of session entities
        """
        if not self._initialized:
            await self.initialize()

        return await self._entity_memory.get_session_entities()

    async def get_entity_context(
        self,
        entity_names: List[str],
        include_related: bool = True,
        max_related: int = 5,
    ) -> Dict[str, Any]:
        """Get rich context for specified entities.

        Args:
            entity_names: Names of entities to get context for
            include_related: Include related entities
            max_related: Maximum related entities per entity

        Returns:
            Dictionary with entity details and relationships
        """
        if not self._initialized:
            await self.initialize()

        context: Dict[str, Any] = {
            "entities": [],
            "relationships": [],
        }

        for name in entity_names:
            # Search for entity
            entities = await self._entity_memory.search(name, limit=1)
            if not entities:
                continue

            entity = entities[0]
            entity_info = {
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type.value,
                "description": entity.description,
                "mentions": entity.mentions,
                "confidence": entity.confidence,
            }
            context["entities"].append(entity_info)

            # Get related entities
            if include_related:
                related = await self._entity_memory.get_related(entity.id, limit=max_related)
                for rel_entity, relation in related:
                    context["relationships"].append(
                        {
                            "source": entity.name,
                            "target": rel_entity.name,
                            "type": relation.relation_type.value,
                            "strength": relation.strength,
                        }
                    )

        return context

    async def build_entity_prompt_context(
        self,
        recent_messages: List[str],
        max_entities: int = 10,
    ) -> str:
        """Build entity context for system prompt enhancement.

        Extracts key entities from recent messages and formats
        them as context for the LLM.

        Args:
            recent_messages: Recent conversation messages
            max_entities: Maximum entities to include

        Returns:
            Formatted entity context string
        """
        if not self._initialized:
            await self.initialize()

        # Get recent entities from session
        session_entities = await self.get_session_entities()

        # Sort by mentions (most discussed first)
        session_entities.sort(key=lambda e: e.mentions, reverse=True)
        top_entities = session_entities[:max_entities]

        if not top_entities:
            return ""

        # Format entity context
        lines = ["## Entities Discussed"]
        for entity in top_entities:
            type_str = entity.entity_type.value.title()
            desc = entity.description or "No description"
            lines.append(f"- **{entity.name}** ({type_str}): {desc}")

        return "\n".join(lines)

    async def clear_session(self) -> None:
        """Clear current session entities."""
        await self._entity_memory.clear_session()

    @property
    def entity_memory(self) -> EntityMemory:
        """Access underlying EntityMemory for advanced operations."""
        return self._entity_memory


def create_entity_integration(
    project_path: Optional[Path] = None,
) -> EntityMemoryIntegration:
    """Factory to create entity memory integration.

    Args:
        project_path: Project path for database location

    Returns:
        Configured EntityMemoryIntegration
    """
    if project_path:
        db_path = str(project_path / ".victor" / "entities.db")
    else:
        from victor.config.settings import get_project_paths

        paths = get_project_paths()
        db_path = str(paths.project_victor_dir / "entities.db")

    return EntityMemoryIntegration(db_path=db_path)
