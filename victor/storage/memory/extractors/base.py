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

"""Base entity extractor protocol and shared utilities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from victor.storage.memory.entity_types import Entity, EntityRelation, EntityType


@dataclass
class ExtractionResult:
    """Result of entity extraction.

    Attributes:
        entities: List of extracted entities
        relations: List of inferred relationships
        confidence: Overall extraction confidence
        metadata: Additional extraction metadata
    """

    entities: List[Entity] = field(default_factory=list)
    relations: List[EntityRelation] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def merge(self, other: "ExtractionResult") -> "ExtractionResult":
        """Merge with another extraction result."""
        # Deduplicate entities by ID
        seen_ids: Set[str] = set()
        merged_entities: List[Entity] = []

        for entity in self.entities + other.entities:
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                merged_entities.append(entity)

        # Deduplicate relations by ID
        seen_rel_ids: Set[str] = set()
        merged_relations: List[EntityRelation] = []

        for relation in self.relations + other.relations:
            if relation.id not in seen_rel_ids:
                seen_rel_ids.add(relation.id)
                merged_relations.append(relation)

        return ExtractionResult(
            entities=merged_entities,
            relations=merged_relations,
            confidence=min(self.confidence, other.confidence),
            metadata={**self.metadata, **other.metadata},
        )

    def filter_by_type(self, entity_types: Set[EntityType]) -> "ExtractionResult":
        """Filter entities by type."""
        filtered = [e for e in self.entities if e.entity_type in entity_types]
        return ExtractionResult(
            entities=filtered,
            relations=self.relations,
            confidence=self.confidence,
            metadata=self.metadata,
        )

    def filter_by_confidence(self, min_confidence: float) -> "ExtractionResult":
        """Filter entities by minimum confidence."""
        filtered = [e for e in self.entities if e.confidence >= min_confidence]
        return ExtractionResult(
            entities=filtered,
            relations=self.relations,
            confidence=self.confidence,
            metadata=self.metadata,
        )


class EntityExtractor(ABC):
    """Abstract base class for entity extractors.

    Extractors analyze text/code and identify entities with their
    types, descriptions, and relationships.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Extractor identifier."""
        ...

    @property
    @abstractmethod
    def supported_types(self) -> Set[EntityType]:
        """Entity types this extractor can identify."""
        ...

    @abstractmethod
    async def extract(
        self,
        content: str,
        source: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """Extract entities from content.

        Args:
            content: Text or code to analyze
            source: Source identifier (file path, message ID, etc.)
            context: Additional context for extraction

        Returns:
            ExtractionResult with entities and relations
        """
        ...

    def can_extract(self, entity_type: EntityType) -> bool:
        """Check if this extractor supports a given entity type."""
        return entity_type in self.supported_types
