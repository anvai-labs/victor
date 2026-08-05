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

"""Entity type definitions for the memory system.

Provides structured entity types for tracking people, organizations,
code elements, and concepts across conversations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib


class EntityType(str, Enum):
    """Types of entities that can be tracked.

    Categories:
    - People/Organizations: PERSON, ORGANIZATION, TEAM
    - Code Elements: FILE, FUNCTION, CLASS, MODULE, VARIABLE
    - Projects: PROJECT, REPOSITORY, PACKAGE
    - Concepts: CONCEPT, TECHNOLOGY, PATTERN, REQUIREMENT
    - Infrastructure: SERVICE, ENDPOINT, DATABASE
    """

    # People and organizations
    PERSON = "person"
    ORGANIZATION = "organization"
    TEAM = "team"

    # Code elements
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    VARIABLE = "variable"
    INTERFACE = "interface"

    # Project elements
    PROJECT = "project"
    REPOSITORY = "repository"
    PACKAGE = "package"
    DEPENDENCY = "dependency"

    # Concepts and patterns
    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    PATTERN = "pattern"
    REQUIREMENT = "requirement"
    BUG = "bug"
    FEATURE = "feature"

    # Infrastructure
    SERVICE = "service"
    ENDPOINT = "endpoint"
    DATABASE = "database"
    CONFIG = "config"

    # Generic
    OTHER = "other"


class RelationType(str, Enum):
    """Types of relationships between entities."""

    # Structural relationships
    CONTAINS = "contains"  # Parent contains child
    BELONGS_TO = "belongs_to"  # Child belongs to parent
    IMPORTS = "imports"  # A imports B
    IMPLEMENTS = "implements"  # Class implements interface
    EXTENDS = "extends"  # Class extends another
    DEPENDS_ON = "depends_on"  # A depends on B

    # Semantic relationships
    RELATED_TO = "related_to"  # General association
    SIMILAR_TO = "similar_to"  # Semantic similarity
    REFERENCES = "references"  # A references B
    USED_BY = "used_by"  # A is used by B

    # Code relationships.
    # NOTE: these values are intentionally UPPERCASE (unlike the lowercase semantic
    # relations above). They mirror the canonical program-analysis edge-type vocabulary
    # in ``victor/storage/graph/edge_types.py`` (EdgeType.CALLS="CALLS", CDG="CDG",
    # DDG_DEF_USE="DDG_DEF_USE") that the graph index emits and graph/code_search edge
    # filters match on. Do NOT lowercase them — that would break alignment with the graph
    # edge vocabulary and silently drop these relations from edge-type-filtered queries.
    CALLS = "CALLS"  # Function A calls function B (== EdgeType.CALLS)
    DATA_DEP = "DDG_DEF_USE"  # B depends on data from A (== EdgeType.DDG_DEF_USE)
    CONTROL_DEP = "CDG"  # B execution depends on A (== EdgeType.CDG)

    # Ownership/authorship
    CREATED_BY = "created_by"
    OWNED_BY = "owned_by"
    MAINTAINED_BY = "maintained_by"

    # Temporal relationships
    PRECEDED_BY = "preceded_by"
    FOLLOWED_BY = "followed_by"
    REPLACED_BY = "replaced_by"


def _generate_entity_id(entity_type: EntityType, name: str) -> str:
    """Generate a unique entity ID based on type and name."""
    content = f"{entity_type.value}:{name.lower()}"
    hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"ent_{hash_val}"


@dataclass
class Entity:
    """Represents an entity tracked in memory.

    Attributes:
        id: Unique identifier for this entity
        name: Display name of the entity
        entity_type: Type classification
        description: Optional description
        attributes: Key-value attributes
        source: Where this entity was extracted from
        confidence: Extraction confidence (0.0-1.0)
        mentions: Number of times mentioned
        first_seen: When first encountered
        last_seen: When last encountered
        embedding: Optional vector embedding for semantic search
    """

    id: str
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    confidence: float = 1.0
    mentions: int = 1
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    embedding: Optional[List[float]] = None

    @classmethod
    def create(
        cls,
        name: str,
        entity_type: EntityType,
        description: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        confidence: float = 1.0,
    ) -> "Entity":
        """Factory method to create a new entity with generated ID.

        Args:
            name: Entity name
            entity_type: Entity type
            description: Optional description
            attributes: Optional attributes dict
            source: Optional source identifier
            confidence: Extraction confidence

        Returns:
            New Entity instance
        """
        entity_id = _generate_entity_id(entity_type, name)
        return cls(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            description=description,
            attributes=attributes or {},
            source=source,
            confidence=confidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize entity to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "attributes": self.attributes,
            "source": self.source,
            "confidence": self.confidence,
            "mentions": self.mentions,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            # Embedding excluded for size
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Deserialize entity from dictionary."""
        first_seen = data.get("first_seen")
        if isinstance(first_seen, str):
            first_seen = datetime.fromisoformat(first_seen)
        elif first_seen is None:
            first_seen = datetime.now(timezone.utc)

        last_seen = data.get("last_seen")
        if isinstance(last_seen, str):
            last_seen = datetime.fromisoformat(last_seen)
        elif last_seen is None:
            last_seen = datetime.now(timezone.utc)

        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            description=data.get("description"),
            attributes=data.get("attributes", {}),
            source=data.get("source"),
            confidence=data.get("confidence", 1.0),
            mentions=data.get("mentions", 1),
            first_seen=first_seen,
            last_seen=last_seen,
            embedding=data.get("embedding"),
        )

    def merge_with(self, other: "Entity") -> "Entity":
        """Merge this entity with another (same entity, different observations).

        Args:
            other: Another entity observation to merge

        Returns:
            New merged entity
        """
        # Use higher confidence description
        description = (
            (other.description if other.confidence > self.confidence else self.description)
            or self.description
            or other.description
        )

        # Merge attributes
        merged_attrs = {**self.attributes, **other.attributes}

        return Entity(
            id=self.id,
            name=self.name,  # Keep original name
            entity_type=self.entity_type,
            description=description,
            attributes=merged_attrs,
            source=self.source or other.source,
            confidence=max(self.confidence, other.confidence),
            mentions=self.mentions + other.mentions,
            first_seen=min(self.first_seen, other.first_seen),
            last_seen=max(self.last_seen, other.last_seen),
            embedding=other.embedding or self.embedding,
        )


@dataclass
class EntityRelation:
    """Represents a relationship between two entities.

    Attributes:
        source_id: ID of the source entity
        target_id: ID of the target entity
        relation_type: Type of relationship
        strength: Relationship strength (0.0-1.0)
        attributes: Additional relationship attributes
        first_seen: When relationship was first observed
        last_seen: When relationship was last observed
    """

    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def id(self) -> str:
        """Generate unique relation ID."""
        content = f"{self.source_id}:{self.relation_type.value}:{self.target_id}"
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"rel_{hash_val}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize relation to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "strength": self.strength,
            "attributes": self.attributes,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityRelation":
        """Deserialize relation from dictionary."""
        first_seen = data.get("first_seen")
        if isinstance(first_seen, str):
            first_seen = datetime.fromisoformat(first_seen)
        elif first_seen is None:
            first_seen = datetime.now(timezone.utc)

        last_seen = data.get("last_seen")
        if isinstance(last_seen, str):
            last_seen = datetime.fromisoformat(last_seen)
        elif last_seen is None:
            last_seen = datetime.now(timezone.utc)

        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            strength=data.get("strength", 1.0),
            attributes=data.get("attributes", {}),
            first_seen=first_seen,
            last_seen=last_seen,
        )


# Entity type categories for filtering
CODE_ENTITY_TYPES = {
    EntityType.FILE,
    EntityType.FUNCTION,
    EntityType.CLASS,
    EntityType.MODULE,
    EntityType.VARIABLE,
    EntityType.INTERFACE,
}

PROJECT_ENTITY_TYPES = {
    EntityType.PROJECT,
    EntityType.REPOSITORY,
    EntityType.PACKAGE,
    EntityType.DEPENDENCY,
}

CONCEPT_ENTITY_TYPES = {
    EntityType.CONCEPT,
    EntityType.TECHNOLOGY,
    EntityType.PATTERN,
    EntityType.REQUIREMENT,
    EntityType.BUG,
    EntityType.FEATURE,
}

PEOPLE_ENTITY_TYPES = {
    EntityType.PERSON,
    EntityType.ORGANIZATION,
    EntityType.TEAM,
}
