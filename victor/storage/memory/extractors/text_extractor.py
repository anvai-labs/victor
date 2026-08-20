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

"""Text entity extractor for natural language content.

Extracts entities like people, organizations, technologies,
concepts, and projects from conversational text.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

from victor.storage.memory.entity_types import (
    Entity,
    EntityRelation,
    EntityType,
    RelationType,
)
from victor.storage.memory.extractors.base import EntityExtractor, ExtractionResult

logger = logging.getLogger(__name__)


# Known technology terms (expandable)
KNOWN_TECHNOLOGIES = {
    # Languages
    "python",
    "javascript",
    "typescript",
    "rust",
    "go",
    "golang",
    "java",
    "c++",
    "cpp",
    "c#",
    "csharp",
    "ruby",
    "php",
    "swift",
    "kotlin",
    "scala",
    # Frameworks
    "react",
    "vue",
    "angular",
    "django",
    "flask",
    "fastapi",
    "express",
    "spring",
    "rails",
    "laravel",
    "nextjs",
    "nuxt",
    "svelte",
    # Libraries
    "numpy",
    "pandas",
    "tensorflow",
    "pytorch",
    "keras",
    "scikit-learn",
    "langchain",
    "llamaindex",
    "transformers",
    "huggingface",
    # Databases
    "postgresql",
    "postgres",
    "mysql",
    "mongodb",
    "redis",
    "elasticsearch",
    "sqlite",
    "duckdb",
    "lancedb",
    "pinecone",
    "weaviate",
    "chromadb",
    # Tools
    "docker",
    "kubernetes",
    "k8s",
    "git",
    "github",
    "gitlab",
    "jenkins",
    "terraform",
    "ansible",
    "aws",
    "azure",
    "gcp",
    "vercel",
    "netlify",
    # AI/ML
    "openai",
    "anthropic",
    "claude",
    "gpt",
    "gpt-4",
    "llama",
    "mistral",
    "gemini",
    "ollama",
    "deepseek",
    "groq",
    "together",
    "replicate",
    # Protocols
    "rest",
    "graphql",
    "grpc",
    "websocket",
    "http",
    "https",
    "mcp",
}

# Organization patterns
ORGANIZATION_PATTERNS = [
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Co|Technologies|Labs|AI))\b",
    r"\b((?:Open|Deep|Neural|Cloud|Data|Meta|Micro|Anthropic|Google|Microsoft|Amazon|Apple)[\w]*)\b",
]

# Project/repository patterns
PROJECT_PATTERNS = [
    r"(?:repo|repository|project):\s*([a-zA-Z][\w-]+)",
    r"github\.com/[\w-]+/([a-zA-Z][\w-]+)",
    r"(?:working on|building|developing)\s+([A-Z][a-zA-Z]+)",
]

# Concept patterns
CONCEPT_PATTERNS = [
    r"(?:implement|use|apply)\s+(?:the\s+)?([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s+pattern",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:architecture|pattern|approach|design)",
    r"(?:using|with)\s+([a-zA-Z]+)\s+(?:approach|method|technique|strategy)",
]

# Requirement/feature patterns
REQUIREMENT_PATTERNS = [
    r"(?:need to|should|must)\s+([a-z]+(?:\s+[a-z]+){0,3})",
    r"(?:feature|requirement):\s*([^.!?]+)",
    r"(?:add|implement|create)\s+(?:a\s+)?([a-z]+(?:\s+[a-z]+){0,2})\s+(?:feature|functionality)",
]

# Bug patterns
BUG_PATTERNS = [
    r"(?:bug|issue|error|problem):\s*([^.!?]+)",
    r"(?:fix|fixing|fixed)\s+(?:the\s+)?([a-z]+(?:\s+[a-z]+){0,3})\s+(?:bug|issue|error)",
]


class TextEntityExtractor(EntityExtractor):
    """Extractor for natural language text entities.

    Identifies people, organizations, technologies, concepts,
    and other entities from conversational content.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        custom_technologies: Optional[Set[str]] = None,
    ):
        """Initialize text extractor.

        Args:
            min_confidence: Minimum confidence threshold
            custom_technologies: Additional technology terms to recognize
        """
        self._min_confidence = min_confidence
        self._technologies = KNOWN_TECHNOLOGIES.copy()
        if custom_technologies:
            self._technologies.update(custom_technologies)

    @property
    def name(self) -> str:
        return "text_extractor"

    @property
    def supported_types(self) -> Set[EntityType]:
        return {
            EntityType.PERSON,
            EntityType.ORGANIZATION,
            EntityType.TECHNOLOGY,
            EntityType.CONCEPT,
            EntityType.PROJECT,
            EntityType.REQUIREMENT,
            EntityType.FEATURE,
            EntityType.BUG,
        }

    async def extract(
        self,
        content: str,
        source: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """Extract text entities from content.

        Args:
            content: Natural language text
            source: Source identifier
            context: Additional context

        Returns:
            ExtractionResult with text entities
        """
        entities: List[Entity] = []
        relations: List[EntityRelation] = []

        # Extract technologies
        entities.extend(self._extract_technologies(content, source))

        # Extract organizations
        entities.extend(self._extract_organizations(content, source))

        # Extract projects
        entities.extend(self._extract_projects(content, source))

        # Extract concepts
        entities.extend(self._extract_concepts(content, source))

        # Extract requirements/features
        entities.extend(self._extract_requirements(content, source))

        # Extract bugs
        entities.extend(self._extract_bugs(content, source))

        # Infer relationships
        relations = self._infer_relationships(content, entities)

        # Deduplicate and filter
        entities = self._deduplicate_entities(entities)
        entities = [e for e in entities if e.confidence >= self._min_confidence]

        return ExtractionResult(
            entities=entities,
            relations=relations,
            confidence=self._calculate_overall_confidence(entities),
            metadata={"extractor": self.name},
        )

    def _extract_technologies(self, content: str, source: Optional[str]) -> List[Entity]:
        """Extract technology entities."""
        entities: List[Entity] = []
        content_lower = content.lower()

        for tech in self._technologies:
            # Use word boundary matching
            pattern = rf"\b{re.escape(tech)}\b"
            if re.search(pattern, content_lower, re.IGNORECASE):
                # Count mentions for confidence
                mentions = len(re.findall(pattern, content_lower, re.IGNORECASE))
                confidence = min(0.9, 0.6 + (mentions * 0.1))

                entity = Entity.create(
                    name=tech.title() if len(tech) > 3 else tech.upper(),
                    entity_type=EntityType.TECHNOLOGY,
                    description=f"Technology: {tech}",
                    source=source,
                    confidence=confidence,
                    attributes={"mentions": mentions},
                )
                entity.mentions = mentions
                entities.append(entity)

        return entities

    def _extract_organizations(self, content: str, source: Optional[str]) -> List[Entity]:
        """Extract organization entities."""
        entities: List[Entity] = []

        for pattern in ORGANIZATION_PATTERNS:
            matches = re.finditer(pattern, content)
            for match in matches:
                name = match.group(1).strip()
                if len(name) < 3:
                    continue

                entity = Entity.create(
                    name=name,
                    entity_type=EntityType.ORGANIZATION,
                    description=f"Organization: {name}",
                    source=source,
                    confidence=0.7,
                )
                entities.append(entity)

        return entities

    def _extract_projects(self, content: str, source: Optional[str]) -> List[Entity]:
        """Extract project entities."""
        entities: List[Entity] = []

        for pattern in PROJECT_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if len(name) < 2:
                    continue

                entity = Entity.create(
                    name=name,
                    entity_type=EntityType.PROJECT,
                    description=f"Project: {name}",
                    source=source,
                    confidence=0.65,
                )
                entities.append(entity)

        return entities

    def _extract_concepts(self, content: str, source: Optional[str]) -> List[Entity]:
        """Extract concept entities."""
        entities: List[Entity] = []

        for pattern in CONCEPT_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if len(name) < 3:
                    continue

                entity = Entity.create(
                    name=name.title(),
                    entity_type=EntityType.CONCEPT,
                    description=f"Concept/Pattern: {name}",
                    source=source,
                    confidence=0.6,
                )
                entities.append(entity)

        return entities

    def _extract_requirements(self, content: str, source: Optional[str]) -> List[Entity]:
        """Extract requirement and feature entities."""
        entities: List[Entity] = []

        for pattern in REQUIREMENT_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if len(name) < 5 or len(name) > 100:
                    continue

                entity = Entity.create(
                    name=name.capitalize(),
                    entity_type=EntityType.REQUIREMENT,
                    description=f"Requirement: {name}",
                    source=source,
                    confidence=0.55,
                )
                entities.append(entity)

        return entities

    def _extract_bugs(self, content: str, source: Optional[str]) -> List[Entity]:
        """Extract bug entities."""
        entities: List[Entity] = []

        for pattern in BUG_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if len(name) < 5 or len(name) > 100:
                    continue

                entity = Entity.create(
                    name=name.capitalize(),
                    entity_type=EntityType.BUG,
                    description=f"Bug/Issue: {name}",
                    source=source,
                    confidence=0.6,
                )
                entities.append(entity)

        return entities

    def _infer_relationships(self, content: str, entities: List[Entity]) -> List[EntityRelation]:
        """Infer relationships between entities."""
        relations: List[EntityRelation] = []

        # Group entities by type for relationship inference
        techs = [e for e in entities if e.entity_type == EntityType.TECHNOLOGY]
        projects = [e for e in entities if e.entity_type == EntityType.PROJECT]
        concepts = [e for e in entities if e.entity_type == EntityType.CONCEPT]

        # Projects use technologies
        for project in projects:
            for tech in techs:
                relation = EntityRelation(
                    source_id=project.id,
                    target_id=tech.id,
                    relation_type=RelationType.DEPENDS_ON,
                    strength=0.5,
                )
                relations.append(relation)

        # Concepts relate to each other
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i + 1 :]:
                relation = EntityRelation(
                    source_id=concept1.id,
                    target_id=concept2.id,
                    relation_type=RelationType.RELATED_TO,
                    strength=0.4,
                )
                relations.append(relation)

        return relations

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities."""
        seen: Dict[str, Entity] = {}

        for entity in entities:
            if entity.id in seen:
                # Merge mentions
                seen[entity.id].mentions += entity.mentions
                if entity.confidence > seen[entity.id].confidence:
                    seen[entity.id].confidence = entity.confidence
            else:
                seen[entity.id] = entity

        return list(seen.values())

    def _calculate_overall_confidence(self, entities: List[Entity]) -> float:
        """Calculate overall extraction confidence."""
        if not entities:
            return 0.0
        return sum(e.confidence for e in entities) / len(entities)
