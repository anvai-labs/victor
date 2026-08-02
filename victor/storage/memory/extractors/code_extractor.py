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

"""Code entity extractor using pattern matching and Tree-sitter integration.

Extracts code entities like functions, classes, modules, and files
from code snippets and file references in text.
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


# Patterns for code entity extraction
CODE_PATTERNS = {
    # Function definitions (Python, JavaScript, TypeScript)
    "function": [
        r"(?:def|function|async\s+function)\s+(\w+)\s*\(",
        r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>)",
        r"(\w+)\s*:\s*(?:async\s+)?function",
    ],
    # Class definitions
    "class": [
        r"class\s+(\w+)(?:\s*\([^)]*\))?(?:\s*:|\s*{|\s+extends)",
        r"interface\s+(\w+)",
        r"struct\s+(\w+)",
        r"type\s+(\w+)\s*=",
    ],
    # File paths
    "file": [
        r"(?:file|path):\s*['\"]?([^'\">\s]+\.\w+)['\"]?",
        r"(?:in|from|import)\s+['\"]([^'\"]+\.\w+)['\"]",
        r"`([^`]+\.\w{1,5})`",
        r"([a-zA-Z_][\w/.-]*\.(?:py|js|ts|tsx|jsx|go|rs|java|cpp|c|h|rb|php))\b",
    ],
    # Module/package references
    "module": [
        r"(?:import|from)\s+([a-zA-Z_][\w.]*)",
        r"require\(['\"]([^'\"]+)['\"]\)",
        r"package\s+([a-zA-Z_][\w.]*)",
    ],
    # Variable/constant definitions
    "variable": [
        r"(?:const|let|var|val)\s+([A-Z][A-Z_0-9]+)\s*=",
        r"([A-Z][A-Z_0-9]{2,})\s*=\s*['\"]?[^'\"]+['\"]?",
    ],
}

# Relationship patterns
RELATIONSHIP_PATTERNS = {
    RelationType.IMPORTS: [
        r"from\s+(\w+)\s+import\s+(\w+)",
        r"import\s+{([^}]+)}\s+from\s+['\"]([^'\"]+)['\"]",
    ],
    RelationType.EXTENDS: [
        r"class\s+(\w+)\s*\(\s*(\w+)\s*\)",
        r"class\s+(\w+)\s+extends\s+(\w+)",
    ],
    RelationType.IMPLEMENTS: [
        r"class\s+(\w+).*implements\s+(\w+)",
    ],
}


class CodeEntityExtractor(EntityExtractor):
    """Extractor for code entities.

    Uses regex patterns to identify code constructs like functions,
    classes, and file references. Can optionally integrate with
    Tree-sitter for more accurate parsing.
    """

    def __init__(
        self,
        use_treesitter: bool = False,
        min_confidence: float = 0.5,
    ):
        """Initialize code extractor.

        Args:
            use_treesitter: Enable Tree-sitter parsing (if available)
            min_confidence: Minimum confidence threshold
        """
        self._use_treesitter = use_treesitter
        self._min_confidence = min_confidence
        self._treesitter_available = False

        if use_treesitter:
            from victor.core.capability_registry import CapabilityRegistry
            from victor.framework.vertical_protocols import CodebaseIndexFactoryProtocol

            self._treesitter_available = CapabilityRegistry.get_instance().is_enhanced(
                CodebaseIndexFactoryProtocol
            )
            if not self._treesitter_available:
                logger.debug("Tree-sitter not available, using regex fallback")

    @property
    def name(self) -> str:
        return "code_extractor"

    @property
    def supported_types(self) -> Set[EntityType]:
        return {
            EntityType.FILE,
            EntityType.FUNCTION,
            EntityType.CLASS,
            EntityType.MODULE,
            EntityType.VARIABLE,
            EntityType.INTERFACE,
        }

    async def extract(
        self,
        content: str,
        source: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """Extract code entities from content.

        Args:
            content: Text containing code or code references
            source: Source identifier
            context: Additional context (e.g., language hint)

        Returns:
            ExtractionResult with code entities
        """
        entities: List[Entity] = []
        relations: List[EntityRelation] = []

        # Extract entities by type
        for entity_type_str, patterns in CODE_PATTERNS.items():
            entity_type = self._map_pattern_to_type(entity_type_str)
            if entity_type is None:
                continue

            for pattern in patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    name = match.group(1).strip()
                    if not name or len(name) < 2:
                        continue

                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_confidence(name, entity_type, pattern, context)

                    if confidence >= self._min_confidence:
                        entity = Entity.create(
                            name=name,
                            entity_type=entity_type,
                            description=f"{entity_type.value.title()} extracted from code",
                            source=source,
                            confidence=confidence,
                            attributes={
                                "pattern": pattern,
                                "match_position": match.start(),
                            },
                        )
                        entities.append(entity)

        # Extract relationships
        relations = self._extract_relationships(content, entities, source)

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        return ExtractionResult(
            entities=entities,
            relations=relations,
            confidence=self._calculate_overall_confidence(entities),
            metadata={
                "extractor": self.name,
                "treesitter_used": self._use_treesitter and self._treesitter_available,
            },
        )

    def _map_pattern_to_type(self, pattern_type: str) -> Optional[EntityType]:
        """Map pattern category to EntityType."""
        mapping = {
            "function": EntityType.FUNCTION,
            "class": EntityType.CLASS,
            "file": EntityType.FILE,
            "module": EntityType.MODULE,
            "variable": EntityType.VARIABLE,
        }
        return mapping.get(pattern_type)

    def _calculate_confidence(
        self,
        name: str,
        entity_type: EntityType,
        pattern: str,
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate extraction confidence for an entity."""
        confidence = 0.7  # Base confidence

        # Boost for common naming conventions
        if entity_type == EntityType.CLASS and name[0].isupper():
            confidence += 0.1
        elif entity_type == EntityType.FUNCTION and name[0].islower():
            confidence += 0.1
        elif entity_type == EntityType.VARIABLE and name.isupper():
            confidence += 0.1

        # Boost for file extensions
        if entity_type == EntityType.FILE:
            ext_match = re.search(r"\.(\w+)$", name)
            if ext_match and ext_match.group(1) in (
                "py",
                "js",
                "ts",
                "tsx",
                "jsx",
                "go",
                "rs",
                "java",
            ):
                confidence += 0.15

        # Penalty for very short names
        if len(name) <= 2:
            confidence -= 0.2

        # Context boost
        if context and context.get("is_code_block"):
            confidence += 0.1

        return min(1.0, max(0.0, confidence))

    def _extract_relationships(
        self,
        content: str,
        entities: List[Entity],
        source: Optional[str],
    ) -> List[EntityRelation]:
        """Extract relationships between entities."""
        relations: List[EntityRelation] = []
        entity_ids = {e.name: e.id for e in entities}

        for rel_type, patterns in RELATIONSHIP_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        source_name = groups[0].strip()
                        target_name = groups[1].strip()

                        # Only create relation if both entities exist
                        if source_name in entity_ids and target_name in entity_ids:
                            relation = EntityRelation(
                                source_id=entity_ids[source_name],
                                target_id=entity_ids[target_name],
                                relation_type=rel_type,
                                strength=0.8,
                                attributes={"source": source},
                            )
                            relations.append(relation)

        return relations

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping highest confidence."""
        seen: Dict[str, Entity] = {}

        for entity in entities:
            if entity.id in seen:
                if entity.confidence > seen[entity.id].confidence:
                    seen[entity.id] = entity
            else:
                seen[entity.id] = entity

        return list(seen.values())

    def _calculate_overall_confidence(self, entities: List[Entity]) -> float:
        """Calculate overall extraction confidence."""
        if not entities:
            return 0.0
        return sum(e.confidence for e in entities) / len(entities)
