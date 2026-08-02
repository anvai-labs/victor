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

"""Tree-sitter based entity extractor for accurate code parsing.

Uses Victor's existing Tree-sitter infrastructure to extract code entities
with high accuracy through AST parsing rather than regex patterns.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from victor.storage.memory.entity_types import (
    Entity,
    EntityRelation,
    EntityType,
    RelationType,
)
from victor.storage.memory.extractors.base import EntityExtractor, ExtractionResult

logger = logging.getLogger(__name__)


class TreeSitterEntityExtractor(EntityExtractor):
    """Extract code entities using Tree-sitter AST parsing.

    Leverages Victor's TreeSitterExtractor for accurate code parsing,
    converting extracted symbols and edges to Entity Memory format.
    """

    def __init__(self, auto_discover_plugins: bool = True):
        """Initialize the Tree-sitter entity extractor.

        Args:
            auto_discover_plugins: Whether to auto-discover language plugins
        """
        self._extractor = None
        self._auto_discover = auto_discover_plugins

    @property
    def name(self) -> str:
        """Get extractor name."""
        return "tree_sitter"

    @property
    def supported_types(self) -> Set[EntityType]:
        """Get supported entity types."""
        return {
            EntityType.FUNCTION,
            EntityType.CLASS,
            EntityType.MODULE,
            EntityType.FILE,
            EntityType.VARIABLE,
        }

    def _get_extractor(self):
        """Lazily resolve the Tree-sitter extractor capability.

        Returns the enhanced provider, or ``None`` when the capability is not
        registered. Returning ``None`` (rather than raising) keeps the absence of an
        optional dependency on the normal control-flow path — the graceful-degradation
        pattern — so genuine extraction errors are not conflated with "not installed".
        """
        if self._extractor is None:
            from victor.core.capability_registry import CapabilityRegistry
            from victor.framework.vertical_protocols import TreeSitterExtractorProtocol

            registry = CapabilityRegistry.get_instance()
            provider = registry.get(TreeSitterExtractorProtocol)
            if provider is not None and registry.is_enhanced(TreeSitterExtractorProtocol):
                self._extractor = provider
            else:
                logger.debug("Tree-sitter extractor not available: no enhanced provider registered")
                return None
        return self._extractor

    def is_available(self) -> bool:
        """Return whether the Tree-sitter extractor capability is registered."""
        return self._get_extractor() is not None

    async def extract(
        self,
        content: str,
        source: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """Extract entities from code using Tree-sitter.

        Args:
            content: Source code content
            source: Source file path (required for language detection)
            context: Optional context with 'language' key

        Returns:
            ExtractionResult with entities and relations
        """
        entities: List[Entity] = []
        relations: List[EntityRelation] = []

        # Need a file path for Tree-sitter language detection
        if source is None:
            # Try to get from context
            source = context.get("file_path") if context else None

        if source is None:
            # Fall back to parsing content inline (limited support)
            return await self._extract_inline(content, context)

        # Optional-dependency check on the normal path: if the capability is not
        # registered, degrade to an empty result instead of routing through an exception.
        extractor = self._get_extractor()
        if extractor is None:
            logger.debug("Tree-sitter extractor unavailable; returning empty result")
            return ExtractionResult(entities=[], relations=[])

        try:
            # Determine language
            file_path = Path(source)
            language = context.get("language") if context else None

            if language is None:
                language = extractor.detect_language(file_path)

            if language is None:
                logger.debug(f"Could not detect language for {source}")
                return ExtractionResult(entities=[], relations=[])

            # Write content to temp file or use existing file
            if file_path.exists():
                # Extract from existing file
                symbols, edges = extractor.extract_all(file_path, language)
            else:
                # Write to temp file for parsing
                import tempfile

                suffix = file_path.suffix or self._get_extension_for_language(language)
                with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
                    f.write(content)
                    temp_path = Path(f.name)

                try:
                    symbols, edges = extractor.extract_all(temp_path, language)
                finally:
                    temp_path.unlink(missing_ok=True)

            # Convert symbols to entities
            entity_map: Dict[str, Entity] = {}

            for symbol in symbols:
                entity_type = self._symbol_type_to_entity_type(symbol.type)
                if entity_type:
                    entity = Entity.create(
                        name=symbol.name,
                        entity_type=entity_type,
                        confidence=0.95,  # High confidence from AST
                        source=str(source),
                    )
                    # Add line number to attributes
                    entity.attributes["line_number"] = symbol.line_number
                    if symbol.end_line:
                        entity.attributes["end_line"] = symbol.end_line
                    if symbol.parent_symbol:
                        entity.attributes["parent"] = symbol.parent_symbol

                    entities.append(entity)
                    entity_map[symbol.name] = entity

            # Convert edges to relations
            for edge in edges:
                relation_type = self._edge_type_to_relation_type(edge.edge_type)
                if relation_type:
                    # Find source and target entities
                    source_entity = entity_map.get(edge.source)
                    target_entity = entity_map.get(edge.target)

                    # Create placeholder entities if not found
                    if source_entity is None:
                        source_entity = Entity.create(
                            name=edge.source,
                            entity_type=EntityType.FUNCTION,  # Assume function for calls
                            source=str(source),
                        )
                        entities.append(source_entity)
                        entity_map[edge.source] = source_entity

                    if target_entity is None:
                        target_entity = Entity.create(
                            name=edge.target,
                            entity_type=EntityType.FUNCTION,  # Assume function for calls
                            source=str(source),
                        )
                        entities.append(target_entity)
                        entity_map[edge.target] = target_entity

                    relation = EntityRelation(
                        source_id=source_entity.id,
                        target_id=target_entity.id,
                        relation_type=relation_type,
                        strength=0.9,  # High confidence from AST
                    )
                    relations.append(relation)

            # Add file entity
            file_entity = Entity.create(
                name=file_path.name,
                entity_type=EntityType.FILE,
                source=str(source),
                confidence=1.0,
            )
            file_entity.attributes["language"] = language
            file_entity.attributes["path"] = str(source)
            entities.append(file_entity)

            # Add CONTAINS relations for file -> symbols
            for entity in list(entities):
                if entity.entity_type in (EntityType.CLASS, EntityType.FUNCTION):
                    relation = EntityRelation(
                        source_id=file_entity.id,
                        target_id=entity.id,
                        relation_type=RelationType.CONTAINS,
                        strength=1.0,
                    )
                    relations.append(relation)

            return ExtractionResult(entities=entities, relations=relations)

        except Exception as e:
            logger.warning(f"Tree-sitter extraction failed: {e}")
            return ExtractionResult(entities=[], relations=[])

    async def _extract_inline(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """Extract entities from inline code without file path.

        Uses language from context or tries common languages.
        """
        language = context.get("language") if context else None

        if language is None:
            # Try to detect from content
            if "def " in content and ":" in content:
                language = "python"
            elif "function " in content or "=>" in content:
                language = "javascript"
            elif "class " in content and "{" in content:
                language = "java"
            else:
                language = "python"  # Default

        # Write to temp file with appropriate extension
        import tempfile

        suffix = self._get_extension_for_language(language)

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
                f.write(content)
                temp_path = Path(f.name)

            try:
                extractor = self._get_extractor()
                if extractor is None:
                    logger.debug("Tree-sitter extractor unavailable; returning empty result")
                    return ExtractionResult(entities=[], relations=[])
                symbols, edges = extractor.extract_all(temp_path, language)

                entities = []
                for symbol in symbols:
                    entity_type = self._symbol_type_to_entity_type(symbol.type)
                    if entity_type:
                        entity = Entity.create(
                            name=symbol.name,
                            entity_type=entity_type,
                            confidence=0.95,
                            source="inline",
                        )
                        entity.attributes["line_number"] = symbol.line_number
                        entities.append(entity)

                return ExtractionResult(entities=entities, relations=[])

            finally:
                temp_path.unlink(missing_ok=True)

        except Exception as e:
            logger.debug(f"Inline extraction failed: {e}")
            return ExtractionResult(entities=[], relations=[])

    def _symbol_type_to_entity_type(self, symbol_type: str) -> Optional[EntityType]:
        """Convert Tree-sitter symbol type to EntityType."""
        mapping = {
            "class": EntityType.CLASS,
            "function": EntityType.FUNCTION,
            "method": EntityType.FUNCTION,
            "module": EntityType.MODULE,
            "variable": EntityType.VARIABLE,
            "constant": EntityType.VARIABLE,
            "interface": EntityType.CLASS,
            "struct": EntityType.CLASS,
            "enum": EntityType.CLASS,
            "trait": EntityType.CLASS,
        }
        return mapping.get(symbol_type.lower())

    def _edge_type_to_relation_type(self, edge_type: str) -> Optional[RelationType]:
        """Convert Tree-sitter edge type to RelationType."""
        mapping = {
            "CALLS": RelationType.CALLS,
            "DATA_DEP": RelationType.DATA_DEP,
            "CONTROL_DEP": RelationType.CONTROL_DEP,
            "INHERITS": RelationType.EXTENDS,
            "IMPLEMENTS": RelationType.IMPLEMENTS,
            "COMPOSITION": RelationType.CONTAINS,
        }
        return mapping.get(edge_type)

    def _get_extension_for_language(self, language: str) -> str:
        """Get file extension for language."""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "tsx": ".tsx",
            "java": ".java",
            "go": ".go",
            "rust": ".rs",
            "c": ".c",
            "cpp": ".cpp",
            "c_sharp": ".cs",
            "ruby": ".rb",
            "php": ".php",
            "kotlin": ".kt",
            "swift": ".swift",
            "scala": ".scala",
        }
        return extensions.get(language, ".txt")


class TreeSitterFileExtractor(EntityExtractor):
    """Extract entities from source files using Tree-sitter.

    Designed for batch processing of entire files or directories.
    """

    def __init__(
        self,
        include_references: bool = False,
        auto_discover_plugins: bool = True,
    ):
        """Initialize the file extractor.

        Args:
            include_references: Whether to extract symbol references
            auto_discover_plugins: Whether to auto-discover language plugins
        """
        self._inner = TreeSitterEntityExtractor(auto_discover_plugins)
        self._include_references = include_references

    @property
    def name(self) -> str:
        """Get extractor name."""
        return "tree_sitter_file"

    @property
    def supported_types(self) -> Set[EntityType]:
        """Get supported entity types."""
        return self._inner.supported_types

    async def extract(
        self,
        content: str,
        source: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """Extract entities from file content."""
        return await self._inner.extract(content, source, context)

    async def extract_file(self, file_path: Path) -> ExtractionResult:
        """Extract entities from a source file.

        Args:
            file_path: Path to source file

        Returns:
            ExtractionResult with entities and relations
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            return await self.extract(
                content=content,
                source=str(file_path),
                context={"file_path": str(file_path)},
            )
        except Exception as e:
            logger.warning(f"Failed to extract from {file_path}: {e}")
            return ExtractionResult(entities=[], relations=[])

    async def extract_directory(
        self,
        directory: Path,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
    ) -> ExtractionResult:
        """Extract entities from all source files in a directory.

        Args:
            directory: Directory to scan
            recursive: Whether to scan recursively
            file_patterns: File glob patterns (e.g., ["*.py", "*.js"])

        Returns:
            Combined ExtractionResult
        """
        all_entities: List[Entity] = []
        all_relations: List[EntityRelation] = []

        if file_patterns is None:
            file_patterns = [
                "*.py",
                "*.js",
                "*.ts",
                "*.tsx",
                "*.java",
                "*.go",
                "*.rs",
                "*.rb",
                "*.c",
                "*.cpp",
                "*.h",
                "*.hpp",
            ]

        for pattern in file_patterns:
            if recursive:
                files = directory.rglob(pattern)
            else:
                files = directory.glob(pattern)

            for file_path in files:
                if file_path.is_file():
                    result = await self.extract_file(file_path)
                    all_entities.extend(result.entities)
                    all_relations.extend(result.relations)

        # Add module entity for directory
        module_entity = Entity.create(
            name=directory.name,
            entity_type=EntityType.MODULE,
            source=str(directory),
            confidence=1.0,
        )
        module_entity.attributes["path"] = str(directory)
        all_entities.append(module_entity)

        return ExtractionResult(entities=all_entities, relations=all_relations)
