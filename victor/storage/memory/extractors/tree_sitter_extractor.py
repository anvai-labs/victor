# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Entity-memory projections backed by the shared victor-codegraph parser.

The historic class names remain API-compatible. They no longer own tree-sitter
discovery, temporary files, parsing, or relation inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from victor.core.codegraph_adapter import codegraph_available, codegraph_file_id, parse_code
from victor.storage.memory.entity_types import Entity, EntityRelation, EntityType, RelationType
from victor.storage.memory.extractors.base import EntityExtractor, ExtractionResult

logger = logging.getLogger(__name__)

_SYMBOL_TYPES = {
    "FILE": EntityType.FILE,
    "MODULE": EntityType.MODULE,
    "PACKAGE": EntityType.PACKAGE,
    "CLASS": EntityType.CLASS,
    "INTERFACE": EntityType.INTERFACE,
    "TRAIT": EntityType.CLASS,
    "STRUCT": EntityType.CLASS,
    "ENUM": EntityType.CLASS,
    "FUNCTION": EntityType.FUNCTION,
    "METHOD": EntityType.FUNCTION,
    "CONSTRUCTOR": EntityType.FUNCTION,
    "PROPERTY": EntityType.VARIABLE,
    "FIELD": EntityType.VARIABLE,
    "CONSTANT": EntityType.VARIABLE,
    "VARIABLE": EntityType.VARIABLE,
}

_RELATION_TYPES = {
    "CALLS": RelationType.CALLS,
    "EXTENDS": RelationType.EXTENDS,
    "IMPLEMENTS": RelationType.IMPLEMENTS,
    "IMPORTS": RelationType.IMPORTS,
    "DEPENDS_ON": RelationType.DEPENDS_ON,
    "CONTAINS": RelationType.CONTAINS,
    "REFERENCES": RelationType.REFERENCES,
}


class TreeSitterEntityExtractor(EntityExtractor):
    """Compatibility facade projecting canonical CodeGraph results into memory."""

    def __init__(self, auto_discover_plugins: bool = True) -> None:
        del auto_discover_plugins

    @property
    def name(self) -> str:
        return "tree_sitter"

    @property
    def supported_types(self) -> Set[EntityType]:
        return set(_SYMBOL_TYPES.values())

    def is_available(self) -> bool:
        return codegraph_available()

    async def extract(
        self,
        content: str,
        source: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        file_path = source or (context or {}).get("file_path") or "<inline>"
        language = (context or {}).get("language")
        parsed = parse_code(
            content,
            file_path=str(file_path),
            language=language,
            repo_root=(context or {}).get("repo_root"),
        )
        if parsed is None:
            return ExtractionResult(entities=[], relations=[])

        entities: list[Entity] = []
        by_symbol_id: dict[str, Entity] = {}
        for symbol in parsed.symbols:
            entity_type = _SYMBOL_TYPES.get(symbol.symbol_type.name)
            if entity_type is None:
                continue
            entity = Entity(
                id=symbol.id,
                name=symbol.simple_name,
                entity_type=entity_type,
                description=symbol.documentation,
                attributes={
                    "symbol_id": symbol.id,
                    "fully_qualified_name": symbol.fully_qualified_name,
                    "line_number": symbol.location.start_line,
                    "end_line": symbol.location.end_line,
                    "parent": symbol.scope_chain[-1] if symbol.scope_chain else None,
                    "language": symbol.language,
                    "signature": symbol.signature,
                    "identity_version": symbol.identity_version,
                },
                source=source or "inline",
                confidence=0.95,
            )
            entities.append(entity)
            by_symbol_id[symbol.id] = entity

        relations: list[EntityRelation] = []
        for relation in parsed.relations:
            relation_type = _RELATION_TYPES.get(relation.relation_type.name)
            if (
                relation_type is None
                or relation.from_symbol_id not in by_symbol_id
                or relation.to_symbol_id not in by_symbol_id
            ):
                continue
            relations.append(
                EntityRelation(
                    source_id=relation.from_symbol_id,
                    target_id=relation.to_symbol_id,
                    relation_type=relation_type,
                    strength=relation.confidence,
                    attributes={"provenance": relation.provenance},
                )
            )

        if source is not None:
            canonical_file_id = codegraph_file_id(parsed.file_path, parsed.language)
            if canonical_file_id is not None:
                file_entity = Entity(
                    id=canonical_file_id,
                    name=Path(source).name,
                    entity_type=EntityType.FILE,
                    source=source,
                    confidence=1.0,
                    attributes={
                        "symbol_id": canonical_file_id,
                        "language": parsed.language,
                        "path": parsed.file_path,
                        "identity_version": "v2",
                    },
                )
                entities.append(file_entity)
                nested_ids = {
                    relation.to_symbol_id
                    for relation in parsed.relations
                    if relation.relation_type.name == "CONTAINS"
                }
                for symbol_id in by_symbol_id.keys() - nested_ids:
                    relations.append(
                        EntityRelation(
                            source_id=file_entity.id,
                            target_id=symbol_id,
                            relation_type=RelationType.CONTAINS,
                        )
                    )

        return ExtractionResult(
            entities=entities,
            relations=relations,
            confidence=0.95 if entities else 0.0,
            metadata={
                "extractor": "victor_codegraph",
                "parse_status": parsed.status.value,
                "capability_tier": parsed.capability_tier.value,
            },
        )


class TreeSitterFileExtractor(EntityExtractor):
    """Compatibility batch facade around ``TreeSitterEntityExtractor``."""

    def __init__(
        self,
        include_references: bool = False,
        auto_discover_plugins: bool = True,
    ) -> None:
        del include_references
        self._inner = TreeSitterEntityExtractor(auto_discover_plugins)

    @property
    def name(self) -> str:
        return "tree_sitter_file"

    @property
    def supported_types(self) -> Set[EntityType]:
        return self._inner.supported_types

    async def extract(
        self,
        content: str,
        source: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        return await self._inner.extract(content, source, context)

    async def extract_file(self, file_path: Path) -> ExtractionResult:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            return ExtractionResult(entities=[], relations=[])
        return await self.extract(content=content, source=str(file_path))

    async def extract_directory(
        self,
        directory: Path,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
    ) -> ExtractionResult:
        patterns = file_patterns or [
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
        combined = ExtractionResult()
        for pattern in patterns:
            paths = directory.rglob(pattern) if recursive else directory.glob(pattern)
            for file_path in paths:
                if file_path.is_file():
                    combined = combined.merge(await self.extract_file(file_path))
        module = Entity.create(
            name=directory.name,
            entity_type=EntityType.MODULE,
            source=str(directory),
            confidence=1.0,
            attributes={"path": str(directory)},
        )
        combined.entities.append(module)
        return combined


__all__ = ["TreeSitterEntityExtractor", "TreeSitterFileExtractor"]
