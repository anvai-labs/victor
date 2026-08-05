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

"""Composite entity extractor combining multiple extractors."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from victor.storage.memory.entity_types import EntityType
from victor.storage.memory.extractors.base import EntityExtractor, ExtractionResult

logger = logging.getLogger(__name__)


class CompositeExtractor(EntityExtractor):
    """Combines multiple extractors for comprehensive entity extraction.

    Runs all configured extractors in parallel and merges their results,
    handling deduplication and confidence aggregation.

    Example:
        extractor = CompositeExtractor([
            CodeEntityExtractor(),
            TextEntityExtractor(),
        ])
        result = await extractor.extract(content)
    """

    def __init__(
        self,
        extractors: Optional[List[EntityExtractor]] = None,
        parallel: bool = True,
        dedup_strategy: str = "highest_confidence",
    ):
        """Initialize composite extractor.

        Args:
            extractors: List of extractors to combine
            parallel: Run extractors in parallel (faster)
            dedup_strategy: How to handle duplicates:
                - "highest_confidence": Keep entity with highest confidence
                - "merge": Merge attributes from all extractions
                - "first": Keep first extraction
        """
        self._extractors = extractors or []
        self._parallel = parallel
        self._dedup_strategy = dedup_strategy

    @property
    def name(self) -> str:
        return "composite_extractor"

    @property
    def supported_types(self) -> Set[EntityType]:
        """Union of all extractor supported types."""
        all_types: Set[EntityType] = set()
        for extractor in self._extractors:
            all_types.update(extractor.supported_types)
        return all_types

    def add_extractor(self, extractor: EntityExtractor) -> "CompositeExtractor":
        """Add an extractor to the composite.

        Args:
            extractor: Extractor to add

        Returns:
            Self for chaining
        """
        self._extractors.append(extractor)
        return self

    async def extract(
        self,
        content: str,
        source: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """Extract entities using all configured extractors.

        Args:
            content: Content to analyze
            source: Source identifier
            context: Additional context

        Returns:
            Merged ExtractionResult from all extractors
        """
        if not self._extractors:
            return ExtractionResult()

        if self._parallel:
            results = await self._extract_parallel(content, source, context)
        else:
            results = await self._extract_sequential(content, source, context)

        # Merge all results
        merged = ExtractionResult()
        for result in results:
            merged = merged.merge(result)

        # Apply deduplication strategy
        merged = self._apply_dedup_strategy(merged)

        # Add composite metadata
        merged.metadata["composite"] = True
        merged.metadata["extractors_used"] = [e.name for e in self._extractors]

        return merged

    async def _extract_parallel(
        self,
        content: str,
        source: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> List[ExtractionResult]:
        """Run extractors in parallel."""
        tasks = [extractor.extract(content, source, context) for extractor in self._extractors]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results: List[ExtractionResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Extractor {self._extractors[i].name} failed: {result}")
            else:
                valid_results.append(result)

        return valid_results

    async def _extract_sequential(
        self,
        content: str,
        source: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> List[ExtractionResult]:
        """Run extractors sequentially."""
        results: List[ExtractionResult] = []

        for extractor in self._extractors:
            try:
                result = await extractor.extract(content, source, context)
                results.append(result)
            except Exception as e:
                logger.warning(f"Extractor {extractor.name} failed: {e}")

        return results

    def _apply_dedup_strategy(self, result: ExtractionResult) -> ExtractionResult:
        """Apply deduplication strategy to merged results."""
        if self._dedup_strategy == "highest_confidence":
            # Keep highest confidence version of each entity
            seen: Dict[str, Any] = {}
            for entity in result.entities:
                if entity.id not in seen:
                    seen[entity.id] = entity
                elif entity.confidence > seen[entity.id].confidence:
                    seen[entity.id] = entity

            result.entities = list(seen.values())

        elif self._dedup_strategy == "merge":
            # Merge attributes from duplicates
            from victor.storage.memory.entity_types import Entity

            seen: Dict[str, Entity] = {}
            for entity in result.entities:
                if entity.id in seen:
                    seen[entity.id] = seen[entity.id].merge_with(entity)
                else:
                    seen[entity.id] = entity

            result.entities = list(seen.values())

        # For "first" strategy, merge() already keeps first

        return result


def create_default_extractor() -> CompositeExtractor:
    """Create a composite extractor with default extractors.

    Returns:
        CompositeExtractor with code and text extractors
    """
    from victor.storage.memory.extractors.code_extractor import CodeEntityExtractor
    from victor.storage.memory.extractors.text_extractor import TextEntityExtractor

    return CompositeExtractor(
        extractors=[
            CodeEntityExtractor(),
            TextEntityExtractor(),
        ],
        parallel=True,
        dedup_strategy="merge",
    )


# Add class method alias
CompositeExtractor.create_default = staticmethod(create_default_extractor)
