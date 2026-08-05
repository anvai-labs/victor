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

"""Query enhancement protocols and data types.

This module defines the Strategy pattern interface for query enhancement,
enabling pluggable transformation strategies across verticals.

Design Patterns:
- Strategy Pattern: Different enhancement strategies for different query types
- Composite Pattern: Pipeline aggregates multiple strategies
- Dependency Inversion: Core code depends on IQueryEnhancementStrategy interface

Usage:
    from victor.protocols.query_enhancement import (
        QueryEnhancementPipeline,
        EnhancementContext,
        EnhancementTechnique,
    )

    pipeline = QueryEnhancementPipeline(config)
    result = await pipeline.enhance(query, context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class EnhancementTechnique(str, Enum):
    """Available query enhancement techniques.

    Each technique represents a different approach to query transformation:
    - REWRITE: Normalize and expand the query
    - HYDE: Generate hypothetical document for embedding
    - STEP_BACK: Generate broader context query
    - MULTI_QUERY: Generate multiple query variants
    - DECOMPOSITION: Break complex query into sub-queries
    - ENTITY_EXPAND: Expand with entity metadata (no LLM)
    """

    REWRITE = "rewrite"
    HYDE = "hyde"
    STEP_BACK = "step_back"
    MULTI_QUERY = "multi_query"
    DECOMPOSITION = "decomposition"
    ENTITY_EXPAND = "entity_expand"


@dataclass
class EnhancementContext:
    """Context for query enhancement.

    Provides domain-specific information to guide enhancement strategies.

    Attributes:
        domain: Domain hint (e.g., "code", "financial", "general")
        task_type: Task type hint (e.g., "search", "edit", "analyze")
        entity_metadata: Domain entities mentioned (e.g., code symbols, companies)
        conversation_history: Recent conversation for context
        max_variants: Maximum number of query variants to generate
        metadata: Additional domain-specific metadata
    """

    domain: str = "general"
    task_type: Optional[str] = None
    entity_metadata: List[Dict[str, Any]] = field(default_factory=list)
    conversation_history: List[str] = field(default_factory=list)
    max_variants: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_entity_names(self) -> List[str]:
        """Get list of entity names from metadata."""
        names = []
        for entity in self.entity_metadata:
            if name := entity.get("name"):
                names.append(name)
            if ticker := entity.get("ticker"):
                names.append(ticker)
        return names


@dataclass
class EnhancedQuery:
    """Result of query enhancement.

    Attributes:
        original: Original user query
        enhanced: Primary enhanced/rewritten query
        technique: Primary enhancement technique used
        variants: Additional query variants (for multi-query strategies)
        sub_queries: Decomposed sub-queries (for decomposition)
        hypothetical_doc: Hypothetical document (for HyDE)
        reasoning: Chain of thought reasoning (if applicable)
        confidence: Enhancement confidence score (0.0-1.0)
        metadata: Additional enhancement metadata
    """

    original: str
    enhanced: str
    technique: EnhancementTechnique
    variants: List[str] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    hypothetical_doc: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_all_queries(self) -> List[str]:
        """Get all query variants including enhanced query and sub-queries."""
        queries = [self.enhanced]
        queries.extend(self.variants)
        queries.extend(self.sub_queries)
        return list(dict.fromkeys(queries))  # Preserve order, remove duplicates

    def __repr__(self) -> str:
        return (
            f"EnhancedQuery(original={self.original!r}, "
            f"enhanced={self.enhanced!r}, "
            f"technique={self.technique.value}, "
            f"variants={len(self.variants)}, "
            f"sub_queries={len(self.sub_queries)})"
        )


@dataclass
class QueryEnhancementConfig:
    """Configuration for query enhancement pipeline.

    Attributes:
        techniques: List of techniques to apply (in order)
        enable_llm: Whether to use LLM-based enhancement
        fallback_to_expansion: Use entity expansion as fallback
        max_enhancement_time_ms: Maximum time for enhancement
        cache_ttl_seconds: Cache TTL for enhanced queries
        provider: Optional LLM provider override
        model: Optional model override
        temperature: Temperature for LLM calls
    """

    techniques: List[EnhancementTechnique] = field(
        default_factory=lambda: [EnhancementTechnique.ENTITY_EXPAND]
    )
    enable_llm: bool = False  # Global default: disabled (opt-out pattern)
    fallback_to_expansion: bool = True
    max_enhancement_time_ms: float = 60000.0  # 60s for consumer hardware
    cache_ttl_seconds: int = 300
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.3

    def __post_init__(self):
        """Ensure techniques is a list of EnhancementTechnique."""
        if self.techniques:
            self.techniques = [
                EnhancementTechnique(t) if isinstance(t, str) else t for t in self.techniques
            ]


@runtime_checkable
class IQueryEnhancementStrategy(Protocol):
    """Strategy interface for query enhancement.

    Each strategy handles a specific type of query transformation.
    Strategies can be composed for multi-step enhancement.

    Implementations should:
    - Be stateless (all state in context)
    - Handle LLM unavailability gracefully
    - Respect timeout constraints
    - Cache results when appropriate
    """

    @property
    def name(self) -> str:
        """Return strategy name."""
        ...

    @property
    def technique(self) -> EnhancementTechnique:
        """Return the enhancement technique this strategy implements."""
        ...

    @property
    def requires_llm(self) -> bool:
        """Return whether this strategy requires LLM access."""
        ...

    async def enhance(
        self,
        query: str,
        context: EnhancementContext,
    ) -> EnhancedQuery:
        """Enhance a query.

        Args:
            query: Original query to enhance
            context: Enhancement context with domain hints

        Returns:
            Enhanced query result
        """
        ...

    def get_prompt_template(self, domain: str) -> str:
        """Get domain-specific prompt template.

        Args:
            domain: Domain identifier (e.g., "code", "financial", "general")

        Returns:
            Prompt template string with {query} and {context} placeholders
        """
        ...


@runtime_checkable
class IQueryEnhancementPipeline(Protocol):
    """Pipeline interface for orchestrating query enhancement.

    The pipeline applies multiple strategies in sequence and handles:
    - Strategy selection based on config
    - Timeout enforcement
    - Caching
    - Fallback to non-LLM strategies
    """

    async def enhance(
        self,
        query: str,
        context: EnhancementContext,
    ) -> EnhancedQuery:
        """Apply enhancement pipeline to query.

        Args:
            query: Original query
            context: Enhancement context

        Returns:
            Enhanced query with all transformations applied
        """
        ...

    def is_llm_available(self) -> bool:
        """Check if LLM is available for enhancement."""
        ...
