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

"""RAG Vertical Enrichment Strategy.

Provides prompt enrichment capabilities specific to RAG (Retrieval-Augmented Generation):
- Dynamic entity resolution from document store metadata
- LLM-based query enhancement (rewriting, HyDE, multi-query)
- Query expansion with synonyms and related terms
- Document context formatting for LLM synthesis
- Conversation history integration

This module implements best-of-class RAG enrichment techniques:
1. EntityResolver - Dynamically extracts entities from document metadata
2. QueryEnhancer - LLM-based query rewriting and expansion
3. Contextual enrichment - Adds relevant context for synthesis

Example:
    from victor_rag.enrichment import RAGEnrichmentStrategy

    strategy = RAGEnrichmentStrategy()
    await strategy.initialize(document_store)

    # Analyze and enhance query
    analysis = await strategy.analyze_query_async("Compare Apple and Microsoft revenue")
    # Returns: {
    #     "entities": [EntityInfo(name="Apple Inc", ticker="AAPL", ...),
    #                  EntityInfo(name="Microsoft Corporation", ticker="MSFT", ...)],
    #     "is_comparison": True,
    #     "recommended_k": 10,
    #     "enhanced_query": "Compare Apple Inc AAPL and Microsoft Corporation MSFT revenue net sales"
    # }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor_contracts.enrichment_runtime import (
    ContextEnrichment,
    EnrichmentContext,
    EnrichmentPriority,
    EnrichmentType,
)

if TYPE_CHECKING:
    from victor_rag.document_store import DocumentStore
    from victor_rag.entity_resolver import EntityInfo, EntityResolver
    from victor_rag.query_enhancer import QueryEnhancer

logger = logging.getLogger(__name__)


@dataclass
class RAGEnrichmentConfig:
    """Configuration for RAG enrichment.

    Attributes:
        use_entity_resolution: Whether to resolve entities from document metadata
        use_llm_enhancement: Whether to use LLM for query enhancement
            NOTE: Enabled by default for RAG vertical (like entity extraction)
        enhancement_techniques: List of LLM enhancement techniques to use
        include_metadata: Whether to include document metadata in enrichment
        context_window_chars: Maximum characters of context to include
        prioritize_recent: Whether to prioritize recent documents
        llm_provider: Provider for LLM enhancement (None = use default)
        llm_model: Model for LLM enhancement (None = use default)
    """

    use_entity_resolution: bool = True
    use_llm_enhancement: bool = True  # Enabled by default for RAG vertical
    enhancement_techniques: List[str] = None  # ["rewrite", "decomposition", "entity_expand"]
    include_metadata: bool = True
    context_window_chars: int = 4000
    prioritize_recent: bool = True
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

    def __post_init__(self):
        if self.enhancement_techniques is None:
            # Default: rewrite + entity_expand for RAG
            self.enhancement_techniques = ["rewrite", "entity_expand"]


class RAGEnrichmentStrategy:
    """Enrichment strategy for RAG vertical.

    Enhances prompts with:
    - Dynamic entity resolution from document store
    - LLM-based query enhancement (optional)
    - Structured context formatting
    - Source metadata enrichment
    - Conversation history snippets
    """

    def __init__(
        self,
        config: Optional[RAGEnrichmentConfig] = None,
        document_store: Optional["DocumentStore"] = None,
    ):
        """Initialize RAG enrichment strategy.

        Args:
            config: Optional configuration overrides
            document_store: Optional document store for entity resolution
        """
        self.config = config or RAGEnrichmentConfig()
        self._document_store = document_store
        self._entity_resolver: Optional["EntityResolver"] = None
        self._query_enhancer: Optional["QueryEnhancer"] = None
        self._initialized = False

    async def initialize(
        self,
        document_store: Optional["DocumentStore"] = None,
    ) -> None:
        """Initialize the enrichment strategy.

        Args:
            document_store: Document store for entity resolution
        """
        if self._initialized:
            return

        if document_store:
            self._document_store = document_store

        # Initialize entity resolver
        if self.config.use_entity_resolution and self._document_store:
            from victor_rag.entity_resolver import EntityResolver

            self._entity_resolver = EntityResolver(self._document_store)
            await self._entity_resolver.initialize()
            logger.info(
                f"Entity resolver initialized with {self._entity_resolver.get_entity_count()} entities"
            )

        # Initialize query enhancer if LLM enhancement is enabled
        if self.config.use_llm_enhancement:
            from victor_rag.query_enhancer import QueryEnhancer

            self._query_enhancer = QueryEnhancer(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
            )
            logger.info("Query enhancer initialized")

        self._initialized = True

    def get_enrichment_priority(self) -> int:
        """Get the priority level for this strategy.

        Returns:
            Priority value (higher = processed first)
        """
        return 80  # High priority for RAG context

    async def get_context_enrichments(
        self,
        prompt: str,
        context: EnrichmentContext,
    ) -> List[ContextEnrichment]:
        """Get context enrichments for a RAG prompt.

        Args:
            prompt: The user's query
            context: Enrichment context with task metadata

        Returns:
            List of enrichments to apply
        """
        enrichments: List[ContextEnrichment] = []

        # Ensure initialized
        await self.initialize()

        # 1. Entity resolution enrichment
        if self._entity_resolver:
            entities = await self._entity_resolver.resolve_entities(prompt)
            if entities:
                entity_hints = self._format_entity_hints(entities)
                enrichments.append(
                    ContextEnrichment(
                        type=EnrichmentType.PROJECT_CONTEXT,
                        content=entity_hints,
                        priority=EnrichmentPriority.HIGH,
                        source="entity_resolution",
                        metadata={
                            "entity_count": len(entities),
                            "entities": [e.name for e in entities],
                        },
                    )
                )

        # 2. Document metadata enrichment
        if self.config.include_metadata:
            doc_sources = context.metadata.get("doc_sources", [])
            if doc_sources:
                metadata_content = self._format_source_metadata(doc_sources)
                enrichments.append(
                    ContextEnrichment(
                        type=EnrichmentType.PROJECT_CONTEXT,
                        content=metadata_content,
                        priority=EnrichmentPriority.HIGH,
                        source="doc_metadata",
                        metadata={"source_count": len(doc_sources)},
                    )
                )

        # 3. Conversation history enrichment
        if context.tool_history:
            history_content = self._format_conversation_history(context.tool_history)
            if history_content:
                enrichments.append(
                    ContextEnrichment(
                        type=EnrichmentType.CONVERSATION,
                        content=history_content,
                        priority=EnrichmentPriority.NORMAL,
                        source="conversation_history",
                    )
                )

        return enrichments

    def _format_entity_hints(self, entities: List["EntityInfo"]) -> str:
        """Format entity information as enrichment hints.

        Args:
            entities: List of resolved entities

        Returns:
            Formatted hints string
        """
        lines = ["[Identified Entities:]"]
        for entity in entities:
            parts = [f"  - {entity.name}"]
            if entity.ticker:
                parts.append(f"(ticker: {entity.ticker})")
            if entity.sector:
                parts.append(f"[{entity.sector}]")
            lines.append(" ".join(parts))

        return "\n".join(lines)

    async def analyze_query_async(
        self,
        query: str,
        use_llm: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Analyze query with full async support.

        This is the main entry point for query analysis. It:
        1. Resolves entities from document metadata
        2. Optionally enhances query via LLM
        3. Returns comprehensive analysis for search optimization

        Args:
            query: The search query
            use_llm: Override config for LLM enhancement

        Returns:
            Dict with analysis results
        """
        await self.initialize()

        # Resolve entities
        entities: List["EntityInfo"] = []
        if self._entity_resolver:
            entities = await self._entity_resolver.resolve_entities(query)

        # Get base analysis from entity resolver
        if self._entity_resolver:
            analysis = self._entity_resolver.analyze_query(query, entities)
        else:
            analysis = self._fallback_analyze(query)

        # LLM enhancement if enabled
        should_use_llm = use_llm if use_llm is not None else self.config.use_llm_enhancement
        if should_use_llm and self._query_enhancer:
            from victor_rag.query_enhancer import EnhancementTechnique

            techniques = [EnhancementTechnique(t) for t in self.config.enhancement_techniques]

            enhanced = await self._query_enhancer.enhance(
                query=query,
                entities=entities,
                techniques=techniques,
            )

            analysis["enhanced_query"] = enhanced.enhanced
            analysis["query_variants"] = enhanced.variants
            analysis["hypothetical_doc"] = enhanced.hypothetical_doc
            analysis["enhancement_technique"] = enhanced.technique.value
        else:
            # Build enhanced query from entity terms (no LLM)
            expansion_terms = analysis.get("expansion_terms", [])
            if expansion_terms:
                # Filter terms not already in query
                new_terms = [t for t in expansion_terms if t.lower() not in query.lower()]
                analysis["enhanced_query"] = f"{query} {' '.join(new_terms[:6])}"
            else:
                analysis["enhanced_query"] = query

        return analysis

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Synchronous query analysis (for backward compatibility).

        Uses fallback logic without async entity resolution.
        For full functionality, use analyze_query_async().

        Args:
            query: The search query

        Returns:
            Dict with 'entities', 'is_comparison', 'recommended_k', 'expanded_query'
        """
        return self._fallback_analyze(query)

    def _fallback_analyze(self, query: str) -> Dict[str, Any]:
        """Fallback query analysis without entity resolver.

        Args:
            query: The search query

        Returns:
            Basic analysis dict
        """
        query_lower = query.lower()

        # Detect comparison keywords
        comparison_keywords = [
            "compare",
            "versus",
            "vs",
            "difference",
            "between",
            "against",
        ]
        is_comparison = any(kw in query_lower for kw in comparison_keywords)

        # Basic financial term detection
        financial_terms = []
        for term in [
            "revenue",
            "profit",
            "growth",
            "margin",
            "sales",
            "income",
            "earnings",
        ]:
            if term in query_lower:
                financial_terms.append(term)

        return {
            "entities": [],
            "entity_names": [],
            "is_comparison": is_comparison,
            "financial_terms": financial_terms,
            "recommended_k": 10 if is_comparison else 5,
            "expanded_query": "",
            "enhanced_query": query,
            "entity_count": 0,
            "expansion_terms": [],
        }

    def _format_source_metadata(self, sources: List[str]) -> str:
        """Format document source metadata for enrichment.

        Args:
            sources: List of document source paths/URLs

        Returns:
            Formatted metadata string
        """
        if not sources:
            return ""

        lines = ["[Source documents:]"]
        for i, source in enumerate(sources[:5], 1):  # Max 5 sources
            # Extract filename from path or URL
            name = source.split("/")[-1] if "/" in source else source
            lines.append(f"  {i}. {name}")

        return "\n".join(lines)

    def _format_conversation_history(self, tool_history: List[Dict[str, Any]]) -> str:
        """Format relevant conversation history.

        Args:
            tool_history: List of recent tool calls

        Returns:
            Formatted history string
        """
        if not tool_history:
            return ""

        # Filter for RAG-related tool calls
        rag_history = [t for t in tool_history if t.get("tool_name", "").startswith("rag_")]

        if not rag_history:
            return ""

        lines = ["[Previous queries:]"]
        for item in rag_history[-3:]:  # Last 3 RAG queries
            query = item.get("args", {}).get("query", "")
            if query:
                lines.append(f"  - {query[:100]}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def enrich_synthesis_prompt(
        self,
        question: str,
        context: str,
        sources: List[str],
        enrichments: Optional[List[ContextEnrichment]] = None,
    ) -> str:
        """Enrich the synthesis prompt with additional context.

        This is the main entry point for enriching RAG synthesis prompts.

        Args:
            question: The user's question
            context: Retrieved document context
            sources: List of source citations
            enrichments: Optional additional enrichments

        Returns:
            Enriched prompt string for LLM synthesis
        """
        parts = []

        # Add enrichment hints if available
        if enrichments:
            hints = [e.content for e in enrichments if e.content]
            if hints:
                parts.append("ENRICHMENT HINTS:\n" + "\n".join(hints))

        # Add structured context
        parts.append(f"CONTEXT FROM {len(sources)} SOURCES:")
        parts.append(context)

        # Add question
        parts.append(f"QUESTION: {question}")

        # Add synthesis instruction
        parts.append(
            "Provide a clear, comprehensive answer based ONLY on the provided context. "
            "Cite sources using [Source N] format. If the context doesn't contain "
            "sufficient information, acknowledge what's missing."
        )

        return "\n\n".join(parts)

    def refresh(self) -> None:
        """Refresh the enrichment strategy (reload entity index, etc.)."""
        self._initialized = False
        if self._entity_resolver:
            self._entity_resolver.refresh()


# Singleton instance for easy access
_strategy_instance: Optional[RAGEnrichmentStrategy] = None


def get_rag_enrichment_strategy(
    config: Optional[RAGEnrichmentConfig] = None,
    document_store: Optional["DocumentStore"] = None,
) -> RAGEnrichmentStrategy:
    """Get the RAG enrichment strategy singleton.

    Args:
        config: Optional configuration (only used on first call)
        document_store: Optional document store for entity resolution

    Returns:
        RAGEnrichmentStrategy instance
    """
    global _strategy_instance
    if _strategy_instance is None:
        _strategy_instance = RAGEnrichmentStrategy(config, document_store)
    elif document_store and not _strategy_instance._document_store:
        _strategy_instance._document_store = document_store
    return _strategy_instance


def reset_rag_enrichment_strategy() -> None:
    """Reset the global enrichment strategy (for testing)."""
    global _strategy_instance
    _strategy_instance = None
