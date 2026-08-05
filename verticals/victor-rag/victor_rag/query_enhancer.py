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

"""Query Enhancer - LLM-based query enhancement for RAG.

Implements best-of-class query enhancement techniques:
1. Query Rewriting - Reformulates query for better retrieval
2. HyDE (Hypothetical Document Embeddings) - Generates hypothetical answer
3. Step-back Prompting - Generates broader context questions
4. Multi-query Generation - Creates query variants for comprehensive coverage

Example:
    enhancer = QueryEnhancer(provider="ollama", model="llama3.2:3b")

    # Simple query rewriting
    enhanced = await enhancer.rewrite_query("What's AAPL revenue?")
    # Returns: "What is Apple Inc's total revenue and net sales?"

    # HyDE - generate hypothetical document
    hyde_doc = await enhancer.generate_hypothetical_document(
        "Compare Apple and Microsoft profit margins"
    )

    # Multi-query expansion
    queries = await enhancer.generate_query_variants(
        "How did Tesla perform last quarter?"
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor_rag.entity_resolver import EntityInfo

logger = logging.getLogger(__name__)


class EnhancementTechnique(Enum):
    """Available query enhancement techniques."""

    REWRITE = "rewrite"  # Basic query rewriting
    HYDE = "hyde"  # Hypothetical Document Embeddings
    STEP_BACK = "step_back"  # Step-back prompting
    MULTI_QUERY = "multi_query"  # Generate multiple query variants
    ENTITY_EXPAND = "entity_expand"  # Expand with entity metadata


@dataclass
class EnhancedQuery:
    """Result of query enhancement.

    Attributes:
        original: Original user query
        enhanced: Enhanced/rewritten query
        technique: Enhancement technique used
        variants: Additional query variants (for multi-query)
        hypothetical_doc: Hypothetical document (for HyDE)
        metadata: Additional enhancement metadata
    """

    original: str
    enhanced: str
    technique: EnhancementTechnique
    variants: List[str] = None
    hypothetical_doc: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.variants is None:
            self.variants = []
        if self.metadata is None:
            self.metadata = {}

    def get_all_queries(self) -> List[str]:
        """Get all query variants including enhanced query."""
        queries = [self.enhanced]
        queries.extend(self.variants)
        return list(set(queries))


# Prompts for different enhancement techniques
REWRITE_PROMPT = """You are a search query optimizer. Rewrite the user's query to improve search results.

Rules:
1. Expand abbreviations (e.g., "AAPL" → "Apple Inc", "rev" → "revenue")
2. Add relevant synonyms for key terms
3. Keep the query focused and specific
4. Do NOT add information not implied by the original query
5. Return ONLY the rewritten query, no explanation

User query: {query}

Entity context (if available):
{entity_context}

Rewritten query:"""

HYDE_PROMPT = """You are a financial analyst. Based on the user's question, write a hypothetical paragraph that would answer this question if it appeared in a company's SEC filing or financial report.

Rules:
1. Write as if this is an excerpt from an actual SEC 10-K filing
2. Include specific but realistic placeholder numbers
3. Use formal financial reporting language
4. Keep it to 2-3 sentences
5. Focus on the specific metrics/topics asked about

User question: {query}

Entity context:
{entity_context}

Hypothetical document excerpt:"""

STEP_BACK_PROMPT = """You are a research assistant. Given the user's specific question, generate a broader "step-back" question that would help provide context.

Rules:
1. Make the question more general but still relevant
2. The broader question should help understand the context needed for the specific question
3. Return ONLY the step-back question, no explanation

User question: {query}

Step-back question:"""

MULTI_QUERY_PROMPT = """You are a search expert. Generate 3 different versions of the user's query to improve search coverage.

Rules:
1. Each variant should capture a different aspect or phrasing
2. Use different keywords and synonyms
3. Keep all variants relevant to the original intent
4. Return as a JSON array of strings

User query: {query}

Entity context:
{entity_context}

Query variants (JSON array):"""


class QueryEnhancer:
    """LLM-based query enhancement for improved RAG retrieval.

    Supports multiple enhancement techniques that can be combined
    for best results.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        timeout: float = 10.0,
    ):
        """Initialize query enhancer.

        Args:
            provider: LLM provider name (e.g., "ollama", "anthropic")
            model: Model name for enhancement
            temperature: LLM temperature (lower = more focused)
            timeout: Request timeout in seconds
        """
        self._provider_name = provider
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._provider_instance = None

    async def _get_provider(self):
        """Get or create LLM provider instance."""
        if self._provider_instance is not None:
            return self._provider_instance

        from victor.config.settings import load_settings
        from victor_contracts.provider_runtime import ProviderRegistry

        settings = load_settings()

        provider_name = self._provider_name or settings.default_provider or "ollama"
        model = self._model or settings.default_model

        try:
            self._provider_instance = ProviderRegistry.create(provider_name)
            if not self._model:
                self._model = model
            return self._provider_instance
        except Exception as e:
            logger.warning(f"Failed to create provider {provider_name}: {e}")
            return None

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM with prompt and return response.

        Args:
            prompt: Prompt to send

        Returns:
            LLM response text, or None on failure
        """
        provider = await self._get_provider()
        if not provider:
            return None

        if not self._model:
            logger.warning("No model configured for query enhancement")
            return None

        from victor_contracts.provider_runtime import Message

        try:
            messages = [Message(role="user", content=prompt)]
            response = await provider.chat(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
            )

            if hasattr(response, "content"):
                return response.content.strip()
            elif hasattr(response, "message"):
                return response.message.get("content", "").strip()
            return str(response).strip()

        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return None

    def _format_entity_context(self, entities: List["EntityInfo"]) -> str:
        """Format entity information for prompt context.

        Args:
            entities: List of resolved entities

        Returns:
            Formatted context string
        """
        if not entities:
            return "No specific entities identified."

        lines = []
        for entity in entities:
            parts = [entity.name]
            if entity.ticker:
                parts.append(f"(ticker: {entity.ticker})")
            if entity.sector:
                parts.append(f"[{entity.sector}]")
            lines.append(" ".join(parts))

        return "\n".join(lines)

    async def rewrite_query(
        self,
        query: str,
        entities: Optional[List["EntityInfo"]] = None,
    ) -> EnhancedQuery:
        """Rewrite query for better retrieval.

        Args:
            query: Original user query
            entities: Optional resolved entities for context

        Returns:
            EnhancedQuery with rewritten query
        """
        entity_context = self._format_entity_context(entities or [])
        prompt = REWRITE_PROMPT.format(query=query, entity_context=entity_context)

        enhanced = await self._call_llm(prompt)

        if not enhanced:
            # Fallback: return original with entity expansion
            enhanced = self._fallback_rewrite(query, entities)

        return EnhancedQuery(
            original=query,
            enhanced=enhanced,
            technique=EnhancementTechnique.REWRITE,
            metadata={"entities": [e.name for e in (entities or [])]},
        )

    async def generate_hypothetical_document(
        self,
        query: str,
        entities: Optional[List["EntityInfo"]] = None,
    ) -> EnhancedQuery:
        """Generate hypothetical document for HyDE technique.

        HyDE embeds the hypothetical answer and uses it for retrieval,
        which can find relevant documents even with vocabulary mismatch.

        Args:
            query: Original user query
            entities: Optional resolved entities

        Returns:
            EnhancedQuery with hypothetical document
        """
        entity_context = self._format_entity_context(entities or [])
        prompt = HYDE_PROMPT.format(query=query, entity_context=entity_context)

        hypothetical = await self._call_llm(prompt)

        if not hypothetical:
            hypothetical = query  # Fallback to original

        return EnhancedQuery(
            original=query,
            enhanced=query,  # Keep original for search
            technique=EnhancementTechnique.HYDE,
            hypothetical_doc=hypothetical,
            metadata={"entities": [e.name for e in (entities or [])]},
        )

    async def generate_step_back_question(self, query: str) -> EnhancedQuery:
        """Generate step-back question for broader context.

        Step-back prompting generates a more general question that
        helps retrieve relevant context documents.

        Args:
            query: Original specific query

        Returns:
            EnhancedQuery with step-back question as variant
        """
        prompt = STEP_BACK_PROMPT.format(query=query)

        step_back = await self._call_llm(prompt)

        return EnhancedQuery(
            original=query,
            enhanced=query,
            technique=EnhancementTechnique.STEP_BACK,
            variants=[step_back] if step_back else [],
        )

    async def generate_query_variants(
        self,
        query: str,
        entities: Optional[List["EntityInfo"]] = None,
        num_variants: int = 3,
    ) -> EnhancedQuery:
        """Generate multiple query variants for comprehensive retrieval.

        Args:
            query: Original user query
            entities: Optional resolved entities
            num_variants: Number of variants to generate

        Returns:
            EnhancedQuery with query variants
        """
        entity_context = self._format_entity_context(entities or [])
        prompt = MULTI_QUERY_PROMPT.format(query=query, entity_context=entity_context)

        response = await self._call_llm(prompt)

        variants = []
        if response:
            try:
                # Try to parse as JSON array
                # Handle potential markdown code blocks
                clean_response = response.strip()
                if clean_response.startswith("```"):
                    clean_response = clean_response.split("```")[1]
                    if clean_response.startswith("json"):
                        clean_response = clean_response[4:]

                variants = json.loads(clean_response)
                if not isinstance(variants, list):
                    variants = [str(variants)]
            except json.JSONDecodeError:
                # Fallback: split by newlines
                variants = [v.strip() for v in response.split("\n") if v.strip()]

        return EnhancedQuery(
            original=query,
            enhanced=query,
            technique=EnhancementTechnique.MULTI_QUERY,
            variants=variants[:num_variants],
            metadata={"entities": [e.name for e in (entities or [])]},
        )

    async def enhance(
        self,
        query: str,
        entities: Optional[List["EntityInfo"]] = None,
        techniques: Optional[List[EnhancementTechnique]] = None,
    ) -> EnhancedQuery:
        """Apply enhancement techniques to query.

        Args:
            query: Original user query
            entities: Optional resolved entities
            techniques: Techniques to apply (default: REWRITE + ENTITY_EXPAND)

        Returns:
            EnhancedQuery with all enhancements applied
        """
        if techniques is None:
            techniques = [
                EnhancementTechnique.REWRITE,
                EnhancementTechnique.ENTITY_EXPAND,
            ]

        # Start with entity expansion (always fast, no LLM needed)
        enhanced_query = self._fallback_rewrite(query, entities)
        variants = []
        hypothetical = None
        all_metadata = {"entities": [e.name for e in (entities or [])]}

        for technique in techniques:
            if technique == EnhancementTechnique.REWRITE:
                result = await self.rewrite_query(query, entities)
                enhanced_query = result.enhanced
                all_metadata.update(result.metadata or {})

            elif technique == EnhancementTechnique.HYDE:
                result = await self.generate_hypothetical_document(query, entities)
                hypothetical = result.hypothetical_doc
                all_metadata["hyde"] = True

            elif technique == EnhancementTechnique.STEP_BACK:
                result = await self.generate_step_back_question(query)
                variants.extend(result.variants)
                all_metadata["step_back"] = True

            elif technique == EnhancementTechnique.MULTI_QUERY:
                result = await self.generate_query_variants(query, entities)
                variants.extend(result.variants)
                all_metadata["multi_query"] = True

            elif technique == EnhancementTechnique.ENTITY_EXPAND:
                # Already done via _fallback_rewrite
                pass

        return EnhancedQuery(
            original=query,
            enhanced=enhanced_query,
            technique=(techniques[0] if techniques else EnhancementTechnique.ENTITY_EXPAND),
            variants=list(set(variants)),
            hypothetical_doc=hypothetical,
            metadata=all_metadata,
        )

    def _fallback_rewrite(
        self,
        query: str,
        entities: Optional[List["EntityInfo"]] = None,
    ) -> str:
        """Fallback rewrite using entity metadata (no LLM).

        Args:
            query: Original query
            entities: Resolved entities

        Returns:
            Enhanced query with entity terms added
        """
        if not entities:
            return query

        # Collect expansion terms
        expansion_terms = []
        for entity in entities:
            terms = entity.get_search_terms()
            for term in terms:
                # Don't duplicate terms already in query
                if term.lower() not in query.lower():
                    expansion_terms.append(term)

        if expansion_terms:
            return f"{query} {' '.join(expansion_terms[:6])}"

        return query


# Singleton instance
_enhancer_instance: Optional[QueryEnhancer] = None


def get_query_enhancer(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> QueryEnhancer:
    """Get or create query enhancer instance.

    Args:
        provider: Optional provider override
        model: Optional model override

    Returns:
        QueryEnhancer instance
    """
    global _enhancer_instance
    if _enhancer_instance is None:
        _enhancer_instance = QueryEnhancer(provider=provider, model=model)
    return _enhancer_instance


def reset_query_enhancer() -> None:
    """Reset the global query enhancer instance."""
    global _enhancer_instance
    _enhancer_instance = None
