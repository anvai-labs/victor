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

"""RAG Query Tool - Query with automatic context retrieval and LLM synthesis."""

import logging
import re
from typing import Any, Dict, List, Optional

from victor_contracts.enrichment_runtime import EnrichmentContext
from victor.tools.base import BaseTool, CostTier, ToolResult
from victor_rag.enrichment import get_rag_enrichment_strategy

logger = logging.getLogger(__name__)


# Patterns that indicate metadata queries (not content search)
METADATA_QUERY_PATTERNS = [
    r"\blist\s+(all\s+)?(tickers?|companies|stocks?|symbols?)\b",
    r"\bwhat\s+(tickers?|companies|stocks?|symbols?)\s+(are|have been)\s+ingested\b",
    r"\bshow\s+(all\s+)?(tickers?|companies|stocks?|symbols?)\b",
    r"\bwhich\s+(companies|tickers?|stocks?)\b",
    r"\bingested\s+(tickers?|companies|stocks?)\b",
    r"\b(how many|count)\s+(documents?|filings?|tickers?|companies)\b",
]


def is_metadata_query(query: str) -> bool:
    """Check if query is asking for metadata rather than content search.

    Args:
        query: The user's query string

    Returns:
        True if this is a metadata query
    """
    query_lower = query.lower()
    for pattern in METADATA_QUERY_PATTERNS:
        if re.search(pattern, query_lower):
            return True
    return False


# Default RAG system prompt for answer synthesis
RAG_SYSTEM_PROMPT = """You are a helpful assistant answering questions based on retrieved documents.

Instructions:
1. Base your answer ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite sources using [Source N] format
4. Be concise but comprehensive
5. If multiple sources agree, synthesize them into a coherent answer
6. If sources conflict, acknowledge the discrepancy
7. Use **markdown formatting** for clarity:
   - Use tables for numerical/comparative data
   - Use bullet points for lists
   - Use bold for key figures and terms
   - Use headers (##) to organize long answers

Do NOT make up information not present in the context."""


class RAGQueryTool(BaseTool):
    """Query the RAG knowledge base with automatic context retrieval and LLM synthesis.

    Retrieves relevant context and uses an LLM to synthesize an answer,
    including source citations.

    Example:
        # Get context only
        result = await tool.execute(question="What is the auth flow?", synthesize=False)

        # Get synthesized answer (default)
        result = await tool.execute(question="What is the auth flow?")

        # Use specific provider/model
        result = await tool.execute(
            question="What is the auth flow?",
            provider="ollama",
            model="llama3.2:3b"
        )
    """

    name = "rag_query"
    description = (
        "Query the RAG knowledge base and synthesize an answer using an LLM. "
        "Returns an answer with source citations grounded in retrieved documents."
    )

    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Question to answer using the knowledge base",
            },
            "k": {
                "type": "integer",
                "description": "Number of context chunks to retrieve (default: 5)",
                "default": 5,
            },
            "synthesize": {
                "type": "boolean",
                "description": "Use LLM to synthesize answer (default: True)",
                "default": True,
            },
            "provider": {
                "type": "string",
                "description": "LLM provider to use (e.g., 'ollama', 'anthropic', 'openai')",
            },
            "model": {
                "type": "string",
                "description": "Model to use for synthesis (provider-specific)",
            },
            "max_context_chars": {
                "type": "integer",
                "description": "Maximum characters of context to use (increase for more sources)",
                "default": 10240,
            },
        },
        "required": ["question"],
    }

    @property
    def cost_tier(self) -> CostTier:
        # MEDIUM when synthesizing (LLM call), LOW for context-only
        return CostTier.MEDIUM

    async def execute(
        self,
        question: str,
        k: int = 5,
        synthesize: bool = True,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_context_chars: int = 10240,
        **kwargs,
    ) -> ToolResult:
        """Execute RAG query with optional LLM synthesis.

        Args:
            question: Question to answer
            k: Number of context chunks
            synthesize: Whether to use LLM to synthesize answer
            provider: LLM provider (e.g., 'ollama', 'anthropic')
            model: Model name for synthesis
            max_context_chars: Maximum context length

        Returns:
            ToolResult with synthesized answer or formatted context
        """

        try:
            store = self._get_document_store()
            await store.initialize()

            # Check if this is a metadata query (e.g., "list all tickers")
            if is_metadata_query(question):
                return await self._handle_metadata_query(question, store)

            # Initialize enrichment strategy with document store for entity resolution
            from victor_rag.enrichment import (
                get_rag_enrichment_strategy,
            )

            # Get or create enrichment strategy with document store
            enrichment_strategy = get_rag_enrichment_strategy(document_store=store)
            await enrichment_strategy.initialize(document_store=store)

            # Analyze query with async entity resolution
            query_analysis = await enrichment_strategy.analyze_query_async(question)

            # Use recommended k if higher than user-specified k
            effective_k = max(k, query_analysis["recommended_k"])

            # Build enhanced search query
            search_query = query_analysis.get("enhanced_query", question)

            # Log enhancement details
            entity_names = query_analysis.get("entity_names", [])
            if entity_names:
                logger.info(
                    f"Enhanced query: '{search_query}' "
                    f"(entities: {entity_names}, k: {effective_k})"
                )

            # For comparison queries with multiple entities, do multi-search
            entities = query_analysis.get("entities", [])
            if query_analysis["is_comparison"] and len(entities) > 1:
                results = await self._multi_entity_search(store, question, entities, effective_k)
            else:
                # Build metadata filter from entities for single-entity queries
                filter_metadata = None
                if entities and len(entities) == 1:
                    entity = entities[0]
                    # Get ticker/symbol from entity
                    ticker = getattr(entity, "ticker", None) or (
                        entity.get("ticker") if isinstance(entity, dict) else None
                    )
                    if ticker:
                        filter_metadata = {"symbol": ticker}
                        logger.info(f"Applying metadata filter: symbol={ticker}")

                # Standard search with optional metadata filter
                results = await store.search(
                    query=search_query,
                    k=effective_k,
                    filter_metadata=filter_metadata,
                    use_hybrid=True,
                )

                # If metadata filter returned no results, fall back to unfiltered search
                if not results and filter_metadata:
                    logger.info("No results with metadata filter, trying unfiltered search")
                    results = await store.search(
                        query=search_query,
                        k=effective_k,
                        use_hybrid=True,
                    )

            if not results:
                return ToolResult(
                    success=True,
                    output=(
                        f"No relevant context found for: '{question}'\n\n"
                        "The knowledge base may not contain information about this topic. "
                        "Consider ingesting relevant documents first."
                    ),
                )

            # Build formatted context
            context_parts = []
            sources = []
            total_chars = 0

            for i, result in enumerate(results, 1):
                chunk = result.chunk
                source = result.doc_source or chunk.metadata.get("source", "unknown")

                # Check if we have space for this chunk
                chunk_text = chunk.content
                if total_chars + len(chunk_text) > max_context_chars:
                    # Truncate to fit
                    remaining = max_context_chars - total_chars
                    if remaining > 100:
                        chunk_text = chunk_text[:remaining] + "..."
                    else:
                        break

                context_parts.append(f"[Source {i}: {source}]\n{chunk_text}")
                sources.append(f"{i}. {source} (relevance: {result.score:.2f})")
                total_chars += len(chunk_text)

            # Build context string
            context_str = "\n\n---\n\n".join(context_parts)

            # If not synthesizing, return context only
            if not synthesize:
                output = (
                    f"QUESTION: {question}\n\n"
                    f"RETRIEVED CONTEXT ({len(context_parts)} sources):\n"
                    f"{'=' * 50}\n\n" + context_str + f"\n\n{'=' * 50}\n"
                    f"SOURCES:\n"
                    + "\n".join(sources)
                    + "\n\nUse this context to answer the question. "
                    "Cite sources by their number (e.g., [1], [2])."
                )
                return ToolResult(success=True, output=output)

            # Get enrichment strategy and enrich the prompt
            enrichment_strategy = get_rag_enrichment_strategy()
            enrichment_context = EnrichmentContext(
                task_type="rag_query",
                metadata={
                    "doc_sources": [s.split(" ")[0] for s in sources],  # Extract source paths
                },
            )

            # Get enrichments
            enrichments = await enrichment_strategy.get_context_enrichments(
                prompt=question,
                context=enrichment_context,
            )

            # Synthesize answer using LLM with enriched prompt
            answer = await self._synthesize_answer(
                question=question,
                context=context_str,
                sources=sources,
                provider=provider,
                model=model,
                enrichments=enrichments,
            )

            # Format final output
            output = (
                f"QUESTION: {question}\n\n"
                f"ANSWER:\n{answer}\n\n"
                f"{'=' * 50}\n"
                f"SOURCES USED:\n" + "\n".join(sources)
            )

            return ToolResult(success=True, output=output)

        except Exception as e:
            logger.exception(f"Query failed: {e}")
            return ToolResult(
                success=False,
                output=f"Query failed: {str(e)}",
            )

    async def _synthesize_answer(
        self,
        question: str,
        context: str,
        sources: List[str],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        enrichments: Optional[List[Any]] = None,
    ) -> str:
        """Synthesize an answer using an LLM provider with enrichment.

        Args:
            question: The user's question
            context: Retrieved context from documents
            sources: List of source citations
            provider: LLM provider name
            model: Model name
            enrichments: Optional list of enrichments to include

        Returns:
            Synthesized answer string
        """
        from victor.config.settings import load_settings
        from victor_contracts.provider_runtime import Message
        from victor_contracts.provider_runtime import ProviderRegistry

        settings = load_settings()

        # Determine provider and model from settings (same as victor chat)
        if not provider:
            provider = settings.default_provider or "ollama"
        if not model:
            model = settings.default_model

        # Get provider instance
        try:
            provider_instance = ProviderRegistry.create(provider)
        except Exception as e:
            logger.warning(f"Failed to get provider {provider}: {e}")
            # Fallback: return context with instruction
            return (
                f"[Could not connect to {provider} for synthesis]\n\n"
                f"Based on the retrieved context:\n{context}\n\n"
                f"Please answer: {question}"
            )

        # Validate model is set
        if not model:
            logger.error(f"No model specified for provider {provider}")
            return (
                f"[No model configured for {provider}]\n\n"
                "Please specify a model with --model or set default_model in your profile.\n\n"
                f"Based on the retrieved context:\n{context[:1000]}..."
            )

        logger.debug(f"Using provider={provider}, model={model} for RAG synthesis")

        # Build enriched prompt for synthesis
        enrichment_strategy = get_rag_enrichment_strategy()
        user_prompt = enrichment_strategy.enrich_synthesis_prompt(
            question=question,
            context=context,
            sources=sources,
            enrichments=enrichments,
        )

        messages = [
            Message(role="system", content=RAG_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]

        # Call provider
        try:
            response = await provider_instance.chat(
                messages=messages,
                model=model,
                temperature=0.3,  # Lower temperature for factual answers
            )

            # Extract answer from response
            if hasattr(response, "content"):
                return response.content
            elif hasattr(response, "message"):
                return response.message.get("content", str(response))
            else:
                return str(response)

        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return (
                f"[Synthesis failed: {str(e)}]\n\n"
                f"Retrieved context for your question:\n{context[:1000]}..."
            )

    def _get_document_store(self):
        """Get document store instance."""
        from victor_rag.document_store import DocumentStore

        if not hasattr(self, "_store"):
            self._store = DocumentStore()
        return self._store

    async def _multi_entity_search(
        self,
        store,
        question: str,
        entities: List[Any],
        k: int,
    ) -> List[Any]:
        """Search for multiple entities and combine results.

        For comparison queries like "Compare Apple and Microsoft revenue",
        this searches for each entity separately to ensure balanced coverage.

        Args:
            store: Document store instance
            question: Original question
            entities: List of EntityInfo objects or entity names
            k: Total number of results to return

        Returns:
            Combined and deduplicated search results
        """
        # Calculate per-entity k (ensure at least 3 results per entity)
        per_entity_k = max(3, k // len(entities))

        all_results = []
        seen_chunk_ids = set()

        # Common financial terms for query enhancement
        financial_terms_map = {
            "revenue": ["net sales", "total revenue", "sales"],
            "profit": ["net income", "earnings", "operating income"],
            "growth": ["increase", "year-over-year", "change"],
            "margin": ["gross margin", "operating margin", "profit margin"],
        }

        for entity in entities:
            # Handle both EntityInfo objects and plain strings
            ticker = None
            if hasattr(entity, "get_search_terms"):
                # EntityInfo object - use its search terms
                entity_terms = entity.get_search_terms()
                entity_name = entity.name
                ticker = getattr(entity, "ticker", None)
            else:
                # Plain string (backward compatibility)
                entity_terms = [str(entity)]
                entity_name = str(entity)

            # Extract financial terms from the question
            question_lower = question.lower()
            query_financial_terms = []
            for term, expansions in financial_terms_map.items():
                if term in question_lower:
                    query_financial_terms.append(term)
                    query_financial_terms.extend(expansions[:2])

            # Build search query: entity terms + financial terms
            search_terms = entity_terms + query_financial_terms
            entity_query = " ".join(search_terms[:10])  # Limit to 10 terms

            # Build metadata filter for this entity
            filter_metadata = {"symbol": ticker} if ticker else None

            logger.info(
                f"Multi-entity search for '{entity_name}': "
                f"query='{entity_query}', filter={filter_metadata}"
            )

            try:
                entity_results = await store.search(
                    query=entity_query,
                    k=per_entity_k,
                    filter_metadata=filter_metadata,
                    use_hybrid=True,
                )

                # If no results with filter, try without
                if not entity_results and filter_metadata:
                    entity_results = await store.search(
                        query=entity_query,
                        k=per_entity_k,
                        use_hybrid=True,
                    )

                # Deduplicate by chunk ID
                for result in entity_results:
                    chunk_id = getattr(result.chunk, "id", None) or id(result.chunk)
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        all_results.append(result)

            except Exception as e:
                logger.warning(f"Search for entity '{entity_name}' failed: {e}")

        # Sort combined results by relevance score
        all_results.sort(key=lambda r: r.score, reverse=True)

        # Return top k results
        return all_results[:k]

    async def _handle_metadata_query(self, question: str, store) -> ToolResult:
        """Handle metadata queries like 'list all tickers'.

        Args:
            question: The metadata query
            store: Document store instance

        Returns:
            ToolResult with metadata information
        """
        docs = await store.list_documents()

        if not docs:
            return ToolResult(
                success=True,
                output=(
                    "No documents ingested yet.\n\n"
                    "To ingest SEC filings, use:\n"
                    "  victor rag demo-sec --preset faang"
                ),
            )

        # Group SEC filings
        sec_docs = [d for d in docs if d.id.startswith("sec_")]
        other_docs = [d for d in docs if not d.id.startswith("sec_")]

        output_parts = [f"METADATA QUERY: {question}\n", "=" * 50]

        if sec_docs:
            # Group by ticker
            by_ticker: Dict[str, list] = {}
            for doc in sec_docs:
                symbol = doc.metadata.get("symbol", "UNKNOWN")
                if symbol not in by_ticker:
                    by_ticker[symbol] = []
                by_ticker[symbol].append(doc)

            output_parts.append(f"\nSEC FILINGS ({len(sec_docs)} documents):")
            output_parts.append(f"\nIngested Tickers ({len(by_ticker)} companies):")

            # List all tickers
            tickers = sorted(by_ticker.keys())
            output_parts.append(f"  {', '.join(tickers)}")

            # Group by sector
            by_sector: Dict[str, int] = {}
            for doc in sec_docs:
                sector = doc.metadata.get("sector", "Other")
                by_sector[sector] = by_sector.get(sector, 0) + 1

            output_parts.append("\nBy Sector:")
            for sector, count in sorted(by_sector.items(), key=lambda x: -x[1]):
                sector_tickers = [
                    doc.metadata.get("symbol", "?")
                    for doc in sec_docs
                    if doc.metadata.get("sector") == sector
                ]
                unique_tickers = sorted(set(sector_tickers))
                output_parts.append(f"  {sector} ({count}): {', '.join(unique_tickers)}")

            # Summary per ticker
            output_parts.append("\nFilings per company:")
            for symbol in tickers:
                ticker_docs = by_ticker[symbol]
                company = ticker_docs[0].metadata.get("company", symbol)
                filings = [
                    f"{d.metadata.get('filing_type', 'N/A')} ({d.metadata.get('filing_date', 'N/A')})"
                    for d in ticker_docs
                ]
                output_parts.append(f"  {symbol:8} {company}: {', '.join(filings)}")

        if other_docs:
            output_parts.append(f"\nOther Documents: {len(other_docs)}")
            for doc in other_docs[:5]:  # Show max 5
                output_parts.append(f"  - {doc.id}: {doc.source}")
            if len(other_docs) > 5:
                output_parts.append(f"  ... and {len(other_docs) - 5} more")

        output_parts.append(f"\n{'=' * 50}")
        output_parts.append("\nFor content search, try:")
        output_parts.append('  victor rag query "What is Apple\'s revenue?"')

        return ToolResult(
            success=True,
            output="\n".join(output_parts),
        )
