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

"""Query Decomposition Strategy.

Breaks complex queries into simpler sub-queries for:
- Multi-step reasoning tasks
- Complex analysis requests
- Tasks requiring multiple information sources

Example:
    "Compare AAPL and MSFT revenue growth and identify risks"
    →
    [
        "AAPL Apple Inc revenue growth year over year",
        "MSFT Microsoft Corporation revenue growth year over year",
        "AAPL Apple Inc risk factors",
        "MSFT Microsoft Corporation risk factors"
    ]
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from victor.protocols.query_enhancement import (
    EnhancedQuery,
    EnhancementContext,
    EnhancementTechnique,
)
from victor.core.query_enhancement.strategies.base import BaseQueryEnhancementStrategy

logger = logging.getLogger(__name__)


# Domain-specific decomposition templates
FINANCIAL_DECOMPOSITION_TEMPLATE = """You are a financial analyst query decomposer.

Break this complex query into 2-4 simpler sub-queries that can be searched independently:
- Each sub-query should focus on one aspect (e.g., revenue, risks, cash flow)
- Include company names and tickers in each sub-query
- Use SEC filing terminology
- Keep sub-queries focused and searchable

Query: {query}
{context}

Return a JSON array of sub-queries. Example format:
["sub-query 1", "sub-query 2", "sub-query 3"]

Return ONLY the JSON array, no explanation:"""

CODE_DECOMPOSITION_TEMPLATE = """You are a code search query decomposer.

Break this complex query into 2-4 simpler sub-queries that can be searched independently:
- Each sub-query should focus on one code aspect (e.g., implementation, tests, config)
- Include relevant technical terms
- Keep sub-queries focused on specific code patterns
- Make each sub-query independently searchable

Query: {query}
{context}

Return a JSON array of sub-queries. Example format:
["sub-query 1", "sub-query 2", "sub-query 3"]

Return ONLY the JSON array, no explanation:"""

GENERAL_DECOMPOSITION_TEMPLATE = """You are a search query decomposer.

Break this complex query into 2-4 simpler sub-queries that can be searched independently:
- Each sub-query should focus on one aspect of the original question
- Keep sub-queries clear and specific
- Make each sub-query independently searchable

Query: {query}
{context}

Return a JSON array of sub-queries. Example format:
["sub-query 1", "sub-query 2", "sub-query 3"]

Return ONLY the JSON array, no explanation:"""


class DecompositionStrategy(BaseQueryEnhancementStrategy):
    """Query decomposition strategy using LLM.

    Breaks complex queries into simpler sub-queries that can be
    searched independently and results combined. Useful for:
    - Multi-entity comparisons
    - Multi-aspect analysis
    - Complex research questions
    """

    # Minimum word count to consider decomposition
    MIN_WORDS_FOR_DECOMPOSITION = 8

    @property
    def name(self) -> str:
        return "decomposition"

    @property
    def technique(self) -> EnhancementTechnique:
        return EnhancementTechnique.DECOMPOSITION

    def _register_default_templates(self) -> None:
        """Register domain-specific decomposition templates."""
        self._prompt_templates = {
            "financial": FINANCIAL_DECOMPOSITION_TEMPLATE,
            "code": CODE_DECOMPOSITION_TEMPLATE,
            "general": GENERAL_DECOMPOSITION_TEMPLATE,
            "research": GENERAL_DECOMPOSITION_TEMPLATE,  # Use general for now
        }

    async def _enhance_impl(
        self,
        query: str,
        context: EnhancementContext,
        llm_response: Optional[str],
    ) -> EnhancedQuery:
        """Apply decomposition enhancement.

        Args:
            query: Original query
            context: Enhancement context
            llm_response: LLM-generated sub-queries JSON (or None)

        Returns:
            Enhanced query with sub-queries
        """
        sub_queries: List[str] = []

        if llm_response:
            sub_queries = self._parse_sub_queries(llm_response)

        # If LLM failed or returned nothing, try heuristic decomposition
        if not sub_queries:
            sub_queries = self._heuristic_decompose(query, context)

        if sub_queries:
            return EnhancedQuery(
                original=query,
                enhanced=query,  # Keep original as main query
                technique=self.technique,
                sub_queries=sub_queries,
                confidence=0.8 if llm_response else 0.6,
                metadata={
                    "domain": context.domain,
                    "sub_query_count": len(sub_queries),
                },
            )

        # No decomposition possible
        return EnhancedQuery(
            original=query,
            enhanced=query,
            technique=self.technique,
            confidence=0.5,
            metadata={"domain": context.domain, "no_decomposition": True},
        )

    def _parse_sub_queries(self, response: str) -> List[str]:
        """Parse sub-queries from LLM response.

        Args:
            response: Raw LLM response (expected JSON array)

        Returns:
            List of sub-query strings
        """
        # Try to extract JSON array from response
        response = response.strip()

        # Find JSON array in response
        try:
            # Direct JSON parse
            if response.startswith("["):
                sub_queries = json.loads(response)
                if isinstance(sub_queries, list):
                    return [str(q).strip() for q in sub_queries if q]
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in text
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            try:
                sub_queries = json.loads(match.group())
                if isinstance(sub_queries, list):
                    return [str(q).strip() for q in sub_queries if q]
            except json.JSONDecodeError:
                pass

        # Fallback: split by newlines if contains numbered list
        lines = response.split("\n")
        sub_queries = []
        for line in lines:
            # Match numbered list items
            match = re.match(r"^\d+[\.\)]\s*(.+)$", line.strip())
            if match:
                sub_queries.append(match.group(1).strip())

        return sub_queries

    def _heuristic_decompose(self, query: str, context: EnhancementContext) -> List[str]:
        """Heuristic decomposition without LLM.

        Uses simple rules to break down queries:
        - Split on "and" for multiple aspects
        - Create per-entity queries for comparisons
        - Split multi-faceted analysis requests

        Args:
            query: Original query
            context: Enhancement context

        Returns:
            List of sub-queries
        """
        sub_queries: List[str] = []
        query_lower = query.lower()

        # Check if query is complex enough to decompose
        word_count = len(query.split())
        if word_count < self.MIN_WORDS_FOR_DECOMPOSITION:
            return []

        # Get entities from context
        entities = context.entity_metadata or []

        # For comparison queries with multiple entities
        if len(entities) >= 2 and any(
            kw in query_lower for kw in ["compare", "versus", "vs", "difference"]
        ):
            # Extract the analysis aspect from query
            aspects = self._extract_aspects(query)

            for entity in entities[:3]:  # Max 3 entities
                name = entity.get("name", "")
                ticker = entity.get("ticker", "")
                entity_term = f"{name} {ticker}".strip() if ticker else name

                for aspect in aspects[:2]:  # Max 2 aspects per entity
                    sub_queries.append(f"{entity_term} {aspect}")

        # For multi-aspect analysis (contains "and" with different topics)
        elif " and " in query_lower:
            parts = re.split(r"\s+and\s+", query, flags=re.IGNORECASE)
            if len(parts) >= 2:
                # Get common subject from first part
                base_subject = self._extract_subject(parts[0])
                for part in parts:
                    sub_queries.append(f"{base_subject} {part.strip()}")

        return sub_queries[:4]  # Max 4 sub-queries

    def _extract_aspects(self, query: str) -> List[str]:
        """Extract analysis aspects from query.

        Args:
            query: Original query

        Returns:
            List of aspect terms
        """
        # Common financial aspects
        financial_aspects = [
            "revenue",
            "income",
            "profit",
            "growth",
            "margin",
            "cash flow",
            "risk",
            "balance sheet",
            "earnings",
            "expenses",
        ]

        aspects = []
        query_lower = query.lower()

        for aspect in financial_aspects:
            if aspect in query_lower:
                aspects.append(aspect)

        # Default aspects if none found
        if not aspects:
            aspects = ["performance", "metrics"]

        return aspects

    def _extract_subject(self, text: str) -> str:
        """Extract the subject from beginning of query.

        Args:
            text: Query text

        Returns:
            Subject term
        """
        # Remove common query starters
        starters = [
            "what is",
            "what are",
            "show me",
            "find",
            "get",
            "compare",
            "analyze",
        ]

        text_lower = text.lower().strip()
        for starter in starters:
            if text_lower.startswith(starter):
                text = text[len(starter) :].strip()
                break

        # Take first 3-5 words as subject
        words = text.split()[:5]
        return " ".join(words)
