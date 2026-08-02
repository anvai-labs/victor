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

"""Entity Expand Strategy.

Fast query expansion using entity metadata - no LLM required.
Appends entity names, tickers, and aliases to improve search recall.

This is the fallback strategy when LLM is unavailable or disabled.

Example:
    Query: "What is Apple's revenue?"
    Entities: [{"name": "Apple Inc", "ticker": "AAPL"}]
    →
    Enhanced: "What is Apple's revenue? Apple Inc AAPL"
"""

from __future__ import annotations

import logging
from typing import Optional

from victor.protocols.query_enhancement import (
    EnhancedQuery,
    EnhancementContext,
    EnhancementTechnique,
)
from victor.core.query_enhancement.strategies.base import BaseQueryEnhancementStrategy

logger = logging.getLogger(__name__)


class EntityExpandStrategy(BaseQueryEnhancementStrategy):
    """Entity expansion strategy (no LLM required).

    Expands queries by appending entity metadata (names, tickers, aliases)
    to improve search recall. This is fast and works without LLM access.

    Used as:
    - Primary strategy in air-gapped mode
    - Fallback when LLM enhancement is disabled
    - Base expansion before other enhancements
    """

    # Maximum number of expansion terms to add
    MAX_EXPANSION_TERMS = 6

    def __init__(self, **kwargs):
        """Initialize strategy (ignores LLM parameters)."""
        # Don't call super().__init__ to avoid LLM setup
        self._prompt_templates = {}

    @property
    def name(self) -> str:
        return "entity_expand"

    @property
    def technique(self) -> EnhancementTechnique:
        return EnhancementTechnique.ENTITY_EXPAND

    @property
    def requires_llm(self) -> bool:
        """Entity expansion does not require LLM."""
        return False

    def _register_default_templates(self) -> None:
        """No templates needed for entity expansion."""
        pass

    async def enhance(
        self,
        query: str,
        context: EnhancementContext,
    ) -> EnhancedQuery:
        """Enhance query by appending entity terms.

        Args:
            query: Original query
            context: Enhancement context with entity metadata

        Returns:
            Enhanced query with entity terms appended
        """
        return await self._enhance_impl(query, context, None)

    async def _enhance_impl(
        self,
        query: str,
        context: EnhancementContext,
        llm_response: Optional[str],
    ) -> EnhancedQuery:
        """Apply entity expansion.

        Args:
            query: Original query
            context: Enhancement context
            llm_response: Ignored (no LLM)

        Returns:
            Enhanced query with entity terms
        """
        expansion_terms = self._get_expansion_terms(query, context)

        if expansion_terms:
            # Append unique terms not already in query
            enhanced = f"{query} {' '.join(expansion_terms)}"
            return EnhancedQuery(
                original=query,
                enhanced=enhanced,
                technique=self.technique,
                confidence=0.7,
                metadata={
                    "domain": context.domain,
                    "expansion_terms": expansion_terms,
                },
            )

        # No expansion possible
        return EnhancedQuery(
            original=query,
            enhanced=query,
            technique=self.technique,
            confidence=0.5,
            metadata={"domain": context.domain, "no_expansion": True},
        )

    def _get_expansion_terms(self, query: str, context: EnhancementContext) -> list[str]:
        """Extract expansion terms from context.

        Args:
            query: Original query (to avoid duplicates)
            context: Enhancement context

        Returns:
            List of expansion terms
        """
        terms = []
        query_lower = query.lower()

        for entity in context.entity_metadata:
            # Add entity name if not in query
            name = entity.get("name", "")
            if name and name.lower() not in query_lower:
                terms.append(name)

            # Add ticker if not in query
            ticker = entity.get("ticker", "")
            if ticker and ticker.lower() not in query_lower:
                terms.append(ticker)

            # Add aliases if not in query
            aliases = entity.get("aliases", [])
            for alias in aliases[:2]:  # Max 2 aliases per entity
                if alias and alias.lower() not in query_lower:
                    terms.append(alias)

            # Add sector if present
            sector = entity.get("sector", "")
            if sector and sector.lower() not in query_lower:
                # Only add sector if query seems to need context
                if len(context.entity_metadata) > 1 or "compare" in query_lower:
                    terms.append(sector)

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique_terms.append(term)

        return unique_terms[: self.MAX_EXPANSION_TERMS]

    def __repr__(self) -> str:
        return "EntityExpandStrategy(requires_llm=False)"
