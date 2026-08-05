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

"""Query Rewrite Strategy.

Reformulates queries for better search/retrieval using LLM by:
- Expanding abbreviations and acronyms
- Adding relevant synonyms
- Normalizing terminology
- Maintaining focus on original intent

Domain-specific templates handle different vocabulary:
- Financial: Expand tickers (AAPL → Apple Inc), abbreviations (rev → revenue)
- Code: Expand code terms (fn → function, impl → implementation)
- Research: Academic terminology normalization
- General: Basic query optimization
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


# Domain-specific prompt templates
FINANCIAL_REWRITE_TEMPLATE = """You are a financial query optimizer for SEC filing search.

Rewrite the query to improve search results:
- Expand ticker symbols to full company names (e.g., AAPL → Apple Inc)
- Expand financial abbreviations (e.g., rev → revenue, eps → earnings per share, yoy → year-over-year)
- Add relevant financial terms that would appear in SEC filings
- Keep the query focused and specific
- Preserve the original intent

Query: {query}
{context}

Return ONLY the rewritten query, no explanation:"""

CODE_REWRITE_TEMPLATE = """You are a code search query optimizer.

Rewrite the query to improve code search results:
- Expand programming abbreviations (e.g., fn → function, impl → implementation, cfg → config)
- Add language-specific terms if context suggests a language
- Include common naming patterns (camelCase, snake_case variations)
- Keep focus on code patterns and structures
- Preserve the original intent

Query: {query}
{context}

Return ONLY the rewritten query, no explanation:"""

RESEARCH_REWRITE_TEMPLATE = """You are a research query optimizer for academic search.

Rewrite the query to improve research paper search:
- Expand abbreviations to full academic terms
- Add relevant synonyms and related concepts
- Use formal academic terminology
- Keep focus on research topics
- Preserve the original intent

Query: {query}
{context}

Return ONLY the rewritten query, no explanation:"""

GENERAL_REWRITE_TEMPLATE = """You are a search query optimizer.

Rewrite the query to improve search results:
- Expand abbreviations and acronyms
- Add relevant synonyms for key terms
- Use clear, specific terminology
- Keep the query focused and concise
- Preserve the original intent

Query: {query}
{context}

Return ONLY the rewritten query, no explanation:"""


class RewriteStrategy(BaseQueryEnhancementStrategy):
    """Query rewrite strategy using LLM.

    Reformulates queries for better search by expanding abbreviations,
    adding synonyms, and normalizing terminology. Uses domain-specific
    templates for financial, code, research, and general domains.
    """

    @property
    def name(self) -> str:
        return "rewrite"

    @property
    def technique(self) -> EnhancementTechnique:
        return EnhancementTechnique.REWRITE

    def _register_default_templates(self) -> None:
        """Register domain-specific rewrite templates."""
        self._prompt_templates = {
            "financial": FINANCIAL_REWRITE_TEMPLATE,
            "code": CODE_REWRITE_TEMPLATE,
            "research": RESEARCH_REWRITE_TEMPLATE,
            "general": GENERAL_REWRITE_TEMPLATE,
        }

    async def _enhance_impl(
        self,
        query: str,
        context: EnhancementContext,
        llm_response: Optional[str],
    ) -> EnhancedQuery:
        """Apply rewrite enhancement.

        Args:
            query: Original query
            context: Enhancement context
            llm_response: LLM-generated rewrite (or None)

        Returns:
            Enhanced query with rewritten text
        """
        if llm_response:
            # Clean up LLM response
            enhanced = self._clean_response(llm_response)

            # Validate the response is reasonable
            if self._is_valid_rewrite(query, enhanced):
                return EnhancedQuery(
                    original=query,
                    enhanced=enhanced,
                    technique=self.technique,
                    confidence=0.9,
                    metadata={"domain": context.domain},
                )

        # Fallback: return original query
        return EnhancedQuery(
            original=query,
            enhanced=query,
            technique=self.technique,
            confidence=0.5,
            metadata={"domain": context.domain, "fallback": True},
        )

    def _clean_response(self, response: str) -> str:
        """Clean LLM response to extract just the rewritten query.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned query text
        """
        # Remove common prefixes LLMs might add
        prefixes_to_remove = [
            "Rewritten query:",
            "Rewritten:",
            "Enhanced query:",
            "Here is the rewritten query:",
            "Query:",
        ]

        cleaned = response.strip()
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].strip()

        # Remove quotes if present
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]

        return cleaned.strip()

    def _is_valid_rewrite(self, original: str, rewritten: str) -> bool:
        """Validate the rewrite is reasonable.

        Args:
            original: Original query
            rewritten: Rewritten query

        Returns:
            True if rewrite is valid
        """
        if not rewritten:
            return False

        # Rewrite shouldn't be too short
        if len(rewritten) < len(original) / 3:
            return False

        # Rewrite shouldn't be excessively long
        if len(rewritten) > len(original) * 5:
            return False

        # Rewrite shouldn't be identical (no enhancement)
        if rewritten.lower().strip() == original.lower().strip():
            return False

        return True
