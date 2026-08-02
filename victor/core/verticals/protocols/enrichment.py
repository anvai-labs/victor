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

"""Enrichment Protocols (ISP: Interface Segregation Principle).

This module contains protocols specifically for prompt enrichment strategies.
Following ISP, these protocols are focused on a single responsibility:
enriching prompts with vertical-specific context.

Usage:
    from victor.core.verticals.protocols.enrichment import (
        EnrichmentStrategyProtocol,
        VerticalEnrichmentProviderProtocol,
    )

    class CodingEnrichmentStrategy(EnrichmentStrategyProtocol):
        async def get_enrichments(self, prompt, context):
            symbols = await self.graph.search(prompt)
            return [ContextEnrichment(type="symbols", content=symbols)]
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, runtime_checkable

# =============================================================================
# Enrichment Strategy Protocol
# =============================================================================


@runtime_checkable
class EnrichmentStrategyProtocol(Protocol):
    """Protocol for vertical-specific prompt enrichment strategies.

    Enables auto prompt optimization where prompts are enriched
    with relevant context from vertical-specific sources:
    - Coding: Knowledge graph symbols, related code snippets
    - Research: Web search results, source citations
    - DevOps: Infrastructure context, command patterns
    - Data Analysis: Schema context, query patterns

    Example:
        class CodingEnrichmentStrategy:
            async def get_enrichments(
                self,
                prompt: str,
                context: "EnrichmentContext",
            ) -> List["ContextEnrichment"]:
                # Query knowledge graph for relevant symbols
                symbols = await self.graph.search(prompt)
                return [
                    ContextEnrichment(
                        type=EnrichmentType.KNOWLEDGE_GRAPH,
                        content=format_symbols(symbols),
                        priority=EnrichmentPriority.HIGH,
                    )
                ]

            def get_priority(self) -> int:
                return 50

            def get_token_allocation(self) -> float:
                return 0.4  # Use up to 40% of token budget
    """

    async def get_enrichments(
        self,
        prompt: str,
        context: Any,  # EnrichmentContext from victor.framework.enrichment
    ) -> List[Any]:  # List[ContextEnrichment]
        """Get enrichments for a prompt.

        Args:
            prompt: The prompt to enrich
            context: EnrichmentContext with task metadata

        Returns:
            List of ContextEnrichment objects to apply
        """
        ...

    def get_priority(self) -> int:
        """Get priority for this strategy.

        Lower values are processed first.

        Returns:
            Priority value (default 50)
        """
        ...

    def get_token_allocation(self) -> float:
        """Get fraction of token budget this strategy can use.

        Returns:
            Float between 0.0 and 1.0 (e.g., 0.4 for 40%)
        """
        ...


# =============================================================================
# Vertical Enrichment Provider Protocol
# =============================================================================


@runtime_checkable
class VerticalEnrichmentProviderProtocol(Protocol):
    """Protocol for verticals providing enrichment strategies.

    This protocol enables type-safe isinstance() checks when integrating
    vertical prompt enrichment with the framework.

    Example:
        class CodingVertical(VerticalBase, VerticalEnrichmentProviderProtocol):
            @classmethod
            def get_enrichment_strategy(cls) -> Optional[EnrichmentStrategyProtocol]:
                return CodingEnrichmentStrategy()
    """

    @classmethod
    def get_enrichment_strategy(cls) -> Optional[EnrichmentStrategyProtocol]:
        """Get the enrichment strategy for this vertical.

        Returns:
            EnrichmentStrategyProtocol implementation or None
        """
        ...


__all__ = [
    "EnrichmentStrategyProtocol",
    "VerticalEnrichmentProviderProtocol",
]
