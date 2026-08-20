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

"""Generic query expansion for semantic search improvement.

This module provides domain-agnostic query expansion capabilities.
Verticals can provide domain-specific expansion dictionaries.

The expansion algorithm adds synonyms and related terms to user queries,
improving semantic search recall by capturing different ways users might
express the same concept.

Usage:
    from victor.framework.search.query_expansion import (
        QueryExpander,
        QueryExpansionConfig,
    )

    # Create expander with custom expansions
    config = QueryExpansionConfig(
        expansions={
            "error": ["exception", "failure", "issue"],
            "config": ["configuration", "settings", "options"],
        },
        max_expansions=5,
    )
    expander = QueryExpander(config)

    # Expand a query
    variations = expander.expand("fix error handling")
    # ['fix error handling', 'exception', 'failure', 'issue']
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Set, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class QueryExpansionConfig:
    """Configuration for query expansion.

    Attributes:
        expansions: Mapping of trigger terms to expansion synonyms.
            Keys are matched case-insensitively against queries.
            Values are lists of synonyms/related terms to add.
        max_expansions: Maximum total query variations to return
            (including the original query).
        deduplicate: Whether to remove duplicate terms (default: True).
    """

    expansions: Dict[str, List[str]] = field(default_factory=dict)
    max_expansions: int = 5
    deduplicate: bool = True


@dataclass(frozen=True)
class ExpandedQuery:
    """Result of query expansion.

    Attributes:
        original: The original user query.
        variations: List of expanded query variations (includes original first).
        matched_patterns: Patterns that triggered expansion.
        expansion_count: Number of expansions added.
    """

    original: str
    variations: List[str]
    matched_patterns: List[str]

    @property
    def expansion_count(self) -> int:
        """Number of expansions added (excluding original)."""
        return max(0, len(self.variations) - 1)

    @property
    def was_expanded(self) -> bool:
        """Whether any expansions were added."""
        return self.expansion_count > 0


@runtime_checkable
class QueryExpanderProtocol(Protocol):
    """Protocol for query expanders (for DIP compliance)."""

    def expand(self, query: str, max_expansions: Optional[int] = None) -> ExpandedQuery:
        """Expand a query with synonyms/related terms."""
        ...

    def is_expandable(self, query: str) -> bool:
        """Check if query has available expansions."""
        ...


class QueryExpander:
    """Expands user queries with synonyms and related terms.

    This is a generic implementation that can be used with any
    domain-specific expansion dictionary.

    Example:
        # Coding vertical might use:
        coding_expansions = {
            "tool registration": ["register tool", "@tool decorator"],
            "provider": ["LLM provider", "BaseProvider"],
        }

        # RAG vertical might use:
        rag_expansions = {
            "document": ["doc", "file", "content"],
            "search": ["query", "find", "retrieve"],
        }
    """

    def __init__(self, config: Optional[QueryExpansionConfig] = None):
        """Initialize query expander.

        Args:
            config: Expansion configuration. If None, creates empty config.
        """
        self._config = config or QueryExpansionConfig()

    @property
    def expansions(self) -> Dict[str, List[str]]:
        """Get the expansion dictionary."""
        return self._config.expansions

    @property
    def max_expansions(self) -> int:
        """Get the maximum expansions limit."""
        return self._config.max_expansions

    def expand(
        self,
        query: str,
        max_expansions: Optional[int] = None,
    ) -> ExpandedQuery:
        """Expand query with synonyms and related terms.

        Args:
            query: Original user query.
            max_expansions: Override max expansions (uses config default if None).

        Returns:
            ExpandedQuery with variations and metadata.

        Example:
            >>> expander = QueryExpander(QueryExpansionConfig(
            ...     expansions={"error": ["exception", "failure"]}
            ... ))
            >>> result = expander.expand("fix error handling")
            >>> result.variations
            ['fix error handling', 'exception', 'failure']
        """
        limit = max_expansions or self._config.max_expansions
        query_lower = query.lower().strip()

        # Always include original query first
        variations: List[str] = [query]
        seen: Set[str] = {query_lower}
        matched_patterns: List[str] = []

        # Find matching patterns and add expansions
        for pattern, synonyms in self._config.expansions.items():
            if pattern in query_lower:
                matched_patterns.append(pattern)

                for synonym in synonyms:
                    synonym_lower = synonym.lower()
                    if self._config.deduplicate and synonym_lower in seen:
                        continue

                    variations.append(synonym)
                    seen.add(synonym_lower)

                    if len(variations) >= limit:
                        break

            if len(variations) >= limit:
                break

        if matched_patterns:
            logger.debug(
                "Expanded query '%s' via patterns %s: %d variations",
                query,
                matched_patterns,
                len(variations),
            )

        return ExpandedQuery(
            original=query,
            variations=variations,
            matched_patterns=matched_patterns,
        )

    def is_expandable(self, query: str) -> bool:
        """Check if query has available expansions.

        Args:
            query: User query to check.

        Returns:
            True if query matches any expansion patterns.
        """
        query_lower = query.lower().strip()
        return any(pattern in query_lower for pattern in self._config.expansions)

    def get_expansion_terms(self, query: str) -> Set[str]:
        """Get all expansion terms for a query.

        Args:
            query: User query.

        Returns:
            Set of all matching expansion terms (excluding original).
        """
        query_lower = query.lower().strip()
        terms: Set[str] = set()

        for pattern, synonyms in self._config.expansions.items():
            if pattern in query_lower:
                terms.update(synonyms)

        return terms

    def add_expansions(self, expansions: Dict[str, List[str]]) -> None:
        """Add additional expansions to the dictionary.

        Args:
            expansions: New expansions to merge in.
        """
        for pattern, synonyms in expansions.items():
            if pattern in self._config.expansions:
                # Merge, avoiding duplicates
                existing = set(self._config.expansions[pattern])
                for synonym in synonyms:
                    if synonym not in existing:
                        self._config.expansions[pattern].append(synonym)
            else:
                self._config.expansions[pattern] = list(synonyms)


def create_query_expander(
    expansions: Optional[Dict[str, List[str]]] = None,
    max_expansions: int = 5,
) -> QueryExpander:
    """Factory function to create a QueryExpander.

    Args:
        expansions: Expansion dictionary (empty if None).
        max_expansions: Maximum query variations to return.

    Returns:
        Configured QueryExpander instance.
    """
    config = QueryExpansionConfig(
        expansions=expansions or {},
        max_expansions=max_expansions,
    )
    return QueryExpander(config)


__all__ = [
    "QueryExpander",
    "QueryExpansionConfig",
    "ExpandedQuery",
    "QueryExpanderProtocol",
    "create_query_expander",
]
