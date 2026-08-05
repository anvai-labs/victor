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

"""Search protocol interfaces for semantic and hybrid search.

This module defines protocols for search implementations across verticals,
enabling Dependency Inversion Principle (DIP) compliance. Verticals implement
these protocols while depending only on abstractions.

Protocols:
- ISemanticSearch: Protocol for semantic search implementations
- IIndexable: Protocol for indexable content sources

Usage:
    from victor.protocols.search import ISemanticSearch

    class CodebaseSearch:
        '''Coding vertical semantic search implementation.'''

        async def search(
            self,
            query: str,
            max_results: int = 10,
            filter_metadata: Optional[Dict[str, Any]] = None,
        ) -> List[SearchHit]:
            # Implementation using code-specific embeddings
            ...

    # Type checking verifies protocol compliance
    searcher: ISemanticSearch = CodebaseSearch()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from victor.core.search_types import SearchHit


@runtime_checkable
class ISemanticSearch(Protocol):
    """Protocol for semantic search implementations.

    This protocol defines the interface for semantic search across all verticals.
    Implementations may use different embedding models, vector stores, and
    indexing strategies while conforming to this interface.

    Examples:
        - Coding vertical: AST-aware code search with symbol embeddings
        - RAG vertical: Document chunk search with hybrid retrieval
        - Research vertical: Paper section search with citation awareness

    Attributes:
        is_indexed: Whether the search provider has indexed content
    """

    @property
    def is_indexed(self) -> bool:
        """Whether the search provider has indexed content.

        Returns:
            True if there is searchable indexed content, False otherwise
        """
        ...

    async def search(
        self,
        query: str,
        max_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchHit]:
        """Execute semantic search on indexed content.

        Args:
            query: Natural language search query
            max_results: Maximum number of results to return
            filter_metadata: Optional metadata filters (e.g., {"file_type": "py"})

        Returns:
            List of SearchHit objects ordered by relevance score (descending)

        Example:
            results = await searcher.search(
                query="authentication error handling",
                max_results=5,
                filter_metadata={"file_type": "py"}
            )
            for hit in results:
                print(f"{hit.file_path}:{hit.line_number} - {hit.score:.2f}")
        """
        ...


@runtime_checkable
class IIndexable(Protocol):
    """Protocol for indexable content sources.

    This protocol defines the interface for content that can be indexed
    for semantic search. Implementations handle the specifics of content
    extraction, chunking, and embedding generation.
    """

    async def index_document(
        self,
        file_path: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Index a document for semantic search.

        Args:
            file_path: Path or identifier for the document
            content: Text content to index
            metadata: Optional metadata to store with the document

        Raises:
            IndexError: If indexing fails
        """
        ...

    async def remove_document(self, file_path: str) -> bool:
        """Remove a document from the index.

        Args:
            file_path: Path or identifier for the document to remove

        Returns:
            True if document was removed, False if not found
        """
        ...

    async def clear_index(self) -> None:
        """Clear all indexed content.

        Use with caution - this removes all indexed data.
        """
        ...


@runtime_checkable
class IIssueLocalizer(Protocol):
    """Optional protocol for graph-guided issue localization capabilities.

    Implementations return file-level or symbol-level candidates for a natural
    language issue description, typically by combining semantic retrieval with
    structural graph expansion.
    """

    async def localize_issue(
        self,
        issue_description: str,
        language: Optional[str] = None,
        top_k: int = 10,
        include_graph_context: bool = True,
        context_limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return ranked localization candidates for an issue description."""
        ...


@runtime_checkable
class IChangeImpactAnalyzer(Protocol):
    """Optional protocol for graph-guided change-impact / blast-radius analysis."""

    async def analyze_change_impact(
        self,
        change_description: str,
        language: Optional[str] = None,
        top_k: int = 10,
        include_graph_context: bool = True,
        context_limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return ranked impact candidates for a proposed change."""
        ...


@runtime_checkable
class ISemanticSearchWithIndexing(ISemanticSearch, IIndexable, Protocol):
    """Combined protocol for searchable and indexable implementations.

    Use this when you need both search and indexing capabilities in a
    single implementation.

    Example:
        class CodebaseIndex:
            '''Full-featured codebase search with indexing.'''

            @property
            def is_indexed(self) -> bool:
                return self._index_count > 0

            async def search(self, query: str, ...) -> List[SearchHit]:
                ...

            async def index_document(self, file_path: str, ...) -> None:
                ...

            async def remove_document(self, file_path: str) -> bool:
                ...

            async def clear_index(self) -> None:
                ...
    """

    pass


__all__ = [
    "IChangeImpactAnalyzer",
    "IIssueLocalizer",
    "ISemanticSearch",
    "IIndexable",
    "ISemanticSearchWithIndexing",
]
