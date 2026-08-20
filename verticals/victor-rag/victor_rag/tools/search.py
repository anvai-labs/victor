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

"""RAG Search Tool - Search for relevant chunks in the RAG store."""

import logging
from typing import List, Optional

from victor.tools.base import BaseTool, CostTier, ToolResult

logger = logging.getLogger(__name__)


class RAGSearchTool(BaseTool):
    """Search for relevant chunks in the RAG knowledge base.

    Uses hybrid search (vector + full-text) to find the most relevant
    document chunks for a given query.

    Example:
        result = await tool.execute(
            query="How does authentication work?",
            k=10,
        )
    """

    name = "rag_search"
    description = "Search the RAG knowledge base for relevant document chunks"

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant chunks",
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return (default: 10)",
                "default": 10,
            },
            "doc_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of document IDs to search within",
            },
            "use_hybrid": {
                "type": "boolean",
                "description": "Use hybrid search (vector + full-text)",
                "default": True,
            },
        },
        "required": ["query"],
    }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.LOW

    async def execute(
        self,
        query: str,
        k: int = 10,
        doc_ids: Optional[List[str]] = None,
        use_hybrid: bool = True,
        **kwargs,
    ) -> ToolResult:
        """Execute search query.

        Args:
            query: Search query
            k: Number of results
            doc_ids: Optional document filter
            use_hybrid: Use hybrid search

        Returns:
            ToolResult with search results
        """

        try:
            store = self._get_document_store()
            await store.initialize()

            # Perform search
            results = await store.search(
                query=query,
                k=k,
                filter_doc_ids=doc_ids,
                use_hybrid=use_hybrid,
            )

            if not results:
                return ToolResult(
                    success=True,
                    output=f"No results found for query: '{query}'",
                )

            # Format results
            output_parts = [f"Found {len(results)} relevant chunks for: '{query}'\n"]

            for i, result in enumerate(results, 1):
                chunk = result.chunk
                source = result.doc_source or chunk.metadata.get("source", "unknown")

                output_parts.append(
                    f"[{i}] Score: {result.score:.3f} | Source: {source}\n"
                    f"    {chunk.content[:300]}..."
                    if len(chunk.content) > 300
                    else f"[{i}] Score: {result.score:.3f} | Source: {source}\n"
                    f"    {chunk.content}"
                )
                output_parts.append("")

            return ToolResult(
                success=True,
                output="\n".join(output_parts),
            )

        except Exception as e:
            logger.exception(f"Search failed: {e}")
            return ToolResult(
                success=False,
                output=f"Search failed: {str(e)}",
            )

    def _get_document_store(self):
        """Get document store instance."""
        from victor_rag.document_store import DocumentStore

        if not hasattr(self, "_store"):
            self._store = DocumentStore()
        return self._store
