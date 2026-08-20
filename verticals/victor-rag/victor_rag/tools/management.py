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

"""RAG Management Tools - List, delete, and get stats for the RAG store."""

import logging
from datetime import datetime
from typing import Dict

from victor.tools.base import BaseTool, CostTier, ToolResult

logger = logging.getLogger(__name__)


class RAGListTool(BaseTool):
    """List all documents in the RAG knowledge base."""

    name = "rag_list"
    description = "List all documents in the RAG knowledge base"

    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.FREE

    async def execute(self, **kwargs) -> ToolResult:
        """List all documents.

        Returns:
            ToolResult with document list
        """

        try:
            store = self._get_document_store()
            await store.initialize()

            docs = await store.list_documents()

            if not docs:
                return ToolResult(
                    success=True,
                    output="No documents in the knowledge base. Use rag_ingest to add documents.",
                )

            # Group SEC filings separately for better display
            sec_docs = [d for d in docs if d.id.startswith("sec_")]
            other_docs = [d for d in docs if not d.id.startswith("sec_")]

            output_parts = [f"Documents in knowledge base ({len(docs)} total):\n"]

            # Show SEC filings summary if any
            if sec_docs:
                output_parts.append(f"\n{'='*50}")
                output_parts.append(f"SEC FILINGS ({len(sec_docs)} documents):")
                output_parts.append(f"{'='*50}")

                # Group by ticker
                by_ticker: Dict[str, list] = {}
                for doc in sec_docs:
                    symbol = doc.metadata.get("symbol", "UNKNOWN")
                    if symbol not in by_ticker:
                        by_ticker[symbol] = []
                    by_ticker[symbol].append(doc)

                # Show tickers
                output_parts.append(f"\nIngested Tickers ({len(by_ticker)} companies):")
                tickers_line = ", ".join(sorted(by_ticker.keys()))
                output_parts.append(f"  {tickers_line}")

                # Group by sector
                by_sector: Dict[str, int] = {}
                for doc in sec_docs:
                    sector = doc.metadata.get("sector", "Other")
                    by_sector[sector] = by_sector.get(sector, 0) + 1

                output_parts.append("\nBy Sector:")
                for sector, count in sorted(by_sector.items(), key=lambda x: -x[1]):
                    output_parts.append(f"  {sector}: {count} filings")

                # Show details per ticker
                output_parts.append("\nDetails:")
                for symbol in sorted(by_ticker.keys()):
                    ticker_docs = by_ticker[symbol]
                    company = ticker_docs[0].metadata.get("company", symbol)
                    filing_types = [d.metadata.get("filing_type", "N/A") for d in ticker_docs]
                    dates = [d.metadata.get("filing_date", "N/A") for d in ticker_docs]
                    output_parts.append(f"  {symbol:8} {company}")
                    for ft, dt in zip(filing_types, dates):
                        output_parts.append(f"           - {ft} ({dt})")

            # Show other documents
            if other_docs:
                output_parts.append(f"\n{'='*50}")
                output_parts.append(f"OTHER DOCUMENTS ({len(other_docs)} documents):")
                output_parts.append(f"{'='*50}")

                for doc in other_docs:
                    created = datetime.fromtimestamp(doc.created_at).strftime("%Y-%m-%d %H:%M")
                    output_parts.append(
                        f"  [{doc.id}]\n"
                        f"    Source: {doc.source}\n"
                        f"    Type: {doc.doc_type}\n"
                        f"    Characters: {len(doc.content):,}\n"
                        f"    Created: {created}"
                    )

            return ToolResult(
                success=True,
                output="\n".join(output_parts),
            )

        except Exception as e:
            logger.exception(f"Failed to list documents: {e}")
            return ToolResult(
                success=False,
                output=f"Failed to list documents: {str(e)}",
            )

    def _get_document_store(self):
        """Get document store instance."""
        from victor_rag.document_store import DocumentStore

        if not hasattr(self, "_store"):
            self._store = DocumentStore()
        return self._store


class RAGDeleteTool(BaseTool):
    """Delete a document from the RAG knowledge base."""

    name = "rag_delete"
    description = "Delete a document from the RAG knowledge base by ID"

    parameters = {
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": "ID of the document to delete",
            },
        },
        "required": ["doc_id"],
    }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.LOW

    async def execute(self, doc_id: str, **kwargs) -> ToolResult:
        """Delete a document.

        Args:
            doc_id: Document ID to delete

        Returns:
            ToolResult with deletion status
        """

        try:
            store = self._get_document_store()
            await store.initialize()

            # Check if document exists
            doc = await store.get_document(doc_id)
            if not doc:
                return ToolResult(
                    success=False,
                    output=f"Document not found: {doc_id}",
                )

            # Delete document
            await store.delete_document(doc_id)

            return ToolResult(
                success=True,
                output=f"Successfully deleted document: {doc_id}\n" f"Source: {doc.source}",
            )

        except Exception as e:
            logger.exception(f"Failed to delete document: {e}")
            return ToolResult(
                success=False,
                output=f"Failed to delete document: {str(e)}",
            )

    def _get_document_store(self):
        """Get document store instance."""
        from victor_rag.document_store import DocumentStore

        if not hasattr(self, "_store"):
            self._store = DocumentStore()
        return self._store


class RAGStatsTool(BaseTool):
    """Get statistics for the RAG knowledge base."""

    name = "rag_stats"
    description = "Get statistics about the RAG knowledge base"

    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.FREE

    async def execute(self, **kwargs) -> ToolResult:
        """Get store statistics.

        Returns:
            ToolResult with statistics
        """

        try:
            store = self._get_document_store()
            await store.initialize()

            stats = await store.get_stats()
            docs = await store.list_documents()

            # Calculate additional stats
            total_chars = sum(len(doc.content) for doc in docs)
            doc_types = {}
            for doc in docs:
                doc_types[doc.doc_type] = doc_types.get(doc.doc_type, 0) + 1

            last_updated = ""
            if stats.get("last_updated"):
                last_updated = datetime.fromtimestamp(stats["last_updated"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            output = (
                f"RAG Knowledge Base Statistics\n"
                f"{'=' * 40}\n"
                f"Total Documents: {stats.get('total_documents', 0)}\n"
                f"Total Chunks: {stats.get('total_chunks', 0)}\n"
                f"Total Characters: {total_chars:,}\n"
                f"Total Searches: {stats.get('total_searches', 0)}\n"
                f"Last Updated: {last_updated or 'Never'}\n\n"
                f"Document Types:\n"
            )

            for dtype, count in doc_types.items():
                output += f"  {dtype}: {count}\n"

            return ToolResult(
                success=True,
                output=output,
            )

        except Exception as e:
            logger.exception(f"Failed to get stats: {e}")
            return ToolResult(
                success=False,
                output=f"Failed to get statistics: {str(e)}",
            )

    def _get_document_store(self):
        """Get document store instance."""
        from victor_rag.document_store import DocumentStore

        if not hasattr(self, "_store"):
            self._store = DocumentStore()
        return self._store
