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

"""RAG vertical compute handlers.

Domain-specific handlers for RAG workflows:
- vector_search: Similarity search in vector stores
- chunk_processor: Document chunking for embedding

Usage:
    from victor.rag import handlers
    handlers.register_handlers()

    # In YAML workflow:
    - id: search_docs
      type: compute
      handler: vector_search
      inputs:
        query: $ctx.user_query
        top_k: 5
        collection: documents
      output: search_results
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor_contracts.workflow_runtime import ComputeNode
    from victor_contracts.workflow_runtime import NodeResult, WorkflowContext

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchHandler:
    """Execute vector similarity search.

    Searches vector store for similar documents.

    Example YAML:
        - id: search_docs
          type: compute
          handler: vector_search
          inputs:
            query: $ctx.user_query
            top_k: 5
            collection: documents
          output: search_results
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor_contracts.workflow_runtime import NodeResult, ExecutorNodeStatus

        start_time = time.time()

        query = node.input_mapping.get("query", "")
        if isinstance(query, str) and query.startswith("$ctx."):
            query = context.get(query[5:]) or query

        top_k = node.input_mapping.get("top_k", 5)
        collection = node.input_mapping.get("collection", "default")

        try:
            result = await tool_registry.execute(
                "vector_search",
                query=query,
                top_k=top_k,
                collection=collection,
            )

            output = {
                "query": query,
                "results": result.output if result.success else [],
                "count": len(result.output) if result.success and result.output else 0,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=(
                    ExecutorNodeStatus.COMPLETED if result.success else ExecutorNodeStatus.FAILED
                ),
                output=output,
                error=result.error if not result.success else None,
                duration_seconds=time.time() - start_time,
                tool_calls_used=1,
            )
        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


@dataclass
class ChunkProcessorHandler:
    """Process documents into chunks for embedding.

    Splits documents using configurable strategies.

    Example YAML:
        - id: chunk_docs
          type: compute
          handler: chunk_processor
          inputs:
            documents: $ctx.raw_docs
            strategy: semantic
            chunk_size: 1024
            overlap: 128
          output: chunks
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor_contracts.workflow_runtime import NodeResult, ExecutorNodeStatus

        start_time = time.time()

        docs_key = node.input_mapping.get("documents")
        documents = context.get(docs_key) if docs_key else []
        strategy = node.input_mapping.get("strategy", "fixed")
        chunk_size = node.input_mapping.get("chunk_size", 1024)
        overlap = node.input_mapping.get("overlap", 128)

        try:
            chunks = []
            doc_list = documents if isinstance(documents, list) else [documents]
            for doc in doc_list:
                doc_chunks = self._chunk_document(doc, strategy, chunk_size, overlap)
                chunks.extend(doc_chunks)

            output = {
                "strategy": strategy,
                "chunk_size": chunk_size,
                "total_chunks": len(chunks),
                "chunks": chunks,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output=output,
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _chunk_document(
        self, document: Any, strategy: str, chunk_size: int, overlap: int
    ) -> List[Dict[str, Any]]:
        """Chunk a single document."""
        text = str(document) if not isinstance(document, str) else document
        chunks = []

        if strategy == "fixed":
            pos = 0
            while pos < len(text):
                end = min(pos + chunk_size, len(text))
                chunks.append({"text": text[pos:end], "start": pos, "end": end})
                pos += chunk_size - overlap
        elif strategy == "sentence":
            sentences = re.split(r"(?<=[.!?])\s+", text)
            current_chunk = ""
            start = 0
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > chunk_size:
                    if current_chunk:
                        chunks.append(
                            {
                                "text": current_chunk.strip(),
                                "start": start,
                                "end": start + len(current_chunk),
                            }
                        )
                    current_chunk = sentence
                    start = start + len(current_chunk)
                else:
                    current_chunk += " " + sentence
            if current_chunk:
                chunks.append(
                    {
                        "text": current_chunk.strip(),
                        "start": start,
                        "end": start + len(current_chunk),
                    }
                )
        elif strategy == "paragraph":
            paragraphs = text.split("\n\n")
            pos = 0
            for para in paragraphs:
                if para.strip():
                    chunks.append({"text": para.strip(), "start": pos, "end": pos + len(para)})
                pos += len(para) + 2

        return chunks


HANDLERS = {
    "vector_search": VectorSearchHandler(),
    "chunk_processor": ChunkProcessorHandler(),
}


def register_handlers() -> None:
    """Register RAG handlers with the workflow executor."""
    from victor_contracts.workflow_runtime import register_compute_handler

    for name, handler in HANDLERS.items():
        register_compute_handler(name, handler)
        logger.debug(f"Registered RAG handler: {name}")


__all__ = [
    "VectorSearchHandler",
    "ChunkProcessorHandler",
    "HANDLERS",
    "register_handlers",
]
