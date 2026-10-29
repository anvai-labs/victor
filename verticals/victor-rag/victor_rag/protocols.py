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

"""Victor extension protocol implementations for victor-rag.

This module provides protocol implementations that can be discovered via
the Victor extension entry point system, enabling the RAG vertical to
register capabilities with the framework without direct dependencies.
"""

from __future__ import annotations

import logging

# Import victor-contracts protocols (NO runtime dependency on victor-ai!)
from victor_contracts.verticals import (
    make_pattern_safety_provider,
    make_prompt_provider,
    make_stage_tool_strategy,
    make_tool_provider,
    make_workflow_provider,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Provider
# =============================================================================


RAG_TOOLS = [
    # Core filesystem tools
    "read",
    "write",
    "grep",
    "ls",
    # Vector database tools
    "vector_search",
    "vector_add",
    "vector_delete",
    "vector_index",
    # Document processing
    "document_ingest",
    "document_split",
    "document_chunk",
    "document_parse",
    # Embedding tools
    "embedding_create",
    "embedding_batch",
    "embedding_search",
    # Knowledge base tools
    "knowledge_query",
    "knowledge_add",
    "knowledge_update",
    # Retrieval tools
    "semantic_search",
    "hybrid_search",
    "rerank_results",
    # Citation tools
    "citation_extract",
    "citation_verify",
]

RAG_STAGE_TOOLS = {
    "ingest": [
        "document_ingest",
        "document_parse",
        "document_chunk",
        "vector_add",
    ],
    "index": ["embedding_create", "vector_index", "embedding_batch"],
    "retrieve": ["semantic_search", "vector_search", "hybrid_search"],
    "rerank": ["rerank_results", "vector_search"],
    "generate": ["knowledge_query", "citation_extract"],
}

RAG_DANGEROUS_PATTERNS = [
    # Vector database dangerous commands
    {
        "pattern": "vector_delete --all",
        "description": "Delete entire vector index",
    },
    {"pattern": "vector_index --drop", "description": "Drop vector collection"},
    {
        "pattern": "knowledge_delete --force",
        "description": "Force delete knowledge",
    },
]

RAG_TOOL_RESTRICTIONS = {
    "vector_delete": ["--all", "--force"],
    "knowledge_delete": ["--all", "--force"],
}

RAG_PROMPT_SECTIONS = {
    "role": (
        "You are a RAG (Retrieval-Augmented Generation) assistant specializing "
        "in knowledge retrieval and document analysis."
    ),
    "expertise": (
        "You have expertise in vector databases, semantic search, document "
        "processing, and citation management."
    ),
    "retrieval": (
        "Always use retrieved context to inform your answers. Cite sources "
        "when referencing specific documents."
    ),
    "hallucination": (
        "Avoid hallucinations by only stating information that is supported "
        "by the retrieved context."
    ),
}

RAG_TASK_TYPE_HINTS = {
    "ingest": {
        "hint": "[INGEST] Process and index documents into the knowledge base.",
        "tool_budget": 10,
    },
    "query": {
        "hint": "[QUERY] Search the knowledge base and provide relevant information.",
        "tool_budget": 5,
    },
    "answer": {
        "hint": ("[ANSWER] Generate answers based on retrieved context with " "proper citations."),
        "tool_budget": 8,
    },
}

RAG_WORKFLOWS = {
    "ingest_documents": {
        "name": "Ingest Documents",
        "description": "Process and index documents into the knowledge base",
        "stages": ["parse", "chunk", "embed", "index"],
    },
    "knowledge_query": {
        "name": "Knowledge Query",
        "description": "Query the knowledge base and generate answers",
        "stages": ["retrieve", "rerank", "generate"],
    },
}

RAGToolProvider = make_tool_provider(
    "RAGToolProvider", RAG_TOOLS, "Tool provider for the RAG vertical."
)
RAGToolSelectionStrategy = make_stage_tool_strategy(
    "RAGToolSelectionStrategy",
    RAG_STAGE_TOOLS,
    ["semantic_search", "knowledge_query"],
    "Stage-aware tool selection for RAG tasks.",
)
RAGSafetyProvider = make_pattern_safety_provider(
    "RAGSafetyProvider",
    RAG_DANGEROUS_PATTERNS,
    RAG_TOOL_RESTRICTIONS,
    doc="Safety provider for RAG knowledge-base operations.",
)
RAGPromptProvider = make_prompt_provider(
    "RAGPromptProvider",
    RAG_PROMPT_SECTIONS,
    RAG_TASK_TYPE_HINTS,
    "Prompt provider for RAG knowledge-retrieval tasks.",
)
RAGWorkflowProvider = make_workflow_provider(
    "RAGWorkflowProvider", RAG_WORKFLOWS, "Workflow provider for the RAG vertical."
)


__all__ = [
    "RAGToolProvider",
    "RAGToolSelectionStrategy",
    "RAGSafetyProvider",
    "RAGPromptProvider",
    "RAGWorkflowProvider",
]
