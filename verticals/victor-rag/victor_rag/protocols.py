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
from typing import Any, Dict, List, Optional

# Import victor-contracts protocols (NO runtime dependency on victor-ai!)
from victor_contracts.verticals.protocols import (
    ToolProvider,
    ToolSelectionStrategy,
    SafetyProvider,
    PromptProvider,
    WorkflowProvider,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Provider
# =============================================================================


class RAGToolProvider(ToolProvider):
    """Tool provider for RAG vertical.

    Provides the list of tools available to the RAG assistant.
    """

    def get_tools(self) -> List[str]:
        """Return list of tool names for RAG vertical."""
        return [
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


class RAGToolSelectionStrategy(ToolSelectionStrategy):
    """Stage-aware tool selection for RAG tasks."""

    def get_tools_for_stage(self, stage: str, task_type: str) -> List[str]:
        """Return optimized tools for given stage and task type."""
        stage_tools: Dict[str, List[str]] = {
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

        return stage_tools.get(stage, ["semantic_search", "knowledge_query"])


# =============================================================================
# Safety Provider
# =============================================================================


class RAGSafetyProvider(SafetyProvider):
    """Safety provider for RAG vertical.

    Provides RAG-specific safety patterns for knowledge base operations.
    """

    def __init__(self):
        self._dangerous_patterns = [
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

    def get_extensions(self) -> List[Any]:
        """Return safety extensions for RAG."""
        return []

    def get_bash_patterns(self) -> List[Any]:
        """Return bash command patterns to monitor."""
        return self._dangerous_patterns

    def get_file_patterns(self) -> List[Any]:
        """Return file operation patterns to monitor."""
        return []

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Return tool-specific restrictions."""
        return {
            "vector_delete": ["--all", "--force"],
            "knowledge_delete": ["--all", "--force"],
        }


# =============================================================================
# Prompt Provider
# =============================================================================


class RAGPromptProvider(PromptProvider):
    """Prompt provider for RAG vertical.

    Provides system prompt sections for knowledge retrieval tasks.
    """

    def get_system_prompt_sections(self) -> Dict[str, str]:
        """Return system prompt sections."""
        return {
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

    def get_task_type_hints(self) -> Dict[str, Any]:
        """Return task type hints for RAG."""
        return {
            "ingest": {
                "hint": "[INGEST] Process and index documents into the knowledge base.",
                "tool_budget": 10,
            },
            "query": {
                "hint": "[QUERY] Search the knowledge base and provide relevant information.",
                "tool_budget": 5,
            },
            "answer": {
                "hint": (
                    "[ANSWER] Generate answers based on retrieved context with " "proper citations."
                ),
                "tool_budget": 8,
            },
        }

    def get_prompt_contributors(self) -> List[Any]:
        """Return prompt contributors for RAG."""
        return []


# =============================================================================
# Workflow Provider
# =============================================================================


class RAGWorkflowProvider(WorkflowProvider):
    """Workflow provider for RAG vertical.

    Provides RAG-specific workflow definitions.
    """

    def get_workflows(self) -> Dict[str, Any]:
        """Return workflow specifications."""
        return {
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

    def get_workflow(self, name: str) -> Optional[Any]:
        """Get a specific workflow by name."""
        return self.get_workflows().get(name)

    def list_workflows(self) -> List[str]:
        """List available workflow names."""
        return list(self.get_workflows().keys())


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "RAGToolProvider",
    "RAGToolSelectionStrategy",
    "RAGSafetyProvider",
    "RAGPromptProvider",
    "RAGWorkflowProvider",
]
