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

"""RAG (Retrieval-Augmented Generation) Vertical Package.

This vertical provides a complete RAG implementation showcasing:
- Document ingestion from multiple formats (PDF, Markdown, Text, Code)
- Vector storage with LanceDB (embedded, no server needed)
- Hybrid search (vector + full-text)
- Semantic chunking with overlap
- Interactive TUI for document management and querying
- Reranking for improved relevance

Package Structure:
    assistant.py        - RAGAssistant vertical class
    document_store.py   - LanceDB-based document storage
    chunker.py          - Intelligent document chunking
    tools/              - RAG-specific tools (ingest, search, query)
    ui/                 - Interactive TUI components
    workflows/          - RAG-specific workflows

Usage:
    from victor_rag import RAGAssistant

    # Get vertical configuration
    config = RAGAssistant.get_config()

    # Create agent with RAG vertical
    agent = await Agent.create(
        tools=config.tools,
        vertical=RAGAssistant,
    )
"""

from typing import TYPE_CHECKING, Any

from victor_rag.assistant import RAGAssistant
from victor_rag.prompts import RAGPromptContributor
from victor_rag.mode_config import RAGModeConfigProvider
from victor_rag.capabilities import RAGCapabilityProvider

if TYPE_CHECKING:
    from victor_rag.document_store import (
        Document,
        DocumentChunk,
        DocumentSearchResult,
        DocumentStore,
        DocumentStoreConfig,
    )
    from victor_rag.chunker import DocumentChunker, ChunkingConfig
    from victor_rag.tools import (
        RAGIngestTool,
        RAGSearchTool,
        RAGQueryTool,
        RAGListTool,
        RAGDeleteTool,
        RAGStatsTool,
    )


__all__ = [
    # Main vertical
    "RAGAssistant",
    # Document store
    "Document",
    "DocumentChunk",
    "DocumentSearchResult",
    "DocumentStore",
    "DocumentStoreConfig",
    # Chunking
    "DocumentChunker",
    "ChunkingConfig",
    # Extensions
    "RAGPromptContributor",
    "RAGModeConfigProvider",
    "RAGCapabilityProvider",
    # Tools
    "RAGIngestTool",
    "RAGSearchTool",
    "RAGQueryTool",
    "RAGListTool",
    "RAGDeleteTool",
    "RAGStatsTool",
]

# Enhanced features with new coordinators
from victor_rag.safety_enhanced import (
    RAGSafetyRules,
    EnhancedRAGSafetyExtension,
)
from victor_rag.conversation_enhanced import (
    RAGContext,
    EnhancedRAGConversationManager,
)

__all__.extend(
    [
        "RAGSafetyRules",
        "EnhancedRAGSafetyExtension",
        "RAGContext",
        "EnhancedRAGConversationManager",
    ]
)


_DOCUMENT_EXPORTS = frozenset(
    {
        "Document",
        "DocumentChunk",
        "DocumentSearchResult",
        "DocumentStore",
        "DocumentStoreConfig",
    }
)
_TOOL_EXPORTS = frozenset(
    {
        "RAGIngestTool",
        "RAGSearchTool",
        "RAGQueryTool",
        "RAGListTool",
        "RAGDeleteTool",
        "RAGStatsTool",
    }
)


def __getattr__(name: str) -> Any:
    """Load storage backends only when their public symbols are requested."""
    if name in _DOCUMENT_EXPORTS:
        from victor_rag import document_store

        value = getattr(document_store, name)
    elif name in {"DocumentChunker", "ChunkingConfig"}:
        from victor_rag import chunker

        value = getattr(chunker, name)
    elif name in _TOOL_EXPORTS:
        from victor_rag import tools

        value = getattr(tools, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
