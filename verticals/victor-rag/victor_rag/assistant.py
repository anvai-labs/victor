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

"""RAGAssistant - Retrieval-Augmented Generation vertical.

This module defines the RAGAssistant vertical showcasing a complete RAG
implementation with document ingestion, vector search, and query generation.

Features:
- Document ingestion from multiple formats (PDF, Markdown, Text, Code)
- LanceDB vector storage (embedded, no server)
- Hybrid search combining vector + full-text
- Semantic chunking with configurable overlap
- Interactive TUI for document management
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from victor_contracts.verticals.protocols.base import VerticalBase
from victor_contracts.core.types import StageDefinition, TieredToolConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Protocol types used in annotations only

# Runtime-resolved names (lazy imports inside methods that need them)
# MiddlewareProtocol, SafetyExtensionProtocol, TieredToolConfig,
# ModeConfigProviderProtocol, PromptContributorProtocol,
# ToolDependencyProviderProtocol — accessed via victor.framework.extensions
# in the methods that use them, not at module level.

# ToolNames accessed at class body level — use SDK re-export
from victor_contracts.constants.tool_names import ToolNames


class RAGAssistant(VerticalBase):
    """Retrieval-Augmented Generation assistant vertical.

    A complete RAG implementation demonstrating:
    - Document ingestion and indexing
    - Vector search with LanceDB
    - Query processing with context retrieval
    - Source attribution and citations

    Example:
        from victor.rag import RAGAssistant

        config = RAGAssistant.get_config()
        agent = await Agent.create(
            tools=config.tools,
            vertical=RAGAssistant,
        )
    """

    name = "rag"
    description = "Retrieval-Augmented Generation assistant for document Q&A"
    version = "1.0.0"

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        return cls.description

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get tools for RAG operations.

        Uses canonical tool names from victor.tools.tool_names.

        Returns:
            List of RAG-specific tool names
        """
        return [
            # RAG-specific tools
            "rag_ingest",  # Ingest documents into the store
            "rag_search",  # Search for relevant chunks
            "rag_query",  # Query with context retrieval
            "rag_list",  # List indexed documents
            "rag_delete",  # Delete documents
            "rag_stats",  # Get store statistics
            # Filesystem for document access (canonical names)
            ToolNames.READ,
            ToolNames.LS,
            # Web for fetching web content
            ToolNames.WEB_FETCH,
            # Shell for document processing
            ToolNames.SHELL,
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get RAG-focused system prompt.

        Returns:
            System prompt optimized for RAG operations
        """
        return """You are Victor, a Retrieval-Augmented Generation (RAG) assistant.

Your capabilities:
- Ingest documents from files (PDF, Markdown, Text, Code)
- Index documents with semantic embeddings using LanceDB
- Search for relevant information using hybrid search
- Answer questions with source citations
- Manage the document knowledge base

Guidelines:
- Always search the knowledge base before answering questions
- Cite sources by referencing document names and page/section numbers
- If information isn't in the knowledge base, say so clearly
- Use rag_ingest to add new documents when requested
- Use rag_search for exploratory queries
- Use rag_query for direct Q&A with automatic context retrieval

Workflow:
1. For questions: Use rag_query to search and get relevant context
2. For exploration: Use rag_search to find related chunks
3. For adding docs: Use rag_ingest with file paths or URLs
4. Always cite your sources in answers

Example interaction:
User: "What does the documentation say about authentication?"
You: [Use rag_query tool with query="authentication"]
     Then synthesize the results with citations.
"""

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get RAG-specific workflow stages.

        Uses canonical tool names from victor.tools.tool_names.

        Returns:
            Stage definitions for RAG workflow
        """
        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Ready to accept RAG queries",
                tools={"rag_search", "rag_query", "rag_list", "rag_stats"},
                next_stages={"INGESTING", "SEARCHING", "QUERYING"},
            ),
            "INGESTING": StageDefinition(
                name="INGESTING",
                description="Ingesting documents into knowledge base",
                tools={"rag_ingest", ToolNames.READ, ToolNames.LS, ToolNames.WEB_FETCH},
                next_stages={"INITIAL", "SEARCHING"},
            ),
            "SEARCHING": StageDefinition(
                name="SEARCHING",
                description="Searching knowledge base",
                tools={"rag_search", "rag_query"},
                next_stages={"INITIAL", "QUERYING", "SYNTHESIZING"},
            ),
            "QUERYING": StageDefinition(
                name="QUERYING",
                description="Processing query with retrieved context",
                tools={"rag_query"},
                next_stages={"SYNTHESIZING", "SEARCHING"},
            ),
            "SYNTHESIZING": StageDefinition(
                name="SYNTHESIZING",
                description="Synthesizing answer from retrieved context",
                tools=set(),  # LLM response only
                next_stages={"INITIAL"},
            ),
        }

    @classmethod
    def get_tiered_tool_config(cls) -> Optional[TieredToolConfig]:
        """Get tiered tool configuration for RAG.

        RAG has unique requirements:
        - No grep tool (document-focused, not code-focused)
        - Custom stage_tools for RAG workflow stages

        Returns:
            Tool configuration with tiers
        """
        return TieredToolConfig(
            # Mandatory: essential tools for any RAG task
            # RAG is document-focused, so no grep (unlike other verticals)
            mandatory={ToolNames.READ, ToolNames.LS},
            # Vertical core: RAG-specific tools always available
            vertical_core={
                "rag_search",
                "rag_query",
                "rag_list",
                "rag_stats",
            },
            # Stage tools: tools available at specific workflow stages
            # Note: Uses sets as per core schema (not lists)
            stage_tools={
                "INGESTING": {
                    "rag_ingest",
                    ToolNames.READ,
                    ToolNames.LS,
                    ToolNames.WEB_FETCH,
                },
                "SEARCHING": {"rag_search"},
                "QUERYING": {"rag_query"},
            },
            # Analysis tasks should not have write tools
            readonly_only_for_analysis=True,
        )

    # Extension providers delegated to base class (OCP/caching compliance)
    # RAG previously overrode get_extensions() directly, bypassing base class caching.
    # Now we implement the individual getter methods and let VerticalBase.get_extensions()
    # handle caching and aggregation.

    @classmethod
    def get_middleware(cls) -> list:
        """Get RAG middleware (none needed).

        Returns:
            Empty list - RAG doesn't use custom middleware
        """
        return []

    @classmethod
    def get_safety_extension(cls):
        """Get RAG safety extension.

        Returns:
            RAGSafetyExtension instance
        """
        return cls._get_extension_factory("safety_extension", "victor_rag.safety")

    @classmethod
    def get_prompt_contributor(cls):
        """Get RAG prompt contributor.

        Returns:
            RAGPromptContributor instance
        """
        return cls._get_extension_factory("prompt_contributor", "victor_rag.prompts")

    @classmethod
    def get_rl_config_provider(cls):
        """Get the RL configuration provider for RAG vertical.

        Returns:
            RAGRLConfig instance
        """
        return cls._get_extension_factory("rl_config_provider", "victor_rag.rl")

    @classmethod
    def get_team_spec_provider(cls):
        """Get the team specification provider for RAG vertical.

        Returns:
            RAGTeamSpecProvider instance
        """
        return cls._get_extension_factory("team_spec_provider", "victor_rag.teams")

    @classmethod
    def get_capability_configs(cls) -> Dict[str, Any]:
        """Get RAG capability configurations for centralized storage.

        Returns default RAG configuration for VerticalContext storage.
        This replaces direct orchestrator.rag_config assignment.

        Returns:
            Dict with default RAG capability configurations
        """
        from victor_rag.capabilities import get_capability_configs

        return get_capability_configs()
