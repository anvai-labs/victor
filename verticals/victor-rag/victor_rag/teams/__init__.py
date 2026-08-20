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

"""Teams integration for RAG vertical.

This package provides team specifications for common RAG tasks with
rich persona attributes for natural agent characterization.

Example:
    from victor_rag.teams import (
        get_team_for_task,
        RAG_TEAM_SPECS,
    )

    # Get team for a task type
    team_spec = get_team_for_task("search")
    print(f"Team: {team_spec.name}")
    print(f"Members: {len(team_spec.members)}")

Teams are auto-registered with the global TeamSpecRegistry on import,
enabling cross-vertical team discovery via:
    from victor_contracts.team_schema import get_runtime_team_registry as get_team_registry
    registry = get_team_registry()
    rag_teams = registry.find_by_vertical("rag")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from victor_contracts import TeamFormation, TeamMemberSpec


@dataclass
class RAGRoleConfig:
    """Configuration for a RAG-specific role.

    Attributes:
        base_role: Base agent role (researcher, planner, executor, reviewer)
        tools: Tools available to this role
        tool_budget: Default tool budget
        description: Role description
    """

    base_role: str
    tools: List[str]
    tool_budget: int
    description: str = ""


# RAG-specific roles with tool allocations
RAG_ROLES: Dict[str, RAGRoleConfig] = {
    "document_parser": RAGRoleConfig(
        base_role="researcher",
        tools=["read", "ls", "web_fetch"],
        tool_budget=15,
        description="Parses and extracts content from documents",
    ),
    "content_indexer": RAGRoleConfig(
        base_role="executor",
        tools=["rag_ingest", "rag_stats"],
        tool_budget=20,
        description="Indexes content into the knowledge base",
    ),
    "search_specialist": RAGRoleConfig(
        base_role="researcher",
        tools=["rag_search", "rag_query", "rag_list"],
        tool_budget=20,
        description="Searches and retrieves relevant information",
    ),
    "answer_synthesizer": RAGRoleConfig(
        base_role="executor",
        tools=["rag_query", "rag_search"],
        tool_budget=15,
        description="Synthesizes answers from retrieved context",
    ),
    "fact_checker": RAGRoleConfig(
        base_role="reviewer",
        tools=["rag_search", "rag_query"],
        tool_budget=15,
        description="Verifies facts and citations in answers",
    ),
    "index_manager": RAGRoleConfig(
        base_role="executor",
        tools=["rag_list", "rag_delete", "rag_stats"],
        tool_budget=15,
        description="Manages and maintains the knowledge base index",
    ),
}


@dataclass
class RAGTeamSpec:
    """Specification for a RAG team.

    Attributes:
        name: Team name
        description: Team description
        formation: How agents are organized
        members: Team member specifications
        total_tool_budget: Total tool budget for the team
        max_iterations: Maximum iterations
    """

    name: str
    description: str
    formation: TeamFormation
    members: List[TeamMemberSpec]
    total_tool_budget: int = 100
    max_iterations: int = 50


# Pre-defined team specifications with rich personas
RAG_TEAM_SPECS: Dict[str, RAGTeamSpec] = {
    "ingest_team": RAGTeamSpec(
        name="Document Ingestion Team",
        description="Parallel document processing for efficient ingestion",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Parse and extract content from input documents",
                name="Document Parser",
                tool_budget=15,
                backstory=(
                    "You are an expert document parser with deep knowledge of various file formats "
                    "including PDF, Markdown, HTML, and plain text. You understand document structure, "
                    "can identify sections, headers, and metadata. You extract content cleanly while "
                    "preserving important formatting and context."
                ),
                expertise=["document parsing", "content extraction", "file formats"],
                personality="meticulous and thorough; preserves document structure",
                memory=True,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Process and chunk content for optimal retrieval",
                name="Content Indexer",
                tool_budget=20,
                backstory=(
                    "You are a content indexing specialist who understands semantic chunking and "
                    "embedding strategies. You know how to split documents to preserve meaning while "
                    "keeping chunks at optimal sizes for retrieval. You handle overlapping context "
                    "and maintain document relationships."
                ),
                expertise=["semantic chunking", "embeddings", "index optimization"],
                personality="efficient and systematic; optimizes for retrieval quality",
                cache=True,
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Validate ingestion and ensure content quality",
                name="Quality Validator",
                tool_budget=10,
                backstory=(
                    "You are a quality assurance specialist who verifies that documents are "
                    "properly ingested. You check for completeness, proper chunking, and that "
                    "metadata is preserved. You identify any issues that might affect retrieval."
                ),
                expertise=["quality assurance", "validation", "data integrity"],
                personality="detail-oriented; catches issues early",
            ),
        ],
        total_tool_budget=45,
    ),
    "search_team": RAGTeamSpec(
        name="Multi-Strategy Search Team",
        description="Parallel multi-strategy search for comprehensive retrieval",
        formation=TeamFormation.PARALLEL,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Perform semantic vector search for relevant content",
                name="Semantic Searcher",
                tool_budget=15,
                backstory=(
                    "You are a semantic search specialist who understands embedding spaces and "
                    "similarity matching. You craft queries that capture semantic meaning and "
                    "find contextually relevant results even when exact keywords don't match."
                ),
                expertise=["semantic search", "embeddings", "similarity matching"],
                personality="intuitive; finds connections others miss",
                memory=True,
            ),
            TeamMemberSpec(
                role="researcher",
                goal="Perform keyword and metadata-based search",
                name="Keyword Searcher",
                tool_budget=15,
                backstory=(
                    "You are a keyword search expert who understands full-text search, filters, "
                    "and metadata queries. You use precise terminology and filters to find "
                    "exact matches and leverage document metadata effectively."
                ),
                expertise=["keyword search", "filtering", "metadata queries"],
                personality="precise and methodical; leaves no stone unturned",
            ),
            TeamMemberSpec(
                role="planner",
                goal="Merge and rank results from multiple search strategies",
                name="Result Ranker",
                tool_budget=10,
                backstory=(
                    "You are a search result ranking specialist who combines results from "
                    "multiple strategies. You understand relevance scoring, deduplication, "
                    "and how to present the most useful results to users."
                ),
                expertise=["result ranking", "deduplication", "relevance scoring"],
                personality="analytical; balances precision and recall",
            ),
        ],
        total_tool_budget=40,
    ),
    "synthesis_team": RAGTeamSpec(
        name="Answer Synthesis Team",
        description="Answer generation with fact-checking and citation",
        formation=TeamFormation.PIPELINE,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Retrieve and organize relevant context for the query",
                name="Context Gatherer",
                tool_budget=15,
                backstory=(
                    "You are an expert at gathering relevant context from the knowledge base. "
                    "You understand what information is needed to answer questions accurately "
                    "and comprehensively. You organize context logically for synthesis."
                ),
                expertise=["context retrieval", "information organization"],
                personality="thorough; ensures no relevant context is missed",
                memory=True,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Synthesize a comprehensive answer from retrieved context",
                name="Answer Synthesizer",
                tool_budget=15,
                backstory=(
                    "You are an expert at synthesizing clear, accurate answers from multiple "
                    "sources. You weave together information coherently, handle contradictions "
                    "gracefully, and always ground your answers in the retrieved context."
                ),
                expertise=["answer synthesis", "writing", "information integration"],
                personality="articulate and clear; explains complex topics simply",
            ),
            TeamMemberSpec(
                role="reviewer",
                goal="Verify facts and ensure proper source citations",
                name="Fact Checker",
                tool_budget=15,
                backstory=(
                    "You are a rigorous fact checker who verifies every claim against sources. "
                    "You ensure citations are accurate, detect any hallucinations, and confirm "
                    "that answers are fully grounded in the retrieved documents."
                ),
                expertise=["fact checking", "citation verification", "accuracy"],
                personality="skeptical and thorough; trusts but verifies",
            ),
        ],
        total_tool_budget=45,
    ),
    "maintenance_team": RAGTeamSpec(
        name="Index Maintenance Team",
        description="Knowledge base maintenance and optimization",
        formation=TeamFormation.SEQUENTIAL,
        members=[
            TeamMemberSpec(
                role="researcher",
                goal="Analyze index health and identify issues",
                name="Index Analyst",
                tool_budget=15,
                backstory=(
                    "You are an expert at analyzing knowledge base health. You identify stale "
                    "content, duplicates, indexing issues, and performance bottlenecks. You "
                    "provide actionable insights for maintenance."
                ),
                expertise=["index analysis", "performance monitoring"],
                personality="analytical; spots issues before they become problems",
                memory=True,
            ),
            TeamMemberSpec(
                role="executor",
                goal="Clean up and optimize the knowledge base",
                name="Index Maintainer",
                tool_budget=20,
                backstory=(
                    "You are a knowledge base maintainer who keeps the index healthy and "
                    "performant. You remove stale content, optimize structures, and ensure "
                    "the system runs efficiently."
                ),
                expertise=["maintenance", "cleanup", "optimization"],
                personality="efficient and careful; never breaks what's working",
            ),
        ],
        total_tool_budget=35,
    ),
}


def get_team_for_task(task_type: str) -> Optional[RAGTeamSpec]:
    """Get appropriate team specification for task type.

    Args:
        task_type: Type of task (ingest, search, query, synthesis, etc.)

    Returns:
        RAGTeamSpec or None if no matching team
    """
    mapping = {
        # Ingest tasks
        "ingest": "ingest_team",
        "index": "ingest_team",
        "add": "ingest_team",
        "import": "ingest_team",
        # Search tasks
        "search": "search_team",
        "find": "search_team",
        "retrieve": "search_team",
        "lookup": "search_team",
        # Synthesis tasks
        "query": "synthesis_team",
        "answer": "synthesis_team",
        "synthesis": "synthesis_team",
        "generate": "synthesis_team",
        # Maintenance tasks
        "maintenance": "maintenance_team",
        "cleanup": "maintenance_team",
        "optimize": "maintenance_team",
        "manage": "maintenance_team",
    }
    spec_name = mapping.get(task_type.lower())
    if spec_name:
        return RAG_TEAM_SPECS.get(spec_name)
    return None


def get_role_config(role_name: str) -> Optional[RAGRoleConfig]:
    """Get configuration for a RAG role.

    Args:
        role_name: Role name

    Returns:
        RAGRoleConfig or None
    """
    return RAG_ROLES.get(role_name.lower())


def list_team_types() -> List[str]:
    """List all available team types.

    Returns:
        List of team type names
    """
    return list(RAG_TEAM_SPECS.keys())


def list_roles() -> List[str]:
    """List all available RAG roles.

    Returns:
        List of role names
    """
    return list(RAG_ROLES.keys())


class RAGTeamSpecProvider:
    """Team specification provider for RAG vertical.

    Implements TeamSpecProviderProtocol interface for consistent
    ISP compliance across all verticals.
    """

    def get_team_specs(self) -> Dict[str, RAGTeamSpec]:
        """Get all RAG team specifications.

        Returns:
            Dictionary mapping team names to RAGTeamSpec instances
        """
        return RAG_TEAM_SPECS

    def get_team_for_task(self, task_type: str) -> Optional[RAGTeamSpec]:
        """Get appropriate team for a task type.

        Args:
            task_type: Type of task

        Returns:
            RAGTeamSpec or None if no matching team
        """
        return get_team_for_task(task_type)

    def list_team_types(self) -> List[str]:
        """List all available team types.

        Returns:
            List of team type names
        """
        return list_team_types()


__all__ = [
    # Types
    "RAGRoleConfig",
    "RAGTeamSpec",
    # Provider
    "RAGTeamSpecProvider",
    # Role configurations
    "RAG_ROLES",
    # Team specifications
    "RAG_TEAM_SPECS",
    # Helper functions
    "get_team_for_task",
    "get_role_config",
    "list_team_types",
    "list_roles",
]

logger = logging.getLogger(__name__)


def register_rag_teams() -> int:
    """Register RAG teams with global registry.

    This function is called during vertical integration by the framework's
    step handlers. Import-time auto-registration has been removed to avoid
    load-order coupling and duplicate registration.

    Returns:
        Number of teams registered.
    """
    try:
        from victor_contracts.team_schema import (
            get_runtime_team_registry as get_team_registry,
        )

        registry = get_team_registry()
        count = registry.register_from_vertical("rag", RAG_TEAM_SPECS)
        logger.debug(f"Registered {count} RAG teams via framework integration")
        return count
    except Exception as e:
        logger.warning(f"Failed to register RAG teams: {e}")
        return 0


# NOTE: Import-time auto-registration removed (SOLID compliance)
# Registration now happens during vertical integration via step_handlers.py
# This avoids load-order coupling and duplicate registration issues.
