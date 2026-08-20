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

"""Victor extension protocol implementations for victor-research.

This module provides protocol implementations that can be discovered via
the Victor extension entry point system, enabling the research vertical to
register capabilities with the framework without direct dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# Import victor-contracts protocols (NO runtime dependency on victor-ai!)
try:
    from victor_contracts.verticals.protocols import (
        PromptProvider,
        SafetyProvider,
        ToolProvider,
        ToolSelectionStrategy,
        WorkflowProvider,
    )
except ImportError:
    from typing import Protocol

    class ToolProvider(Protocol):
        def get_tools(self) -> List[str]: ...

    class ToolSelectionStrategy(Protocol):
        def get_tools_for_stage(self, stage: str, task_type: str) -> List[str]: ...

    class SafetyProvider(Protocol):
        def get_extensions(self) -> List[Any]: ...
        def get_bash_patterns(self) -> List[Any]: ...
        def get_file_patterns(self) -> List[Any]: ...
        def get_tool_restrictions(self) -> Dict[str, List[str]]: ...

    class PromptProvider(Protocol):
        def get_system_prompt_sections(self) -> Dict[str, str]: ...
        def get_task_type_hints(self) -> Dict[str, Any]: ...
        def get_prompt_contributors(self) -> List[Any]: ...

    class WorkflowProvider(Protocol):
        def get_workflows(self) -> Dict[str, Any]: ...
        def get_workflow(self, name: str) -> Optional[Any]: ...
        def list_workflows(self) -> List[str]: ...


logger = logging.getLogger(__name__)


# =============================================================================
# Tool Provider
# =============================================================================


class ResearchToolProvider(ToolProvider):
    """Tool provider for Research vertical.

    Provides the list of tools available to the Research assistant.
    """

    def get_tools(self) -> List[str]:
        """Return list of tool names for Research vertical."""
        return [
            # Core filesystem tools
            "read",
            "write",
            "grep",
            "ls",
            # Web search tools
            "web_search",
            "web_fetch",
            "web_scrape",
            "web_crawl",
            # Academic research tools
            "arxiv_search",
            "scholar_search",
            "pubmed_search",
            # Document processing
            "pdf_extract",
            "document_parse",
            # Citation tools
            "citation_extract",
            "citation_format",
            "bibliography_generate",
            # Knowledge synthesis
            "literature_review",
            "fact_check",
            "competitive_analysis",
            # Summary tools
            "summary_generate",
            "key_points_extract",
        ]


class ResearchToolSelectionStrategy(ToolSelectionStrategy):
    """Stage-aware tool selection for Research tasks."""

    def get_tools_for_stage(self, stage: str, task_type: str) -> List[str]:
        """Return optimized tools for given stage and task type."""
        stage_tools: Dict[str, List[str]] = {
            "discover": ["web_search", "arxiv_search", "scholar_search"],
            "collect": ["web_fetch", "web_scrape", "pdf_extract"],
            "analyze": ["read", "citation_extract", "fact_check"],
            "synthesize": ["literature_review", "summary_generate", "write"],
            "verify": ["fact_check", "web_fetch", "read"],
        }

        return stage_tools.get(stage, ["web_search", "read", "write"])


# =============================================================================
# Safety Provider
# =============================================================================


class ResearchSafetyProvider(SafetyProvider):
    """Safety provider for Research vertical.

    Provides research-specific safety patterns for web operations.
    """

    def __init__(self):
        self._dangerous_patterns = [
            # Web scraping dangerous patterns
            {
                "pattern": "web_scrape --infinite",
                "description": "Infinite scraping loop",
            },
            {
                "pattern": "web_crawl --depth 100",
                "description": "Excessive crawl depth",
            },
            # File operations
            {
                "pattern": "write --force /etc/",
                "description": "Writing to system directories",
            },
        ]

    def get_extensions(self) -> List[Any]:
        """Return safety extensions for Research."""
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
            "web_scrape": ["--infinite", "--no-delay"],
            "web_crawl": ["--depth", "max_depth=10"],
        }


# =============================================================================
# Prompt Provider
# =============================================================================


class ResearchPromptProvider(PromptProvider):
    """Prompt provider for Research vertical.

    Provides system prompt sections for research tasks.
    """

    def get_system_prompt_sections(self) -> Dict[str, str]:
        """Return system prompt sections."""
        return {
            "role": (
                "You are a Research assistant specializing in information gathering, "
                "literature review, and knowledge synthesis."
            ),
            "expertise": (
                "You have expertise in web research, academic databases, "
                "fact-checking, and competitive analysis."
            ),
            "methodology": (
                "Use systematic research approaches: define questions, discover "
                "sources, evaluate credibility, synthesize findings."
            ),
            "citations": (
                "Always cite sources when presenting information. Use standard "
                "citation formats (APA, MLA, Chicago)."
            ),
            "verification": (
                "Verify claims by cross-referencing multiple sources. Distinguish "
                "between facts and opinions."
            ),
        }

    def get_task_type_hints(self) -> Dict[str, Any]:
        """Return task type hints for Research."""
        return {
            "search": {
                "hint": "[SEARCH] Search multiple sources, collect relevant information.",
                "tool_budget": 8,
            },
            "literature_review": {
                "hint": "[LITERATURE] Review academic literature, synthesize findings.",
                "tool_budget": 15,
            },
            "fact_check": {
                "hint": "[VERIFY] Verify claims across multiple sources.",
                "tool_budget": 10,
            },
            "competitive": {
                "hint": "[COMPETITIVE] Analyze competitive landscape, gather intelligence.",
                "tool_budget": 12,
            },
        }

    def get_prompt_contributors(self) -> List[Any]:
        """Return prompt contributors for Research."""
        return []


# =============================================================================
# Workflow Provider
# =============================================================================


class ResearchWorkflowProvider(WorkflowProvider):
    """Workflow provider for Research vertical.

    Provides research-specific workflow definitions.
    """

    def get_workflows(self) -> Dict[str, Any]:
        """Return workflow specifications."""
        return {
            "literature_review": {
                "name": "Literature Review",
                "description": "Conduct a comprehensive literature review on a topic",
                "stages": ["discover", "collect", "analyze", "synthesize"],
            },
            "fact_check": {
                "name": "Fact Check",
                "description": "Verify claims across multiple sources",
                "stages": ["discover", "collect", "verify"],
            },
            "competitive_analysis": {
                "name": "Competitive Analysis",
                "description": "Analyze competitive landscape",
                "stages": ["discover", "collect", "analyze", "synthesize"],
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
    "ResearchToolProvider",
    "ResearchToolSelectionStrategy",
    "ResearchSafetyProvider",
    "ResearchPromptProvider",
    "ResearchWorkflowProvider",
]
