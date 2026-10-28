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


RESEARCH_TOOLS = [
    "read",
    "write",
    "grep",
    "ls",
    "web_search",
    "web_fetch",
    "web_scrape",
    "web_crawl",
    "arxiv_search",
    "scholar_search",
    "pubmed_search",
    "pdf_extract",
    "document_parse",
    "citation_extract",
    "citation_format",
    "bibliography_generate",
    "literature_review",
    "fact_check",
    "competitive_analysis",
    "summary_generate",
    "key_points_extract",
]

RESEARCH_STAGE_TOOLS = {
    "discover": ["web_search", "arxiv_search", "scholar_search"],
    "collect": ["web_fetch", "web_scrape", "pdf_extract"],
    "analyze": ["read", "citation_extract", "fact_check"],
    "synthesize": ["literature_review", "summary_generate", "write"],
    "verify": ["fact_check", "web_fetch", "read"],
}

RESEARCH_DANGEROUS_PATTERNS = [
    {
        "pattern": "web_scrape --infinite",
        "description": "Infinite scraping loop",
    },
    {
        "pattern": "web_crawl --depth 100",
        "description": "Excessive crawl depth",
    },
    {
        "pattern": "write --force /etc/",
        "description": "Writing to system directories",
    },
]

RESEARCH_TOOL_RESTRICTIONS = {
    "web_scrape": ["--infinite", "--no-delay"],
    "web_crawl": ["--depth", "max_depth=10"],
}

RESEARCH_PROMPT_SECTIONS = {
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

RESEARCH_TASK_TYPE_HINTS = {
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

RESEARCH_WORKFLOWS = {
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

ResearchToolProvider = make_tool_provider(
    "ResearchToolProvider", RESEARCH_TOOLS, "Tool provider for the research vertical."
)
ResearchToolSelectionStrategy = make_stage_tool_strategy(
    "ResearchToolSelectionStrategy",
    RESEARCH_STAGE_TOOLS,
    ["web_search", "read", "write"],
    "Stage-aware tool selection for research tasks.",
)
ResearchSafetyProvider = make_pattern_safety_provider(
    "ResearchSafetyProvider",
    RESEARCH_DANGEROUS_PATTERNS,
    RESEARCH_TOOL_RESTRICTIONS,
    doc="Safety provider for research operations.",
)
ResearchPromptProvider = make_prompt_provider(
    "ResearchPromptProvider",
    RESEARCH_PROMPT_SECTIONS,
    RESEARCH_TASK_TYPE_HINTS,
    "Prompt provider for research tasks.",
)
ResearchWorkflowProvider = make_workflow_provider(
    "ResearchWorkflowProvider", RESEARCH_WORKFLOWS, "Workflow provider for the research vertical."
)


__all__ = [
    "ResearchToolProvider",
    "ResearchToolSelectionStrategy",
    "ResearchSafetyProvider",
    "ResearchPromptProvider",
    "ResearchWorkflowProvider",
]
