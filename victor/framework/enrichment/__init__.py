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

"""Framework enrichment infrastructure for prompt enhancement.

This module provides the core infrastructure for automatic prompt enrichment
with relevant context from various sources. It is vertical-agnostic -
each vertical implements its own EnrichmentStrategyProtocol to provide
domain-specific enrichments.

Core Components:
- PromptEnrichmentService: Main service that coordinates enrichment
- EnrichmentContext: Context object passed to strategies
- ContextEnrichment: Individual enrichment items
- EnrichedPrompt: Result containing enriched prompt and metadata
- EnrichmentStrategyProtocol: Protocol for vertical-specific strategies

Utility Components:
- Search term extraction with native Rust acceleration
- Web search result formatting
- Tool history context extraction

Usage:
    from victor.framework.enrichment import (
        PromptEnrichmentService,
        EnrichmentContext,
        EnrichmentType,
        extract_search_terms,
        format_web_results,
    )

    # Create enrichment service
    service = PromptEnrichmentService(max_tokens=2000)
    service.register_strategy("coding", CodingEnrichmentStrategy())

    # Enrich a prompt
    context = EnrichmentContext(task_type="edit")
    enriched = await service.enrich(prompt, "coding", context)

    # Use utility functions
    terms = extract_search_terms("Find authentication bugs")
    formatted = format_web_results(search_results, max_results=3)
"""

# Core enrichment infrastructure
from victor.framework.enrichment.core import (
    # Enums
    EnrichmentType,
    EnrichmentPriority,
    # Data Classes
    EnrichmentContext,
    ContextEnrichment,
    EnrichedPrompt,
    EnrichmentOutcome,
    # Protocols
    EnrichmentStrategyProtocol,
    # Services
    PromptEnrichmentService,
    EnrichmentCache,
)

# Search term extraction
from victor.framework.enrichment.search_terms import (
    SearchTermExtractor,
    extract_search_terms,
    get_search_term_patterns,
)

# Web search formatting
from victor.framework.enrichment.web_search import (
    WebSearchFormatter,
    format_web_results,
    truncate_snippet,
)

# Tool history extraction
from victor.framework.enrichment.tool_history import (
    ToolHistoryExtractor,
    extract_tool_context,
    get_relevant_tool_results,
)

# Identifier extraction (consolidated from verticals)
from victor.framework.enrichment.identifiers import (
    PATTERNS as IDENTIFIER_PATTERNS,
    COMMON_WORDS,
    extract_identifiers,
    extract_camelcase,
    extract_snake_case,
    extract_dotted_paths,
    extract_quoted_identifiers,
    IdentifierExtractor,
)

# File pattern matching (consolidated from verticals)
from victor.framework.enrichment.file_patterns import (
    DEVOPS_PATTERNS,
    DATA_PATTERNS,
    CODE_PATTERNS,
    FilePatternMatcher,
    create_combined_matcher,
)

# Keyword classification (consolidated from verticals)
from victor.framework.enrichment.keyword_classifier import (
    ANALYSIS_TYPES,
    INFRA_TYPES,
    RESEARCH_TYPES,
    KeywordClassifier,
    create_combined_classifier,
)

__all__ = [
    # Core: Enums
    "EnrichmentType",
    "EnrichmentPriority",
    # Core: Data Classes
    "EnrichmentContext",
    "ContextEnrichment",
    "EnrichedPrompt",
    "EnrichmentOutcome",
    # Core: Protocols
    "EnrichmentStrategyProtocol",
    # Core: Services
    "PromptEnrichmentService",
    "EnrichmentCache",
    # Utilities: Search terms
    "SearchTermExtractor",
    "extract_search_terms",
    "get_search_term_patterns",
    # Utilities: Web search
    "WebSearchFormatter",
    "format_web_results",
    "truncate_snippet",
    # Utilities: Tool history
    "ToolHistoryExtractor",
    "extract_tool_context",
    "get_relevant_tool_results",
    # Utilities: Identifier extraction
    "IDENTIFIER_PATTERNS",
    "COMMON_WORDS",
    "extract_identifiers",
    "extract_camelcase",
    "extract_snake_case",
    "extract_dotted_paths",
    "extract_quoted_identifiers",
    "IdentifierExtractor",
    # Utilities: File pattern matching
    "DEVOPS_PATTERNS",
    "DATA_PATTERNS",
    "CODE_PATTERNS",
    "FilePatternMatcher",
    "create_combined_matcher",
    # Utilities: Keyword classification
    "ANALYSIS_TYPES",
    "INFRA_TYPES",
    "RESEARCH_TYPES",
    "KeywordClassifier",
    "create_combined_classifier",
]
