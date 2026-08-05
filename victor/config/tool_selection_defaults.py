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

"""Centralized defaults for tool selection strategies.

This module consolidates hardcoded values from tool selection components
into a single source of truth. This makes configuration easier to find,
modify, and document.

Replaces hardcoded values scattered across:
- SemanticToolSelector
- HybridToolSelector
- KeywordToolSelector
- ToolPipeline

Usage:
    from victor.config.tool_selection_defaults import (
        SemanticSelectorDefaults,
        HybridSelectorDefaults,
        ToolPipelineDefaults,
        FallbackTools,
        IdempotentTools,
    )

    # Use in code
    if len(tools) < SemanticSelectorDefaults.MIN_TOOLS_THRESHOLD:
        tools.extend(FallbackTools.COMMON_FALLBACK_TOOLS)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Final, FrozenSet, List, Set

# =============================================================================
# Enums for Tool Categories
# =============================================================================


class ToolCategory(str, Enum):
    """Canonical tool category names for classification."""

    FILE_OPS = "file_ops"
    GIT_OPS = "git_ops"
    CODE_ANALYSIS = "code_analysis"
    SEARCH = "search"
    EDIT = "edit"


# =============================================================================
# Semantic Selector Defaults
# =============================================================================


@dataclass(frozen=True)
class SemanticSelectorDefaults:
    """Default values for semantic tool selection.

    These control the behavior of SemanticToolSelector including
    similarity thresholds, cost awareness, and result limits.
    """

    # Minimum number of tools to return before adding fallbacks
    MIN_TOOLS_THRESHOLD: Final[int] = 2

    # Cost penalty factor for expensive tools (0.0-1.0)
    # Higher values penalize expensive tools more
    COST_PENALTY_FACTOR: Final[float] = 0.05

    # Enable cost-aware selection by default
    COST_AWARE_SELECTION: Final[bool] = True

    # Minimum similarity score for semantic matching (0.0-1.0)
    SIMILARITY_THRESHOLD: Final[float] = 0.15

    # Maximum tools to consider in initial semantic search
    MAX_SEMANTIC_CANDIDATES: Final[int] = 20

    # Default number of tools to return
    DEFAULT_RESULT_LIMIT: Final[int] = 10

    # Score assigned to mandatory tools to ensure they rank high
    MANDATORY_TOOL_SCORE: Final[float] = 0.9

    # Score assigned to fallback tools when added
    FALLBACK_TOOL_SCORE: Final[float] = 0.5

    # Maximum boost from usage/success/recency/context factors
    MAX_USAGE_BOOST: Final[float] = 0.05
    MAX_SUCCESS_BOOST: Final[float] = 0.05
    MAX_RECENCY_BOOST: Final[float] = 0.05
    MAX_CONTEXT_BOOST: Final[float] = 0.05
    MAX_TOTAL_BOOST: Final[float] = 0.2

    # Task-type alignment boost
    TASK_TYPE_BOOST: Final[float] = 0.05


# =============================================================================
# Hybrid Selector Defaults
# =============================================================================


@dataclass(frozen=True)
class HybridSelectorDefaults:
    """Default values for hybrid tool selection.

    These control the blending of semantic and keyword selection
    strategies in HybridToolSelector.
    """

    # Weight for semantic selection results (0.0-1.0)
    SEMANTIC_WEIGHT: Final[float] = 0.7

    # Weight for keyword selection results (0.0-1.0)
    KEYWORD_WEIGHT: Final[float] = 0.3

    # Minimum tools to get from semantic selector
    MIN_SEMANTIC_TOOLS: Final[int] = 3

    # Minimum tools to get from keyword selector
    MIN_KEYWORD_TOOLS: Final[int] = 2

    # Maximum total tools in final result
    MAX_TOTAL_TOOLS: Final[int] = 15

    # Weight boost for RL-recommended tools (0.0-1.0)
    RL_BOOST_WEIGHT: Final[float] = 0.15

    # RL exploration mode: top-k tools to shuffle for diversity
    EXPLORATION_TOP_K: Final[int] = 3

    # Default Q-value for unknown tools in RL ranking
    DEFAULT_Q_VALUE: Final[float] = 0.5

    # Default confidence for unknown tools in RL ranking
    DEFAULT_CONFIDENCE: Final[float] = 0.3

    # Default grounding scores for outcome recording
    DEFAULT_GROUNDING_SCORE_SUCCESS: Final[float] = 0.5
    DEFAULT_GROUNDING_SCORE_FAILURE: Final[float] = 0.2
    DEFAULT_EFFICIENCY_SCORE: Final[float] = 0.5


# =============================================================================
# Tool Pipeline Defaults
# =============================================================================


@dataclass(frozen=True)
class ToolPipelineDefaults:
    """Default values for tool pipeline execution.

    These control tool execution budgets, concurrency, and caching.
    Note: Some of these duplicate values in victor/config/defaults.py
    for backward compatibility. The canonical source is defaults.py.
    """

    # Maximum tool calls per session
    TOOL_BUDGET: Final[int] = 25

    # Maximum concurrent tool executions
    MAX_CONCURRENT_TOOLS: Final[int] = 5

    # Batch size for parallel tool execution
    PARALLEL_BATCH_SIZE: Final[int] = 10

    # Timeout for individual tool execution (seconds)
    PARALLEL_TIMEOUT_PER_TOOL: Final[float] = 60.0

    # Maximum entries in idempotent tool result cache
    IDEMPOTENT_CACHE_MAX_SIZE: Final[int] = 50

    # TTL for cached tool results (seconds)
    IDEMPOTENT_CACHE_TTL: Final[float] = 300.0


# =============================================================================
# Fallback Tools Configuration
# =============================================================================


@dataclass(frozen=True)
class FallbackTools:
    """Fallback tool sets for when selection returns too few results.

    These are added to selection results when the initial selection
    doesn't return enough tools.
    """

    # Fallback tools for conceptual/architectural queries
    # Used when query appears to be about design patterns, inheritance, etc.
    CONCEPTUAL_FALLBACK_TOOLS: Final[List[str]] = (
        "search",
        "read",
    )

    # General fallback tools for any query type
    # Used when selection returns fewer than MIN_TOOLS_THRESHOLD tools
    COMMON_FALLBACK_TOOLS: Final[List[str]] = (
        "read",
        "grep",
        "search",
        "ls",
        "shell",
        "write",
        "edit",
    )


# =============================================================================
# Idempotent Tools Configuration
# =============================================================================


class IdempotentTools:
    """Tools whose results can be safely cached.

    Idempotent tools return the same result for the same arguments,
    so their results can be cached to avoid redundant execution.
    """

    # Frozen set of tool names that are idempotent
    IDEMPOTENT_TOOLS: Final[FrozenSet[str]] = frozenset(
        {
            "read",
            "grep",
            "search",
            "ls",
            "glob",
            "find",
            "file_info",
            "git_status",
            "git_diff",
            "git_log",
            "get_diagnostics",
            "get_symbols",
            "get_references",
        }
    )

    @classmethod
    def is_idempotent(cls, tool_name: str) -> bool:
        """Check if a tool is idempotent.

        Args:
            tool_name: Name of the tool to check

        Returns:
            True if tool results can be cached
        """
        return tool_name in cls.IDEMPOTENT_TOOLS


# =============================================================================
# Query Pattern Configuration
# =============================================================================


@dataclass(frozen=True)
class QueryPatterns:
    """Patterns for classifying query types.

    Used by semantic selector to identify conceptual vs operational queries.
    """

    # Patterns indicating conceptual/architectural queries
    CONCEPTUAL_PATTERNS: Final[List[str]] = (
        "inherit",
        "implement",
        "extend",
        "subclass",
        "pattern",
        "architecture",
        "design",
        "interface",
        "abstract",
        "protocol",
        "relationship",
        "hierarchy",
        "structure",
        "organize",
        "dependency",
        "module",
        "component",
        "layer",
        "flow",
        "diagram",
    )

    @classmethod
    def is_conceptual_query(cls, query: str) -> bool:
        """Check if a query appears to be conceptual/architectural.

        Args:
            query: The query string to check

        Returns:
            True if query contains conceptual patterns
        """
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in cls.CONCEPTUAL_PATTERNS)


# =============================================================================
# Category Aliases
# =============================================================================


@dataclass(frozen=True)
class CategoryAliases:
    """Aliases for tool categories.

    Maps common category names to their canonical forms for
    consistent tool filtering and selection.
    """

    # File operations
    FILE_OPS: Final[Set[str]] = frozenset({"file", "files", "file_ops", "filesystem"})

    # Git operations
    GIT_OPS: Final[Set[str]] = frozenset({"git", "version_control", "vcs", "scm"})

    # Code analysis
    CODE_ANALYSIS: Final[Set[str]] = frozenset({"code", "analysis", "ast", "lint", "parse"})

    # Search operations
    SEARCH: Final[Set[str]] = frozenset({"search", "find", "grep", "locate", "query"})

    # Edit operations
    EDIT: Final[Set[str]] = frozenset({"edit", "write", "modify", "change", "update"})

    @classmethod
    def get_canonical_category(cls, category: str) -> ToolCategory:
        """Get the canonical category name for an alias.

        Args:
            category: Category name or alias

        Returns:
            ToolCategory enum value
        """
        category_lower = category.lower()

        if category_lower in cls.FILE_OPS:
            return ToolCategory.FILE_OPS
        if category_lower in cls.GIT_OPS:
            return ToolCategory.GIT_OPS
        if category_lower in cls.CODE_ANALYSIS:
            return ToolCategory.CODE_ANALYSIS
        if category_lower in cls.SEARCH:
            return ToolCategory.SEARCH
        if category_lower in cls.EDIT:
            return ToolCategory.EDIT

        # Return as-is (could add more categories dynamically)
        return ToolCategory(category_lower)  # type: ignore


__all__ = [
    "SemanticSelectorDefaults",
    "HybridSelectorDefaults",
    "ToolPipelineDefaults",
    "FallbackTools",
    "IdempotentTools",
    "QueryPatterns",
    "CategoryAliases",
]
