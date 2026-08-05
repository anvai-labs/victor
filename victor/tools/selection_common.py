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

"""Stateful utility functions for tool selection.

This module contains utility functions with dependencies on ToolRegistry and other
stateful components. Extracted from tool_selection.py as part of HIGH-002: Unified
Tool Selection Architecture - Release 1, Phase 0.

These functions interact with the tool registry and metadata systems to discover
tools dynamically based on decorator metadata.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional, Set

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Fallback critical tools for cases where registry is unavailable.
# Critical tools are detected via priority=Priority.CRITICAL in @tool decorator.
FALLBACK_CRITICAL_TOOLS: Set[str] = {
    "read",  # read_file → read
    "write",  # write_file → write
    "ls",  # list_directory → ls
    "shell",  # execute_bash → shell
    "edit",  # edit_files → edit
    "search",  # code_search → search (always needed for code exploration)
}

# Fallback category keywords for goal inference.
# NOTE: This is a FALLBACK when registry.detect_categories_from_text() is unavailable.
# The registry-based approach dynamically builds categories from @tool(keywords=[...])
# decorators and is preferred. This static list is only used when registry is None.
FALLBACK_CATEGORY_KEYWORDS: Dict[str, Set[str]] = {
    "security": {
        "security",
        "vulnerability",
        "scan",
        "audit",
        "cve",
        "exploit",
        "owasp",
    },
    "metrics": {
        "metrics",
        "complexity",
        "coverage",
        "analyze",
        "statistics",
        "cyclomatic",
    },
    "testing": {"test", "unittest", "pytest", "spec", "coverage", "mock"},
    "git": {"git", "commit", "branch", "merge", "push", "pull", "rebase"},
    "docker": {"docker", "container", "dockerfile", "compose", "image"},
    "web": {"web", "http", "api", "fetch", "url", "search"},
}

# Alias for backward compatibility
CATEGORY_KEYWORDS = FALLBACK_CATEGORY_KEYWORDS


# ============================================================================
# Tool Discovery Functions
# ============================================================================


def get_critical_tools(registry: Optional["ToolRegistry"] = None) -> Set[str]:
    """Dynamically discover critical tools from registry using priority-based detection.

    Critical tools are those with priority=Priority.CRITICAL in their @tool decorator.
    These tools are always available for selection regardless of task type.
    Falls back to hardcoded list if registry is unavailable.

    Args:
        registry: Optional tool registry to query

    Returns:
        Set of canonical tool names that are critical priority tools
    """
    if registry is None:
        return FALLBACK_CRITICAL_TOOLS.copy()

    critical_tools: Set[str] = set()
    for tool in registry.list_tools(only_enabled=True):
        # Use is_critical property which checks priority=Priority.CRITICAL
        if hasattr(tool, "is_critical") and tool.is_critical:
            critical_tools.add(tool.name)

    # Fallback if no critical tools found (shouldn't happen with proper setup)
    if not critical_tools:
        logger.warning(
            "No tools with priority=Priority.CRITICAL found in registry. "
            "Using fallback critical tools. Check tool decorator definitions."
        )
        return FALLBACK_CRITICAL_TOOLS.copy()

    return critical_tools


def get_tools_by_category(
    registry: Optional["ToolRegistry"] = None, category: str = ""
) -> Set[str]:
    """Dynamically discover tools in a specific category from registry.

    Tools declare their category via @tool(category="...") decorator.

    Args:
        registry: Tool registry to query
        category: Category name to filter by (e.g., "git", "web", "testing")

    Returns:
        Set of tool names in the specified category
    """
    if registry is None or not category:
        return set()

    tools_in_category: Set[str] = set()
    for tool in registry.list_tools(only_enabled=True):
        if hasattr(tool, "category") and tool.category == category:
            tools_in_category.add(tool.name)

    return tools_in_category


def get_web_tools(registry: Optional["ToolRegistry"] = None) -> Set[str]:
    """Get web-related tools from registry.

    Web tools are those with category='web'. Tools must declare their category
    via @tool(category="web") decorator.

    Args:
        registry: Tool registry to query (required for proper operation)

    Returns:
        Set of canonical web tool names.
        Returns fallback set if registry is None.
    """
    if registry is None:
        # Fallback to known web tools
        return {"web_search", "web_fetch"}

    return get_tools_by_category(registry, category="web")


def get_all_categories(registry: Optional["ToolRegistry"] = None) -> Set[str]:
    """Get all unique categories from tools in registry.

    Args:
        registry: Tool registry to query

    Returns:
        Set of unique category names
    """
    if registry is None:
        return set()

    categories: Set[str] = set()
    for tool in registry.list_tools(only_enabled=True):
        if hasattr(tool, "category") and tool.category:
            categories.add(tool.category)

    return categories


def get_category_to_tools_map(
    registry: Optional["ToolRegistry"] = None,
) -> Dict[str, Set[str]]:
    """Build a mapping from categories to tool names.

    Dynamically discovers categories from tool metadata. Tools must declare
    their category via @tool(category="...") decorator.

    Args:
        registry: Tool registry to query (required for proper operation)

    Returns:
        Dict mapping category name to set of tool names.
        Empty dict if registry is None.

    Raises:
        ValueError: If registry is provided but no tools have categories
    """
    if registry is None:
        logger.warning(
            "get_category_to_tools_map called without registry. "
            "Pass a ToolRegistry for proper category discovery."
        )
        return {}

    category_map: Dict[str, Set[str]] = {}
    for tool in registry.list_tools(only_enabled=True):
        if hasattr(tool, "category") and tool.category:
            category = tool.category
            if category not in category_map:
                category_map[category] = set()
            category_map[category].add(tool.name)

    if not category_map:
        logger.warning(
            "No tools with categories found in registry. "
            "Add category metadata to @tool decorators."
        )

    return category_map


def detect_categories_from_message(message: str) -> Set[str]:
    """Detect relevant tool categories from keywords in a message.

    Uses registry-based detection with decorator-driven keywords when available,
    falling back to static CATEGORY_KEYWORDS if registry is empty.

    Args:
        message: User message text to analyze

    Returns:
        Set of category names that match keywords in the message

    Example:
        >>> detect_categories_from_message("run a security scan")
        {'security'}
        >>> detect_categories_from_message("analyze code complexity and metrics")
        {'metrics'}
    """
    # Try registry-based detection first (decorator-driven)
    try:
        from victor.tools.metadata_registry import detect_categories_from_text

        registry_detected = detect_categories_from_text(message)
        if registry_detected:
            logger.debug(f"Registry detected categories: {registry_detected}")
            return registry_detected
    except ImportError as e:
        logger.debug(f"Registry module not available for category detection: {e}")
    except Exception as e:
        logger.warning(
            "Registry category detection failed",
            exc_info=e,
            extra={"message_length": len(message)},
        )

    # Fallback to static keywords
    message_lower = message.lower()
    detected: Set[str] = set()

    for category, keywords in FALLBACK_CATEGORY_KEYWORDS.items():
        if any(kw in message_lower for kw in keywords):
            detected.add(category)

    if detected:
        logger.debug(f"Fallback detected categories: {detected}")

    return detected


def get_tools_for_categories(
    categories: Set[str], registry: Optional["ToolRegistry"] = None
) -> Set[str]:
    """Get tool names for a set of categories.

    Aggregates tools from multiple categories into a single set.
    Uses the ToolMetadataRegistry for lookup if registry is not provided.

    Args:
        categories: Set of category names (e.g., {"security", "metrics"})
        registry: Optional tool registry to query

    Returns:
        Set of tool names belonging to any of the specified categories
    """
    tools: Set[str] = set()

    # Try registry-based lookup first
    if registry is not None:
        for category in categories:
            category_tools = get_tools_by_category(registry, category)
            tools.update(category_tools)
    else:
        # Fall back to metadata registry
        try:
            from victor.tools.metadata_registry import get_global_registry

            meta_registry = get_global_registry()
            for category in categories:
                category_tools = meta_registry.get_tools_by_category(category)
                tools.update(category_tools)
        except ImportError as e:
            logger.debug(f"Metadata registry module not available: {e}")
        except Exception as e:
            logger.warning(
                "Failed to get tools for categories via registry",
                exc_info=e,
                extra={"categories": list(categories)},
            )

    return tools


def get_tools_from_message(message: str) -> Set[str]:
    """Get tool names from message using metadata registry keyword matching.

    Uses the global ToolMetadataRegistry to match keywords defined in @tool decorators.

    Args:
        message: User message text

    Returns:
        Set of tool names that match keywords in the message
    """
    try:
        from victor.tools.metadata_registry import get_global_registry

        registry = get_global_registry()
        return registry.get_tools_matching_text(message)
    except ImportError as e:
        logger.debug(f"Metadata registry module not available: {e}")
        return set()
    except Exception as e:
        logger.warning(
            "Failed to get tools from message via registry",
            exc_info=e,
            extra={"message_length": len(message)},
        )
        return set()


def select_tools_by_keywords(
    message: str,
    all_tool_names: Set[str],
    registry: Optional["ToolRegistry"] = None,
    is_small: bool = False,
    max_tools_for_small: int = 10,
) -> Set[str]:
    """Select tools using keyword matching via ToolMetadataRegistry.

    Uses keywords defined in @tool decorators for tool selection.
    This is the recommended approach - tools define their own keywords.

    Args:
        message: User message
        all_tool_names: Set of all available tool names
        registry: Optional tool registry for critical tools discovery
        is_small: Whether this is a small model
        max_tools_for_small: Max tools for small models

    Returns:
        Set of selected tool names
    """
    # Start with critical tools (dynamic discovery)
    core_tools = get_critical_tools(registry)
    selected = core_tools.copy()

    # Add tools matching keywords from @tool decorators
    keyword_matches = get_tools_from_message(message)
    selected.update(keyword_matches)

    # Filter to only available tools
    selected = selected.intersection(all_tool_names)

    # Limit for small models
    if is_small and len(selected) > max_tools_for_small:
        # Keep core tools, limit others
        core_in_selected = selected.intersection(core_tools)
        others = list(selected - core_tools)
        max_others = max(0, max_tools_for_small - len(core_in_selected))
        selected = core_in_selected.union(set(others[:max_others]))

    return selected
