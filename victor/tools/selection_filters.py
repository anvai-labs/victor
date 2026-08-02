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

"""Pure utility functions for tool selection filtering.

This module contains stateless filtering functions extracted from tool_selection.py
as part of HIGH-002: Unified Tool Selection Architecture - Release 1, Phase 0.

These functions have no side effects and minimal dependencies, making them suitable
for use across different tool selector implementations (Keyword, Semantic, Hybrid).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Set

if TYPE_CHECKING:
    from victor.providers.base import ToolDefinition

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Web-related keywords for explicit web tool inclusion
WEB_KEYWORDS: List[str] = ["search", "web", "online", "lookup", "http", "https"]

# Small model indicators (for limiting tools)
SMALL_MODEL_INDICATORS: List[str] = [":0.5b", ":1.5b", ":3b"]


# ============================================================================
# Pure Filtering Functions
# ============================================================================


def is_small_model(model_name: str, provider_name: str = "") -> bool:
    """Check if a model is considered small (for tool limiting).

    Args:
        model_name: Model name/identifier
        provider_name: Optional provider name

    Returns:
        True if model is small
    """
    model_lower = model_name.lower()

    # Only apply small model logic for Ollama currently
    if provider_name.lower() == "ollama":
        return any(indicator in model_lower for indicator in SMALL_MODEL_INDICATORS)

    return False


def needs_web_tools(message: str) -> bool:
    """Check if message suggests web tools are needed.

    Args:
        message: User message text

    Returns:
        True if web tools should be included
    """
    message_lower = message.lower()
    return any(kw in message_lower for kw in WEB_KEYWORDS)


def deduplicate_tools(tools: List["ToolDefinition"]) -> List["ToolDefinition"]:
    """Deduplicate tools by name, preserving order.

    Extracted from tool_selection.py lines 1272-1275.

    Args:
        tools: List of tool definitions (may contain duplicates)

    Returns:
        Deduplicated list of tools
    """
    dedup: Dict[str, "ToolDefinition"] = {}
    for t in tools:
        dedup[t.name] = t
    return list(dedup.values())


def blend_tool_results(
    semantic_tools: List["ToolDefinition"],
    keyword_tools: List["ToolDefinition"],
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> List["ToolDefinition"]:
    """Blend semantic and keyword tool selection results.

    Combines results from both selection strategies, avoiding duplicates.
    Semantic tools come first (higher priority), then keyword tools.

    Extracted from tool_selection.py lines 1250-1254.

    Args:
        semantic_tools: Tools selected by semantic similarity
        keyword_tools: Tools selected by keyword matching
        semantic_weight: Weight for semantic results (currently unused, for future scoring)
        keyword_weight: Weight for keyword results (currently unused, for future scoring)

    Returns:
        Combined list of tools with duplicates removed
    """
    # Start with semantic tools
    blended = semantic_tools.copy()

    # Add keyword tools that aren't already present
    existing = {t.name for t in semantic_tools}
    blended.extend([t for t in keyword_tools if t.name not in existing])

    return blended


def cap_tools_to_max(tools: List["ToolDefinition"], max_tools: int) -> List["ToolDefinition"]:
    """Cap tool list to maximum length.

    Extracted from tool_selection.py lines 1298-1299.

    Args:
        tools: List of tools
        max_tools: Maximum number of tools to return

    Returns:
        Tools capped to max_tools
    """
    if len(tools) > max_tools:
        return tools[:max_tools]
    return tools


def get_tool_names_set(tools: List["ToolDefinition"]) -> Set[str]:
    """Extract tool names as a set.

    Helper for checking tool membership.

    Args:
        tools: List of tool definitions

    Returns:
        Set of tool names
    """
    return {t.name for t in tools}
