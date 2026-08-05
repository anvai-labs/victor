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

"""Tool history context extraction utilities.

Provides domain-agnostic extraction of relevant context from tool call history
for prompt enrichment across any vertical.

Example:
    from victor.framework.enrichment.tool_history import extract_tool_context

    history = [{"tool": "web_search", "result": {...}}, ...]
    context = extract_tool_context(history, tool_names=["web_search", "web_fetch"])
"""

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def get_relevant_tool_results(
    tool_history: List[Dict[str, Any]],
    tool_names: Optional[Set[str]] = None,
    max_results: int = 10,
    min_content_length: int = 50,
) -> List[Dict[str, Any]]:
    """Extract relevant results from tool history.

    Filters tool history to find successful results with substantial content.

    Args:
        tool_history: List of tool call records
        tool_names: Set of tool names to include (None = all)
        max_results: Maximum recent results to check (default: 10)
        min_content_length: Minimum content length to include (default: 50)

    Returns:
        List of relevant result dicts with tool name and content
    """
    if not tool_history:
        return []

    relevant = []

    for call in tool_history[-max_results:]:
        tool_name = call.get("tool", "")

        # Filter by tool name if specified
        if tool_names and tool_name not in tool_names:
            continue

        # Check for successful result
        result = call.get("result", {})

        if isinstance(result, dict) and result.get("success"):
            content = result.get("content", "")

            if content and len(content) >= min_content_length:
                relevant.append(
                    {
                        "tool": tool_name,
                        "content": content,
                        "metadata": result.get("metadata", {}),
                    }
                )

    return relevant


def extract_tool_context(
    tool_history: List[Dict[str, Any]],
    tool_names: Optional[Set[str]] = None,
    max_results: int = 3,
    max_content_length: int = 300,
    header: Optional[str] = None,
) -> str:
    """Extract context from tool history for prompt enrichment.

    Creates a formatted string of relevant prior tool results.

    Args:
        tool_history: List of tool call records
        tool_names: Set of tool names to include (None = all)
        max_results: Maximum results to include (default: 3)
        max_content_length: Maximum content length per result (default: 300)
        header: Optional header text

    Returns:
        Formatted context string, or empty string if no relevant results
    """
    relevant = get_relevant_tool_results(
        tool_history,
        tool_names=tool_names,
        max_results=max_results * 3,  # Get extra for filtering
    )

    if not relevant:
        return ""

    header = header or "Prior results in this session:"
    parts = [header]

    for item in relevant[:max_results]:
        tool = item["tool"]
        content = item["content"]

        # Truncate content
        if len(content) > max_content_length:
            content = content[:max_content_length].rstrip() + "..."

        parts.append(f"\n- From {tool}:")
        parts.append(f"  {content}")

    return "\n".join(parts)


class ToolHistoryExtractor:
    """Extractor for tool history context.

    Provides configurable extraction of relevant context from
    tool call history with consistent formatting.

    Example:
        extractor = ToolHistoryExtractor(
            tool_names={"web_search", "web_fetch"},
            max_results=3,
        )
        context = extractor.extract(history)
    """

    def __init__(
        self,
        tool_names: Optional[Set[str]] = None,
        max_results: int = 3,
        max_content_length: int = 300,
        min_content_length: int = 50,
        header: Optional[str] = None,
    ):
        """Initialize the extractor.

        Args:
            tool_names: Set of tool names to include (None = all)
            max_results: Maximum results to include (default: 3)
            max_content_length: Maximum content length per result (default: 300)
            min_content_length: Minimum content length to consider (default: 50)
            header: Optional header text
        """
        self._tool_names = tool_names
        self._max_results = max_results
        self._max_content_length = max_content_length
        self._min_content_length = min_content_length
        self._header = header

    def extract(self, tool_history: List[Dict[str, Any]]) -> str:
        """Extract context from tool history.

        Args:
            tool_history: List of tool call records

        Returns:
            Formatted context string
        """
        return extract_tool_context(
            tool_history,
            tool_names=self._tool_names,
            max_results=self._max_results,
            max_content_length=self._max_content_length,
            header=self._header,
        )

    def get_relevant_results(
        self,
        tool_history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Get relevant results without formatting.

        Args:
            tool_history: List of tool call records

        Returns:
            List of relevant result dicts
        """
        return get_relevant_tool_results(
            tool_history,
            tool_names=self._tool_names,
            max_results=self._max_results * 3,
            min_content_length=self._min_content_length,
        )

    def add_tool_name(self, tool_name: str) -> None:
        """Add a tool name to the filter.

        Args:
            tool_name: Tool name to add
        """
        if self._tool_names is None:
            self._tool_names = set()
        self._tool_names.add(tool_name)

    def remove_tool_name(self, tool_name: str) -> None:
        """Remove a tool name from the filter.

        Args:
            tool_name: Tool name to remove
        """
        if self._tool_names:
            self._tool_names.discard(tool_name)

    @property
    def tool_names(self) -> Optional[Set[str]]:
        """Get tool name filter."""
        return self._tool_names

    @property
    def max_results(self) -> int:
        """Get max results setting."""
        return self._max_results

    @max_results.setter
    def max_results(self, value: int) -> None:
        """Set max results."""
        self._max_results = max(1, value)


__all__ = [
    "ToolHistoryExtractor",
    "extract_tool_context",
    "get_relevant_tool_results",
]
