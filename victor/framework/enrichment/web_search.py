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

"""Web search result formatting utilities.

Provides domain-agnostic formatting for web search results to be used
in prompt enrichment across any vertical.

Example:
    from victor.framework.enrichment.web_search import format_web_results

    results = [
        {"title": "Python Docs", "snippet": "...", "url": "https://..."},
    ]
    formatted = format_web_results(results, max_results=3)
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def truncate_snippet(
    snippet: str,
    max_length: int = 200,
    suffix: str = "...",
) -> str:
    """Truncate a snippet to a maximum length.

    Truncates at word boundaries when possible.

    Args:
        snippet: The text to truncate
        max_length: Maximum length (default: 200)
        suffix: Suffix to add when truncated (default: "...")

    Returns:
        Truncated snippet
    """
    if not snippet or len(snippet) <= max_length:
        return snippet

    # Try to truncate at word boundary
    truncated = snippet[:max_length]

    # Find last space
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.6:  # Don't truncate too much
        truncated = truncated[:last_space]

    return truncated.rstrip() + suffix


def format_web_results(
    results: List[Dict[str, Any]],
    max_results: int = 3,
    max_snippet_length: int = 200,
    include_urls: bool = True,
    header: Optional[str] = None,
) -> str:
    """Format web search results for prompt injection.

    Creates a structured, readable format suitable for LLM context.

    Args:
        results: List of search result dicts with title, snippet, url
        max_results: Maximum results to include (default: 3)
        max_snippet_length: Maximum snippet length (default: 200)
        include_urls: Whether to include source URLs (default: True)
        header: Optional header text (default: "Relevant web search results:")

    Returns:
        Formatted string of search results
    """
    if not results:
        return ""

    header = header or "Relevant web search results:"
    parts = [header]

    for i, result in enumerate(results[:max_results], 1):
        title = result.get("title", "Untitled")
        snippet = result.get("snippet", "")
        url = result.get("url", "")

        # Format result
        parts.append(f"\n{i}. **{title}**")

        if snippet:
            truncated = truncate_snippet(snippet, max_snippet_length)
            parts.append(f"   {truncated}")

        if include_urls and url:
            parts.append(f"   Source: {url}")

    return "\n".join(parts)


class WebSearchFormatter:
    """Formatter for web search results.

    Provides configurable formatting for web search results
    with consistent styling across verticals.

    Example:
        formatter = WebSearchFormatter(max_results=5)
        formatted = formatter.format(results)
    """

    def __init__(
        self,
        max_results: int = 3,
        max_snippet_length: int = 200,
        include_urls: bool = True,
        header: Optional[str] = None,
    ):
        """Initialize the formatter.

        Args:
            max_results: Maximum results to include (default: 3)
            max_snippet_length: Maximum snippet length (default: 200)
            include_urls: Whether to include URLs (default: True)
            header: Optional header text
        """
        self._max_results = max_results
        self._max_snippet_length = max_snippet_length
        self._include_urls = include_urls
        self._header = header

    def format(
        self,
        results: List[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> str:
        """Format search results.

        Args:
            results: List of search result dicts
            query: Optional query for context

        Returns:
            Formatted string
        """
        if query and not self._header:
            header = f"Web search results for: {query}"
        else:
            header = self._header

        return format_web_results(
            results,
            max_results=self._max_results,
            max_snippet_length=self._max_snippet_length,
            include_urls=self._include_urls,
            header=header,
        )

    def format_for_citation(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Format results for citation purposes.

        Returns structured data suitable for building citations.

        Args:
            results: List of search result dicts

        Returns:
            List of citation-ready dicts
        """
        citations = []

        for result in results[: self._max_results]:
            citations.append(
                {
                    "title": result.get("title", "Untitled"),
                    "url": result.get("url", ""),
                    "snippet": truncate_snippet(
                        result.get("snippet", ""),
                        self._max_snippet_length,
                    ),
                }
            )

        return citations

    @property
    def max_results(self) -> int:
        """Get max results setting."""
        return self._max_results

    @max_results.setter
    def max_results(self, value: int) -> None:
        """Set max results."""
        self._max_results = max(1, value)


__all__ = [
    "WebSearchFormatter",
    "format_web_results",
    "truncate_snippet",
]
