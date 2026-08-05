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

"""Coding-vertical query expansion for semantic code search.

This module provides coding-specific query expansion capabilities.
The generic QueryExpander base class is in victor/framework/search/query_expansion.py.

This vertical-specific module:
- Defines SEMANTIC_QUERY_EXPANSIONS for coding domain concepts
- Provides backward-compatible QueryExpander class wrapping framework base
- Maintains existing API (get_query_expander, expand_query functions)
"""

import logging
from typing import Dict, List, Optional, Set

from victor_contracts.search_runtime import (
    QueryExpander as FrameworkQueryExpander,
    QueryExpansionConfig,
)

logger = logging.getLogger(__name__)


# Conceptual query expansions for semantic code search
# Maps conceptual terms to implementation-specific variations
SEMANTIC_QUERY_EXPANSIONS: Dict[str, List[str]] = {
    # Tool/Plugin Architecture
    "tool registration": [
        "register tool",
        "@tool decorator",
        "tool registry",
        "register_tool",
        "ToolRegistry",
        "tool.register",
        "add_tool",
    ],
    "plugin registration": [
        "register plugin",
        "plugin registry",
        "register_plugin",
        "PluginRegistry",
        "load_plugin",
    ],
    # Provider/Model Architecture
    "provider": [
        "LLM provider",
        "model provider",
        "BaseProvider",
        "provider class",
        "provider implementation",
    ],
    "provider implementation": [
        "provider class",
        "BaseProvider",
        "chat method",
        "stream_chat",
        "provider adapter",
    ],
    # Error Handling
    "error handling": [
        "exception",
        "try catch",
        "try except",
        "error recovery",
        "exception handling",
        "raise",
        "except",
    ],
    "exception": [
        "error handling",
        "try except",
        "exception class",
        "raise exception",
        "catch exception",
    ],
    # Configuration
    "configuration": [
        "config",
        "settings",
        "Settings class",
        "ProfileConfig",
        "configuration file",
        "config.yaml",
    ],
    "settings": [
        "configuration",
        "config",
        "Settings",
        "environment variables",
        "profiles",
    ],
    # Testing
    "test": [
        "unit test",
        "integration test",
        "pytest",
        "test case",
        "test fixture",
        "test function",
    ],
    "testing": [
        "test",
        "unit test",
        "pytest",
        "test suite",
        "test coverage",
    ],
    # Logging/Monitoring
    "logging": [
        "logger",
        "log",
        "logging.getLogger",
        "debug",
        "info",
        "warning",
        "error",
    ],
    "monitoring": [
        "metrics",
        "telemetry",
        "usage analytics",
        "performance tracking",
        "instrumentation",
    ],
    # Authentication/Authorization
    "authentication": [
        "auth",
        "login",
        "credentials",
        "api key",
        "token",
        "authenticate",
    ],
    "authorization": [
        "permissions",
        "access control",
        "RBAC",
        "authorize",
        "allowed",
    ],
    # Validation
    "validation": [
        "validate",
        "validator",
        "validation error",
        "check",
        "verify",
        "pydantic",
    ],
    # Caching
    "caching": [
        "cache",
        "cached",
        "cache_dir",
        "TTL",
        "cache hit",
        "cache miss",
    ],
    "cache": [
        "caching",
        "cached",
        "cache_file",
        "cache_key",
        "LRU cache",
    ],
    # Database/Storage
    "database": [
        "db",
        "storage",
        "store",
        "persist",
        "SQLite",
        "DuckDB",
    ],
    "storage": [
        "database",
        "store",
        "persist",
        "save",
        "load",
    ],
    # API/Endpoints
    "api": [
        "endpoint",
        "REST API",
        "HTTP API",
        "API route",
        "API handler",
    ],
    "endpoint": [
        "api",
        "route",
        "handler",
        "URL",
        "HTTP endpoint",
    ],
    # Code Structure
    "class": [
        "class definition",
        "class method",
        "class attribute",
        "class inheritance",
        "base class",
    ],
    "function": [
        "function definition",
        "def",
        "method",
        "callable",
        "function call",
    ],
    "method": [
        "function",
        "class method",
        "instance method",
        "static method",
        "def",
    ],
    # Documentation
    "documentation": [
        "docs",
        "docstring",
        "README",
        "documentation file",
        "markdown",
    ],
    "docstring": [
        "documentation",
        "doc comment",
        "triple quotes",
        "function documentation",
    ],
    # Imports/Dependencies
    "import": [
        "import statement",
        "from import",
        "module import",
        "dependency",
        "package",
    ],
    "dependency": [
        "import",
        "requirement",
        "package",
        "library",
        "module",
    ],
    # Async/Concurrency
    "async": [
        "asynchronous",
        "await",
        "async def",
        "asyncio",
        "coroutine",
    ],
    "concurrency": [
        "async",
        "parallel",
        "threading",
        "multiprocessing",
        "concurrent",
    ],
}


class QueryExpander:
    """Coding-vertical query expander with code-specific expansions.

    This is a thin wrapper around the framework's QueryExpander that provides
    backward compatibility and uses coding-specific expansion terms.

    Example:
        >>> expander = QueryExpander()
        >>> expander.expand_query("tool registration")
        ['tool registration', 'register tool', '@tool decorator', ...]
    """

    def __init__(self, expansions: Optional[Dict[str, List[str]]] = None):
        """Initialize query expander.

        Args:
            expansions: Custom expansion dictionary. If None, uses SEMANTIC_QUERY_EXPANSIONS.
        """
        self.expansions = expansions or SEMANTIC_QUERY_EXPANSIONS
        self._base = FrameworkQueryExpander(QueryExpansionConfig(expansions=self.expansions))

    def expand_query(self, query: str, max_expansions: int = 5) -> List[str]:
        """Expand query with synonyms and related terms.

        Args:
            query: Original user query
            max_expansions: Maximum number of expanded queries to return (including original)

        Returns:
            List of query variations, starting with the original query

        Example:
            >>> expander = QueryExpander()
            >>> expander.expand_query("tool registration")
            [
                'tool registration',
                'register tool',
                '@tool decorator',
                'tool registry',
                'register_tool',
            ]
        """
        result = self._base.expand(query, max_expansions=max_expansions)
        return result.variations

    def is_expandable(self, query: str) -> bool:
        """Check if query has available expansions.

        Args:
            query: User query

        Returns:
            True if query can be expanded, False otherwise
        """
        return self._base.is_expandable(query)

    def get_expansion_terms(self, query: str) -> Set[str]:
        """Get all expansion terms for a query without deduplication.

        Args:
            query: User query

        Returns:
            Set of all expansion terms (excluding original query)
        """
        return self._base.get_expansion_terms(query)


# Global singleton instance
_query_expander: Optional[QueryExpander] = None


def get_query_expander() -> QueryExpander:
    """Get or create global QueryExpander instance.

    Returns:
        QueryExpander singleton instance
    """
    global _query_expander
    if _query_expander is None:
        _query_expander = QueryExpander()
    return _query_expander


def expand_query(query: str, max_expansions: int = 5) -> List[str]:
    """Convenience function to expand a query using global expander.

    Args:
        query: Original user query
        max_expansions: Maximum number of expanded queries

    Returns:
        List of query variations
    """
    expander = get_query_expander()
    return expander.expand_query(query, max_expansions)
