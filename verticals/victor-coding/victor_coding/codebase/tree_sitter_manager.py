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

"""Backward-compatible wrappers around :class:`TreeSitterService`.

This module preserves the historical ``get_language()`` / ``get_parser()`` /
``run_query()`` entry points so existing callers keep working. New code should
use ``get_tree_sitter_service()`` directly to get a ``ParsedSource`` it can
reuse across symbol/edge/import/chunk extraction.
"""

from typing import TYPE_CHECKING, Dict, List

from tree_sitter import Language, Parser, Query, QueryCursor

from victor_coding.codebase.tree_sitter_service import (
    LANGUAGE_MODULES,
    get_tree_sitter_service,
)

if TYPE_CHECKING:
    from tree_sitter import Node, Tree


__all__ = ["LANGUAGE_MODULES", "get_language", "get_parser", "run_query"]


def get_language(language: str) -> Language:
    """Return a cached :class:`Language` for ``language`` or raise.

    Preserves the legacy behavior of raising ``ValueError``/``ImportError``/
    ``AttributeError`` rather than returning ``None``. New code should prefer
    ``TreeSitterService.get_language`` which returns ``None`` on failure.
    """
    service = get_tree_sitter_service()
    normalized = service.normalize_language(language)
    if normalized not in LANGUAGE_MODULES:
        raise ValueError(f"Unsupported language for tree-sitter: {language}")

    obj = service.get_language(normalized)
    if obj is not None:
        return obj

    # Reproduce the original error messages by re-attempting the import.
    module_name, func_name = LANGUAGE_MODULES[normalized]
    try:
        module = __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Language package '{module_name}' not installed. "
            f"Install it with: pip install {module_name.replace('_', '-')}"
        ) from exc
    try:
        func = getattr(module, func_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Language module '{module_name}' does not have function '{func_name}'. "
            f"Check the tree-sitter package version and update LANGUAGE_MODULES."
        ) from exc
    raw = func()
    return Language(raw) if not isinstance(raw, Language) else raw


def get_parser(language: str) -> Parser:
    """Return a per-thread cached :class:`Parser` for ``language``.

    Parsers are no longer shared across worker threads; each thread that
    calls this gets its own instance from :class:`TreeSitterService`.
    """
    service = get_tree_sitter_service()
    parser = service.get_parser(language)
    if parser is not None:
        return parser
    # Service returned None — re-raise via get_language so callers see the
    # historical error type.
    get_language(language)
    raise RuntimeError(f"Tree-sitter parser unavailable for language: {language}")


def run_query(tree: "Tree", query_src: str, language: str) -> Dict[str, List["Node"]]:
    """Run a tree-sitter query using the modern ``QueryCursor`` API.

    This wrapper compiles the query per call (it does not consult the
    ``(language, kind)`` cache because no kind is supplied here). For
    hot-path code, use ``TreeSitterService.run_query`` with a stable kind so
    the compiled ``Query`` is cached.
    """
    lang = get_language(language)
    query = Query(lang, query_src)
    cursor = QueryCursor(query)
    return cursor.captures(tree.root_node)
