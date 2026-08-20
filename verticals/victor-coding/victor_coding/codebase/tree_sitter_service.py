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

"""Canonical tree-sitter service for victor-coding.

Owns three caches with distinct lifetimes:

- ``_languages``: process-wide ``Language`` objects (immutable once loaded).
- ``_parsers``: per-thread ``Parser`` instances via ``threading.local``;
  ``Parser`` is not safe to share across concurrent ``parse()`` calls.
- ``_queries``: process-wide compiled ``Query`` objects keyed by
  ``(language, kind)`` so query text is compiled once per kind, not per call.

Replaces the module-level mutable cache pattern that previously lived in
``tree_sitter_manager.py`` (which is now a thin compatibility shim).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tree_sitter import Language, Parser, Query, QueryCursor

logger = logging.getLogger(__name__)


# Canonical language key → (grammar module name, factory function name).
# Adding a new language is a one-line edit here plus a plugin.
LANGUAGE_MODULES: Dict[str, Tuple[str, str]] = {
    # Core languages (commonly used)
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "java": ("tree_sitter_java", "language"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    # Additional languages
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "c_sharp": ("tree_sitter_c_sharp", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "php": ("tree_sitter_php", "language_php"),
    "kotlin": ("tree_sitter_kotlin", "language"),
    "swift": ("tree_sitter_swift", "language"),
    "scala": ("tree_sitter_scala", "language"),
    "bash": ("tree_sitter_bash", "language"),
    "sql": ("tree_sitter_sql", "language"),
    # Web languages
    "html": ("tree_sitter_html", "language"),
    "css": ("tree_sitter_css", "language"),
    "json": ("tree_sitter_json", "language"),
    "yaml": ("tree_sitter_yaml", "language"),
    "toml": ("tree_sitter_toml", "language"),
    # Other
    "lua": ("tree_sitter_lua", "language"),
    "elixir": ("tree_sitter_elixir", "language"),
    "haskell": ("tree_sitter_haskell", "language"),
    "r": ("tree_sitter_r", "language"),
    # Tier-2 coding languages added in the post-TSA plugin sweep.
    "zig": ("tree_sitter_zig", "language"),
    "julia": ("tree_sitter_julia", "language"),
    "ocaml": ("tree_sitter_ocaml", "language_ocaml"),
    "solidity": ("tree_sitter_solidity", "language"),
    "perl": ("tree_sitter_perl", "language"),
    "objc": ("tree_sitter_objc", "language"),
    # Tier-3 build / scripting / schema languages.
    "make": ("tree_sitter_make", "language"),
    "cmake": ("tree_sitter_cmake", "language"),
    "graphql": ("tree_sitter_graphql", "language"),
    "groovy": ("tree_sitter_groovy", "language"),
    "hcl": ("tree_sitter_hcl", "language"),
    # Markup with meaningful symbols. Markdown headings become "function"
    # symbols so code-intelligence can answer "where is the # API section
    # in this README?". Fenced code blocks surface as "class" symbols keyed
    # on the info string language (python, rust, etc.).
    "markdown": ("tree_sitter_markdown", "language"),
    # Tier-4 hardware-description / shader languages. CCG is meaningful
    # for all three (process/always blocks + generate loops in HDLs;
    # standard C-style control flow in GLSL).
    "vhdl": ("tree_sitter_vhdl", "language"),
    "verilog": ("tree_sitter_verilog", "language"),
    "glsl": ("tree_sitter_glsl", "language"),
}


# Aliases users may pass in; mapped to canonical keys above.
_LANGUAGE_ALIASES: Dict[str, str] = {
    "typescriptreact": "tsx",
    "csharp": "c_sharp",
    "cs": "c_sharp",
    "c++": "cpp",
    "cxx": "cpp",
    "cc": "cpp",
    "js": "javascript",
    "node": "javascript",
    "py": "python",
    "rs": "rust",
    "kt": "kotlin",
    "rb": "ruby",
}


@dataclass(frozen=True)
class ParsedSource:
    """One canonical parsed-source handle shared across extractors.

    The ``tree`` and ``root_node`` come from a single ``parser.parse`` call so
    that symbol/edge/import/chunk extraction can be driven from the same
    underlying parse without re-parsing.
    """

    language: str
    content: bytes
    tree: Any
    root_node: Any
    file_path: Optional[str] = None


class TreeSitterService:
    """Process-wide tree-sitter facade.

    Thread-safety guarantees:

    - ``_languages`` reads are lock-free; the lock only protects the
      first-writer race when multiple threads request the same uncached
      language simultaneously.
    - ``_parsers`` lives in ``threading.local`` — each thread sees its own
      ``Parser`` instance, so concurrent ``parse()`` calls never collide.
    - ``_queries`` reads are lock-free; the lock protects against duplicate
      compilation under the same race.
    """

    def __init__(self) -> None:
        self._languages: Dict[str, Language] = {}
        self._language_lock = threading.Lock()
        self._parsers_tls = threading.local()
        self._queries: Dict[Tuple[str, str], Query] = {}
        self._query_lock = threading.Lock()
        self._logged_failures: set[Tuple[str, str]] = set()
        self._log_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Language resolution
    # ------------------------------------------------------------------ #

    def normalize_language(self, language: Optional[str]) -> str:
        """Map a user-provided language string to a canonical key.

        Returns an empty string for ``None``/empty input. Unknown languages
        are returned lowercased so callers can still pass them to
        ``supports_language`` for a False result.
        """
        if not language:
            return ""
        normalized = language.lower()
        return _LANGUAGE_ALIASES.get(normalized, normalized)

    def supports_language(self, language: str) -> bool:
        lang = self.normalize_language(language)
        if lang not in LANGUAGE_MODULES:
            return False
        return self.get_language(lang) is not None

    def get_language(self, language: str) -> Optional[Language]:
        lang = self.normalize_language(language)
        cached = self._languages.get(lang)
        if cached is not None:
            return cached

        module_info = LANGUAGE_MODULES.get(lang)
        if module_info is None:
            self._log_once(lang, "unsupported")
            return None

        module_name, func_name = module_info
        try:
            module = __import__(module_name)
            func = getattr(module, func_name)
            raw = func()
            obj = Language(raw) if not isinstance(raw, Language) else raw
        except ImportError:
            self._log_once(lang, "grammar_missing")
            return None
        except AttributeError:
            self._log_once(lang, "grammar_function_missing")
            return None
        except Exception:  # pragma: no cover - defensive
            self._log_once(lang, "grammar_load_failed")
            return None

        with self._language_lock:
            existing = self._languages.get(lang)
            if existing is not None:
                return existing
            self._languages[lang] = obj
        return obj

    # ------------------------------------------------------------------ #
    # Parser (per-thread)
    # ------------------------------------------------------------------ #

    def _parser_cache(self) -> Dict[str, Parser]:
        cache = getattr(self._parsers_tls, "cache", None)
        if cache is None:
            cache = {}
            self._parsers_tls.cache = cache
        return cache

    def get_parser(self, language: str) -> Optional[Parser]:
        lang = self.normalize_language(language)
        cache = self._parser_cache()
        existing = cache.get(lang)
        if existing is not None:
            return existing
        ts_lang = self.get_language(lang)
        if ts_lang is None:
            return None
        parser = Parser(ts_lang)
        cache[lang] = parser
        return parser

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #

    def parse(
        self,
        content: bytes,
        language: str,
        *,
        file_path: Optional[str] = None,
    ) -> Optional[ParsedSource]:
        lang = self.normalize_language(language)
        parser = self.get_parser(lang)
        if parser is None:
            return None
        try:
            tree = parser.parse(content)
        except Exception:  # pragma: no cover - defensive
            self._log_once(lang, "parse_failed")
            return None
        return ParsedSource(
            language=lang,
            content=content,
            tree=tree,
            root_node=tree.root_node,
            file_path=file_path,
        )

    # ------------------------------------------------------------------ #
    # Queries (compiled, cached by (language, kind))
    # ------------------------------------------------------------------ #

    def get_query(self, language: str, kind: str, source: str) -> Optional[Query]:
        # Cache key includes `source` so that multiple distinct queries
        # sharing the same kind (e.g. all of markdown's h1/h2/h3/.../h6
        # patterns ship as `symbol_type="function"`) get independent
        # entries instead of colliding on the same kind and returning the
        # first-cached query for every subsequent lookup. `kind` stays in
        # the key so the same source registered under different kinds
        # caches separately (rare but valid).
        lang = self.normalize_language(language)
        key = (lang, f"{kind}\x00{source}")
        existing = self._queries.get(key)
        if existing is not None:
            return existing
        ts_lang = self.get_language(lang)
        if ts_lang is None:
            return None
        try:
            query = Query(ts_lang, source)
        except Exception:
            self._log_once(lang, f"query_compile_failed:{kind}")
            return None
        with self._query_lock:
            existing = self._queries.get(key)
            if existing is not None:
                return existing
            self._queries[key] = query
        return query

    def run_query(self, parsed: ParsedSource, kind: str, source: str) -> Dict[str, List[Any]]:
        query = self.get_query(parsed.language, kind, source)
        if query is None:
            return {}
        cursor = QueryCursor(query)
        return cursor.captures(parsed.root_node)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _log_once(self, language: str, reason: str) -> None:
        key = (language, reason)
        with self._log_lock:
            if key in self._logged_failures:
                return
            self._logged_failures.add(key)
        logger.debug("tree-sitter %s: %s", reason, language)


_service_instance: Optional[TreeSitterService] = None
_service_lock = threading.Lock()


def get_tree_sitter_service() -> TreeSitterService:
    """Return the process-wide ``TreeSitterService`` singleton."""
    global _service_instance
    if _service_instance is not None:
        return _service_instance
    with _service_lock:
        if _service_instance is None:
            _service_instance = TreeSitterService()
    return _service_instance


def _reset_for_tests() -> None:
    """Drop the singleton so the next call rebuilds a fresh service.

    Intended for use in test fixtures only.
    """
    global _service_instance
    with _service_lock:
        _service_instance = None
