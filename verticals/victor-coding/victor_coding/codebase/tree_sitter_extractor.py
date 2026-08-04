# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Unified tree-sitter symbol and relationship extraction.

This module provides a registry-based approach to code analysis using
tree-sitter, eliminating language-specific branching in the indexer.
All language-specific queries are defined in the LanguagePlugin classes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from tree_sitter import Query, QueryCursor

from victor_coding.languages.registry import LanguageRegistry, get_language_registry

if TYPE_CHECKING:
    from tree_sitter import Node, Parser, Tree

logger = logging.getLogger(__name__)

# ADR-014 step 2 (parse-layer de-duplication): source symbol extraction from the shared,
# neutral ``victor-codegraph`` package — the genuinely-duplicated tree-sitter parsing —
# rather than re-implementing it here. Victor-coding keeps its own hierarchical embedding
# chunker (CodeChunker); only the underlying symbol *extraction* is delegated. Soft import
# so the extractor works without the package (editable-only until published:
# ``pip install -e ../../victor-codegraph``); default ON when available, disable with
# ``VICTOR_CODING_USE_CODEGRAPH=0``.
try:  # pragma: no cover - availability depends on the optional package
    import victor_codegraph as _victor_codegraph
except Exception:
    _victor_codegraph = None


def _codegraph_enabled() -> bool:
    import os

    return _victor_codegraph is not None and os.getenv(
        "VICTOR_CODING_USE_CODEGRAPH", "1"
    ).strip().lower() not in ("0", "false", "no", "off")


@dataclass
class ExtractedSymbol:
    """A symbol extracted from source code."""

    name: str
    type: str  # "class", "function", etc.
    file_path: str
    line_number: int
    end_line: Optional[int] = None
    parent_symbol: Optional[str] = None
    symbol_id: Optional[str] = None
    legacy_symbol_id: Optional[str] = None


@dataclass
class ExtractedEdge:
    """A relationship edge extracted from source code."""

    source: str  # caller, child class, owner class
    target: str  # callee, base class, type
    edge_type: str  # "CALLS", "INHERITS", "IMPLEMENTS", "COMPOSITION"
    file_path: str
    line_number: int


@dataclass
class ExtractedReference:
    """A reference to a symbol extracted from source code."""

    name: str  # referenced symbol name
    file_path: str
    line_number: int
    column: int
    enclosing_scope: Optional[str] = None  # function/class containing this reference


class TreeSitterExtractor:
    """Unified tree-sitter extraction using language registry.

    This class eliminates the need for `if language == "python"` branching
    by delegating all language-specific logic to the registered plugins.
    """

    def __init__(self, registry: Optional[LanguageRegistry] = None, auto_discover: bool = True):
        """Initialize extractor with optional custom registry.

        Args:
            registry: Language registry to use. If None, uses global registry.
            auto_discover: If True, automatically discover and register plugins.
        """
        self.registry = registry or get_language_registry()
        if auto_discover and not self.registry._plugins:
            self.registry.discover_plugins()
        self._parsers: Dict[str, "Parser"] = {}

    def _get_parser(self, language: str) -> Optional["Parser"]:
        """Get tree-sitter parser for a language.

        Per-thread caching now lives in ``TreeSitterService``; this method
        retains its public shape for callers and any test seams that patch
        ``self._parsers``.
        """
        if language in self._parsers:
            return self._parsers[language]

        from victor_coding.codebase.tree_sitter_service import get_tree_sitter_service

        parser = get_tree_sitter_service().get_parser(language)
        if parser is None:
            return None
        self._parsers[language] = parser
        return parser

    def _run_query(self, tree: "Tree", query_src: str, parser: "Parser") -> Dict[str, List["Node"]]:
        """Run a tree-sitter query using the new QueryCursor API.

        Args:
            tree: Parsed syntax tree
            query_src: Query source string
            parser: Parser (for getting language)

        Returns:
            Dictionary mapping capture names to lists of matching nodes
        """
        try:
            query = Query(parser.language, query_src)
            cursor = QueryCursor(query)
            return cursor.captures(tree.root_node)
        except Exception as e:
            logger.debug(f"Query failed: {e}")
            return {}

    def detect_language(self, file_path: Path) -> Optional[str]:
        """Detect language for a file using the registry.

        Args:
            file_path: Path to the file

        Returns:
            Language name or None if not detected
        """
        return self.registry.detect_language(file_path)

    def _extract_symbols_via_codegraph(
        self, file_path: Path, language: str
    ) -> Optional[List[ExtractedSymbol]]:
        """Delegate symbol extraction to victor-codegraph (ADR-014, parse layer).

        Returns ``None`` on failure or empty result so the caller falls back to the
        registry/tree-sitter path. Maps ``CodeSymbol`` -> ``ExtractedSymbol``.
        """
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            parsed = _victor_codegraph.parse(content, language=language, file_path=str(file_path))
        except Exception as e:  # noqa: BLE001 - fall back on any delegation failure
            logger.debug(f"victor-codegraph symbol delegation failed for {file_path}: {e}")
            return None
        if not parsed.symbols:
            return None
        return [
            ExtractedSymbol(
                name=s.simple_name,
                type=s.symbol_type.name.lower(),
                file_path=str(file_path),
                line_number=s.location.start_line,
                end_line=s.location.end_line or None,
                parent_symbol=s.scope_chain[-1] if s.scope_chain else None,
                symbol_id=s.id,
                legacy_symbol_id=s.legacy_id,
            )
            for s in parsed.symbols
        ]

    def extract_symbols(
        self, file_path: Path, language: Optional[str] = None
    ) -> List[ExtractedSymbol]:
        """Extract symbols from a file using registered queries.

        Args:
            file_path: Path to the source file
            language: Language name (auto-detected if None)

        Returns:
            List of extracted symbols
        """
        if language is None:
            language = self.detect_language(file_path)
            if language is None:
                return []

        # ADR-014: prefer the shared parser for symbol extraction when available.
        if _codegraph_enabled():
            delegated = self._extract_symbols_via_codegraph(file_path, language)
            if delegated is not None:
                return delegated

        try:
            plugin = self.registry.get(language)
        except KeyError:
            return []

        queries = plugin.tree_sitter_queries
        if not queries.symbols:
            return []

        parser = self._get_parser(language)
        if parser is None:
            return []

        try:
            content = file_path.read_bytes()
            tree = parser.parse(content)
        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return []

        symbols: List[ExtractedSymbol] = []
        relative_path = str(file_path)

        for pattern in queries.symbols:
            captures = self._run_query(tree, pattern.query, parser)

            # Get name and def captures (def is for end_line boundaries)
            name_nodes = captures.get("name", [])
            def_nodes = captures.get("def", [])

            # Build a map of line -> def node for end_line lookup
            # This handles queries that capture both @name and @def
            def_by_start_line = {}
            for def_node in def_nodes:
                def_by_start_line[def_node.start_point[0]] = def_node

            for node in name_nodes:
                text = node.text.decode("utf-8", errors="ignore")
                if text:
                    # Look up corresponding def node for end_line
                    # The def node contains the name node, so they share the same start area
                    end_line = node.end_point[0] + 1  # Default: name's end line
                    name_line = node.start_point[0]

                    # Search for def node that contains this name
                    for def_start, def_node in def_by_start_line.items():
                        if def_start <= name_line and def_node.end_point[0] >= name_line:
                            end_line = def_node.end_point[0] + 1
                            break

                    symbols.append(
                        ExtractedSymbol(
                            name=text,
                            type=pattern.symbol_type,
                            file_path=relative_path,
                            line_number=node.start_point[0] + 1,
                            end_line=end_line,
                        )
                    )

        return symbols

    def extract_call_edges(
        self, file_path: Path, language: Optional[str] = None
    ) -> List[ExtractedEdge]:
        """Extract caller->callee edges from a file.

        Args:
            file_path: Path to the source file
            language: Language name (auto-detected if None)

        Returns:
            List of call edges
        """
        if language is None:
            language = self.detect_language(file_path)
            if language is None:
                return []

        try:
            plugin = self.registry.get(language)
        except KeyError:
            return []

        queries = plugin.tree_sitter_queries
        if not queries.calls:
            return []

        parser = self._get_parser(language)
        if parser is None:
            return []

        try:
            content = file_path.read_bytes()
            tree = parser.parse(content)
        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return []

        relative_path = str(file_path)
        edges: List[ExtractedEdge] = []

        captures = self._run_query(tree, queries.calls, parser)
        for capture_name, nodes in captures.items():
            if capture_name == "callee":
                for node in nodes:
                    callee = node.text.decode("utf-8", errors="ignore")
                    if callee:
                        caller = self._find_enclosing_symbol(node, queries.enclosing_scopes)
                        if caller:
                            edges.append(
                                ExtractedEdge(
                                    source=caller,
                                    target=callee,
                                    edge_type="CALLS",
                                    file_path=relative_path,
                                    line_number=node.start_point[0] + 1,
                                )
                            )

        return edges

    def extract_inheritance_edges(
        self, file_path: Path, language: Optional[str] = None
    ) -> List[ExtractedEdge]:
        """Extract class inheritance edges from a file.

        Args:
            file_path: Path to the source file
            language: Language name (auto-detected if None)

        Returns:
            List of inheritance edges (child -> base)
        """
        if language is None:
            language = self.detect_language(file_path)
            if language is None:
                return []

        try:
            plugin = self.registry.get(language)
        except KeyError:
            return []

        queries = plugin.tree_sitter_queries
        if not queries.inheritance:
            return []

        parser = self._get_parser(language)
        if parser is None:
            return []

        try:
            content = file_path.read_bytes()
            tree = parser.parse(content)
        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return []

        relative_path = str(file_path)
        edges: List[ExtractedEdge] = []

        captures = self._run_query(tree, queries.inheritance, parser)

        # Process captures to pair up child and base
        child_nodes = captures.get("child", [])
        base_nodes = captures.get("base", [])

        # Match child and base by position (they appear in order)
        for i, child_node in enumerate(child_nodes):
            child = child_node.text.decode("utf-8", errors="ignore")
            # Find corresponding base(s)
            if i < len(base_nodes):
                base = base_nodes[i].text.decode("utf-8", errors="ignore")
                if child and base:
                    edges.append(
                        ExtractedEdge(
                            source=child,
                            target=base,
                            edge_type="INHERITS",
                            file_path=relative_path,
                            line_number=child_node.start_point[0] + 1,
                        )
                    )

        return edges

    def extract_implements_edges(
        self, file_path: Path, language: Optional[str] = None
    ) -> List[ExtractedEdge]:
        """Extract interface implementation edges from a file.

        Args:
            file_path: Path to the source file
            language: Language name (auto-detected if None)

        Returns:
            List of implements edges (class -> interface)
        """
        if language is None:
            language = self.detect_language(file_path)
            if language is None:
                return []

        try:
            plugin = self.registry.get(language)
        except KeyError:
            return []

        queries = plugin.tree_sitter_queries
        if not queries.implements:
            return []

        parser = self._get_parser(language)
        if parser is None:
            return []

        try:
            content = file_path.read_bytes()
            tree = parser.parse(content)
        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return []

        relative_path = str(file_path)
        edges: List[ExtractedEdge] = []

        captures = self._run_query(tree, queries.implements, parser)

        child_nodes = captures.get("child", [])
        interface_nodes = captures.get("interface", []) or captures.get("base", [])

        for i, child_node in enumerate(child_nodes):
            child = child_node.text.decode("utf-8", errors="ignore")
            if i < len(interface_nodes):
                interface = interface_nodes[i].text.decode("utf-8", errors="ignore")
                if child and interface:
                    edges.append(
                        ExtractedEdge(
                            source=child,
                            target=interface,
                            edge_type="IMPLEMENTS",
                            file_path=relative_path,
                            line_number=child_node.start_point[0] + 1,
                        )
                    )

        return edges

    def extract_composition_edges(
        self, file_path: Path, language: Optional[str] = None
    ) -> List[ExtractedEdge]:
        """Extract composition/has-a relationship edges from a file.

        Args:
            file_path: Path to the source file
            language: Language name (auto-detected if None)

        Returns:
            List of composition edges (owner -> member_type)
        """
        if language is None:
            language = self.detect_language(file_path)
            if language is None:
                return []

        try:
            plugin = self.registry.get(language)
        except KeyError:
            return []

        queries = plugin.tree_sitter_queries
        if not queries.composition:
            return []

        parser = self._get_parser(language)
        if parser is None:
            return []

        try:
            content = file_path.read_bytes()
            tree = parser.parse(content)
        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return []

        relative_path = str(file_path)
        edges: List[ExtractedEdge] = []

        captures = self._run_query(tree, queries.composition, parser)

        owner_nodes = captures.get("owner", [])
        type_nodes = captures.get("type", [])

        for i, owner_node in enumerate(owner_nodes):
            owner = owner_node.text.decode("utf-8", errors="ignore")
            if i < len(type_nodes):
                member_type = type_nodes[i].text.decode("utf-8", errors="ignore")
                if owner and member_type:
                    edges.append(
                        ExtractedEdge(
                            source=owner,
                            target=member_type,
                            edge_type="COMPOSITION",
                            file_path=relative_path,
                            line_number=owner_node.start_point[0] + 1,
                        )
                    )

        return edges

    def extract_references(
        self, file_path: Path, language: Optional[str] = None
    ) -> List[ExtractedReference]:
        """Extract all identifier references from a file.

        This is useful for building "find references" functionality and
        understanding symbol usage patterns across the codebase.

        Args:
            file_path: Path to the source file
            language: Language name (auto-detected if None)

        Returns:
            List of references found in the file
        """
        if language is None:
            language = self.detect_language(file_path)
            if language is None:
                return []

        try:
            plugin = self.registry.get(language)
        except KeyError:
            return []

        queries = plugin.tree_sitter_queries
        if not queries.references:
            return []

        parser = self._get_parser(language)
        if parser is None:
            return []

        try:
            content = file_path.read_bytes()
            tree = parser.parse(content)
        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            return []

        relative_path = str(file_path)
        references: List[ExtractedReference] = []

        captures = self._run_query(tree, queries.references, parser)
        for capture_name, nodes in captures.items():
            if capture_name == "name":
                for node in nodes:
                    name = node.text.decode("utf-8", errors="ignore")
                    if name:
                        enclosing = self._find_enclosing_symbol(node, queries.enclosing_scopes)
                        references.append(
                            ExtractedReference(
                                name=name,
                                file_path=relative_path,
                                line_number=node.start_point[0] + 1,
                                column=node.start_point[1],
                                enclosing_scope=enclosing,
                            )
                        )

        return references

    def extract_all(
        self, file_path: Path, language: Optional[str] = None
    ) -> Tuple[List[ExtractedSymbol], List[ExtractedEdge]]:
        """Extract all symbols and edges from a file.

        This is the main entry point for comprehensive file analysis.

        Args:
            file_path: Path to the source file
            language: Language name (auto-detected if None)

        Returns:
            Tuple of (symbols, edges) where edges include calls,
            inheritance, implements, and composition relationships.
        """
        if language is None:
            language = self.detect_language(file_path)
            if language is None:
                return [], []

        symbols = self.extract_symbols(file_path, language)

        edges: List[ExtractedEdge] = []
        edges.extend(self.extract_call_edges(file_path, language))
        edges.extend(self.extract_inheritance_edges(file_path, language))
        edges.extend(self.extract_implements_edges(file_path, language))
        edges.extend(self.extract_composition_edges(file_path, language))

        return symbols, edges

    def extract_all_with_references(
        self, file_path: Path, language: Optional[str] = None
    ) -> Tuple[List[ExtractedSymbol], List[ExtractedEdge], List[ExtractedReference]]:
        """Extract all symbols, edges, and references from a file.

        Extended version of extract_all that also includes references.
        Useful for comprehensive analysis including "find references" support.

        Args:
            file_path: Path to the source file
            language: Language name (auto-detected if None)

        Returns:
            Tuple of (symbols, edges, references)
        """
        if language is None:
            language = self.detect_language(file_path)
            if language is None:
                return [], [], []

        symbols, edges = self.extract_all(file_path, language)
        references = self.extract_references(file_path, language)

        return symbols, edges, references

    def _find_enclosing_symbol(
        self, node: "Node", enclosing_scopes: List[Tuple[str, str]]
    ) -> Optional[str]:
        """Find the name of the enclosing function/method for a node.

        Args:
            node: The AST node to find enclosing scope for
            enclosing_scopes: List of (node_type, name_field) tuples

        Returns:
            Enclosing symbol name (e.g., "ClassName.method_name") or None
        """
        if not enclosing_scopes:
            return None

        current = node.parent
        method_name: Optional[str] = None
        class_name: Optional[str] = None

        while current is not None:
            for node_type, field_name in enclosing_scopes:
                if current.type == node_type:
                    field = current.child_by_field_name(field_name)
                    if field:
                        text = field.text.decode("utf-8", errors="ignore")
                        # Determine if this is a class or method
                        if node_type in (
                            "class_declaration",
                            "class_definition",
                            "interface_declaration",
                            "class_specifier",
                            "struct_item",
                            "type_declaration",
                        ):
                            class_name = class_name or text
                        else:
                            method_name = method_name or text
            current = current.parent

        if method_name:
            if class_name:
                return f"{class_name}.{method_name}"
            return method_name
        return class_name
