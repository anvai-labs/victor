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

"""Plugin-backed tree-sitter analysis provider.

Implements the root ``TreeSitterAnalysisProtocol`` so root framework code can
ask victor-coding for symbols, edges, imports, and chunk context without
importing any language-specific internals. Wraps :class:`TreeSitterService`
and consults :class:`LanguageRegistry` for per-language query packs.

Output shapes follow the LLD "Data Contracts" section:

- Symbol dict keys: ``name``, ``symbol_type``, ``file_path``, ``line_start``,
  ``line_end``, ``parent_symbol`` (optional), ``ast_kind`` (optional).
- Edge dict keys: ``source``, ``target``, ``edge_type``, ``file_path``,
  ``line_number``; optional ``receiver_type``, ``is_method_call``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from victor_coding.codebase.tree_sitter_service import (
    ParsedSource,
    TreeSitterService,
    get_tree_sitter_service,
)
from victor_coding.languages.registry import LanguageRegistry, get_language_registry

logger = logging.getLogger(__name__)

# ADR-014/015: source symbols from the shared victor-codegraph parser when importable,
# so the symbol ids/shape are consistent across Victor core, ProximaDB, and AnvaiOps.
# Soft import; default ON when available (disable with VICTOR_CODING_USE_CODEGRAPH=0).
try:  # pragma: no cover - availability depends on the optional package
    import victor_codegraph as _victor_codegraph
except Exception:
    _victor_codegraph = None

# victor-codegraph symbol-type -> ast_kind (the analysis provider's tree-sitter node-type
# field). Functions/methods/constructors map to a function-definition kind; class-likes to
# their definition kind.
_CODEGRAPH_AST_KIND = {
    "class": "class_definition",
    "struct": "struct_item",
    "interface": "interface_declaration",
    "trait": "trait_item",
    "enum": "enum_declaration",
}


def _codegraph_symbols_enabled() -> bool:
    import os

    return _victor_codegraph is not None and os.getenv(
        "VICTOR_CODING_USE_CODEGRAPH", "1"
    ).strip().lower() not in ("0", "false", "no", "off")


# AST parent-types that indicate a captured callee is a method dispatch.
_METHOD_CALL_PARENT_TYPES = frozenset(
    {
        "attribute",  # python: obj.method
        "member_expression",  # js/ts: obj.method
        "field_expression",  # rust: obj.method
        "field_access",  # java: obj.method
        "selector_expression",  # go: obj.method
        "member_access_expression",  # c#: obj.method
    }
)


@dataclass(frozen=True)
class _ChunkContextView:
    """Minimal duck-typed object accepted by root's TreeSitterParseContext.

    Root constructs its own ``TreeSitterParseContext.from_content`` using
    ``root_node`` and ``content``; we hand both back in a stable shape.
    """

    root_node: Any
    content: str
    language: str
    file_path: Optional[str] = None


class TreeSitterAnalysisProvider:
    """Concrete provider for the root ``TreeSitterAnalysisProtocol``.

    Each ``extract_*`` call parses the content once via
    :class:`TreeSitterService`, then runs the language plugin's pre-defined
    queries. Callers can also call :meth:`parse` first and reuse the
    ``ParsedSource`` across multiple extractions via :meth:`extract_from_parsed`.
    """

    def __init__(
        self,
        *,
        service: Optional[TreeSitterService] = None,
        registry: Optional[LanguageRegistry] = None,
    ) -> None:
        self._service = service or get_tree_sitter_service()
        self._registry = registry or get_language_registry()
        if not self._registry._plugins:
            try:
                self._registry.discover_plugins()
            except Exception:  # pragma: no cover - defensive
                logger.debug("Plugin discovery failed", exc_info=True)

    # ------------------------------------------------------------------ #
    # Protocol surface
    # ------------------------------------------------------------------ #

    def supports_language(self, language: str) -> bool:
        lang = self._service.normalize_language(language)
        if not self._service.supports_language(lang):
            return False
        try:
            self._registry.get(lang)
            return True
        except KeyError:
            return False

    def parse(
        self,
        content: bytes,
        language: str,
        *,
        file_path: Optional[str] = None,
    ) -> Optional[ParsedSource]:
        return self._service.parse(content, language, file_path=file_path)

    def extract_symbols(
        self,
        content: bytes,
        language: str,
        *,
        file_path: str,
    ) -> List[Dict[str, Any]]:
        if _codegraph_symbols_enabled():
            delegated = self._symbols_via_codegraph(content, language, file_path)
            if delegated is not None:
                return delegated
        parsed = self.parse(content, language, file_path=file_path)
        if parsed is None:
            return []
        return self._symbols_from_parsed(parsed, file_path)

    def _symbols_via_codegraph(
        self, content: bytes, language: str, file_path: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Source symbols from victor-codegraph (ADR-015). Returns None on failure so the
        caller falls back to the tree-sitter path. Output matches the symbol-dict contract.
        """
        try:
            src = (
                content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
            )
            parsed = _victor_codegraph.parse(src, language=language, file_path=file_path)
        except Exception as e:  # noqa: BLE001 - fall back on any failure
            logger.debug("victor-codegraph symbol delegation failed for %s: %s", file_path, e)
            return None
        if not parsed.symbols:
            return None
        by_scope = {(*s.scope_chain, s.simple_name): s.id for s in parsed.symbols}
        out: List[Dict[str, Any]] = []
        for s in parsed.symbols:
            stype = s.symbol_type.name.lower()
            out.append(
                {
                    "name": s.simple_name,
                    "symbol_type": stype,
                    "file_path": s.location.file_path or file_path,
                    "line_start": s.location.start_line,
                    "line_end": s.location.end_line,
                    "parent_symbol": s.scope_chain[-1] if s.scope_chain else None,
                    "ast_kind": _CODEGRAPH_AST_KIND.get(stype, "function_definition"),
                    "symbol_id": s.id,
                    "legacy_symbol_id": s.legacy_id,
                    "parent_id": by_scope.get(tuple(s.scope_chain)),
                    "signature": s.signature,
                    "docstring": s.documentation,
                    "visibility": "private" if "private" in s.modifiers else "public",
                }
            )
        return out

    def extract_edges(
        self,
        content: bytes,
        language: str,
        *,
        file_path: str,
    ) -> List[Dict[str, Any]]:
        if _codegraph_symbols_enabled():
            try:
                src = content.decode("utf-8", errors="replace")
                codegraph = _victor_codegraph.parse(src, language=language, file_path=file_path)
                if codegraph.symbols:
                    names = {s.id: s.simple_name for s in codegraph.symbols}
                    mapped: List[Dict[str, Any]] = []
                    for relation in codegraph.relations:
                        source = names.get(relation.from_symbol_id)
                        target = names.get(relation.to_symbol_id)
                        if source is None:
                            continue
                        target = target or (
                            relation.target_ref.name
                            if relation.target_ref is not None
                            else relation.to_symbol_id
                        )
                        edge_type = (
                            "INHERITS"
                            if relation.relation_type.name == "EXTENDS"
                            else relation.relation_type.name
                        )
                        mapped.append(
                            {
                                "source": source,
                                "target": target,
                                "source_id": relation.from_symbol_id,
                                "target_id": (
                                    relation.to_symbol_id if relation.target_ref is None else None
                                ),
                                "edge_type": edge_type,
                                "file_path": file_path,
                                "line_number": (
                                    relation.call_site.start_line
                                    if relation.call_site is not None
                                    else 0
                                ),
                                "is_method_call": bool(
                                    relation.target_ref and relation.target_ref.qualifier
                                ),
                                "confidence": relation.confidence,
                            }
                        )
                    return mapped
            except Exception as e:  # noqa: BLE001 - preserve the legacy fallback
                logger.debug("victor-codegraph edge delegation failed for %s: %s", file_path, e)
        parsed = self.parse(content, language, file_path=file_path)
        if parsed is None:
            return []
        return self._edges_from_parsed(parsed, file_path)

    def extract_imports(
        self,
        content: bytes,
        language: str,
        *,
        file_path: Optional[str] = None,
    ) -> List[str]:
        if _codegraph_symbols_enabled() and file_path is not None:
            try:
                src = content.decode("utf-8", errors="replace")
                codegraph = _victor_codegraph.parse(src, language=language, file_path=file_path)
                if codegraph.status.value in {"success", "partial"}:
                    return list(codegraph.imports)
            except Exception as e:  # noqa: BLE001 - preserve the legacy fallback
                logger.debug("victor-codegraph import delegation failed for %s: %s", file_path, e)
        parsed = self.parse(content, language, file_path=file_path)
        if parsed is None:
            return []
        return self._imports_from_parsed(parsed)

    def build_chunk_context(
        self,
        content: str,
        language: str,
        *,
        file_path: Optional[str] = None,
    ) -> Optional[_ChunkContextView]:
        parsed = self.parse(content.encode("utf-8"), language, file_path=file_path)
        if parsed is None:
            return None
        return _ChunkContextView(
            root_node=parsed.root_node,
            content=content,
            language=parsed.language,
            file_path=file_path,
        )

    # ------------------------------------------------------------------ #
    # Parse reuse — callers can parse once and run all extractions
    # ------------------------------------------------------------------ #

    def extract_from_parsed(
        self, parsed: ParsedSource, *, file_path: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run symbol + edge extraction on an already-parsed source."""
        symbols = self._symbols_from_parsed(parsed, file_path)
        edges = self._edges_from_parsed(parsed, file_path)
        return symbols, edges

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _plugin_queries(self, language: str):
        try:
            plugin = self._registry.get(language)
        except KeyError:
            return None
        return plugin.tree_sitter_queries

    def _symbols_from_parsed(self, parsed: ParsedSource, file_path: str) -> List[Dict[str, Any]]:
        queries = self._plugin_queries(parsed.language)
        if queries is None or not queries.symbols:
            return []

        out: List[Dict[str, Any]] = []
        for pattern in queries.symbols:
            kind = f"symbols:{pattern.symbol_type}"
            captures = self._service.run_query(parsed, kind, pattern.query)
            name_nodes = captures.get("name", [])
            def_nodes = captures.get("def", [])
            def_by_start = {n.start_point[0]: n for n in def_nodes}

            for node in name_nodes:
                text = node.text.decode("utf-8", errors="ignore")
                if not text:
                    continue
                start_line = node.start_point[0]
                end_line = node.end_point[0] + 1
                # Match the existing extractor's pairing: prefer a @def node
                # whose span contains the @name line.
                for def_start, def_node in def_by_start.items():
                    if def_start <= start_line and def_node.end_point[0] >= start_line:
                        end_line = def_node.end_point[0] + 1
                        break
                out.append(
                    {
                        "name": text,
                        "symbol_type": pattern.symbol_type,
                        "file_path": file_path,
                        "line_start": start_line + 1,
                        "line_end": end_line,
                        "ast_kind": (node.parent.type if node.parent is not None else None),
                    }
                )
        return out

    def _edges_from_parsed(self, parsed: ParsedSource, file_path: str) -> List[Dict[str, Any]]:
        queries = self._plugin_queries(parsed.language)
        if queries is None:
            return []

        edges: List[Dict[str, Any]] = []
        edges.extend(self._calls_edges(parsed, queries, file_path))
        edges.extend(
            self._pair_edges(parsed, queries.inheritance, "INHERITS", "child", "base", file_path)
        )
        implements_pairs = (("child", "interface"), ("child", "base"))
        for child_cap, target_cap in implements_pairs:
            new_edges = self._pair_edges(
                parsed,
                queries.implements,
                "IMPLEMENTS",
                child_cap,
                target_cap,
                file_path,
            )
            if new_edges:
                edges.extend(new_edges)
                break
        edges.extend(
            self._pair_edges(parsed, queries.composition, "COMPOSITION", "owner", "type", file_path)
        )
        return edges

    def _calls_edges(
        self,
        parsed: ParsedSource,
        queries,
        file_path: str,
    ) -> List[Dict[str, Any]]:
        if not queries.calls:
            return []
        captures = self._service.run_query(parsed, "edges:calls", queries.calls)
        callee_nodes = captures.get("callee", [])
        out: List[Dict[str, Any]] = []
        for node in callee_nodes:
            callee = node.text.decode("utf-8", errors="ignore")
            if not callee:
                continue
            caller = self._find_enclosing_symbol(node, queries.enclosing_scopes)
            if not caller:
                continue
            is_method = node.parent is not None and node.parent.type in _METHOD_CALL_PARENT_TYPES
            out.append(
                {
                    "source": caller,
                    "target": callee,
                    "edge_type": "CALLS",
                    "file_path": file_path,
                    "line_number": node.start_point[0] + 1,
                    "is_method_call": is_method,
                    "receiver_type": None,
                }
            )
        return out

    def _pair_edges(
        self,
        parsed: ParsedSource,
        query_src: Optional[str],
        edge_type: str,
        source_capture: str,
        target_capture: str,
        file_path: str,
    ) -> List[Dict[str, Any]]:
        if not query_src:
            return []
        captures = self._service.run_query(parsed, f"edges:{edge_type.lower()}", query_src)
        sources = captures.get(source_capture, [])
        targets = captures.get(target_capture, [])
        out: List[Dict[str, Any]] = []
        for i, src_node in enumerate(sources):
            if i >= len(targets):
                break
            src = src_node.text.decode("utf-8", errors="ignore")
            tgt = targets[i].text.decode("utf-8", errors="ignore")
            if not src or not tgt:
                continue
            out.append(
                {
                    "source": src,
                    "target": tgt,
                    "edge_type": edge_type,
                    "file_path": file_path,
                    "line_number": src_node.start_point[0] + 1,
                }
            )
        return out

    def _imports_from_parsed(self, parsed: ParsedSource) -> List[str]:
        """Best-effort import extraction.

        Walks the parse tree for nodes whose type matches a small set of
        import constructs across the supported grammars. Returns the raw
        text of each import, which is enough for graph wiring; callers
        that need symbolic module paths should re-parse upstream.
        """
        lang = parsed.language
        wanted = _IMPORT_NODE_TYPES.get(lang)
        if not wanted:
            return []

        out: List[str] = []
        stack: List[Any] = [parsed.root_node]
        while stack:
            node = stack.pop()
            if node.type in wanted:
                # export_statement only acts as an import when it re-exports
                # from a source (`export * from './x'`). Plain export
                # declarations would dump entire class/function bodies into
                # the raw-import buffer.
                if node.type == "export_statement" and not any(
                    child.type == "string" for child in node.children
                ):
                    continue
                text = node.text.decode("utf-8", errors="ignore").strip()
                if text:
                    out.append(text)
                # Don't descend into an import — its children are part of it.
                continue
            for child in node.children:
                stack.append(child)
        return out

    def _find_enclosing_symbol(
        self, node, enclosing_scopes: List[Tuple[str, str]]
    ) -> Optional[str]:
        if not enclosing_scopes:
            return None
        current = node.parent
        method_name: Optional[str] = None
        class_name: Optional[str] = None
        class_node_types = {
            "class_declaration",
            "class_definition",
            "interface_declaration",
            "class_specifier",
            "struct_item",
            "type_declaration",
        }
        while current is not None:
            for node_type, field_name in enclosing_scopes:
                if current.type == node_type:
                    field = current.child_by_field_name(field_name)
                    if field is not None:
                        text = field.text.decode("utf-8", errors="ignore")
                        if node_type in class_node_types:
                            class_name = class_name or text
                        else:
                            method_name = method_name or text
            current = current.parent
        if method_name:
            if class_name:
                return f"{class_name}.{method_name}"
            return method_name
        return class_name


# Per-language top-level import node types. Conservative — better than
# nothing for the common cases; extend per plugin as needed.
_IMPORT_NODE_TYPES: Dict[str, frozenset] = {
    "python": frozenset({"import_statement", "import_from_statement"}),
    # export_statement covers barrel-file re-exports (`export * from './x'`);
    # _imports_from_parsed keeps only export statements with a string source.
    "javascript": frozenset({"import_statement", "export_statement"}),
    "typescript": frozenset({"import_statement", "export_statement"}),
    "tsx": frozenset({"import_statement", "export_statement"}),
    "go": frozenset({"import_declaration"}),
    "rust": frozenset({"use_declaration"}),
    "java": frozenset({"import_declaration"}),
    "scala": frozenset({"import_declaration"}),
    "c": frozenset({"preproc_include"}),
    "cpp": frozenset({"preproc_include"}),
}
