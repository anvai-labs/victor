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

"""Plugin-backed Code Context Graph builder.

The root Victor framework owns the generic graph pipeline.  Language-specific
syntax knowledge belongs in this package, where the language registry and
tree-sitter plugins already live.  This builder implements the framework
``CCGBuilderProtocol`` without importing root Victor graph types until build
time, so victor-coding remains importable in contract-only contexts.
"""

from __future__ import annotations

import hashlib
import keyword
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from victor_coding.languages.base import LanguagePlugin
from victor_coding.languages.registry import LanguageRegistry, get_language_registry

logger = logging.getLogger(__name__)

STATEMENT_CONDITION = "condition"
STATEMENT_LOOP = "loop"
STATEMENT_TRY = "try"
STATEMENT_CATCH = "catch"
STATEMENT_FINALLY = "finally"
STATEMENT_SWITCH = "switch"
STATEMENT_CASE = "case"
STATEMENT_DEFAULT = "default"
STATEMENT_ASSIGNMENT = "assignment"
STATEMENT_CALL = "call"
STATEMENT_RETURN = "return"
STATEMENT_YIELD = "yield"
STATEMENT_AWAIT = "await"
STATEMENT_THROW = "throw"
STATEMENT_FUNCTION_DEF = "function_def"
STATEMENT_CLASS_DEF = "class_def"
STATEMENT_VARIABLE_DEF = "variable_def"
STATEMENT_BLOCK = "block"
STATEMENT_EXPRESSION = "expression"

NODE_TYPE_TO_STATEMENT_TYPE: Dict[str, str] = {}


def _register(statement_type: str, *node_types: str) -> None:
    for node_type in node_types:
        NODE_TYPE_TO_STATEMENT_TYPE[node_type] = statement_type


# The node names below cover the tree-sitter grammars registered by
# victor_coding.languages.registry.  They intentionally live in victor-coding,
# not victor core, because this package owns language syntax knowledge.
_register(
    STATEMENT_CONDITION,
    "if_statement",
    "if_expression",
    "if",
    "elif_clause",
    "else_clause",
    "conditional_expression",
    "ternary_expression",
    "when_expression",
    "unless",
    # VHDL uses `elsif_statement` instead of `elif_clause`.
    "elsif_statement",
    # Verilog wraps all if/else in `conditional_statement`.
    "conditional_statement",
)
_register(
    STATEMENT_LOOP,
    "for_statement",
    "for_in_statement",
    "for_expression",
    "for",
    "enhanced_for_statement",
    "while_statement",
    "while_expression",
    "while",
    "do_statement",
    "do_while_statement",
    "loop_expression",
    "repeat_statement",
    "for_range",
    "range_clause",
    # Perl distinguishes `for(;;)` (cstyle_for_statement) from `foreach`,
    # and uses `loop_statement` for plain `while`/`until` constructs.
    "cstyle_for_statement",
    "loop_statement",
    # VHDL: `for ... loop ... end loop;` and `while ... loop ... end loop;`
    # parse as `for_loop` / `while_loop`; iterative generate constructs are
    # for_generate_statement.
    "for_loop",
    "while_loop",
    "for_generate_statement",
)
_register(
    STATEMENT_TRY,
    "try_statement",
    "try_expression",
    "try",
    "try_with_resources_statement",
)
_register(
    STATEMENT_CATCH,
    "except_clause",
    "except_block",
    "catch_clause",
    "catch_formal_parameter",
    # Additional grammar-specific catch node names surfaced by the
    # per-language CCG coverage audit:
    "rescue",  # ruby
    "rescue_block",  # elixir
    "catch_block",  # kotlin, swift
    "catch_declaration",  # c#
)
_register(STATEMENT_FINALLY, "finally_clause", "finally")
_register(
    STATEMENT_SWITCH,
    "switch_statement",
    "switch_expression",
    "match_statement",
    "match_expression",
    "when_statement",
    # haskell `case ... of` parses as a bare `match` node; go's
    # type/expression switches use distinct node names from
    # `switch_statement`.
    "match",
    "expression_switch_statement",
    "type_switch_statement",
)
_register(
    STATEMENT_CASE,
    "case_statement",
    "case_clause",
    "match_clause",
    "match_arm",
    "switch_case",
    "case",
    # Additional grammar-specific arm/case node names.
    "match_case",  # ocaml
    "case_block",  # scala
    "expression_case",  # go
    "type_case",  # go type switch
    "case_statement_alternative",  # vhdl
    "case_item",  # verilog
)
_register(STATEMENT_DEFAULT, "default_case", "switch_default", "default")
_register(
    STATEMENT_ASSIGNMENT,
    "assignment",
    "augmented_assignment",
    "named_expression",
    "assignment_expression",
    "assignment_statement",
    "compound_assignment_expr",
    "compound_assignment_expression",
    "short_var_declaration",
    "let_declaration",
    "let_condition",
    "const_item",
    "static_item",
    "variable_declaration",
    "variable_declarator",
    "lexical_declaration",
    "const_declaration",
    "property_declaration",
    "field_declaration",
    "local_declaration_statement",
    "declaration",
    "binary_operator",
)
_register(
    STATEMENT_CALL,
    "call",
    "call_expression",
    "function_call",
    "method_call",
    "method_invocation",
    "invocation_expression",
    "macro_invocation",
    "command",
    "subscript_expression",
)
_register(STATEMENT_RETURN, "return_statement", "return_expression", "return")
_register(STATEMENT_YIELD, "yield_statement", "yield_expression", "yield")
_register(STATEMENT_AWAIT, "await", "await_expression")
_register(
    STATEMENT_THROW,
    "throw_statement",
    "throw_expression",
    "raise_statement",
    "panic_expression",
)
_register(
    STATEMENT_FUNCTION_DEF,
    "function_definition",
    "async_function_definition",
    "function_declaration",
    "function_item",
    "function_signature_item",
    "function_expression",
    "arrow_function",
    "method_definition",
    "method_declaration",
    "method_spec",
    "method_elem",
    "closure_expression",
    "lambda",
    "lambda_expression",
    "anonymous_function",
    "proc",
    "subroutine",
    "constructor_declaration",
    "destructor_declaration",
    "init_declaration",
)
_register(
    STATEMENT_CLASS_DEF,
    "class_definition",
    "class_declaration",
    "interface_declaration",
    "interface_type",
    "trait_item",
    "impl_item",
    "struct_item",
    "struct_specifier",
    "union_specifier",
    "enum_item",
    "enum_declaration",
    "enum_specifier",
    "type_declaration",
    "type_alias_declaration",
    "type_item",
    "type_spec",
    "object_declaration",
    "module",
    "module_declaration",
    "mod_item",
    "namespace_definition",
    "namespace_declaration",
    "macro_definition",
)
_register(
    STATEMENT_EXPRESSION,
    "expression_statement",
    "expression_list",
    "parenthesized_expression",
    "binary_expression",
    "unary_expression",
)
_register(
    STATEMENT_BLOCK,
    "block",
    "statement_block",
    "compound_statement",
    "class_body",
    "declaration_list",
)

SCOPE_STATEMENT_TYPES = {
    STATEMENT_CONDITION,
    STATEMENT_LOOP,
    STATEMENT_TRY,
    STATEMENT_CATCH,
    STATEMENT_FINALLY,
    STATEMENT_SWITCH,
    STATEMENT_CASE,
    STATEMENT_DEFAULT,
    STATEMENT_FUNCTION_DEF,
    STATEMENT_CLASS_DEF,
    STATEMENT_BLOCK,
}

CONTROL_STATEMENT_TYPES = {
    STATEMENT_CONDITION,
    STATEMENT_LOOP,
    STATEMENT_TRY,
    STATEMENT_CATCH,
    STATEMENT_FINALLY,
    STATEMENT_SWITCH,
    STATEMENT_CASE,
    STATEMENT_DEFAULT,
}

DEFINITION_STATEMENT_TYPES = {
    STATEMENT_ASSIGNMENT,
    STATEMENT_FUNCTION_DEF,
    STATEMENT_CLASS_DEF,
    STATEMENT_VARIABLE_DEF,
}

IDENTIFIER_NODE_TYPES = {
    "identifier",
    "type_identifier",
    "field_identifier",
    "property_identifier",
    "shorthand_property_identifier",
    "simple_identifier",
    "variable_name",
    "constant",
    "name",
}

SKIP_IDENTIFIER_DESCENT_TYPES = {
    "comment",
    "string",
    "string_literal",
    "raw_string_literal",
    "interpreted_string_literal",
    "template_string",
    "string_content",
    "character",
    "char_literal",
    "character_literal",
    "number",
    "integer",
    "float",
    "integer_literal",
    "float_literal",
    "decimal_integer_literal",
}

BROAD_KEYWORDS = {
    *keyword.kwlist,
    "abstract",
    "alias",
    "and",
    "as",
    "async",
    "await",
    "become",
    "bool",
    "boolean",
    "break",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "crate",
    "data",
    "def",
    "defer",
    "del",
    "do",
    "double",
    "dyn",
    "elif",
    "else",
    "enum",
    "except",
    "export",
    "extends",
    "extern",
    "false",
    "final",
    "finally",
    "float",
    "fn",
    "for",
    "from",
    "func",
    "function",
    "global",
    "go",
    "guard",
    "if",
    "impl",
    "import",
    "in",
    "inline",
    "int",
    "interface",
    "is",
    "let",
    "loop",
    "macro",
    "match",
    "mod",
    "module",
    "mut",
    "namespace",
    "new",
    "nil",
    "none",
    "not",
    "null",
    "object",
    "operator",
    "or",
    "override",
    "package",
    "pass",
    "private",
    "protected",
    "pub",
    "public",
    "raise",
    "ref",
    "return",
    "self",
    "static",
    "struct",
    "super",
    "switch",
    "this",
    "throw",
    "throws",
    "trait",
    "true",
    "try",
    "type",
    "typeof",
    "union",
    "unsafe",
    "use",
    "using",
    "val",
    "var",
    "void",
    "where",
    "while",
    "with",
    "yield",
}


@dataclass(frozen=True)
class SymbolHint:
    """Symbol metadata extracted from a plugin tree-sitter query."""

    name: str
    symbol_type: str


@dataclass
class StatementRecord:
    """Internal statement record used while building CCG edges."""

    graph_node: Any
    ast_node: Any
    statement_type: str
    identifiers: Set[str] = field(default_factory=set)
    definitions: Set[str] = field(default_factory=set)
    calls: Set[str] = field(default_factory=set)


def _get_graph_types() -> tuple[type, type, type]:
    """Lazy import graph types from the Victor runtime package."""
    from victor.storage.graph.edge_types import EdgeType
    from victor.storage.graph.protocol import GraphEdge, GraphNode

    return GraphNode, GraphEdge, EdgeType


class PluginBackedCCGBuilder:
    """Build CCG nodes and edges using victor-coding language plugins.

    The implementation supports every registered language plugin that declares
    a tree-sitter grammar.  It emits statement nodes with CFG, CDG, and DDG
    edges and enriches nodes with plugin-derived symbol/call metadata.
    """

    def __init__(
        self,
        registry: Optional[LanguageRegistry] = None,
        auto_discover: bool = True,
    ) -> None:
        self.registry = registry or get_language_registry()
        if auto_discover:
            self._ensure_discovered()
        self._parsers: Dict[str, Any] = {}
        self._supported_language_cache: Optional[Set[str]] = None

    @property
    def supported_languages(self) -> List[str]:
        """Languages with registered plugins, grammars, AND CCG opt-in."""
        if self._supported_language_cache is None:
            self._ensure_discovered()
            languages = set()
            for language in self.registry.list_languages():
                try:
                    plugin = self.registry.get(language)
                except KeyError:
                    continue
                if not plugin.capabilities.supports_control_flow_graph:
                    continue
                parser_language = self._parser_language_for_plugin(plugin, language)
                if parser_language and self._tree_sitter_language_known(parser_language):
                    languages.add(language)
            self._supported_language_cache = languages
        return sorted(self._supported_language_cache)

    def supports_language(self, language: str) -> bool:
        """Return whether this builder can handle a language plugin.

        A plugin qualifies iff: (a) it exists, (b) its grammar is loadable,
        and (c) the plugin opts into CCG via
        ``capabilities.supports_control_flow_graph``. The capability gate
        prevents markup / build / schema / data plugins (markdown, make,
        cmake, graphql, hcl, html, css, json, yaml, toml, ini, hocon,
        xml, sql) from being asked to produce a control-flow graph they
        don't structurally have.
        """
        try:
            plugin = self._get_plugin(language)
        except KeyError:
            return False
        if not plugin.capabilities.supports_control_flow_graph:
            return False
        parser_language = self._parser_language_for_plugin(plugin, language)
        return bool(parser_language and self._tree_sitter_language_known(parser_language))

    async def build_ccg_for_file(
        self,
        file_path: Path | str,
        language: str | None = None,
    ) -> Tuple[List[Any], List[Any]]:
        """Build statement-level CFG/CDG/DDG nodes and edges for a source file."""
        path = Path(file_path)
        detected_language = language or self.registry.detect_language(path)
        if not detected_language:
            return [], []

        try:
            plugin = self._get_plugin(detected_language)
        except KeyError:
            return [], []

        parser_language = self._parser_language_for_plugin(plugin, detected_language, path)
        if not parser_language:
            return [], []

        try:
            parser = self._get_parser(parser_language)
        except Exception as exc:
            logger.debug("Tree-sitter parser unavailable for %s: %s", parser_language, exc)
            return [], []

        try:
            source_bytes = path.read_bytes()
            source_code = source_bytes.decode("utf-8", errors="ignore")
            tree = parser.parse(source_bytes)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            logger.debug("Failed to parse %s as %s: %s", path, detected_language, exc)
            return [], []

        source_lines = source_code.splitlines()
        symbol_hints = self._extract_symbol_hints(tree, parser, plugin)
        graph_nodes: List[Any] = []
        records: List[StatementRecord] = []
        canonical_language = plugin.config.name

        self._walk_statements(
            tree.root_node,
            file_path=path,
            source_bytes=source_bytes,
            source_lines=source_lines,
            language=canonical_language,
            parser_language=parser_language,
            plugin=plugin,
            symbol_hints=symbol_hints,
            graph_nodes=graph_nodes,
            records=records,
        )

        _, GraphEdge, EdgeType = _get_graph_types()
        graph_edges = []
        graph_edges.extend(self._build_cfg_edges(records, GraphEdge, EdgeType))
        graph_edges.extend(self._build_cdg_edges(records, GraphEdge, EdgeType))
        graph_edges.extend(self._build_ddg_edges(records, GraphEdge, EdgeType, str(path)))
        graph_edges = self._dedupe_edges(graph_edges)

        logger.debug(
            "Built plugin-backed CCG for %s (%s): %s nodes, %s edges",
            path,
            canonical_language,
            len(graph_nodes),
            len(graph_edges),
        )
        return graph_nodes, graph_edges

    def get_builder_info(self) -> Dict[str, Any]:
        """Return metadata useful for diagnostics and health checks."""
        return {
            "provider": "victor-coding",
            "builder": self.__class__.__name__,
            "supported_languages": self.supported_languages,
        }

    def _ensure_discovered(self) -> None:
        if not self.registry.list_languages():
            self.registry.discover_plugins()

    def _get_plugin(self, language: str) -> LanguagePlugin:
        self._ensure_discovered()
        return self.registry.get(language)

    def _tree_sitter_language_known(self, parser_language: str) -> bool:
        try:
            from victor_coding.codebase.tree_sitter_manager import LANGUAGE_MODULES
        except Exception:
            return True
        return parser_language in LANGUAGE_MODULES

    def _parser_language_for_plugin(
        self,
        plugin: LanguagePlugin,
        language: str,
        file_path: Optional[Path] = None,
    ) -> Optional[str]:
        parser_language = plugin.config.tree_sitter_language
        if not parser_language:
            return None

        if plugin.config.name == "typescript" and file_path is not None:
            if file_path.suffix.lower() == ".tsx":
                return "tsx"
        return parser_language

    def _get_parser(self, parser_language: str) -> Any:
        if parser_language in self._parsers:
            return self._parsers[parser_language]

        from victor_coding.codebase.tree_sitter_manager import get_parser

        parser = get_parser(parser_language)
        self._parsers[parser_language] = parser
        return parser

    def _extract_symbol_hints(
        self,
        tree: Any,
        parser: Any,
        plugin: LanguagePlugin,
    ) -> Dict[Tuple[int, int, str], SymbolHint]:
        hints: Dict[Tuple[int, int, str], SymbolHint] = {}
        queries = plugin.tree_sitter_queries
        if not queries.symbols:
            return hints

        try:
            from tree_sitter import Query, QueryCursor
        except Exception:
            return hints

        for pattern in queries.symbols:
            try:
                query = Query(parser.language, pattern.query)
                captures = QueryCursor(query).captures(tree.root_node)
            except Exception as exc:
                logger.debug(
                    "Symbol query failed for %s/%s: %s",
                    plugin.config.name,
                    pattern.symbol_type,
                    exc,
                )
                continue

            name_nodes = captures.get("name", [])
            definition_nodes = captures.get("def", [])
            for name_node in name_nodes:
                symbol_name = self._node_text(name_node).strip()
                if not symbol_name:
                    continue
                owner = self._find_symbol_owner(name_node, definition_nodes)
                hints[self._node_key(owner)] = SymbolHint(symbol_name, pattern.symbol_type)

        return hints

    def _find_symbol_owner(self, name_node: Any, definition_nodes: List[Any]) -> Any:
        best_owner = None
        best_span = None
        for def_node in definition_nodes:
            if (
                def_node.start_byte <= name_node.start_byte
                and def_node.end_byte >= name_node.end_byte
            ):
                span = def_node.end_byte - def_node.start_byte
                if best_span is None or span < best_span:
                    best_owner = def_node
                    best_span = span

        if best_owner is not None:
            return best_owner

        current = getattr(name_node, "parent", None)
        while current is not None:
            if self._classify_node(getattr(current, "type", "")) in {
                STATEMENT_FUNCTION_DEF,
                STATEMENT_CLASS_DEF,
            }:
                return current
            current = getattr(current, "parent", None)

        return name_node

    def _walk_statements(
        self,
        node: Any,
        *,
        file_path: Path,
        source_bytes: bytes,
        source_lines: List[str],
        language: str,
        parser_language: str,
        plugin: LanguagePlugin,
        symbol_hints: Dict[Tuple[int, int, str], SymbolHint],
        graph_nodes: List[Any],
        records: List[StatementRecord],
        parent_scope: Optional[str] = None,
    ) -> None:
        GraphNode, _, _ = _get_graph_types()
        node_type = getattr(node, "type", "")
        statement_type = self._classify_node(node_type)
        next_scope = parent_scope

        if statement_type:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            node_id = self._generate_statement_id(str(file_path), start_line, end_line, node_type)
            symbol_hint = symbol_hints.get(self._node_key(node))
            symbol_name = (
                symbol_hint.name
                if symbol_hint is not None
                else self._extract_declared_name(node, statement_type)
            )
            content_preview = self._content_preview(node, source_bytes, source_lines)
            definitions = self._extract_definitions(node, statement_type)
            identifiers = self._collect_identifiers(node)
            calls = self._extract_call_names(node)
            uses = {name for name in identifiers if name not in definitions}
            visibility = self._determine_visibility(symbol_name, content_preview)

            metadata: Dict[str, Any] = {
                "node_type": node_type,
                "content_preview": content_preview[:160],
                "source": "victor-coding-language-plugin",
                "language_plugin": plugin.config.name,
                "language_display_name": plugin.config.display_name,
                "tree_sitter_language": parser_language,
                "definitions": sorted(definitions)[:25],
                "uses": sorted(uses)[:25],
            }
            if symbol_hint is not None:
                metadata["symbol_type"] = symbol_hint.symbol_type
            if symbol_name:
                metadata["symbol_name"] = symbol_name
            if calls:
                metadata["calls"] = sorted(calls)[:25]

            graph_node = GraphNode(
                node_id=node_id,
                type="statement",
                name=self._statement_name(node_type, statement_type, symbol_name, start_line),
                file=str(file_path),
                line=start_line,
                end_line=end_line,
                lang=language,
                ast_kind=node_type,
                scope_id=parent_scope,
                statement_type=statement_type,
                visibility=visibility,
                metadata=metadata,
            )
            graph_nodes.append(graph_node)
            records.append(
                StatementRecord(
                    graph_node=graph_node,
                    ast_node=node,
                    statement_type=statement_type,
                    identifiers=identifiers,
                    definitions=definitions,
                    calls=calls,
                )
            )

            if statement_type in SCOPE_STATEMENT_TYPES:
                next_scope = node_id

        for child in getattr(node, "children", []):
            child_type = getattr(child, "type", "")
            if child_type in SKIP_IDENTIFIER_DESCENT_TYPES:
                continue
            self._walk_statements(
                child,
                file_path=file_path,
                source_bytes=source_bytes,
                source_lines=source_lines,
                language=language,
                parser_language=parser_language,
                plugin=plugin,
                symbol_hints=symbol_hints,
                graph_nodes=graph_nodes,
                records=records,
                parent_scope=next_scope,
            )

    def _classify_node(self, node_type: str) -> Optional[str]:
        return NODE_TYPE_TO_STATEMENT_TYPE.get(node_type)

    def _build_cfg_edges(self, records: List[StatementRecord], GraphEdge: type, EdgeType: type):
        edges = []
        records_by_scope: Dict[Optional[str], List[StatementRecord]] = {}
        for record in records:
            records_by_scope.setdefault(record.graph_node.scope_id, []).append(record)

        for scoped_records in records_by_scope.values():
            ordered = sorted(
                scoped_records,
                key=lambda record: (
                    record.graph_node.line or 0,
                    record.graph_node.end_line or 0,
                    record.graph_node.node_id,
                ),
            )
            for current, next_record in zip(ordered, ordered[1:], strict=False):
                if current.graph_node.node_id == next_record.graph_node.node_id:
                    continue
                edges.append(
                    GraphEdge(
                        src=current.graph_node.node_id,
                        dst=next_record.graph_node.node_id,
                        type=self._cfg_edge_type(current, next_record, EdgeType),
                        weight=1.0,
                        metadata={
                            "file": current.graph_node.file,
                            "source": "victor-coding-language-plugin",
                        },
                    )
                )

        record_by_id = {record.graph_node.node_id: record for record in records}
        for record in records:
            parent = record_by_id.get(record.graph_node.scope_id)
            if parent is None or parent.statement_type not in CONTROL_STATEMENT_TYPES:
                continue
            edge_type = self._control_entry_edge_type(parent, EdgeType)
            edges.append(
                GraphEdge(
                    src=parent.graph_node.node_id,
                    dst=record.graph_node.node_id,
                    type=edge_type,
                    weight=1.0,
                    metadata={
                        "file": record.graph_node.file,
                        "scope": parent.statement_type,
                        "source": "victor-coding-language-plugin",
                    },
                )
            )

        return edges

    def _build_cdg_edges(self, records: List[StatementRecord], GraphEdge: type, EdgeType: type):
        edges = []
        record_by_id = {record.graph_node.node_id: record for record in records}
        for record in records:
            parent = record_by_id.get(record.graph_node.scope_id)
            if parent is None or parent.statement_type not in CONTROL_STATEMENT_TYPES:
                continue
            edges.append(
                GraphEdge(
                    src=parent.graph_node.node_id,
                    dst=record.graph_node.node_id,
                    type=self._cdg_edge_type(parent, EdgeType),
                    weight=1.0,
                    metadata={
                        "file": record.graph_node.file,
                        "scope": parent.statement_type,
                        "source": "victor-coding-language-plugin",
                    },
                )
            )
        return edges

    def _build_ddg_edges(
        self,
        records: List[StatementRecord],
        GraphEdge: type,
        EdgeType: type,
        file_path: str,
    ):
        edges = []
        latest_definition: Dict[str, StatementRecord] = {}
        ordered = sorted(
            records,
            key=lambda record: (
                record.graph_node.line or 0,
                record.graph_node.end_line or 0,
                record.graph_node.node_id,
            ),
        )

        for record in ordered:
            uses = record.identifiers - record.definitions
            for name in sorted(uses):
                defining_record = latest_definition.get(name)
                if defining_record is None:
                    continue
                if defining_record.graph_node.node_id == record.graph_node.node_id:
                    continue
                edges.append(
                    GraphEdge(
                        src=defining_record.graph_node.node_id,
                        dst=record.graph_node.node_id,
                        type=EdgeType.DDG_DEF_USE,
                        weight=1.0,
                        metadata={
                            "variable": name,
                            "file": file_path,
                            "source": "victor-coding-language-plugin",
                        },
                    )
                )

            for name in sorted(record.definitions):
                latest_definition[name] = record

        return edges

    def _cfg_edge_type(
        self, current: StatementRecord, next_record: StatementRecord, EdgeType: type
    ):
        if current.statement_type == STATEMENT_CONDITION:
            return EdgeType.CFG_TRUE_BRANCH
        if current.statement_type == STATEMENT_LOOP:
            return EdgeType.CFG_LOOP_ENTRY
        if current.statement_type in {STATEMENT_SWITCH, STATEMENT_CASE}:
            return EdgeType.CFG_CASE
        if current.statement_type == STATEMENT_TRY:
            return EdgeType.CFG_EXCEPTION
        if current.statement_type == STATEMENT_CATCH:
            return EdgeType.CFG_CATCH
        if current.statement_type == STATEMENT_FINALLY:
            return EdgeType.CFG_FINALLY
        if current.statement_type == STATEMENT_RETURN:
            return EdgeType.CFG_RETURN
        if current.statement_type == STATEMENT_THROW:
            return EdgeType.CFG_EXCEPTION
        return EdgeType.CFG_SUCCESSOR

    def _control_entry_edge_type(self, parent: StatementRecord, EdgeType: type):
        if parent.statement_type == STATEMENT_LOOP:
            return EdgeType.CFG_LOOP_ENTRY
        if parent.statement_type in {STATEMENT_SWITCH, STATEMENT_CASE}:
            return EdgeType.CFG_CASE
        if parent.statement_type == STATEMENT_TRY:
            return EdgeType.CFG_EXCEPTION
        if parent.statement_type == STATEMENT_CATCH:
            return EdgeType.CFG_CATCH
        if parent.statement_type == STATEMENT_FINALLY:
            return EdgeType.CFG_FINALLY
        return EdgeType.CFG_TRUE_BRANCH

    def _cdg_edge_type(self, parent: StatementRecord, EdgeType: type):
        if parent.statement_type == STATEMENT_LOOP:
            return EdgeType.CDG_LOOP
        if parent.statement_type in {STATEMENT_SWITCH, STATEMENT_CASE}:
            return EdgeType.CDG_CASE
        return EdgeType.CDG

    def _dedupe_edges(self, edges: Iterable[Any]) -> List[Any]:
        deduped = []
        seen = set()
        for edge in edges:
            key = (
                edge.src,
                edge.dst,
                str(edge.type),
                tuple(sorted((edge.metadata or {}).items())),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(edge)
        return deduped

    def _extract_definitions(self, node: Any, statement_type: str) -> Set[str]:
        if statement_type not in DEFINITION_STATEMENT_TYPES:
            return set()

        if statement_type in {STATEMENT_FUNCTION_DEF, STATEMENT_CLASS_DEF}:
            names = set()
            declared_name = self._extract_declared_name(node, statement_type)
            if declared_name:
                names.add(declared_name)
            names.update(self._extract_parameter_names(node))
            return {name for name in names if not self._is_keyword(name)}

        target_nodes = []
        for field_name in (
            "left",
            "lhs",
            "name",
            "pattern",
            "declarator",
            "declarators",
            "target",
            "key",
            "value",
        ):
            child = self._child_by_field_name(node, field_name)
            if child is not None:
                target_nodes.append(child)

        if not target_nodes:
            target_nodes = list(getattr(node, "children", [])[:1])

        names = set()
        for target in target_nodes:
            names.update(self._collect_identifiers(target))
        return names

    def _extract_declared_name(self, node: Any, statement_type: str) -> Optional[str]:
        for field_name in ("name", "declarator", "declarators", "type", "path"):
            child = self._child_by_field_name(node, field_name)
            name = self._first_identifier(child)
            if name:
                return name

        if statement_type in {STATEMENT_FUNCTION_DEF, STATEMENT_CLASS_DEF}:
            return self._first_identifier(node)
        return None

    def _extract_parameter_names(self, node: Any) -> Set[str]:
        names = set()
        for field_name in ("parameters", "parameter", "arguments"):
            child = self._child_by_field_name(node, field_name)
            if child is not None:
                names.update(self._collect_identifiers(child))
        return names

    def _extract_call_names(self, node: Any) -> Set[str]:
        names = set()

        def walk(current: Any) -> None:
            current_type = getattr(current, "type", "")
            if current_type in SKIP_IDENTIFIER_DESCENT_TYPES:
                return
            if self._classify_node(current_type) == STATEMENT_CALL:
                callee = self._call_name(current)
                if callee:
                    names.add(callee)
            for child in getattr(current, "children", []):
                walk(child)

        walk(node)
        return names

    def _call_name(self, node: Any) -> Optional[str]:
        for field_name in ("function", "method", "name", "command"):
            child = self._child_by_field_name(node, field_name)
            name = self._qualified_identifier(child)
            if name:
                return name
        return self._qualified_identifier(node)

    def _collect_identifiers(self, node: Any) -> Set[str]:
        identifiers = set()
        if node is None:
            return identifiers

        def walk(current: Any) -> None:
            current_type = getattr(current, "type", "")
            if current_type in SKIP_IDENTIFIER_DESCENT_TYPES:
                return
            if current_type in IDENTIFIER_NODE_TYPES:
                text = self._node_text(current).strip()
                if text and not self._is_keyword(text):
                    identifiers.add(text)
            for child in getattr(current, "children", []):
                walk(child)

        walk(node)
        return identifiers

    def _qualified_identifier(self, node: Any) -> Optional[str]:
        if node is None:
            return None
        if getattr(node, "type", "") in IDENTIFIER_NODE_TYPES:
            text = self._node_text(node).strip()
            return None if self._is_keyword(text) else text

        pieces = []
        for child in getattr(node, "children", []):
            if getattr(child, "type", "") in IDENTIFIER_NODE_TYPES:
                text = self._node_text(child).strip()
                if text and not self._is_keyword(text):
                    pieces.append(text)

        if not pieces:
            return self._first_identifier(node)
        return ".".join(pieces[-3:])

    def _first_identifier(self, node: Any) -> Optional[str]:
        if node is None:
            return None
        if getattr(node, "type", "") in IDENTIFIER_NODE_TYPES:
            text = self._node_text(node).strip()
            return None if self._is_keyword(text) else text
        for child in getattr(node, "children", []):
            name = self._first_identifier(child)
            if name:
                return name
        return None

    def _child_by_field_name(self, node: Any, field_name: str) -> Optional[Any]:
        if node is None or not hasattr(node, "child_by_field_name"):
            return None
        try:
            return node.child_by_field_name(field_name)
        except Exception:
            return None

    def _content_preview(self, node: Any, source_bytes: bytes, source_lines: List[str]) -> str:
        text = ""
        try:
            text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
        except Exception:
            text = ""
        if not text:
            start_line = node.start_point[0]
            if 0 <= start_line < len(source_lines):
                text = source_lines[start_line]
        return " ".join(text.strip().split())

    def _determine_visibility(
        self, symbol_name: Optional[str], content_preview: str
    ) -> Optional[str]:
        lowered = content_preview.lower()
        if "private" in lowered:
            return "private"
        if "protected" in lowered:
            return "protected"
        if "public" in lowered or lowered.startswith("pub ") or " pub " in lowered:
            return "public"
        if symbol_name and symbol_name.startswith("_") and not symbol_name.startswith("__"):
            return "private"
        return None

    def _statement_name(
        self,
        node_type: str,
        statement_type: str,
        symbol_name: Optional[str],
        start_line: int,
    ) -> str:
        if symbol_name:
            return f"{statement_type}:{symbol_name}:{start_line}"
        return f"{node_type}:{start_line}"

    def _node_text(self, node: Any) -> str:
        text = getattr(node, "text", b"")
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return str(text)

    def _node_key(self, node: Any) -> Tuple[int, int, str]:
        return (
            int(getattr(node, "start_byte", 0)),
            int(getattr(node, "end_byte", 0)),
            str(getattr(node, "type", "")),
        )

    def _generate_statement_id(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        node_type: str,
    ) -> str:
        id_string = f"{file_path}:{start_line}:{end_line}:{node_type}"
        return hashlib.sha256(id_string.encode("utf-8")).hexdigest()[:16]

    def _is_keyword(self, name: str) -> bool:
        return name.lower() in BROAD_KEYWORDS
