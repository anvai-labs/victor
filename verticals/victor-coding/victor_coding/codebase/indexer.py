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

"""Codebase indexing for intelligent code awareness.

This is the HIGHEST PRIORITY feature to match Claude Code capabilities.

Supports both keyword search and semantic search (with embeddings).

Features:
- AST-based symbol extraction
- Keyword and semantic search
- File watching for automatic staleness detection
- Lazy reindexing when stale
- Incremental updates for changed files
"""

import ast
import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field
from tree_sitter import Query

from victor_coding.codebase.graph.protocol import GraphEdge, GraphNode
from victor_coding.codebase.tree_sitter_extractor import TreeSitterExtractor
from victor_coding.codebase.unified_extractor import (
    UnifiedSymbolExtractor,
    EnrichedSymbol,
)
from victor_coding.languages.registry import get_language_registry
from victor_coding.languages.tiers import get_tier, LanguageTier
from victor_coding.codebase.graph.registry import create_graph_store
from victor_coding.codebase.symbol_resolver import SymbolResolver
from victor_contracts.utils.ast_helpers import (
    build_signature,
    extract_base_classes,
    is_stdlib_module as _is_stdlib_module,
)

if TYPE_CHECKING:
    from victor_coding.codebase.embeddings.base import BaseEmbeddingProvider
    from victor_coding.codebase.graph.protocol import GraphStoreProtocol


logger = logging.getLogger(__name__)

# STDLIB_MODULES imported from victor.core.utils.ast_helpers


# =============================================================================
# PRIMITIVE / CONTAINER TYPES (exclude from COMPOSED_OF phantom nodes)
# =============================================================================
# These are language-builtin or stdlib types that would create meaningless
# composition edges (e.g. every struct with a String field → phantom node).
_PRIMITIVE_TYPES = frozenset(
    {
        # Rust
        "String",
        "str",
        "Vec",
        "Option",
        "Result",
        "Box",
        "Arc",
        "Rc",
        "HashMap",
        "HashSet",
        "BTreeMap",
        "BTreeSet",
        "VecDeque",
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "isize",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
        "f32",
        "f64",
        "bool",
        "char",
        "PathBuf",
        "Duration",
        "Instant",
        "Cow",
        # JS/TS
        "string",
        "number",
        "boolean",
        "any",
        "void",
        "null",
        "undefined",
        "Array",
        "Map",
        "Set",
        "Promise",
        "Date",
        "RegExp",
        "Error",
        "Record",
        "Partial",
        "Required",
        "Readonly",
        # Java/C#
        "int",
        "long",
        "float",
        "double",
        "byte",
        "short",
        "Integer",
        "Long",
        "Float",
        "Double",
        "Boolean",
        "List",
        "ArrayList",
        "LinkedList",
        "Object",
    }
)


# _is_stdlib_module imported from victor.core.utils.ast_helpers


# =============================================================================
# PARALLEL INDEXING SUPPORT
# =============================================================================
# Module-level function for ProcessPoolExecutor (must be picklable)
# Processes a single file and returns extracted data as a dictionary.
# This enables 3-8x speedup on multi-core systems.


def _get_plugin_query(language: str, field: str) -> Optional[str]:
    """Get a tree-sitter query from the language plugin, or None if unavailable.

    Safe to call from subprocesses — imports the registry on demand and
    discovers plugins if needed.
    """
    try:
        from victor_coding.languages.registry import get_language_registry

        registry = get_language_registry()
        if not registry._plugins:
            registry.discover_plugins()
        plugin = registry.get(language)
        if plugin:
            value = getattr(plugin.tree_sitter_queries, field, None)
            if value:
                return value
    except Exception:
        pass
    return None


def _get_plugin_enclosing_scopes(language: str) -> List[Tuple[str, str]]:
    """Get enclosing scope definitions from language plugin, or empty list."""
    try:
        from victor_coding.languages.registry import get_language_registry

        registry = get_language_registry()
        if not registry._plugins:
            registry.discover_plugins()
        plugin = registry.get(language)
        if plugin and plugin.tree_sitter_queries.enclosing_scopes:
            return plugin.tree_sitter_queries.enclosing_scopes
    except Exception:
        pass
    return []


# Tree-sitter import queries for non-Python languages (used by parallel path).
_PARALLEL_IMPORT_QUERIES: Dict[str, str] = {
    "javascript": """
        (import_statement source: (string) @source)
        (call_expression
            function: (identifier) @_fn
            arguments: (arguments (string) @source)
            (#eq? @_fn "require"))
    """,
    "typescript": """
        (import_statement source: (string) @source)
        (call_expression
            function: (identifier) @_fn
            arguments: (arguments (string) @source)
            (#eq? @_fn "require"))
    """,
    "rust": """
        (use_declaration argument: (_) @source)
    """,
    "go": """
        (import_spec path: (interpreted_string_literal) @source)
    """,
    "java": """
        (import_declaration (scoped_identifier) @source)
    """,
}


def _process_file_parallel(
    file_path_str: str,
    root_str: str,
    language: str,
) -> Optional[Dict[str, Any]]:
    """Process a single file for indexing in a subprocess.

    This is a module-level function (not a method) so it can be pickled
    for use with ProcessPoolExecutor.

    Uses plugin-first query lookup for all extraction types, falling back
    to static dictionaries for languages not yet migrated to plugins.

    Args:
        file_path_str: Absolute path to the file
        root_str: Absolute path to the codebase root
        language: Detected language for the file

    Returns:
        Dictionary with extracted file data, or None on error
    """
    import ast as py_ast
    from pathlib import Path

    file_path = Path(file_path_str)
    root = Path(root_str)

    try:
        stat = file_path.stat()
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return None

    # Extract symbols using tree-sitter
    symbols_data: List[Dict[str, Any]] = []
    call_edges: List[Tuple[str, str]] = []
    imports: List[str] = []
    inherit_edges: List[Tuple[str, str]] = []
    implements_edges: List[Tuple[str, str]] = []
    compose_edges: List[Tuple[str, str]] = []
    references: List[str] = []

    # Tree-sitter symbol extraction
    try:
        from victor_coding.codebase.tree_sitter_manager import get_parser
        from tree_sitter import Query, QueryCursor

        parser = get_parser(language)
        if parser:
            content_bytes = file_path.read_bytes()
            tree = parser.parse(content_bytes)

            # Symbol extraction — plugin-first, then static fallback
            # Uses @name capture for symbol name/start_line, @def capture for end_line
            query_defs = SYMBOL_QUERIES.get(language, [])
            if not query_defs:
                # Try language plugin for symbol queries
                try:
                    from victor_coding.languages.registry import get_language_registry

                    _reg = get_language_registry()
                    if not _reg._plugins:
                        _reg.discover_plugins()
                    _plugin = _reg.get(language)
                    if _plugin and _plugin.tree_sitter_queries.symbols:
                        query_defs = [
                            (qp.symbol_type, qp.query) for qp in _plugin.tree_sitter_queries.symbols
                        ]
                except Exception:
                    pass
            for sym_type, query_src in query_defs:
                try:
                    query = Query(parser.language, query_src)
                    cursor = QueryCursor(query)
                    captures_dict = cursor.captures(tree.root_node)

                    # Get @name and @def captures
                    name_nodes = captures_dict.get("name", [])
                    def_nodes = captures_dict.get("def", [])

                    # Map def nodes by start line for matching
                    def_by_start_line = {}
                    for def_node in def_nodes:
                        def_by_start_line[def_node.start_point[0]] = def_node

                    for node in name_nodes:
                        text = node.text.decode("utf-8", errors="ignore")
                        if text:
                            name_line = node.start_point[0]
                            # Default end_line to name node's end
                            end_line = node.end_point[0] + 1

                            # Find matching @def node for proper end_line
                            for def_start, def_node in def_by_start_line.items():
                                if def_start <= name_line <= def_node.end_point[0]:
                                    end_line = def_node.end_point[0] + 1
                                    break

                            symbols_data.append(
                                {
                                    "name": text,
                                    "type": sym_type,
                                    "file_path": str(file_path.relative_to(root)),
                                    "line_number": name_line + 1,
                                    "end_line": end_line,
                                }
                            )
                except Exception:
                    continue

            # Call edge extraction — plugin-first, then static fallback
            call_query_src = _get_plugin_query(language, "calls")
            if not call_query_src:
                call_query_src = CALL_QUERIES.get(language)
            if call_query_src:
                try:
                    query = Query(parser.language, call_query_src)
                    cursor = QueryCursor(query)
                    captures_dict = cursor.captures(tree.root_node)

                    callee_nodes = captures_dict.get("callee", [])
                    for node in callee_nodes:
                        callee = node.text.decode("utf-8", errors="ignore")
                        # Find enclosing function as caller
                        caller = _find_enclosing_function(node, language)
                        if caller and callee and callee not in {"function", caller}:
                            call_edges.append((caller, callee))
                except Exception:
                    pass

            # Reference extraction — plugin-first, then static fallback
            ref_query_src = _get_plugin_query(language, "references")
            if not ref_query_src:
                ref_query_src = REFERENCE_QUERIES.get(language)
            if ref_query_src:
                try:
                    query = Query(parser.language, ref_query_src)
                    cursor = QueryCursor(query)
                    captures_dict = cursor.captures(tree.root_node)
                    for _capture_name, nodes in captures_dict.items():
                        for node in nodes:
                            ref = node.text.decode("utf-8", errors="ignore")
                            if ref and len(ref) > 1:  # Skip single-char identifiers
                                references.append(ref)
                except Exception:
                    pass

            # Inheritance extraction — plugin-first, then static fallback
            inherit_query_src = _get_plugin_query(language, "inheritance")
            if not inherit_query_src:
                inherit_query_src = INHERITS_QUERIES.get(language)
            if inherit_query_src:
                try:
                    query = Query(parser.language, inherit_query_src)
                    cursor = QueryCursor(query)
                    for _pat_idx, cap_dict in cursor.matches(tree.root_node):
                        child_nodes = cap_dict.get("child", [])
                        base_nodes = cap_dict.get("base", [])
                        if child_nodes and base_nodes:
                            child_text = child_nodes[0].text.decode("utf-8", errors="ignore")
                            base_text = base_nodes[0].text.decode("utf-8", errors="ignore")
                            if child_text and base_text:
                                inherit_edges.append((child_text, base_text))
                except Exception:
                    pass

            # Implements extraction — plugin-first, then static fallback
            impl_query_src = _get_plugin_query(language, "implements")
            if not impl_query_src:
                impl_query_src = IMPLEMENTS_QUERIES.get(language)
            if impl_query_src:
                try:
                    query = Query(parser.language, impl_query_src)
                    cursor = QueryCursor(query)
                    for _pat_idx, cap_dict in cursor.matches(tree.root_node):
                        child_nodes = cap_dict.get("child", [])
                        iface_nodes = cap_dict.get("interface", []) or cap_dict.get("base", [])
                        if child_nodes and iface_nodes:
                            child_text = child_nodes[0].text.decode("utf-8", errors="ignore")
                            iface_text = iface_nodes[0].text.decode("utf-8", errors="ignore")
                            if child_text and iface_text:
                                implements_edges.append((child_text, iface_text))
                except Exception:
                    pass

            # Composition extraction — plugin-first, then static fallback
            comp_query_src = _get_plugin_query(language, "composition")
            if not comp_query_src:
                comp_query_src = COMPOSITION_QUERIES.get(language)
            if comp_query_src:
                try:
                    query = Query(parser.language, comp_query_src)
                    cursor = QueryCursor(query)
                    for _pat_idx, cap_dict in cursor.matches(tree.root_node):
                        owner_nodes = cap_dict.get("owner", [])
                        type_nodes = cap_dict.get("type", [])
                        if owner_nodes and type_nodes:
                            owner_text = owner_nodes[0].text.decode("utf-8", errors="ignore")
                            type_text = type_nodes[0].text.decode("utf-8", errors="ignore")
                            if owner_text and type_text:
                                compose_edges.append((owner_text, type_text))
                except Exception:
                    pass

            # Import extraction for non-Python (tree-sitter based)
            if language != "python":
                import_query_src = _PARALLEL_IMPORT_QUERIES.get(language)
                if import_query_src:
                    try:
                        query = Query(parser.language, import_query_src)
                        cursor = QueryCursor(query)
                        captures_dict = cursor.captures(tree.root_node)
                        for node in captures_dict.get("source", []):
                            text = node.text.decode("utf-8", errors="ignore")
                            cleaned = text.strip("'\"")
                            if cleaned:
                                imports.append(cleaned)
                    except Exception:
                        pass
    except Exception:
        pass

    # Python-specific: extract imports via ast (more reliable than tree-sitter)
    if language == "python":
        try:
            from victor_contracts.utils.ast_helpers import (
                extract_base_classes as _extract_bases,
            )

            tree = py_ast.parse(content, filename=file_path_str)
            for node in py_ast.walk(tree):
                if isinstance(node, py_ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, py_ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Extract inheritance from AST (more complete than tree-sitter)
            for node in py_ast.walk(tree):
                if isinstance(node, py_ast.ClassDef):
                    for base_name in _extract_bases(node):
                        inherit_edges.append((node.name, base_name))
        except Exception:
            pass

    return {
        "path": str(file_path.relative_to(root)),
        "language": language,
        "symbols": symbols_data,
        "imports": imports,
        "call_edges": call_edges,
        "inherit_edges": inherit_edges,
        "implements_edges": implements_edges,
        "compose_edges": compose_edges,
        "references": list(set(references)),  # Dedupe
        "last_modified": stat.st_mtime,
        "size": stat.st_size,
        "lines": content.count("\n") + 1,
    }


def _find_enclosing_function(node: Any, language: str) -> Optional[str]:
    """Find the enclosing function name for a node.

    Helper for parallel processing - walks up the tree to find parent function.
    Uses plugin enclosing_scopes first, then falls back to static map.
    """
    # PRIMARY: Get enclosing scope config from language plugin
    plugin_scopes = _get_plugin_enclosing_scopes(language)
    if plugin_scopes:
        current = node.parent
        while current:
            for node_type, field_name in plugin_scopes:
                if current.type == node_type:
                    field = current.child_by_field_name(field_name)
                    if field:
                        # For C++ function_declarator, drill into nested declarator
                        if field.type == "function_declarator":
                            inner = field.child_by_field_name("declarator")
                            if inner:
                                field = inner
                        return field.text.decode("utf-8", errors="ignore")
            current = current.parent
        return None

    # FALLBACK: Static enclosing type map
    enclosing_types = {
        "python": ("function_definition",),
        "javascript": ("function_declaration", "method_definition", "arrow_function"),
        "typescript": ("function_declaration", "method_definition", "arrow_function"),
        "go": ("function_declaration", "method_declaration"),
        "java": ("method_declaration",),
        "cpp": ("function_definition",),
    }

    types = enclosing_types.get(language, ("function_definition",))
    current = node.parent
    while current:
        if current.type in types:
            # Find the name child
            for child in current.children:
                if child.type in (
                    "identifier",
                    "property_identifier",
                    "name",
                    "field_identifier",
                ):
                    return child.text.decode("utf-8", errors="ignore")
        current = current.parent
    return None


# =============================================================================
# LEGACY QUERY DICTIONARIES
# =============================================================================
# These hardcoded dictionaries are LEGACY and will be DEPRECATED.
# The new plugin-based architecture (victor/languages/plugins/) provides
# these queries via TreeSitterQueries in each LanguagePlugin.
#
# Migration path:
# 1. Use LanguageRegistry.get(language).tree_sitter_queries for new code
# 2. These dicts remain for backward compatibility during transition
# 3. Eventually remove once all callers migrate to plugin-based approach
#
# See: victor/languages/plugins/*.py for the new query definitions
# See: victor/codebase/tree_sitter_extractor.py for the new unified extractor
# =============================================================================

REFERENCE_QUERIES: Dict[str, str] = {
    "python": """
        (call function: (identifier) @name)
        (call function: (attribute attribute: (identifier) @name))
        (attribute object: (_) attribute: (identifier) @name)
        (identifier) @name
    """,
    "javascript": """
        (call_expression function: (identifier) @name)
        (call_expression function: (member_expression property: (property_identifier) @name))
        (member_expression property: (property_identifier) @name)
        (new_expression constructor: (identifier) @name)
        (identifier) @name
    """,
    "typescript": """
        (call_expression function: (identifier) @name)
        (call_expression function: (member_expression property: (property_identifier) @name))
        (member_expression property: (property_identifier) @name)
        (new_expression constructor: (identifier) @name)
        (identifier) @name
    """,
    "java": """
        (method_invocation name: (identifier) @name)
        (method_invocation object: (identifier) @name)
        (field_access field: (identifier) @name)
    """,
    "go": """
        (call_expression function: (identifier) @name)
        (call_expression function: (selector_expression field: (field_identifier) @name))
        (selector_expression field: (field_identifier) @name)
        (identifier) @name
    """,
}

# Map file extensions to tree-sitter language ids
EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".java": "java",
    ".json": "config-json",
    ".yaml": "config-yaml",
    ".yml": "config-yaml",
    ".toml": "config-toml",
    ".ini": "config-ini",
    ".properties": "config-properties",
    ".conf": "config-hocon",
    ".hocon": "config-hocon",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
}

# Tree-sitter symbol queries per language for lightweight multi-language graph capture.
# Uses @def capture for end_line (function body boundaries) and @name for symbol name.
SYMBOL_QUERIES: Dict[str, List[tuple[str, str]]] = {
    "python": [
        ("class", "(class_definition name: (identifier) @name) @def"),
        ("function", "(function_definition name: (identifier) @name) @def"),
    ],
    "javascript": [
        ("class", "(class_declaration name: (identifier) @name) @def"),
        ("function", "(function_declaration name: (identifier) @name) @def"),
        ("function", "(method_definition name: (property_identifier) @name) @def"),
        (
            "function",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function))) @def",
        ),
        (
            "function",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression))) @def",
        ),
        (
            "function",
            "(assignment_expression left: (identifier) @name right: (arrow_function)) @def",
        ),
    ],
    "typescript": [
        ("class", "(class_declaration name: (identifier) @name) @def"),
        ("function", "(function_declaration name: (identifier) @name) @def"),
        ("function", "(method_signature name: (property_identifier) @name) @def"),
        ("function", "(method_definition name: (property_identifier) @name) @def"),
        (
            "function",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function))) @def",
        ),
        (
            "function",
            "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression))) @def",
        ),
        (
            "function",
            "(assignment_expression left: (identifier) @name right: (arrow_function)) @def",
        ),
    ],
    "go": [
        ("function", "(function_declaration name: (identifier) @name) @def"),
        ("function", "(method_declaration name: (field_identifier) @name) @def"),
        ("class", "(type_declaration (type_spec name: (type_identifier) @name)) @def"),
    ],
    "java": [
        ("class", "(class_declaration name: (identifier) @name) @def"),
        ("class", "(interface_declaration name: (identifier) @name) @def"),
        ("function", "(method_declaration name: (identifier) @name) @def"),
    ],
    "cpp": [
        ("class", "(class_specifier name: (type_identifier) @name) @def"),
        (
            "function",
            "(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def",
        ),
        (
            "function",
            "(function_definition declarator: (function_declarator declarator: (field_identifier) @name)) @def",
        ),
    ],
}

INHERITS_QUERIES: Dict[str, str] = {
    "python": """
        (class_definition
            name: (identifier) @child
            superclasses: (argument_list (identifier) @base))
    """,
    "javascript": """
        (class_declaration
            name: (identifier) @child
            (class_heritage (identifier) @base))
    """,
    "typescript": """
        (class_declaration
            name: (identifier) @child
            (class_heritage (identifier) @base))
    """,
    "java": """
        (class_declaration
            name: (identifier) @child
            super_classes: (superclass (type_identifier) @base))
    """,
    "cpp": """
        (class_specifier
            name: (type_identifier) @child
            (base_class_clause (base_class (type_identifier) @base))
        )
    """,
}

IMPLEMENTS_QUERIES: Dict[str, str] = {
    "typescript": """
        (class_declaration
            name: (type_identifier) @child
            (class_heritage
                (implements_clause (type_identifier) @interface)))
    """,
    "java": """
        (class_declaration
            name: (identifier) @child
            interfaces: (super_interfaces (type_list (type_identifier) @interface)))
        (interface_declaration
            name: (identifier) @child
            interfaces: (super_interfaces (type_list (type_identifier) @interface)))
    """,
    "cpp": """
        (class_specifier
            name: (type_identifier) @child
            (base_class_clause (base_class (type_identifier) @base))
        )
    """,
}

COMPOSITION_QUERIES: Dict[str, str] = {
    "javascript": """
        (class_declaration
            name: (identifier) @owner
            body: (class_body
                (method_definition
                    body: (statement_block
                        (expression_statement
                            (assignment_expression
                                left: (member_expression object: (this) property: (property_identifier))
                                right: (new_expression constructor: (identifier) @type)))))))
    """,
    "typescript": """
        (class_declaration
            name: (identifier) @owner
            body: (class_body
                (field_definition
                    type: (type_annotation (type_identifier) @type))
                (public_field_definition
                    type: (type_annotation (type_identifier) @type))
                (method_definition
                    body: (statement_block
                        (expression_statement
                            (assignment_expression
                                left: (member_expression object: (this) property: (property_identifier))
                                right: (new_expression constructor: (identifier) @type)))))))
    """,
    "go": """
        (type_declaration
            (type_spec
                name: (type_identifier) @owner
                type: (struct_type
                    (field_declaration
                        type: (type_identifier) @type))))
    """,
    "java": """
        (class_declaration
            name: (identifier) @owner
            body: (class_body
                (field_declaration
                    type: (type_identifier) @type)))
    """,
    "cpp": """
        (class_specifier
            name: (type_identifier) @owner
            body: (field_declaration_list
                (field_declaration
                    type: (type_identifier) @type)))
    """,
}

# Tree-sitter call queries (callee only) for multi-language call/reference edges.
CALL_QUERIES: Dict[str, str] = {
    "python": """
        (call function: (identifier) @callee)
        (call function: (attribute attribute: (identifier) @callee))
    """,
    "javascript": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (member_expression property: (property_identifier) @callee))
        (call_expression function: (subscript_expression index: (property_identifier) @callee))
        (new_expression constructor: (identifier) @callee)
    """,
    "typescript": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (member_expression property: (property_identifier) @callee))
        (call_expression function: (subscript_expression index: (property_identifier) @callee))
        (new_expression constructor: (identifier) @callee)
    """,
    "go": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (selector_expression field: (field_identifier) @callee))
        (type_conversion_expression type: (type_identifier) @callee)
    """,
    "java": """
        (method_invocation name: (identifier) @callee)
        (object_creation_expression type: (type_identifier) @callee)
        (super_method_invocation name: (identifier) @callee)
    """,
    "cpp": """
        (call_expression function: (identifier) @callee)
        (call_expression function: (field_expression field: (field_identifier) @callee))
        (new_expression type: (type_identifier) @callee)
    """,
}

# Mapping of function/method node types to name field for caller resolution.
ENCLOSING_NAME_FIELDS: Dict[str, List[tuple[str, str]]] = {
    "python": [
        ("function_definition", "name"),
        ("class_definition", "name"),
    ],
    "javascript": [
        ("function_declaration", "name"),
        ("method_definition", "name"),
        ("class_declaration", "name"),  # used for Class.method combination
    ],
    "typescript": [
        ("function_declaration", "name"),
        ("method_definition", "name"),
        ("method_signature", "name"),
        ("class_declaration", "name"),
    ],
    "go": [
        ("function_declaration", "name"),
        ("method_declaration", "name"),
    ],
    "java": [
        ("method_declaration", "name"),
        ("class_declaration", "name"),
        ("interface_declaration", "name"),
    ],
    "cpp": [
        ("function_definition", "declarator"),
        ("class_specifier", "name"),
    ],
}


# Try to import watchdog for file watching
try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object


# Module-level function for ProcessPoolExecutor (must be picklable)
def _parse_file_worker(args: Tuple[str, str]) -> Optional[Dict[str, Any]]:
    """Parse a single Python file and extract metadata.

    This is a module-level function for use with ProcessPoolExecutor.
    Returns a dict with file metadata that can be converted to FileMetadata.

    Args:
        args: Tuple of (file_path_str, root_path_str)

    Returns:
        Dict with file metadata or None if parsing failed
    """
    file_path_str, root_path_str = args
    file_path = Path(file_path_str)
    root_path = Path(root_path_str)

    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))

        # Extract metadata
        stat = file_path.stat()
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        rel_path = str(file_path.relative_to(root_path))

        # Extract symbols using shared helpers
        from victor_contracts.utils.ast_helpers import extract_symbols as _extract_syms

        raw_symbols = _extract_syms(tree)
        symbols = [
            {
                "name": s.name,
                "type": s.type,
                "file_path": rel_path,
                "line_number": s.line_number,
                "docstring": s.docstring,
                "signature": s.signature,
            }
            for s in raw_symbols
        ]

        # Extract imports (module names for dependency edges)
        imports: list = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return {
            "path": rel_path,
            "language": "python",
            "last_modified": stat.st_mtime,
            "indexed_at": time.time(),
            "size": stat.st_size,
            "lines": content.count("\n") + 1,
            "content_hash": content_hash,
            "symbols": symbols,
            "imports": imports,
        }

    except Exception as e:
        logger.debug(f"Failed to parse {file_path}: {e}")
        return None


class CodebaseFileHandler(FileSystemEventHandler):
    """File system event handler for tracking codebase changes.

    Tracks file modifications, creations, and deletions to mark
    the index as stale when relevant files change.
    """

    def __init__(
        self,
        on_change: Callable[[str], None],
        file_patterns: List[str] = None,
        ignore_patterns: List[str] = None,
    ):
        """Initialize file handler.

        Args:
            on_change: Callback when a file changes (receives file path)
            file_patterns: File patterns to watch (e.g., ["*.py"])
            ignore_patterns: Patterns to ignore
        """
        super().__init__()
        self.on_change = on_change
        self.file_patterns = file_patterns or ["*.py"]
        self.ignore_patterns = ignore_patterns or [
            "__pycache__",
            ".git",
            "node_modules",
            ".pytest_cache",
            "venv",
            ".venv",
        ]
        self._debounce_lock = threading.Lock()
        self._pending_changes: Set[str] = set()
        self._debounce_timer: Optional[threading.Timer] = None
        self._debounce_delay = 0.5  # 500ms debounce

    def _should_process(self, path: str) -> bool:
        """Check if path should be processed."""
        path_obj = Path(path)

        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if pattern in str(path_obj):
                return False

        # Check file patterns
        for pattern in self.file_patterns:
            if path_obj.match(pattern):
                return True

        return False

    def _debounced_notify(self) -> None:
        """Notify of changes after debounce period."""
        with self._debounce_lock:
            changes = list(self._pending_changes)
            self._pending_changes.clear()
            self._debounce_timer = None

        for path in changes:
            try:
                self.on_change(path)
            except Exception as e:
                logger.warning(f"Error in file change callback: {e}")

    def _schedule_notification(self, path: str) -> None:
        """Schedule a debounced notification."""
        with self._debounce_lock:
            self._pending_changes.add(path)

            # Cancel existing timer
            if self._debounce_timer:
                self._debounce_timer.cancel()

            # Schedule new timer
            self._debounce_timer = threading.Timer(self._debounce_delay, self._debounced_notify)
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def on_modified(self, event) -> None:
        """Handle file modification."""
        if not event.is_directory and self._should_process(event.src_path):
            self._schedule_notification(event.src_path)

    def on_created(self, event) -> None:
        """Handle file creation."""
        if not event.is_directory and self._should_process(event.src_path):
            self._schedule_notification(event.src_path)

    def on_deleted(self, event) -> None:
        """Handle file deletion."""
        if not event.is_directory and self._should_process(event.src_path):
            self._schedule_notification(event.src_path)


class IndexedSymbol(BaseModel):
    """Code symbol stored in the codebase index.

    Renamed from Symbol to be semantically distinct:
    - IndexedSymbol (here): Pydantic model for index storage
    - NativeSymbol (victor.native.protocols): Rust-extracted symbols (frozen)
    - RefactorSymbol (victor_coding.refactor.protocol): Refactoring symbol

    Note: Body content is NOT stored here - read from file via line_number/end_line.
    This keeps the index lightweight while allowing full body retrieval on demand.
    """

    name: str
    type: str  # function, class, variable, import
    file_path: str
    line_number: int
    end_line: Optional[int] = None  # end line - use with line_number to read body from file
    docstring: Optional[str] = None
    signature: Optional[str] = None
    parent_symbol: Optional[str] = None  # parent symbol name (for methods in classes)
    references: List[str] = Field(default_factory=list)  # Files that reference this symbol
    base_classes: List[str] = Field(default_factory=list)  # inheritance targets
    composition: List[tuple[str, str]] = Field(default_factory=list)  # (owner, member) for has-a


# Backward compatibility alias
Symbol = IndexedSymbol


class FileMetadata(BaseModel):
    """Metadata about a source file."""

    path: str
    language: str
    symbols: List[Symbol] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)  # Files this file depends on
    call_edges: List[tuple[str, str]] = Field(default_factory=list)  # (caller, callee) pairs
    inherit_edges: List[tuple[str, str]] = Field(default_factory=list)  # (child, base)
    implements_edges: List[tuple[str, str]] = Field(default_factory=list)  # (child, interface)
    compose_edges: List[tuple[str, str]] = Field(default_factory=list)  # (owner, member)
    references: List[str] = Field(default_factory=list)  # Identifier references (tree-sitter/AST)
    last_modified: float  # File mtime when indexed
    indexed_at: float = 0.0  # When this file was indexed
    size: int
    lines: int
    content_hash: Optional[str] = None  # SHA256 hash for change detection


class CodebaseIndex:
    """Indexes codebase for intelligent code understanding.

    This is the foundation for matching Claude Code's codebase awareness.

    Supports:
    - AST-based symbol extraction
    - Keyword search
    - Semantic search (with embeddings)
    - Dependency graph analysis
    """

    # All source file patterns to watch (multi-language support)
    WATCHED_PATTERNS = [
        "*.py",
        "*.pyw",  # Python
        "*.js",
        "*.jsx",
        "*.mjs",  # JavaScript
        "*.ts",
        "*.tsx",  # TypeScript
        "*.go",  # Go
        "*.rs",  # Rust
        "*.java",
        "*.kt",
        "*.scala",  # JVM
        "*.rb",  # Ruby
        "*.php",  # PHP
        "*.cs",  # C#
        "*.cpp",
        "*.cc",
        "*.c",
        "*.h",
        "*.hpp",  # C/C++
        "*.swift",  # Swift
        "*.dart",  # Dart
        "*.json",
        "*.yaml",
        "*.yml",
        "*.toml",
        "*.ini",
        "*.properties",
        "*.conf",
        "*.hocon",
    ]

    # Unified ID generation for graph-embedding correlation
    @staticmethod
    def make_symbol_id(file_path: str, symbol_name: str) -> str:
        """Generate unified symbol ID for graph and embedding correlation.

        Format: symbol:{file_path}:{symbol_name}

        This ID is used as:
        - node_id in graph_node (SQLite)
        - doc_id in embeddings (LanceDB)

        Enables bidirectional lookup:
        - Semantic search → node_id → graph traversal
        - Graph query → node_id → embedding lookup
        """
        return f"symbol:{file_path}:{symbol_name}"

    @staticmethod
    def make_file_id(file_path: str) -> str:
        """Generate unified file ID for graph nodes."""
        return f"file:{file_path}"

    def __init__(
        self,
        root_path: str,
        ignore_patterns: Optional[List[str]] = None,
        use_embeddings: bool = True,
        embedding_config: Optional[Dict[str, Any]] = None,
        enable_watcher: bool = True,
        graph_store: Optional["GraphStoreProtocol"] = None,
        graph_store_name: Optional[str] = None,
        graph_path: Optional[Path] = None,
        parallel_workers: int = 0,
    ):
        """Initialize codebase indexer.

        Graph and embeddings are always coupled - they share unified IDs for
        correlation. When you query semantic search, you can use the returned
        unified_id to traverse the graph. When you traverse the graph, you can
        lookup semantic similarity for nodes.

        Args:
            root_path: Root directory of the codebase
            ignore_patterns: Patterns to ignore (e.g., ["venv/", "node_modules/"])
            use_embeddings: Enable semantic search with embeddings (default: True).
                Graph and embeddings are coupled via unified IDs.
            embedding_config: Configuration for embedding provider (optional)
            enable_watcher: Whether to enable file watching for auto-staleness detection
            graph_store: Optional graph store for symbol relationships. If None,
                a per-repo store is created under .victor/graph/graph.db.
            graph_store_name: Optional graph backend name (sqlite, proxima,
                memory, or "auto" to resolve the repo's .victor/graph_backend
                marker; None = "auto")
            graph_path: Optional explicit graph store path
            parallel_workers: Number of parallel workers for file indexing.
                0 = auto-detect (min(cpu_count, 8)), 1 = sequential (default: 0)
                Use parallel processing for 3-8x speedup on large codebases.
        """
        self.root = Path(root_path).resolve()

        # Parallel indexing configuration
        if parallel_workers == 0:
            # Auto-detect: cap at 4 workers (benchmarks show only 2x speedup,
            # so loading more tree-sitter parsers has diminishing returns)
            import multiprocessing

            self._parallel_workers = min(multiprocessing.cpu_count(), 4)
        else:
            self._parallel_workers = parallel_workers
        # Note: Hidden directories (starting with '.') are excluded automatically
        # by _should_ignore(), so no need to list .git/, .venv/, .pytest_cache/, etc.
        self.ignore_patterns = ignore_patterns or [
            "venv/",
            "env/",
            "node_modules/",
            "__pycache__/",
            "*.pyc",
            "dist/",
            "build/",
            "out/",
            # Coverage and docs
            "htmlcov/",
            "coverage/",
            # Third party / vendor
            "vendor/",
            "third_party/",
            # Archive/legacy code (not actively maintained)
            "archive/",
        ]

        # Indexed data
        self.files: Dict[str, FileMetadata] = {}
        self.symbols: Dict[str, Symbol] = {}  # symbol_name -> Symbol
        self.symbol_index: Dict[str, List[str]] = {}  # file -> symbol names

        # Staleness tracking
        self._is_indexed = False
        self._is_stale = False
        self._changed_files: Set[str] = set()
        self._last_indexed: Optional[float] = None
        self._staleness_lock = threading.Lock()

        # File watcher
        self._watcher_enabled = enable_watcher and WATCHDOG_AVAILABLE
        self._observer: Optional[Observer] = None
        self._file_handler: Optional[CodebaseFileHandler] = None

        # Callbacks for change notifications (e.g., SymbolStore)
        self._change_callbacks: List[Callable[[str], None]] = []

        # Graph store (per-repo; backend resolved via graph_store_name or the
        # repo's .victor/graph_backend marker — this parameter was previously
        # accepted and silently dropped, so callers passing "proxima" always
        # got sqlite)
        if graph_store is None:
            self.graph_store: Optional["GraphStoreProtocol"] = create_graph_store(
                name=graph_store_name or "auto",
                project_path=self.root,
            )
        else:
            self.graph_store = graph_store
        self._graph_nodes: List[GraphNode] = []
        self._graph_edges: List[GraphEdge] = []
        self._pending_call_edges: List[tuple[str, str, str]] = []  # caller_id, callee_name, file
        self._pending_inherit_edges: List[tuple[str, str, str]] = []  # child_id, base_name, file
        self._pending_implements_edges: List[tuple[str, str, str]] = []  # child_id, interface, file
        self._pending_compose_edges: List[tuple[str, str, str]] = []  # owner_id, member_type, file
        self._symbol_resolver = SymbolResolver()

        # Embedding support (optional)
        self.use_embeddings = use_embeddings
        self.embedding_provider: Optional["BaseEmbeddingProvider"] = None
        if use_embeddings:
            self._initialize_embeddings(embedding_config)

        # Unified tree-sitter extractor using language registry
        self._language_registry = get_language_registry()
        self._language_registry.discover_plugins()
        self._tree_sitter_extractor = TreeSitterExtractor(self._language_registry)

        # Tier-aware unified symbol extractor (Phase 3 of tiered language support)
        # Provides enhanced symbol extraction with native AST and LSP enrichment
        self._unified_extractor = UnifiedSymbolExtractor(
            tree_sitter=self._tree_sitter_extractor,
            lsp_service=None,  # LSP service set later if available
            enable_lsp_enrichment=True,
        )

    def _reset_graph_buffers(self) -> None:
        self._graph_nodes = []
        self._graph_edges = []
        self._pending_call_edges = []
        self._pending_inherit_edges = []
        self._pending_implements_edges = []
        self._pending_compose_edges = []
        self._symbol_resolver = SymbolResolver()

    def _enriched_to_symbol(self, enriched: EnrichedSymbol, relative_path: str) -> Symbol:
        """Convert an EnrichedSymbol to the legacy Symbol format.

        This maintains backward compatibility while enabling tier-aware extraction.
        The EnrichedSymbol may contain additional fields (return_type, parameters,
        decorators, is_async) that aren't directly in Symbol but can be included
        in the signature field.

        Args:
            enriched: EnrichedSymbol from unified extractor
            relative_path: File path relative to project root

        Returns:
            Symbol compatible with existing indexer logic
        """
        # Build enhanced signature if we have type info
        signature = enriched.signature
        if not signature and (enriched.parameters or enriched.return_type):
            if enriched.symbol_type in ("function", "method"):
                params = ", ".join(enriched.parameters) if enriched.parameters else ""
                ret = f" -> {enriched.return_type}" if enriched.return_type else ""
                prefix = "async " if enriched.is_async else ""
                signature = f"{prefix}def {enriched.name}({params}){ret}"

        return Symbol(
            name=enriched.name,
            type=enriched.symbol_type,
            file_path=relative_path,
            line_number=enriched.line_number,
            end_line=enriched.end_line,
            docstring=enriched.docstring,
            signature=signature,
            parent_symbol=enriched.parent_symbol,
        )

    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on ignore patterns.

        Also excludes hidden directories (starting with '.') by convention.
        Only checks path parts WITHIN the project root, not parent directories.
        """
        # Get path relative to root to avoid ignoring due to parent directories
        # e.g., ~/.victor/swe_bench_cache/repo should not be ignored because of .victor
        try:
            rel_path = path.relative_to(self.root)
            path_str = str(rel_path)
        except ValueError:
            # Path is not under root - check full path
            path_str = str(path)
            rel_path = path

        # Skip hidden directories (Unix convention: directories starting with '.')
        # This excludes .git/, .vscode-test/, .vscode-victor/, etc.
        for part in rel_path.parts:
            if part.startswith(".") and part not in (".", ".."):
                return True

        for pattern in self.ignore_patterns:
            if pattern.endswith("/"):
                # Directory pattern
                if pattern[:-1] in path_str or f"/{pattern[:-1]}/" in path_str:
                    return True
            elif "*" in pattern:
                # Glob pattern
                import fnmatch

                if fnmatch.fnmatch(path.name, pattern):
                    return True
            else:
                # Exact match
                if pattern in path_str:
                    return True
        return False

    async def index_codebase(self) -> None:
        """Index the entire codebase.

        Scans all source files matching WATCHED_PATTERNS and extracts:
        - Symbols (classes, functions, methods)
        - Imports and dependencies
        - Call edges for function relationships

        After indexing:
        - `self.files` contains FileMetadata for each indexed file
        - `self.symbols` contains all extracted symbols
        - Graph store (if configured) contains relationship data

        Note: This is a FULL rebuild. It clears the graph store first to remove
        stale entries from renamed/deleted files, then rebuilds from scratch.

        Performance: Uses parallel processing with ProcessPoolExecutor when
        _parallel_workers > 1, providing 3-8x speedup on multi-core systems.
        """
        self._reset_graph_buffers()
        self.files.clear()
        self.symbols.clear()
        self.symbol_index.clear()

        # Clear graph store first to remove stale entries from renamed/deleted files
        if self.graph_store:
            try:
                await self.graph_store.delete_by_repo()
                logger.debug("Cleared graph store for full rebuild")
            except Exception as e:
                logger.warning(f"Failed to clear graph store: {e}")

        # Discover all files matching watched patterns
        files_to_index: List[Tuple[Path, str]] = []
        for pattern in self.WATCHED_PATTERNS:
            for file_path in self.root.rglob(pattern):
                if file_path.is_file() and not self._should_ignore(file_path):
                    language = self._detect_language(file_path)
                    files_to_index.append((file_path, language))

        # Use parallel processing for large codebases (3-8x speedup)
        if self._parallel_workers > 1 and len(files_to_index) > 50:
            await self._index_files_parallel(files_to_index)
        else:
            # Sequential fallback for small codebases or single-worker mode
            for file_path, language in files_to_index:
                try:
                    await self._index_tree_sitter_file(file_path, language)
                except Exception as exc:
                    logger.debug(f"Failed to index {file_path}: {exc}")

        # Resolve cross-file dependencies
        self._resolve_cross_file_calls()
        self._build_dependency_graph()

        # Flush graph buffers to store
        if self.graph_store and self._graph_nodes:
            await self.graph_store.upsert_nodes(self._graph_nodes)
        if self.graph_store and self._graph_edges:
            await self.graph_store.upsert_edges(self._graph_edges)

        # Build embeddings for all indexed symbols (batched for performance)
        if self.use_embeddings and self.embedding_provider:
            # Clear existing embeddings first (handles schema changes like adding end_line)
            try:
                await self.embedding_provider.clear_index()
                logger.debug("Cleared embedding index for full rebuild")
            except Exception as e:
                logger.warning(f"Failed to clear embedding index: {e}")

            # Collect all documents for batch embedding
            # Uses unified IDs for graph-embedding correlation
            documents = []
            for rel_path, file_meta in self.files.items():
                for symbol in file_meta.symbols:
                    text_for_embedding = self._get_symbol_embedding_text(symbol)
                    if text_for_embedding:
                        # Use unified ID for correlation with graph nodes
                        unified_id = self.make_symbol_id(rel_path, symbol.name)
                        documents.append(
                            {
                                "id": unified_id,
                                "content": text_for_embedding,
                                "metadata": {
                                    "file_path": rel_path,
                                    "symbol_name": symbol.name,
                                    "symbol_type": symbol.type,
                                    "line_number": symbol.line_number,
                                    "end_line": symbol.end_line,  # For precise reads
                                },
                            }
                        )

            # Batch embed for performance (process in chunks of 500)
            batch_size = 500
            embedding_count = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                try:
                    await self.embedding_provider.index_documents(batch)
                    embedding_count += len(batch)
                    if i > 0 and i % 5000 == 0:
                        logger.info(f"Embedded {embedding_count}/{len(documents)} symbols...")
                except Exception as e:
                    logger.warning(f"Failed to embed batch at {i}: {e}")

            logger.info(f"Created {embedding_count} embeddings for semantic search")

        self._is_indexed = True
        self._is_stale = False
        self._last_indexed = time.time()
        logger.info(f"Indexed {len(self.files)} files with {len(self.symbols)} symbols")

    async def _index_files_parallel(self, files_to_index: List[Tuple[Path, str]]) -> None:
        """Index files using parallel processing with ProcessPoolExecutor.

        This method provides 3-8x speedup on multi-core systems by processing
        files in parallel. The CPU-intensive tree-sitter parsing is offloaded
        to worker processes.

        Args:
            files_to_index: List of (file_path, language) tuples to index
        """
        start_time = time.time()
        root_str = str(self.root)
        total_files = len(files_to_index)
        processed = 0
        errors = 0

        logger.info(
            f"Starting parallel indexing: {total_files} files, " f"{self._parallel_workers} workers"
        )

        # Prepare arguments for parallel processing
        tasks = [(str(file_path), root_str, language) for file_path, language in files_to_index]

        # Process files in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self._parallel_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(_process_file_parallel, *task): task for task in tasks}

            # Process results as they complete
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        # Convert result dict to FileMetadata and merge
                        self._merge_parallel_result(result)
                        processed += 1
                    else:
                        errors += 1
                except Exception as exc:
                    logger.debug(f"Parallel index failed for {task[0]}: {exc}")
                    errors += 1

                # Progress logging every 500 files
                if (processed + errors) % 500 == 0:
                    logger.debug(
                        f"Progress: {processed + errors}/{total_files} files "
                        f"({processed} ok, {errors} failed)"
                    )

        elapsed = time.time() - start_time
        files_per_sec = total_files / elapsed if elapsed > 0 else 0
        logger.info(
            f"Parallel indexing complete: {processed}/{total_files} files in {elapsed:.2f}s "
            f"({files_per_sec:.1f} files/sec, {errors} errors)"
        )

    def _merge_parallel_result(self, result: Dict[str, Any]) -> None:
        """Merge a parallel processing result into the index.

        Converts the result dictionary from _process_file_parallel into
        FileMetadata and Symbol objects, then records them in the index.

        Args:
            result: Dictionary from _process_file_parallel with file data
        """
        # Convert symbol dicts to Symbol objects
        symbols = [
            Symbol(
                name=s["name"],
                type=s["type"],
                file_path=s["file_path"],
                line_number=s["line_number"],
                end_line=s.get("end_line"),
            )
            for s in result.get("symbols", [])
        ]

        # Create FileMetadata
        metadata = FileMetadata(
            path=result["path"],
            language=result["language"],
            symbols=symbols,
            imports=result.get("imports", []),
            call_edges=result.get("call_edges", []),
            inherit_edges=result.get("inherit_edges", []),
            implements_edges=result.get("implements_edges", []),
            compose_edges=result.get("compose_edges", []),
            references=result.get("references", []),
            last_modified=result["last_modified"],
            indexed_at=time.time(),
            size=result["size"],
            lines=result["lines"],
        )

        # Store in index
        self.files[metadata.path] = metadata

        # Record symbols and build graph nodes/edges
        self._record_symbols(metadata)

    async def ensure_indexed(self, auto_reindex: bool = True) -> None:
        """Ensure the index is ready for querying.

        If the index hasn't been built yet, builds it. If auto_reindex is True
        and the index is stale, rebuilds it.

        Args:
            auto_reindex: If True, automatically reindex when stale (default True)
        """
        if not self._is_indexed:
            # Never indexed - do a full index
            logger.debug("Index not built, building initial index")
            await self.index_codebase()
        elif auto_reindex and self._is_stale:
            # Index exists but is stale - rebuild
            logger.debug("Index is stale, rebuilding")
            await self.index_codebase()

    async def incremental_reindex(self, files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform incremental reindexing of changed files only.

        This method efficiently updates the index by only processing files that:
        1. Have been modified since last indexing (detected via mtime)
        2. Have been added since last indexing
        3. Are explicitly specified in the `files` parameter

        Files that have been deleted are automatically removed from the index.

        Args:
            files: Optional list of specific file paths to reindex.
                   If None, auto-detects changed files via mtime comparison.

        Returns:
            Dictionary with reindex statistics:
            - updated: List of files that were re-indexed
            - added: List of new files that were indexed
            - removed: List of files removed from index
            - unchanged: Count of files that didn't need updating
            - errors: List of files that failed to index
        """
        stats = {
            "updated": [],
            "added": [],
            "removed": [],
            "unchanged": 0,
            "errors": [],
        }

        if not self._is_indexed:
            # No existing index - do full index instead
            logger.info("No existing index, performing full index")
            await self.index_codebase()
            stats["added"] = list(self.files.keys())
            return stats

        # Determine which files to process
        if files:
            # Explicit file list provided
            files_to_check = [Path(f) if not isinstance(f, Path) else f for f in files]
        else:
            # Auto-detect: check all watched files for changes
            files_to_check = []
            for pattern in self.WATCHED_PATTERNS:
                for file_path in self.root.rglob(pattern):
                    if file_path.is_file() and not self._should_ignore(file_path):
                        files_to_check.append(file_path)

        # Track current files to detect deletions
        current_files: Set[str] = set()

        for file_path in files_to_check:
            if not file_path.exists():
                continue

            rel_path = str(file_path.relative_to(self.root))
            current_files.add(rel_path)

            try:
                current_mtime = file_path.stat().st_mtime

                # Check if file is new or modified
                if rel_path in self.files:
                    existing_mtime = self.files[rel_path].last_modified
                    if current_mtime <= existing_mtime:
                        # File unchanged
                        stats["unchanged"] += 1
                        continue

                    # File modified - reindex it
                    language = self._detect_language(file_path)
                    await self._index_single_file(file_path, language)
                    stats["updated"].append(rel_path)
                    logger.debug(f"Updated index for: {rel_path}")
                else:
                    # New file - add to index
                    language = self._detect_language(file_path)
                    await self._index_single_file(file_path, language)
                    stats["added"].append(rel_path)
                    logger.debug(f"Added to index: {rel_path}")

            except Exception as e:
                logger.warning(f"Failed to reindex {rel_path}: {e}")
                stats["errors"].append({"file": rel_path, "error": str(e)})

        # Detect and remove deleted files (only if we scanned all files)
        if not files:
            for indexed_path in list(self.files.keys()):
                if indexed_path not in current_files:
                    # File was deleted - remove from index
                    await self._remove_file_from_index(indexed_path)
                    stats["removed"].append(indexed_path)
                    logger.debug(f"Removed from index: {indexed_path}")

        # Clear staleness flag and update changed files set
        with self._staleness_lock:
            self._is_stale = False
            self._changed_files.clear()
            self._last_indexed = time.time()

        # Log summary
        total_changes = len(stats["updated"]) + len(stats["added"]) + len(stats["removed"])
        if total_changes > 0:
            logger.info(
                f"Incremental reindex: {len(stats['updated'])} updated, "
                f"{len(stats['added'])} added, {len(stats['removed'])} removed, "
                f"{stats['unchanged']} unchanged"
            )
        else:
            logger.debug(f"Incremental reindex: no changes detected ({stats['unchanged']} files)")

        return stats

    async def _index_single_file(self, file_path: Path, language: str) -> None:
        """Index a single file and update the index structures.

        This is used by incremental_reindex to update individual files
        without rebuilding the entire index.

        Args:
            file_path: Path to the file to index
            language: Detected language of the file
        """
        rel_path = str(file_path.relative_to(self.root))

        # Remove existing symbols for this file
        if rel_path in self.symbol_index:
            for symbol_key in self.symbol_index[rel_path]:
                self.symbols.pop(symbol_key, None)
            self.symbol_index[rel_path] = []

        # Use the existing tree-sitter indexing logic
        await self._index_tree_sitter_file(file_path, language)

        # Update embeddings if enabled
        if self.use_embeddings and self.embedding_provider and rel_path in self.files:
            file_meta = self.files[rel_path]
            for symbol in file_meta.symbols:
                # Generate embedding for symbol if it has meaningful content
                text_for_embedding = self._get_symbol_embedding_text(symbol)
                if text_for_embedding:
                    try:
                        # Use unified ID for graph-embedding correlation
                        unified_id = self.make_symbol_id(rel_path, symbol.name)
                        await self.embedding_provider.index_document(
                            doc_id=unified_id,
                            content=text_for_embedding,
                            metadata={
                                "file_path": rel_path,
                                "symbol_name": symbol.name,
                                "symbol_type": symbol.type,
                                "line_number": symbol.line_number,
                            },
                        )
                    except Exception as e:
                        logger.debug(f"Failed to embed symbol {symbol.name}: {e}")

    async def _remove_file_from_index(self, rel_path: str) -> None:
        """Remove a file and its symbols from the index.

        Args:
            rel_path: Relative path of the file to remove
        """
        # Remove symbols
        if rel_path in self.symbol_index:
            for symbol_key in self.symbol_index[rel_path]:
                self.symbols.pop(symbol_key, None)

                # Remove from embeddings if enabled
                if self.use_embeddings and self.embedding_provider:
                    try:
                        await self.embedding_provider.delete_document(symbol_key)
                    except Exception:
                        pass  # Ignore deletion errors

            del self.symbol_index[rel_path]

        # Remove file metadata
        self.files.pop(rel_path, None)

        # Remove from graph store if available
        if self.graph_store:
            try:
                # Delete nodes and edges associated with this file
                await self.graph_store.delete_by_file(rel_path)
            except Exception as e:
                logger.debug(f"Failed to remove graph nodes for {rel_path}: {e}")

    def _get_symbol_embedding_text(self, symbol: Symbol) -> str:
        """Generate text representation of a symbol for embedding.

        Args:
            symbol: The symbol to generate embedding text for

        Returns:
            Text suitable for embedding, or empty string if not embeddable
        """
        parts = []

        # Add symbol type and name
        parts.append(f"{symbol.type} {symbol.name}")

        # Add signature if available
        if symbol.signature:
            parts.append(symbol.signature)

        # Add docstring if available
        if symbol.docstring:
            parts.append(symbol.docstring[:500])  # Limit docstring length

        return " ".join(parts)

    def _detect_language(self, file_path: Path, default: str = "python") -> str:
        """Detect language from extension for tree-sitter queries.

        Uses the language registry for detection, with fallback to legacy
        EXTENSION_TO_LANGUAGE dict for config files.
        """
        # Try registry first (unified approach)
        detected = self._language_registry.detect_language(file_path)
        if detected:
            return detected
        # Fallback to legacy dict (for config files not in registry)
        return EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower(), default)

    def _is_config_language(self, language: str) -> bool:
        """Return True if the language is a config/metadata file."""
        return language.startswith("config")

    def _extract_references(
        self,
        file_path: Path,
        language: str,
        fallback_calls: List[str],
        imports: List[str],
    ) -> List[str]:
        """Extract identifier references using tree-sitter when available."""
        refs: Set[str] = set(fallback_calls) | set(imports)

        # PRIMARY: Get reference query from language plugin
        query_src = None
        try:
            plugin = self._language_registry.get(language)
            if plugin and plugin.tree_sitter_queries.references:
                query_src = plugin.tree_sitter_queries.references
        except Exception:
            pass

        # FALLBACK: Use legacy static dictionary (for languages not yet migrated)
        if not query_src:
            query_src = REFERENCE_QUERIES.get(language)

        if not query_src:
            return list(refs)
        try:
            from victor_coding.codebase.tree_sitter_manager import get_parser
        except Exception:
            return list(refs)

        try:
            parser = get_parser(language)
        except Exception:
            return list(refs)

        if parser is None:
            return list(refs)

        try:
            from tree_sitter import QueryCursor

            content = file_path.read_bytes()
            tree = parser.parse(content)
            query = Query(parser.language, query_src)
            cursor = QueryCursor(query)
            captures_dict = cursor.captures(tree.root_node)
            for _capture_name, nodes in captures_dict.items():
                for node in nodes:
                    text = node.text.decode("utf-8", errors="ignore")
                    if text:
                        refs.add(text)
        except Exception:
            # Graceful degradation; fall back to existing refs
            pass

        # Regex fallback to catch simple identifier usage when tree-sitter misses
        if not refs:
            try:
                text = file_path.read_text(encoding="utf-8")
                for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", text):
                    refs.add(match.group(0))
            except Exception:
                return list(refs)

        return list(refs)

    def _extract_config_keys(self, content: str, language: str) -> List[tuple[str, int]]:
        """Extract top-level-ish config keys for JSON/YAML/INI/property files."""
        keys: dict[str, int] = {}

        def _walk(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    dotted = f"{prefix}.{k}" if prefix else str(k)
                    keys.setdefault(dotted, 1)
                    _walk(v, dotted)
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    dotted = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                    keys.setdefault(dotted, 1)
                    _walk(item, dotted)

        try:
            if language == "config-json":
                data = json.loads(content)
                _walk(data)
            elif language == "config-yaml":
                try:
                    import yaml  # type: ignore

                    data = yaml.safe_load(content)
                    _walk(data)
                except Exception:
                    pass
        except Exception:
            # Fall back to regex below
            pass

        if not keys:
            # Regex fallback for generic key/value formats
            for match in re.finditer(
                r'^[\s"\']*([A-Za-z0-9_.\-]+)\s*[:=]', content, flags=re.MULTILINE
            ):
                key = match.group(1)
                line_no = content.count("\n", 0, match.start()) + 1
                keys.setdefault(key, line_no)

        return list(keys.items())

    # ========================================================================
    # ARCHIVED: First duplicate block of tree-sitter extraction methods
    # These methods were duplicated and have been consolidated.
    # The active versions are now the LAST definitions in the class.
    # Lines 866-1197 removed on 2025-12-11 during plugin architecture refactoring.
    # ========================================================================

    def _extract_symbols_with_tree_sitter(self, file_path: Path, language: str) -> List[Symbol]:
        """Extract lightweight symbol declarations for non-Python languages via tree-sitter."""
        query_defs = SYMBOL_QUERIES.get(language)
        if not query_defs:
            return []
        symbols: List[Symbol] = []
        parser = None
        try:
            from victor_coding.codebase.tree_sitter_manager import get_parser

            try:
                parser = get_parser(language)
            except Exception:
                parser = None
        except Exception:
            parser = None

        if parser is not None:
            try:
                content = file_path.read_bytes()
                tree = parser.parse(content)
                for sym_type, query_src in query_defs:
                    try:
                        from tree_sitter import QueryCursor

                        query = Query(parser.language, query_src)
                        cursor = QueryCursor(query)
                        captures_dict = cursor.captures(tree.root_node)
                        for _capture_name, nodes in captures_dict.items():
                            for node in nodes:
                                text = node.text.decode("utf-8", errors="ignore")
                                if not text:
                                    continue
                                symbols.append(
                                    Symbol(
                                        name=text,
                                        type=sym_type,
                                        file_path=str(file_path.relative_to(self.root)),
                                        line_number=node.start_point[0] + 1,
                                    )
                                )
                    except Exception:
                        continue
            except Exception:
                symbols = []
        if symbols:
            return symbols

        # Regex fallback when grammar support is unavailable
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            return []

        regex_fallbacks: Dict[str, list[tuple[str, str]]] = {
            "javascript": [
                ("class", r"class\s+(\w+)"),
                ("function", r"function\s+(\w+)"),
            ],
            "typescript": [
                ("class", r"class\s+(\w+)"),
                ("function", r"function\s+(\w+)"),
            ],
            "go": [
                ("function", r"func\s+(?:\([\w\*\s,]+\)\s*)?(\w+)"),
                ("class", r"type\s+(\w+)\s+struct"),
            ],
            "java": [
                ("class", r"(?:class|interface)\s+(\w+)"),
                ("function", r"(?:public|private|protected|\s)+\s*\w+\s+(\w+)\s*\("),
            ],
        }
        for sym_type, pattern in regex_fallbacks.get(language, []):
            for match in re.finditer(pattern, text):
                name = match.group(1)
                symbols.append(
                    Symbol(
                        name=name,
                        type=sym_type,
                        file_path=str(file_path.relative_to(self.root)),
                        line_number=text.count("\n", 0, match.start()) + 1,
                    )
                )
        return symbols

    def _extract_inheritance(
        self, file_path: Path, language: str, symbols: List[Symbol]
    ) -> List[tuple[str, str]]:
        """Extract child->base inheritance edges."""
        edges: List[tuple[str, str]] = []
        # First check if symbols have base_classes populated (from AST path)
        if language == "python":
            for sym in symbols:
                if sym.type == "class" and sym.base_classes:
                    for base in sym.base_classes:
                        edges.append((sym.name, base))
            # If we found edges via AST, return them; otherwise try tree-sitter
            if edges:
                return edges
            # Fallback: parse AST directly to extract bases (tree-sitter symbols may not include them)
            try:
                import ast as py_ast

                tree = py_ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
                for node in py_ast.walk(tree):
                    if isinstance(node, py_ast.ClassDef):
                        for base_name in extract_base_classes(node):
                            edges.append((node.name, base_name))
            except Exception:
                pass
            if edges:
                return edges

        # PRIMARY: Get inheritance query from language plugin
        query_src = None
        try:
            plugin = self._language_registry.get(language)
            if plugin and plugin.tree_sitter_queries.inheritance:
                query_src = plugin.tree_sitter_queries.inheritance
        except Exception:
            pass
        # FALLBACK: Use legacy static dictionary
        if not query_src:
            query_src = INHERITS_QUERIES.get(language)

        parser = None
        try:
            from victor_coding.codebase.tree_sitter_manager import get_parser

            parser = get_parser(language)
        except Exception:
            parser = None

        if parser is not None and query_src:
            try:
                from tree_sitter import QueryCursor

                content = file_path.read_bytes()
                tree = parser.parse(content)
                query = Query(parser.language, query_src)
                cursor = QueryCursor(query)
                for _pat_idx, cap_dict in cursor.matches(tree.root_node):
                    child_nodes = cap_dict.get("child", [])
                    base_nodes = cap_dict.get("base", [])
                    if child_nodes and base_nodes:
                        child_text = child_nodes[0].text.decode("utf-8", errors="ignore")
                        base_text = base_nodes[0].text.decode("utf-8", errors="ignore")
                        if child_text and base_text:
                            edges.append((child_text, base_text))
            except Exception:
                pass

        if not edges:
            # Regex fallback
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                return edges
            for match in re.finditer(r"class\s+(\w+)\s+extends\s+(\w+)", text):
                edges.append((match.group(1), match.group(2)))
        return edges

    def _extract_implements(
        self, file_path: Path, language: str, symbols: List[Symbol]
    ) -> List[tuple[str, str]]:
        """Extract child->interface implements edges for typed languages."""
        edges: List[tuple[str, str]] = []
        # PRIMARY: Get implements query from language plugin
        query_src = None
        try:
            plugin = self._language_registry.get(language)
            if plugin and plugin.tree_sitter_queries.implements:
                query_src = plugin.tree_sitter_queries.implements
        except Exception:
            pass
        # FALLBACK: Use legacy static dictionary
        if not query_src:
            query_src = IMPLEMENTS_QUERIES.get(language)
        parser = None
        try:
            from victor_coding.codebase.tree_sitter_manager import get_parser

            parser = get_parser(language)
        except Exception:
            parser = None

        if parser is not None and query_src:
            try:
                from tree_sitter import QueryCursor

                content = file_path.read_bytes()
                tree = parser.parse(content)
                query = Query(parser.language, query_src)
                cursor = QueryCursor(query)
                for _pat_idx, cap_dict in cursor.matches(tree.root_node):
                    child_nodes = cap_dict.get("child", [])
                    iface_nodes = cap_dict.get("interface", []) or cap_dict.get("base", [])
                    if child_nodes and iface_nodes:
                        child_text = child_nodes[0].text.decode("utf-8", errors="ignore")
                        iface_text = iface_nodes[0].text.decode("utf-8", errors="ignore")
                        if child_text and iface_text:
                            edges.append((child_text, iface_text))
            except Exception:
                pass

        if not edges:
            # Regex fallback for implements
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                return edges
            for match in re.finditer(r"class\s+(\w+)\s+implements\s+([\w, ]+)", text):
                child = match.group(1)
                bases = [b.strip() for b in match.group(2).split(",") if b.strip()]
                for base in bases:
                    edges.append((child, base))
        return edges

    def _extract_composition(
        self, file_path: Path, language: str, symbols: List[Symbol]
    ) -> List[tuple[str, str]]:
        """Extract has-a/composition edges (owner -> member type)."""
        edges: List[tuple[str, str]] = []

        # Python handled via AST visitor (class attributes)
        if language == "python":
            for sym in symbols:
                if sym.type == "class" and hasattr(sym, "composition"):
                    edges.extend(sym.composition)
            return edges

        # PRIMARY: Get composition query from language plugin
        query_src = None
        try:
            plugin = self._language_registry.get(language)
            if plugin and plugin.tree_sitter_queries.composition:
                query_src = plugin.tree_sitter_queries.composition
        except Exception:
            pass
        # FALLBACK: Use legacy static dictionary
        if not query_src:
            query_src = COMPOSITION_QUERIES.get(language)

        parser = None
        try:
            from victor_coding.codebase.tree_sitter_manager import get_parser

            parser = get_parser(language)
        except Exception:
            parser = None

        if parser is not None and query_src:
            try:
                from tree_sitter import QueryCursor

                content = file_path.read_bytes()
                tree = parser.parse(content)
                query = Query(parser.language, query_src)
                cursor = QueryCursor(query)
                for _pat_idx, cap_dict in cursor.matches(tree.root_node):
                    owner_nodes = cap_dict.get("owner", [])
                    type_nodes = cap_dict.get("type", [])
                    if owner_nodes and type_nodes:
                        owner_text = owner_nodes[0].text.decode("utf-8", errors="ignore")
                        type_text = type_nodes[0].text.decode("utf-8", errors="ignore")
                        if owner_text and type_text:
                            edges.append((owner_text, type_text))
            except Exception:
                pass

        if edges:
            return edges

        # Regex fallback for typed declarations and new expressions
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            return edges

        owner: Optional[str] = None
        for line in text.splitlines():
            class_match = re.search(r"class\s+(\w+)", line)
            if class_match:
                owner = class_match.group(1)
                continue
            if line.strip().startswith("}"):
                owner = owner if "class" in line else None
            # TypeScript/Java style property: field: Type or Type field;
            field_match = re.search(r"(\w+)\s*[:]\s*(\w+)", line)
            java_field = re.search(r"(\w+)\s+(\w+)\s*;", line)
            new_expr = re.search(r"new\s+(\w+)\s*\(", line)
            target_type = None
            if field_match:
                target_type = field_match.group(2)
            elif java_field and owner:
                target_type = java_field.group(1)
            elif new_expr:
                target_type = new_expr.group(1)
            if owner and target_type:
                edges.append((owner, target_type))
        return edges

    def _find_enclosing_symbol_name(self, node, language: str) -> Optional[str]:
        """Best-effort caller lookup by walking ancestors."""
        # PRIMARY: Get enclosing scopes from language plugin
        fields = None
        try:
            plugin = self._language_registry.get(language)
            if plugin and plugin.tree_sitter_queries.enclosing_scopes:
                fields = plugin.tree_sitter_queries.enclosing_scopes
        except Exception:
            pass
        # FALLBACK: Use legacy static dictionary
        if not fields:
            fields = ENCLOSING_NAME_FIELDS.get(language, [])
        current = node.parent
        method_name: Optional[str] = None
        class_name: Optional[str] = None
        while current is not None:
            for node_type, field_name in fields:
                if current.type == node_type:
                    field = current.child_by_field_name(field_name)
                    if not field:
                        continue
                    # For C++ function_declarator, drill into nested declarator
                    # to get the bare identifier (e.g. "main" not "main()")
                    if field.type == "function_declarator":
                        inner = field.child_by_field_name("declarator")
                        if inner:
                            field = inner
                    text = field.text.decode("utf-8", errors="ignore")
                    if node_type in (
                        "class_declaration",
                        "interface_declaration",
                        "class_specifier",
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

    def _extract_calls_with_tree_sitter(
        self, file_path: Path, language: str
    ) -> List[tuple[str, str]]:
        """Extract caller->callee pairs using tree-sitter (non-Python)."""
        # PRIMARY: Get call query from language plugin
        query_src = None
        try:
            plugin = self._language_registry.get(language)
            if plugin and plugin.tree_sitter_queries.calls:
                query_src = plugin.tree_sitter_queries.calls
        except Exception:
            pass
        # FALLBACK: Use legacy static dictionary
        if not query_src:
            query_src = CALL_QUERIES.get(language)
        if not query_src:
            return []
        try:
            from victor_coding.codebase.tree_sitter_manager import get_parser
        except Exception:
            return []
        try:
            parser = get_parser(language)
        except Exception:
            return []
        if parser is None:
            return []

        content = file_path.read_bytes()
        tree = parser.parse(content)
        try:
            query = Query(parser.language, query_src)
        except Exception:
            return []

        call_edges: List[tuple[str, str]] = []
        try:
            from tree_sitter import QueryCursor

            cursor = QueryCursor(query)
            captures_dict = cursor.captures(tree.root_node)
            for _capture_name, nodes in captures_dict.items():
                for node in nodes:
                    callee = node.text.decode("utf-8", errors="ignore")
                    caller = self._find_enclosing_symbol_name(node, language)
                    if caller and callee:
                        call_edges.append((caller, callee))
        except Exception:
            call_edges = []

        if call_edges:
            return call_edges

        # Regex fallback when tree-sitter capture fails
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception:
            return []
        pattern = re.compile(r"(\w+)\s*\(")
        caller = None
        for line in text.splitlines():
            func_decl = re.search(r"function\s+(\w+)", line)
            if func_decl:
                caller = func_decl.group(1)
            method_decl = re.search(r"(\w+)\s*\([^)]*\)\s*\{", line)
            if method_decl:
                caller = method_decl.group(1)
            for callee in pattern.findall(line):
                if caller and callee and callee not in {"function", caller}:
                    call_edges.append((caller, callee))
        return call_edges

    # Tree-sitter import queries for non-Python languages.
    _IMPORT_QUERIES: Dict[str, str] = {
        "javascript": """
            (import_statement source: (string) @source)
            (call_expression
                function: (identifier) @_fn
                arguments: (arguments (string) @source)
                (#eq? @_fn "require"))
        """,
        "typescript": """
            (import_statement source: (string) @source)
            (call_expression
                function: (identifier) @_fn
                arguments: (arguments (string) @source)
                (#eq? @_fn "require"))
        """,
        "rust": """
            (use_declaration argument: (_) @source)
        """,
        "go": """
            (import_spec path: (interpreted_string_literal) @source)
        """,
        "java": """
            (import_declaration (scoped_identifier) @source)
        """,
    }

    def _extract_imports_with_tree_sitter(self, file_path: Path, language: str) -> List[str]:
        """Extract import/require/use statements using tree-sitter (non-Python)."""
        query_src = self._IMPORT_QUERIES.get(language)
        if not query_src:
            return []
        try:
            from victor_coding.codebase.tree_sitter_manager import get_parser
        except Exception:
            return []
        try:
            parser = get_parser(language)
        except Exception:
            return []
        if parser is None:
            return []

        imports: List[str] = []
        try:
            from tree_sitter import QueryCursor

            content = file_path.read_bytes()
            tree = parser.parse(content)
            query = Query(parser.language, query_src)
            cursor = QueryCursor(query)
            captures_dict = cursor.captures(tree.root_node)
            for node in captures_dict.get("source", []):
                text = node.text.decode("utf-8", errors="ignore")
                # Strip quotes from string literals
                cleaned = text.strip("'\"")
                if cleaned:
                    imports.append(cleaned)
        except Exception:
            pass
        return imports

    async def _index_tree_sitter_file(self, file_path: Path, language: str) -> None:
        """Index a file using tier-aware symbol extraction.

        For Tier 1/2 languages (Python, TypeScript, Go, Rust, etc.), uses the
        UnifiedSymbolExtractor which provides enhanced type information.
        Falls back to tree-sitter only for Tier 3 languages.
        """
        try:
            stat = file_path.stat()
            content = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.debug(f"Skipping {file_path} due to read error: {exc}")
            return

        relative_path = str(file_path.relative_to(self.root))

        # Use unified extractor for tier-aware symbol extraction
        # This provides enhanced type info for Tier 1/2 languages
        tier_config = get_tier(language)
        symbols: List[Symbol] = []

        if tier_config.tier in (LanguageTier.TIER_1, LanguageTier.TIER_2):
            # Try unified extractor first (provides enriched symbols)
            try:
                enriched = await self._unified_extractor.extract_symbols(
                    file_path, language, content
                )
                if enriched:
                    symbols = [self._enriched_to_symbol(s, relative_path) for s in enriched]
                    logger.debug(
                        f"Unified extractor: {len(symbols)} symbols from {file_path.name} "
                        f"(tier={tier_config.tier.name})"
                    )
            except Exception as e:
                logger.debug(f"Unified extraction failed for {file_path}: {e}")

        # Fall back to legacy tree-sitter extraction if needed
        if not symbols:
            symbols = self._extract_symbols_with_tree_sitter(file_path, language)

        call_edges = self._extract_calls_with_tree_sitter(file_path, language)
        imports: List[str] = []

        # Lightweight Python import extraction (tree-sitter path omits imports today)
        if language == "python":
            try:
                tree = ast.parse(content, filename=str(file_path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
            except Exception as exc:
                logger.debug(f"Failed to parse imports for {file_path}: {exc}")
        else:
            # Tree-sitter based import extraction for non-Python languages
            imports = self._extract_imports_with_tree_sitter(file_path, language)

        metadata = FileMetadata(
            path=str(file_path.relative_to(self.root)),
            language=language,
            symbols=symbols,
            imports=imports,
            last_modified=stat.st_mtime,
            indexed_at=time.time(),
            size=stat.st_size,
            lines=content.count("\n") + 1,
            call_edges=call_edges,
        )

        metadata.inherit_edges = self._extract_inheritance(file_path, language, symbols)
        metadata.implements_edges = self._extract_implements(file_path, language, symbols)
        metadata.compose_edges = self._extract_composition(file_path, language, symbols)

        metadata.references = self._extract_references(
            file_path,
            language,
            [callee for _, callee in call_edges],
            metadata.imports,
        )

        self.files[metadata.path] = metadata
        self._record_symbols(metadata)

    def _record_symbols(self, metadata: FileMetadata) -> None:
        """Record symbol metadata and populate graph buffers."""
        symbol_names = {s.name for s in metadata.symbols}

        # Always create a file node so config/docs files without symbols still appear in the graph.
        # Uses unified ID format for graph-embedding correlation.
        file_node_id: Optional[str] = None
        if self.graph_store:
            file_node_id = self.make_file_id(metadata.path)
            self._graph_nodes.append(
                GraphNode(
                    node_id=file_node_id,
                    type="file",
                    name=Path(metadata.path).name,
                    file=metadata.path,
                    line=None,
                    lang=metadata.language,
                    metadata={"lines": metadata.lines, "size": metadata.size},
                )
            )

        for symbol in metadata.symbols:
            # Symbol registry - use unified ID format
            unified_id = self.make_symbol_id(metadata.path, symbol.name)
            self.symbols[unified_id] = symbol
            if metadata.path not in self.symbol_index:
                self.symbol_index[metadata.path] = []
            self.symbol_index[metadata.path].append(symbol.name)

            if self.graph_store:
                # Determine parent_id for nested symbols (methods in classes)
                parent_id = None
                if symbol.parent_symbol:
                    parent_id = self.make_symbol_id(metadata.path, symbol.parent_symbol)

                self._graph_nodes.append(
                    GraphNode(
                        node_id=unified_id,
                        type=symbol.type,
                        name=symbol.name,
                        file=metadata.path,
                        line=symbol.line_number,
                        end_line=symbol.end_line,  # Use with line to read body from file
                        lang=metadata.language,
                        signature=symbol.signature,
                        docstring=symbol.docstring,
                        parent_id=parent_id,
                        embedding_ref=unified_id,  # Link to vector store entry
                        metadata={},
                    )
                )
                self._graph_edges.append(
                    GraphEdge(
                        src=file_node_id or self.make_file_id(metadata.path),
                        dst=unified_id,
                        type="CONTAINS",
                        metadata={"path": metadata.path},
                    )
                )

        # Add simple intra-file CALLS edges when both endpoints are known symbols
        # Uses unified IDs for graph-embedding correlation
        if self.graph_store and metadata.call_edges:
            for caller, callee in metadata.call_edges:
                caller_id = self.make_symbol_id(metadata.path, caller)
                if caller not in symbol_names or callee not in symbol_names:
                    # Track for potential cross-file resolution
                    self._pending_call_edges.append((caller_id, callee, metadata.path))
                    continue
                callee_id = self.make_symbol_id(metadata.path, callee)
                self._graph_edges.append(
                    GraphEdge(
                        src=caller_id,
                        dst=callee_id,
                        type="CALLS",
                        metadata={"path": metadata.path},
                    )
                )

        # Inheritance edges (child -> base) - uses unified IDs
        if self.graph_store and metadata.inherit_edges:
            for child, base in metadata.inherit_edges:
                child_id = self.make_symbol_id(metadata.path, child)
                if child not in symbol_names:
                    continue
                # if base is in current file, link directly, else resolve later
                if base in symbol_names:
                    base_id = self.make_symbol_id(metadata.path, base)
                    self._graph_edges.append(
                        GraphEdge(
                            src=child_id,
                            dst=base_id,
                            type="INHERITS",
                            metadata={"path": metadata.path},
                        )
                    )
                else:
                    self._pending_inherit_edges.append((child_id, base, metadata.path))

        # Implements edges (child -> interface/abstract) - uses unified IDs
        if self.graph_store and metadata.implements_edges:
            for child, base in metadata.implements_edges:
                child_id = self.make_symbol_id(metadata.path, child)
                if child not in symbol_names:
                    continue
                if base in symbol_names:
                    base_id = self.make_symbol_id(metadata.path, base)
                    self._graph_edges.append(
                        GraphEdge(
                            src=child_id,
                            dst=base_id,
                            type="IMPLEMENTS",
                            metadata={"path": metadata.path},
                        )
                    )
                else:
                    self._pending_implements_edges.append((child_id, base, metadata.path))

        # Composition edges (owner -> member type) - uses unified IDs
        if self.graph_store and metadata.compose_edges:
            for owner, member in metadata.compose_edges:
                owner_id = self.make_symbol_id(metadata.path, owner)
                if owner not in symbol_names:
                    continue
                if member in symbol_names:
                    member_id = self.make_symbol_id(metadata.path, member)
                    self._graph_edges.append(
                        GraphEdge(
                            src=owner_id,
                            dst=member_id,
                            type="COMPOSED_OF",
                            metadata={"path": metadata.path},
                        )
                    )
                else:
                    self._pending_compose_edges.append((owner_id, member, metadata.path))

        # Add IMPORTS edges from file to imported module names (cross-file reference scaffold)
        # Mark stdlib modules so they can be excluded from PageRank while keeping edges
        if self.graph_store and metadata.imports:
            for imp in metadata.imports:
                is_stdlib = _is_stdlib_module(imp)
                module_node_id = f"module:{imp}"  # module:pkg format for external modules
                self._graph_nodes.append(
                    GraphNode(
                        node_id=module_node_id,
                        type="module" if not is_stdlib else "stdlib_module",
                        name=imp,
                        file=metadata.path,
                        lang=metadata.language,
                    )
                )
                self._graph_edges.append(
                    GraphEdge(
                        src=self.make_file_id(metadata.path),
                        dst=module_node_id,
                        type="IMPORTS",
                        metadata={"path": metadata.path, "is_stdlib": is_stdlib},
                    )
                )

    def _resolve_cross_file_calls(self) -> None:
        """Resolve pending cross-file edges (CALLS, INHERITS, IMPLEMENTS, COMPOSED_OF)."""
        has_pending = (
            self._pending_call_edges
            or self._pending_inherit_edges
            or self._pending_implements_edges
            or self._pending_compose_edges
        )
        if not has_pending:
            return

        # Build resolver index from graph nodes - self.symbols keys are already unified IDs
        node_ids = list(self.symbols.keys())
        self._symbol_resolver.ingest(node_ids)

        # Track deduplicated phantom nodes for external types
        _seen_external: set[str] = set()

        # Cross-file CALLS resolution
        for caller_id, callee_name, file_path in self._pending_call_edges:
            target_id = self._symbol_resolver.resolve(callee_name, preferred_file=file_path)
            if not target_id:
                # Try short name heuristic (after ingest it exists already)
                target_id = self._symbol_resolver.resolve(
                    callee_name.split(".")[-1], preferred_file=file_path
                )
            if not target_id:
                continue
            self._graph_edges.append(
                GraphEdge(
                    src=caller_id,
                    dst=target_id,
                    type="CALLS",
                    metadata={"path": file_path, "resolved": True},
                )
            )

        # INHERITS resolution — create phantom nodes for external base types
        for child_id, base_name, file_path in self._pending_inherit_edges:
            target_id = self._symbol_resolver.resolve(base_name, preferred_file=file_path)
            if not target_id:
                target_id = self._symbol_resolver.resolve(
                    base_name.split(".")[-1], preferred_file=file_path
                )
            if not target_id:
                target_id = f"external_type:{base_name}"
                if target_id not in _seen_external:
                    _seen_external.add(target_id)
                    self._graph_nodes.append(
                        GraphNode(
                            node_id=target_id,
                            type="external_type",
                            name=base_name,
                            file=file_path,
                            metadata={"external": True},
                        )
                    )
            self._graph_edges.append(
                GraphEdge(
                    src=child_id,
                    dst=target_id,
                    type="INHERITS",
                    metadata={
                        "path": file_path,
                        "resolved": target_id.startswith("symbol:"),
                    },
                )
            )

        # IMPLEMENTS resolution — create phantom nodes for external traits/interfaces
        for child_id, base_name, file_path in self._pending_implements_edges:
            target_id = self._symbol_resolver.resolve(base_name, preferred_file=file_path)
            if not target_id:
                target_id = self._symbol_resolver.resolve(
                    base_name.split(".")[-1], preferred_file=file_path
                )
            if not target_id:
                target_id = f"external_type:{base_name}"
                if target_id not in _seen_external:
                    _seen_external.add(target_id)
                    self._graph_nodes.append(
                        GraphNode(
                            node_id=target_id,
                            type="external_type",
                            name=base_name,
                            file=file_path,
                            metadata={"external": True},
                        )
                    )
            self._graph_edges.append(
                GraphEdge(
                    src=child_id,
                    dst=target_id,
                    type="IMPLEMENTS",
                    metadata={
                        "path": file_path,
                        "resolved": target_id.startswith("symbol:"),
                    },
                )
            )

        # COMPOSES/has-a relationships — phantom nodes for non-primitive external types
        for owner_id, member_name, file_path in self._pending_compose_edges:
            target_id = self._symbol_resolver.resolve(member_name, preferred_file=file_path)
            if not target_id:
                target_id = self._symbol_resolver.resolve(
                    member_name.split(".")[-1], preferred_file=file_path
                )
            if not target_id:
                # Skip primitive/container types (too noisy for composition)
                if member_name in _PRIMITIVE_TYPES:
                    continue
                target_id = f"external_type:{member_name}"
                if target_id not in _seen_external:
                    _seen_external.add(target_id)
                    self._graph_nodes.append(
                        GraphNode(
                            node_id=target_id,
                            type="external_type",
                            name=member_name,
                            file=file_path,
                            metadata={"external": True},
                        )
                    )
            self._graph_edges.append(
                GraphEdge(
                    src=owner_id,
                    dst=target_id,
                    type="COMPOSED_OF",
                    metadata={
                        "path": file_path,
                        "resolved": target_id.startswith("symbol:"),
                    },
                )
            )

        # REFERENCES edges (file -> symbol) for any referenced identifier
        for metadata in self.files.values():
            if not metadata.references:
                continue
            file_node = f"file:{metadata.path}"
            for ref in metadata.references:
                target_id = self._symbol_resolver.resolve(ref, preferred_file=metadata.path)
                if not target_id:
                    target_id = self._symbol_resolver.resolve(
                        ref.split(".")[-1], preferred_file=metadata.path
                    )
                if not target_id:
                    continue
                self._graph_edges.append(
                    GraphEdge(
                        src=file_node,
                        dst=target_id,
                        type="REFERENCES",
                        metadata={"path": metadata.path, "resolved": True},
                    )
                )

    def _build_dependency_graph(self) -> None:
        """Build dependency graph between files."""
        for _file_path, metadata in self.files.items():
            for imp in metadata.imports:
                # Try to resolve import to file path
                # This is a simplified version - full implementation would be more robust
                possible_paths = [
                    f"{imp.replace('.', '/')}.py",
                    f"{imp.replace('.', '/')}/__init__.py",
                ]

                for possible_path in possible_paths:
                    if possible_path in self.files:
                        metadata.dependencies.append(possible_path)
                        break

    async def find_relevant_files(
        self,
        query: str,
        max_files: int = 10,
        auto_reindex: bool = True,
    ) -> List[FileMetadata]:
        """Find files relevant to a query.

        Automatically reindexes if the index is stale (lazy reindexing).

        Args:
            query: Search query
            max_files: Maximum number of files to return
            auto_reindex: If True, automatically reindex when stale

        Returns:
            List of relevant file metadata
        """
        # Lazy reindexing - ensure index is up to date
        await self.ensure_indexed(auto_reindex=auto_reindex)

        results = []

        # Simple keyword search for now
        query_lower = query.lower()

        for file_path, metadata in self.files.items():
            # Check if query matches:
            # 1. File name
            # 2. Symbol names
            # 3. Imports
            relevance_score = 0

            if query_lower in file_path.lower():
                relevance_score += 10

            for symbol in metadata.symbols:
                if query_lower in symbol.name.lower():
                    relevance_score += 5
                if symbol.docstring and query_lower in symbol.docstring.lower():
                    relevance_score += 3

            for imp in metadata.imports:
                if query_lower in imp.lower():
                    relevance_score += 2

            if relevance_score > 0:
                results.append((relevance_score, metadata))

        # Sort by relevance and return top N
        results.sort(key=lambda x: x[0], reverse=True)
        return [metadata for _, metadata in results[:max_files]]

    def find_symbol(self, symbol_name: str) -> Optional[Symbol]:
        """Find a symbol by name.

        Args:
            symbol_name: Name of symbol to find

        Returns:
            Symbol if found, None otherwise
        """
        # Search all files
        for _key, symbol in self.symbols.items():
            if symbol.name == symbol_name:
                return symbol
        return None

    def get_file_context(self, file_path: str) -> Dict[str, Any]:
        """Get full context for a file including dependencies.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file context
        """
        if file_path not in self.files:
            return {}

        metadata = self.files[file_path]

        return {
            "file": metadata,
            "symbols": metadata.symbols,
            "imports": metadata.imports,
            "dependencies": [self.files[dep] for dep in metadata.dependencies if dep in self.files],
            "dependents": self._find_dependents(file_path),
        }

    def _find_dependents(self, file_path: str) -> List[FileMetadata]:
        """Find files that depend on this file."""
        dependents = []
        for metadata in self.files.values():
            if file_path in metadata.dependencies:
                dependents.append(metadata)
        return dependents

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics including staleness information."""
        with self._staleness_lock:
            is_stale = self._is_stale
            changed_count = len(self._changed_files)
            last_indexed = self._last_indexed

        stats = {
            "total_files": len(self.files),
            "total_symbols": len(self.symbols),
            "total_lines": sum(f.lines for f in self.files.values()),
            "languages": {"python": len(self.files)},
            "embeddings_enabled": self.use_embeddings,
            "is_indexed": self._is_indexed,
            "is_stale": is_stale,
            "changed_files_count": changed_count,
            "last_indexed": last_indexed,
            "watcher_enabled": self._watcher_enabled,
            "watcher_running": self._observer is not None,
        }
        if self.use_embeddings and self.embedding_provider:
            stats["embedding_stats"] = asyncio.run(self.embedding_provider.get_stats())
        return stats

    def _initialize_embeddings(self, config: Optional[Dict[str, Any]]) -> None:
        """Initialize embedding provider.

        Embeddings are stored in {rootrepo}/.victor/embeddings/ directory by default.
        This keeps all index data co-located with the repository.

        Configuration is read from settings.py with sensible defaults:
        - vector_store: lancedb (disk-based ANN, lower memory)
        - embedding_model: BAAI/bge-small-en-v1.5 (384-dim, excellent for code)

        Args:
            config: Embedding configuration dict (overrides settings if provided)
        """
        try:
            from victor_coding.codebase.embeddings import (
                EmbeddingConfig,
                EmbeddingRegistry,
            )

            # Create config with defaults from settings
            if not config:
                config = {}

            # Load settings for defaults
            from victor_coding.compat.settings import get_project_paths, load_settings

            settings = load_settings()
            default_persist_dir = get_project_paths(self.root).embeddings_dir

            # Use settings as defaults, allow config to override
            embedding_config = EmbeddingConfig(
                vector_store=config.get(
                    "vector_store",
                    getattr(settings, "codebase_vector_store", "lancedb"),
                ),
                embedding_model_type=config.get(
                    "embedding_model_type",
                    getattr(settings, "codebase_embedding_provider", "sentence-transformers"),
                ),
                embedding_model_name=config.get(
                    "embedding_model_name",
                    getattr(settings, "codebase_embedding_model", "BAAI/bge-small-en-v1.5"),
                ),
                persist_directory=config.get("persist_directory", str(default_persist_dir)),
                extra_config=config.get("extra_config", {}),
            )

            # Create embedding provider
            self.embedding_provider = EmbeddingRegistry.create(embedding_config)

            # Detect if persistent embeddings already exist on disk.
            # If so, mark as indexed to avoid full rebuild when ensure_indexed() is called.
            persist_dir = Path(embedding_config.persist_directory)
            if persist_dir.exists() and any(persist_dir.iterdir()):
                self._is_indexed = True
                self._last_indexed = persist_dir.stat().st_mtime
                logger.info(
                    f"Detected existing embeddings at {persist_dir}, " "skipping full rebuild"
                )
            else:
                print(
                    f"✓ Embeddings enabled: {embedding_config.embedding_model_name} + "
                    f"{embedding_config.vector_store}"
                )
                print(f"  Storage: {embedding_config.persist_directory}")

        except ImportError as e:
            print(f"⚠️  Warning: Embeddings not available: {e}")
            print("   Install with: pip install chromadb sentence-transformers")
            self.use_embeddings = False
            self.embedding_provider = None

    async def semantic_search(
        self,
        query: str,
        max_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        auto_reindex: bool = True,
        similarity_threshold: Optional[float] = None,
        expand_query: bool = True,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings with query expansion and threshold filtering.

        Automatically reindexes if the index is stale (lazy reindexing).

        Args:
            query: Search query (natural language)
            max_results: Maximum number of results
            filter_metadata: Optional metadata filters
            auto_reindex: If True, automatically reindex when stale
            similarity_threshold: Minimum similarity score [0.0-1.0] to include result
            expand_query: Enable query expansion with synonyms/related terms

        Returns:
            List of search results with file paths, symbols, and relevance scores
        """
        if not self.use_embeddings or not self.embedding_provider:
            raise ValueError("Embeddings not enabled. Initialize with use_embeddings=True")

        # Lazy reindexing - ensure index is up to date
        await self.ensure_indexed(auto_reindex=auto_reindex)

        # Ensure provider is initialized
        if not self.embedding_provider._initialized:
            await self.embedding_provider.initialize()

        # Query expansion to improve recall (fix false negatives)
        queries_to_search = [query]
        if expand_query:
            from victor_coding.codebase.query_expander import expand_query as expand_fn

            queries_to_search = expand_fn(query, max_expansions=5)
            if len(queries_to_search) > 1:
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(
                    f"Semantic search: expanded '{query}' to {len(queries_to_search)} queries"
                )

        # Search with all query variations and merge results
        all_results = []
        seen_docs = set()  # Deduplicate by (file_path, line_number)

        for search_query in queries_to_search:
            results = await self.embedding_provider.search_similar(
                query=search_query,
                limit=max_results * 2,  # Get more for dedup/filtering
                filter_metadata=filter_metadata,
            )

            for result in results:
                # Deduplicate by file + line
                doc_key = (result.file_path, result.line_number or 0)
                if doc_key not in seen_docs:
                    all_results.append(result)
                    seen_docs.add(doc_key)

        # Apply similarity threshold if specified
        if similarity_threshold is not None:
            all_results = [r for r in all_results if r.score >= similarity_threshold]
            if not all_results:
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(
                    f"Semantic search: threshold {similarity_threshold:.2f} filtered all results. "
                    "Consider lowering threshold or checking query."
                )

        # Sort by score (highest first) and limit
        all_results.sort(key=lambda r: r.score, reverse=True)
        all_results = all_results[:max_results]

        # Convert to dict format with end_line for precise reads
        return [
            {
                "file_path": result.file_path,
                "symbol_name": result.symbol_name,
                "content": result.content,
                "score": result.score,
                "line_number": result.line_number,
                "end_line": result.metadata.get("end_line"),  # For precise reads
                "metadata": result.metadata,
            }
            for result in all_results
        ]

    def _build_symbol_context(self, symbol: Symbol) -> str:
        """Build context string for a symbol (for embedding).

        Args:
            symbol: Symbol to build context for

        Returns:
            Context string combining symbol information
        """
        parts = [
            f"Symbol: {symbol.name}",
            f"Type: {symbol.type}",
            f"File: {symbol.file_path}",
        ]

        if symbol.signature:
            parts.append(f"Signature: {symbol.signature}")

        if symbol.docstring:
            parts.append(f"Documentation: {symbol.docstring}")

        return "\n".join(parts)


class SymbolVisitor(ast.NodeVisitor):
    """AST visitor to extract symbols from Python code."""

    def __init__(self, metadata: FileMetadata):
        self.metadata = metadata
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None
        self.call_edges: List[tuple[str, str]] = []
        self.composition_edges: List[tuple[str, str]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        bases = extract_base_classes(node)
        symbol = Symbol(
            name=node.name,
            type="class",
            file_path=self.metadata.path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            base_classes=bases,
        )
        self.metadata.symbols.append(symbol)

        # Visit class methods
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        name = node.name
        if self.current_class:
            name = f"{self.current_class}.{name}"

        signature = build_signature(node)

        symbol = Symbol(
            name=name,
            type="function",
            file_path=self.metadata.path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            signature=signature,
        )
        self.metadata.symbols.append(symbol)
        old_function = self.current_function
        self.current_function = name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement."""
        for alias in node.names:
            self.metadata.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from...import statement."""
        if node.module:
            self.metadata.imports.append(node.module)

    def visit_Call(self, node: ast.Call) -> None:
        """Capture simple call relationships for intra-file graph edges."""
        if self.current_function:
            callee = None
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr

            if callee:
                self.call_edges.append((self.current_function, callee))

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Detect has-a relationships for class attributes."""
        if self.current_class:
            target_type: Optional[str] = None
            if isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name):
                    target_type = func.id
                elif isinstance(func, ast.Attribute):
                    target_type = func.attr
            elif isinstance(node.value, ast.Name):
                target_type = node.value.id
            if target_type:
                self.composition_edges.append((self.current_class, target_type))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Capture annotated attributes inside classes for composition edges."""
        if self.current_class:
            target_type: Optional[str] = None
            if isinstance(node.annotation, ast.Name):
                target_type = node.annotation.id
            elif isinstance(node.annotation, ast.Attribute):
                target_type = node.annotation.attr
            if target_type:
                self.composition_edges.append((self.current_class, target_type))
        self.generic_visit(node)


# TODO: Future enhancements
# [DONE] 1. Add semantic search with embeddings (ChromaDB, LanceDB)
# 2. Add support for more languages (JavaScript, TypeScript, Go, etc.)
# [DONE] 3. Add incremental indexing (only reindex changed files)
# [DONE] 4. Add file watching for automatic staleness detection
# [DONE] 5. Add background indexer service for periodic updates
# 6. Add symbol reference tracking (who calls what)
# 7. Add type information extraction
# 8. Add test coverage mapping
# 9. Add documentation extraction
# 10. Add complexity metrics


# =============================================================================
# Background Indexer Service
# =============================================================================


class BackgroundIndexerService:
    """Background service for periodic incremental reindexing.

    Uses mtime-based change detection to efficiently update the index
    without blocking user operations. Runs as a daemon thread.

    Features:
    - Periodic polling (configurable interval, default 60s)
    - mtime-based change detection (no unnecessary work)
    - Graceful shutdown on session end
    - Thread-safe singleton pattern
    """

    _instance: Optional["BackgroundIndexerService"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        root: Path,
        interval_seconds: float = 60.0,
        auto_start: bool = False,
    ):
        self.root = root
        self.interval_seconds = interval_seconds
        self._indexer: Optional["CodebaseIndex"] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._last_run: Optional[float] = None
        self._stats: Dict[str, Any] = {"runs": 0, "files_updated": 0, "errors": 0}

        if auto_start:
            self.start()

    @classmethod
    def get_instance(
        cls,
        root: Optional[Path] = None,
        interval_seconds: float = 60.0,
    ) -> "BackgroundIndexerService":
        """Get or create the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                if root is None:
                    root = Path.cwd()
                cls._instance = cls(root, interval_seconds)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for testing."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None

    def start(self) -> None:
        """Start the background indexer thread."""
        if self._running:
            logger.debug("Background indexer already running")
            return

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="BackgroundIndexer",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"Background indexer started (interval={self.interval_seconds}s, root={self.root})"
        )

    def stop(self) -> None:
        """Stop the background indexer gracefully."""
        if not self._running:
            return

        self._stop_event.set()
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        logger.info("Background indexer stopped")

    def _run_loop(self) -> None:
        """Main loop for periodic reindexing."""
        while not self._stop_event.is_set():
            try:
                self._run_incremental_reindex()
            except Exception as e:
                logger.warning(f"Background indexer error: {e}")
                self._stats["errors"] += 1

            # Wait for next interval or stop event
            self._stop_event.wait(self.interval_seconds)

    def _run_incremental_reindex(self) -> None:
        """Perform incremental reindex using mtime detection."""
        if self._indexer is None:
            self._indexer = CodebaseIndex(self.root)

        # Only reindex if there are changes (mtime-based detection)
        if not self._indexer._is_stale and self._indexer._is_indexed:
            logger.debug("Background indexer: no changes detected, skipping")
            return

        # Run incremental reindex in asyncio
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self._indexer.incremental_reindex())
            self._stats["runs"] += 1
            self._stats["files_updated"] += len(result.get("updated", []))
            self._stats["files_updated"] += len(result.get("added", []))
            self._last_run = time.time()

            total_changes = (
                len(result.get("updated", []))
                + len(result.get("added", []))
                + len(result.get("removed", []))
            )
            if total_changes > 0:
                logger.info(
                    f"Background reindex: {len(result.get('updated', []))} updated, "
                    f"{len(result.get('added', []))} added, "
                    f"{len(result.get('removed', []))} removed"
                )
        finally:
            loop.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get background indexer statistics."""
        return {
            **self._stats,
            "running": self._running,
            "interval_seconds": self.interval_seconds,
            "last_run": self._last_run,
        }


def start_background_indexer(
    root: Optional[Path] = None,
    interval_seconds: float = 60.0,
) -> BackgroundIndexerService:
    """Start the background indexer service (convenience function).

    Args:
        root: Project root directory (defaults to cwd)
        interval_seconds: Polling interval in seconds (default 60)

    Returns:
        BackgroundIndexerService instance
    """
    service = BackgroundIndexerService.get_instance(root, interval_seconds)
    service.start()
    return service
