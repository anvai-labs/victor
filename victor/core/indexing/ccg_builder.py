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

"""Code Context Graph (CCG) builder - implements GraphCoder methodology.

This module builds three types of program graphs from source code:
1. **Control Flow Graph (CFG)**: Represents execution paths through code
2. **Control Dependence Graph (CDG)**: Represents control dependencies between statements
3. **Data Dependence Graph (DDG)**: Represents variable definitions and uses

Based on research from:
- GraphCoder: Enhancing Code Generation with Control Flow Graphs
- GraphCodeAgent: Multi-Granularity Dependency Graphs for Code Analysis

Usage:
    builder = CodeContextGraphBuilder(graph_store)
    nodes, edges = await builder.build_ccg_for_file(
        file_path=Path("src/main.py"),
        language="python"
    )
    await graph_store.upsert_nodes(nodes)
    await graph_store.upsert_edges(edges)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.storage.graph.protocol import GraphEdge, GraphNode, GraphStoreProtocol

logger = logging.getLogger(__name__)


def _get_graph_types() -> tuple[type, type, type]:
    """Lazy import of graph types."""
    from victor.storage.graph.protocol import GraphEdge, GraphNode
    from victor.storage.graph.edge_types import EdgeType

    return GraphNode, GraphEdge, EdgeType


# Supported languages for CCG building
SUPPORTED_CCG_LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "go",
    "rust",
    "java",
    "c",
    "cpp",
    "c_sharp",
}


class StatementType(str, Enum):
    """Statement types for CCG node classification."""

    # Control flow
    CONDITION = "condition"
    LOOP = "loop"
    TRY = "try"
    CATCH = "catch"
    FINALLY = "finally"
    SWITCH = "switch"
    CASE = "case"
    DEFAULT = "default"

    # Data operations
    ASSIGNMENT = "assignment"
    CALL = "call"
    RETURN = "return"
    YIELD = "yield"
    AWAIT = "await"
    THROW = "throw"

    # Definitions
    FUNCTION_DEF = "function_def"
    CLASS_DEF = "class_def"
    VARIABLE_DEF = "variable_def"

    # Other
    BLOCK = "block"
    EXPRESSION = "expression"
    UNKNOWN = "unknown"


@dataclass
class BasicBlock:
    """A basic block in the control flow graph.

    A basic block is a maximal sequence of consecutive instructions
    with a single entry point and single exit point.
    """

    block_id: str
    entry_line: int
    exit_line: int
    statements: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    is_loop_body: bool = False
    is_conditional: bool = False
    loop_type: Optional[str] = None  # for, while, do_while
    condition: Optional[str] = None


@dataclass
class VariableInfo:
    """Information about a variable for DDG construction."""

    name: str
    defining_node: str  # node_id where this variable is defined
    use_sites: List[str] = field(default_factory=list)  # node_ids where it's used
    scope: str = ""  # scope_id for hierarchical tracking
    type: Optional[str] = None


class CodeContextGraphBuilder:
    """Builds CFG, CDG, and DDG from source code using Tree-sitter.

    This class implements the GraphCoder methodology for constructing
    program graphs that capture:
    - Control flow relationships (CFG)
    - Control dependencies (CDG)
    - Data dependencies (DDG)

    The builder will first check the CapabilityRegistry for enhanced
    builders registered by external packages (e.g., victor-coding).
    If no enhanced builder is available, it falls back to the built-in
    implementation.

    Attributes:
        graph_store: The graph store for persisting nodes and edges
        language: The programming language being analyzed
    """

    def __init__(
        self,
        graph_store: Optional[GraphStoreProtocol] = None,
        language: str = "python",
    ) -> None:
        """Initialize the CCG builder.

        Args:
            graph_store: Optional graph store for persistence
            language: Programming language (must be in SUPPORTED_CCG_LANGUAGES)
        """
        self.graph_store = graph_store
        self.language = language.lower()
        self._tree_sitter_parser: Optional[Any] = None
        self._tree_sitter_language: Optional[Any] = None
        self._tree_sitter_parser_cache: Dict[str, Any] = {}

        # Try to get enhanced builder from capability registry
        self._enhanced_builder = self._get_enhanced_builder(language)
        self._enhanced_builder_cache: Dict[str, Optional[Any]] = {}

        if self.language not in SUPPORTED_CCG_LANGUAGES:
            logger.warning(
                f"Language '{self.language}' not in supported CCG languages. "
                f"Supported: {SUPPORTED_CCG_LANGUAGES}. "
                f"CCG building may be limited."
            )

    def _get_enhanced_builder(self, language: str) -> Optional[Any]:
        """Try to get an enhanced CCG builder from the capability registry.

        Args:
            language: Language identifier

        Returns:
            Enhanced builder if available, None otherwise
        """
        try:
            from victor.core.capability_registry import CapabilityRegistry
            from victor.framework.vertical_protocols import CCGBuilderProtocol

            registry = CapabilityRegistry.get_instance()
            provider = registry.get(CCGBuilderProtocol)

            if provider is not None and hasattr(provider, "supports_language"):
                if provider.supports_language(language):
                    logger.debug(f"Using enhanced CCG builder for {language}")
                    return provider
        except Exception as e:
            logger.debug(f"Could not get enhanced CCG builder: {e}")

        return None

    def _enhanced_builder_supports_language(self, builder: Any, language: str) -> bool:
        """Return whether an enhanced builder can handle a language."""
        supports_language = getattr(builder, "supports_language", None)
        return not callable(supports_language) or bool(supports_language(language))

    def _resolve_enhanced_builder(self, language: str) -> Optional[Any]:
        """Resolve an enhanced builder for the current file language.

        A builder instance can be reused across languages if it advertises support.
        This matters for mixed-language batches where the parent builder may have
        been constructed with a default language such as Python.
        """
        lang = language.lower()
        if lang in self._enhanced_builder_cache:
            return self._enhanced_builder_cache[lang]

        if self._enhanced_builder is not None and self._enhanced_builder_supports_language(
            self._enhanced_builder, lang
        ):
            self._enhanced_builder_cache[lang] = self._enhanced_builder
            return self._enhanced_builder

        builder = self._get_enhanced_builder(lang)
        self._enhanced_builder_cache[lang] = builder
        if builder is not None:
            self._enhanced_builder = builder
        return builder

    async def build_ccg_for_file(
        self,
        file_path: Path,
        language: str | None = None,
    ) -> Tuple[List[Any], List[Any]]:
        """Build all three graphs (CFG, CDG, DDG) for a file.

        Args:
            file_path: Path to the source file
            language: Optional language override (auto-detected from extension if None)

        Returns:
            Tuple of (nodes, edges) for the complete CCG
        """
        lang = (language or self._detect_language(file_path)).lower()

        # Delegate to enhanced builder if available for this file's actual language.
        enhanced_builder = self._resolve_enhanced_builder(lang)
        if enhanced_builder:
            try:
                if self._enhanced_builder_supports_language(enhanced_builder, lang):
                    return await enhanced_builder.build_ccg_for_file(file_path, lang)
            except Exception as e:
                logger.warning(f"Enhanced CCG builder failed: {e}, falling back to built-in")

        if lang not in SUPPORTED_CCG_LANGUAGES:
            logger.debug(f"Skipping CCG for unsupported language: {lang}")
            return [], []

        try:
            source_code = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return [], []

        previous_language = self.language
        self.language = lang
        try:
            # Parse the AST
            ast_root = self._parse_source(source_code, file_path, lang)
            if ast_root is None:
                return [], []

            # Build all three graphs
            cfg_nodes, cfg_edges = await self._build_cfg(ast_root, file_path, source_code)
            cdg_edges = await self._build_cdg(cfg_nodes, cfg_edges)
            ddg_edges = await self._build_ddg(ast_root, cfg_nodes, file_path, source_code)

            all_nodes = cfg_nodes
            all_edges = cfg_edges + cdg_edges + ddg_edges
        finally:
            self.language = previous_language

        logger.debug(
            f"Built CCG for {file_path}: " f"{len(all_nodes)} nodes, {len(all_edges)} edges"
        )

        return all_nodes, all_edges

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension.

        Args:
            file_path: Path to the source file

        Returns:
            Language identifier string
        """
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "c_sharp",
        }
        return ext_map.get(file_path.suffix.lower(), "unknown")

    def _get_tree_sitter_parser(self, language: str) -> Optional[Any]:
        """Get or create a cached Tree-sitter parser for a language."""
        cached = self._tree_sitter_parser_cache.get(language)
        if cached is not None:
            return cached

        try:
            import tree_sitter as ts
        except ImportError:
            logger.debug("Tree-sitter not available, skipping CCG construction")
            return None

        lang_modules = {
            "python": ("tree_sitter_python", "language"),
            "javascript": ("tree_sitter_javascript", "language"),
            "typescript": ("tree_sitter_typescript", "language_typescript"),
            "go": ("tree_sitter_go", "language"),
            "rust": ("tree_sitter_rust", "language"),
            "java": ("tree_sitter_java", "language"),
            "c": ("tree_sitter_c", "language"),
            "cpp": ("tree_sitter_cpp", "language"),
        }

        if language not in lang_modules:
            logger.debug(f"Language {language} not in CCG language modules")
            return None

        try:
            module_name, func_name = lang_modules[language]
            lang_module = __import__(module_name)
            lang_func = getattr(lang_module, func_name)
            lang_obj = lang_func()
            ts_language = (
                ts.Language(lang_obj) if not isinstance(lang_obj, ts.Language) else lang_obj
            )

            parser = ts.Parser(ts_language)
            self._tree_sitter_parser_cache[language] = parser
            return parser
        except (AttributeError, ValueError, ImportError) as e:
            logger.debug(f"Tree-sitter parser not available for {language}: {e}")
            return None

    def _parse_source(
        self,
        source_code: str,
        file_path: Path,
        language: Optional[str] = None,
    ) -> Optional[Any]:
        """Parse source code using Tree-sitter.

        Args:
            source_code: Source code to parse
            file_path: Path for error reporting

        Returns:
            Tree-sitter AST root node, or None if parsing fails
        """
        try:
            lang = (language or self.language).lower()
            parser = self._get_tree_sitter_parser(lang)
            if parser is None:
                return None

            tree = parser.parse(bytes(source_code, "utf-8"))
            return tree.root_node

        except Exception as e:
            logger.debug(f"Failed to parse {file_path} with Tree-sitter: {e}")
            return None

    async def _build_cfg(
        self,
        ast_root: Any,
        file_path: Path,
        source_code: str,
    ) -> Tuple[List[Any], List[Any]]:
        """Build Control Flow Graph from AST.

        Args:
            ast_root: Tree-sitter AST root node
            file_path: Path to source file
            source_code: Source code for extracting line numbers

        Returns:
            Tuple of (nodes, edges) for the CFG
        """
        GraphNode, GraphEdge, EdgeType = _get_graph_types()

        nodes: List[Any] = []
        edges: List[Any] = []

        if ast_root is None:
            return nodes, edges

        try:
            source_lines = source_code.splitlines()
        except Exception:
            source_lines = []

        # Walk the AST and build CFG nodes
        statement_nodes = self._extract_statement_nodes(ast_root, file_path, source_lines)
        nodes.extend(statement_nodes)

        # Build CFG edges by analyzing control flow
        cfg_edges = self._build_cfg_edges(statement_nodes, ast_root, file_path)
        edges.extend(cfg_edges)

        return nodes, edges

    def _extract_statement_nodes(
        self,
        ast_root: Any,
        file_path: Path,
        source_lines: List[str],
    ) -> List[Any]:
        """Extract statement-level nodes from AST.

        Args:
            ast_root: Tree-sitter AST root node
            file_path: Path to source file
            source_lines: Source code lines for content extraction

        Returns:
            List of GraphNode instances representing statements
        """
        GraphNode, _, _ = _get_graph_types()

        nodes: List[Any] = []
        file_str = str(file_path)

        def walk(node: Any, parent_scope: str = "") -> None:
            """Recursively walk the AST and extract statement nodes."""

            if hasattr(node, "type") and hasattr(node, "children"):
                node_type = node.type

                # Determine if this is a statement we should track
                statement_type = self._classify_statement(node_type)
                if statement_type != StatementType.UNKNOWN:
                    # Get line numbers
                    start_line = node.start_point[0] + 1 if hasattr(node, "start_point") else 0
                    end_line = node.end_point[0] + 1 if hasattr(node, "end_point") else 0

                    # Extract statement content
                    content = ""
                    if 0 <= start_line - 1 < len(source_lines):
                        content = source_lines[start_line - 1].strip()

                    # Generate unique node ID
                    node_id = self._generate_statement_id(file_str, start_line, end_line, node_type)

                    # Determine visibility
                    visibility = self._determine_visibility(node, content)

                    # Create the node
                    graph_node = GraphNode(
                        node_id=node_id,
                        type="statement",
                        name=f"{node_type}:{start_line}",
                        file=file_str,
                        line=start_line,
                        end_line=end_line,
                        lang=self.language,
                        ast_kind=node_type,
                        scope_id=parent_scope or None,
                        statement_type=statement_type.value,
                        visibility=visibility,
                        metadata={
                            "node_type": node_type,
                            "content_preview": content[:100] if content else "",
                        },
                    )
                    nodes.append(graph_node)

                    # Update scope for children
                    new_scope = node_id
                else:
                    new_scope = parent_scope

                # Recurse into children
                for child in node.children:
                    walk(child, new_scope)

        walk(ast_root)
        return nodes

    def _classify_statement(self, node_type: str) -> StatementType:
        """Classify a Tree-sitter node type into a StatementType.

        Args:
            node_type: Tree-sitter node type string

        Returns:
            Classified StatementType
        """
        # Python-specific mappings
        if self.language == "python":
            if node_type in {"if_statement", "elif_clause"}:
                return StatementType.CONDITION
            if node_type in {"for_statement", "while_statement"}:
                return StatementType.LOOP
            if node_type == "try_statement":
                return StatementType.TRY
            if node_type in {"except_clause", "except_block"}:
                return StatementType.CATCH
            if node_type == "finally_clause":
                return StatementType.FINALLY
            if node_type == "match_statement":
                return StatementType.SWITCH
            if node_type in {"match_clause", "case_clause"}:
                return StatementType.CASE
            if node_type in {"assignment", "augmented_assignment", "named_expression"}:
                return StatementType.ASSIGNMENT
            if node_type == "call":
                return StatementType.CALL
            if node_type == "return_statement":
                return StatementType.RETURN
            if node_type == "yield_statement":
                return StatementType.YIELD
            if node_type in {"function_definition", "async_function_definition"}:
                return StatementType.FUNCTION_DEF
            if node_type == "class_definition":
                return StatementType.CLASS_DEF
            if node_type in {"expression_statement", "expression_list"}:
                return StatementType.EXPRESSION

        # JavaScript/TypeScript mappings
        elif self.language in {"javascript", "typescript"}:
            if node_type in {"if_statement", "else_clause"}:
                return StatementType.CONDITION
            if node_type in {
                "for_statement",
                "for_in_statement",
                "while_statement",
                "do_statement",
            }:
                return StatementType.LOOP
            if node_type == "try_statement":
                return StatementType.TRY
            if node_type in {"catch_clause", "finally_clause"}:
                return StatementType.CATCH
            if node_type == "switch_statement":
                return StatementType.SWITCH
            if node_type in {"switch_case", "switch_default"}:
                return StatementType.CASE
            if node_type in {
                "variable_declaration",
                "lexical_declaration",
                "assignment_expression",
                "augmented_assignment_expression",
            }:
                return StatementType.ASSIGNMENT
            if node_type == "call_expression":
                return StatementType.CALL
            if node_type == "return_statement":
                return StatementType.RETURN
            if node_type in {
                "function_declaration",
                "function_expression",
                "arrow_function",
                "method_definition",
            }:
                return StatementType.FUNCTION_DEF
            if node_type == "class_declaration":
                return StatementType.CLASS_DEF
            if node_type == "expression_statement":
                # Parity with the Python classifier: without this, statements
                # like `x -= 1;` anchor nothing (no loop back-edges in JS/TS).
                return StatementType.EXPRESSION

        # Go mappings
        elif self.language == "go":
            if node_type == "if_statement":
                return StatementType.CONDITION
            if node_type in {"for_statement", "for_range"}:
                return StatementType.LOOP
            if node_type in {"assignment_statement", "short_var_declaration"}:
                return StatementType.ASSIGNMENT
            if node_type == "call_expression":
                return StatementType.CALL
            if node_type == "return_statement":
                return StatementType.RETURN
            if node_type == "function_declaration":
                return StatementType.FUNCTION_DEF
            if node_type == "type_declaration":
                return StatementType.CLASS_DEF

        return StatementType.UNKNOWN

    def _determine_visibility(self, node: Any, content: str) -> Optional[str]:
        """Determine visibility of a node (public/private/protected).

        Args:
            node: Tree-sitter AST node
            content: Statement content

        Returns:
            Visibility string or None
        """
        content_lower = content.lower()

        # Python: check for underscore prefix
        if self.language == "python":
            if content.startswith("_") and not content.startswith("__"):
                return "private"
            if content.startswith("__") and not content.endswith("__"):
                return "private"
            if content.startswith("__") and content.endswith("__"):
                return "public"  # dunder methods are public
            return "public"

        # JavaScript/TypeScript
        elif self.language in {"javascript", "typescript"}:
            if "private" in content_lower:
                return "private"
            if "protected" in content_lower:
                return "protected"
            if "public" in content_lower:
                return "public"
            return "public"

        return None

    def _generate_statement_id(
        self, file_path: str, start_line: int, end_line: int, node_type: str
    ) -> str:
        """Generate a unique stable ID for a statement node.

        Args:
            file_path: Path to source file
            start_line: Starting line number
            end_line: Ending line number
            node_type: Tree-sitter node type

        Returns:
            Unique node ID string
        """
        # Create a stable hash based on location
        id_string = f"{file_path}:{start_line}:{end_line}:{node_type}"
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]

    def _build_cfg_edges(
        self,
        nodes: List[Any],
        ast_root: Any,
        file_path: Path,
    ) -> List[Any]:
        """Build CFG edges between statement nodes.

        Args:
            nodes: List of statement nodes
            ast_root: Tree-sitter AST root
            file_path: Path to source file

        Returns:
            List of GraphEdge instances representing CFG
        """
        _, GraphEdge, EdgeType = _get_graph_types()

        edges: List[Any] = []

        if not nodes:
            return edges

        # Sort nodes by line number
        sorted_nodes = sorted(nodes, key=lambda n: n.line or 0)

        # Build simple sequential CFG edges
        for i in range(len(sorted_nodes) - 1):
            current = sorted_nodes[i]
            next_node = sorted_nodes[i + 1]

            # Check if we should create a CFG edge
            if self._should_connect_cfg(current, next_node):
                edge_type = self._determine_cfg_edge_type(current, next_node)
                edge = GraphEdge(
                    src=current.node_id,
                    dst=next_node.node_id,
                    type=edge_type,
                    weight=1.0,
                    metadata={"file": current.file} if current.file else {},
                )
                edges.append(edge)

        # Structural edges from the AST (closes the old TODO): false branches,
        # loop back-edges, exception/catch/finally flow, switch/match dispatch.
        seen = {(e.src, e.dst, e.type) for e in edges}
        for edge in self._build_structural_cfg_edges(sorted_nodes, ast_root):
            key = (edge.src, edge.dst, edge.type)
            if key not in seen:
                seen.add(key)
                edges.append(edge)

        return edges

    # AST node types per construct. Node types are grammar-specific, so union
    # sets are safe to apply across every CCG language (a Go tree simply has no
    # `except_clause` nodes, a Python tree no `switch_statement`).
    _IF_NODES = frozenset({"if_statement", "if_expression"})
    _ELSE_NODES = frozenset({"elif_clause", "else_clause", "else_statement"})
    _LOOP_NODES = frozenset(
        {
            "for_statement",
            "while_statement",
            "do_statement",
            "for_in_statement",
            "for_of_statement",
            "enhanced_for_statement",
            "for_range_loop",
            "loop_expression",
            "while_expression",
            "for_expression",
        }
    )
    _TRY_NODES = frozenset({"try_statement", "try_expression"})
    _CATCH_NODES = frozenset({"except_clause", "except_group_clause", "catch_clause"})
    _FINALLY_NODES = frozenset({"finally_clause"})
    _SWITCH_NODES = frozenset(
        {"switch_statement", "switch_expression", "match_statement", "match_expression"}
    )
    _CASE_NODES = frozenset(
        {
            "case_clause",
            "switch_case",
            "case_statement",
            "match_arm",
            "switch_block_statement_group",
            "switch_rule",
        }
    )
    _DEFAULT_CASE_NODES = frozenset({"switch_default", "default_case"})

    def _statement_id_at_line(self, nodes: List[Any], line: int) -> Optional[str]:
        """Statement node id for a line: exact start match, else narrowest span.

        Compound statements span their whole body, so plain containment would
        resolve an inner line to the outermost construct; preferring the exact
        start line and then the narrowest containing span picks the innermost
        statement instead.
        """
        best_id: Optional[str] = None
        best_span = float("inf")
        for node in nodes:
            if not node.line:
                continue
            if node.line == line:
                return node.node_id
            if node.line <= line <= (node.end_line or node.line):
                span = (node.end_line or node.line) - node.line
                if span < best_span:
                    best_span = span
                    best_id = node.node_id
        return best_id

    def _build_structural_cfg_edges(self, nodes: List[Any], ast_root: Any) -> List[Any]:
        """CFG edges derived from AST structure rather than line adjacency.

        Emits, per construct:
        - if/elif/else:  condition -> alternative head   (CFG_FALSE_BRANCH)
        - loops:         last body statement -> header   (CFG_LOOP_BACK)
        - try:           try head -> each handler        (CFG_CATCH / CFG_FINALLY)
        - switch/match:  dispatch head -> each case head (CFG_CASE / CFG_DEFAULT)

        Best-effort: an edge is only emitted when both endpoints resolve to
        extracted statement nodes.
        """
        _, GraphEdge, EdgeType = _get_graph_types()
        edges: List[Any] = []

        def _first_statement_in_span(start: int, end: int) -> Optional[str]:
            """First exact statement start within [start, end] (line order)."""
            best_id: Optional[str] = None
            best_line = end + 1
            for node in nodes:
                if node.line and start <= node.line <= end and node.line < best_line:
                    best_line = node.line
                    best_id = node.node_id
            return best_id

        def _last_statement_in_span(start: int, end: int) -> Optional[str]:
            """Last statement starting within (start, end] — brace languages end
            constructs on a bare ``}`` line where no statement starts."""
            best_id: Optional[str] = None
            best_line = start
            for node in nodes:
                if node.line and start < node.line <= end and node.line > best_line:
                    best_line = node.line
                    best_id = node.node_id
            return best_id

        def _emit(src_line: int, dst_node: Any, edge_type: str) -> None:
            """Edge from the statement at src_line to the head of dst_node.

            The destination construct (else_clause, catch, case) may not itself
            be an extracted statement — resolve to the first statement inside
            its span instead.
            """
            dst_head = dst_node.start_point[0] + 1
            dst_end = dst_node.end_point[0] + 1
            src = self._statement_id_at_line(nodes, src_line)
            dst = _first_statement_in_span(dst_head, dst_end)
            if src and dst and src != dst:
                edges.append(GraphEdge(src=src, dst=dst, type=edge_type, weight=1.0))

        stack = [ast_root]
        while stack:
            n = stack.pop()
            t = getattr(n, "type", None)
            children = list(getattr(n, "children", []) or [])
            if t is not None:
                line = n.start_point[0] + 1
                if t in self._IF_NODES:
                    seen_alt_lines: Set[int] = set()
                    alts = [n.child_by_field_name("alternative")]
                    alts.extend(ch for ch in children if ch.type in self._ELSE_NODES)
                    for alt in alts:
                        if alt is None:
                            continue
                        alt_line = alt.start_point[0] + 1
                        if alt_line in seen_alt_lines:
                            continue
                        seen_alt_lines.add(alt_line)
                        _emit(line, alt, EdgeType.CFG_FALSE_BRANCH)
                elif t in self._LOOP_NODES:
                    last_line = n.end_point[0] + 1
                    if last_line > line:
                        src = _last_statement_in_span(line, last_line)
                        dst = self._statement_id_at_line(nodes, line)
                        if src and dst and src != dst:
                            edges.append(
                                GraphEdge(src=src, dst=dst, type=EdgeType.CFG_LOOP_BACK, weight=1.0)
                            )
                elif t in self._TRY_NODES:
                    for ch in children:
                        if ch.type in self._CATCH_NODES:
                            _emit(line, ch, EdgeType.CFG_CATCH)
                        elif ch.type in self._FINALLY_NODES:
                            _emit(line, ch, EdgeType.CFG_FINALLY)
                elif t in self._SWITCH_NODES:
                    body = n.child_by_field_name("body") or n
                    for ch in getattr(body, "children", []) or []:
                        if ch.type in self._CASE_NODES:
                            _emit(line, ch, EdgeType.CFG_CASE)
                        elif ch.type in self._DEFAULT_CASE_NODES:
                            _emit(line, ch, EdgeType.CFG_DEFAULT)
            stack.extend(children)

        return edges

    def _should_connect_cfg(self, current: GraphNode, next_node: GraphNode) -> bool:
        """Determine if two nodes should be connected in the CFG.

        Args:
            current: Current node
            next_node: Potential successor node

        Returns:
            True if nodes should be connected
        """
        # Basic heuristic: connect if next_node is immediately after
        if current.line and next_node.line:
            # Allow gaps for blank lines or comments
            return 0 < (next_node.line - current.line) <= 10
        return False

    def _determine_cfg_edge_type(self, current: Any, next_node: Any) -> str:
        """Determine the CFG edge type between two nodes.

        Args:
            current: Current node
            next_node: Successor node

        Returns:
            Edge type string
        """
        _, _, EdgeType = _get_graph_types()

        if current.statement_type == StatementType.CONDITION.value:
            # Check if this is the true or false branch
            # For now, assume true branch
            return EdgeType.CFG_TRUE_BRANCH
        elif current.statement_type == StatementType.LOOP.value:
            return EdgeType.CFG_LOOP_ENTRY
        elif current.statement_type in {
            StatementType.SWITCH.value,
            StatementType.CASE.value,
        }:
            return EdgeType.CFG_CASE
        elif current.statement_type == StatementType.RETURN.value:
            return EdgeType.CFG_RETURN
        else:
            return EdgeType.CFG_SUCCESSOR

    async def _build_cdg(
        self,
        nodes: List[Any],
        cfg_edges: List[Any],
    ) -> List[Any]:
        """Build Control Dependence Graph from CFG.

        Uses the post-dominance algorithm from Cytron et al.

        Args:
            nodes: CFG nodes
            cfg_edges: CFG edges

        Returns:
            List of GraphEdge instances representing CDG
        """
        _, GraphEdge, EdgeType = _get_graph_types()

        edges: List[Any] = []

        if not nodes or not cfg_edges:
            return edges

        # Build adjacency list for CFG
        successors: Dict[str, List[str]] = {}
        for node in nodes:
            successors[node.node_id] = []

        for edge in cfg_edges:
            if edge.src in successors:
                successors[edge.src].append(edge.dst)

        # Find condition nodes (if statements, loops)
        condition_nodes = [
            n
            for n in nodes
            if n.statement_type
            in {
                StatementType.CONDITION.value,
                StatementType.LOOP.value,
                StatementType.SWITCH.value,
            }
        ]

        # Create CDG edges from conditions to their dependent nodes
        for condition in condition_nodes:
            # Get all nodes dominated by this condition
            dominated = self._find_dominated_nodes(condition.node_id, successors, nodes)

            for dominated_id in dominated:
                if dominated_id != condition.node_id:
                    edge_type = self._determine_cdg_edge_type(condition)
                    edge = GraphEdge(
                        src=condition.node_id,
                        dst=dominated_id,
                        type=edge_type,
                        weight=1.0,
                        metadata={"file": condition.file} if condition.file else {},
                    )
                    edges.append(edge)

        return edges

    def _find_dominated_nodes(
        self,
        start_id: str,
        successors: Dict[str, List[str]],
        nodes: List[Any],
    ) -> List[str]:
        """Find all nodes dominated by start_id using simple reachability.

        Args:
            start_id: Starting node ID
            successors: Adjacency list of successors
            nodes: All nodes in the graph

        Returns:
            List of dominated node IDs
        """
        dominated: Set[str] = set()
        to_visit = [start_id]
        visited: Set[str] = set()

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)

            if current != start_id:
                dominated.add(current)

            # Add successors
            for succ in successors.get(current, []):
                if succ not in visited:
                    to_visit.append(succ)

        return list(dominated)

    def _determine_cdg_edge_type(self, condition: Any) -> str:
        """Determine the CDG edge type for a condition node.

        Args:
            condition: Condition node

        Returns:
            Edge type string
        """
        _, _, EdgeType = _get_graph_types()

        if condition.statement_type == StatementType.LOOP.value:
            return EdgeType.CDG_LOOP
        elif condition.statement_type == StatementType.CONDITION.value:
            return EdgeType.CDG
        else:
            return EdgeType.CDG

    async def _build_ddg(
        self,
        ast_root: Any,
        nodes: List[Any],
        file_path: Path,
        source_code: str,
    ) -> List[Any]:
        """Build Data Dependence Graph from AST.

        Tracks variable definitions and uses (def-use chains).

        Args:
            ast_root: Tree-sitter AST root
            nodes: Statement nodes
            file_path: Path to source file
            source_code: Source code

        Returns:
            List of GraphEdge instances representing DDG
        """
        _, GraphEdge, EdgeType = _get_graph_types()

        edges: List[Any] = []

        if not nodes or ast_root is None:
            return edges

        # Build symbol table (variable definitions)
        symbol_table: Dict[str, VariableInfo] = {}
        scope_stack: List[str] = [""]  # Stack of scope IDs

        # Walk AST and track variable definitions/uses
        def walk(node: Any, depth: int = 0) -> None:
            if not hasattr(node, "type"):
                return

            node_type = node.type

            # Track variable definitions
            if self._is_definition_node(node_type):
                var_name = self._extract_variable_name(node)
                if var_name:
                    current_scope = scope_stack[-1] if scope_stack else ""
                    node_id = self._find_node_at_line(nodes, node.start_point[0] + 1)

                    var_info = VariableInfo(
                        name=var_name,
                        defining_node=node_id,
                        scope=current_scope,
                    )
                    key = f"{current_scope}:{var_name}"
                    symbol_table[key] = var_info

            # Track variable uses
            var_uses = self._extract_variable_uses(node)
            for var_name in var_uses:
                current_scope = scope_stack[-1] if scope_stack else ""
                key = f"{current_scope}:{var_name}"
                if key in symbol_table:
                    var_info = symbol_table[key]
                    use_node_id = self._find_node_at_line(nodes, node.start_point[0] + 1)
                    if use_node_id and use_node_id != var_info.defining_node:
                        var_info.use_sites.append(use_node_id)

            # Track scope changes
            if self._is_scope_boundary(node_type):
                new_scope = self._generate_statement_id(
                    str(file_path),
                    node.start_point[0] + 1,
                    node.end_point[0] + 1,
                    node_type,
                )
                scope_stack.append(new_scope)

            # Recurse into children
            for child in node.children:
                walk(child, depth + 1)

            # Pop scope if we pushed one
            if self._is_scope_boundary(node_type):
                if len(scope_stack) > 1:
                    scope_stack.pop()

        try:
            self._walk_ast(ast_root, walk)
        except Exception as e:
            logger.warning(f"Error building DDG: {e}")

        # Create DDG edges from def-use chains
        for var_info in symbol_table.values():
            if var_info.defining_node:
                for use_site in var_info.use_sites:
                    edge = GraphEdge(
                        src=var_info.defining_node,
                        dst=use_site,
                        type=EdgeType.DDG_DEF_USE,
                        weight=1.0,
                        metadata={"variable": var_info.name, "file": str(file_path)},
                    )
                    edges.append(edge)

        return edges

    def _is_definition_node(self, node_type: str) -> bool:
        """Check if a node type represents a variable definition.

        Args:
            node_type: Tree-sitter node type

        Returns:
            True if node is a definition
        """
        # Python definitions
        if self.language == "python":
            return node_type in {
                "assignment",
                "named_expression",
                "for_statement",  # loop variable
                "with_clause",  # context manager variable
                "pattern",  # match statement patterns
            }

        # JavaScript/TypeScript definitions
        elif self.language in {"javascript", "typescript"}:
            return node_type in {
                "variable_declaration",
                "assignment_expression",
            }

        # Go definitions
        elif self.language == "go":
            return node_type in {
                "assignment_statement",
                "short_var_declaration",
                "for_statement",  # loop variable
                "range_clause",  # range variable
            }

        return False

    def _is_scope_boundary(self, node_type: str) -> bool:
        """Check if a node type represents a scope boundary.

        Args:
            node_type: Tree-sitter node type

        Returns:
            True if node is a scope boundary
        """
        scope_types = {
            "function_definition",
            "class_definition",
            "if_statement",
            "for_statement",
            "while_statement",
            "try_statement",
            "with_statement",
        }
        return node_type in scope_types

    def _extract_variable_name(self, node: Any) -> Optional[str]:
        """Extract variable name from a definition node.

        Args:
            node: Tree-sitter node

        Returns:
            Variable name or None
        """
        # This is a simplified implementation
        # Full implementation would traverse the node's children
        # to find the identifier
        try:
            if hasattr(node, "children"):
                for child in node.children:
                    if hasattr(child, "type") and child.type == "identifier":
                        if hasattr(child, "text"):
                            return (
                                child.text.decode() if isinstance(child.text, bytes) else child.text
                            )
        except Exception:
            pass
        return None

    def _extract_variable_uses(self, node: Any) -> List[str]:
        """Extract variable uses from a node.

        Args:
            node: Tree-sitter node

        Returns:
            List of variable names used in this node
        """
        # Simplified implementation
        uses: List[str] = []

        def extract(n: Any) -> None:
            if hasattr(n, "type") and n.type == "identifier":
                if hasattr(n, "text"):
                    name = n.text.decode() if isinstance(n.text, bytes) else n.text
                    # Filter out keywords
                    if name and not self._is_keyword(name):
                        uses.append(name)
            for child in getattr(n, "children", []):
                extract(child)

        try:
            extract(node)
        except Exception:
            pass

        return uses

    def _is_keyword(self, name: str) -> bool:
        """Check if a name is a language keyword.

        Args:
            name: Identifier name

        Returns:
            True if name is a keyword
        """
        # Combined keywords set (unique across all languages)
        keywords = {
            # Python
            "False",
            "None",
            "True",
            "and",
            "as",
            "assert",
            "async",
            "await",
            "break",
            "class",
            "continue",
            "def",
            "del",
            "elif",
            "else",
            "except",
            "finally",
            "for",
            "from",
            "global",
            "if",
            "import",
            "in",
            "is",
            "lambda",
            "nonlocal",
            "not",
            "or",
            "pass",
            "raise",
            "return",
            "try",
            "while",
            "with",
            "yield",
            # JavaScript/TypeScript
            "case",
            "catch",
            "const",
            "debugger",
            "default",
            "delete",
            "do",
            "enum",
            "export",
            "extends",
            "false",
            "function",
            "implements",
            "instanceof",
            "interface",
            "let",
            "new",
            "null",
            "of",
            "package",
            "private",
            "protected",
            "public",
            "super",
            "switch",
            "static",
            "this",
            "throw",
            "true",
            "typeof",
            "var",
            "void",
            # Go
            "chan",
            "defer",
            "fallthrough",
            "func",
            "go",
            "goto",
            "map",
            "range",
            "select",
            "struct",
            "type",
        }
        return name in keywords

    def _decay_score(self, base_score: float, distance: int, max_distance: int) -> float:
        """Apply distance decay to a relevance score.

        Args:
            base_score: Original relevance score (0-1)
            distance: Hop distance from source
            max_distance: Maximum distance for decay calculation

        Returns:
            Decayed score
        """
        if distance == 0:
            return base_score
        decay_factor = 1.0 - (distance / max_distance)
        return base_score * max(decay_factor, 0.1)

    def _should_connect_cfg(self, node1: Any, node2: Any) -> bool:
        """Determine if two nodes should have a CFG edge between them.

        Args:
            node1: First node
            node2: Second node

        Returns:
            True if nodes should be connected
        """
        # Connect if nodes are close in the same file
        if not node1.file or not node2.file:
            return False
        if node1.file != node2.file:
            return False
        if not node1.line or not node2.line:
            return True

        # Connect if within 30 lines
        return abs(node1.line - node2.line) <= 30

    def _find_node_at_line(self, nodes: List[Any], line: int) -> Optional[str]:
        """Find a node ID at a specific line number.

        Args:
            nodes: List of graph nodes
            line: Line number

        Returns:
            Node ID or None
        """
        for node in nodes:
            if node.line and node.line <= line <= (node.end_line or line):
                return node.node_id
        return None

    def _walk_ast(self, node: Any, visitor: Any, depth: int = 0) -> None:
        """Walk the AST and call visitor on each node.

        Args:
            node: Tree-sitter node
            visitor: Visitor function
            depth: Current depth
        """
        visitor(node, depth)
        for child in getattr(node, "children", []):
            self._walk_ast(child, visitor, depth + 1)

    def supports_language(self, language: str) -> bool:
        """Check if this builder supports the given language.

        Args:
            language: Language identifier (e.g., "python", "javascript")

        Returns:
            True if this builder can handle the language
        """
        lang = language.lower()
        return lang in SUPPORTED_CCG_LANGUAGES


__all__ = ["CodeContextGraphBuilder", "StatementType", "SUPPORTED_CCG_LANGUAGES"]
