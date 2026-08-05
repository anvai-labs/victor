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

"""Base types for language plugins.

Defines the interfaces and data structures used by language plugins
to provide language-specific functionality.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tree-sitter Query Types
# ---------------------------------------------------------------------------


@dataclass
class QueryPattern:
    """Single tree-sitter query pattern for symbol extraction.

    Attributes:
        symbol_type: The type of symbol this query extracts (e.g., "class", "function")
        query: The tree-sitter query string with @name capture
    """

    symbol_type: str
    query: str


@dataclass
class TreeSitterQueries:
    """Collection of tree-sitter queries for a language.

    All queries should use the new tree-sitter API format and capture
    nodes using @name syntax. Queries are optional - if not provided,
    that feature will not be available for the language.

    Example Python queries:
        symbols: [
            QueryPattern("class", "(class_definition name: (identifier) @name)"),
            QueryPattern("function", "(function_definition name: (identifier) @name)"),
        ]
        calls: "(call function: (identifier) @callee)"

    Attributes:
        symbols: List of query patterns for symbol extraction (classes, functions, etc.)
        calls: Query for call expressions (captures @callee)
        references: Query for identifier references
        inheritance: Query for class inheritance (captures @child and @base)
        implements: Query for interface implementation (captures @child and @interface)
        composition: Query for has-a relationships (captures @owner and @type)
        enclosing_scopes: List of (node_type, name_field) for caller resolution
    """

    # Symbol extraction queries
    symbols: List[QueryPattern] = field(default_factory=list)

    # Relationship extraction queries
    calls: Optional[str] = None
    references: Optional[str] = None
    inheritance: Optional[str] = None
    implements: Optional[str] = None
    composition: Optional[str] = None

    # For resolving enclosing scope (e.g., which function a call is inside)
    # List of (node_type, name_field) tuples
    enclosing_scopes: List[Tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph Edge Detection Types (for Graph RAG)
# ---------------------------------------------------------------------------


@dataclass
class CallEdge:
    """A detected function call edge for graph construction.

    Attributes:
        caller_name: Name of the calling function/symbol
        callee_name: Name of the called function/symbol
        caller_line: Line number where call occurs (for debugging)
        receiver_type: For method calls `obj.method()`, the inferred static
            type of the receiver (e.g., `"Foo"`). None for plain function
            calls, untracked variables, or when type inference is best-effort
            and could not determine the type. Used downstream to bind the
            call to the correct `impl T` block instead of fanning out to
            every method with the same leaf name.
        is_method_call: True for `obj.method()` dot dispatch (regardless of
            whether receiver_type could be inferred). False for plain
            `func()` calls and path calls like `Foo::bar()`. The downstream
            resolver uses this to decide fallback policy: method calls with
            no inferable receiver type are dropped (name-only fallback would
            bind to unrelated user-defined methods with the same leaf name,
            almost always wrong); plain function calls keep the name-only
            fallback with a fanout cap.
    """

    caller_name: str
    callee_name: str
    caller_line: Optional[int] = None
    receiver_type: Optional[str] = None
    is_method_call: bool = False


@dataclass
class EdgeDetectionResult:
    """Result from edge detection for a file.

    Attributes:
        calls: List of CALLS edges detected
        metadata: Additional metadata (language-specific)
    """

    calls: List[CallEdge]
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CodeChunk:
    """A semantic code chunk for embedding.

    Represents a meaningful unit of code extracted using AST-aware
    chunking. This preserves semantic boundaries like function/class
    definitions rather than arbitrary text splits.

    Attributes:
        text: The actual code text for this chunk
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)
        chunk_type: Type of chunk ("function", "class", "module_header", etc.)
        symbol_name: Name of the symbol if this chunk represents one
        parent_symbol: Name of parent symbol (e.g., class name for a method)
        file_path: Path to the source file
    """

    text: str
    start_line: int
    end_line: int
    chunk_type: str
    symbol_name: Optional[str] = None
    parent_symbol: Optional[str] = None
    file_path: Optional[str] = None


class ChunkStrategy(Protocol):
    """Protocol for AST-aware code chunking strategies.

    Implementations split code into semantic chunks using the parsed
    AST tree rather than arbitrary text boundaries. This produces
    better embeddings for code search.
    """

    def chunk_file(
        self, tree: "Tree", content: bytes, file_path: Optional[str] = None
    ) -> List[CodeChunk]:
        """Split a file into semantic chunks using AST.

        Args:
            tree: Parsed tree-sitter tree
            content: Raw file content as bytes
            file_path: Optional file path for metadata

        Returns:
            List of semantic code chunks
        """
        ...


class CommentStyle(Enum):
    """Comment styles for different languages."""

    C_STYLE = "c_style"  # /* */ and //
    HASH = "hash"  # #
    DOUBLE_DASH = "double_dash"  # --
    SEMICOLON = "semicolon"  # ;
    HTML = "html"  # <!-- -->
    NONE = "none"


@dataclass
class DocCommentPattern:
    """How documentation comments look for a language.

    Attributes:
        line_prefixes: Prefixes for doc comment lines (e.g., ["///", "//!"] for Rust)
        block_start: Start marker for block doc comments (e.g., "/**")
        block_end: End marker for block doc comments (e.g., "*/")
        location: Where doc comments appear relative to the symbol:
            "before" (Rust/Go/Java/JS) or "inside" (Python)
    """

    line_prefixes: List[str] = field(default_factory=list)
    block_start: Optional[str] = None
    block_end: Optional[str] = None
    location: str = "before"


@dataclass
class LanguageConfig:
    """Configuration for a programming language.

    Contains all the metadata needed to work with a language,
    from syntax details to tooling commands.
    """

    # Identity
    name: str  # Canonical name (e.g., "python")
    display_name: str  # Human-readable name (e.g., "Python")
    aliases: List[str] = field(default_factory=list)  # Alternative names

    # File identification
    extensions: List[str] = field(default_factory=list)  # .py, .pyw
    filenames: List[str] = field(default_factory=list)  # Makefile, Dockerfile
    shebangs: List[str] = field(default_factory=list)  # #!/usr/bin/python

    # Syntax
    comment_style: CommentStyle = CommentStyle.HASH
    line_comment: Optional[str] = "#"
    block_comment_start: Optional[str] = None
    block_comment_end: Optional[str] = None
    string_delimiters: List[str] = field(default_factory=lambda: ['"', "'"])

    # Indentation
    indent_size: int = 4
    use_tabs: bool = False

    # Tooling
    package_managers: List[str] = field(default_factory=list)  # pip, npm
    build_systems: List[str] = field(default_factory=list)  # make, cargo
    test_frameworks: List[str] = field(default_factory=list)  # pytest, jest

    # LSP
    language_server: Optional[str] = None  # Command to start LSP
    language_server_name: Optional[str] = None  # Name for identification

    # Tree-sitter
    tree_sitter_language: Optional[str] = None  # tree-sitter grammar name

    # Documentation comment pattern
    doc_comment_pattern: Optional["DocCommentPattern"] = None


@dataclass
class TestRunner:
    """Configuration for a test runner."""

    name: str
    command: List[str]  # Base command (e.g., ["pytest"])
    file_pattern: str = "test_*.py"  # Pattern for test files
    discover_args: List[str] = field(default_factory=list)  # Args for discovery
    run_args: List[str] = field(default_factory=list)  # Args for running
    coverage_args: List[str] = field(default_factory=list)  # Args for coverage
    parallel_args: List[str] = field(default_factory=list)  # Args for parallel run

    # Output parsing
    output_format: str = "text"  # text, json, junit
    json_args: List[str] = field(default_factory=list)  # Args for JSON output


@dataclass
class BuildSystem:
    """Configuration for a build system."""

    name: str
    build_command: List[str]  # e.g., ["cargo", "build"]
    run_command: List[str]  # e.g., ["cargo", "run"]
    clean_command: List[str] = field(default_factory=list)
    install_command: List[str] = field(default_factory=list)

    # Build modes
    debug_args: List[str] = field(default_factory=list)
    release_args: List[str] = field(default_factory=list)

    # Manifest file
    manifest_file: Optional[str] = None  # Cargo.toml, package.json


@dataclass
class Formatter:
    """Configuration for a code formatter."""

    name: str
    command: List[str]  # e.g., ["black", "."]
    check_args: List[str] = field(default_factory=list)  # Args for check mode
    config_file: Optional[str] = None  # pyproject.toml, .prettierrc


@dataclass
class Linter:
    """Configuration for a linter."""

    name: str
    command: List[str]  # e.g., ["ruff", "check"]
    fix_args: List[str] = field(default_factory=list)  # Args for auto-fix
    config_file: Optional[str] = None
    output_format: str = "text"  # text, json, sarif


@dataclass
class LanguageCapabilities:
    """Capabilities of a language plugin."""

    # Analysis
    supports_syntax_analysis: bool = False
    supports_semantic_analysis: bool = False
    supports_type_checking: bool = False

    # Refactoring
    supports_rename: bool = False
    supports_extract_function: bool = False
    supports_inline: bool = False
    supports_organize_imports: bool = False

    # Testing
    supports_test_discovery: bool = False
    supports_test_execution: bool = False
    supports_coverage: bool = False

    # Debugging
    supports_debugging: bool = False
    supports_breakpoints: bool = False
    supports_step_debugging: bool = False

    # Other
    supports_formatting: bool = False
    supports_linting: bool = False
    supports_completion: bool = False

    # Graph analysis
    # True iff the language has real control flow (if/for/while/try/etc.)
    # and benefits from a Code Context Graph (CFG/CDG/DDG). Markup, build,
    # schema, and pure-data formats (markdown, make, cmake, graphql, hcl,
    # html, css, json, yaml, toml, etc.) keep this False so CCG builders
    # know to skip them even when symbol queries are defined.
    supports_control_flow_graph: bool = False


@runtime_checkable
class LanguagePlugin(Protocol):
    """Protocol for language plugins.

    Each language implements this protocol to provide
    language-specific functionality to Victor.
    """

    @property
    def config(self) -> LanguageConfig:
        """Get language configuration."""
        ...

    @property
    def capabilities(self) -> LanguageCapabilities:
        """Get language capabilities."""
        ...

    @property
    def tree_sitter_queries(self) -> TreeSitterQueries:
        """Get tree-sitter queries for symbol/call extraction."""
        ...

    def detect_from_file(self, path: Path) -> bool:
        """Check if this language handles the given file.

        Args:
            path: File path to check

        Returns:
            True if this language handles the file
        """
        ...

    def detect_from_content(self, content: str, filename: Optional[str] = None) -> float:
        """Estimate confidence that content is in this language.

        Args:
            content: File content
            filename: Optional filename hint

        Returns:
            Confidence score 0.0 to 1.0
        """
        ...

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Get appropriate test runner for a project.

        Args:
            project_root: Project root directory

        Returns:
            TestRunner config or None if not available
        """
        ...

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Get appropriate build system for a project.

        Args:
            project_root: Project root directory

        Returns:
            BuildSystem config or None if not available
        """
        ...

    def get_formatter(self, project_root: Path) -> Optional[Formatter]:
        """Get appropriate formatter for a project.

        Args:
            project_root: Project root directory

        Returns:
            Formatter config or None if not available
        """
        ...

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Get appropriate linter for a project.

        Args:
            project_root: Project root directory

        Returns:
            Linter config or None if not available
        """
        ...

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in parsed source code for Graph RAG.

        Analyzes the parsed tree-sitter tree to find function/method calls
        and their relationships. This is used to build call graphs for
        code intelligence and retrieval.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        ...


class BaseLanguagePlugin(ABC):
    """Base class for language plugins with common functionality."""

    def __init__(self):
        """Initialize plugin."""
        self._config: Optional[LanguageConfig] = None
        self._capabilities: Optional[LanguageCapabilities] = None
        self._tree_sitter_queries: Optional[TreeSitterQueries] = None

    @property
    def config(self) -> LanguageConfig:
        """Get language configuration."""
        if self._config is None:
            self._config = self._create_config()
        return self._config

    @property
    def capabilities(self) -> LanguageCapabilities:
        """Get language capabilities."""
        if self._capabilities is None:
            self._capabilities = self._create_capabilities()
        return self._capabilities

    @property
    def tree_sitter_queries(self) -> TreeSitterQueries:
        """Get tree-sitter queries for this language.

        Returns queries for symbol extraction, call analysis,
        inheritance detection, etc. using tree-sitter.
        """
        if self._tree_sitter_queries is None:
            self._tree_sitter_queries = self._create_tree_sitter_queries()
        return self._tree_sitter_queries

    @abstractmethod
    def _create_config(self) -> LanguageConfig:
        """Create language configuration."""
        ...

    @abstractmethod
    def _create_capabilities(self) -> LanguageCapabilities:
        """Create capabilities description."""
        ...

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        """Create tree-sitter queries for this language.

        Override in subclasses to provide language-specific queries.
        Default returns empty queries (no tree-sitter support).
        """
        return TreeSitterQueries()

    def detect_from_file(self, path: Path) -> bool:
        """Check if this language handles the file."""
        # Check extension
        if path.suffix.lower() in [e.lower() for e in self.config.extensions]:
            return True

        # Check filename
        if path.name in self.config.filenames:
            return True

        # Check shebang
        if self.config.shebangs and path.exists():
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    first_line = f.readline().strip()
                    for shebang in self.config.shebangs:
                        if first_line.startswith("#!") and shebang in first_line:
                            return True
            except Exception:
                pass

        return False

    def detect_from_content(self, content: str, filename: Optional[str] = None) -> float:
        """Estimate confidence from content."""
        score = 0.0

        # Check filename hint
        if filename:
            path = Path(filename)
            if path.suffix.lower() in [e.lower() for e in self.config.extensions]:
                score += 0.5
            if path.name in self.config.filenames:
                score += 0.5

        # Check shebang in content
        if content.startswith("#!"):
            first_line = content.split("\n")[0]
            for shebang in self.config.shebangs:
                if shebang in first_line:
                    score += 0.3
                    break

        return min(score, 1.0)

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Default: no test runner."""
        return None

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Default: no build system."""
        return None

    def get_formatter(self, project_root: Path) -> Optional[Formatter]:
        """Default: no formatter."""
        return None

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Default: no linter."""
        return None

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Default implementation: no edge detection.

        Override in subclasses to provide language-specific call graph detection.
        """
        return EdgeDetectionResult(calls=[], metadata={"language": self.config.name})


@dataclass
class TraversalConfig:
    """Configuration for AST traversal to eliminate duplicate traverse() implementations.

    Each language plugin can define its node type mappings instead of duplicating
    the traverse logic. This reduces code duplication from ~100 lines per plugin
    to a simple configuration dict.

    Attributes:
        function_types: Node types that represent functions/methods (e.g., ["function_definition"])
        class_types: Node types that represent classes/structs (e.g., ["class_declaration"])
        call_types: Node types that represent calls/invocations (e.g., ["call", "method_invocation"])
        name_field: Field name to extract symbol name (e.g., "identifier" or "name")
        scope_body_types: For classes/containers, the child type that contains members (e.g., ["class_body"])
        skip_recursion_in_types: Node types where we should stop recursing (e.g., ["string_literal"])
    """

    function_types: List[str] = field(default_factory=list)
    class_types: List[str] = field(default_factory=list)
    call_types: List[str] = field(default_factory=list)
    name_field: str = "identifier"
    scope_body_types: List[str] = field(default_factory=list)
    skip_recursion_in_types: List[str] = field(
        default_factory=lambda: ["string_literal", "comment", "bytes"]
    )


class ConfigurableASTTraverser:
    """Configurable AST traverser to eliminate duplicate traverse() implementations.

    Each language plugin (Python, Java, C++, etc.) previously had its own ~50-line
    traverse() function that was nearly identical except for node type names.

    Example usage in PythonPlugin:
        config = TraversalConfig(
            function_types=["function_definition"],
            class_types=["class_definition"],
            call_types=["call"],
            name_field="identifier",
        )
        traverser = ConfigurableASTTraverser(config)
        results = traverser.find_call_nodes(tree.root_node)

    Example usage in JavaPlugin:
        config = TraversalConfig(
            function_types=["method_declaration", "constructor_declaration"],
            class_types=["class_declaration", "interface_declaration"],
            call_types=["method_invocation", "object_creation_expression"],
            name_field="identifier",
            scope_body_types=["class_body"],
        )
        traverser = ConfigurableASTTraverser(config)
        results = traverser.find_call_nodes(tree.root_node)
    """

    def __init__(self, config: TraversalConfig, get_node_text_fn: callable) -> None:
        """Initialize traverser with configuration.

        Args:
            config: Traversal configuration for this language
            get_node_text_fn: Function to extract text from a node (typically self._get_node_text)
        """
        self.config = config
        self._get_node_text = get_node_text_fn

    def find_call_nodes(self, root: "Node") -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        This replaces the duplicated traverse() functions in each language plugin.

        Args:
            root: Root tree-sitter node to traverse

        Returns:
            List of (call_node, enclosing_function_name, line_number) tuples
        """
        results: List[tuple["Node", str, Optional[int]]] = []

        def traverse(node: "Node", enclosing_function: Optional[str] = None) -> None:
            # Skip recursion into leaf nodes
            if node.type in self.config.skip_recursion_in_types:
                return

            node_type = node.type

            # Check if this is a function/method definition
            if node_type in self.config.function_types:
                func_name = self._extract_name(node)
                # Recurse with new enclosing context
                for child in node.children:
                    traverse(child, func_name)
                return

            # Check if this is a class/interface definition
            if node_type in self.config.class_types:
                class_name = self._extract_name(node)
                # For class bodies, only recurse into the body
                if self.config.scope_body_types:
                    for child in node.children:
                        if child.type in self.config.scope_body_types:
                            for grandchild in child.children:
                                traverse(grandchild, class_name)
                else:
                    for child in node.children:
                        traverse(child, class_name)
                return

            # Check if this is a call/invocation node
            if node_type in self.config.call_types:
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Recurse into children
            for child in node.children:
                traverse(child, enclosing_function)

        traverse(root)
        return results

    def _extract_name(self, node: "Node") -> Optional[str]:
        """Extract the name from a definition node.

        Searches children for a node of type matching self.config.name_field.

        Args:
            node: The definition node (function, class, etc.)

        Returns:
            Extracted name or None
        """
        for child in node.children:
            if child.type == self.config.name_field:
                return self._get_node_text(child)
        return None
