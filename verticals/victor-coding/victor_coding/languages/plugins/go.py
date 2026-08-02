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

"""Go language plugin."""

import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from victor_coding.languages.base import (
    BaseLanguagePlugin,
    BuildSystem,
    CallEdge,
    CommentStyle,
    ConfigurableASTTraverser,
    DocCommentPattern,
    EdgeDetectionResult,
    Formatter,
    LanguageCapabilities,
    LanguageConfig,
    Linter,
    QueryPattern,
    TestRunner,
    TraversalConfig,
    TreeSitterQueries,
)

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)


class GoPlugin(BaseLanguagePlugin):
    """Go language plugin.

    Supports:
    - Testing: go test
    - Formatting: gofmt, goimports
    - Linting: golangci-lint, staticcheck
    - Building: go build
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="go",
            display_name="Go",
            aliases=["golang"],
            extensions=[".go"],
            filenames=["go.mod", "go.sum"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', "`"],
            indent_size=4,
            use_tabs=True,  # Go convention
            package_managers=["go mod"],
            build_systems=["go build"],
            test_frameworks=["go test"],
            language_server="gopls",
            language_server_name="gopls",
            tree_sitter_language="go",
            doc_comment_pattern=DocCommentPattern(
                line_prefixes=["//"],
            ),
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=True,
            supports_rename=True,
            supports_extract_function=True,
            supports_inline=True,
            supports_organize_imports=True,
            supports_test_discovery=True,
            supports_test_execution=True,
            supports_coverage=True,
            supports_debugging=True,
            supports_breakpoints=True,
            supports_step_debugging=True,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
            supports_control_flow_graph=True,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        """Create tree-sitter queries for Go symbol/call extraction."""
        return TreeSitterQueries(
            symbols=[
                QueryPattern("function", "(function_declaration name: (identifier) @name)"),
                QueryPattern("function", "(method_declaration name: (field_identifier) @name)"),
                QueryPattern(
                    "class",
                    "(type_declaration (type_spec name: (type_identifier) @name))",
                ),
            ],
            calls="""
                (call_expression function: (identifier) @callee)
                (call_expression function: (selector_expression field: (field_identifier) @callee))
                (type_conversion_expression type: (type_identifier) @callee)
            """,
            references="""
                (call_expression function: (identifier) @name)
                (call_expression function: (selector_expression field: (field_identifier) @name))
                (selector_expression field: (field_identifier) @name)
                (identifier) @name
            """,
            composition="""
                (type_declaration
                    (type_spec
                        name: (type_identifier) @owner
                        type: (struct_type
                            (field_declaration_list
                                (field_declaration
                                    type: (type_identifier) @type)))))
            """,
            enclosing_scopes=[
                ("function_declaration", "name"),
                ("method_declaration", "name"),
            ],
        )

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Get go test runner."""
        go_mod = project_root / "go.mod"

        if not go_mod.exists():
            return None

        return TestRunner(
            name="go test",
            command=["go", "test"],
            file_pattern="*_test.go",
            discover_args=["-list", "."],
            run_args=["-v", "./..."],
            coverage_args=["-cover", "-coverprofile=coverage.out"],
            parallel_args=["-parallel", "auto"],
            output_format="json",
            json_args=["-json"],
        )

    def get_formatter(self, project_root: Path) -> Optional[Formatter]:
        """Get gofmt formatter."""
        return Formatter(
            name="gofmt",
            command=["gofmt", "-w", "."],
            check_args=["-d"],
        )

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Get golangci-lint linter."""
        return Linter(
            name="golangci-lint",
            command=["golangci-lint", "run"],
            fix_args=["--fix"],
            config_file=".golangci.yml",
            output_format="json",
        )

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Get go build system."""
        go_mod = project_root / "go.mod"

        if not go_mod.exists():
            return None

        return BuildSystem(
            name="go",
            build_command=["go", "build", "./..."],
            run_command=["go", "run", "."],
            clean_command=["go", "clean"],
            install_command=["go", "install", "."],
            manifest_file="go.mod",
        )

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in Go source code.

        Finds function calls, method calls, and goroutine calls.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []
        call_nodes = self._find_call_nodes(tree.root_node)

        for call_node, caller_name, caller_line in call_nodes:
            callee_name = self._extract_callee_name(call_node)
            if callee_name and caller_name:
                calls.append(
                    CallEdge(
                        caller_name=caller_name,
                        callee_name=callee_name,
                        caller_line=caller_line,
                    )
                )

        logger.debug(f"Detected {len(calls)} CALLS edges in {file_path.name}")

        return EdgeDetectionResult(
            calls=calls,
            metadata={
                "language": "go",
                "file": str(file_path),
            },
        )

    def _find_call_nodes(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Uses the shared ConfigurableASTTraverser to eliminate code duplication.
        """
        config = TraversalConfig(
            function_types=["function_declaration", "method_declaration"],
            call_types=["call_expression"],
            name_field="identifier",
        )
        traverser = ConfigurableASTTraverser(config, self._get_node_text)
        return traverser.find_call_nodes(root)

    def _extract_callee_name(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called function from a call node.

        Handles:
        - Simple calls: foo()
        - Selector calls: obj.method()
        - Package calls: pkg.Func()
        """
        # Go call_expression structure: (function: identifier or selector_expression)
        for child in call_node.children:
            if child.type == "selector_expression":
                # obj.method() -> extract "method"
                return self._extract_selector_name(child)
            elif child.type == "identifier":
                # foo() -> extract "foo"
                return self._get_node_text(child)
            elif child.type == "parenthesized_expression":
                # Handle parenthesized expressions
                return self._extract_callee_name(child)

        return None

    def _extract_selector_name(self, selector_node: "Node") -> Optional[str]:
        """Extract selector name from a selector_expression node.

        For obj.method, extracts "method".
        For pkg.Type.method, extracts "method" (the final selector).
        """
        # selector_expression: operand (identifier or selector_expression) . selector (field_identifier)
        for child in reversed(selector_node.children):
            if child.type == "field_identifier":
                return self._get_node_text(child)
            elif child.type == "selector_expression":
                result = self._extract_selector_name(child)
                if result:
                    return result

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node."""
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text
