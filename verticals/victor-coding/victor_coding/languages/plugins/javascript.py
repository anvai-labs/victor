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

"""JavaScript language plugin."""

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


class JavaScriptPlugin(BaseLanguagePlugin):
    """JavaScript language plugin.

    Supports:
    - Testing: jest, mocha, vitest
    - Formatting: prettier
    - Linting: eslint
    - Bundling: webpack, vite, esbuild
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="javascript",
            display_name="JavaScript",
            aliases=["js", "jsx", "mjs", "cjs"],
            extensions=[".js", ".jsx", ".mjs", ".cjs"],
            filenames=["package.json", ".eslintrc", ".prettierrc"],
            shebangs=["node"],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', "'", "`"],
            indent_size=2,
            use_tabs=False,
            package_managers=["npm", "yarn", "pnpm", "bun"],
            build_systems=["webpack", "vite", "esbuild", "rollup"],
            test_frameworks=["jest", "mocha", "vitest", "ava"],
            language_server="typescript-language-server",
            language_server_name="TypeScript Language Server",
            tree_sitter_language="javascript",
            doc_comment_pattern=DocCommentPattern(
                line_prefixes=["///"],
                block_start="/**",
                block_end="*/",
            ),
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=False,  # No native types
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
        """Create tree-sitter queries for JavaScript symbol/call extraction."""
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_declaration name: (identifier) @name)"),
                QueryPattern("function", "(function_declaration name: (identifier) @name)"),
                QueryPattern("function", "(method_definition name: (property_identifier) @name)"),
                QueryPattern(
                    "function",
                    "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function)))",
                ),
                QueryPattern(
                    "function",
                    "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression)))",
                ),
                QueryPattern(
                    "function",
                    "(assignment_expression left: (identifier) @name right: (arrow_function))",
                ),
            ],
            calls="""
                (call_expression function: (identifier) @callee)
                (call_expression function: (member_expression property: (property_identifier) @callee))
                (new_expression constructor: (identifier) @callee)
            """,
            references="""
                (call_expression function: (identifier) @name)
                (call_expression function: (member_expression property: (property_identifier) @name))
                (member_expression property: (property_identifier) @name)
                (new_expression constructor: (identifier) @name)
                (identifier) @name
            """,
            inheritance="""
                (class_declaration
                    name: (identifier) @child
                    (class_heritage (identifier) @base))
            """,
            composition="""
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
            enclosing_scopes=[
                ("function_declaration", "name"),
                ("method_definition", "name"),
                ("class_declaration", "name"),
            ],
        )

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Get jest, mocha, or vitest runner."""
        package_json = project_root / "package.json"

        if package_json.exists():
            content = package_json.read_text()

            # Check for vitest
            if '"vitest"' in content:
                return TestRunner(
                    name="vitest",
                    command=["npx", "vitest"],
                    file_pattern="*.test.js",
                    run_args=["run"],
                    coverage_args=["--coverage"],
                    parallel_args=["--pool", "threads"],
                    output_format="json",
                    json_args=["--reporter=json"],
                )

            # Check for jest
            if '"jest"' in content:
                return TestRunner(
                    name="jest",
                    command=["npx", "jest"],
                    file_pattern="*.test.js",
                    coverage_args=["--coverage"],
                    parallel_args=["--maxWorkers=auto"],
                    output_format="json",
                    json_args=["--json"],
                )

            # Check for mocha
            if '"mocha"' in content:
                return TestRunner(
                    name="mocha",
                    command=["npx", "mocha"],
                    file_pattern="*.test.js",
                    run_args=["--recursive"],
                )

        # Default to jest
        return TestRunner(
            name="jest",
            command=["npx", "jest"],
            file_pattern="*.test.js",
        )

    def get_formatter(self, project_root: Path) -> Optional[Formatter]:
        """Get prettier formatter."""
        return Formatter(
            name="prettier",
            command=["npx", "prettier", "--write", "."],
            check_args=["--check"],
            config_file=".prettierrc",
        )

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Get eslint linter."""
        return Linter(
            name="eslint",
            command=["npx", "eslint", "."],
            fix_args=["--fix"],
            config_file=".eslintrc",
            output_format="json",
        )

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Get npm/yarn/pnpm build system."""
        package_json = project_root / "package.json"

        if not package_json.exists():
            return None

        # Detect package manager
        if (project_root / "pnpm-lock.yaml").exists():
            pm = "pnpm"
        elif (project_root / "yarn.lock").exists():
            pm = "yarn"
        else:
            pm = "npm"

        return BuildSystem(
            name=pm,
            build_command=[pm, "run", "build"],
            run_command=[pm, "run", "start"],
            install_command=[pm, "install"],
            manifest_file="package.json",
        )

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in JavaScript/TypeScript source code.

        Finds function calls, method calls, and property access chains.

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

        return EdgeDetectionResult(
            calls=calls,
            metadata={
                "language": "javascript/typescript",
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
            function_types=[
                "function_declaration",
                "function_expression",
                "method_definition",
                "arrow_function",
            ],
            class_types=["class_declaration"],
            call_types=["call_expression", "new_expression"],
            name_field="identifier",
            scope_body_types=["class_body"],
        )
        traverser = ConfigurableASTTraverser(config, self._get_node_text)
        return traverser.find_call_nodes(root)

    def _extract_function_name(self, node: "Node") -> Optional[str]:
        """Extract function name from a function node."""
        if node.type == "function_declaration":
            for child in node.children:
                if child.type == "identifier":
                    return self._get_node_text(child)

        if node.type == "method_definition":
            for child in node.children:
                if child.type in ("property_identifier", "identifier"):
                    return self._get_node_text(child)

        if node.type == "function_expression":
            found_function = False
            for child in node.children:
                if child.type == "identifier":
                    if found_function:
                        return self._get_node_text(child)
                elif child.type == "function":
                    found_function = True

        return None

    def _extract_callee_name(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called function from a call node."""
        for child in call_node.children:
            if child.type == "member_expression":
                return self._extract_member_name(child)
            elif child.type == "identifier":
                return self._get_node_text(child)
            elif child.type == "subscript_expression":
                return self._extract_subscript_name(child)

        return None

    def _extract_member_name(self, member_node: "Node") -> Optional[str]:
        """Extract member name from a member_expression node."""
        for child in reversed(member_node.children):
            if child.type == "property_identifier":
                return self._get_node_text(child)
            elif child.type == "member_expression":
                result = self._extract_member_name(child)
                if result:
                    return result

        return None

    def _extract_subscript_name(self, subscript_node: "Node") -> Optional[str]:
        """Extract method name from subscript expression."""
        for child in subscript_node.children:
            if child.type == "string":
                text = self._get_node_text(child)
                if text:
                    return text.strip("\"'")
        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node."""
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text
