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

"""TypeScript language plugin."""

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


class TypeScriptPlugin(BaseLanguagePlugin):
    """TypeScript language plugin.

    Supports:
    - Testing: jest, vitest
    - Formatting: prettier
    - Linting: eslint, typescript-eslint
    - Type checking: tsc
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="typescript",
            display_name="TypeScript",
            aliases=["ts", "tsx"],
            extensions=[".ts", ".tsx", ".mts", ".cts"],
            filenames=["tsconfig.json"],
            shebangs=["ts-node", "tsx"],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', "'", "`"],
            indent_size=2,
            use_tabs=False,
            package_managers=["npm", "yarn", "pnpm", "bun"],
            build_systems=["tsc", "webpack", "vite", "esbuild"],
            test_frameworks=["jest", "vitest", "mocha"],
            language_server="typescript-language-server",
            language_server_name="TypeScript Language Server",
            tree_sitter_language="typescript",
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
        """Create tree-sitter queries for TypeScript symbol/call extraction.

        The @def capture is used for end_line (function body boundaries).
        The @name capture is used for the symbol name.
        """
        return TreeSitterQueries(
            symbols=[
                # tree-sitter-typescript: class_declaration name is type_identifier,
                # NOT identifier. function_declaration still uses identifier.
                QueryPattern("class", "(class_declaration name: (type_identifier) @name) @def"),
                QueryPattern("function", "(function_declaration name: (identifier) @name) @def"),
                QueryPattern(
                    "function",
                    "(method_signature name: (property_identifier) @name) @def",
                ),
                QueryPattern(
                    "function",
                    "(method_definition name: (property_identifier) @name) @def",
                ),
                QueryPattern(
                    "function",
                    "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function))) @def",
                ),
                QueryPattern(
                    "function",
                    "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression))) @def",
                ),
                QueryPattern(
                    "function",
                    "(assignment_expression left: (identifier) @name right: (arrow_function)) @def",
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
                    name: (type_identifier) @child
                    (class_heritage
                        (extends_clause (identifier) @base)))
            """,
            implements="""
                (class_declaration
                    name: (type_identifier) @child
                    (class_heritage
                        (implements_clause (type_identifier) @interface)))
            """,
            composition="""
                (class_declaration
                    name: (type_identifier) @owner
                    body: (class_body
                        (public_field_definition
                            type: (type_annotation (type_identifier) @type))))
            """,
            enclosing_scopes=[
                ("function_declaration", "name"),
                ("method_definition", "name"),
                ("method_signature", "name"),
                ("class_declaration", "name"),
            ],
        )

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Get jest or vitest runner."""
        package_json = project_root / "package.json"

        if package_json.exists():
            content = package_json.read_text()

            if '"vitest"' in content:
                return TestRunner(
                    name="vitest",
                    command=["npx", "vitest"],
                    file_pattern="*.test.ts",
                    run_args=["run"],
                    coverage_args=["--coverage"],
                    output_format="json",
                    json_args=["--reporter=json"],
                )

            if '"jest"' in content:
                return TestRunner(
                    name="jest",
                    command=["npx", "jest"],
                    file_pattern="*.test.ts",
                    coverage_args=["--coverage"],
                    output_format="json",
                    json_args=["--json"],
                )

        return TestRunner(
            name="jest",
            command=["npx", "jest"],
            file_pattern="*.test.ts",
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
        """Get eslint linter with TypeScript support."""
        return Linter(
            name="eslint",
            command=["npx", "eslint", "--ext", ".ts,.tsx", "."],
            fix_args=["--fix"],
            config_file=".eslintrc",
        )

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Get TypeScript build system."""
        tsconfig = project_root / "tsconfig.json"

        if not tsconfig.exists():
            return None

        return BuildSystem(
            name="tsc",
            build_command=["npx", "tsc"],
            run_command=["npx", "ts-node"],
            clean_command=["rm", "-rf", "dist"],
            manifest_file="tsconfig.json",
        )

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in TypeScript source code.

        Finds function calls, method calls, and property access chains.
        Handles TypeScript-specific features like interfaces and type annotations.

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
                "language": "typescript",
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
                "method_signature",  # TypeScript specific
            ],
            class_types=["class_declaration", "interface_declaration"],
            call_types=["call_expression", "new_expression"],
            name_field="identifier",
            scope_body_types=["class_body", "interface_body"],
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

        if node.type == "method_signature":
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
