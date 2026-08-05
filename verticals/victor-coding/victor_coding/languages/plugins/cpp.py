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

"""C++ language plugin."""

import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from victor_coding.languages.base import (
    BaseLanguagePlugin,
    BuildSystem,
    CallEdge,
    CommentStyle,
    DocCommentPattern,
    EdgeDetectionResult,
    Formatter,
    LanguageCapabilities,
    LanguageConfig,
    Linter,
    QueryPattern,
    TestRunner,
    TreeSitterQueries,
)

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)


class CppPlugin(BaseLanguagePlugin):
    """C++ language plugin.

    Supports:
    - Testing: Google Test, Catch2
    - Formatting: clang-format
    - Linting: clang-tidy, cppcheck
    - Building: CMake, Make
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="cpp",
            display_name="C++",
            aliases=["c++", "cxx", "cc"],
            extensions=[".cpp", ".cc", ".cxx", ".hpp", ".hxx", ".h", ".c"],
            filenames=["CMakeLists.txt", "Makefile"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', "'"],
            indent_size=4,
            use_tabs=False,
            package_managers=["conan", "vcpkg"],
            build_systems=["cmake", "make", "ninja"],
            test_frameworks=["gtest", "catch2", "doctest"],
            language_server="clangd",
            language_server_name="clangd",
            tree_sitter_language="cpp",
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
        """Create tree-sitter queries for C++ symbol/call extraction."""
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_specifier name: (type_identifier) @name)"),
                QueryPattern("class", "(struct_specifier name: (type_identifier) @name)"),
                QueryPattern(
                    "function",
                    "(function_definition declarator: (function_declarator declarator: (identifier) @name))",
                ),
                QueryPattern(
                    "function",
                    "(function_definition declarator: (function_declarator declarator: (field_identifier) @name))",
                ),
            ],
            calls="""
                (call_expression function: (identifier) @callee)
                (call_expression function: (field_expression field: (field_identifier) @callee))
                (new_expression type: (type_identifier) @callee)
            """,
            inheritance="""
                (class_specifier
                    name: (type_identifier) @child
                    (base_class_clause (type_identifier) @base)
                )
            """,
            # C++ has no distinct "implements" — interfaces are abstract classes
            # represented as base classes. Reuse the inheritance pattern so
            # protocol-level inheritance still gets surfaced as IMPLEMENTS-style
            # edges when the consumer asks for them.
            implements="""
                (class_specifier
                    name: (type_identifier) @child
                    (base_class_clause (type_identifier) @base)
                )
            """,
            composition="""
                (class_specifier
                    name: (type_identifier) @owner
                    body: (field_declaration_list
                        (field_declaration
                            type: (type_identifier) @type)))
            """,
            enclosing_scopes=[
                ("function_definition", "declarator"),
                ("class_specifier", "name"),
            ],
        )

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Get Google Test or Catch2 runner."""
        cmake_file = project_root / "CMakeLists.txt"

        if cmake_file.exists():
            content = cmake_file.read_text()

            # Check for Google Test
            if "gtest" in content.lower() or "googletest" in content.lower():
                return TestRunner(
                    name="gtest",
                    command=["ctest", "--output-on-failure"],
                    file_pattern="*_test.cpp",
                    run_args=["--verbose"],
                    parallel_args=["-j", "auto"],
                )

            # Check for Catch2
            if "catch2" in content.lower():
                return TestRunner(
                    name="catch2",
                    command=["ctest", "--output-on-failure"],
                    file_pattern="*_test.cpp",
                    run_args=["--verbose"],
                )

        return TestRunner(
            name="ctest",
            command=["ctest", "--output-on-failure"],
            file_pattern="*_test.cpp",
        )

    def get_formatter(self, project_root: Path) -> Optional[Formatter]:
        """Get clang-format formatter."""
        return Formatter(
            name="clang-format",
            command=["clang-format", "-i", "**/*.cpp", "**/*.hpp"],
            check_args=["--dry-run", "--Werror"],
            config_file=".clang-format",
        )

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Get clang-tidy linter."""
        return Linter(
            name="clang-tidy",
            command=["clang-tidy", "**/*.cpp"],
            fix_args=["--fix"],
            config_file=".clang-tidy",
        )

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Get CMake or Make build system."""
        cmake_file = project_root / "CMakeLists.txt"

        if cmake_file.exists():
            return BuildSystem(
                name="cmake",
                build_command=["cmake", "--build", "build"],
                run_command=["./build/main"],
                clean_command=["cmake", "--build", "build", "--target", "clean"],
                install_command=["cmake", "--install", "build"],
                debug_args=["--config", "Debug"],
                release_args=["--config", "Release"],
                manifest_file="CMakeLists.txt",
            )

        makefile = project_root / "Makefile"
        if makefile.exists():
            return BuildSystem(
                name="make",
                build_command=["make"],
                run_command=["./main"],
                clean_command=["make", "clean"],
                install_command=["make", "install"],
                manifest_file="Makefile",
            )

        return None

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in C++ source code.

        Finds function calls, method calls, and constructor calls.
        Handles C++ features like namespaces and templates (basic support).

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
                "language": "cpp",
                "file": str(file_path),
            },
        )

    def _find_call_nodes(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Tracks namespace context for proper function resolution.
        """
        results: List[tuple["Node", str, Optional[int]]] = []

        def traverse(
            node: "Node",
            enclosing_function: Optional[str] = None,
            namespace_context: List[str] = None,
        ) -> None:
            """Recursively traverse tree finding calls."""
            if namespace_context is None:
                namespace_context = []

            # Check if this is a namespace definition
            if node.type == "namespace_definition":
                # Get namespace name
                ns_name = None
                for child in node.children:
                    if child.type == "identifier":
                        ns_name = self._get_node_text(child)
                        break

                new_namespace = namespace_context.copy()
                if ns_name:
                    new_namespace.append(ns_name)

                # Process children with namespace context
                for child in node.children:
                    traverse(child, enclosing_function, new_namespace)
                return

            # Check if this is a function definition
            if node.type == "function_definition":
                func_name = self._extract_function_name(node)
                if not func_name:
                    func_name = enclosing_function

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name, namespace_context)
                return

            # Check if this is a class/struct specifier
            if node.type in ("class_specifier", "struct_specifier"):
                class_name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        class_name = self._get_node_text(child)
                        break

                # Process class body
                for child in node.children:
                    if child.type == "field_declaration_list":
                        for grandchild in child.children:
                            traverse(
                                grandchild,
                                class_name or enclosing_function,
                                namespace_context,
                            )
                    else:
                        traverse(child, enclosing_function, namespace_context)
                return

            # Check if this is a call expression
            if node.type == "call_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Check for new expressions (constructor calls)
            if node.type == "new_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Recurse into children
            for child in node.children:
                traverse(child, enclosing_function, namespace_context)

        traverse(root)
        return results

    def _extract_function_name(self, node: "Node") -> Optional[str]:
        """Extract function name from a function definition node.

        C++ functions can have complex declarators. This handles the common cases.
        """
        # Look for function_declarator
        for child in node.children:
            if child.type == "function_declarator":
                # Get the declarator (identifier or field_identifier)
                for grandchild in child.children:
                    if grandchild.type == "declarator":
                        for ggchild in grandchild.children:
                            if ggchild.type in ("identifier", "field_identifier"):
                                return self._get_node_text(ggchild)
                    elif grandchild.type in ("identifier", "field_identifier"):
                        return self._get_node_text(grandchild)

        # Fallback: look for identifier directly
        for child in node.children:
            if child.type in ("identifier", "field_identifier"):
                return self._get_node_text(child)

        return None

    def _extract_callee_name(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called function from a call node.

        Handles:
        - Simple calls: foo()
        - Field access calls: obj.method()
        - Template calls: foo<T>()
        - Namespace-qualified calls: std::foo()
        """
        # Get the function part of the call
        for child in call_node.children:
            if child.type == "field_expression":
                # obj.method() -> extract "method"
                return self._extract_field_name(child)
            elif child.type == "template_function":
                # foo<T>() -> extract "foo"
                return self._extract_template_name(child)
            elif child.type == "identifier":
                # foo() -> extract "foo"
                return self._get_node_text(child)
            elif child.type == "qualified_identifier":
                # std::foo -> extract "foo"
                return self._extract_qualified_name(child)

        return None

    def _extract_field_name(self, field_node: "Node") -> Optional[str]:
        """Extract field name from a field_expression node.

        For obj.method, extracts "method".
        For obj.field1.field2, extracts "field2" (the final field).
        """
        # field_expression: object (identifier or field_expression) . field: (field_identifier)
        for child in reversed(field_node.children):
            if child.type == "field_identifier":
                return self._get_node_text(child)
            elif child.type == "field_expression":
                result = self._extract_field_name(child)
                if result:
                    return result

        return None

    def _extract_template_name(self, template_node: "Node") -> Optional[str]:
        """Extract the base name from a template function.

        For foo<T>(), extracts "foo".
        """
        for child in template_node.children:
            if child.type == "identifier":
                return self._get_node_text(child)
            elif child.type == "field_expression":
                return self._extract_field_name(child)

        return None

    def _extract_qualified_name(self, qualified_node: "Node") -> Optional[str]:
        """Extract the final identifier from a qualified identifier.

        For std::foo, extracts "foo".
        For ns::Class::method, extracts "method".
        """
        # qualified_identifier: identifier :: identifier (or nested)
        # We want the last identifier
        for child in reversed(qualified_node.children):
            if child.type == "identifier":
                return self._get_node_text(child)
            elif child.type == "qualified_identifier":
                result = self._extract_qualified_name(child)
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
