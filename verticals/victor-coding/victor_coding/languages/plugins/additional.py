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

"""Additional language plugins for less common but supported languages.

These plugins support embedding-based indexing for:
- C (clang, gcc)
- Kotlin (kotlinc, gradle)
- C# (dotnet, msbuild)
- Ruby (bundler, rspec)
- PHP (composer, phpunit)
- Swift (swift build, xcode)
- Scala (sbt, mill)
- Bash/Shell scripts
- SQL
- HTML/CSS/SCSS
- Lua
- Elixir
- Haskell
- R
- Markdown/reStructuredText
"""

import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from victor_coding.languages.base import (
    BaseLanguagePlugin,
    CallEdge,
    CommentStyle,
    EdgeDetectionResult,
    LanguageCapabilities,
    LanguageConfig,
    QueryPattern,
    TreeSitterQueries,
)

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)


class CPlugin(BaseLanguagePlugin):
    """C language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="c",
            display_name="C",
            aliases=["clang"],
            extensions=[".c", ".h"],
            filenames=["Makefile", "CMakeLists.txt"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"'],
            indent_size=4,
            use_tabs=False,
            package_managers=[],
            build_systems=["make", "cmake", "meson"],
            test_frameworks=["unity", "cunit"],
            language_server="clangd",
            language_server_name="clangd",
            tree_sitter_language="c",
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
        """Create tree-sitter queries for C/C++ symbol extraction.

        The @def capture is used for end_line (function body boundaries).
        """
        return TreeSitterQueries(
            symbols=[
                QueryPattern(
                    "function",
                    "(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def",
                ),
                QueryPattern("class", "(struct_specifier name: (type_identifier) @name) @def"),
                QueryPattern("class", "(enum_specifier name: (type_identifier) @name) @def"),
            ],
            calls="""
                (call_expression function: (identifier) @callee)
                (call_expression function: (field_expression field: (field_identifier) @callee))
            """,
            references="""
                (call_expression function: (identifier) @name)
                (identifier) @name
            """,
            enclosing_scopes=[
                ("function_definition", "declarator"),
            ],
        )

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in C source code.

        Finds function calls and method calls through pointers.
        Handles C-specific features like function pointers and struct methods.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []
        call_nodes = self._find_call_nodes_c(tree.root_node)

        for call_node, caller_name, caller_line in call_nodes:
            callee_name = self._extract_callee_name_c(call_node)
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
                "language": "c",
                "file": str(file_path),
            },
        )

    def _find_call_nodes_c(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Args:
            root: Tree-sitter root node

        Returns:
            List of (call_node, caller_name, caller_line) tuples
        """
        results: List[tuple["Node", str, Optional[int]]] = []

        def traverse(node: "Node", enclosing_function: Optional[str] = None) -> None:
            # Check if this is a function definition
            if node.type == "function_definition":
                func_name = self._extract_function_name_c(node)
                if not func_name:
                    func_name = enclosing_function

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name)
                return

            # Check if this is a struct specifier
            if node.type == "struct_specifier":
                struct_name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        struct_name = self._get_node_text(child)
                        break

                # Process struct body
                for child in node.children:
                    if child.type == "field_declaration_list":
                        for grandchild in child.children:
                            traverse(grandchild, struct_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check if this is a call expression
            if node.type == "call_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Recurse into children
            for child in node.children:
                traverse(child, enclosing_function)

        traverse(root)
        return results

    def _extract_function_name_c(self, node: "Node") -> Optional[str]:
        """Extract function name from a function definition node.

        Args:
            node: Function definition node

        Returns:
            Function name or None
        """
        # Look for function_declarator
        for child in node.children:
            if child.type == "function_declarator":
                # Get the declarator (identifier)
                for grandchild in child.children:
                    if grandchild.type == "declarator":
                        for ggchild in grandchild.children:
                            if ggchild.type == "identifier":
                                return self._get_node_text(ggchild)
                    elif grandchild.type == "identifier":
                        return self._get_node_text(grandchild)

        # Fallback: look for identifier directly
        for child in node.children:
            if child.type == "identifier":
                return self._get_node_text(child)

        return None

    def _extract_callee_name_c(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called function from a call node.

        Handles:
        - Simple calls: foo()
        - Field expression calls: ptr->method()

        Args:
            call_node: Call expression node

        Returns:
            Callee name or None
        """
        # Get the function part of the call
        for child in call_node.children:
            if child.type == "field_expression":
                # ptr->method() -> extract "method"
                return self._extract_field_name_c(child)
            elif child.type == "identifier":
                # foo() -> extract "foo"
                return self._get_node_text(child)

        return None

    def _extract_field_name_c(self, field_node: "Node") -> Optional[str]:
        """Extract field name from a field_expression node.

        For ptr->method, extracts "method".

        Args:
            field_node: Field expression node

        Returns:
            Field name or None
        """
        # field_expression: argument . field: (field_identifier)
        # or for pointer access: argument -> field: (field_identifier)
        for child in reversed(field_node.children):
            if child.type == "field_identifier":
                return self._get_node_text(child)

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node.

        Args:
            node: Tree-sitter node

        Returns:
            Node text or None
        """
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text


class KotlinPlugin(BaseLanguagePlugin):
    """Kotlin language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="kotlin",
            display_name="Kotlin",
            aliases=["kt", "kts"],
            extensions=[".kt", ".kts"],
            filenames=["build.gradle.kts", "settings.gradle.kts"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', '"""'],
            indent_size=4,
            use_tabs=False,
            package_managers=["gradle", "maven"],
            build_systems=["gradle", "maven"],
            test_frameworks=["junit", "kotest"],
            language_server="kotlin-language-server",
            language_server_name="Kotlin Language Server",
            tree_sitter_language="kotlin",
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
        # tree-sitter-kotlin uses `identifier` everywhere — `simple_identifier`
        # and `type_identifier` no longer exist as distinct node types.
        # Delegation goes through `delegation_specifiers` -> `delegation_specifier`,
        # and the call form is `call_expression` with a bare `identifier` or
        # a `navigation_expression` for `obj.method()`.
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_declaration (identifier) @name)"),
                QueryPattern("function", "(function_declaration (identifier) @name)"),
            ],
            calls="""
                (call_expression (identifier) @callee)
                (call_expression (navigation_expression (identifier) @callee .))
            """,
            references="""
                (identifier) @name
            """,
            inheritance="""
                (class_declaration
                    (identifier) @child
                    (delegation_specifiers
                        (delegation_specifier
                            (constructor_invocation (user_type (identifier) @base)))))
            """,
            implements="""
                (class_declaration
                    (identifier) @child
                    (delegation_specifiers
                        (delegation_specifier
                            (user_type (identifier) @interface))))
            """,
            composition="""
                (class_declaration
                    (identifier) @owner
                    (class_body
                        (property_declaration
                            (variable_declaration (user_type (identifier) @type)))))
            """,
            enclosing_scopes=[
                ("function_declaration", "name"),
                ("class_declaration", "name"),
            ],
        )

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in Kotlin source code.

        Finds function calls, method calls, and property access chains.
        Handles Kotlin-specific features like safe call operators and extension functions.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []
        call_nodes = self._find_call_nodes_kotlin(tree.root_node)

        for call_node, caller_name, caller_line in call_nodes:
            callee_name = self._extract_callee_name_kotlin(call_node)
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
                "language": "kotlin",
                "file": str(file_path),
            },
        )

    def _find_call_nodes_kotlin(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Args:
            root: Tree-sitter root node

        Returns:
            List of (call_node, caller_name, caller_line) tuples
        """
        results: List[tuple["Node", str, Optional[int]]] = []

        def traverse(node: "Node", enclosing_function: Optional[str] = None) -> None:
            # Check if this is a function declaration
            if node.type == "function_declaration":
                func_name = None
                for child in node.children:
                    if child.type == "identifier":
                        func_name = self._get_node_text(child)
                        break

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name or enclosing_function)
                return

            # Check for class declarations
            if node.type == "class_declaration":
                class_name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        class_name = self._get_node_text(child)
                        break

                # Process class body
                for child in node.children:
                    if child.type == "class_body":
                        for grandchild in child.children:
                            traverse(grandchild, class_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check for object declarations (Kotlin singletons)
            if node.type == "object_declaration":
                obj_name = None
                for child in node.children:
                    if child.type == "simple_identifier":
                        obj_name = self._get_node_text(child)
                        break

                # Process object body
                for child in node.children:
                    if child.type == "class_body":
                        for grandchild in child.children:
                            traverse(grandchild, obj_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check if this is a call expression
            if node.type == "call_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Recurse into children
            for child in node.children:
                traverse(child, enclosing_function)

        traverse(root)
        return results

    def _extract_callee_name_kotlin(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called function from a call node.

        Handles:
        - Simple calls: foo()
        - Navigation expressions: obj.method()
        - Safe navigation: obj?.method()

        Args:
            call_node: Call expression node

        Returns:
            Callee name or None
        """
        for child in call_node.children:
            if child.type == "navigation_expression":
                # obj.method() -> extract "method"
                return self._extract_navigation_name(child)
            elif child.type == "identifier":
                # foo() -> extract "foo"
                return self._get_node_text(child)

        return None

    def _extract_navigation_name(self, navigation_node: "Node") -> Optional[str]:
        """Extract method name from a navigation_expression node.

        For obj.method, extracts "method".
        Handles safe navigation: obj?.method

        Args:
            navigation_node: Navigation expression node

        Returns:
            Method name or None
        """
        # navigation_expression can be nested: obj1.obj2.method
        # We want the last identifier (not simple_identifier!)
        for child in reversed(navigation_node.children):
            if child.type == "identifier":
                return self._get_node_text(child)
            elif child.type == "navigation_expression":
                result = self._extract_navigation_name(child)
                if result:
                    return result

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node.

        Args:
            node: Tree-sitter node

        Returns:
            Node text or None
        """
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text


class CSharpPlugin(BaseLanguagePlugin):
    """C# language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="csharp",
            display_name="C#",
            aliases=["cs", "c_sharp"],
            extensions=[".cs"],
            filenames=["*.csproj", "*.sln"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', '@"', '"""'],
            indent_size=4,
            use_tabs=False,
            package_managers=["nuget"],
            build_systems=["dotnet", "msbuild"],
            test_frameworks=["xunit", "nunit", "mstest"],
            language_server="omnisharp",
            language_server_name="OmniSharp",
            tree_sitter_language="c_sharp",
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_declaration name: (identifier) @name)"),
                QueryPattern("class", "(interface_declaration name: (identifier) @name)"),
                QueryPattern("function", "(method_declaration name: (identifier) @name)"),
            ],
            calls="""
                (invocation_expression (identifier) @callee)
                (invocation_expression (member_access_expression name: (identifier) @callee))
                (object_creation_expression type: (identifier) @callee)
            """,
            references="""
                (identifier) @name
            """,
            inheritance="""
                (class_declaration
                    name: (identifier) @child
                    (base_list (identifier) @base))
            """,
            # Same physical shape as inheritance — C#'s grammar puts both
            # base classes and interfaces into one `base_list`. The consumer
            # disambiguates by symbol classification (an interface name
            # by convention starts with `I`, or by looking up the target
            # symbol's declaration kind).
            implements="""
                (class_declaration
                    name: (identifier) @child
                    (base_list (identifier) @interface))
            """,
            composition="""
                (class_declaration
                    name: (identifier) @owner
                    (declaration_list
                        (field_declaration
                            (variable_declaration (identifier) @type))))
            """,
            enclosing_scopes=[
                ("method_declaration", "name"),
                ("class_declaration", "name"),
            ],
        )

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in C# source code.

        Finds method calls, constructor calls, and static method calls.
        Handles C#-specific features like generics, extension methods, and LINQ.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []
        call_nodes = self._find_call_nodes_csharp(tree.root_node)

        for call_node, caller_name, caller_line in call_nodes:
            callee_name = self._extract_callee_name_csharp(call_node)
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
                "language": "csharp",
                "file": str(file_path),
            },
        )

    def _find_call_nodes_csharp(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Args:
            root: Tree-sitter root node

        Returns:
            List of (call_node, caller_name, caller_line) tuples
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

            # Check if this is a namespace declaration
            if node.type == "namespace_declaration":
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

            # Check if this is a method declaration
            if node.type == "method_declaration":
                func_name = None
                for child in node.children:
                    if child.type == "identifier":
                        func_name = self._get_node_text(child)
                        break

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name or enclosing_function, namespace_context)
                return

            # Check for class/interface declarations
            if node.type in (
                "class_declaration",
                "struct_declaration",
                "interface_declaration",
            ):
                class_name = None
                for child in node.children:
                    if child.type == "identifier":
                        class_name = self._get_node_text(child)
                        break

                # Process class body
                for child in node.children:
                    if child.type == "declaration_list":
                        for grandchild in child.children:
                            traverse(
                                grandchild,
                                class_name or enclosing_function,
                                namespace_context,
                            )
                    else:
                        traverse(child, enclosing_function, namespace_context)
                return

            # Check if this is an invocation expression (method call)
            if node.type == "invocation_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Check for object creation expressions (constructor calls)
            if node.type == "object_creation_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Recurse into children
            for child in node.children:
                traverse(child, enclosing_function, namespace_context)

        traverse(root)
        return results

    def _extract_callee_name_csharp(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called method from a call node.

        Handles:
        - Simple calls: Method()
        - Member access calls: obj.Method()
        - Static calls: ClassName.Method()

        Args:
            call_node: Call expression node

        Returns:
            Callee name or None
        """
        # For invocation_expression, find the function being called
        for child in call_node.children:
            if child.type == "member_access_expression":
                # obj.Method() -> extract "Method"
                return self._extract_member_name_csharp(child)
            elif child.type == "identifier":
                # Method() -> extract "Method"
                return self._get_node_text(child)

        # For object_creation_expression
        if call_node.type == "object_creation_expression":
            for child in call_node.children:
                if child.type == "identifier":
                    return self._get_node_text(child)
                elif child.type == "generic_name":
                    # GenericType<>() -> extract "GenericType"
                    for ggchild in child.children:
                        if ggchild.type == "identifier":
                            return self._get_node_text(ggchild)

        return None

    def _extract_member_name_csharp(self, member_node: "Node") -> Optional[str]:
        """Extract member name from a member_access_expression node.

        For obj.Method, extracts "Method".
        For obj1.obj2.Method, extracts "Method" (the final member).

        Args:
            member_node: Member access expression node

        Returns:
            Member name or None
        """
        # member_access_expression: object . name: (identifier)
        for child in reversed(member_node.children):
            if child.type == "identifier":
                return self._get_node_text(child)
            elif child.type == "generic_name":
                # Generic method: Method<T>()
                for ggchild in child.children:
                    if ggchild.type == "identifier":
                        return self._get_node_text(ggchild)
            elif child.type == "member_access_expression":
                result = self._extract_member_name_csharp(child)
                if result:
                    return result

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node.

        Args:
            node: Tree-sitter node

        Returns:
            Node text or None
        """
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text


class RubyPlugin(BaseLanguagePlugin):
    """Ruby language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="ruby",
            display_name="Ruby",
            aliases=["rb"],
            extensions=[".rb", ".rake", ".gemspec"],
            filenames=["Gemfile", "Rakefile", ".ruby-version"],
            shebangs=["ruby"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            block_comment_start="=begin",
            block_comment_end="=end",
            string_delimiters=['"', "'", '"""'],
            indent_size=2,
            use_tabs=False,
            package_managers=["bundler", "gem"],
            build_systems=["rake"],
            test_frameworks=["rspec", "minitest"],
            language_server="solargraph",
            language_server_name="Solargraph",
            tree_sitter_language="ruby",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=False,  # Ruby is dynamic
            supports_rename=True,
            supports_extract_function=True,
            supports_inline=True,
            supports_organize_imports=False,
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class name: (constant) @name)"),
                QueryPattern("class", "(module name: (constant) @name)"),
                QueryPattern("function", "(method name: (identifier) @name)"),
                QueryPattern("function", "(singleton_method name: (identifier) @name)"),
            ],
            calls="""
                (call method: (identifier) @callee)
                (call receiver: (identifier) method: (identifier) @callee)
            """,
            references="""
                (identifier) @name
                (constant) @name
            """,
            inheritance="""
                (class
                    name: (constant) @child
                    superclass: (superclass (constant) @base))
            """,
            implements="""
                (class
                    name: (constant) @child
                    body: (body_statement
                        (call method: (identifier) @_include (#eq? @_include "include")
                            arguments: (argument_list (constant) @interface))))
            """,
            enclosing_scopes=[
                ("method", "name"),
                ("class", "name"),
                ("module", "name"),
            ],
        )

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in Ruby source code.

        Finds method calls, including blocks and dynamic calls.
        Handles Ruby-specific features like safe navigation and singleton methods.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []
        call_nodes = self._find_call_nodes_ruby(tree.root_node)

        for call_node, caller_name, caller_line in call_nodes:
            callee_name = self._extract_callee_name_ruby(call_node)
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
                "language": "ruby",
                "file": str(file_path),
            },
        )

    def _find_call_nodes_ruby(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Args:
            root: Tree-sitter root node

        Returns:
            List of (call_node, caller_name, caller_line) tuples
        """
        results: List[tuple["Node", str, Optional[int]]] = []

        def traverse(node: "Node", enclosing_function: Optional[str] = None) -> None:
            # Check if this is a method definition
            if node.type == "method":
                func_name = None
                for child in node.children:
                    if child.type == "identifier":
                        func_name = self._get_node_text(child)
                        break

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name or enclosing_function)
                return

            # Check for singleton method definitions (def self.method)
            if node.type == "singleton_method":
                func_name = None
                for child in node.children:
                    if child.type == "identifier":
                        func_name = self._get_node_text(child)
                        break

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name or enclosing_function)
                return

            # Check for class declarations
            if node.type == "class":
                class_name = None
                for child in node.children:
                    if child.type == "constant":
                        class_name = self._get_node_text(child)
                        break

                # Process class body
                for child in node.children:
                    if child.type == "body":
                        for grandchild in child.children:
                            traverse(grandchild, class_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check for module declarations
            if node.type == "module":
                module_name = None
                for child in node.children:
                    if child.type == "constant":
                        module_name = self._get_node_text(child)
                        break

                # Process module body
                for child in node.children:
                    if child.type == "body":
                        for grandchild in child.children:
                            traverse(grandchild, module_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check if this is a call expression
            if node.type == "call":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Check for bare identifiers as method calls (Ruby allows foo without parens)
            # Only capture identifiers that are direct children of body_statement
            # This avoids capturing variable declarations, parameters, etc.
            if node.type == "body_statement" and enclosing_function:
                for child in node.children:
                    if child.type == "identifier":
                        # This is likely a bare method call like: foo
                        # Create a synthetic call node for consistency
                        results.append((child, enclosing_function or "", child.start_point[0] + 1))
                        break  # Only capture the first identifier per statement

            # Recurse into children
            for child in node.children:
                traverse(child, enclosing_function)

        traverse(root)
        return results

    def _extract_callee_name_ruby(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called method from a call node.

        Handles:
        - Simple calls: method (identifier node itself)
        - Method calls with receiver: obj.method (call node)
        - Safe navigation: obj&.method

        For simple calls, returns the identifier's text.
        For calls with receivers (obj.method), returns the last identifier.

        Args:
            call_node: Call node

        Returns:
            Callee name or None
        """
        # If the node itself is an identifier (bare method call)
        if call_node.type == "identifier":
            return self._get_node_text(call_node)

        # For method calls with receivers (obj.method), the LAST identifier is the method name
        # (first would be the receiver like "obj" in "obj.method")
        identifiers = []
        for child in call_node.children:
            if child.type == "identifier":
                identifiers.append(self._get_node_text(child))

        # Return the last identifier (method name), or first if only one
        if identifiers:
            return identifiers[-1] if len(identifiers) > 1 else identifiers[0]

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node.

        Args:
            node: Tree-sitter node

        Returns:
            Node text or None
        """
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text


class PhpPlugin(BaseLanguagePlugin):
    """PHP language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="php",
            display_name="PHP",
            aliases=[],
            extensions=[".php", ".phtml", ".php5", ".php7"],
            filenames=["composer.json", "composer.lock"],
            shebangs=["php"],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', "'"],
            indent_size=4,
            use_tabs=False,
            package_managers=["composer"],
            build_systems=[],
            test_frameworks=["phpunit", "pest"],
            language_server="intelephense",
            language_server_name="Intelephense",
            tree_sitter_language="php",
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_declaration name: (name) @name)"),
                QueryPattern("class", "(interface_declaration name: (name) @name)"),
                QueryPattern("class", "(trait_declaration name: (name) @name)"),
                QueryPattern("function", "(function_definition name: (name) @name)"),
                QueryPattern("function", "(method_declaration name: (name) @name)"),
            ],
            calls="""
                (function_call_expression function: (name) @callee)
                (member_call_expression name: (name) @callee)
                (object_creation_expression (name) @callee)
            """,
            references="""
                (name) @name
            """,
            inheritance="""
                (class_declaration
                    name: (name) @child
                    (base_clause (name) @base))
            """,
            implements="""
                (class_declaration
                    name: (name) @child
                    (class_interface_clause (name) @interface))
            """,
            composition="""
                (class_declaration
                    name: (name) @owner
                    body: (declaration_list
                        (property_declaration
                            type: (named_type (name) @type))))
            """,
            enclosing_scopes=[
                ("function_definition", "name"),
                ("method_declaration", "name"),
                ("class_declaration", "name"),
            ],
        )

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in PHP source code.

        Finds function calls, method calls, and constructor calls.
        Handles PHP-specific features like static methods and namespaces.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []
        call_nodes = self._find_call_nodes_php(tree.root_node)

        for call_node, caller_name, caller_line in call_nodes:
            callee_name = self._extract_callee_name_php(call_node)
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
                "language": "php",
                "file": str(file_path),
            },
        )

    def _find_call_nodes_php(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Args:
            root: Tree-sitter root node

        Returns:
            List of (call_node, caller_name, caller_line) tuples
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
                    if child.type == "namespace_name":
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
                func_name = None
                for child in node.children:
                    if child.type == "name":
                        func_name = self._get_node_text(child)
                        break

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name or enclosing_function, namespace_context)
                return

            # Check if this is a method declaration
            if node.type == "method_declaration":
                func_name = None
                for child in node.children:
                    if child.type == "name":
                        func_name = self._get_node_text(child)
                        break

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name or enclosing_function, namespace_context)
                return

            # Check for class/interface/trait declarations
            if node.type in (
                "class_declaration",
                "interface_declaration",
                "trait_declaration",
            ):
                class_name = None
                for child in node.children:
                    if child.type == "name":
                        class_name = self._get_node_text(child)
                        break

                # Process class body
                for child in node.children:
                    if child.type == "declaration_list":
                        for grandchild in child.children:
                            traverse(
                                grandchild,
                                class_name or enclosing_function,
                                namespace_context,
                            )
                    else:
                        traverse(child, enclosing_function, namespace_context)
                return

            # Check if this is a function call expression
            if node.type == "function_call_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Check if this is a member call expression ($obj->method())
            if node.type == "member_call_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Check for scoped call expression (ClassName::method())
            if node.type == "scoped_call_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Check for object creation expression (new Class())
            if node.type == "object_creation_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Recurse into children
            for child in node.children:
                traverse(child, enclosing_function, namespace_context)

        traverse(root)
        return results

    def _extract_callee_name_php(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called function from a call node.

        Handles:
        - Simple calls: function()
        - Member calls: $obj->method()
        - Static calls: ClassName::method()
        - Object creation: new ClassName()

        Args:
            call_node: Call expression node

        Returns:
            Callee name or None
        """
        # For function_call_expression
        if call_node.type == "function_call_expression":
            for child in call_node.children:
                if child.type == "name":
                    return self._get_node_text(child)

        # For member_call_expression ($obj->method())
        if call_node.type == "member_call_expression":
            for child in call_node.children:
                if child.type == "name":
                    return self._get_node_text(child)

        # For scoped_call_expression (ClassName::method())
        if call_node.type == "scoped_call_expression":
            # The last name child is the method name
            names = []
            for child in call_node.children:
                if child.type == "name":
                    names.append(self._get_node_text(child))
            if names:
                # Return the last name (method name)
                return names[-1]

        # For object_creation_expression
        if call_node.type == "object_creation_expression":
            for child in call_node.children:
                if child.type == "name":
                    return self._get_node_text(child)

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node.

        Args:
            node: Tree-sitter node

        Returns:
            Node text or None
        """
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text


class SwiftPlugin(BaseLanguagePlugin):
    """Swift language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="swift",
            display_name="Swift",
            aliases=[],
            extensions=[".swift"],
            filenames=["Package.swift"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', '"""'],
            indent_size=4,
            use_tabs=False,
            package_managers=["swift package manager"],
            build_systems=["swift build", "xcodebuild"],
            test_frameworks=["xctest"],
            language_server="sourcekit-lsp",
            language_server_name="SourceKit-LSP",
            tree_sitter_language="swift",
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
        # tree-sitter-swift: there is no `struct_declaration` — structs and
        # classes share `class_declaration`. Inheritance/protocol-conformance
        # goes through repeated `inheritance_specifier` children, each
        # wrapping `user_type -> type_identifier`. Property types are
        # `type_annotation -> user_type -> type_identifier`, not bare
        # `type_identifier`.
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_declaration (type_identifier) @name)"),
                QueryPattern("class", "(protocol_declaration (type_identifier) @name)"),
                QueryPattern("function", "(function_declaration (simple_identifier) @name)"),
            ],
            calls="""
                (call_expression (simple_identifier) @callee)
            """,
            references="""
                (simple_identifier) @name
                (type_identifier) @name
            """,
            inheritance="""
                (class_declaration
                    (type_identifier) @child
                    (inheritance_specifier (user_type (type_identifier) @base)))
            """,
            # Same physical pattern as inheritance — Swift doesn't syntactically
            # distinguish class extension from protocol conformance at the
            # grammar level; both are inheritance_specifier children. The
            # consumer can disambiguate via runtime symbol classification.
            implements="""
                (class_declaration
                    (type_identifier) @child
                    (inheritance_specifier (user_type (type_identifier) @interface)))
            """,
            composition="""
                (class_declaration
                    (type_identifier) @owner
                    (class_body
                        (property_declaration
                            (type_annotation (user_type (type_identifier) @type)))))
            """,
            enclosing_scopes=[
                ("function_declaration", "name"),
                ("class_declaration", "name"),
            ],
        )

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in Swift source code.

        Finds function calls and method calls.
        Handles Swift-specific features like optional chaining and protocol methods.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []
        call_nodes = self._find_call_nodes_swift(tree.root_node)

        for call_node, caller_name, caller_line in call_nodes:
            callee_name = self._extract_callee_name_swift(call_node)
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
                "language": "swift",
                "file": str(file_path),
            },
        )

    def _find_call_nodes_swift(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Args:
            root: Tree-sitter root node

        Returns:
            List of (call_node, caller_name, caller_line) tuples
        """
        results: List[tuple["Node", str, Optional[int]]] = []

        def traverse(node: "Node", enclosing_function: Optional[str] = None) -> None:
            # Check if this is a function declaration
            if node.type == "function_declaration":
                func_name = None
                for child in node.children:
                    if child.type == "simple_identifier":
                        func_name = self._get_node_text(child)
                        break

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name or enclosing_function)
                return

            # Check for class declarations
            if node.type == "class_declaration":
                class_name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        class_name = self._get_node_text(child)
                        break

                # Process class body
                for child in node.children:
                    if child.type == "class_body":
                        for grandchild in child.children:
                            traverse(grandchild, class_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check for struct declarations
            if node.type == "struct_declaration":
                struct_name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        struct_name = self._get_node_text(child)
                        break

                # Process struct body
                for child in node.children:
                    if child.type == "class_body":
                        for grandchild in child.children:
                            traverse(grandchild, struct_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check for protocol declarations
            if node.type == "protocol_declaration":
                protocol_name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        protocol_name = self._get_node_text(child)
                        break

                # Process protocol body
                for child in node.children:
                    if child.type == "class_body":
                        for grandchild in child.children:
                            traverse(grandchild, protocol_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check if this is a call expression
            if node.type == "call_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Recurse into children
            for child in node.children:
                traverse(child, enclosing_function)

        traverse(root)
        return results

    def _extract_callee_name_swift(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called function from a call node.

        Handles:
        - Simple calls: function()
        - Member calls: obj.method()
        - Optional chaining: obj?.method

        Args:
            call_node: Call expression node

        Returns:
            Callee name or None
        """
        for child in call_node.children:
            if child.type == "navigation_expression":
                # obj.method() -> extract "method" from navigation_suffix
                return self._extract_navigation_method_name(child)
            elif child.type == "simple_identifier":
                # foo() -> extract "foo"
                return self._get_node_text(child)

        return None

    def _extract_navigation_method_name(self, navigation_node: "Node") -> Optional[str]:
        """Extract method name from a navigation_expression node.

        For obj.method, extracts "method" from navigation_suffix.
        Handles nested navigation: obj1.obj2.method

        Args:
            navigation_node: Navigation expression node

        Returns:
            Method name or None
        """
        # Look for navigation_suffix containing the method name
        for child in navigation_node.children:
            if child.type == "navigation_suffix":
                for grandchild in child.children:
                    if grandchild.type == "simple_identifier":
                        return self._get_node_text(grandchild)
            elif child.type == "navigation_expression":
                result = self._extract_navigation_method_name(child)
                if result:
                    return result

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node.

        Args:
            node: Tree-sitter node

        Returns:
            Node text or None
        """
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text


class ScalaPlugin(BaseLanguagePlugin):
    """Scala language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="scala",
            display_name="Scala",
            aliases=[],
            extensions=[".scala", ".sc"],
            filenames=["build.sbt"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', '"""'],
            indent_size=2,
            use_tabs=False,
            package_managers=["sbt", "mill"],
            build_systems=["sbt", "mill"],
            test_frameworks=["scalatest", "specs2"],
            language_server="metals",
            language_server_name="Metals",
            tree_sitter_language="scala",
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_definition name: (identifier) @name)"),
                QueryPattern("class", "(object_definition name: (identifier) @name)"),
                QueryPattern("class", "(trait_definition name: (identifier) @name)"),
                QueryPattern("function", "(function_definition name: (identifier) @name)"),
            ],
            calls="""
                (call_expression function: (identifier) @callee)
            """,
            references="""
                (identifier) @name
            """,
            inheritance="""
                (class_definition
                    name: (identifier) @child
                    (extends_clause (type_identifier) @base))
            """,
            implements="""
                (class_definition
                    name: (identifier) @child
                    (extends_clause
                        (type_identifier)
                        (type_identifier) @interface))
            """,
            composition="""
                (class_definition
                    name: (identifier) @owner
                    body: (template_body
                        (val_definition
                            pattern: (identifier)
                            type: (type_identifier) @type)))
            """,
            enclosing_scopes=[
                ("function_definition", "name"),
                ("class_definition", "name"),
            ],
        )


class BashPlugin(BaseLanguagePlugin):
    """Bash/Shell script plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="bash",
            display_name="Bash",
            aliases=["sh", "shell", "zsh"],
            extensions=[".sh", ".bash", ".zsh", ".ksh"],
            filenames=[".bashrc", ".zshrc", ".profile", ".bash_profile"],
            shebangs=["bash", "sh", "zsh"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            block_comment_start="",
            block_comment_end="",
            string_delimiters=['"', "'"],
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=["bats"],
            language_server="bash-language-server",
            language_server_name="Bash Language Server",
            tree_sitter_language="bash",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=True,
            supports_test_execution=True,
            supports_coverage=False,
            supports_debugging=True,
            supports_breakpoints=True,
            supports_step_debugging=True,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
            supports_control_flow_graph=True,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        return TreeSitterQueries(
            symbols=[
                QueryPattern("function", "(function_definition name: (word) @name)"),
            ],
            calls="""
                (command name: (command_name (word) @callee))
            """,
            references="""
                (word) @name
                (variable_name) @name
            """,
            enclosing_scopes=[
                ("function_definition", "name"),
            ],
        )


class SqlPlugin(BaseLanguagePlugin):
    """SQL language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="sql",
            display_name="SQL",
            aliases=["mysql", "postgresql", "sqlite"],
            extensions=[".sql"],
            filenames=[],
            shebangs=[],
            comment_style=CommentStyle.DOUBLE_DASH,
            line_comment="--",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=["'", '"'],
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=[],
            language_server="sql-language-server",
            language_server_name="SQL Language Server",
            tree_sitter_language="sql",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
            # PL/pgSQL / T-SQL / MySQL stored procs DO have control flow,
            # but the installed `tree-sitter-sql` grammar is ANSI-SELECT
            # focused — it emits ERROR nodes for BEGIN/IF/THEN/WHILE/LOOP
            # blocks, so a CCG walker can't find statements even when
            # they exist in source. Verified via the CCG coverage probe.
            # Flip to True only when a procedural-SQL grammar is wired.
            supports_control_flow_graph=False,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        return TreeSitterQueries(
            symbols=[
                # tree-sitter-sql wraps the create-statement nodes in `create_*`
                # (no `_statement` suffix); the name is an `object_reference`
                # containing an `identifier`.
                QueryPattern(
                    "function",
                    "(create_function (object_reference (identifier) @name))",
                ),
                QueryPattern(
                    "class",
                    "(create_table (object_reference (identifier) @name))",
                ),
            ],
            enclosing_scopes=[],
        )


class HtmlPlugin(BaseLanguagePlugin):
    """HTML language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="html",
            display_name="HTML",
            aliases=["htm"],
            extensions=[".html", ".htm", ".xhtml"],
            filenames=["index.html"],
            shebangs=[],
            comment_style=CommentStyle.HTML,
            line_comment="",
            block_comment_start="<!--",
            block_comment_end="-->",
            string_delimiters=['"', "'"],
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=[],
            language_server="html-languageserver",
            language_server_name="HTML Language Server",
            tree_sitter_language="html",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
            supports_control_flow_graph=False,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        return TreeSitterQueries()


class CssPlugin(BaseLanguagePlugin):
    """CSS language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="css",
            display_name="CSS",
            aliases=["scss", "less", "sass"],
            extensions=[".css", ".scss", ".less", ".sass"],
            filenames=[],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",  # Only for SCSS/Less
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', "'"],
            indent_size=2,
            use_tabs=False,
            package_managers=["npm"],
            build_systems=["sass", "less"],
            test_frameworks=[],
            language_server="css-languageserver",
            language_server_name="CSS Language Server",
            tree_sitter_language="css",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
            supports_control_flow_graph=False,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        return TreeSitterQueries()


class LuaPlugin(BaseLanguagePlugin):
    """Lua language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="lua",
            display_name="Lua",
            aliases=[],
            extensions=[".lua"],
            filenames=[],
            shebangs=["lua"],
            comment_style=CommentStyle.DOUBLE_DASH,
            line_comment="--",
            block_comment_start="--[[",
            block_comment_end="]]",
            string_delimiters=['"', "'", "[[", "]]"],
            indent_size=2,
            use_tabs=False,
            package_managers=["luarocks"],
            build_systems=[],
            test_frameworks=["busted"],
            language_server="lua-language-server",
            language_server_name="Lua Language Server",
            tree_sitter_language="lua",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=False,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=True,
            supports_test_execution=True,
            supports_coverage=False,
            supports_debugging=True,
            supports_breakpoints=True,
            supports_step_debugging=True,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
            supports_control_flow_graph=True,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        # tree-sitter-lua: there is no separate `local_function_declaration`
        # node — both `function foo()` and `local function foo()` parse as
        # `function_declaration` (the `local` keyword is a child token).
        # One pattern covers both forms.
        return TreeSitterQueries(
            symbols=[
                QueryPattern("function", "(function_declaration name: (identifier) @name)"),
            ],
            calls="""
                (function_call name: (identifier) @callee)
            """,
            references="""
                (identifier) @name
            """,
            enclosing_scopes=[
                ("function_declaration", "name"),
            ],
        )


class ElixirPlugin(BaseLanguagePlugin):
    """Elixir language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="elixir",
            display_name="Elixir",
            aliases=["ex", "exs"],
            extensions=[".ex", ".exs"],
            filenames=["mix.exs"],
            shebangs=["elixir"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            block_comment_start="",
            block_comment_end="",
            string_delimiters=['"', '"""'],
            indent_size=2,
            use_tabs=False,
            package_managers=["mix", "hex"],
            build_systems=["mix"],
            test_frameworks=["exunit"],
            language_server="elixir-ls",
            language_server_name="ElixirLS",
            tree_sitter_language="elixir",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=True,  # Via Dialyzer
            supports_rename=True,
            supports_extract_function=True,
            supports_inline=False,
            supports_organize_imports=False,
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern(
                    "class",
                    '(call target: (identifier) @_defmodule (#eq? @_defmodule "defmodule") (arguments (alias) @name))',
                ),
                QueryPattern(
                    "function",
                    '(call target: (identifier) @_def (#match? @_def "^def") (arguments (call target: (identifier) @name)))',
                ),
            ],
            calls="""
                (call target: (identifier) @callee)
            """,
            references="""
                (identifier) @name
                (alias) @name
            """,
            enclosing_scopes=[],
        )


class HaskellPlugin(BaseLanguagePlugin):
    """Haskell language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="haskell",
            display_name="Haskell",
            aliases=["hs"],
            extensions=[".hs", ".lhs"],
            filenames=["stack.yaml", "package.yaml", "*.cabal"],
            shebangs=["runhaskell"],
            comment_style=CommentStyle.DOUBLE_DASH,
            line_comment="--",
            block_comment_start="{-",
            block_comment_end="-}",
            string_delimiters=['"'],
            indent_size=2,
            use_tabs=False,
            package_managers=["cabal", "stack"],
            build_systems=["cabal", "stack"],
            test_frameworks=["hspec", "quickcheck"],
            language_server="haskell-language-server",
            language_server_name="Haskell Language Server",
            tree_sitter_language="haskell",
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
        # tree-sitter-haskell: nodes are `type_synomym` (the grammar's own
        # typo for "synonym"), `newtype`, `data_type`; their child holding
        # the type name is `(name)`, not `name: (type)`. Function
        # application is `(apply . (variable))` — the leading anchor `.`
        # captures only the function position, not argument variables.
        return TreeSitterQueries(
            symbols=[
                QueryPattern("function", "(function name: (variable) @name)"),
                QueryPattern("class", "(type_synomym (name) @name)"),
                QueryPattern("class", "(newtype (name) @name)"),
                QueryPattern("class", "(data_type (name) @name)"),
            ],
            calls="""
                (apply . (variable) @callee)
            """,
            references="""
                (variable) @name
                (constructor) @name
            """,
            enclosing_scopes=[
                ("function", "name"),
            ],
        )


class RPlugin(BaseLanguagePlugin):
    """R language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="r",
            display_name="R",
            aliases=["rlang"],
            extensions=[".r", ".R", ".Rmd"],
            filenames=["DESCRIPTION", "NAMESPACE"],
            shebangs=["Rscript"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            block_comment_start="",
            block_comment_end="",
            string_delimiters=['"', "'"],
            indent_size=2,
            use_tabs=False,
            package_managers=["cran", "devtools"],
            build_systems=[],
            test_frameworks=["testthat"],
            language_server="languageserver",
            language_server_name="R Language Server",
            tree_sitter_language="r",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=False,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern(
                    "function",
                    "(binary_operator lhs: (identifier) @name rhs: (function_definition))",
                ),
            ],
            calls="""
                (call function: (identifier) @callee)
            """,
            references="""
                (identifier) @name
            """,
            enclosing_scopes=[],
        )


class MarkdownPlugin(BaseLanguagePlugin):
    """Markdown documentation plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="markdown",
            display_name="Markdown",
            aliases=["md"],
            extensions=[".md", ".markdown", ".mdown"],
            filenames=["README.md", "CHANGELOG.md", "CONTRIBUTING.md"],
            shebangs=[],
            comment_style=CommentStyle.HTML,
            line_comment="",
            block_comment_start="<!--",
            block_comment_end="-->",
            string_delimiters=[],
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=[],
            language_server="marksman",
            language_server_name="Marksman",
            tree_sitter_language="markdown",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
            supports_control_flow_graph=False,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        # Markdown headings (#, ##, ###...) surface as `function` symbols so
        # code-intelligence can answer "where is the ## API section?" — the
        # @def capture is attached so end_line tracks the heading's span.
        # Fenced code blocks surface as `class` symbols keyed on the info
        # string's language (python, rust, etc.), making "find all python
        # samples in the docs" possible.
        return TreeSitterQueries(
            symbols=[
                QueryPattern(
                    "function",
                    "(atx_heading (atx_h1_marker) (inline) @name) @def",
                ),
                QueryPattern(
                    "function",
                    "(atx_heading (atx_h2_marker) (inline) @name) @def",
                ),
                QueryPattern(
                    "function",
                    "(atx_heading (atx_h3_marker) (inline) @name) @def",
                ),
                QueryPattern(
                    "function",
                    "(atx_heading (atx_h4_marker) (inline) @name) @def",
                ),
                QueryPattern(
                    "function",
                    "(atx_heading (atx_h5_marker) (inline) @name) @def",
                ),
                QueryPattern(
                    "function",
                    "(atx_heading (atx_h6_marker) (inline) @name) @def",
                ),
                # Setext-style headings (text underlined with === or ---).
                QueryPattern(
                    "function",
                    "(setext_heading (paragraph) @name) @def",
                ),
                # Fenced code blocks keyed on their info-string language.
                QueryPattern(
                    "class",
                    "(fenced_code_block (info_string (language) @name)) @def",
                ),
            ],
            references="""
                (link_destination) @name
            """,
            enclosing_scopes=[],
        )


class XmlPlugin(BaseLanguagePlugin):
    """XML language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="xml",
            display_name="XML",
            aliases=["xsl", "xslt"],
            extensions=[".xml", ".xsl", ".xslt", ".xsd", ".wsdl", ".svg"],
            filenames=["pom.xml", "web.xml", "AndroidManifest.xml"],
            shebangs=[],
            comment_style=CommentStyle.HTML,
            line_comment="",
            block_comment_start="<!--",
            block_comment_end="-->",
            string_delimiters=['"', "'"],
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=[],
            language_server="lemminx",
            language_server_name="LemMinX",
            tree_sitter_language=None,
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=False,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
            supports_control_flow_graph=False,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        return TreeSitterQueries()


# ============================================================================
# Tier-2 coding languages: zig, julia, ocaml, solidity, perl, objc
#
# Each plugin ships a minimal query pack — symbols + calls — sufficient for
# graph indexing and code-intelligence symbol lookup. Inheritance /
# implements / composition are added only where the grammar exposes a
# straightforward shape (e.g. ObjC class_interface, Solidity inheritance
# specifiers). For languages that don't have those concepts at all
# (Zig structs, Julia, OCaml functional code), the patterns are omitted.
# ============================================================================


class ZigPlugin(BaseLanguagePlugin):
    """Zig systems programming language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="zig",
            display_name="Zig",
            aliases=["zig"],
            extensions=[".zig"],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            indent_size=4,
            use_tabs=False,
            package_managers=["zig"],
            build_systems=["zig build"],
            test_frameworks=["zig test"],
            language_server="zls",
            language_server_name="Zig Language Server",
            tree_sitter_language="zig",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=True,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=True,
            supports_test_execution=True,
            supports_coverage=False,
            supports_debugging=True,
            supports_breakpoints=True,
            supports_step_debugging=True,
            supports_formatting=True,
            supports_linting=False,
            supports_completion=True,
            supports_control_flow_graph=True,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        # Zig has no classes/inheritance — only structs declared as
        # const-bound types. The class pattern captures `const X = struct {...}`.
        return TreeSitterQueries(
            symbols=[
                QueryPattern("function", "(function_declaration (identifier) @name)"),
                QueryPattern(
                    "class",
                    "(variable_declaration (identifier) @name (struct_declaration))",
                ),
            ],
            calls="""
                (call_expression (identifier) @callee)
            """,
            references="""
                (identifier) @name
            """,
            enclosing_scopes=[
                ("function_declaration", "name"),
            ],
        )


class JuliaPlugin(BaseLanguagePlugin):
    """Julia scientific computing language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="julia",
            display_name="Julia",
            aliases=["jl"],
            extensions=[".jl"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            block_comment_start="#=",
            block_comment_end="=#",
            indent_size=4,
            use_tabs=False,
            package_managers=["Pkg"],
            build_systems=[],
            test_frameworks=["Test"],
            language_server="LanguageServer.jl",
            language_server_name="Julia Language Server",
            tree_sitter_language="julia",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=True,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
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
        # Julia function signature is nested inside a call_expression that
        # itself sits under `signature`. struct_definition wraps its name
        # in `type_head`.
        return TreeSitterQueries(
            symbols=[
                QueryPattern(
                    "function",
                    "(function_definition (signature (call_expression (identifier) @name)))",
                ),
                QueryPattern(
                    "class",
                    "(struct_definition (type_head (identifier) @name))",
                ),
            ],
            calls="""
                (call_expression (identifier) @callee)
            """,
            references="""
                (identifier) @name
            """,
            enclosing_scopes=[
                ("function_definition", "name"),
            ],
        )


class OcamlPlugin(BaseLanguagePlugin):
    """OCaml functional programming language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="ocaml",
            display_name="OCaml",
            aliases=["ml"],
            extensions=[".ml", ".mli"],
            comment_style=CommentStyle.C_STYLE,
            line_comment=None,
            block_comment_start="(*",
            block_comment_end="*)",
            indent_size=2,
            use_tabs=False,
            package_managers=["opam", "dune"],
            build_systems=["dune", "ocamlbuild"],
            test_frameworks=["alcotest", "ounit"],
            language_server="ocamllsp",
            language_server_name="OCaml Language Server",
            tree_sitter_language="ocaml",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=True,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=True,
            supports_organize_imports=False,
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
        # OCaml `let foo x = ...` is a value_definition wrapping a let_binding.
        # Modules surface through module_definition. Function application is
        # `(application_expression (value_path ...))` — the value_path's
        # leaf value_name is the callee.
        return TreeSitterQueries(
            symbols=[
                QueryPattern(
                    "function",
                    "(value_definition (let_binding (value_name) @name))",
                ),
                QueryPattern(
                    "class",
                    "(module_definition (module_binding (module_name) @name))",
                ),
            ],
            calls="""
                (application_expression (value_path (value_name) @callee))
            """,
            references="""
                (value_name) @name
            """,
            enclosing_scopes=[
                ("let_binding", "value_name"),
                ("module_binding", "module_name"),
            ],
        )


class SolidityPlugin(BaseLanguagePlugin):
    """Solidity smart-contract language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="solidity",
            display_name="Solidity",
            aliases=["sol"],
            extensions=[".sol"],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            indent_size=4,
            use_tabs=False,
            package_managers=["npm", "yarn"],
            build_systems=["hardhat", "foundry", "truffle"],
            test_frameworks=["hardhat", "foundry"],
            language_server="solidity-ls",
            language_server_name="Solidity Language Server",
            tree_sitter_language="solidity",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=True,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=False,
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(contract_declaration (identifier) @name)"),
                QueryPattern("class", "(interface_declaration (identifier) @name)"),
                QueryPattern("class", "(library_declaration (identifier) @name)"),
                QueryPattern("function", "(function_definition (identifier) @name)"),
                QueryPattern("function", "(modifier_definition (identifier) @name)"),
            ],
            calls="""
                (call_expression (expression (identifier) @callee))
            """,
            references="""
                (identifier) @name
            """,
            enclosing_scopes=[
                ("function_definition", "name"),
                ("contract_declaration", "name"),
            ],
        )


class PerlPlugin(BaseLanguagePlugin):
    """Perl scripting language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="perl",
            display_name="Perl",
            aliases=["pl"],
            extensions=[".pl", ".pm", ".t"],
            shebangs=["perl"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            indent_size=4,
            use_tabs=False,
            package_managers=["cpan", "cpanm"],
            build_systems=["make"],
            test_frameworks=["Test::More", "prove"],
            language_server="perlnavigator",
            language_server_name="Perl Navigator",
            tree_sitter_language="perl",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern(
                    "function",
                    "(subroutine_declaration_statement (bareword) @name)",
                ),
                QueryPattern("class", "(package_statement (package) @name)"),
            ],
            references="""
                (bareword) @name
            """,
            enclosing_scopes=[
                ("subroutine_declaration_statement", "bareword"),
                ("package_statement", "package"),
            ],
        )


class ObjcPlugin(BaseLanguagePlugin):
    """Objective-C language plugin (iOS / macOS native)."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="objc",
            display_name="Objective-C",
            aliases=["objective-c", "objectivec", "m"],
            extensions=[".m", ".mm", ".h"],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            indent_size=4,
            use_tabs=False,
            package_managers=["cocoapods", "carthage"],
            build_systems=["xcodebuild", "make"],
            test_frameworks=["XCTest"],
            language_server="clangd",
            language_server_name="clangd",
            tree_sitter_language="objc",
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
        # ObjC class_interface has children [identifier (class name), identifier (superclass)].
        # The `.` anchor on the symbols query captures only the first child (class name).
        # The inheritance pattern captures both.
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_interface (identifier) @name . )"),
                QueryPattern("class", "(class_implementation (identifier) @name)"),
                QueryPattern("class", "(protocol_declaration (identifier) @name)"),
                QueryPattern(
                    "function",
                    "(method_definition (method_type) (identifier) @name)",
                ),
            ],
            calls="""
                (message_expression . (identifier) (identifier) @callee)
            """,
            references="""
                (identifier) @name
            """,
            inheritance="""
                (class_interface
                    (identifier) @child
                    .
                    (identifier) @base)
            """,
            enclosing_scopes=[
                ("class_interface", "name"),
                ("class_implementation", "name"),
                ("method_definition", "name"),
            ],
        )


# ============================================================================
# Tier-3 build / scripting / schema languages: make, cmake, graphql, groovy, hcl
#
# These aren't traditional "coding" languages but they appear extensively in
# real projects (Makefiles, CMakeLists.txt, GraphQL schemas, Gradle/Jenkins
# Groovy scripts, Terraform/HCL configs) and benefit from symbol extraction
# so code-intelligence / graph indexing can find their definitions.
# ============================================================================


class MakePlugin(BaseLanguagePlugin):
    """Makefile language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="make",
            display_name="Make",
            aliases=["makefile"],
            extensions=[".mk", ".make"],
            filenames=["Makefile", "makefile", "GNUmakefile"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            indent_size=4,
            use_tabs=True,  # Makefiles REQUIRE tabs for recipe indentation
            package_managers=[],
            build_systems=["make", "gmake"],
            test_frameworks=[],
            language_server=None,
            language_server_name=None,
            tree_sitter_language="make",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=True,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=False,
            supports_linting=True,
            supports_completion=False,
            supports_control_flow_graph=False,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        # Makefile targets surface as `function` symbols so code-intelligence
        # tools can find them by name (the most common use: "where is the
        # `test` target defined?").
        return TreeSitterQueries(
            symbols=[
                QueryPattern("function", "(rule (targets (word) @name))"),
            ],
            references="""
                (word) @name
            """,
            enclosing_scopes=[],
        )


class CmakePlugin(BaseLanguagePlugin):
    """CMake build language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="cmake",
            display_name="CMake",
            aliases=["cmake"],
            extensions=[".cmake"],
            filenames=["CMakeLists.txt"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=["cmake"],
            test_frameworks=["ctest"],
            language_server="neocmakelsp",
            language_server_name="CMake Language Server",
            tree_sitter_language="cmake",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=True,
            supports_test_execution=True,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
            supports_control_flow_graph=False,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        # CMake function/macro definitions wrap arguments under `argument_list`.
        # The leading anchor `.` skips the `function`/`macro` keyword node so
        # the first `argument` (which holds the name) is the one captured.
        return TreeSitterQueries(
            symbols=[
                QueryPattern(
                    "function",
                    "(function_def (function_command . (function) (argument_list "
                    "(argument (unquoted_argument) @name))))",
                ),
                QueryPattern(
                    "function",
                    "(macro_def (macro_command . (macro) (argument_list "
                    "(argument (unquoted_argument) @name))))",
                ),
            ],
            calls="""
                (normal_command (identifier) @callee)
            """,
            references="""
                (identifier) @name
            """,
            enclosing_scopes=[],
        )


class GraphqlPlugin(BaseLanguagePlugin):
    """GraphQL schema language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="graphql",
            display_name="GraphQL",
            aliases=["gql"],
            extensions=[".graphql", ".gql"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            indent_size=2,
            use_tabs=False,
            package_managers=["npm", "yarn"],
            build_systems=["graphql-codegen"],
            test_frameworks=[],
            language_server="graphql-language-service",
            language_server_name="GraphQL Language Service",
            tree_sitter_language="graphql",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=True,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
            supports_control_flow_graph=False,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        # GraphQL "types" cover Object/Interface/Union/Enum/Scalar/Input.
        # Fields inside types are surfaced as `function` symbols so symbol
        # lookup can find Query.hello, Mutation.setName, etc.
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(object_type_definition (name) @name)"),
                QueryPattern("class", "(interface_type_definition (name) @name)"),
                QueryPattern("class", "(union_type_definition (name) @name)"),
                QueryPattern("class", "(scalar_type_definition (name) @name)"),
                QueryPattern("class", "(enum_type_definition (name) @name)"),
                QueryPattern("class", "(input_object_type_definition (name) @name)"),
                QueryPattern("function", "(field_definition (name) @name)"),
            ],
            references="""
                (name) @name
            """,
            enclosing_scopes=[
                ("object_type_definition", "name"),
                ("interface_type_definition", "name"),
            ],
        )


class GroovyPlugin(BaseLanguagePlugin):
    """Groovy / Gradle script language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="groovy",
            display_name="Groovy",
            aliases=["gradle"],
            extensions=[".groovy", ".gradle", ".gvy", ".gy", ".gsh"],
            filenames=["Jenkinsfile", "build.gradle"],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            indent_size=4,
            use_tabs=False,
            package_managers=["grape"],
            build_systems=["gradle"],
            test_frameworks=["spock", "junit"],
            language_server="groovy-language-server",
            language_server_name="Groovy Language Server",
            tree_sitter_language="groovy",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=False,
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_declaration (identifier) @name)"),
                QueryPattern("function", "(method_declaration (identifier) @name)"),
            ],
            references="""
                (identifier) @name
            """,
            inheritance="""
                (class_declaration
                    (identifier) @child
                    (superclass (type_identifier) @base))
            """,
            enclosing_scopes=[
                ("class_declaration", "name"),
                ("method_declaration", "name"),
            ],
        )


class HclPlugin(BaseLanguagePlugin):
    """HashiCorp Configuration Language plugin (Terraform, Vault, Nomad, etc.)."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="hcl",
            display_name="HCL",
            aliases=["terraform", "tf"],
            extensions=[".hcl", ".tf", ".tfvars"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            block_comment_start="/*",
            block_comment_end="*/",
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=["terraform"],
            test_frameworks=["terratest"],
            language_server="terraform-ls",
            language_server_name="Terraform Language Server",
            tree_sitter_language="hcl",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=True,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
            supports_control_flow_graph=False,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        # HCL/Terraform blocks (resource/variable/module/data/provider) surface
        # as `class` symbols keyed on the block-type identifier. Attribute
        # assignments inside a block surface as `function` symbols.
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(block (identifier) @name)"),
                QueryPattern("function", "(attribute (identifier) @name)"),
            ],
            references="""
                (identifier) @name
            """,
            enclosing_scopes=[
                ("block", "identifier"),
            ],
        )


# ============================================================================
# Tier-4 hardware-description / shader languages: vhdl, verilog, glsl
#
# HDLs have real control flow inside `process`/`always` blocks (if/case/for/
# while/loop) and `generate` constructs. Shader code (GLSL) is C-like and
# uses the standard if/for/while/switch node names. CCG is meaningful for
# all three.
# ============================================================================


class VhdlPlugin(BaseLanguagePlugin):
    """VHDL hardware description language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="vhdl",
            display_name="VHDL",
            aliases=["vhd"],
            extensions=[".vhd", ".vhdl"],
            comment_style=CommentStyle.DOUBLE_DASH,
            line_comment="--",
            indent_size=4,
            use_tabs=False,
            package_managers=[],
            build_systems=["ghdl", "vivado", "questa", "modelsim"],
            test_frameworks=["vunit"],
            language_server="vhdl-tool",
            language_server_name="VHDL Tool",
            tree_sitter_language="vhdl",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=True,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
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
        # VHDL design units surface as `class` symbols (entity / architecture
        # / package — each is a top-level scope holder). Subprograms wrap
        # their function name inside `function_specification`.
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(entity_declaration (identifier) @name)"),
                QueryPattern("class", "(architecture_definition (identifier) @name)"),
                QueryPattern("class", "(package_declaration (identifier) @name)"),
                QueryPattern(
                    "function",
                    "(subprogram_declaration (function_specification (identifier) @name))",
                ),
            ],
            references="""
                (identifier) @name
            """,
            enclosing_scopes=[
                ("entity_declaration", "identifier"),
                ("architecture_definition", "identifier"),
                ("package_declaration", "identifier"),
            ],
        )


class VerilogPlugin(BaseLanguagePlugin):
    """Verilog / SystemVerilog hardware description language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="verilog",
            display_name="Verilog",
            aliases=["systemverilog", "sv"],
            extensions=[".v", ".vh", ".sv", ".svh"],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            indent_size=4,
            use_tabs=False,
            package_managers=[],
            build_systems=["verilator", "iverilog", "vivado", "questa"],
            test_frameworks=["cocotb", "uvm"],
            language_server="svls",
            language_server_name="SystemVerilog Language Server",
            tree_sitter_language="verilog",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=True,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
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
        # Modules surface as `class` symbols; functions and tasks both surface
        # as `function` (Verilog `task` is a procedure that may have side
        # effects but isn't structurally different from a function for the
        # symbol graph). Note: function_identifier / task_identifier are
        # text-bearing nodes — the `simple_identifier` child is empty here.
        return TreeSitterQueries(
            symbols=[
                QueryPattern(
                    "class",
                    "(module_declaration (module_header (simple_identifier) @name))",
                ),
                QueryPattern(
                    "function",
                    "(function_declaration (function_body_declaration "
                    "(function_identifier) @name))",
                ),
                QueryPattern(
                    "function",
                    "(task_declaration (task_body_declaration (task_identifier) @name))",
                ),
            ],
            references="""
                (simple_identifier) @name
            """,
            enclosing_scopes=[
                ("module_declaration", "simple_identifier"),
                ("function_declaration", "function_identifier"),
                ("task_declaration", "task_identifier"),
            ],
        )


class GlslPlugin(BaseLanguagePlugin):
    """GLSL shader language plugin (OpenGL / WebGL / Vulkan shaders)."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="glsl",
            display_name="GLSL",
            aliases=["shader"],
            extensions=[
                ".glsl",
                ".vert",
                ".frag",
                ".vs",
                ".fs",
                ".gs",
                ".tcs",
                ".tes",
                ".comp",
            ],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            indent_size=4,
            use_tabs=False,
            package_managers=[],
            build_systems=["glslang", "glslc"],
            test_frameworks=[],
            language_server="glsl_analyzer",
            language_server_name="GLSL Analyzer",
            tree_sitter_language="glsl",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=True,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
            supports_control_flow_graph=True,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        # GLSL is C-like — function definitions wrap the name in
        # function_declarator. Calls are standard C call_expression.
        return TreeSitterQueries(
            symbols=[
                QueryPattern(
                    "function",
                    "(function_definition declarator: "
                    "(function_declarator declarator: (identifier) @name))",
                ),
            ],
            calls="""
                (call_expression function: (identifier) @callee)
            """,
            references="""
                (identifier) @name
            """,
            enclosing_scopes=[
                ("function_definition", "declarator"),
            ],
        )
