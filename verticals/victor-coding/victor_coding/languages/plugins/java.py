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

"""Java language plugin."""

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


class JavaPlugin(BaseLanguagePlugin):
    """Java language plugin.

    Supports:
    - Testing: JUnit, Maven Surefire, Gradle Test
    - Formatting: google-java-format
    - Linting: checkstyle, SpotBugs
    - Building: Maven, Gradle
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="java",
            display_name="Java",
            aliases=["jav"],
            extensions=[".java"],
            filenames=["pom.xml", "build.gradle", "build.gradle.kts"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"'],
            indent_size=4,
            use_tabs=False,
            package_managers=["maven", "gradle"],
            build_systems=["maven", "gradle"],
            test_frameworks=["junit", "testng"],
            language_server="jdtls",
            language_server_name="Eclipse JDT Language Server",
            tree_sitter_language="java",
            doc_comment_pattern=DocCommentPattern(
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
        """Create tree-sitter queries for Java symbol/call extraction.

        The @def capture is used for end_line (function body boundaries).
        The @name capture is used for the symbol name.
        """
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_declaration name: (identifier) @name) @def"),
                QueryPattern("class", "(interface_declaration name: (identifier) @name) @def"),
                QueryPattern("function", "(method_declaration name: (identifier) @name) @def"),
            ],
            calls="""
                (method_invocation name: (identifier) @callee)
                (object_creation_expression type: (type_identifier) @callee)
            """,
            references="""
                (method_invocation name: (identifier) @name)
                (method_invocation object: (identifier) @name)
                (field_access field: (identifier) @name)
            """,
            inheritance="""
                (class_declaration
                    name: (identifier) @child
                    superclass: (superclass (type_identifier) @base))
            """,
            implements="""
                (class_declaration
                    name: (identifier) @child
                    interfaces: (super_interfaces (type_list (type_identifier) @interface)))
                (interface_declaration
                    name: (identifier) @child
                    (extends_interfaces (type_list (type_identifier) @interface)))
            """,
            composition="""
                (class_declaration
                    name: (identifier) @owner
                    body: (class_body
                        (field_declaration
                            type: (type_identifier) @type)))
            """,
            enclosing_scopes=[
                ("method_declaration", "name"),
                ("class_declaration", "name"),
                ("interface_declaration", "name"),
            ],
        )

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Get Maven or Gradle test runner."""
        # Check for Gradle
        if (project_root / "build.gradle").exists() or (project_root / "build.gradle.kts").exists():
            return TestRunner(
                name="gradle test",
                command=["./gradlew", "test"],
                file_pattern="*Test.java",
                run_args=["--info"],
                coverage_args=["jacocoTestReport"],
                parallel_args=["--parallel"],
            )

        # Check for Maven
        if (project_root / "pom.xml").exists():
            return TestRunner(
                name="maven test",
                command=["mvn", "test"],
                file_pattern="*Test.java",
                run_args=["-Dtest=*"],
                coverage_args=["jacoco:report"],
            )

        return None

    def get_formatter(self, project_root: Path) -> Optional[Formatter]:
        """Get google-java-format formatter."""
        return Formatter(
            name="google-java-format",
            command=["google-java-format", "--replace", "src/**/*.java"],
            check_args=["--dry-run", "--set-exit-if-changed"],
        )

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Get checkstyle linter."""
        # Check for Gradle
        if (project_root / "build.gradle").exists():
            return Linter(
                name="checkstyle",
                command=["./gradlew", "checkstyleMain"],
                config_file="config/checkstyle/checkstyle.xml",
            )

        # Check for Maven
        if (project_root / "pom.xml").exists():
            return Linter(
                name="checkstyle",
                command=["mvn", "checkstyle:check"],
                config_file="checkstyle.xml",
            )

        return None

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Get Maven or Gradle build system."""
        # Check for Gradle
        if (project_root / "build.gradle").exists() or (project_root / "build.gradle.kts").exists():
            return BuildSystem(
                name="gradle",
                build_command=["./gradlew", "build"],
                run_command=["./gradlew", "run"],
                clean_command=["./gradlew", "clean"],
                install_command=["./gradlew", "install"],
                manifest_file="build.gradle",
            )

        # Check for Maven
        if (project_root / "pom.xml").exists():
            return BuildSystem(
                name="maven",
                build_command=["mvn", "package"],
                run_command=["mvn", "exec:java"],
                clean_command=["mvn", "clean"],
                install_command=["mvn", "install"],
                manifest_file="pom.xml",
            )

        return None

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in Java source code.

        Finds method calls, constructor calls, and static method calls.

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
                "language": "java",
                "file": str(file_path),
            },
        )

    def _find_call_nodes(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing method context.

        Uses the shared ConfigurableASTTraverser to eliminate code duplication.
        """
        config = TraversalConfig(
            function_types=["method_declaration", "constructor_declaration"],
            class_types=["class_declaration", "interface_declaration"],
            call_types=["method_invocation", "object_creation_expression"],
            name_field="identifier",
            scope_body_types=["class_body", "interface_body"],
        )
        traverser = ConfigurableASTTraverser(config, self._get_node_text)
        return traverser.find_call_nodes(root)

    def _extract_callee_name(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called method from a method invocation node.

        Handles:
        - Simple calls: foo()
        - Field access calls: obj.method()
        - Static calls: Class.method()
        - Chained calls: obj.method1().method2()
        """
        # For object creation expressions, extract the class name
        if call_node.type == "object_creation_expression":
            for child in call_node.children:
                if child.type == "identifier":
                    # Simple: new Foo()
                    return self._get_node_text(child)
                elif child.type == "type_identifier":
                    # Qualified: new com.example.Foo()
                    return self._get_qualified_name(child)

        # For method invocations, Java uses: identifier . identifier (argument_list)
        # We need to find the last identifier before the argument_list
        last_identifier = None
        for child in call_node.children:
            if child.type == "identifier":
                last_identifier = self._get_node_text(child)
            elif child.type == "argument_list":
                # We've reached the arguments, the last identifier is the method name
                if last_identifier:
                    return last_identifier

        # Fallback: try to find any identifier
        for child in call_node.children:
            if child.type == "identifier":
                return self._get_node_text(child)

        return None

    def _extract_field_name(self, field_node: "Node") -> Optional[str]:
        """Extract field name from a field_access node.

        For obj.method, extracts "method".
        For obj.field1.field2, extracts "field2" (the final field).
        """
        for child in reversed(field_node.children):
            if child.type == "identifier":
                return self._get_node_text(child)
            elif child.type == "field_access":
                result = self._extract_field_name(child)
                if result:
                    return result

        return None

    def _get_qualified_name(self, node: "Node") -> Optional[str]:
        """Extract the final identifier from a qualified name.

        For com.example.Foo, extracts "Foo".
        """
        parts: List[str] = []

        def collect(n: "Node") -> None:
            if n.type == "identifier":
                parts.append(self._get_node_text(n) or "")
            elif n.type == "type_identifier":
                parts.append(self._get_node_text(n) or "")
            for child in n.children:
                collect(child)

        collect(node)

        if parts:
            return parts[-1]

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node."""
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text
