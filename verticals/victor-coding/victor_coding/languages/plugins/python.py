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

"""Python language plugin."""

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


class PythonPlugin(BaseLanguagePlugin):
    """Python language plugin.

    Supports:
    - Testing: pytest, unittest
    - Formatting: black, autopep8
    - Linting: ruff, flake8, pylint
    - Type checking: mypy, pyright
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="python",
            display_name="Python",
            aliases=["py", "python3", "python2"],
            extensions=[".py", ".pyw", ".pyi", ".pyx"],
            filenames=["Pipfile", "pyproject.toml", "setup.py", "requirements.txt"],
            shebangs=["python", "python3", "python2"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            block_comment_start='"""',
            block_comment_end='"""',
            string_delimiters=['"', "'", '"""', "'''"],
            indent_size=4,
            use_tabs=False,
            package_managers=["pip", "pipenv", "poetry", "uv"],
            build_systems=["setuptools", "poetry", "flit"],
            test_frameworks=["pytest", "unittest", "nose2"],
            language_server="pylsp",
            language_server_name="Python Language Server",
            tree_sitter_language="python",
            doc_comment_pattern=DocCommentPattern(location="inside"),
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
        """Create tree-sitter queries for Python symbol/call extraction.

        The @def capture is used for end_line (function body boundaries).
        The @name capture is used for the symbol name.
        """
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_definition name: (identifier) @name) @def"),
                QueryPattern("function", "(function_definition name: (identifier) @name) @def"),
            ],
            calls="""
                (call function: (identifier) @callee)
                (call function: (attribute attribute: (identifier) @callee))
            """,
            references="""
                (call function: (identifier) @name)
                (call function: (attribute attribute: (identifier) @name))
                (attribute object: (_) attribute: (identifier) @name)
                (identifier) @name
            """,
            inheritance="""
                (class_definition
                    name: (identifier) @child
                    superclasses: (argument_list (identifier) @base))
            """,
            composition="""
                (class_definition
                    name: (identifier) @owner
                    body: (block
                        (function_definition
                            name: (identifier) @_init (#eq? @_init "__init__")
                            body: (block
                                (expression_statement
                                    (assignment
                                        left: (attribute object: (identifier) @_self (#eq? @_self "self"))
                                        right: (call function: (identifier) @type)))))))
            """,
            enclosing_scopes=[
                ("function_definition", "name"),
                ("class_definition", "name"),
            ],
        )

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Get pytest or unittest runner."""
        # Check for pytest
        pyproject = project_root / "pyproject.toml"
        setup_cfg = project_root / "setup.cfg"
        pytest_ini = project_root / "pytest.ini"

        if pyproject.exists() or setup_cfg.exists() or pytest_ini.exists():
            return TestRunner(
                name="pytest",
                command=["pytest"],
                file_pattern="test_*.py",
                discover_args=["--collect-only", "-q"],
                run_args=["-v"],
                coverage_args=["--cov", "--cov-report=term-missing"],
                parallel_args=["-n", "auto"],
                output_format="json",
                json_args=["--json-report", "--json-report-file=-"],
            )

        # Fall back to unittest
        return TestRunner(
            name="unittest",
            command=["python", "-m", "unittest"],
            file_pattern="test_*.py",
            discover_args=["discover", "-v"],
            run_args=["-v"],
        )

    def get_formatter(self, project_root: Path) -> Optional[Formatter]:
        """Get black or autopep8 formatter."""
        pyproject = project_root / "pyproject.toml"

        # Check for black config
        if pyproject.exists():
            content = pyproject.read_text()
            if "[tool.black]" in content:
                return Formatter(
                    name="black",
                    command=["black", "."],
                    check_args=["--check", "--diff"],
                    config_file="pyproject.toml",
                )

        # Default to black
        return Formatter(
            name="black",
            command=["black", "."],
            check_args=["--check", "--diff"],
        )

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Get ruff, flake8, or pylint linter."""
        pyproject = project_root / "pyproject.toml"

        # Check for ruff config
        if pyproject.exists():
            content = pyproject.read_text()
            if "[tool.ruff]" in content:
                return Linter(
                    name="ruff",
                    command=["ruff", "check", "."],
                    fix_args=["--fix"],
                    config_file="pyproject.toml",
                    output_format="json",
                )

        # Check for flake8
        flake8_cfg = project_root / ".flake8"
        if flake8_cfg.exists():
            return Linter(
                name="flake8",
                command=["flake8", "."],
                config_file=".flake8",
            )

        # Default to ruff
        return Linter(
            name="ruff",
            command=["ruff", "check", "."],
            fix_args=["--fix"],
        )

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Get pip or poetry build system."""
        pyproject = project_root / "pyproject.toml"

        if pyproject.exists():
            content = pyproject.read_text()
            if "[tool.poetry]" in content:
                return BuildSystem(
                    name="poetry",
                    build_command=["poetry", "build"],
                    run_command=["poetry", "run", "python"],
                    install_command=["poetry", "install"],
                    manifest_file="pyproject.toml",
                )

        # Default pip
        return BuildSystem(
            name="pip",
            build_command=["python", "-m", "build"],
            run_command=["python"],
            install_command=["pip", "install", "-e", "."],
            manifest_file="pyproject.toml",
        )

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in Python source code.

        Finds function calls within the same file by analyzing:
        - Call nodes (call)
        - Enclosing function context (function_definition)
        - Method calls (attribute expressions)

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []

        # Find enclosing function for each call
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
                "language": "python",
                "file": str(file_path),
            },
        )

    def _find_call_nodes(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Uses the shared ConfigurableASTTraverser to eliminate code duplication.

        Args:
            root: Tree-sitter root node

        Returns:
            List of (call_node, caller_name, caller_line) tuples
        """
        config = TraversalConfig(
            function_types=["function_definition", "async_function_definition"],
            call_types=["call"],
            name_field="identifier",
        )
        traverser = ConfigurableASTTraverser(config, self._get_node_text)
        return traverser.find_call_nodes(root)

    def _extract_callee_name(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called function from a call node.

        Handles:
        - Simple calls: foo()
        - Attribute calls: obj.method()
        - Chained calls: obj.method1().method2()

        Args:
            call_node: Tree-sitter call node

        Returns:
            Callee name or None
        """
        # Get the function part of the call
        for child in call_node.children:
            if child.type == "attribute":
                # obj.method() -> extract "method"
                return self._extract_attribute_name(child)
            elif child.type == "identifier":
                # foo() -> extract "foo"
                return self._get_node_text(child)

        return None

    def _extract_attribute_name(self, attr_node: "Node") -> Optional[str]:
        """Extract attribute name from an attribute node.

        For obj.method, extracts "method".
        For obj.method1.method2, extracts "method2" (the final attribute).

        Args:
            attr_node: Tree-sitter attribute node

        Returns:
            Attribute name or None
        """
        # Attribute node structure: (object (attribute) attribute: (identifier))
        # We want the last identifier (the method being called)
        for child in reversed(attr_node.children):
            if child.type == "identifier":
                return self._get_node_text(child)
            elif child.type == "attribute":
                # Recurse for chained attributes
                result = self._extract_attribute_name(child)
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
