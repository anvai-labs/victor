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

"""Shared base for the ECMAScript-family (JavaScript/TypeScript) plugins.

Holds the logic both plugins share verbatim -- capabilities, package.json test
runner detection, prettier formatting, call-node traversal and callee/member
name extraction. The genuine differences are expressed as class-attribute
hooks that ``JavaScriptPlugin`` and ``TypeScriptPlugin`` override.
"""

from pathlib import Path
from typing import ClassVar, List, Optional, Tuple, TYPE_CHECKING

from victor_coding.languages.base import (
    BaseLanguagePlugin,
    ConfigurableASTTraverser,
    Formatter,
    LanguageCapabilities,
    TestRunner,
    TraversalConfig,
)

if TYPE_CHECKING:
    from tree_sitter import Node


class EcmaScriptPluginBase(BaseLanguagePlugin):
    """Common behaviour of the JavaScript and TypeScript plugins.

    Subclasses provide ``_create_config``/``_create_tree_sitter_queries`` and
    may override the class-attribute hooks below.
    """

    # --- Hooks --------------------------------------------------------------
    _supports_type_checking: ClassVar[bool] = False
    _test_file_pattern: ClassVar[str] = "*.test.js"
    _vitest_parallel_args: ClassVar[Tuple[str, ...]] = ()
    _jest_parallel_args: ClassVar[Tuple[str, ...]] = ()
    _detects_mocha: ClassVar[bool] = False
    # Node types whose name is the first property_identifier/identifier child.
    _method_node_types: ClassVar[Tuple[str, ...]] = ("method_definition",)
    _extra_function_types: ClassVar[Tuple[str, ...]] = ()
    _class_types: ClassVar[Tuple[str, ...]] = ("class_declaration",)
    _scope_body_types: ClassVar[Tuple[str, ...]] = ("class_body",)

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=self._supports_type_checking,
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

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Detect vitest, jest (or mocha, when enabled) from package.json; default jest."""
        package_json = project_root / "package.json"
        pattern = self._test_file_pattern

        if package_json.exists():
            content = package_json.read_text()

            if '"vitest"' in content:
                return TestRunner(
                    name="vitest",
                    command=["npx", "vitest"],
                    file_pattern=pattern,
                    run_args=["run"],
                    coverage_args=["--coverage"],
                    parallel_args=list(self._vitest_parallel_args),
                    output_format="json",
                    json_args=["--reporter=json"],
                )

            if '"jest"' in content:
                return TestRunner(
                    name="jest",
                    command=["npx", "jest"],
                    file_pattern=pattern,
                    coverage_args=["--coverage"],
                    parallel_args=list(self._jest_parallel_args),
                    output_format="json",
                    json_args=["--json"],
                )

            if self._detects_mocha and '"mocha"' in content:
                return TestRunner(
                    name="mocha",
                    command=["npx", "mocha"],
                    file_pattern=pattern,
                    run_args=["--recursive"],
                )

        return TestRunner(
            name="jest",
            command=["npx", "jest"],
            file_pattern=pattern,
        )

    def get_formatter(self, project_root: Path) -> Optional[Formatter]:
        """Get prettier formatter."""
        return Formatter(
            name="prettier",
            command=["npx", "prettier", "--write", "."],
            check_args=["--check"],
            config_file=".prettierrc",
        )

    def _find_call_nodes(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context."""
        config = TraversalConfig(
            function_types=[
                "function_declaration",
                "function_expression",
                "method_definition",
                "arrow_function",
                *self._extra_function_types,
            ],
            class_types=list(self._class_types),
            call_types=["call_expression", "new_expression"],
            name_field="identifier",
            scope_body_types=list(self._scope_body_types),
        )
        traverser = ConfigurableASTTraverser(config, self._get_node_text)
        return traverser.find_call_nodes(root)

    def _extract_function_name(self, node: "Node") -> Optional[str]:
        """Extract function name from a function node."""
        if node.type == "function_declaration":
            for child in node.children:
                if child.type == "identifier":
                    return self._get_node_text(child)

        if node.type in self._method_node_types:
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
