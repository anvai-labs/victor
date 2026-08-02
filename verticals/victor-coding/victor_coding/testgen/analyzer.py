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

"""Code analyzer for test generation.

Analyzes source code to extract function signatures, type hints,
and documentation for generating appropriate test cases.
"""

import ast
import logging
import re
from pathlib import Path
from typing import Optional

from victor_coding.testgen.protocol import ClassSignature, FunctionSignature

logger = logging.getLogger(__name__)


class TestTargetAnalyzer:
    """Analyzes code to extract testable targets."""

    def analyze_file(self, file_path: Path) -> tuple[list[FunctionSignature], list[ClassSignature]]:
        """Analyze a Python file for testable targets.

        Args:
            file_path: Path to Python file

        Returns:
            Tuple of (functions, classes)
        """
        try:
            source = file_path.read_text()
            return self.analyze_source(source)
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            return [], []

    def analyze_source(self, source: str) -> tuple[list[FunctionSignature], list[ClassSignature]]:
        """Analyze Python source code.

        Args:
            source: Python source code

        Returns:
            Tuple of (functions, classes)
        """
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.warning(f"Syntax error in source: {e}")
            return [], []

        functions = []
        classes = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = self._analyze_function(node)
                if sig and not sig.name.startswith("_"):
                    functions.append(sig)

            elif isinstance(node, ast.ClassDef):
                sig = self._analyze_class(node)
                if sig and not sig.name.startswith("_"):
                    classes.append(sig)

        return functions, classes

    def _analyze_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> Optional[FunctionSignature]:
        """Analyze a function definition."""
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            type_hint = None
            if arg.annotation:
                type_hint = ast.unparse(arg.annotation)
            parameters.append((arg.arg, type_hint))

        # Handle *args and **kwargs
        if node.args.vararg:
            type_hint = None
            if node.args.vararg.annotation:
                type_hint = ast.unparse(node.args.vararg.annotation)
            parameters.append((f"*{node.args.vararg.arg}", type_hint))

        if node.args.kwarg:
            type_hint = None
            if node.args.kwarg.annotation:
                type_hint = ast.unparse(node.args.kwarg.annotation)
            parameters.append((f"**{node.args.kwarg.arg}", type_hint))

        # Extract return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Check decorators
        decorators = []
        is_static = False
        is_classmethod = False

        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
                if decorator.id == "staticmethod":
                    is_static = True
                elif decorator.id == "classmethod":
                    is_classmethod = True
            elif isinstance(decorator, ast.Attribute):
                decorators.append(ast.unparse(decorator))

        return FunctionSignature(
            name=node.name,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_static=is_static,
            is_classmethod=is_classmethod,
            decorators=decorators,
            source_location=(node.lineno, node.end_lineno or node.lineno),
        )

    def _analyze_class(self, node: ast.ClassDef) -> Optional[ClassSignature]:
        """Analyze a class definition."""
        methods = []
        properties = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = self._analyze_function(item)
                if sig:
                    sig.is_method = True
                    # Check if it's a property
                    if "property" in sig.decorators:
                        properties.append(sig.name)
                    else:
                        methods.append(sig)

        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base))

        # Extract docstring
        docstring = ast.get_docstring(node)

        return ClassSignature(
            name=node.name,
            methods=methods,
            properties=properties,
            bases=bases,
            docstring=docstring,
        )

    def extract_docstring_examples(self, docstring: str) -> list[tuple[str, str]]:
        """Extract examples from docstring.

        Args:
            docstring: Function/class docstring

        Returns:
            List of (input, expected_output) tuples
        """
        if not docstring:
            return []

        examples = []

        # Look for doctest-style examples
        # >>> expression
        # result
        pattern = r">>>\s*(.+?)\n\s*(.+?)(?=\n>>>|\n\n|$)"
        matches = re.findall(pattern, docstring, re.MULTILINE)

        for expr, result in matches:
            examples.append((expr.strip(), result.strip()))

        return examples

    def infer_test_values(
        self,
        type_hint: Optional[str],
    ) -> list[tuple[str, any]]:
        """Infer test values from type hints.

        Args:
            type_hint: Type annotation string

        Returns:
            List of (description, value) tuples
        """
        if not type_hint:
            return [("default", None)]

        type_lower = type_hint.lower()
        values = []

        # Basic types
        if "int" in type_lower:
            values = [
                ("zero", 0),
                ("positive", 42),
                ("negative", -1),
                ("large", 10**9),
            ]
        elif "float" in type_lower:
            values = [
                ("zero", 0.0),
                ("positive", 3.14),
                ("negative", -2.5),
                ("small", 0.001),
            ]
        elif "str" in type_lower:
            values = [
                ("empty", ""),
                ("normal", "test"),
                ("spaces", "  spaced  "),
                ("unicode", "héllo 世界"),
            ]
        elif "bool" in type_lower:
            values = [
                ("true", True),
                ("false", False),
            ]
        elif "list" in type_lower or "List" in type_hint:
            values = [
                ("empty", []),
                ("single", [1]),
                ("multiple", [1, 2, 3]),
            ]
        elif "dict" in type_lower or "Dict" in type_hint:
            values = [
                ("empty", {}),
                ("single", {"key": "value"}),
                ("multiple", {"a": 1, "b": 2}),
            ]
        elif "optional" in type_lower or "None" in type_hint:
            values = [
                ("none", None),
            ]
            # Try to infer the inner type
            inner_match = re.search(r"Optional\[(.+)\]", type_hint)
            if inner_match:
                inner_values = self.infer_test_values(inner_match.group(1))
                values.extend(inner_values)
        elif "path" in type_lower:
            values = [
                ("relative", Path("test/file.py")),
                ("absolute", Path("/tmp/test.py")),
            ]
        else:
            values = [("default", None)]

        return values

    def detect_error_conditions(
        self,
        func_sig: FunctionSignature,
    ) -> list[tuple[str, str, type]]:
        """Detect likely error conditions for a function.

        Args:
            func_sig: Function signature

        Returns:
            List of (description, input_expr, expected_exception) tuples
        """
        errors = []

        # Check parameter types for common error cases
        for param_name, type_hint in func_sig.parameters:
            if param_name.startswith("*"):
                continue

            if type_hint:
                type_lower = type_hint.lower()

                # Division by zero
                if "divisor" in param_name.lower() or "denominator" in param_name.lower():
                    errors.append(
                        (
                            f"zero_{param_name}",
                            f"{param_name}=0",
                            ZeroDivisionError,
                        )
                    )

                # File not found
                if "path" in type_lower or "file" in param_name.lower():
                    errors.append(
                        (
                            f"nonexistent_{param_name}",
                            f"{param_name}='/nonexistent/path'",
                            FileNotFoundError,
                        )
                    )

                # Type errors
                if "int" in type_lower:
                    errors.append(
                        (
                            f"wrong_type_{param_name}",
                            f"{param_name}='not_an_int'",
                            TypeError,
                        )
                    )

                # Value errors for empty strings
                if "str" in type_lower and "optional" not in type_lower:
                    errors.append(
                        (
                            f"empty_{param_name}",
                            f"{param_name}=''",
                            ValueError,
                        )
                    )

        return errors
