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

"""Extract refactoring transforms."""

import ast
import textwrap
from pathlib import Path

from victor_coding.refactor.analyzer import BaseCodeAnalyzer
from victor_coding.refactor.protocol import (
    CodeEdit,
    RefactorPreview,
    RefactorRequest,
    RefactorRisk,
    RefactorType,
    SourceLocation,
)
from victor_coding.refactor.transforms.base import BaseTransform


class ExtractFunctionTransform(BaseTransform):
    """Transform for extracting code into a new function."""

    @property
    def refactor_type(self) -> RefactorType:
        return RefactorType.EXTRACT_FUNCTION

    @property
    def risk_level(self) -> RefactorRisk:
        return RefactorRisk.LOW

    def validate(
        self,
        request: RefactorRequest,
        source: str,
    ) -> tuple[bool, list[str]]:
        """Validate extract function request."""
        is_valid, errors = super().validate(request, source)

        if not request.extract_name:
            errors.append("Function name is required for extraction")
            is_valid = False

        if request.extract_name and not request.extract_name.isidentifier():
            errors.append(f"'{request.extract_name}' is not a valid identifier")
            is_valid = False

        return is_valid, errors

    def preview(
        self,
        request: RefactorRequest,
        sources: dict[Path, str],
        analyzer: BaseCodeAnalyzer,
    ) -> RefactorPreview:
        """Generate extract function preview."""
        preview = RefactorPreview(
            request=request,
            risk=self.risk_level,
        )

        target_file = request.target.file_path
        if target_file not in sources:
            preview.errors.append(f"File not found: {target_file}")
            return preview

        source = sources[target_file]
        is_valid, errors = self.validate(request, source)
        if not is_valid:
            preview.errors.extend(errors)
            return preview

        # Get the code to extract
        extracted_code = self._get_source_range(source, request.target)
        if not extracted_code.strip():
            preview.errors.append("No code selected for extraction")
            return preview

        # Analyze the extracted code to find:
        # 1. Variables that are used but defined outside (parameters)
        # 2. Variables that are defined and used outside (return values)
        parameters, return_values = self._analyze_variables(extracted_code, source, request.target)

        # Build the new function
        func_name = request.extract_name
        param_str = ", ".join(parameters) if parameters else ""

        # Determine indentation
        lines = source.split("\n")
        first_line = lines[request.target.start_line - 1]
        base_indent = len(first_line) - len(first_line.lstrip())
        indent = " " * base_indent

        # Build function body
        body_lines = extracted_code.split("\n")
        # Normalize indentation
        min_indent = (
            min(len(line) - len(line.lstrip()) for line in body_lines if line.strip())
            if body_lines
            else 0
        )

        normalized_body = []
        for line in body_lines:
            if line.strip():
                normalized_body.append("    " + line[min_indent:])
            else:
                normalized_body.append("")

        body_str = "\n".join(normalized_body)

        # Build return statement if needed
        return_stmt = ""
        if return_values:
            if len(return_values) == 1:
                return_stmt = f"\n    return {return_values[0]}"
            else:
                return_stmt = f"\n    return {', '.join(return_values)}"

        # Build complete function
        new_function = f"\n\ndef {func_name}({param_str}):\n{body_str}{return_stmt}\n"

        # Build function call
        if return_values:
            if len(return_values) == 1:
                call_stmt = f"{return_values[0]} = {func_name}({param_str})"
            else:
                call_stmt = f"{', '.join(return_values)} = {func_name}({param_str})"
        else:
            call_stmt = f"{func_name}({param_str})"

        # Find insertion point (before the function containing the selection)
        insert_line = self._find_function_start(source, request.target.start_line)

        # Create edits
        # 1. Insert new function
        preview.edits.append(
            CodeEdit(
                location=SourceLocation(
                    file_path=target_file,
                    start_line=insert_line,
                    start_column=0,
                    end_line=insert_line,
                    end_column=0,
                ),
                new_text=new_function,
                description=f"Add new function '{func_name}'",
            )
        )

        # 2. Replace extracted code with function call
        preview.edits.append(
            CodeEdit(
                location=request.target,
                new_text=indent + call_stmt,
                description=f"Replace code with call to '{func_name}'",
            )
        )

        preview.affected_files.append(target_file)

        return preview

    def _analyze_variables(
        self,
        extracted_code: str,
        full_source: str,
        location: SourceLocation,
    ) -> tuple[list[str], list[str]]:
        """Analyze variables in extracted code.

        Returns:
            Tuple of (parameters, return_values)
        """
        try:
            # Parse extracted code
            extracted_tree = ast.parse(extracted_code)
        except SyntaxError:
            # Wrap in function to help parsing
            try:
                wrapped = f"def _temp():\n{textwrap.indent(extracted_code, '    ')}"
                extracted_tree = ast.parse(wrapped)
            except SyntaxError:
                return [], []

        # Find all names in extracted code
        names_used: set[str] = set()
        names_assigned: set[str] = set()

        for node in ast.walk(extracted_tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    names_used.add(node.id)
                elif isinstance(node.ctx, ast.Store):
                    names_assigned.add(node.id)

        # Filter out builtins
        import builtins

        builtin_names = set(dir(builtins))
        names_used -= builtin_names
        names_assigned -= builtin_names

        # Parameters: used but not assigned in extracted code
        parameters = sorted(names_used - names_assigned)

        # Return values: assigned in extracted code (simple heuristic)
        # In practice, we'd check if they're used after the extraction point
        return_values = sorted(names_assigned)

        return parameters, return_values

    def _find_function_start(self, source: str, line_num: int) -> int:
        """Find the start of the function containing line_num."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 1

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if hasattr(node, "end_lineno"):
                    if node.lineno <= line_num <= (node.end_lineno or line_num):
                        return node.lineno

        return 1


class ExtractVariableTransform(BaseTransform):
    """Transform for extracting an expression into a variable."""

    @property
    def refactor_type(self) -> RefactorType:
        return RefactorType.EXTRACT_VARIABLE

    @property
    def risk_level(self) -> RefactorRisk:
        return RefactorRisk.SAFE

    def validate(
        self,
        request: RefactorRequest,
        source: str,
    ) -> tuple[bool, list[str]]:
        """Validate extract variable request."""
        is_valid, errors = super().validate(request, source)

        if not request.extract_name:
            errors.append("Variable name is required")
            is_valid = False

        if request.extract_name and not request.extract_name.isidentifier():
            errors.append(f"'{request.extract_name}' is not a valid identifier")
            is_valid = False

        return is_valid, errors

    def preview(
        self,
        request: RefactorRequest,
        sources: dict[Path, str],
        analyzer: BaseCodeAnalyzer,
    ) -> RefactorPreview:
        """Generate extract variable preview."""
        preview = RefactorPreview(
            request=request,
            risk=self.risk_level,
        )

        target_file = request.target.file_path
        if target_file not in sources:
            preview.errors.append(f"File not found: {target_file}")
            return preview

        source = sources[target_file]
        is_valid, errors = self.validate(request, source)
        if not is_valid:
            preview.errors.extend(errors)
            return preview

        # Get the expression to extract
        expression = self._get_source_range(source, request.target)
        if not expression.strip():
            preview.errors.append("No expression selected")
            return preview

        # Validate it's a valid expression
        try:
            ast.parse(expression, mode="eval")
        except SyntaxError as e:
            preview.errors.append(f"Selection is not a valid expression: {e}")
            return preview

        var_name = request.extract_name

        # Determine indentation
        lines = source.split("\n")
        target_line = lines[request.target.start_line - 1]
        indent = len(target_line) - len(target_line.lstrip())
        indent_str = " " * indent

        # Create variable assignment
        assignment = f"{var_name} = {expression}\n{indent_str}"

        # Create edits
        # 1. Insert variable assignment before the line
        preview.edits.append(
            CodeEdit(
                location=SourceLocation(
                    file_path=target_file,
                    start_line=request.target.start_line,
                    start_column=0,
                    end_line=request.target.start_line,
                    end_column=0,
                ),
                new_text=indent_str + assignment,
                description=f"Add variable '{var_name}'",
            )
        )

        # 2. Replace expression with variable name
        preview.edits.append(
            CodeEdit(
                location=request.target,
                new_text=var_name,
                description=f"Replace expression with '{var_name}'",
            )
        )

        preview.affected_files.append(target_file)

        return preview


class ExtractConstantTransform(BaseTransform):
    """Transform for extracting a value into a module-level constant."""

    @property
    def refactor_type(self) -> RefactorType:
        return RefactorType.EXTRACT_CONSTANT

    @property
    def risk_level(self) -> RefactorRisk:
        return RefactorRisk.SAFE

    def validate(
        self,
        request: RefactorRequest,
        source: str,
    ) -> tuple[bool, list[str]]:
        """Validate extract constant request."""
        is_valid, errors = super().validate(request, source)

        if not request.extract_name:
            errors.append("Constant name is required")
            is_valid = False

        if request.extract_name:
            # Constants should be uppercase
            if not request.extract_name.isupper():
                pass

        return is_valid, errors

    def preview(
        self,
        request: RefactorRequest,
        sources: dict[Path, str],
        analyzer: BaseCodeAnalyzer,
    ) -> RefactorPreview:
        """Generate extract constant preview."""
        preview = RefactorPreview(
            request=request,
            risk=self.risk_level,
        )

        target_file = request.target.file_path
        if target_file not in sources:
            preview.errors.append(f"File not found: {target_file}")
            return preview

        source = sources[target_file]
        is_valid, errors = self.validate(request, source)
        if not is_valid:
            preview.errors.extend(errors)
            return preview

        # Get the value to extract
        value = self._get_source_range(source, request.target)
        const_name = request.extract_name.upper()

        # Check if name suggests uppercase
        if request.extract_name != const_name:
            preview.warnings.append(
                f"Using uppercase name '{const_name}' instead of '{request.extract_name}'"
            )

        # Find insertion point (after imports, before first class/function)
        insert_line = self._find_constant_insert_point(source)

        # Create edits
        # 1. Insert constant at module level
        preview.edits.append(
            CodeEdit(
                location=SourceLocation(
                    file_path=target_file,
                    start_line=insert_line,
                    start_column=0,
                    end_line=insert_line,
                    end_column=0,
                ),
                new_text=f"{const_name} = {value}\n\n",
                description=f"Add constant '{const_name}'",
            )
        )

        # 2. Replace value with constant name
        preview.edits.append(
            CodeEdit(
                location=request.target,
                new_text=const_name,
                description=f"Replace value with '{const_name}'",
            )
        )

        preview.affected_files.append(target_file)

        return preview

    def _find_constant_insert_point(self, source: str) -> int:
        """Find the best line to insert a constant."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 1

        last_import = 0
        first_def = None

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                last_import = max(last_import, node.lineno)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if first_def is None:
                    first_def = node.lineno

        if last_import > 0:
            return last_import + 2  # After imports with blank line
        elif first_def:
            return first_def
        else:
            return 1
