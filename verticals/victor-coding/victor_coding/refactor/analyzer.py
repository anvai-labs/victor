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

"""Code analyzer for refactoring operations.

Uses AST parsing to analyze code structure and find symbols,
references, and refactoring opportunities.
"""

import ast
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from victor_coding.refactor.protocol import (
    RefactorRisk,
    RefactorSuggestion,
    RefactorType,
    SourceLocation,
    Symbol,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class CodeAnalyzer(Protocol):
    """Protocol for code analyzers."""

    @property
    def language(self) -> str:
        """Language this analyzer supports."""
        ...

    def parse(self, source: str, file_path: Path) -> Optional[ast.AST]:
        """Parse source code into AST."""
        ...

    def find_symbol_at(
        self,
        source: str,
        file_path: Path,
        line: int,
        column: int,
    ) -> Optional[Symbol]:
        """Find symbol at a specific location."""
        ...

    def find_all_symbols(
        self,
        source: str,
        file_path: Path,
    ) -> list[Symbol]:
        """Find all symbols in source code."""
        ...

    def find_references(
        self,
        source: str,
        file_path: Path,
        symbol_name: str,
    ) -> list[SourceLocation]:
        """Find all references to a symbol."""
        ...


class BaseCodeAnalyzer(ABC):
    """Abstract base class for code analyzers."""

    @property
    @abstractmethod
    def language(self) -> str:
        """Language this analyzer supports."""
        ...

    @abstractmethod
    def parse(self, source: str, file_path: Path) -> Optional[ast.AST]:
        """Parse source code into AST."""
        ...

    @abstractmethod
    def find_symbol_at(
        self,
        source: str,
        file_path: Path,
        line: int,
        column: int,
    ) -> Optional[Symbol]:
        """Find symbol at a specific location."""
        ...

    @abstractmethod
    def find_all_symbols(
        self,
        source: str,
        file_path: Path,
    ) -> list[Symbol]:
        """Find all symbols in source code."""
        ...

    @abstractmethod
    def find_references(
        self,
        source: str,
        file_path: Path,
        symbol_name: str,
    ) -> list[SourceLocation]:
        """Find all references to a symbol."""
        ...

    def suggest_refactorings(
        self,
        source: str,
        file_path: Path,
    ) -> list[RefactorSuggestion]:
        """Suggest potential refactorings."""
        return []


class PythonAnalyzer(BaseCodeAnalyzer):
    """Code analyzer for Python using the ast module."""

    @property
    def language(self) -> str:
        return "python"

    def parse(self, source: str, file_path: Path) -> Optional[ast.AST]:
        """Parse Python source code."""
        try:
            return ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return None

    def find_symbol_at(
        self,
        source: str,
        file_path: Path,
        line: int,
        column: int,
    ) -> Optional[Symbol]:
        """Find symbol at a specific location in Python code."""
        tree = self.parse(source, file_path)
        if tree is None:
            return None

        # Walk AST to find node at position
        for node in ast.walk(tree):
            if not hasattr(node, "lineno"):
                continue

            # Check if position matches
            if node.lineno == line:
                symbol = self._node_to_symbol(node, file_path)
                if symbol:
                    return symbol

        return None

    def find_all_symbols(
        self,
        source: str,
        file_path: Path,
    ) -> list[Symbol]:
        """Find all symbols in Python source."""
        tree = self.parse(source, file_path)
        if tree is None:
            return []

        symbols = []
        self._collect_symbols(tree, file_path, "", symbols)
        return symbols

    def _collect_symbols(
        self,
        node: ast.AST,
        file_path: Path,
        scope: str,
        symbols: list[Symbol],
    ) -> None:
        """Recursively collect symbols from AST."""
        if isinstance(node, ast.FunctionDef):
            symbol = self._function_to_symbol(node, file_path, scope)
            symbols.append(symbol)
            new_scope = f"{scope}.{node.name}" if scope else node.name
            for child in ast.iter_child_nodes(node):
                self._collect_symbols(child, file_path, new_scope, symbols)

        elif isinstance(node, ast.AsyncFunctionDef):
            symbol = self._function_to_symbol(node, file_path, scope, is_async=True)
            symbols.append(symbol)
            new_scope = f"{scope}.{node.name}" if scope else node.name
            for child in ast.iter_child_nodes(node):
                self._collect_symbols(child, file_path, new_scope, symbols)

        elif isinstance(node, ast.ClassDef):
            symbol = self._class_to_symbol(node, file_path, scope)
            symbols.append(symbol)
            new_scope = f"{scope}.{node.name}" if scope else node.name
            for child in ast.iter_child_nodes(node):
                self._collect_symbols(child, file_path, new_scope, symbols)

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for symbol in self._assignment_to_symbols(node, file_path, scope):
                symbols.append(symbol)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                symbols.append(
                    Symbol(
                        name=name,
                        kind="import",
                        location=SourceLocation(
                            file_path=file_path,
                            start_line=node.lineno,
                            start_column=node.col_offset,
                            end_line=node.lineno,
                            end_column=node.col_offset + len(name),
                        ),
                        scope=scope,
                    )
                )

        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                symbols.append(
                    Symbol(
                        name=name,
                        kind="import",
                        location=SourceLocation(
                            file_path=file_path,
                            start_line=node.lineno,
                            start_column=node.col_offset,
                            end_line=node.lineno,
                            end_column=node.col_offset + len(name),
                        ),
                        scope=scope,
                    )
                )

        else:
            for child in ast.iter_child_nodes(node):
                self._collect_symbols(child, file_path, scope, symbols)

    def _function_to_symbol(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        scope: str,
        is_async: bool = False,
    ) -> Symbol:
        """Convert function node to Symbol."""
        modifiers = []
        if is_async or isinstance(node, ast.AsyncFunctionDef):
            modifiers.append("async")

        # Check for decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id == "staticmethod":
                    modifiers.append("static")
                elif decorator.id == "classmethod":
                    modifiers.append("classmethod")
                elif decorator.id == "property":
                    modifiers.append("property")

        # Get return type annotation
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        # Get docstring
        docstring = ast.get_docstring(node)

        return Symbol(
            name=node.name,
            kind="function" if not scope or "." not in scope else "method",
            location=SourceLocation(
                file_path=file_path,
                start_line=node.lineno,
                start_column=node.col_offset,
                end_line=node.end_lineno or node.lineno,
                end_column=node.end_col_offset or 0,
            ),
            scope=scope,
            type_annotation=return_type,
            docstring=docstring,
            modifiers=modifiers,
        )

    def _class_to_symbol(
        self,
        node: ast.ClassDef,
        file_path: Path,
        scope: str,
    ) -> Symbol:
        """Convert class node to Symbol."""
        docstring = ast.get_docstring(node)

        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base))

        return Symbol(
            name=node.name,
            kind="class",
            location=SourceLocation(
                file_path=file_path,
                start_line=node.lineno,
                start_column=node.col_offset,
                end_line=node.end_lineno or node.lineno,
                end_column=node.end_col_offset or 0,
            ),
            scope=scope,
            docstring=docstring,
            modifiers=bases,  # Store bases in modifiers for now
        )

    def _assignment_to_symbols(
        self,
        node: ast.Assign | ast.AnnAssign,
        file_path: Path,
        scope: str,
    ) -> list[Symbol]:
        """Convert assignment to Symbols."""
        symbols = []

        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                type_ann = ast.unparse(node.annotation) if node.annotation else None
                symbols.append(
                    Symbol(
                        name=node.target.id,
                        kind="variable",
                        location=SourceLocation(
                            file_path=file_path,
                            start_line=node.lineno,
                            start_column=node.col_offset,
                            end_line=node.lineno,
                            end_column=node.col_offset + len(node.target.id),
                        ),
                        scope=scope,
                        type_annotation=type_ann,
                    )
                )

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.append(
                        Symbol(
                            name=target.id,
                            kind="variable",
                            location=SourceLocation(
                                file_path=file_path,
                                start_line=node.lineno,
                                start_column=node.col_offset,
                                end_line=node.lineno,
                                end_column=node.col_offset + len(target.id),
                            ),
                            scope=scope,
                        )
                    )

        return symbols

    def _node_to_symbol(
        self,
        node: ast.AST,
        file_path: Path,
    ) -> Optional[Symbol]:
        """Convert an AST node to a Symbol."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._function_to_symbol(node, file_path, "")
        elif isinstance(node, ast.ClassDef):
            return self._class_to_symbol(node, file_path, "")
        elif isinstance(node, ast.Name):
            return Symbol(
                name=node.id,
                kind="reference",
                location=SourceLocation(
                    file_path=file_path,
                    start_line=node.lineno,
                    start_column=node.col_offset,
                    end_line=node.lineno,
                    end_column=node.col_offset + len(node.id),
                ),
            )
        return None

    def find_references(
        self,
        source: str,
        file_path: Path,
        symbol_name: str,
    ) -> list[SourceLocation]:
        """Find all references to a symbol in Python code."""
        tree = self.parse(source, file_path)
        if tree is None:
            return []

        references = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == symbol_name:
                references.append(
                    SourceLocation(
                        file_path=file_path,
                        start_line=node.lineno,
                        start_column=node.col_offset,
                        end_line=node.lineno,
                        end_column=node.col_offset + len(node.id),
                    )
                )
            elif isinstance(node, ast.Attribute) and node.attr == symbol_name:
                references.append(
                    SourceLocation(
                        file_path=file_path,
                        start_line=node.lineno,
                        start_column=node.col_offset,
                        end_line=node.lineno,
                        end_column=node.end_col_offset or node.col_offset,
                    )
                )

        return references

    def suggest_refactorings(
        self,
        source: str,
        file_path: Path,
    ) -> list[RefactorSuggestion]:
        """Suggest potential refactorings for Python code."""
        tree = self.parse(source, file_path)
        if tree is None:
            return []

        suggestions = []

        for node in ast.walk(tree):
            # Suggest extracting long functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.end_lineno and node.lineno:
                    length = node.end_lineno - node.lineno
                    if length > 50:
                        suggestions.append(
                            RefactorSuggestion(
                                refactor_type=RefactorType.EXTRACT_FUNCTION,
                                target=SourceLocation(
                                    file_path=file_path,
                                    start_line=node.lineno,
                                    start_column=node.col_offset,
                                    end_line=node.end_lineno,
                                    end_column=node.end_col_offset or 0,
                                ),
                                reason=(
                                    f"Function '{node.name}' is {length} lines long, "
                                    "consider extracting"
                                ),
                                confidence=0.7,
                                risk=RefactorRisk.LOW,
                            )
                        )

            # Suggest converting to async
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Await):
                        suggestions.append(
                            RefactorSuggestion(
                                refactor_type=RefactorType.CONVERT_TO_ASYNC,
                                target=SourceLocation(
                                    file_path=file_path,
                                    start_line=node.lineno,
                                    start_column=node.col_offset,
                                    end_line=node.end_lineno or node.lineno,
                                    end_column=node.end_col_offset or 0,
                                ),
                                reason=f"Function '{node.name}' uses await but isn't async",
                                confidence=0.9,
                                risk=RefactorRisk.MEDIUM,
                                auto_fixable=True,
                            )
                        )
                        break

            # Suggest list comprehension
            if isinstance(node, ast.For):
                # Simple pattern: for x in y: result.append(expr)
                if (
                    len(node.body) == 1
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Call)
                ):
                    call = node.body[0].value
                    if isinstance(call.func, ast.Attribute) and call.func.attr == "append":
                        suggestions.append(
                            RefactorSuggestion(
                                refactor_type=RefactorType.CONVERT_LOOP_TO_COMPREHENSION,
                                target=SourceLocation(
                                    file_path=file_path,
                                    start_line=node.lineno,
                                    start_column=node.col_offset,
                                    end_line=node.end_lineno or node.lineno,
                                    end_column=node.end_col_offset or 0,
                                ),
                                reason="Loop can be converted to list comprehension",
                                confidence=0.8,
                                risk=RefactorRisk.SAFE,
                                auto_fixable=True,
                            )
                        )

        return suggestions


# Registry of analyzers
ANALYZERS: dict[str, type[BaseCodeAnalyzer]] = {
    "python": PythonAnalyzer,
}


def get_analyzer(language: str) -> Optional[BaseCodeAnalyzer]:
    """Get an analyzer for a language.

    Args:
        language: Language identifier

    Returns:
        Analyzer instance or None
    """
    analyzer_class = ANALYZERS.get(language.lower())
    if analyzer_class:
        return analyzer_class()
    return None
