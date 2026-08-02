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

"""Refactoring protocol types and data structures.

Defines the core types for AST-based code refactoring operations.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class RefactorType(Enum):
    """Types of refactoring operations."""

    # Rename operations
    RENAME_SYMBOL = "rename_symbol"
    RENAME_FILE = "rename_file"

    # Extract operations
    EXTRACT_FUNCTION = "extract_function"
    EXTRACT_METHOD = "extract_method"
    EXTRACT_VARIABLE = "extract_variable"
    EXTRACT_CLASS = "extract_class"
    EXTRACT_INTERFACE = "extract_interface"
    EXTRACT_CONSTANT = "extract_constant"

    # Inline operations
    INLINE_FUNCTION = "inline_function"
    INLINE_VARIABLE = "inline_variable"
    INLINE_CONSTANT = "inline_constant"

    # Move operations
    MOVE_SYMBOL = "move_symbol"
    MOVE_TO_FILE = "move_to_file"

    # Change signature
    ADD_PARAMETER = "add_parameter"
    REMOVE_PARAMETER = "remove_parameter"
    REORDER_PARAMETERS = "reorder_parameters"
    CHANGE_RETURN_TYPE = "change_return_type"

    # Class operations
    PULL_UP_METHOD = "pull_up_method"
    PUSH_DOWN_METHOD = "push_down_method"
    EXTRACT_SUPERCLASS = "extract_superclass"
    IMPLEMENT_INTERFACE = "implement_interface"

    # Code organization
    ORGANIZE_IMPORTS = "organize_imports"
    REMOVE_UNUSED_IMPORTS = "remove_unused_imports"
    SORT_MEMBERS = "sort_members"

    # Modernization
    CONVERT_TO_ASYNC = "convert_to_async"
    CONVERT_TO_GENERATOR = "convert_to_generator"
    CONVERT_LOOP_TO_COMPREHENSION = "convert_loop_to_comprehension"
    USE_OPTIONAL_CHAINING = "use_optional_chaining"


class RefactorScope(Enum):
    """Scope of refactoring operation."""

    LOCAL = "local"  # Within current file
    PROJECT = "project"  # Across project files
    WORKSPACE = "workspace"  # Across workspace


class RefactorRisk(Enum):
    """Risk level of a refactoring operation."""

    SAFE = "safe"  # No semantic changes
    LOW = "low"  # Minor semantic changes possible
    MEDIUM = "medium"  # May affect behavior
    HIGH = "high"  # Likely affects behavior
    BREAKING = "breaking"  # Definitely breaks API


@dataclass
class SourceLocation:
    """A location in source code."""

    file_path: Path
    start_line: int
    start_column: int
    end_line: int
    end_column: int

    def contains(self, line: int, column: int) -> bool:
        """Check if a position is within this location."""
        if line < self.start_line or line > self.end_line:
            return False
        if line == self.start_line and column < self.start_column:
            return False
        if line == self.end_line and column > self.end_column:
            return False
        return True


@dataclass
class RefactorSymbol:
    """Code symbol for refactoring operations.

    Renamed from Symbol to be semantically distinct:
    - RefactorSymbol (here): Refactoring symbol with SourceLocation and references
    - NativeSymbol (victor.native.protocols): Rust-extracted symbols (frozen)
    - IndexedSymbol (victor_coding.codebase.indexer): Pydantic model for index storage
    """

    name: str
    kind: str  # function, class, variable, parameter, etc.
    location: SourceLocation
    scope: str = ""  # Fully qualified scope
    type_annotation: Optional[str] = None
    docstring: Optional[str] = None
    modifiers: list[str] = field(default_factory=list)  # public, static, async, etc.
    references: list[SourceLocation] = field(default_factory=list)

    @property
    def qualified_name(self) -> str:
        """Get fully qualified name."""
        if self.scope:
            return f"{self.scope}.{self.name}"
        return self.name


# Backward compatibility alias
Symbol = RefactorSymbol


@dataclass
class CodeEdit:
    """A single edit to source code."""

    location: SourceLocation
    new_text: str
    description: str = ""

    @property
    def file_path(self) -> Path:
        return self.location.file_path


@dataclass
class RefactorRequest:
    """A request for a refactoring operation."""

    refactor_type: RefactorType
    target: SourceLocation
    scope: RefactorScope = RefactorScope.PROJECT

    # Type-specific parameters
    new_name: Optional[str] = None  # For rename
    extract_name: Optional[str] = None  # For extract
    destination_file: Optional[Path] = None  # For move
    parameter_changes: list[dict] = field(default_factory=list)  # For signature changes

    # Options
    preview: bool = True  # Generate preview without applying
    update_references: bool = True
    update_imports: bool = True
    preserve_formatting: bool = True


@dataclass
class RefactorPreview:
    """Preview of refactoring changes before applying."""

    request: RefactorRequest
    edits: list[CodeEdit] = field(default_factory=list)
    affected_files: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    risk: RefactorRisk = RefactorRisk.SAFE

    @property
    def is_valid(self) -> bool:
        """Check if refactoring can be applied."""
        return len(self.errors) == 0

    @property
    def edit_count(self) -> int:
        """Total number of edits."""
        return len(self.edits)

    @property
    def file_count(self) -> int:
        """Number of files affected."""
        return len(self.affected_files)


@dataclass
class RefactorResult:
    """Result of applying a refactoring operation."""

    request: RefactorRequest
    success: bool
    edits_applied: int = 0
    files_modified: list[Path] = field(default_factory=list)
    backup_paths: list[Path] = field(default_factory=list)
    error_message: str = ""
    duration_ms: float = 0.0

    def can_undo(self) -> bool:
        """Check if refactoring can be undone."""
        return len(self.backup_paths) > 0


@dataclass
class RefactorCapabilities:
    """Capabilities of a refactoring provider."""

    supported_refactors: list[RefactorType] = field(default_factory=list)
    supported_languages: list[str] = field(default_factory=list)
    supports_preview: bool = True
    supports_undo: bool = True
    supports_cross_file: bool = True
    max_affected_files: int = 100


@dataclass
class CodePattern:
    """A pattern for matching code structures."""

    name: str
    pattern: str  # Pattern syntax (regex, AST pattern, etc.)
    pattern_type: str = "ast"  # ast, regex, semantic
    language: str = ""
    description: str = ""
    replacement: Optional[str] = None  # For automated fixes


@dataclass
class RefactorSuggestion:
    """A suggested refactoring based on code analysis."""

    refactor_type: RefactorType
    target: SourceLocation
    reason: str
    confidence: float = 0.0  # 0-1
    risk: RefactorRisk = RefactorRisk.SAFE
    auto_fixable: bool = False
    suggested_name: Optional[str] = None
